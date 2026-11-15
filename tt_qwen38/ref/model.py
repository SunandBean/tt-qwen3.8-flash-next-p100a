"""Float32 CPU reference of Qwen3.8-Flash-Next (GGUF arch `qwen4exp`, the GSQ-RCO Coder IQ1_M) - M1 `ref_algo`.

Whole-sequence forward, one layer at a time: every layer's dense weights and the union of its routed experts are
dequantized once per call (expert dequant is ~45 ms/expert in numpy, so a token-at-a-time loop would be ~20 s/token).
State (GDN recurrent + conv history, attention K/V and indexer keys, PLE token/conv history) is returned so a later
call continues the sequence: `forward(prompt)` then `forward([next_token], state)` for decode.

Formulas: the sjettTT/tt-qwen-3.8-flash-next CPU oracle (Apache-2.0; reference.py, tt/*.py) with the GGUF storage
conventions of Strata's engine (MIT; notes/qwen4exp_forward_spec_strata.md):
  - zero-centred norms are stored as (1 + w): multiply by the stored weight; `ssm_norm` is a plain weight;
  - `ssm_a` = -exp(A_log): decay = exp(softplus(alpha + dt_bias) * ssm_a);
  - GDN value head h reads key head h % 16 (llama.cpp's tiled order; the HF checkpoint uses h // 3);
  - `attn_q` rows per head are [q 256 | gate 256]; RoPE is NEOX on the first 64 of 256 dims, theta 1e7.
All activations and matmuls are float32 (weights are the dequantized GGUF values).
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

from .gguf_weights import Weights

ARCH = "qwen4exp"


@dataclass
class Config:
    n_layers: int
    hidden: int
    n_head: int
    n_kv: int
    head_dim: int
    rope_dim: int
    rope_theta: float
    eps: float
    n_exp: int
    top_k: int
    hc: int
    full_interval: int
    ssm_state: int
    ssm_groups: int
    ssm_vheads: int
    conv_k: int
    idx_heads: int
    idx_dim: int
    idx_topk: int
    compress: int
    ple_layer: int
    ple_ngram: int
    ple_heads_per_ngram: int
    ple_conv_k: int
    ple_dim: int
    ple_eos: int
    ple_mult: list
    ple_offsets: list
    ple_vocab: list

    @classmethod
    def from_weights(cls, W: Weights) -> "Config":
        k = lambda s: W.kv(f"{ARCH}.{s}")
        return cls(
            n_layers=k("block_count"), hidden=k("embedding_length"), n_head=k("attention.head_count"),
            n_kv=k("attention.head_count_kv"), head_dim=k("attention.key_length"), rope_dim=k("rope.dimension_count"),
            rope_theta=float(k("rope.freq_base")), eps=float(k("attention.layer_norm_rms_epsilon")),
            n_exp=k("expert_count"), top_k=k("expert_used_count"), hc=k("hyper_connection.count"),
            full_interval=k("full_attention_interval"), ssm_state=k("ssm.state_size"), ssm_groups=k("ssm.group_count"),
            ssm_vheads=k("ssm.time_step_rank"), conv_k=k("ssm.conv_kernel"), idx_heads=k("attention.indexer.head_count"),
            idx_dim=k("attention.indexer.key_length"), idx_topk=k("attention.indexer.top_k"),
            compress=max(k("attention.compress_ratios")), ple_layer=k("ple.layers")[0], ple_ngram=k("ple.ngram_size"),
            ple_heads_per_ngram=k("ple.heads_per_ngram"), ple_conv_k=k("ple.conv_kernel"),
            ple_dim=k("embedding_length_per_layer_input"), ple_eos=k("ple.eos_token_id"),
            ple_mult=list(k("ple.layer_multipliers")), ple_offsets=list(k("ple.head_offsets")),
            ple_vocab=list(k("ple.head_vocab_sizes")),
        )

    def is_attention(self, layer: int) -> bool:
        return (layer + 1) % self.full_interval == 0


# ------------------------------------------------------------------ state
@dataclass
class GDNState:
    conv: torch.Tensor      # [conv_k - 1, 10240] previous qkv rows (pre-conv)
    S: torch.Tensor         # [nv, dk, dv]


@dataclass
class AttnState:
    k: torch.Tensor         # [P, n_kv, head_dim] (normed, rotated)
    v: torch.Tensor         # [P, n_kv, head_dim]
    idx_k: torch.Tensor     # [P, idx_dim] raw indexer keys


@dataclass
class PLEState:
    tokens: list            # the last (ngram - 1) token ids (EOS-padded at the start)
    conv: torch.Tensor      # [(conv_k - 1) * dilation, 4H] previous normalized gated rows


@dataclass
class State:
    pos: int = 0
    layers: dict = field(default_factory=dict)
    ple: PLEState | None = None


# ------------------------------------------------------------------ helpers
def rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def grouped_rms(x: torch.Tensor, w: torch.Tensor, group: int, eps: float) -> torch.Tensor:
    """RMS over each `group`-wide slice of the last dim, times the stored (1 + w) weight."""
    s = x.shape
    return (rms(x.reshape(*s[:-1], s[-1] // group, group), eps).reshape(s)) * w


def rope_neox(x: torch.Tensor, pos: torch.Tensor, rot: int, theta: float) -> torch.Tensor:
    """x [..T.., heads, D] with T on dim 0; rotate pairs (i, i + rot/2) of the first `rot` dims."""
    inv = theta ** (-torch.arange(0, rot, 2, dtype=torch.float64) / rot)
    ang = (pos.to(torch.float64)[:, None] * inv[None, :]).to(torch.float32)      # [T, rot/2]
    cos, sin = ang.cos(), ang.sin()
    shape = [x.shape[0]] + [1] * (x.ndim - 2) + [rot // 2]
    cos, sin = cos.view(shape), sin.view(shape)
    x0, x1, rest = x[..., : rot // 2], x[..., rot // 2: rot], x[..., rot:]
    return torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos, rest], dim=-1)


def T(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


# ------------------------------------------------------------------ the model
class Reference:
    def __init__(self, W: Weights | None = None, threads: int = 8):
        self.W = W or Weights()
        self.cfg = Config.from_weights(self.W)
        self.pool = ThreadPoolExecutor(threads)
        self._embd_rows = None
        self.trace: dict = {}   # filled per forward: routing, per-layer residuals (when asked)

    # ---- weights
    def t(self, name: str) -> torch.Tensor:
        return T(self.W.tensor(name))

    def experts(self, layer: int, ids) -> dict:
        """{e: (gate [640, H], up [640, H], down [H, 640])}, dequantized in parallel."""
        names = [f"blk.{layer}.ffn_{p}_exps.weight" for p in ("gate", "up", "down")]
        jobs = {(e, n): self.pool.submit(self.W.expert, n, e) for e in ids for n in names}
        return {e: tuple(T(jobs[(e, n)].result()) for n in names) for e in ids}

    # ---- blocks
    def embed(self, tokens) -> torch.Tensor:
        return T(self.W.rows("token_embd.weight", tokens))

    def gr_read(self, R: torch.Tensor, prefix: str):
        """R [T, 4H] -> (x [T, H], Rn [T, 4H])."""
        c = self.cfg
        Rn = grouped_rms(R, self.t(f"{prefix}_norm.weight"), c.hidden, c.eps)
        lr = F.silu(Rn @ self.t(f"{prefix}_down.weight").T / c.hc)
        g = torch.sigmoid(lr @ self.t(f"{prefix}_up.weight").T).view(-1, c.hc, c.hidden)
        x = (g * Rn.view(-1, c.hc, c.hidden)).mean(1)
        return x, Rn

    def gr_write(self, R: torch.Tensor, Rn: torch.Tensor, y: torch.Tensor, prefix: str) -> torch.Tensor:
        c = self.cfg
        coef = 2 * torch.sigmoid(Rn @ self.t(f"{prefix}_inject.weight").T / c.hc)       # [T, 4]
        return (R.view(-1, c.hc, c.hidden) + coef[:, :, None] * y[:, None, :]).reshape(R.shape)

    def gdn(self, L: int, x: torch.Tensor, st: GDNState | None):
        c = self.cfg
        p = f"blk.{L}."
        dk = c.ssm_state
        nk, nv = c.ssm_groups, c.ssm_vheads
        qkv = x @ self.t(p + "attn_qkv.weight").T                                   # [T, 10240]
        hist = torch.zeros(c.conv_k - 1, qkv.shape[1]) if st is None else st.conv
        full = torch.cat([hist, qkv], 0)
        w = self.t(p + "ssm_conv1d.weight")                                         # [10240, 4], tap 3 = newest
        Tn = x.shape[0]
        conv = sum(full[k: k + Tn] * w[:, k] for k in range(c.conv_k))
        conv = F.silu(conv)
        q, k, v = conv[:, : nk * dk], conv[:, nk * dk: 2 * nk * dk], conv[:, 2 * nk * dk:]
        q = q.view(Tn, nk, dk)
        k = k.view(Tn, nk, dk)
        v = v.view(Tn, nv, dk)
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6) / math.sqrt(dk)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
        kh = torch.arange(nv) % nk                                                  # tiled head map
        q, k = q[:, kh], k[:, kh]                                                   # [T, nv, dk]
        alpha = x @ self.t(p + "ssm_alpha.weight").T
        beta = torch.sigmoid(x @ self.t(p + "ssm_beta.weight").T)
        decay = torch.exp(F.softplus(alpha + self.t(p + "ssm_dt.bias")) * self.t(p + "ssm_a"))   # [T, nv]
        S = torch.zeros(nv, dk, dk) if st is None else st.S.clone()
        out = torch.empty(Tn, nv, dk)
        for t in range(Tn):
            S = S * decay[t][:, None, None]
            rem = torch.einsum("hk,hkv->hv", k[t], S)
            delta = (v[t] - rem) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            out[t] = torch.einsum("hk,hkv->hv", q[t], S)
        z = (x @ self.t(p + "attn_gate.weight").T).view(Tn, nv, dk)
        y = rms(out, c.eps) * self.t(p + "ssm_norm.weight") * torch.sigmoid(z)
        y = y.reshape(Tn, nv * dk) @ self.t(p + "ssm_out.weight").T
        return y, GDNState(conv=full[-(c.conv_k - 1):].clone(), S=S)

    def qsa(self, L: int, x: torch.Tensor, st: AttnState | None, pos0: int):
        c = self.cfg
        p = f"blk.{L}."
        Tn, D, R = x.shape[0], c.head_dim, c.rope_dim
        pos = torch.arange(pos0, pos0 + Tn)
        qg = (x @ self.t(p + "attn_q.weight").T).view(Tn, c.n_head, 2 * D)
        q, gate = qg[..., :D], qg[..., D:]
        q = rope_neox(rms(q, c.eps) * self.t(p + "attn_q_norm.weight"), pos, R, c.rope_theta)
        k = (x @ self.t(p + "attn_k.weight").T).view(Tn, c.n_kv, D)
        k = rope_neox(rms(k, c.eps) * self.t(p + "attn_k_norm.weight"), pos, R, c.rope_theta)
        v = (x @ self.t(p + "attn_v.weight").T).view(Tn, c.n_kv, D)
        iq = (x @ self.t(p + "indexer.q_proj.weight").T).view(Tn, c.idx_heads, c.idx_dim)
        ik = x @ self.t(p + "indexer.k_proj.weight").T                              # [T, idx_dim] raw
        if st is not None:
            k, v, ik = torch.cat([st.k, k]), torch.cat([st.v, v]), torch.cat([st.idx_k, ik])
        P = k.shape[0]
        # QSA selection mask [T, P]: dense while the visible blocks fit the budget
        mask = torch.zeros(Tn, P, dtype=torch.bool)
        budget_blocks = c.idx_topk // c.compress
        iqn = iq_rot = None
        for t in range(Tn):
            vis = pos0 + t + 1
            complete = vis // c.compress
            if complete <= budget_blocks:
                mask[t, :vis] = True
                continue
            if iq_rot is None:
                iqn = rms(iq, c.eps) * self.t(p + "indexer.q_norm.weight")
                iq_rot = rope_neox(iqn, pos, R, c.rope_theta)
                pooled = ik[: (P // c.compress) * c.compress].view(-1, c.compress, c.idx_dim).mean(1)
                starts = torch.arange(pooled.shape[0]) * c.compress
                bk = rope_neox((rms(pooled, c.eps) * self.t(p + "indexer.k_norm.weight"))[:, None, :], starts, R,
                               c.rope_theta)[:, 0]                                    # [blocks, idx_dim]
            sc = torch.relu(iq_rot[t] @ bk[:complete].T).sum(0) / math.sqrt(c.idx_dim)   # [complete]
            chosen = sc.topk(budget_blocks).indices
            for b in chosen.tolist():
                mask[t, b * c.compress: (b + 1) * c.compress] = True
            mask[t, complete * c.compress: vis] = True                              # the incomplete tail
        rep = c.n_head // c.n_kv
        kr, vr = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)           # [P, n_head, D]
        sc = torch.einsum("thd,phd->htp", q, kr) / math.sqrt(D)
        sc = sc.masked_fill(~mask[None], float("-inf"))
        pr = torch.softmax(sc, -1)
        o = torch.einsum("htp,phd->thd", pr, vr) * torch.sigmoid(gate)
        y = o.reshape(Tn, c.n_head * D) @ self.t(p + "attn_output.weight").T
        return y, AttnState(k=k, v=v, idx_k=ik), mask

    def moe(self, L: int, x: torch.Tensor):
        c = self.cfg
        p = f"blk.{L}."
        logits = x @ self.t(p + "ffn_gate_inp.weight").T                             # [T, E]
        probs = torch.softmax(logits, -1)
        wts, ids = probs.topk(c.top_k, -1)
        wts = wts / wts.sum(-1, keepdim=True)
        ex = self.experts(L, sorted(set(ids.flatten().tolist())))
        out = torch.zeros_like(x)
        for t in range(x.shape[0]):
            for j in range(c.top_k):
                g, u, d = ex[int(ids[t, j])]
                out[t] += wts[t, j] * ((F.silu(g @ x[t]) * (u @ x[t])) @ d.T)
        sh = (F.silu(x @ self.t(p + "ffn_gate_shexp.weight").T) * (x @ self.t(p + "ffn_up_shexp.weight").T)) \
            @ self.t(p + "ffn_down_shexp.weight").T
        sh = sh * torch.sigmoid(x @ self.t(p + "ffn_gate_inp_shexp.weight")[:, None])
        return out + sh, (ids, wts, logits)

    # ---- PLE
    def ple_ids(self, tokens: list, prev: list) -> np.ndarray:
        """[T, 16] row ids of the hashed 2-/3-gram heads (port: reference.ngram_token_ids)."""
        c = self.cfg
        n = c.ple_ngram
        hist = np.array(list(prev) + list(tokens), dtype=np.int64)
        eos = c.ple_eos
        H = len(hist)
        posn = np.arange(H)
        eos_pos = np.where(hist == eos, posn, -1)
        prev_eos_incl = np.maximum.accumulate(eos_pos)
        prev_eos = np.concatenate([[-1], prev_eos_incl[:-1]])
        in_seg = posn - (prev_eos + 1)
        shifted = []
        for s in range(n):
            src = np.clip(posn - s, 0, None)
            valid = (in_seg >= s) & (posn - s >= 0)
            shifted.append(np.where(valid, hist[src], eos))
        mult = np.array(c.ple_mult, dtype=np.int64)
        vocab = np.array(c.ple_vocab, dtype=np.int64)
        offs = np.array(c.ple_offsets, dtype=np.int64)
        blocks = []
        with np.errstate(over="ignore"):
            for g in range(2, n + 1):
                a = (g - 2) * c.ple_heads_per_ngram
                mixed = shifted[0] * mult[0]
                for s in range(1, g):
                    mixed = np.bitwise_xor(mixed, shifted[s] * mult[s])
                ids = np.remainder(mixed[:, None], vocab[None, a: a + c.ple_heads_per_ngram])
                blocks.append(ids + offs[None, a: a + c.ple_heads_per_ngram])
        return np.concatenate(blocks, 1)[len(prev):]

    def ple(self, R: torch.Tensor, tokens: list, st: PLEState | None):
        c = self.cfg
        p = f"blk.{c.ple_layer}."
        H, hc = c.hidden, c.hc
        prev = [c.ple_eos] * (c.ple_ngram - 1) if st is None else st.tokens
        ids = self.ple_ids(tokens, prev)                                            # [T, 16]
        rows = self.W.rows("per_layer_token_embd.weight", ids.ravel())              # [T*16, 160]
        emb = T(rows).view(len(tokens), -1)                                         # [T, 2560]
        key = grouped_rms(emb @ self.t(p + "ple_key.weight").T, self.t(p + "ple_norm_key.weight"), H, c.eps)
        val = emb @ self.t(p + "ple_value.weight").T                                # [T, H]
        qry = grouped_rms(R, self.t(p + "ple_norm_query.weight"), H, c.eps)
        gate = (key.view(-1, hc, H) * qry.view(-1, hc, H)).sum(-1, keepdim=True) / math.sqrt(H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = (torch.sigmoid(gate) * val[:, None, :]).reshape(-1, hc * H)
        nrm = grouped_rms(gv, self.t(p + "ple_norm_conv.weight"), H, c.eps)
        dil = c.ple_ngram
        hl = (c.ple_conv_k - 1) * dil
        hist = torch.zeros(hl, hc * H) if st is None else st.conv
        full = torch.cat([hist, nrm], 0)
        w = self.t(p + "ple_conv1d.weight")                                         # [4H, 4]
        Tn = len(tokens)
        conv = sum(full[k * dil: k * dil + Tn] * w[:, k] for k in range(c.ple_conv_k))
        out = gv + F.silu(conv)
        new = PLEState(tokens=(list(prev) + list(tokens))[-(c.ple_ngram - 1):], conv=full[-hl:].clone())
        return out, new

    # ---- whole model
    @torch.no_grad()
    def forward(self, tokens: list, state: State | None = None, layers: int | None = None, keep_residuals=False,
                logits: str = "last"):
        """tokens: list of ids appended at state.pos. Returns (logits [T or 1, V] or None, new State)."""
        c = self.cfg
        st = state or State()
        new = State(pos=st.pos + len(tokens), layers=dict(st.layers), ple=st.ple)
        R = self.embed(tokens).repeat(1, c.hc)
        self.trace = {"routing": {}, "residual": {}, "selected": {}}
        n_layers = c.n_layers if layers is None else layers
        for L in range(n_layers):
            if L == c.ple_layer:
                d, new.ple = self.ple(R, tokens, st.ple)
                R = R + d
            x, Rn = self.gr_read(R, f"blk.{L}.hc_attn")
            if c.is_attention(L):
                y, new.layers[L], mask = self.qsa(L, x, st.layers.get(L), st.pos)
                self.trace["selected"][L] = mask.sum(-1).tolist()
            else:
                y, new.layers[L] = self.gdn(L, x, st.layers.get(L))
            R = self.gr_write(R, Rn, y, f"blk.{L}.hc_attn")
            x, Rn = self.gr_read(R, f"blk.{L}.hc_ffn")
            y, (ids, wts, _) = self.moe(L, x)
            self.trace["routing"][L] = ids.tolist()
            R = self.gr_write(R, Rn, y, f"blk.{L}.hc_ffn")
            if keep_residuals:
                self.trace["residual"][L] = R.clone()
        if n_layers < c.n_layers or logits == "none":
            return None, new, R
        h, _ = self.gr_read(R if logits == "all" else R[-1:], "output_hc")
        out = h @ self.lm_head().T
        return out, new, R

    def lm_head(self) -> torch.Tensor:
        if getattr(self, "_lm", None) is None:
            self._lm = self.t("output.weight")                                       # [V, H], 2.5 GB fp32
        return self._lm
