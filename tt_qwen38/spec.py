"""MTP speculative decoding on the P100a (spec: notes/mtp_spec_strata.md).

Draft layer = the base Flash-Next MTP block (mtp-q2_0.gguf, HF names, raw zero-centred norms -> 1+w), run on the
Coder's final residuals, token embedding and LM head. Dense parts on the card (BFP8 / BF16), the 512 Q2_0 experts and
the 512-way router on the CPU (hostexp, ggml type 42). Its own dense K/V cache, one slot per cell.

Cell c = (R_c main final residual before the output mixer, token c+1) at rope position / KV slot c -> predicts c+2.
One MTP pass runs T rows with the same trace (rows beyond the needed ones are junk and never used):
  catch-up: rows r = 0..a of the last verify window (cells p..p+a, tokens outv[r]); only row a needs the MoE+head
            (rows < a only fill K/V) and gives draft d0;
  step j:   row 0 = the previous step's MTP output residual and draft d_{j-1}, cell p+a+j.
Prompt cells only need K/V: a pass without MoE/head (`front`).

Round (greedy): window [x, d0..d_{K-1}] -> verify -> a = leading matches -> emit outv[0..a] -> commit(a+1) -> MTP.

    python -m tt_qwen38.spec --prepare        # convert the MTP dense weights (host only)
"""
from __future__ import annotations

import argparse
import ctypes
import math
import os
import pathlib
import time

import numpy as np
import torch
import ttnn

from tt_qwen38.ref.gguf_weights import Weights

from .model import HC, RM, TILE, H
from .verify import VerifyModel
from .weights import CACHE, _ttnn_dtype

from tt_qwen38.sources import mtp_gguf   # the MTP draft layer (built on first use)
NE, TOPK, F = 512, 10, 640


def tensors_mtp(Wm: Weights):
    t = lambda n: torch.from_numpy(Wm.tensor("mtp." + n).copy())
    one = lambda n: t(n) + 1.0                                     # raw zero-centred RMSNorm weights
    for blk, hf in (("hc_attn", "layers.0.attn_hyper_connection"), ("hc_ffn", "layers.0.mlp_hyper_connection"),
                    ("out_hc", "hyper_connection_mixer")):
        norm, down, up = one(f"{hf}.hc_norm.weight"), t(f"{hf}.input_mix_weight_down.weight"), \
            t(f"{hf}.input_mix_weight_up.weight")
        inj = t(f"{hf}.block_inject_weight.weight") if blk != "out_hc" else None
        for b in range(HC):
            s = slice(b * H, (b + 1) * H)
            yield f"mtp.{blk}.norm{b}", norm[s].reshape(1, 1, 1, H), "f32row"
            yield f"mtp.{blk}.down{b}", down[:, s].T.contiguous(), "bf16"
            yield f"mtp.{blk}.up{b}", up[s, :].T.contiguous(), "bf16"
            if inj is not None:
                ij = torch.zeros(H, 32)
                ij[:, :HC] = inj[:, s].T
                yield f"mtp.{blk}.inj{b}", ij, "bf16"
    a = "layers.0.self_attn."
    yield "mtp.attn.q", t(a + "q_proj.weight").T.contiguous(), "bfp8"
    yield "mtp.attn.kv", torch.cat([t(a + "k_proj.weight"), t(a + "v_proj.weight")], 0).T.contiguous(), "bfp8"
    yield "mtp.attn.out", t(a + "o_proj.weight").T.contiguous(), "bfp8"
    yield "mtp.attn.q_norm", one(a + "q_norm.weight").reshape(1, 1, 1, -1), "f32row"
    yield "mtp.attn.k_norm", one(a + "k_norm.weight").reshape(1, 1, 1, -1), "f32row"
    m = "layers.0.mlp."
    gu = torch.cat([t(m + "shared_expert.gate_proj.weight"), t(m + "shared_expert.up_proj.weight")], 0)
    yield "mtp.sh.gu", gu.T.contiguous(), "bfp8"
    yield "mtp.sh.down", t(m + "shared_expert.down_proj.weight").T.contiguous(), "bfp8"
    g = torch.zeros(H, 32)
    g[:, 0] = t(m + "shared_expert_gate.weight").reshape(-1)
    yield "mtp.sh.gate", g, "bf16"
    yield "mtp.fc_e", t("fc_embedding.weight").T.contiguous(), "bfp8"
    yield "mtp.fc_h", t("fc_hidden.weight").T.contiguous(), "bfp8"
    yield "mtp.pre_e", one("pre_fc_norm_embedding.weight").reshape(1, 1, 1, H), "f32row"
    ph = one("pre_fc_norm_hidden.weight")
    for b in range(HC):
        yield f"mtp.pre_h{b}", ph[b * H:(b + 1) * H].reshape(1, 1, 1, H), "f32row"


def prepare_mtp(only_missing=True):
    CACHE.mkdir(parents=True, exist_ok=True)
    Wm = Weights([mtp_gguf()])
    t0, n = time.time(), 0
    for key, t, kind in tensors_mtp(Wm):
        f = CACHE / f"{key}.tensorbin"
        if only_missing and f.exists():
            continue
        x = t.to(torch.bfloat16) if kind == "bf16" else t.float()
        if x.ndim == 2:
            x = x.reshape(1, 1, *x.shape)
        ttnn.dump_tensor(str(f), ttnn.from_torch(x, dtype=_ttnn_dtype(ttnn, kind), layout=TILE))
        n += 1
    print(f"prepared {n} MTP tensors in {time.time() - t0:.0f} s -> {CACHE}")


class MtpHost:
    """512-way router (fp32 from BF16) and the Q2_0 experts of the MTP GGUF on the CPU (hostexp)."""

    def __init__(self, hx):
        self.Wm = Weights([mtp_gguf()])
        self.hx = hx                                                # the main HostExperts (same lib, threads)
        g, _, mm = self.Wm.files[0]
        base = np.frombuffer(mm, dtype=np.uint8).ctypes.data + g.data_start
        self._keep = mm
        tg = self.Wm.info("mtp.layers.0.mlp.experts.gate_up_proj")
        td = self.Wm.info("mtp.layers.0.mlp.experts.down_proj")
        assert tg.type_name == "Q2_0" and td.type_name == "Q2_0"
        self.rb_h = H // 64 * 18                                    # one 2560-wide Q2_0 row
        self.per_gu = tg.expected_bytes() // NE                     # 1280 rows
        self.per_d = td.expected_bytes() // NE
        self.gu0, self.d0 = base + tg.offset, base + td.offset
        self.router = self.Wm.tensor("mtp.layers.0.mlp.gate.weight").astype(np.float32)   # [512, 2560]

    def moe(self, x: np.ndarray) -> np.ndarray:
        lg = self.router @ x
        p = np.exp(lg - lg.max())
        p /= p.sum()
        ids = np.argsort(-p, kind="stable")[:TOPK]
        w = (p[ids] / max(p[ids].sum(), 2.0 ** -14)).astype(np.float32)
        P = ctypes.c_void_p * TOPK
        g = P(*[self.gu0 + int(e) * self.per_gu for e in ids])
        u = P(*[self.gu0 + int(e) * self.per_gu + F * self.rb_h for e in ids])
        d = P(*[self.d0 + int(e) * self.per_d for e in ids])
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = np.empty(H, dtype=np.float32)
        rc = self.hx.lib.hx_moe(TOPK, g, u, d, 42, 42, 42, w.ctypes.data, x.ctypes.data, H, F, y.ctypes.data)
        if rc:
            raise RuntimeError("hx_moe failed for the MTP experts")
        return y


class SpecModel(VerifyModel):
    def __init__(self, device, experts=None, drafts: int = 3, min_p: float = 0.0, prefill_rows: int = 28,
                 s_max: int = 32768, prefix_slots: int = 0, **kw):
        super().__init__(device, experts=experts, t_max=drafts + 1, s_max=s_max, max_ctx=s_max, **kw)
        self.pre = self.sibling(prefill_rows) if prefill_rows > self.T else None   # M9 batched prompt reading
        self.K = drafts
        self.use_prefill = True
        self.debug_top2 = False
        self.min_p = min_p   # Strata's --spec-min-p (0.5 there). on the P100a 0.5 was slower (chat 6.76 vs 6.99,
                             # code 9.42 vs 10.03 tok/s): the prob costs ~3 ms per MTP pass, skipped rows save less
        self.mtraces = None
        if drafts > 0:                         # drafts=0: plain decode (1-row verify passes), no MTP layer loaded
            self.mh = MtpHost(self.hx)
            self._alloc_m()
        self.spec_stats = {"rounds": 0, "accepted": 0, "drafted": 0, "emitted": 0}
        if os.environ.get("P100A_GPU") in ("1", "fake"):   # warm experts on the RTX 5070 Ti
            self.attach_gpu(float(os.environ.get("P100A_GPU_GB") or 10), os.environ.get("P100A_GPU_SETS") or None,
                            max_rows=max(prefill_rows, self.T))
        if os.environ.get("P100A_ROUTE_COUNTS"):  # count routed (layer, expert) pairs for warm-set choice
            # decode passes (the T-row verify window) and prompt passes (the 28-row sibling) counted apart: agent
            # prompts are ~25x longer than their answers and route differently (trace-ranked residents from
            # pooled counts were 25% slower on decode)
            self.route_counts = np.zeros((self.cfg.n_layers, self.cfg.n_exp), np.int64)
            if self.pre is not None:
                self.pre.route_counts = np.zeros_like(self.route_counts)
        self.pcache = None                     # F2 prefix cache: slots allocated now, before any trace capture
        if prefix_slots and self.pre is not None:
            from .prefix import PrefixCache
            self.pcache = PrefixCache(self, prefix_slots, prefill_rows)

    def attach_gpu(self, gb: float, sets_path: str | None = None, max_rows: int = 32):
        """Warm tier: the next experts after the P100a's residents (Strata's ranking, or a JSON {layer: [experts]})
        until `gb` GB of GGUF bytes sit in the 5070 Ti's VRAM. Shared by the verify window and the prompt sibling."""
        import json
        from tt_qwen38.hostexp.gpu import FakeGpuExperts, GpuExperts, warm_sets
        res = self.experts.sets if self.experts is not None else {}
        if sets_path:
            sets = {int(k): v for k, v in json.load(open(sets_path)).items()}
        else:
            sets = warm_sets(self.W, self.gpu_ranking(), res, gb, self.cfg.n_layers)
        for L, v in sets.items():                  # never duplicate a resident
            r = set(res.get(L, ()))
            sets[L] = [e for e in v if e not in r]
        if os.environ.get("P100A_GPU") == "fake":
            self.gx = FakeGpuExperts(self.W, sets, self.hx)
        else:
            self.gx = GpuExperts(self.W, sets, max_rows=max_rows, max_k=self.cfg.top_k, n_layers=self.cfg.n_layers)
        if self.pre is not None:
            self.pre.gx = self.gx
        print(f"GPU warm tier: {self.gx.n_warm} experts, {self.gx.vram_bytes / 2**30:.2f} GiB VRAM, "
              f"loaded in {self.gx.load_s:.1f} s", flush=True)

    def gpu_ranking(self):
        """(layer, expert) pairs, hottest first, for the warm set: the step-7 decode routing counts
        (data/route-counts-<model>-decode.npy) by default, P100A_GPU_RANK=strata for Strata's profile alone."""
        from .experts import ranking
        return ranking(os.environ.get("P100A_GPU_RANK", "trace"))

    def gpu_close(self):
        if self.gx is not None:
            self.gx.close()
            self.gx = None
            if self.pre is not None:
                self.pre.gx = None

    def cached_prefix(self, ids) -> int:
        """tokens a request with this prompt would resume from, without touching the card"""
        if self.pcache is None or not self.use_prefill:
            return 0
        return self.pcache.lookup(ids)[1]

    def _alloc_m(self):
        T, z = self.T, torch.zeros
        self.mkc = self._dev(z(2, 1, self.s_max, 256), dtype=ttnn.bfloat16)
        self.mvc = self._dev(z(2, 1, self.s_max, 256), dtype=ttnn.bfloat16)
        self.selA = self._dev(z(1, 1, 32, 32), layout=RM)
        self.selB = self._dev(z(1, 1, 32, 32), layout=RM)
        self.emb_m = self._dev(z(1, 1, T, H), layout=RM)
        self.cos_m = self._dev(torch.ones(1, 1, T, 256), layout=RM)
        self.sin_m = self._dev(z(1, 1, T, 256), layout=RM)
        self.pos_m = [self._dev(torch.zeros(2, dtype=torch.int32), dtype=ttnn.int32, layout=RM) for _ in range(T)]
        self.cur_m = self._dev(torch.zeros(2 * T, dtype=torch.int32), dtype=ttnn.int32, layout=RM)
        self.ycold_m = self._dev(z(1, 1, T, H), layout=RM)
        self.Rm_out = self._dev(z(1, HC, 32, H))

    # ---- device segments
    def mseg_a(self):
        T = self.T
        sa = ttnn.repeat(ttnn.to_layout(self.selA, TILE), ttnn.Shape([1, HC, 1, 1]))
        sb = ttnn.repeat(ttnn.to_layout(self.selB, TILE), ttnn.Shape([1, HC, 1, 1]))
        Rin = ttnn.add(self._mmf(sa, self.R), self._mmf(sb, self.Rm_out))             # [1,4,32,H]
        e = self._mm(ttnn.multiply(self._rms(self._pad32(self.emb_m)), self.w["mtp.pre_e"]), self.w["mtp.fc_e"])
        ms = ttnn.multiply(ttnn.sum(ttnn.mean(ttnn.multiply(Rin, Rin), dim=-1, keepdim=True), dim=1, keepdim=True),
                           1.0 / HC)                                                   # [1,1,32,1] over 10240
        inv = ttnn.repeat(ttnn.rsqrt(ttnn.add(ms, self.cfg.eps)), ttnn.Shape([1, HC, 1, 1]))
        if not hasattr(self, "_preh"):
            self._preh = ttnn.concat([self.w[f"mtp.pre_h{b}"] for b in range(HC)], dim=1)
        hn = ttnn.multiply(ttnn.multiply(Rin, inv), self._preh)
        h = self._mm(hn, self.w["mtp.fc_h"])
        R = ttnn.where(self.mask4_v, ttnn.add(h, ttnn.repeat(e, ttnn.Shape([1, HC, 1, 1]))), self._zero_4)
        x, Rn = self.gr_read(R, "mtp.hc_attn")
        cos, sin = ttnn.to_layout(self.cos_m, TILE), ttnn.to_layout(self.sin_m, TILE)
        cosk, sink = ttnn.reshape(cos, [1, T, 1, 256]), ttnn.reshape(sin, [1, T, 1, 256])
        cosq = ttnn.to_layout(ttnn.repeat_interleave(self.cos_m, 24, dim=2), TILE)
        sinq = ttnn.to_layout(ttnn.repeat_interleave(self.sin_m, 24, dim=2), TILE)
        y = self.vattn(None, x, cosk, sink, cosq, sinq, p="mtp.attn.", kc=self.mkc, vc=self.mvc,
                       pos=self.pos_m, cur=self.cur_m)
        R2 = self._gr_write_v(R, Rn, y, "mtp.hc_attn")
        x2, Rn2 = self.gr_read(R2, "mtp.hc_ffn")
        xrow = ttnn.slice(ttnn.to_layout(ttnn.typecast(x2, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, 1, T, H])
        return {"R2": R2, "Rn2": Rn2, "x2": x2, "xrow": xrow}

    def mseg_b(self, a):
        x = a["x2"]
        gu = self._mm(x, self.w["mtp.sh.gu"])
        hh = ttnn.multiply(ttnn.silu(ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 32, F])),
                           ttnn.slice(gu, [0, 0, 0, F], [1, 1, 32, 2 * F]))
        sh = self._mm(hh, self.w["mtp.sh.down"])
        sg = ttnn.sigmoid(ttnn.slice(self._mm(x, self.w["mtp.sh.gate"]), [0, 0, 0, 0], [1, 1, 32, 1]))
        y = ttnn.add(ttnn.multiply(sh, sg), self._pad32(self.ycold_m))
        R3 = self._gr_write_v(a["R2"], a["Rn2"], y, "mtp.hc_ffn")
        ttnn.copy(R3, self.Rm_out)
        hm, _ = self.gr_read(R3, "mtp.out_hc")
        lg = self._mm(hm, self.w["lm_head"])
        rm = ttnn.slice(ttnn.to_layout(ttnn.typecast(lg, ttnn.bfloat16), RM), [0, 0, 0, 0],
                        [1, 1, self.T, lg.shape[-1]])
        out = {"argmax": ttnn.argmax(rm, dim=-1)}
        if self.min_p > 0:                                                              # prob = 1 / sum exp(l - max)
            mx = ttnn.max(lg, dim=-1, keepdim=True)
            out["sumexp"] = ttnn.sum(ttnn.exp(ttnn.subtract(lg, mx)), dim=-1, keepdim=True)
        return out

    def _mrun(self, name, fn, *args):
        if self.mtraces is None:
            return fn(*args)
        tid, out = self.mtraces[name]
        ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
        return out

    # ---- host side of one MTP pass
    def mtp_pass(self, src: str, tokens: list, pos0: int, head_rows=(), src_row: int = 0, src_off: int = 0):
        """src 'main': row r <- verify residual row r (r < T); 'mtp': row 0 <- MTP output row src_row.
        head_rows: rows that need MoE + head (their argmax is returned); empty = K/V only."""
        T = self.T
        t_in = time.perf_counter()
        try:
            return self._mtp_pass(src, tokens, pos0, head_rows, src_row, T, src_off)
        finally:
            self.phase["mtp"] = self.phase.get("mtp", 0) + time.perf_counter() - t_in
            self.phase["mtp_n"] = self.phase.get("mtp_n", 0) + 1

    def _mtp_pass(self, src, tokens, pos0, head_rows, src_row, T, src_off=0):
        sa, sb = torch.zeros(1, 1, 32, 32), torch.zeros(1, 1, 32, 32)
        if src == "main":                    # MTP row r <- residual row src_off + r (prefill passes have up to 28)
            for r in range(min(T, 32 - src_off)):
                sa[0, 0, r, src_off + r] = 1
        else:
            sb[0, 0, 0, src_row] = 1
        self._write(self.selA, sa)
        self._write(self.selB, sb)
        toks = list(tokens) + [tokens[-1]] * (T - len(tokens))
        e = torch.from_numpy(np.ascontiguousarray(self.W.rows("token_embd.weight", toks), dtype=np.float32))
        self._write(self.emb_m, e.reshape(1, 1, T, H))
        c = self.cfg
        inv = c.rope_theta ** (-torch.arange(0, c.rope_dim, 2, dtype=torch.float64) / c.rope_dim)
        cos, sin = torch.ones(T, 256), torch.zeros(T, 256)
        lim = self.s_max - 1
        for r in range(T):
            ang = (pos0 + r) * inv
            cos[r, : c.rope_dim] = torch.cat([ang.cos(), ang.cos()]).float()
            sin[r, : c.rope_dim] = torch.cat([ang.sin(), ang.sin()]).float()
            pr = min(pos0 + r, lim)
            self._write(self.pos_m[r], torch.tensor([pr, pr], dtype=torch.int32), ttnn.int32)
        self._write(self.cos_m, cos.reshape(1, 1, T, 256))
        self._write(self.sin_m, sin.reshape(1, 1, T, 256))
        self._write(self.cur_m, torch.tensor([min(pos0 + u // 2, lim) for u in range(2 * T)], dtype=torch.int32),
                    ttnn.int32)
        a = self._mrun("ma", self.mseg_a)
        if not head_rows:
            return []
        X = self._read(a["xrow"]).float().reshape(T, H).numpy()
        Y = np.zeros((T, H), np.float32)
        t0 = time.perf_counter()
        for r in head_rows:
            Y[r] = self.mh.moe(X[r])
        self.phase["mtp_experts"] = self.phase.get("mtp_experts", 0) + time.perf_counter() - t0
        self._write(self.ycold_m, torch.from_numpy(Y).reshape(1, 1, T, H))
        b = self._mrun("mb", self.mseg_b, a)
        am = self._read(b["argmax"]).reshape(-1)[:T].long().tolist()
        if "sumexp" in b:
            se = self._read(b["sumexp"]).float().reshape(-1)
            self.last_probs = [float(1.0 / se[r]) for r in head_rows]
        else:
            self.last_probs = [1.0] * len(head_rows)
        return [am[r] for r in head_rows]

    def warm_m(self):
        self.reset()
        self.mtp_pass("main", [198] * self.T, 0, head_rows=(0,))
        self.reset()

    def capture_m(self):
        tr = {}
        tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        a = self.mseg_a()
        ttnn.end_trace_capture(self.dev, tid, cq_id=0)
        tr["ma"] = (tid, a)
        tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        b = self.mseg_b(a)
        ttnn.end_trace_capture(self.dev, tid, cq_id=0)
        tr["mb"] = (tid, b)
        self.mtraces = tr

    def capture_all(self):
        """All eager warm-ups first, then every trace back to back: no device allocation after the first capture.
        The process holds only these traces (never the plain TracedModel set: a second trace set on the card corrupts the first)."""
        self.warm_v()
        if self.pre is not None:
            self.pre.warm_v()
        if self.K:
            self.warm_m()
        self.capture_v(warm=False)
        if self.pre is not None:
            self.pre.capture_v(warm=False)
        if self.K:
            self.capture_m()
        self.reset()

    # ---- generation
    def generate(self, ids: list, max_new: int, eos=(), spec: bool = True, on_token=None, temperature: float = 0.0,
                 top_p: float = 1.0, top_k: int = 0, seed=None):
        """One request start to end (see steps()). Returns (tokens, info)."""
        res = {}
        for _ in self.steps(ids, max_new, eos, spec, on_token, temperature, top_p, top_k, seed, res):
            pass
        return res["out"]

    def steps(self, ids: list, max_new: int, eos=(), spec: bool = True, on_token=None, temperature: float = 0.0,
              top_p: float = 1.0, top_k: int = 0, seed=None, result=None):
        """Generator: yields after every card pass group (one prompt chunk, one decode round), so a scheduler can
        interleave requests (device.sessions swaps the card state between them). result["out"] = (tokens, info)
        at the end. Greedy when temperature == 0, else sampling. spec=False: same verify path, no drafts used.
        Sampling with drafts is exact: the MTP drafts are deterministic (argmax), so draft d is accepted with
        probability p(d) under the target distribution p (temperature, top-k, top-p applied) and, on rejection, the
        token is drawn from p with d removed (renormalised); after all accepted drafts one more token is drawn.
        on_token(t) may return True to stop (stop strings). Returns (tokens, info)."""
        T, K = self.T, self.K
        spec = spec and K > 0
        samp = temperature > 0
        rng = np.random.default_rng(seed)

        def dist(lg):                                       # target distribution of one row (numpy float64)
            z = lg.double() / temperature
            if top_k and top_k < z.numel():
                kth = torch.topk(z, top_k).values[-1]
                z = torch.where(z < kth, torch.full_like(z, -float("inf")), z)
            pr = torch.softmax(z, -1)
            if top_p < 1.0:
                sp, si = torch.sort(pr, descending=True)
                keep = torch.cumsum(sp, 0) - sp < top_p             # smallest prefix with mass >= top_p
                mask = torch.zeros_like(pr, dtype=torch.bool)
                mask[si[keep]] = True
                pr = torch.where(mask, pr, torch.zeros_like(pr))
                pr = pr / pr.sum()
            return pr.numpy()

        def draw(pr):
            return int(rng.choice(pr.size, p=pr))

        stop = {"flag": False}

        def emit(t):
            out.append(t)
            if on_token and on_token(t):
                stop["flag"] = True
        self.reset()
        n = len(ids)
        out = []
        t0 = time.perf_counter()
        # prompt: P rows per pass (P = 28 with the prefill sibling, else T); MTP K/V for prompt cells from
        # (R_c, ids[c+1]), T rows per MTP pass read from the shared residual
        pm = self.pre if (self.pre is not None and self.use_prefill) else self
        if pm is not self:
            pm.reset_host()
        P = pm.T
        i = 0
        pc = self.pcache if (self.pcache is not None and pm is not self) else None   # F2 (needs the aligned 28-row reading)
        saves = set()
        if pc is not None:
            pc.stats["lookups"] += 1
            slot, b = pc.lookup(ids)
            if slot is not None:
                ple = pc.restore(slot)
                pm.pos, pm.tokens, pm.ple_state = b, list(ids[:b]), ple
                i = b
                pc.stats["hits"] += 1
                pc.stats["hit_tokens"] += b
            saves = pc.plan(ids)
            pc.live = []                        # rows >= i are being rewritten from here on
        self.last_cached = cached0 = i
        while i < n:
            chunk = ids[i:i + P]
            nv = len(chunk)
            last = i + nv >= n
            res = pm.verify(chunk + [chunk[-1]] * (P - nv), n_valid=nv, logits=last and (self.debug_top2 or samp))
            outv = res[0] if isinstance(res, tuple) else res
            if isinstance(res, tuple):
                v, ix = res[1][nv - 1].float().topk(2)
                self.last_top2 = [(int(ix[0]), round(float(v[0]), 3)), (int(ix[1]), round(float(v[1]), 3))]
                if samp:
                    outv = list(outv)
                    outv[nv - 1] = draw(dist(res[1][nv - 1]))         # the first generated token, sampled
            pm.commit(nv)
            if spec:
                toks = [ids[c + 1] if c + 1 < n else outv[nv - 1] for c in range(i, i + nv)]
                for off in range(0, nv, T):
                    head = (nv - 1 - off,) if last and off <= nv - 1 < off + T else ()
                    d = self.mtp_pass("main", toks[off:off + T], i + off, head_rows=head, src_off=off)
                    if head:
                        head_row = head[0]
            i += nv
            if pc is not None and i in saves and i < n:
                pc.save(ids, i, pm.ple_state)
            if i < n:
                yield "prompt"
        if pc is not None:
            pc.live = list(ids)
        if pm is not self:
            self.pos, self.tokens, self.ple_state = pm.pos, list(pm.tokens), pm.ple_state
        t1 = time.perf_counter()
        x = outv[nv - 1]
        p = n                                       # position of x
        drafts = []
        if spec:
            drafts = self._chain(d, p - 1, head_row)
        acc_hist, prop_hist = [], []
        while True:
            emit(x)
            if stop["flag"] or x in eos or len(out) >= max_new or p + T > min(self.s_max, self.max_ctx):
                break
            nprop = len(drafts) if spec else 0
            window = [x] + drafts[:nprop] + [x] * (K - nprop)
            if samp:
                _, lg = self.verify(window, n_valid=1 + nprop, logits=True)
                a, final = 0, None
                while a < nprop:
                    pr = dist(lg[a])
                    d = window[a + 1]
                    if rng.random() < pr[d]:
                        a += 1
                        continue
                    pr[d] = 0.0                                   # rejected: draw from p without d
                    final = draw(pr / pr.sum())
                    break
                if final is None:
                    final = draw(dist(lg[a]))
                emitted = window[1:a + 1] + [final]
            else:
                outv = self.verify(window, n_valid=1 + nprop)
                a = 0
                while a < nprop and window[a + 1] == outv[a]:
                    a += 1
                emitted = outv[: a + 1]
            self.commit(a + 1)
            acc_hist.append(a)
            prop_hist.append(nprop)
            for tkn in emitted[:-1]:
                emit(tkn)
                if stop["flag"] or tkn in eos or len(out) >= max_new:
                    break
            else:
                x = emitted[-1]
                if spec:
                    d0 = self.mtp_pass("main", emitted, p, head_rows=(a,))
                    drafts = self._chain(d0, p + a, a)
                p += a + 1
                yield "round"
                continue
            break
        t2 = time.perf_counter()
        self.spec_stats["rounds"] += len(acc_hist)
        self.spec_stats["accepted"] += sum(acc_hist)
        self.spec_stats["drafted"] += sum(prop_hist)
        if self.route_counts is not None:                  # cumulative, rewritten after each request
            np.save(os.environ["P100A_ROUTE_COUNTS"], self.route_counts)
            if self.pre is not None:
                np.save(os.environ["P100A_ROUTE_COUNTS"].replace(".npy", "-prompt.npy"), self.pre.route_counts)
        result["out"] = (out[:max_new], {"prompt_s": t1 - t0, "decode_s": t2 - t1, "rounds": len(acc_hist),
                                         "accepted": acc_hist, "proposed": prop_hist, "cached": cached0})

    def _chain(self, d0: list, cell0: int, src_row: int) -> list:
        """Drafts after the catch-up pass that produced d0 (at cell cell0, MTP output in row src_row): keep chaining
        while the last draft's prob >= min_p (up to K drafts). Returns the proposed drafts (prob >= min_p only)."""
        drafts, probs = list(d0), list(self.last_probs)
        while len(drafts) < self.K and probs[-1] >= self.min_p:
            j = len(drafts)
            drafts += self.mtp_pass("mtp", [drafts[-1]], cell0 + j, head_rows=(0,), src_row=src_row if j == 1 else 0)
            probs += self.last_probs
        n = 0
        while n < len(drafts) and probs[n] >= self.min_p:
            n += 1
        return drafts[:n]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    a = ap.parse_args()
    if a.prepare:
        prepare_mtp()
