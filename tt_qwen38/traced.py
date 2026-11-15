"""traced decode. Same math as tt_qwen38/model.py (M2/M3), but every layer runs as two captured ttnn traces so the
host no longer dispatches thousands of small ops per token:

  A(L): residual -> [PLE add at L=1] -> GR read -> GDN | attention -> GR write -> GR read (ffn) -> x, x row for the host
  host: read x (5 KB), route (fp32), cold experts (hostexp), write sparsity row + cold-expert output (10 KB)
  B(L): resident experts (sparse_matmul) + shared expert + cold output -> GR write -> residual
  final: out_hc GR read -> LM head -> argmax (and the logits row, when asked)

Everything that lives across tokens is allocated before the first capture and updated in place (ttnn.copy /
paged_update_cache): the residual, GDN state and conv history, KV caches, and the host-written inputs (embedding,
position, cos/sin, PLE delta, sparsity rows, cold-expert output).
Attention: one KV cache per attention layer [2, 1, S_max, 256] (the 2 KV heads as 2 'users' at the same position),
`paged_update_cache` + `scaled_dot_product_attention_decode` (MQA per KV head). Dense, i.e. exact up to 2,050 tokens
(Strata's indexer starts dropping blocks after that; not implemented yet).
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import ttnn

from .model import HC, RM, TILE, H, P100aModel
from .weights import is_attention


class TracedModel(P100aModel):
    def __init__(self, device, experts=None, s_max: int = 4096, **kw):
        super().__init__(device, experts=experts, **kw)
        self.s_max = s_max
        self.traces = None
        self.phase = {}
        self._alloc()

    # ---- persistent buffers
    def _dev(self, t: torch.Tensor, dtype=ttnn.float32, layout=TILE):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _alloc(self):
        c = self.cfg
        z = torch.zeros
        self.R = self._dev(z(1, HC, 32, H))
        self.emb_in = self._dev(z(1, HC, 1, H), layout=RM)
        self.ple_in = self._dev(z(1, HC, 1, H), layout=RM)
        self.cos_in = self._dev(torch.ones(1, 1, 1, 256), layout=RM)
        self.sin_in = self._dev(z(1, 1, 1, 256), layout=RM)
        self.pos_in = self._dev(torch.zeros(2, dtype=torch.int32), dtype=ttnn.int32, layout=RM)
        self.ycold_in = self._dev(z(1, 1, 1, H), layout=RM)
        self.sp_in, self.wsel_in = {}, {}
        if self.experts is not None:
            for L, ids in self.experts.sets.items():
                if ids:
                    self.sp_in[L] = self._dev(z(1, 1, 1, len(ids)), dtype=ttnn.bfloat16, layout=RM)
                    self.wsel_in[L] = self._dev(z(1, 1, 1, 32 * len(ids)), dtype=ttnn.bfloat16, layout=RM)
        self.S, self.hist, self.kc, self.vc = {}, {}, {}, {}
        for L in range(c.n_layers):
            if is_attention(L):
                self.kc[L] = self._dev(z(2, 1, self.s_max, 256), dtype=ttnn.bfloat16)
                self.vc[L] = self._dev(z(2, 1, self.s_max, 256), dtype=ttnn.bfloat16)
            else:
                self.S[L] = self._dev(z(1, 48, 128, 128))
                self.hist[L] = [self._dev(z(1, 1, 32, 10240)) for _ in range(3)]
        self._zS = self._dev(z(1, 48, 128, 128))
        self._zh = self._dev(z(1, 1, 32, 10240))
        self.kv_shard = ttnn.create_sharded_memory_config(
            shape=(32, 256), core_grid=ttnn.CoreGrid(x=2, y=1), strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        self.sdpa_cfg = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                               q_chunk_size=0, k_chunk_size=128)

    def reset(self):
        super().reset()
        if getattr(self, "S", None):          # zero the recurrent state in place (KV is masked by position)
            for L in self.S:
                ttnn.copy(self._zS, self.S[L])
                for h in self.hist[L]:
                    ttnn.copy(self._zh, h)

    # ---- layer pieces (device ops only)
    def gdn_t(self, L, x):
        p = f"blk.{L}.gdn."
        qkv = self._mm(x, self.w[p + "qkv"])
        h0, h1, h2 = self.hist[L]
        conv = ttnn.multiply(qkv, self.w[p + "conv3"])
        for k, h in enumerate((h0, h1, h2)):
            conv = ttnn.add(conv, ttnn.multiply(h, self.w[f"{p}conv{k}"]))
        ttnn.copy(h1, h0)
        ttnn.copy(h2, h1)
        ttnn.copy(qkv, h2)
        conv = ttnn.silu(conv)
        q = ttnn.repeat(self._heads(conv, 16, 128, 0), ttnn.Shape([1, 1, 3, 1]))
        k = ttnn.repeat(self._heads(conv, 16, 128, 2048), ttnn.Shape([1, 1, 3, 1]))
        v = self._heads(conv, 48, 128, 4096)
        ab = self._mm(x, self.w[p + "ab"])
        alpha = self._heads(ab, 1, 48, 0)
        beta = ttnn.sigmoid(self._heads(ab, 1, 48, 48))
        g = ttnn.multiply(ttnn.softplus(ttnn.add(alpha, self.w[p + "dt"])), self.w[p + "a"])
        g = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(g, RM), [1, 1, 48]), TILE)
        beta = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(beta, RM), [1, 1, 48]), TILE)
        o, S = self._delta(q, k, v, beta, g, scale=None, initial_state=self.S[L], device=self.dev,
                           high_precision=True)
        ttnn.copy(ttnn.to_memory_config(S, ttnn.DRAM_MEMORY_CONFIG), self.S[L])
        o = ttnn.reshape(o, [1, 1, 48, 128])
        o = ttnn.multiply(ttnn.rms_norm(o, epsilon=self.cfg.eps), self.w[p + "norm"])
        zt = self._heads(self._mm(x, self.w[p + "z"]), 48, 128)
        return self._mm(self._flat(ttnn.multiply(o, ttnn.sigmoid(zt)), 48, 128), self.w[p + "out"])

    def attn_t(self, L, x, cos, sin):
        p = f"blk.{L}.attn."
        qg = self._mm(x, self.w[p + "q"])
        rm = ttnn.reshape(ttnn.slice(ttnn.to_layout(qg, RM), [0, 0, 0, 0], [1, 1, 1, 12288]), [1, 1, 24, 512])
        q = ttnn.to_layout(ttnn.slice(rm, [0, 0, 0, 0], [1, 1, 24, 256]), TILE)
        gate = ttnn.to_layout(ttnn.slice(rm, [0, 0, 0, 256], [1, 1, 24, 512]), TILE)
        q = self._rope(ttnn.multiply(ttnn.rms_norm(q, epsilon=self.cfg.eps), self.w[p + "q_norm"]), cos, sin)
        kvr = ttnn.to_layout(self._heads(self._mm(x, self.w[p + "kv"]), 4, 256), RM)
        k = ttnn.to_layout(ttnn.slice(kvr, [0, 0, 0, 0], [1, 1, 2, 256]), TILE)
        v = ttnn.slice(kvr, [0, 0, 2, 0], [1, 1, 4, 256])
        k = self._rope(ttnn.multiply(ttnn.rms_norm(k, epsilon=self.cfg.eps), self.w[p + "k_norm"]), cos, sin)
        k = ttnn.reshape(ttnn.slice(ttnn.to_layout(k, RM), [0, 0, 0, 0], [1, 1, 2, 256]), [1, 2, 1, 256])
        v = ttnn.reshape(v, [1, 2, 1, 256])
        for src, cache in ((k, self.kc[L]), (v, self.vc[L])):
            sh = ttnn.to_layout(ttnn.pad(ttnn.typecast(src, ttnn.bfloat16) if src.dtype != ttnn.bfloat16 else src,
                                         [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0), TILE)
            ttnn.experimental.paged_update_cache(cache, ttnn.to_memory_config(sh, self.kv_shard),
                                                 update_idxs_tensor=self.pos_in)
        qh = ttnn.reshape(ttnn.slice(ttnn.to_layout(q, RM), [0, 0, 0, 0], [1, 1, 24, 256]), [1, 2, 12, 256])
        qh = ttnn.to_layout(ttnn.typecast(qh, ttnn.bfloat16) if qh.dtype != ttnn.bfloat16 else qh, TILE)
        o = ttnn.transformer.scaled_dot_product_attention_decode(
            qh, self.kc[L], self.vc[L], cur_pos_tensor=self.pos_in, scale=1.0 / math.sqrt(256),
            program_config=self.sdpa_cfg, memory_config=ttnn.DRAM_MEMORY_CONFIG)                 # [1,2,12,256]
        o = ttnn.reshape(ttnn.slice(ttnn.to_layout(o, RM), [0, 0, 0, 0], [1, 2, 12, 256]), [1, 1, 24, 256])
        o = ttnn.multiply(ttnn.to_layout(o, TILE), ttnn.sigmoid(gate))
        return self._mm(self._flat(o, 24, 256), self.w[p + "out"])

    def seg_a(self, L):
        if L == 0:
            R = ttnn.to_layout(ttnn.pad(self.emb_in, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0), TILE)
        else:
            R = self.R
        if L == self.cfg.ple_layer:
            R = ttnn.add(R, ttnn.to_layout(ttnn.pad(self.ple_in, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0), TILE))
        x, Rn = self.gr_read(R, f"blk.{L}.hc_attn")
        if is_attention(L):
            cos, sin = ttnn.to_layout(self.cos_in, TILE), ttnn.to_layout(self.sin_in, TILE)
            y = self.attn_t(L, x, cos, sin)
        else:
            y = self.gdn_t(L, x)
        R2 = self.gr_write(R, Rn, y, f"blk.{L}.hc_attn")
        x2, Rn2 = self.gr_read(R2, f"blk.{L}.hc_ffn")
        xrow = ttnn.slice(ttnn.to_layout(ttnn.typecast(x2, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, 1, 1, H])
        return {"R2": R2, "Rn2": Rn2, "x2": x2, "xrow": xrow}

    def seg_b1(self, L, a):
        """resident experts + shared expert: needs only the routing rows, so it runs while the CPU does the cold experts."""
        p = f"blk.{L}."
        x = a["x2"]
        gu = self._mm(x, self.w[p + "sh.gu"])
        h = ttnn.multiply(ttnn.silu(ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 32, 640])),
                          ttnn.slice(gu, [0, 0, 0, 640], [1, 1, 32, 1280]))
        sh = self._mm(h, self.w[p + "sh.down"])
        sg = ttnn.sigmoid(ttnn.slice(self._mm(x, self.w[p + "sh.gate"]), [0, 0, 0, 0], [1, 1, 32, 1]))
        y = ttnn.multiply(sh, sg)
        if L in self.sp_in:
            y = ttnn.add(y, self.experts.forward_sp(L, x, self.sp_in[L], self._pad32(self.wsel_in[L])))
        return {"y": y}

    def seg_b2(self, L, a, b1):
        y = ttnn.add(b1["y"], self._pad32(self.ycold_in))
        R3 = self.gr_write(a["R2"], a["Rn2"], y, f"blk.{L}.hc_ffn")
        ttnn.copy(R3, self.R)
        return {}

    def seg_final(self):
        h, _ = self.gr_read(self.R, "out_hc")
        lg = self._mm(h, self.w["lm_head"])
        rm = ttnn.slice(ttnn.to_layout(ttnn.typecast(lg, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, 1, 1, lg.shape[-1]])
        return {"logits": rm, "argmax": ttnn.argmax(rm, dim=-1)}

    # ---- host side of a token
    def _write(self, dst, t: torch.Tensor, dtype=ttnn.float32):
        ttnn.copy_host_to_device_tensor(ttnn.from_torch(t.contiguous(), dtype=dtype, layout=RM), dst)
        self.io["h2d_bytes"] += t.numel() * t.element_size()

    def _read(self, t) -> torch.Tensor:
        t0 = time.perf_counter()
        x = ttnn.to_torch(t)
        self.io["d2h_s"] += time.perf_counter() - t0
        self.io["d2h_n"] += 1
        self.io["d2h_bytes"] += x.numel() * x.element_size()
        return x

    def _route(self, L, xh):
        """fp32 routing; writes the resident rows. Returns the cold experts and weights."""
        if L not in self.router:
            self.router[L] = self.W.tensor(f"blk.{L}.ffn_gate_inp.weight")
        logits = self.router[L] @ xh
        pr = np.exp(logits - logits.max())
        pr /= pr.sum()
        ids = np.argsort(-pr, kind="stable")[: self.cfg.top_k]
        wts = pr[ids] / pr[ids].sum()
        if self.experts is not None:
            res, cids, cwts = self.experts.split(L, ids, wts)
            if L in self.sp_in:
                w = torch.zeros(1, 1, 1, len(self.experts.sets[L]))
                for i, wt in res:
                    w[0, 0, 0, i] = wt
                self._write(self.sp_in[L], w, ttnn.bfloat16)
                ws = torch.zeros(1, 1, 1, 32 * w.shape[-1])
                ws[0, 0, 0, ::32] = w[0, 0, 0]
                self._write(self.wsel_in[L], ws, ttnn.bfloat16)
            self.stats["resident"] += len(res)
        else:
            cids, cwts = ids, wts
        self.stats["cold"] += len(cids)
        return cids, cwts

    def _cold(self, L, xh, cids, cwts):
        t0 = time.perf_counter()
        y = self.hx.moe(L, cids, cwts, xh) if len(cids) else np.zeros(H, np.float32)
        self.phase["hostexp_only"] = self.phase.get("hostexp_only", 0) + time.perf_counter() - t0
        self._write(self.ycold_in, torch.from_numpy(y).reshape(1, 1, 1, H))

    def _inputs(self, token):
        c = self.cfg
        e = torch.from_numpy(self.W.rows("token_embd.weight", [token])[0].copy())
        self._write(self.emb_in, e.repeat(HC).reshape(1, HC, 1, H))
        ang = self.pos * (c.rope_theta ** (-torch.arange(0, c.rope_dim, 2, dtype=torch.float64) / c.rope_dim))
        cos, sin = torch.ones(256), torch.zeros(256)
        cos[: c.rope_dim] = torch.cat([ang.cos(), ang.cos()]).float()
        sin[: c.rope_dim] = torch.cat([ang.sin(), ang.sin()]).float()
        self._write(self.cos_in, cos.reshape(1, 1, 1, 256))
        self._write(self.sin_in, sin.reshape(1, 1, 1, 256))
        self._write(self.pos_in, torch.tensor([self.pos, self.pos], dtype=torch.int32), ttnn.int32)

    def _ple(self, token):
        if self.traces is None:
            R = self._read(ttnn.slice(ttnn.to_layout(ttnn.typecast(self.R, ttnn.bfloat16), RM), [0, 0, 0, 0],
                                      [1, HC, 1, H]))
        else:
            self._ple_refresh()
            R = self._read(self._ple_rrow)
        d, self.ple_state = self.ref.ple(R.float().reshape(1, -1), [token], self.ple_state)
        self._write(self.ple_in, d[0].reshape(1, HC, 1, H))

    def _run(self, name, fn, *args):
        """Eager before capture; afterwards replay the captured trace and return its persistent outputs."""
        if self.traces is None:
            return fn(*args)
        tid, out = self.traces[name]
        ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
        return out

    def step(self, token: int, greedy: bool = True, **_):
        if self.pos >= min(self.s_max, self.max_ctx):
            raise RuntimeError("context limit of the traced dense-attention path")
        ph = self.phase
        t = time.perf_counter()
        self._inputs(token)
        t1 = time.perf_counter(); ph["inputs"] = ph.get("inputs", 0) + t1 - t; t = t1
        for L in range(self.cfg.n_layers):
            if L == self.cfg.ple_layer:
                self._ple(token)
                t1 = time.perf_counter(); ph["ple"] = ph.get("ple", 0) + t1 - t; t = t1
            a = self._run(f"a{L}", self.seg_a, L)
            t1 = time.perf_counter(); ph["enqueue"] = ph.get("enqueue", 0) + t1 - t; t = t1
            xh = self._read(a["xrow"]).float().reshape(-1).numpy()
            t1 = time.perf_counter(); ph["wait_x"] = ph.get("wait_x", 0) + t1 - t; t = t1
            cids, cwts = self._route(L, xh)
            b1 = self._run(f"b1_{L}", self.seg_b1, L, a)
            t1 = time.perf_counter(); ph["route+writes"] = ph.get("route+writes", 0) + t1 - t; t = t1
            self._cold(L, xh, cids, cwts)
            t1 = time.perf_counter(); ph["cold+write"] = ph.get("cold+write", 0) + t1 - t; t = t1
            self._run(f"b2_{L}", self.seg_b2, L, a, b1)
            t1 = time.perf_counter(); ph["enqueue"] = ph.get("enqueue", 0) + t1 - t; t = t1
        f = self._run("final", self.seg_final)
        self.pos += 1
        self.tokens.append(token)
        out = int(self._read(f["argmax"]).reshape(-1)[0]) if greedy else self._read(f["logits"]).float().reshape(-1)
        t1 = time.perf_counter(); ph["final"] = ph.get("final", 0) + t1 - t
        return out

    def capture(self, token: int = 198):
        """Run one eager token (compiles every program, loads every weight), then capture all segments."""
        self.reset()
        self.step(token)
        self.reset()
        self._ple_rrow = ttnn.slice(ttnn.to_layout(ttnn.typecast(self.R, ttnn.bfloat16), RM), [0, 0, 0, 0],
                                    [1, HC, 1, H])
        traces = {}
        for L in range(self.cfg.n_layers):
            for name, fn, args in ((f"a{L}", self.seg_a, (L,)), (f"b1_{L}", self.seg_b1, "b1"),
                                   (f"b2_{L}", self.seg_b2, "b2")):
                if args == "b1":
                    args = (L, traces[f"a{L}"][1])
                elif args == "b2":
                    args = (L, traces[f"a{L}"][1], traces[f"b1_{L}"][1])
                tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
                out = fn(*args)
                ttnn.end_trace_capture(self.dev, tid, cq_id=0)
                traces[name] = (tid, out)
        tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        out = self.seg_final()
        ttnn.end_trace_capture(self.dev, tid, cq_id=0)
        traces["final"] = (tid, out)
        # the R row the host reads for PLE must be refreshed after b0: capture it as its own tiny trace
        tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
        rrow = ttnn.slice(ttnn.to_layout(ttnn.typecast(self.R, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, HC, 1, H])
        ttnn.end_trace_capture(self.dev, tid, cq_id=0)
        traces["ple_row"] = (tid, rrow)
        self._ple_rrow = rrow
        self.traces = traces
        self.reset()

    def _ple_refresh(self):
        tid, _ = self.traces["ple_row"]
        ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
