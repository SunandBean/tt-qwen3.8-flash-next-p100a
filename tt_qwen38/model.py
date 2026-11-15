"""Qwen3.8-Flash-Next Coder decode on one P100a, dense path on the device, routed experts on the host CPU.

One token per step. Every vector is a [1,1,32,N] TILE tensor whose row 0 is the token (rows 1..31 stay zero).
The 4-branch hyper-connection residual is kept as four fp32 [1,1,32,2560] tensors.

Device: hyper-connection reads/writes, GDN (tt-metal's decode delta-rule helper), gated attention (dense; exact up to
2,050 tokens, where Strata's indexer starts to drop blocks), shared expert, final mixer, LM head.
Host: token embedding, PLE (layer 1), router (fp32, exact) and routed experts (hostexp, ggml dot products).
Formulas and GGUF conventions: tt_qwen38/ref/model.py (the M1 float32 reference).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import ttnn

from tt_qwen38.hostexp import HostExperts
from tt_qwen38.ref.gguf_weights import Weights
from tt_qwen38.ref.model import Config, PLEState, Reference

from .weights import DeviceWeights, is_attention

H, HC = 2560, 4
RM, TILE = ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT


class P100aModel:
    def __init__(self, device, W: Weights | None = None, threads: int = 8, max_ctx: int = 2048, experts=None):
        self.dev = device
        self.experts = experts            # device.experts.DeviceExperts or None (all routed experts on the CPU)
        self.W = W or Weights()
        self.cfg = Config.from_weights(self.W)
        self.w = DeviceWeights(device)
        self.ref = Reference(self.W, threads=threads)        # host-side PLE math and embedding rows
        self.hx = HostExperts(self.W, threads=int(os.environ.get("P100A_HX_THREADS") or threads))   # F5 sweep knob
        self.router = {}
        self.max_ctx = max_ctx
        self._mm1d = {}
        self.ck = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                   fp32_dest_acc_en=True, packer_l1_acc=True)
        # rotate-half on dims 0..63 of a 256-wide head: x @ P = [-x[32:64], x[0:32], 0...]
        P = torch.zeros(256, 256)
        r = self.cfg.rope_dim // 2
        for i in range(r):
            P[i + r, i] = -1.0
            P[i, i + r] = 1.0
        self.rope_P = self._up(P.reshape(1, 1, 256, 256), ttnn.bfloat16)
        m = torch.zeros(1, 1, 32, H)
        m[0, 0, 0] = 1
        m4 = torch.zeros(1, HC, 32, H)
        m4[:, :, 0] = 1
        self._row0_4 = self._up(m4)
        self._zero_4 = self._up(torch.zeros(1, HC, 32, H))
        self._grw = {}
        # PLE runs on the host every token: keep its weights dequantized instead of re-reading the GGUF each call
        _t, cache = self.ref.t, {}
        self.ref.t = lambda n: cache[n] if n in cache else cache.setdefault(n, _t(n)) if ".ple_" in n else _t(n)
        from models.experimental.gated_attention_gated_deltanet.tt.ttnn_delta_rule_ops import (
            recurrent_gated_delta_rule_decode_ttnn,
        )
        self._delta = recurrent_gated_delta_rule_decode_ttnn
        self.reset()
        self.timing = {}
        self.stats = {"resident": 0, "cold": 0}
        self.io = {"h2d_bytes": 0, "d2h_bytes": 0, "d2h_n": 0, "d2h_s": 0.0}
        self.profile = {} if os.environ.get("P100A_PROFILE") == "1" else None
        self.debug_experts = os.environ.get("P100A_DEBUG_EXPERTS") == "1"
        self.debug_log = []

    # ---- small helpers
    def _up(self, t: torch.Tensor, dtype=ttnn.float32, layout=TILE):
        return ttnn.from_torch(t.contiguous(), dtype=dtype, layout=layout, device=self.dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def _pad32(self, rm):
        """row-major [1,1,r,N] -> TILE [1,1,32,N] with rows r..31 exactly zero (plain tilize leaves them undefined)."""
        r = rm.shape[2]
        if r < 32:
            rm = ttnn.pad(rm, [(0, 0), (0, 0), (0, 32 - r), (0, 0)], 0.0)
        return ttnn.to_layout(rm, TILE)

    def _vec(self, v: torch.Tensor, dtype=ttnn.float32):
        """[N] host vector -> [1,1,32,N] tile, row 0 (uploads only N values; zero padding made on the device)."""
        rm = ttnn.from_torch(v.float().reshape(1, 1, 1, -1).contiguous(), dtype=dtype, layout=RM, device=self.dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.io["h2d_bytes"] += v.numel() * (4 if dtype == ttnn.float32 else 2)
        return self._pad32(rm)

    def _bvec(self, v: torch.Tensor, dtype=ttnn.float32):
        """[N] host vector -> [1,1,1,N] tile that broadcasts over rows (compact upload)."""
        rm = ttnn.from_torch(v.float().reshape(1, 1, 1, -1).contiguous(), dtype=dtype, layout=RM, device=self.dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.io["h2d_bytes"] += v.numel() * (4 if dtype == ttnn.float32 else 2)
        return ttnn.to_layout(rm, TILE)

    def _row(self, t, n: int | None = None) -> torch.Tensor:
        """row 0 of a [1,1,32,N] tile -> host fp32 [N]; moves only row 0, as bf16."""
        t16 = ttnn.typecast(t, ttnn.bfloat16) if t.dtype != ttnn.bfloat16 else t
        rm = ttnn.to_layout(t16, RM)
        N = rm.shape[-1]
        rm = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, 1, N])
        t0 = time.perf_counter()
        x = ttnn.to_torch(rm).float().reshape(-1)
        self.io["d2h_s"] += time.perf_counter() - t0
        self.io["d2h_bytes"] += N * 2
        self.io["d2h_n"] += 1
        return x if n is None else x[:n]

    def _argmax(self, t) -> int:
        rm = ttnn.to_layout(t, RM)
        rm = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, 1, rm.shape[-1]])
        i = ttnn.argmax(rm, dim=-1)
        t0 = time.perf_counter()
        v = int(ttnn.to_torch(i).reshape(-1)[0])
        self.io["d2h_s"] += time.perf_counter() - t0
        self.io["d2h_bytes"] += 4
        self.io["d2h_n"] += 1
        return v

    def _heads(self, t, n_rows: int, width: int, start: int = 0):
        """row 0 of [1,1,32,N] -> tile [1,1,n_rows,width] from row-0 columns [start, start + n_rows*width)."""
        rm = ttnn.to_layout(t, RM)
        rm = ttnn.slice(rm, [0, 0, 0, start], [1, 1, 1, start + n_rows * width])
        rm = ttnn.reshape(rm, [1, 1, n_rows, width])
        return ttnn.to_layout(rm, TILE)

    def _flat(self, t, n_rows: int, width: int):
        """tile [1,1,n_rows,width] -> tile [1,1,32,n_rows*width] (row 0)."""
        rm = ttnn.to_layout(t, RM)
        rm = ttnn.slice(rm, [0, 0, 0, 0], [1, 1, n_rows, width])
        rm = ttnn.reshape(rm, [1, 1, 1, n_rows * width])
        return self._pad32(rm)

    def _ckf(self):
        if getattr(self, "_ckf_", None) is None:
            self._ckf_ = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                          fp32_dest_acc_en=True, packer_l1_acc=True)
        return self._ckf_

    def _rms(self, x):
        """fp32 RMS over the last dim (ttnn.rms_norm on a 2560-wide fp32 tile does not fit one core's L1)."""
        ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True)
        return ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, self.cfg.eps)))

    # S2 (P100A_FUSED contains "mm1d"): explicit 1D (mcast_in0) program configs for the small decode matmuls, cores from
    # tt_qwen38/mm_bench.py (20261111-1603-s2-mm-bench.json); the big projections stay on the default (~300 GB/s already)
    MM1D = {(6144, 2560): 88, (2560, 1024): 66, (2560, 1280): 88, (640, 2560): 55, (2560, 128): 33, (2560, 512): 8,
            (2560, 6144): 66}

    def _mm_cfg(self, a, b):
        if "mm1d" not in os.environ.get("P100A_FUSED", "gdn,moe,hcw,commit,mm1d,kvw,conv,abg").split(","):
            return None
        sh = b.shape
        if len(sh) != 4 or sh[0] * sh[1] != 1 or a.shape[-2] != 32 or a.shape[0] * a.shape[1] != 1:
            return None
        k, n = sh[-2], sh[-1]
        cores = self.MM1D.get((k, n))
        if cores is None:
            return None
        key = (k, n, cores)
        if key not in self._mm1d:
            g = self.dev.compute_with_storage_grid_size()
            cols = min(g.x, cores)
            rows = math.ceil(cores / cols)
            kt, nt = k // 32, math.ceil(n / 32)
            pn = math.ceil(nt / (cols * rows))
            self._mm1d[key] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(cols, rows), in0_block_w=max(d for d in range(1, 9) if kt % d == 0),
                out_subblock_h=1, out_subblock_w=max(i for i in range(1, 5) if pn % i == 0), per_core_M=1,
                per_core_N=pn, fuse_batch=True, fused_activation=None, mcast_in0=True)
        return self._mm1d[key]

    def _mm(self, a, b, dtype=ttnn.float32):
        if a.dtype != ttnn.bfloat16:
            a = ttnn.typecast(a, ttnn.bfloat16)
        pc = self._mm_cfg(a, b)
        if pc is not None:
            return ttnn.matmul(a, b, dtype=dtype, compute_kernel_config=self.ck, program_config=pc)
        return ttnn.matmul(a, b, dtype=dtype, compute_kernel_config=self.ck)

    # ---- state
    def reset(self):
        self.pos = 0
        self.tokens: list[int] = []
        self.gdn_S = {}
        self.gdn_hist = {}
        self.kv = {}
        self.ple_state: PLEState | None = None
        self._mask = None

    # ---- blocks
    # The residual is one fp32 tensor [1, 4, 32, 2560]: branch b in dim 1, the token in row 0 (rows 1..31 zero).
    def _gr_w(self, prefix):
        """Stacked hyper-connection weights for `prefix`, built once on the device from the per-branch tensors."""
        if prefix not in self._grw:
            cat = lambda k: ttnn.concat([self.w[f"{prefix}.{k}{b}"] for b in range(HC)], dim=1)
            # the 1/HC of the branch mean (read), of silu(down/HC) and of sigmoid(inject/HC) all fold into the norm
            # weights: Rn' = Rn/HC makes every one of those three multiplies exact no-ops (M8, 3 ops per pair)
            self._grw[prefix] = {"norm": ttnn.multiply(cat("norm"), 1.0 / HC), "down": cat("down"), "up": cat("up"),
                                 "inj": cat("inj") if not prefix.endswith("out_hc") else None}
        return self._grw[prefix]

    def gr_read(self, R, prefix):
        w = self._gr_w(prefix)
        if os.environ.get("P100A_GR_FUSED") == "1":      # experiment: fused bf16 rms_norm (one op instead of five)
            Rn = ttnn.multiply(ttnn.rms_norm(ttnn.typecast(R, ttnn.bfloat16), epsilon=self.cfg.eps), w["norm"])
        else:   # fused fp32 RMS at HiFi4 (0.047 vs 0.077 ms, 5.7e-4 rel. from the 5-op form; default fidelity 1.5e-2)
            Rn = ttnn.multiply(ttnn.rms_norm(R, epsilon=self.cfg.eps, compute_kernel_config=self._ckf()), w["norm"])
        lo = ttnn.silu(ttnn.sum(self._mm(Rn, w["down"]), dim=1, keepdim=True))            # [1,1,32,320]
        g = ttnn.sigmoid(self._mm(ttnn.repeat(lo, ttnn.Shape([1, HC, 1, 1])), w["up"]))   # [1,4,32,2560]
        x = ttnn.sum(ttnn.multiply(g, Rn), dim=1, keepdim=True)                           # [1,1,32,2560]
        return x, Rn                                                                       # Rn is Rn/HC

    def gr_write(self, R, Rn, y, prefix):
        w = self._gr_w(prefix)
        c = ttnn.sum(self._mm(Rn, w["inj"]), dim=1, keepdim=True)                         # [1,1,32,32], cols 0..3
        coef = ttnn.multiply(ttnn.sigmoid(c), 2.0)                                          # c = inject/HC
        coef = ttnn.slice(ttnn.permute(coef, (0, 3, 2, 1)), [0, 0, 0, 0], [1, HC, 32, 1])  # [1,4,32,1]
        out = ttnn.add(R, ttnn.multiply(ttnn.repeat(y, ttnn.Shape([1, HC, 1, 1])), coef))
        # keep the padding rows at exactly zero (tilize padding is not zeroed; see experts.py)
        return ttnn.where(self._row0_4, out, self._zero_4)

    def _R_from_host(self, v: torch.Tensor):
        """host [10240] -> residual [1,4,32,2560] (4 x 2560 values uploaded)."""
        rm = ttnn.from_torch(v.float().reshape(1, HC, 1, H).contiguous(), dtype=ttnn.float32, layout=RM,
                             device=self.dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.io["h2d_bytes"] += v.numel() * 4
        return ttnn.to_layout(ttnn.pad(rm, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0), TILE)

    def _R_to_host(self, R) -> torch.Tensor:
        rm = ttnn.to_layout(ttnn.typecast(R, ttnn.bfloat16), RM)
        rm = ttnn.slice(rm, [0, 0, 0, 0], [1, HC, 1, H])
        t0 = time.perf_counter()
        x = ttnn.to_torch(rm).float().reshape(-1)
        self.io["d2h_s"] += time.perf_counter() - t0
        self.io["d2h_bytes"] += HC * H * 2
        self.io["d2h_n"] += 1
        return x

    def gdn(self, L, x):
        p = f"blk.{L}.gdn."
        qkv = self._mm(x, self.w[p + "qkv"])                                              # [1,1,32,10240]
        hist = self.gdn_hist.get(L)
        if hist is None:
            z = ttnn.multiply(qkv, 0.0)
            hist = [z, z, z]
        conv = ttnn.multiply(qkv, self.w[p + "conv3"])
        for k in range(3):
            conv = ttnn.add(conv, ttnn.multiply(hist[k], self.w[f"{p}conv{k}"]))
        self.gdn_hist[L] = [hist[1], hist[2], qkv]
        conv = ttnn.silu(conv)
        q = ttnn.repeat(self._heads(conv, 16, 128, 0), ttnn.Shape([1, 1, 3, 1]))          # head h -> key h % 16
        k = ttnn.repeat(self._heads(conv, 16, 128, 2048), ttnn.Shape([1, 1, 3, 1]))
        v = self._heads(conv, 48, 128, 4096)
        ab = self._mm(x, self.w[p + "ab"])                                                # [1,1,32,128]
        alpha = self._heads(ab, 1, 48, 0)
        beta = ttnn.sigmoid(self._heads(ab, 1, 48, 48))
        g = ttnn.multiply(ttnn.softplus(ttnn.add(alpha, self.w[p + "dt"])), self.w[p + "a"])   # log decay [1,1,1,48]
        g = ttnn.reshape(ttnn.to_layout(g, RM), [1, 1, 48])
        beta = ttnn.reshape(ttnn.to_layout(beta, RM), [1, 1, 48])
        g, beta = ttnn.to_layout(g, TILE), ttnn.to_layout(beta, TILE)
        o, S = self._delta(q, k, v, beta, g, scale=None, initial_state=self.gdn_S.get(L), device=self.dev,
                           high_precision=True)
        # the helper leaves the state in L1; 36 layers x 3.1 MB would fill the L1 heap, so park it in DRAM
        self.gdn_S[L] = ttnn.to_memory_config(S, ttnn.DRAM_MEMORY_CONFIG)
        o = ttnn.reshape(o, [1, 1, 48, 128])
        o = ttnn.multiply(ttnn.rms_norm(o, epsilon=self.cfg.eps), self.w[p + "norm"])
        zt = self._heads(self._mm(x, self.w[p + "z"]), 48, 128)
        y = ttnn.multiply(o, ttnn.sigmoid(zt))
        return self._mm(self._flat(y, 48, 128), self.w[p + "out"])

    def _rope(self, t, cos, sin):
        return ttnn.add(ttnn.multiply(t, cos), ttnn.multiply(self._mm(t, self.rope_P), sin))

    def qsa(self, L, x, cos, sin):
        p = f"blk.{L}.attn."
        if self.pos >= self.max_ctx:
            raise RuntimeError("M2 attention is dense; contexts past 2,048 tokens need the indexer")
        qg = self._mm(x, self.w[p + "q"])                                                 # [1,1,32,12288]
        rm = ttnn.reshape(ttnn.slice(ttnn.to_layout(qg, RM), [0, 0, 0, 0], [1, 1, 1, 12288]), [1, 1, 24, 512])
        q = ttnn.to_layout(ttnn.slice(rm, [0, 0, 0, 0], [1, 1, 24, 256]), TILE)
        gate = ttnn.to_layout(ttnn.slice(rm, [0, 0, 0, 256], [1, 1, 24, 512]), TILE)
        q = self._rope(ttnn.multiply(ttnn.rms_norm(q, epsilon=self.cfg.eps), self.w[p + "q_norm"]), cos, sin)
        kv = self._heads(self._mm(x, self.w[p + "kv"]), 4, 256)                           # rows k0 k1 v0 v1
        kvr = ttnn.to_layout(kv, RM)
        k = ttnn.to_layout(ttnn.slice(kvr, [0, 0, 0, 0], [1, 1, 2, 256]), TILE)
        v = ttnn.slice(kvr, [0, 0, 2, 0], [1, 1, 4, 256])
        k = self._rope(ttnn.multiply(ttnn.rms_norm(k, epsilon=self.cfg.eps), self.w[p + "k_norm"]), cos, sin)
        k = ttnn.reshape(ttnn.slice(ttnn.to_layout(k, RM), [0, 0, 0, 0], [1, 1, 2, 256]), [1, 2, 1, 256])
        v = ttnn.reshape(v, [1, 2, 1, 256])
        Kc, Vc = self.kv.get(L, (None, None))
        Kc = k if Kc is None else ttnn.concat([Kc, k], dim=2)                             # RM [1,2,P,256]
        Vc = v if Vc is None else ttnn.concat([Vc, v], dim=2)
        self.kv[L] = (Kc, Vc)
        P = self.pos + 1
        Kt, Vt = ttnn.to_layout(Kc, TILE), ttnn.to_layout(Vc, TILE)
        qh = ttnn.reshape(ttnn.slice(ttnn.to_layout(q, RM), [0, 0, 0, 0], [1, 1, 24, 256]), [1, 2, 12, 256])
        qh = ttnn.to_layout(qh, TILE)
        sc = ttnn.multiply(self._mm(qh, ttnn.transpose(Kt, -2, -1)), 1.0 / math.sqrt(256))   # [1,2,32,Ppad]
        Ppad = sc.shape[-1]
        if Ppad > P:
            if self._mask is None or self._mask[0] != (P, Ppad):
                m = torch.zeros(Ppad)
                m[P:] = -1e30
                rm = ttnn.from_torch(m.reshape(1, 1, 1, Ppad), dtype=ttnn.bfloat16, layout=RM, device=self.dev)
                self._mask = ((P, Ppad), ttnn.to_layout(rm, TILE))            # one row, broadcast over the 32 rows
            sc = ttnn.add(sc, self._mask[1])
        pr = ttnn.softmax(sc, dim=-1)
        o = self._mm(pr, Vt)                                                              # [1,2,32,256]
        o = ttnn.reshape(ttnn.slice(ttnn.to_layout(o, RM), [0, 0, 0, 0], [1, 2, 12, 256]), [1, 1, 24, 256])
        o = ttnn.multiply(ttnn.to_layout(o, TILE), ttnn.sigmoid(gate))
        return self._mm(self._flat(o, 24, 256), self.w[p + "out"])

    def moe(self, L, x):
        p = f"blk.{L}."
        t0 = time.perf_counter()
        xh = self._row(x).numpy()
        if L not in self.router:
            self.router[L] = self.W.tensor(p + "ffn_gate_inp.weight")                      # [256, 2560] fp32
        logits = self.router[L] @ xh
        pr = np.exp(logits - logits.max())
        pr /= pr.sum()
        ids = np.argsort(-pr, kind="stable")[: self.cfg.top_k]
        wts = pr[ids] / pr[ids].sum()
        self.routing.setdefault(L, []).append(ids.tolist())
        y_dev = None
        if self.experts is not None:
            res, cids, cwts = self.experts.split(L, ids, wts)
            self.stats["resident"] += len(res)
            self.stats["cold"] += len(cids)
            y_dev = self.experts.forward(L, x, res)          # enqueued on the device; the CPU works meanwhile
        else:
            cids, cwts = ids, wts
            self.stats["cold"] += len(cids)
        y = self.hx.moe(L, cids, cwts, xh) if len(cids) else np.zeros(H, np.float32)
        if self.debug_experts and y_dev is not None:
            rids = [self.experts.sets[L][i] for i, _ in res]
            ref = self.hx.moe(L, rids, [w for _, w in res], xh)
            got = self._row(y_dev).numpy()
            y2 = self._row(self.experts.forward(L, self._vec(torch.from_numpy(xh), ttnn.bfloat16), res)).numpy()
            x16 = ttnn.typecast(x, ttnn.bfloat16)
            self.debug_x = (float(np.abs(self._row(x16).numpy() - xh).max()), str(x.memory_config()), str(x.dtype),
                            float(ttnn.to_torch(x16).float()[0, 0, 1:].abs().max()))
            got = np.stack([got, y2])
            self.debug_log.append((self.pos, L, len(res), [float(np.linalg.norm(g - ref) / (np.linalg.norm(ref) + 1e-30)) for g in got],
                                   float(np.linalg.norm(ref)), self.debug_x))
        t1 = time.perf_counter()
        gu = self._mm(x, self.w[p.replace("blk.", "blk.") + "sh.gu"])
        h = ttnn.multiply(ttnn.silu(ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 32, 640])),
                          ttnn.slice(gu, [0, 0, 0, 640], [1, 1, 32, 1280]))
        sh = self._mm(h, self.w[p + "sh.down"])
        sg = ttnn.sigmoid(ttnn.slice(self._mm(x, self.w[p + "sh.gate"]), [0, 0, 0, 0], [1, 1, 32, 1]))
        out = ttnn.add(ttnn.multiply(sh, sg), self._vec(torch.from_numpy(y)))
        if y_dev is not None:
            out = ttnn.add(out, y_dev)
        self.timing["host_moe_s"] = self.timing.get("host_moe_s", 0.0) + (t1 - t0)
        return out

    def ple(self, R, token):
        Rh = self._R_to_host(R)[None]                                                      # [1, 10240]
        d, self.ple_state = self.ref.ple(Rh, [token], self.ple_state)
        return ttnn.add(R, self._R_from_host(d[0]))

    # ---- one token
    def step(self, token: int, observe=None, inject=None, greedy: bool = False):
        """Run one token; returns logits [V] (host fp32). observe(L, R) sees each layer's residual;
        inject(L) may return a host [10240] residual to load before layer L (isolated-layer PCC)."""
        c = self.cfg
        self.routing = {}
        e = torch.from_numpy(self.W.rows("token_embd.weight", [token])[0].copy())
        R = self._R_from_host(e.repeat(HC))
        ang = self.pos * (c.rope_theta ** (-torch.arange(0, c.rope_dim, 2, dtype=torch.float64) / c.rope_dim))
        cos = torch.ones(256)
        sin = torch.zeros(256)
        cos[: c.rope_dim] = torch.cat([ang.cos(), ang.cos()]).float()
        sin[: c.rope_dim] = torch.cat([ang.sin(), ang.sin()]).float()
        cos_t = self._bvec(cos)
        sin_t = self._bvec(sin)
        prof = self.profile
        def tick(k, t0):
            if prof is not None:
                ttnn.synchronize_device(self.dev)
                t1 = time.perf_counter()
                prof[k] = prof.get(k, 0.0) + t1 - t0
                return t1
            return t0
        tp = time.perf_counter()
        for L in range(c.n_layers):
            if inject is not None:
                r = inject(L)
                if r is not None:
                    R = self._R_from_host(r)
            if L == c.ple_layer:
                R = self.ple(R, token)
            tp = tick("ple+embed", tp)
            x, Rn = self.gr_read(R, f"blk.{L}.hc_attn")
            tp = tick("gr_read", tp)
            y = self.qsa(L, x, cos_t, sin_t) if is_attention(L) else self.gdn(L, x)
            tp = tick("attention" if is_attention(L) else "gdn", tp)
            R = self.gr_write(R, Rn, y, f"blk.{L}.hc_attn")
            tp = tick("gr_write", tp)
            x, Rn = self.gr_read(R, f"blk.{L}.hc_ffn")
            tp = tick("gr_read", tp)
            y = self.moe(L, x)
            tp = tick("moe(all)", tp)
            R = self.gr_write(R, Rn, y, f"blk.{L}.hc_ffn")
            tp = tick("gr_write", tp)
            if observe is not None:
                observe(L, self._R_to_host(R))
        h, _ = self.gr_read(R, "out_hc")
        lg = self._mm(h, self.w["lm_head"])
        out = self._argmax(lg) if greedy else self._row(lg)
        tick("final+lm_head", tp)
        self.pos += 1
        self.tokens.append(token)
        return out
