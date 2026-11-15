"""multi-row pass ("verify") for speculative decoding, and a building block for batched prefill.

T rows = T consecutive tokens at positions pos .. pos+T-1 go through the main model in one pass (same traces for any
token values; rows T..31 of every tile stay zero). The pass does not move the recurrent state; `commit(n)` then
keeps the first n rows (n <= T): GDN state, conv history, PLE history and the position advance as if exactly those n
tokens had been decoded. KV rows beyond n are left behind and masked by position later.

GDN over T rows is the chunk form of the gated delta rule (one chunk; CPU-checked to 1e-16 against the recurrence):
  gamma = cumsum(g), D[t,i] = exp(gamma_t - gamma_i) for i <= t, A = strict_tril(D * (K K^T)) * beta
  U = (I + A)^-1 beta (V - exp(gamma) K S0), (I + A)^-1 = (I - A)(I + A^2)(I + A^4)... (A nilpotent)
  O = exp(gamma) Q S0 + (D * Q K^T) U
  state after n rows: S_n = exp(gamma_r) S0 + K^T (D[r,:]^T * U), r = n - 1 (the commit trace)
Attention: each row's K/V is written with paged_update_cache (one call per row, positions pos+r), then
paged SDPA-decode with 2T 'users' (row r, KV head h) that share the KV head's pages, cur_pos = pos + r: causal.
Host per row: routing (fp32), cold experts (hostexp), PLE (multi-token, with per-n history for the commit).
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
import torch
import ttnn

from tt_qwen38.ref.model import PLEState, grouped_rms

from .model import HC, RM, TILE, H
from .traced import TracedModel
from .weights import is_attention

_COLD_ROWS = os.environ.get("P100A_COLD_ROWS") == "1"   # cold experts via hx.moe_rows
NB = 32  # KV page size (positions); the [2,1,S,256] caches are viewed as [2*S/NB, 1, NB, 256] pages


class VerifyModel(TracedModel):
    def __init__(self, device, experts=None, t_max: int = 4, gather: bool = False, act_cap: int | None = None, **kw):
        self.T = t_max
        self.gather = gather                       # indexed sparse_matmul, cost follows the active slots
        self.A = act_cap or 8 * t_max
        super().__init__(device, experts=experts, **kw)
        self._alloc_v()
        self.vtraces = None
        self.vstats = {"passes": 0, "rows": 0}
        self.qsa_stats = {"passes": 0, "selected_rows": 0}
        self.force_qsa = False
        self.keep_masks, self.last_masks = False, {}
        self.gx = None                               # hostexp.gpu.GpuExperts (warm tier on the 5070 Ti)
        self._warm_rows = None
        self.route_counts = None                     # [n_layers, n_expert] routed-pair counts (traces)

    # ---- buffers and constants
    def _const(self, t: torch.Tensor, dtype=ttnn.float32):
        return self._dev(t, dtype=dtype)

    def sibling(self, T: int) -> "VerifyModel":
        """A VerifyModel with another row count that shares everything persistent with this one (device weights, the
        residual R, GDN state, conv history, KV caches, constants, host experts) and has its own T-row buffers and
        traces. T=28 for prompt reading next to the T=4 verify window. Warm both before capturing either."""
        sib = VerifyModel.__new__(VerifyModel)
        sib.__dict__.update(self.__dict__)
        sib.T, sib.A = T, 8 * T
        sib.vtraces, sib.traces, sib.gdn_keep = None, None, {}
        sib._alloc_v(shared=False)
        return sib

    def _alloc_v(self, shared: bool = True):
        T, z = self.T, torch.zeros
        assert 1 <= T <= 29                          # E = 3 history rows + T new rows must fit one 32-row tile
        self.emb_v = self._dev(z(1, 1, T, H), layout=RM)
        self.ple_v = self._dev(z(1, HC, T, H), layout=RM)
        self.cos_v = self._dev(torch.ones(1, 1, T, 256), layout=RM)
        self.sin_v = self._dev(z(1, 1, T, 256), layout=RM)
        self.pos_v = [self._dev(torch.zeros(2, dtype=torch.int32), dtype=ttnn.int32, layout=RM) for _ in range(T)]
        self.cur_v = self._dev(torch.zeros(2 * T, dtype=torch.int32), dtype=ttnn.int32, layout=RM)
        nblk = self.s_max // NB
        pt = torch.stack([torch.arange(nblk, dtype=torch.int32) + (u % 2) * nblk for u in range(2 * T)])
        self.ptab_v = self._dev(pt, dtype=ttnn.int32, layout=RM)
        self.ycold_v = self._dev(z(1, 1, T, H), layout=RM)
        self.sp_v, self.wsel_v, self.wsel2_v = {}, {}, {}
        self.idx_v = {}
        if self.experts is not None:
            for L, ids in self.experts.sets.items():
                if ids:
                    self.sp_v[L] = self._dev(z(1, 1, 1, len(ids)), dtype=ttnn.bfloat16, layout=RM)
                    if self.gather:                   # indexed mode needs num_active <= resident slots of L
                        aL = min(self.A, len(ids))
                        self.idx_v[L] = self._dev(torch.zeros(1, 1, 1, aL, dtype=torch.int32),
                                                  dtype=ttnn.uint16, layout=RM)
                        self.wsel_v[L] = self._dev(z(1, 1, T, 32 * aL), dtype=ttnn.bfloat16, layout=RM)
                    else:
                        self.wsel_v[L] = self._dev(z(1, 1, T, 32 * len(ids)), dtype=ttnn.bfloat16, layout=RM)
                        if T > 16:                    # rows >= 16 go through a second down projection (experts.py)
                            self.wsel2_v[L] = self._dev(z(1, 1, T, 32 * len(ids)), dtype=ttnn.bfloat16, layout=RM)
        if T > 16:
            lo, hi = z(32, 32), z(32, 32)
            for r in range(16):
                lo[r, r] = 1
                hi[r, r + 16] = 1
            self.c_sel_lo = self._const(lo.reshape(1, 1, 32, 32), ttnn.bfloat16)
            self.c_sel_hi = self._const(hi.reshape(1, 1, 32, 32), ttnn.bfloat16)
        m4 = z(1, HC, 32, H)
        m4[:, :, :T] = 1
        self.mask4_v = self._const(m4)
        m1 = z(1, 1, 32, H)
        m1[:, :, :T] = 1
        self.mask1_v = self._const(m1, ttnn.bfloat16)
        self.qsa_on = self.s_max > 2052                     # the indexer matters only past 2,051 positions
        if self.qsa_on:
            self.qmask_v = self._dev(z(1, 1, T, self.s_max), dtype=ttnn.bfloat16, layout=RM)
            self.posk_v = [self._dev(torch.zeros(1, dtype=torch.int32), dtype=ttnn.int32, layout=RM) for _ in range(T)]
        # S2 fused kernels; P100A_FUSED = comma list (default: all of gdn,moe,hcw,commit,mm1d,kvw,conv,
        # abg; "none" = the plain ttnn op path of step 3)
        self.fused = set(x for x in os.environ.get("P100A_FUSED", "gdn,moe,hcw,commit,mm1d,kvw,conv,abg").split(",") if x)
        if "gdn" in self.fused:
            from .kgdn import GdnCore
            self.kg = GdnCore(self.dev, T, eps=self.cfg.eps)
        if "hcw" in self.fused:
            from .khc import HcWrite
            self.hw = HcWrite(self.dev, T)
        if "kvw" in self.fused:
            from .kkv import KvWrite
            self.kvw = KvWrite(self.dev, T, 2, 256, self.s_max)
            self.ikw = KvWrite(self.dev, T, 1, 128, self.s_max, ncaches=1)
        if not shared:
            self.gdn_keep = {}
            return
        if "moe" in self.fused and self.experts is not None:  # resident experts, cost follows the active ones
            from .kmoe import MoeKernel
            self.km = MoeKernel(self.dev, self.experts)
            self.km_y = self._dev(z(1, 1, 32, H))
            self.km_zero = self._dev(z(1, 1, 32, H), ttnn.bfloat16)
        if "gdn" in self.fused:                              # outputs shared by every T flavour (fixed addresses)
            self.kg_y = self._dev(z(1, 1, 32, 6144))
            self.kg_kn = {L: self._dev(z(1, 48, 32, 128)) for L in self.S}
            self.kg_U = {L: self._dev(z(1, 48, 32, 128)) for L in self.S}
            if "abg" in self.fused:
                self.kg_g = {L: self._dev(z(1, 48, 32, 32)) for L in self.S}
                rd = lambda k: ttnn.to_torch(self.w[k]).float().flatten()[:48].tolist()
                self.kg_dt = {L: rd(f"blk.{L}.gdn.dt") for L in self.S}
                self.kg_a = {L: rd(f"blk.{L}.gdn.a") for L in self.S}
        if "conv" in self.fused:
            from .kgdn import GdnConv
            self.kconv = GdnConv(self.dev)
        if "commit" in self.fused:
            assert "gdn" in self.fused, "the fused commit reads the fused GDN core's outputs"
            from .kgdn import GdnCommit
            self.kcommit = GdnCommit(self.dev)
        if self.qsa_on:
            S = self.s_max
            self.ikc = {L: self._dev(z(1, 1, S, 128), dtype=ttnn.bfloat16) for L in self.kc}
            self.ik_shard = ttnn.create_sharded_memory_config(
                shape=(32, 128), core_grid=ttnn.CoreGrid(x=1, y=1), strategy=ttnn.ShardStrategy.HEIGHT,
                orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
            pool = z(512, 128)
            for i in range(4):
                pool[i * 128:(i + 1) * 128] = torch.eye(128) * 0.25
            self.c_pool = self._const(pool.reshape(1, 1, 512, 128))
            c = self.cfg
            inv = c.rope_theta ** (-torch.arange(0, c.rope_dim, 2, dtype=torch.float64) / c.rope_dim)
            ang = (torch.arange(S // 4, dtype=torch.float64) * 4)[:, None] * inv[None]          # block starts
            bc, bs = torch.ones(S // 4, 128), z(S // 4, 128)
            bc[:, :c.rope_dim] = torch.cat([ang.cos(), ang.cos()], 1).float()
            bs[:, :c.rope_dim] = torch.cat([ang.sin(), ang.sin()], 1).float()
            self.c_bcos, self.c_bsin = self._const(bc.reshape(1, 1, S // 4, 128)), self._const(bs.reshape(1, 1, S // 4, 128))
            P = z(128, 128)
            rr = c.rope_dim // 2
            for i in range(rr):
                P[i + rr, i] = -1.0
                P[i, i + rr] = 1.0
            self.c_rot128 = self._const(P.reshape(1, 1, 128, 128))
        self.hist3 = {L: self._dev(z(1, 1, 32, 10240)) for L in self.S}
        # row-shift / selection / triangle constants (exact 0/1 in any dtype)
        e = torch.eye(32)
        sh3 = z(32, 32)
        for r in range(29):
            sh3[r + 3, r] = 1                                  # E row r+3 <- qkv row r
        self.c_sh3 = self._const(sh3.reshape(1, 1, 32, 32))

        self.c_up = [self._const(u.reshape(1, 1, 32, 32)) for u in self._ups()]
        self._gdnw = {}
        tril = torch.tril(torch.ones(32, 32))
        self.c_cumsum = self._const(tril.reshape(1, 1, 32, 32))
        self.c_tril48 = self._const(tril.expand(1, 48, 32, 32).contiguous())
        self.c_stril48 = self._const((tril - e).expand(1, 48, 32, 32).contiguous())
        self.c_eye48 = self._const(e.expand(1, 48, 32, 32).contiguous())
        # commit selections (host-written each commit): Esel row 0 = one-hot r; Msel rows <= r = 1; Csel rows 0..2
        # pick E rows n..n+2
        self.esel_v = self._dev(z(1, 1, 32, 32), layout=RM)
        self.msel_v = self._dev(z(1, 1, 32, 48), layout=RM)
        self.csel_v = self._dev(z(1, 1, 32, 32), layout=RM)
        self.ckf = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
                                                    fp32_dest_acc_en=True, packer_l1_acc=True)
        self.sdpa_cfg_v = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=(8, 8), exp_approx_mode=False,
                                                 q_chunk_size=0, k_chunk_size=128)
        self.gdn_keep = {}

    @staticmethod
    def _ups():
        """conv tap k reads E row r + k: [32,32] selection U_k."""
        out = []
        for k in range(4):
            u = torch.zeros(32, 32)
            for r in range(32 - k):
                u[r, r + k] = 1
            out.append(u)
        return out

    def reset(self):
        super().reset()
        if getattr(self, "hist3", None):
            for L in self.hist3:
                ttnn.copy(self._zh, self.hist3[L])

    def reset_host(self):
        """Position / token / PLE bookkeeping only (the device state is shared with a sibling and reset there)."""
        self.pos, self.tokens, self.ple_state = 0, [], None

    def _dbg(self, tag):
        """P100A_VSYNC=1 (eager only): sync and print, so a hang is located from the log."""
        if os.environ.get("P100A_VSYNC") in ("1", "2") and self.vtraces is None:
            ttnn.synchronize_device(self.dev)
            print(f"  [v] {tag} ok {time.strftime('%H:%M:%S')}", flush=True)

    def _mmf(self, a, b, **kw):
        """fp32 x fp32 matmul at HiFi4 (state math). The FPU reads fp32 sources as TF32 (~11 significant bits)."""
        return ttnn.matmul(a, b, dtype=ttnn.float32, compute_kernel_config=self.ckf, **kw)

    def _mm_sel(self, sel, b):
        """sel (exact 0/1) @ b with b kept to ~fp32 precision: b = hi + lo with hi = bf16(b) (exact in TF32) and
        lo = b - hi (its TF32 rounding error is ~2^-19 of b). the in-chunk cumsum of the GDN log-decay over 28
        rows lost up to |gamma| * 2^-11 through one TF32 matmul (deep rows fell to PCC 0.8)."""
        hi = ttnn.typecast(ttnn.typecast(b, ttnn.bfloat16), ttnn.float32)
        return ttnn.add(self._mmf(sel, hi), self._mmf(sel, ttnn.subtract(b, hi)))

    # ---- layer pieces, T rows
    def _gdn_w(self, L):
        """Per-layer GDN extras for the verify path, built once on the device (eager warm-up, before any capture):
        alpha / beta weight halves, and the 4 conv taps stacked as a [128, 10240] row-block weight."""
        if L not in self._gdnw:
            p = f"blk.{L}.gdn."
            ab = self.w[p + "ab"]                                                        # [1,1,2560,128] bf16
            wa = ttnn.slice(ab, [0, 0, 0, 0], [1, 1, H, 48])
            wb = ttnn.slice(ab, [0, 0, 0, 48], [1, 1, H, 96])
            taps = ttnn.concat([ttnn.repeat(self.w[f"{p}conv{k}"], ttnn.Shape([1, 1, 32, 1])) for k in range(4)],
                               dim=2)                                                    # [1,1,128,10240]
            self._gdnw[L] = (wa, wb, taps)
        return self._gdnw[L]

    def vgdn(self, L, x):
        p = f"blk.{L}.gdn."
        T = self.T
        x16 = ttnn.typecast(x, ttnn.bfloat16)
        qkv = self._mm(x16, self.w[p + "qkv"])                                         # [1,1,32,10240]
        if "conv" in self.fused:                # one program for E and the 4-tap conv (tt_qwen38/kgdn.py GdnConv)
            shp = ttnn.Shape([1, 1, 32, 10240])
            E = ttnn.allocate_tensor_on_device(shp, ttnn.float32, TILE, self.dev, ttnn.DRAM_MEMORY_CONFIG)
            conv = ttnn.allocate_tensor_on_device(shp, ttnn.float32, TILE, self.dev, ttnn.DRAM_MEMORY_CONFIG)
            self.kconv(qkv, self.hist3[L], [self.w[f"{p}conv{k}"] for k in range(4)], conv, E)
        else:
            E = ttnn.add(self.hist3[L], self._mmf(self.c_sh3, qkv))                    # rows 0..2 history, 3.. new
            conv = None
            for k in range(4):                 # (one stacked [128-row] form measured 0.76 ms vs 0.19 ms: kept this)
                t = ttnn.multiply(self._mmf(self.c_up[k], E), self.w[f"{p}conv{k}"])
                conv = t if conv is None else ttnn.add(conv, t)
            conv = ttnn.silu(conv)
        self._dbg(f"L{L} gdn conv")
        if "gdn" in self.fused and "abg" in self.fused:   # alpha/beta/gamma inside the GDN core kernel too
            ab = self._mm(x16, self.w[p + "ab"])
            z = self._mm(x16, self.w[p + "z"])
            self.kg(conv, None, None, self.S[L], z, self.w[p + "norm"], self.kg_y, self.kg_kn[L], self.kg_U[L],
                    ab=ab, dt=self.kg_dt[L], a=self.kg_a[L], gout=self.kg_g[L])
            self.gdn_keep[L] = (E, self.kg_kn[L], self.kg_U[L], None, self.kg_g[L])
            return self._mm(self.kg_y, self.w[p + "out"])
        if "gdn" in self.fused:                 # one program from conv to y (tt_qwen38/kgdn.py)
            ab = ttnn.to_layout(self._mm(x16, self.w[p + "ab"]), RM)
            alpha = ttnn.to_layout(ttnn.slice(ab, [0, 0, 0, 0], [1, 1, 32, 48]), TILE)
            beta = ttnn.sigmoid(ttnn.to_layout(ttnn.slice(ab, [0, 0, 0, 48], [1, 1, 32, 96]), TILE))
            g = ttnn.multiply(ttnn.softplus(ttnn.add(alpha, self.w[p + "dt"])), self.w[p + "a"])
            gam = self._mm_sel(self.c_cumsum, g)
            gcol = ttnn.permute(gam, (0, 3, 2, 1))
            bcol = ttnn.permute(beta, (0, 3, 2, 1))
            z = self._mm(x16, self.w[p + "z"])
            self.kg(conv, gcol, bcol, self.S[L], z, self.w[p + "norm"], self.kg_y, self.kg_kn[L], self.kg_U[L])
            self.gdn_keep[L] = (E, self.kg_kn[L], self.kg_U[L], gam, gcol)
            return self._mm(self.kg_y, self.w[p + "out"])
        hd = ttnn.permute(ttnn.reshape(conv, [1, 32, 80, 128]), (0, 2, 1, 3))          # [1,80,32,128] head-major
        q = ttnn.repeat(ttnn.slice(hd, [0, 0, 0, 0], [1, 16, 32, 128]), ttnn.Shape([1, 3, 1, 1]))
        k = ttnn.repeat(ttnn.slice(hd, [0, 16, 0, 0], [1, 32, 32, 128]), ttnn.Shape([1, 3, 1, 1]))
        v = ttnn.slice(hd, [0, 32, 0, 0], [1, 80, 32, 128])
        # l2(q) / sqrt(128) = rms_norm(q, eps/128) / 128 ; l2(k) = rms_norm(k, eps/128) / sqrt(128)
        q = ttnn.multiply(ttnn.rms_norm(q, epsilon=1e-6 / 128, compute_kernel_config=self.ckf), 1.0 / 128)
        k = ttnn.multiply(ttnn.rms_norm(k, epsilon=1e-6 / 128, compute_kernel_config=self.ckf), 128 ** -0.5)
        self._dbg(f"L{L} gdn qkv heads")
        ab = ttnn.to_layout(self._mm(x16, self.w[p + "ab"]), RM)                        # [1,1,32,128]
        alpha = ttnn.to_layout(ttnn.slice(ab, [0, 0, 0, 0], [1, 1, 32, 48]), TILE)
        beta = ttnn.sigmoid(ttnn.to_layout(ttnn.slice(ab, [0, 0, 0, 48], [1, 1, 32, 96]), TILE))
        g = ttnn.multiply(ttnn.softplus(ttnn.add(alpha, self.w[p + "dt"])), self.w[p + "a"])   # [1,1,32,48]
        gam = self._mm_sel(self.c_cumsum, g)                                           # cumsum over rows (~fp32)
        self._dbg(f"L{L} gdn g cumsum")
        gcol = ttnn.permute(gam, (0, 3, 2, 1))                                         # [1,48,32,1]
        grow = ttnn.permute(gam, (0, 3, 1, 2))                                         # [1,48,1,32]
        dlog = ttnn.subtract(ttnn.repeat(gcol, ttnn.Shape([1, 1, 1, 32])), grow)
        D = ttnn.multiply(ttnn.exp(ttnn.minimum(dlog, 0.0)), self.c_tril48)            # [1,48,32,32]
        bcol = ttnn.permute(beta, (0, 3, 2, 1))                                        # [1,48,32,1]
        Gcol = ttnn.exp(gcol)
        self._dbg(f"L{L} gdn D")
        S0 = self.S[L]
        A = ttnn.multiply(ttnn.multiply(self._mmf(k, k, transpose_b=True), D), self.c_stril48)
        A = ttnn.multiply(A, bcol)
        VB = ttnn.multiply(ttnn.subtract(v, ttnn.multiply(self._mmf(k, S0), Gcol)), bcol)
        if T <= 4:      # (I+A)^-1 = (I-A)(I+A^2): exact for T <= 4, entries stay small
            inv = self._mmf(ttnn.subtract(self.c_eye48, A), ttnn.add(self.c_eye48, self._mmf(A, A)))
            self._dbg(f"L{L} gdn inv")
            U = self._mmf(inv, VB)                                                      # [1,48,32,128]
        else:           # doubling factors (I+A^2^j) grow like binomials and cancel only in the product: TF32
            U = VB      # loses deep rows (PCC 0.8 at T=28). Solve (I+A)U = VB by U <- VB - A U, T-1 times: exact
            for _ in range(T - 1):          # for strictly lower A, and the last sweep reads only exact rows
                U = ttnn.subtract(VB, self._mmf(A, U))
            self._dbg(f"L{L} gdn solve")
        O = ttnn.add(ttnn.multiply(self._mmf(q, S0), Gcol),
                     self._mmf(ttnn.multiply(self._mmf(q, k, transpose_b=True), D), U))
        self._dbg(f"L{L} gdn O")
        O = ttnn.multiply(ttnn.rms_norm(O, epsilon=self.cfg.eps, compute_kernel_config=self.ckf), self.w[p + "norm"])
        zt = ttnn.permute(ttnn.reshape(self._mm(x16, self.w[p + "z"]), [1, 32, 48, 128]), (0, 2, 1, 3))
        y = ttnn.multiply(O, ttnn.sigmoid(zt))
        y = ttnn.reshape(ttnn.permute(y, (0, 2, 1, 3)), [1, 1, 32, 6144])
        self.gdn_keep[L] = (E, k, U, gam, None)
        return self._mm(y, self.w[p + "out"])

    def _l2(self, t):
        return ttnn.multiply(t, ttnn.rsqrt(ttnn.add(ttnn.sum(ttnn.multiply(t, t), dim=-1, keepdim=True), 1e-6)))

    def vattn(self, L, x, cos, sin, cosq, sinq, p=None, kc=None, vc=None, pos=None, cur=None):
        pre = self.vattn_pre(L, x, cos, sin, cosq, sinq, p, kc, vc, pos, cur)
        return self.vattn_post(L, pre, p, kc, vc, cur)

    def vattn_pre(self, L, x, cos, sin, cosq, sinq, p=None, kc=None, vc=None, pos=None, cur=None):
        p = p or f"blk.{L}.attn."
        kc = self.kc[L] if kc is None else kc
        vc = self.vc[L] if vc is None else vc
        pos = self.pos_v if pos is None else pos
        T = self.T
        qg = ttnn.slice(ttnn.to_layout(self._mm(x, self.w[p + "q"]), RM), [0, 0, 0, 0], [1, 1, T, 12288])
        rm = ttnn.reshape(qg, [1, 1, 24 * T, 512])                                       # rows (r, head)
        q = ttnn.to_layout(ttnn.slice(rm, [0, 0, 0, 0], [1, 1, 24 * T, 256]), TILE)
        gate = ttnn.to_layout(ttnn.slice(rm, [0, 0, 0, 256], [1, 1, 24 * T, 512]), TILE)
        q = self._rope(ttnn.multiply(ttnn.rms_norm(q, epsilon=self.cfg.eps, compute_kernel_config=self.ckf), self.w[p + "q_norm"]), cosq, sinq)
        kv = ttnn.slice(ttnn.to_layout(self._mm(x, self.w[p + "kv"]), RM), [0, 0, 0, 0], [1, 1, T, 1024])
        kv = ttnn.reshape(kv, [1, T, 4, 256])                                            # per row: k0 k1 v0 v1
        k = ttnn.to_layout(ttnn.slice(kv, [0, 0, 0, 0], [1, T, 2, 256]), TILE)
        v = ttnn.slice(kv, [0, 0, 2, 0], [1, T, 4, 256])
        k = self._rope(ttnn.multiply(ttnn.rms_norm(k, epsilon=self.cfg.eps, compute_kernel_config=self.ckf), self.w[p + "k_norm"]), cos, sin)
        k = ttnn.slice(ttnn.to_layout(ttnn.typecast(k, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, T, 2, 256])
        v = ttnn.typecast(v, ttnn.bfloat16) if v.dtype != ttnn.bfloat16 else v
        self._dbg(f"{p} q k v")
        if "kvw" in self.fused:                 # one program writes all 2T rows (tt_qwen38/kkv.py)
            self.kvw([kc, vc], [k, v], self.cur_v if cur is None else cur)
        for r in range(T if "kvw" not in self.fused else 0):
            for src, cache in ((k, kc), (v, vc)):
                row = ttnn.reshape(ttnn.slice(src, [0, r, 0, 0], [1, r + 1, 2, 256]),
                                   [1, 2, 1, 256])
                sh = ttnn.to_layout(ttnn.pad(row, [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0), TILE)
                ttnn.experimental.paged_update_cache(cache, ttnn.to_memory_config(sh, self.kv_shard),
                                                     update_idxs_tensor=pos[r])
        self._dbg(f"{p} kv writes")
        qh = ttnn.slice(ttnn.to_layout(ttnn.typecast(q, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, 1, 24 * T, 256])
        qh = ttnn.to_layout(ttnn.reshape(qh, [1, 2 * T, 12, 256]), TILE)                 # user = 2r + kv head
        return {"qh": qh, "gate": gate}

    def vattn_post(self, L, pre, p=None, kc=None, vc=None, cur=None, mask=None):
        """mask None: causal paged SDPA-decode up to cur_pos (dense). mask [2T,1,12,S] additive (M10, QSA): non-causal
        over the whole cache, the mask carries both causality and the indexer's block selection."""
        p = p or f"blk.{L}.attn."
        kc = self.kc[L] if kc is None else kc
        vc = self.vc[L] if vc is None else vc
        cur = self.cur_v if cur is None else cur
        T = self.T
        nblk = self.s_max // NB
        kp = ttnn.reshape(kc, [2 * nblk, 1, NB, 256])
        vp = ttnn.reshape(vc, [2 * nblk, 1, NB, 256])
        if mask is None:
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                pre["qh"], kp, vp, self.ptab_v, cur_pos_tensor=cur, scale=1.0 / math.sqrt(256),
                program_config=self.sdpa_cfg_v, memory_config=ttnn.DRAM_MEMORY_CONFIG)    # [1,2T,12,256]
        else:
            o = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                pre["qh"], kp, vp, self.ptab_v, is_causal=False, attn_mask=mask, scale=1.0 / math.sqrt(256),
                program_config=self.sdpa_cfg_v, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self._dbg(f"{p} sdpa")
        o = ttnn.reshape(ttnn.slice(ttnn.to_layout(o, RM), [0, 0, 0, 0], [1, 2 * T, 12, 256]), [1, 1, 24 * T, 256])
        o = ttnn.multiply(ttnn.to_layout(o, TILE), ttnn.sigmoid(pre["gate"]))
        o = ttnn.slice(ttnn.to_layout(o, RM), [0, 0, 0, 0], [1, 1, 24 * T, 256])
        return self._mm(self._pad32(ttnn.reshape(o, [1, 1, T, 6144])), self.w[p + "out"])

    def _ik_write(self, L, x16):
        """raw indexer keys of this pass's rows -> the layer's [1,1,S,128] key cache (both flavours: the QSA passes
        need every earlier position's key)."""
        p, T = f"blk.{L}.idx.", self.T
        ik = ttnn.slice(ttnn.to_layout(self._mm(x16, self.w[p + "k"]), RM), [0, 0, 0, 0], [1, 1, T, 128])
        ik = ttnn.typecast(ik, ttnn.bfloat16)
        if "kvw" in self.fused:
            self.ikw([self.ikc[L]], [ik], self.cur_v)
            return
        for r in range(T):
            row = ttnn.pad(ttnn.slice(ik, [0, 0, r, 0], [1, 1, r + 1, 128]), [(0, 0), (0, 0), (0, 31), (0, 0)], 0.0)
            ttnn.experimental.paged_update_cache(self.ikc[L], ttnn.to_memory_config(ttnn.to_layout(row, TILE),
                                                                                    self.ik_shard),
                                                 update_idxs_tensor=self.posk_v[r])

    def vindexer(self, L, x16, cosi, sini):
        """M10 QSA indexer: writes this pass's raw indexer keys, re-pools every complete 4-token block of the cache
        (mean -> rms * k_norm -> rope at the block start) and scores them for every row:
        score[r, b] = sum_h relu(q_rot[r,h] . k_blk[b]) / sqrt(128)   -> [1,1,T,S/4] fp32 for the host top-k."""
        p, T, S = f"blk.{L}.idx.", self.T, self.s_max
        self._ik_write(L, x16)
        blk = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(self.ikc[L], RM), [1, 1, S // 4, 512]), TILE)
        pooled = self._mmf(blk, self.c_pool)                                            # [1,1,S/4,128] block means
        kb = ttnn.multiply(ttnn.rms_norm(pooled, epsilon=self.cfg.eps, compute_kernel_config=self.ckf),
                           self.w[p + "k_norm"])
        kb = ttnn.add(ttnn.multiply(kb, self.c_bcos), ttnn.multiply(self._mmf(kb, self.c_rot128), self.c_bsin))
        iq = ttnn.slice(ttnn.to_layout(self._mm(x16, self.w[p + "q"]), RM), [0, 0, 0, 0], [1, 1, T, 512])
        iq = ttnn.to_layout(ttnn.reshape(iq, [1, 1, 4 * T, 128]), TILE)                   # rows (r, head)
        iq = ttnn.multiply(ttnn.rms_norm(iq, epsilon=self.cfg.eps, compute_kernel_config=self.ckf), self.w[p + "q_norm"])
        iq = ttnn.add(ttnn.multiply(iq, cosi), ttnn.multiply(self._mmf(iq, self.c_rot128), sini))
        sc = ttnn.relu(self._mmf(iq, kb, transpose_b=True))                             # [1,1,4T,S/4]
        sc = ttnn.sum(ttnn.reshape(sc, [1, T, 4, S // 4]), dim=2, keepdim=True)         # [1,T,1,S/4]
        return ttnn.multiply(ttnn.reshape(sc, [1, 1, T, S // 4]), 1.0 / math.sqrt(128))

    def _gr_write_v(self, R, Rn, y, prefix, out=None):
        w = self._gr_w(prefix)
        c = ttnn.sum(self._mm(Rn, w["inj"]), dim=1, keepdim=True)
        if "hcw" in self.fused:                 # one program for the elementwise tail (tt_qwen38/khc.py)
            if out is None:
                out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, HC, 32, H]), ttnn.float32, TILE, self.dev,
                                                     ttnn.DRAM_MEMORY_CONFIG)
            self.hw(R, y, c, out)
            return out
        coef = ttnn.multiply(ttnn.sigmoid(c), 2.0)                     # Rn carries the 1/HC (see model._gr_w)
        coef = ttnn.slice(ttnn.permute(coef, (0, 3, 2, 1)), [0, 0, 0, 0], [1, HC, 32, 1])
        out = ttnn.add(R, ttnn.multiply(ttnn.repeat(y, ttnn.Shape([1, HC, 1, 1])), coef))
        return ttnn.where(self.mask4_v, out, self._zero_4)

    def vseg_a(self, L):
        if L == 0:
            e = self._pad32(self.emb_v)
            R = ttnn.where(self.mask4_v, ttnn.repeat(e, ttnn.Shape([1, HC, 1, 1])), self._zero_4)
        else:
            R = self.R
        if L == self.cfg.ple_layer:
            R = ttnn.add(R, self._pad32(self.ple_v))
        x, Rn = self.gr_read(R, f"blk.{L}.hc_attn")
        if is_attention(L):
            T = self.T
            cos, sin = ttnn.to_layout(self.cos_v, TILE), ttnn.to_layout(self.sin_v, TILE)   # [1,1,T,256]
            cosk, sink = ttnn.reshape(cos, [1, T, 1, 256]), ttnn.reshape(sin, [1, T, 1, 256])
            cosq = ttnn.to_layout(ttnn.repeat_interleave(self.cos_v, 24, dim=2), TILE)    # [1,1,24T,256]
            sinq = ttnn.to_layout(ttnn.repeat_interleave(self.sin_v, 24, dim=2), TILE)
            y = self.vattn(L, x, cosk, sink, cosq, sinq)
            if self.qsa_on:
                self._ik_write(L, ttnn.typecast(x, ttnn.bfloat16))
        else:
            y = self.vgdn(L, x)
        self._dbg(f"L{L} mixer done")
        R2 = self._gr_write_v(R, Rn, y, f"blk.{L}.hc_attn")
        x2, Rn2 = self.gr_read(R2, f"blk.{L}.hc_ffn")
        xrow = ttnn.slice(ttnn.to_layout(ttnn.typecast(x2, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, 1, self.T, H])
        out = {"R2": R2, "Rn2": Rn2, "x2": x2, "xrow": xrow}
        if is_attention(L):
            out["y"] = y                                    # attention output (M10 reference check)
        return out

    def _seg_in(self, L):
        if L == 0:
            e = self._pad32(self.emb_v)
            R = ttnn.where(self.mask4_v, ttnn.repeat(e, ttnn.Shape([1, HC, 1, 1])), self._zero_4)
        else:
            R = self.R
        if L == self.cfg.ple_layer:
            R = ttnn.add(R, self._pad32(self.ple_v))
        return R

    def _rope_rows(self):
        T = self.T
        cos, sin = ttnn.to_layout(self.cos_v, TILE), ttnn.to_layout(self.sin_v, TILE)       # [1,1,T,256]
        cosk, sink = ttnn.reshape(cos, [1, T, 1, 256]), ttnn.reshape(sin, [1, T, 1, 256])
        cosq = ttnn.to_layout(ttnn.repeat_interleave(self.cos_v, 24, dim=2), TILE)        # [1,1,24T,256]
        sinq = ttnn.to_layout(ttnn.repeat_interleave(self.sin_v, 24, dim=2), TILE)
        return cosk, sink, cosq, sinq

    def vseg_aq1(self, L):
        """QSA flavour, attention layer, part 1: everything up to the indexer scores (host picks the blocks)."""
        R = self._seg_in(L)
        x, Rn = self.gr_read(R, f"blk.{L}.hc_attn")
        cosk, sink, cosq, sinq = self._rope_rows()
        pre = self.vattn_pre(L, x, cosk, sink, cosq, sinq)
        cs = ttnn.slice(self.cos_v, [0, 0, 0, 0], [1, 1, self.T, 128])
        sn = ttnn.slice(self.sin_v, [0, 0, 0, 0], [1, 1, self.T, 128])
        cosi = ttnn.to_layout(ttnn.repeat_interleave(cs, 4, dim=2), TILE)                 # [1,1,4T,128]
        sini = ttnn.to_layout(ttnn.repeat_interleave(sn, 4, dim=2), TILE)
        sc = self.vindexer(L, ttnn.typecast(x, ttnn.bfloat16), cosi, sini)
        return {"R": R, "Rn": Rn, "pre": pre, "scores": sc, "x": x}

    def vseg_aq2(self, L, a1, dense_out=None):
        """QSA flavour, part 2: masked attention, then the same tail as vseg_a. With dense_out (trace capture) the
        results are copied into the dense trace's output tensors, so the vb1/vb2 traces serve both flavours."""
        T, S = self.T, self.s_max
        m = ttnn.reshape(self.qmask_v, [T, 1, 1, S])                                    # RM bf16 additive
        m = ttnn.repeat(ttnn.repeat_interleave(m, 2, dim=0), ttnn.Shape([1, 1, 12, 1]))  # [2T,1,12,S]
        y = self.vattn_post(L, a1["pre"], mask=ttnn.to_layout(m, TILE))
        R2 = self._gr_write_v(a1["R"], a1["Rn"], y, f"blk.{L}.hc_attn")
        x2, Rn2 = self.gr_read(R2, f"blk.{L}.hc_ffn")
        xrow = ttnn.slice(ttnn.to_layout(ttnn.typecast(x2, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, 1, T, H])
        if dense_out is None:       # eager (warm-up): run the same copies into fresh tensors so their programs are
            dense_out = {k_: ttnn.clone(t_) for k_, t_ in             # compiled before capture (M10 lesson)
                         (("R2", R2), ("Rn2", Rn2), ("x2", x2), ("xrow", xrow), ("y", y))}
        for k_, t_ in (("R2", R2), ("Rn2", Rn2), ("x2", x2), ("xrow", xrow), ("y", y)):
            ttnn.copy(t_, dense_out[k_])
        return dense_out

    def vseg_b1(self, L, a):
        p = f"blk.{L}."
        x = a["x2"]
        gu = self._mm(x, self.w[p + "sh.gu"])
        h = ttnn.multiply(ttnn.silu(ttnn.slice(gu, [0, 0, 0, 0], [1, 1, 32, 640])),
                          ttnn.slice(gu, [0, 0, 0, 640], [1, 1, 32, 1280]))
        sh = self._mm(h, self.w[p + "sh.down"])
        sg = ttnn.sigmoid(ttnn.slice(self._mm(x, self.w[p + "sh.gate"]), [0, 0, 0, 0], [1, 1, 32, 1]))
        y = ttnn.multiply(sh, sg)
        if L in self.sp_v and "moe" in self.fused:
            xm = ttnn.where(self.mask1_v, ttnn.typecast(x, ttnn.bfloat16), self.km_zero)
            self.km(L, xm, self.km_y)
            y = ttnn.add(y, self.km_y)
        elif L in self.sp_v:
            if self.gather:
                y = ttnn.add(y, self.experts.forward_ix(L, x, self.sp_v[L], self.idx_v[L],
                                                        self._pad32(self.wsel_v[L]), self.mask1_v))
            elif L in self.wsel2_v:
                y = ttnn.add(y, self.experts.forward_sp(L, x, self.sp_v[L], self._pad32(self.wsel_v[L]),
                                                        mask=self.mask1_v, wsel2=self._pad32(self.wsel2_v[L]),
                                                        sel_lo=self.c_sel_lo, sel_hi=self.c_sel_hi))
            else:
                y = ttnn.add(y, self.experts.forward_sp(L, x, self.sp_v[L], self._pad32(self.wsel_v[L]),
                                                        mask=self.mask1_v))
        return {"y": y}

    def vseg_b2(self, L, a, b1):
        y = ttnn.add(b1["y"], self._pad32(self.ycold_v))
        if "hcw" in self.fused:                 # the kernel writes the new residual straight into self.R
            self._gr_write_v(a["R2"], a["Rn2"], y, f"blk.{L}.hc_ffn", out=self.R)
        else:
            R3 = self._gr_write_v(a["R2"], a["Rn2"], y, f"blk.{L}.hc_ffn")
            ttnn.copy(R3, self.R)
        return {}

    def vseg_final(self):
        h, _ = self.gr_read(self.R, "out_hc")
        lg = self._mm(h, self.w["lm_head"])
        rm = ttnn.slice(ttnn.to_layout(ttnn.typecast(lg, ttnn.bfloat16), RM), [0, 0, 0, 0], [1, 1, self.T, lg.shape[-1]])
        return {"logits": rm, "argmax": ttnn.argmax(rm, dim=-1)}

    def vseg_rows(self):
        """R rows (bf16) after layer 0, for the host PLE."""
        return {"rows": ttnn.slice(ttnn.to_layout(ttnn.typecast(self.R, ttnn.bfloat16), RM), [0, 0, 0, 0],
                                   [1, HC, self.T, H])}

    def vcommit(self):
        """Keep the first n rows: S <- S_n, conv history <- E rows n..n+2 (n from the host-written selections)."""
        esel = ttnn.to_layout(self.esel_v, TILE)
        msel = ttnn.to_layout(self.msel_v, TILE)
        csel = ttnn.to_layout(self.csel_v, TILE)
        for L in self.S:
            self._dbg(f"commit L{L}")
            E, k, U, gam, gcol = self.gdn_keep[L]
            if "commit" in self.fused:          # one program per layer (tt_qwen38/kgdn.py GdnCommit), exact state
                self.kcommit(self.S[L], k, U, gcol)
                ttnn.copy(self._mmf(csel, E), self.hist3[L])
                continue
            gr = ttnn.slice(self._mm_sel(esel, gam), [0, 0, 0, 0], [1, 1, 1, 48])       # gamma_r  [1,1,1,48]
            dcol = ttnn.multiply(ttnn.exp(ttnn.minimum(ttnn.subtract(gr, gam), 0.0)), msel)  # [1,1,32,48]
            dh = ttnn.permute(dcol, (0, 3, 2, 1))                                      # [1,48,32,1]
            Sn = self._mmf(k, ttnn.multiply(U, dh), transpose_a=True)                  # [1,48,128,128]
            Gr = ttnn.permute(ttnn.exp(gr), (0, 3, 1, 2))                              # [1,48,1,1]
            Sn = ttnn.add(Sn, ttnn.multiply(self.S[L], Gr))
            ttnn.copy(Sn, self.S[L])
            ttnn.copy(self._mmf(csel, E), self.hist3[L])
        return {}

    # ---- host side
    def _ple_multi(self, R: torch.Tensor, tokens: list):
        """PLE for T rows; returns (delta [T,4H], states[n] for n = 1..T)."""
        ref, c = self.ref, self.cfg
        st = self.ple_state
        p = f"blk.{c.ple_layer}."
        Hh, hc = c.hidden, c.hc
        prev = [c.ple_eos] * (c.ple_ngram - 1) if st is None else st.tokens
        ids = ref.ple_ids(tokens, prev)
        rows = ref.W.rows("per_layer_token_embd.weight", ids.ravel())
        emb = torch.from_numpy(np.ascontiguousarray(rows, dtype=np.float32)).view(len(tokens), -1)
        key = grouped_rms(emb @ ref.t(p + "ple_key.weight").T, ref.t(p + "ple_norm_key.weight"), Hh, c.eps)
        val = emb @ ref.t(p + "ple_value.weight").T
        qry = grouped_rms(R, ref.t(p + "ple_norm_query.weight"), Hh, c.eps)
        gate = (key.view(-1, hc, Hh) * qry.view(-1, hc, Hh)).sum(-1, keepdim=True) / math.sqrt(Hh)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = (torch.sigmoid(gate) * val[:, None, :]).reshape(-1, hc * Hh)
        nrm = grouped_rms(gv, ref.t(p + "ple_norm_conv.weight"), Hh, c.eps)
        dil = c.ple_ngram
        hl = (c.ple_conv_k - 1) * dil
        hist = torch.zeros(hl, hc * Hh) if st is None else st.conv
        full = torch.cat([hist, nrm], 0)
        w = ref.t(p + "ple_conv1d.weight")
        Tn = len(tokens)
        conv = sum(full[k * dil: k * dil + Tn] * w[:, k] for k in range(c.ple_conv_k))
        out = gv + torch.nn.functional.silu(conv)
        allt = list(prev) + list(tokens)
        states = [PLEState(tokens=allt[: len(prev) + n][-(c.ple_ngram - 1):], conv=full[n: n + hl].clone())
                  for n in range(1, Tn + 1)]
        return out, states

    def _inputs_v(self, tokens):
        c, T = self.cfg, self.T
        e = torch.from_numpy(np.ascontiguousarray(self.W.rows("token_embd.weight", tokens), dtype=np.float32))
        self._write(self.emb_v, e.reshape(1, 1, T, H))
        cos, sin = torch.ones(T, 256), torch.zeros(T, 256)
        inv = c.rope_theta ** (-torch.arange(0, c.rope_dim, 2, dtype=torch.float64) / c.rope_dim)
        for r in range(T):
            ang = (self.pos + r) * inv
            cos[r, : c.rope_dim] = torch.cat([ang.cos(), ang.cos()]).float()
            sin[r, : c.rope_dim] = torch.cat([ang.sin(), ang.sin()]).float()
        self._write(self.cos_v, cos.reshape(1, 1, T, 256))
        self._write(self.sin_v, sin.reshape(1, 1, T, 256))
        lim = self.s_max - 1
        for r in range(T):
            pr = min(self.pos + r, lim)
            self._write(self.pos_v[r], torch.tensor([pr, pr], dtype=torch.int32), ttnn.int32)
        self._write(self.cur_v, torch.tensor([min(self.pos + u // 2, lim) for u in range(2 * T)], dtype=torch.int32),
                    ttnn.int32)
        if self.qsa_on:
            for r in range(T):
                self._write(self.posk_v[r], torch.tensor([min(self.pos + r, lim)], dtype=torch.int32), ttnn.int32)

    def _route_v(self, L, X, n_valid=None):
        """X [T, H] fp32 -> per-row cold (ids, wts); writes the union sparsity + per-row selection rows."""
        if L not in self.router:
            self.router[L] = self.W.tensor(f"blk.{L}.ffn_gate_inp.weight")
        T, k = self.T, self.cfg.top_k
        logits = X @ self.router[L].T                                                   # [T, 256]
        pr = np.exp(logits - logits.max(1, keepdims=True))
        pr /= pr.sum(1, keepdims=True)
        cold = []
        km = "moe" in self.fused and L in self.sp_v
        picks = []
        if L in self.sp_v:
            n = len(self.experts.sets[L])
            sp = torch.zeros(1, 1, 1, n)
            aL = min(self.A, n)
            ws = torch.zeros(1, 1, T, 32 * (aL if self.gather else n))
            ws2 = torch.zeros(1, 1, T, 32 * n) if L in self.wsel2_v else None
            slot_of = {}                                     # gather: resident slot -> compact position
        gx, warm_rows = self.gx, []
        for r in range(T):
            if n_valid is not None and r >= n_valid:        # padding row (no draft proposed): no expert work
                cold.append(([], []))
                warm_rows.append([])
                picks.append([])
                continue
            ids = np.argsort(-pr[r], kind="stable")[:k]
            wts = pr[r, ids] / pr[r, ids].sum()
            if self.route_counts is not None:
                self.route_counts[L, ids] += 1
            if self.experts is not None:
                res, cids, cwts = self.experts.split(L, ids, wts)
                if km:
                    picks.append([(i, float(wt)) for i, wt in res])
                    res = [i for i, _ in res]
                elif L in self.sp_v:
                    keep = []
                    for i, wt in res:
                        if self.gather:
                            if i not in slot_of and len(slot_of) >= aL:           # over the cap: CPU instead
                                cids.append(self.experts.sets[L][i])
                                cwts.append(wt)
                                continue
                            c = slot_of.setdefault(i, len(slot_of))
                            ws[0, 0, r, 32 * c + r] = wt
                        elif ws2 is not None and r >= 16:
                            ws2[0, 0, r, 32 * i + r - 16] = wt
                        else:
                            ws[0, 0, r, 32 * i + r] = wt
                        sp[0, 0, 0, i] = 1.0
                        keep.append(i)
                    res = keep
                self.stats["resident"] += len(res)
            else:
                cids, cwts = ids, wts
            if gx is not None:
                warm, cids, cwts = gx.split(L, cids, cwts)
                warm_rows.append(warm)
                self.stats["warm"] = self.stats.get("warm", 0) + len(warm)
            self.stats["cold"] += len(cids)
            cold.append((cids, cwts))
        self._warm_rows = warm_rows if gx is not None else None
        if km:
            ctl, _ = self.km.control(picks)
            self.km.write_control(L, ctl)
            self.io["h2d_bytes"] += ctl.numel() * 4
        elif L in self.sp_v:
            self._write(self.sp_v[L], sp, ttnn.bfloat16)
            self._write(self.wsel_v[L], ws, ttnn.bfloat16)
            if ws2 is not None:
                self._write(self.wsel2_v[L], ws2, ttnn.bfloat16)
            if self.gather:
                ix = list(slot_of) + [next(iter(slot_of), 0)] * (aL - len(slot_of))       # pad: weight 0
                self._write(self.idx_v[L], torch.tensor(ix, dtype=torch.int32).reshape(1, 1, 1, aL), ttnn.uint16)
        return cold

    def _cold_v(self, L, X, cold):
        t0 = time.perf_counter()
        # the GPU's warm experts run while the CPU does the cold ones; only x / y cross PCIe
        sub = self._warm_rows is not None and self.gx.submit(L, self._warm_rows, X)
        if _COLD_ROWS and sum(1 for c, _ in cold if len(c)) > 1:     # one call, each unique expert read once
            Y = self.hx.moe_rows(L, cold, X)
        else:
            Y = np.zeros((self.T, H), np.float32)
            for r, (cids, cwts) in enumerate(cold):
                if len(cids):
                    Y[r] = self.hx.moe(L, cids, cwts, np.ascontiguousarray(X[r]))
        t1 = time.perf_counter()
        self.phase["hostexp_only"] = self.phase.get("hostexp_only", 0) + t1 - t0
        if sub:
            Y += self.gx.wait()
            self.phase["gpu_wait"] = self.phase.get("gpu_wait", 0) + time.perf_counter() - t1
        self._write(self.ycold_v, torch.from_numpy(Y).reshape(1, 1, self.T, H))

    def _vrun(self, name, fn, *args):
        if self.vtraces is None:
            return fn(*args)
        tid, out = self.vtraces[name]
        ttnn.execute_trace(self.dev, tid, cq_id=0, blocking=False)
        return out

    def verify(self, tokens: list, logits: bool = False, n_valid: int | None = None):
        """Run T tokens at pos..pos+T-1 (no state change until commit). Returns per-row argmax [T] (and logits)."""
        T = self.T
        assert len(tokens) == T
        if self.pos + T > min(self.s_max, self.max_ctx):
            raise RuntimeError("context limit of the traced dense-attention path")
        ph = self.phase
        t = t00 = time.perf_counter()

        def tick(k):
            nonlocal t
            t1 = time.perf_counter()
            ph[k] = ph.get(k, 0) + t1 - t
            t = t1
        self._inputs_v(tokens)
        self._pending_ple = None
        qsa = self.qsa_on and (self.force_qsa or self.pos + T > 2 * 1024 + 3)     # some row sees >= 2,052 positions
        self.qsa_stats["passes"] += int(qsa)
        tick("v.inputs")
        for L in range(self.cfg.n_layers):
            if L == self.cfg.ple_layer:
                rows = self._read(self._vrun("vrows", self.vseg_rows)["rows"]).float()      # [1,4,T,H]
                tick("v.wait_dev")
                R = rows[0].permute(1, 0, 2).reshape(T, HC * H)
                d, self._pending_ple = self._ple_multi(R, tokens)
                self._write(self.ple_v, d.reshape(T, HC, H).permute(1, 0, 2).reshape(1, HC, T, H).contiguous())
                tick("v.ple")
            if qsa and is_attention(L):
                a1 = self._vrun(f"vaq1{L}", self.vseg_aq1, L)
                sc = self._read(a1["scores"]).float().reshape(T, -1).numpy()
                tick("v.wait_dev")
                mk = self._qsa_mask(sc, n_valid)
                if self.keep_masks:
                    self.last_masks[L] = mk
                self._write(self.qmask_v, mk, ttnn.bfloat16)
                tick("v.qsa_select")
                a = self._vrun(f"vaq2{L}", self.vseg_aq2, L, a1)
            else:
                a = self._vrun(f"va{L}", self.vseg_a, L)
            X = self._read(a["xrow"]).float().reshape(T, H).numpy()
            tick("v.wait_dev")
            cold = self._route_v(L, X, n_valid)
            b1 = self._vrun(f"vb1_{L}", self.vseg_b1, L, a)
            tick("v.route+writes")
            self._cold_v(L, X, cold)
            tick("v.cold+write")
            self._vrun(f"vb2_{L}", self.vseg_b2, L, a, b1)
            self._dbg(f"L{L} layer done")
            tick("v.enqueue")
        f = self._vrun("vfinal", self.vseg_final)
        am = self._read(f["argmax"]).reshape(-1)[:T].long().tolist()
        lg = self._read(f["logits"]).float().reshape(T, -1) if logits else None
        tick("v.wait_dev")
        self._v_tokens = list(tokens)
        self.vstats["passes"] += 1
        self.vstats["rows"] += T
        ph["verify"] = ph.get("verify", 0) + time.perf_counter() - t00
        return (am, lg) if logits else am

    def _qsa_mask(self, sc: np.ndarray, n_valid=None) -> torch.Tensor:
        """Host top-k of the QSA indexer (tt_qwen38/ref/model.py qsa): row r sees vis = pos+r+1 positions; while the complete
        4-token blocks fit the 512-block budget it is dense (causal), else the 512 best-scoring complete blocks plus the
        incomplete tail. Returns the additive bf16-ready mask [1,1,T,S] (0 = attend, -1e9 = not)."""
        T, S = self.T, self.s_max
        c = self.cfg
        budget = c.idx_topk // c.compress
        m = np.full((T, S), -1e9, np.float32)
        for r in range(T):
            vis = min(self.pos + r + 1, S)
            complete = vis // c.compress
            if complete <= budget or (n_valid is not None and r >= n_valid):
                m[r, :vis] = 0
                continue
            b = np.argpartition(-sc[r, :complete], budget - 1)[:budget]
            idx = (b[:, None] * c.compress + np.arange(c.compress)[None]).ravel()
            m[r, idx] = 0
            m[r, complete * c.compress: vis] = 0
            self.qsa_stats["selected_rows"] += 1
        return torch.from_numpy(m).reshape(1, 1, T, S)

    def commit(self, n: int):
        """Keep the first n rows of the last verify pass."""
        assert 1 <= n <= self.T
        r = n - 1
        es = torch.zeros(1, 1, 32, 32)
        es[0, 0, 0, r] = 1
        ms = torch.zeros(1, 1, 32, 48)
        ms[0, 0, :n] = 1
        cs = torch.zeros(1, 1, 32, 32)
        for j in range(3):
            cs[0, 0, j, n + j] = 1
        self._write(self.esel_v, es)
        self._write(self.msel_v, ms)
        self._write(self.csel_v, cs)
        if "commit" in self.fused:
            self.kcommit.write_sel(n)
        self._vrun("vcommit", self.vcommit)
        self.ple_state = self._pending_ple[r]
        self.pos += n
        self.tokens.extend(self._v_tokens[:n])

    def warm_v(self, tokens=None):
        """One eager pass + commit: compiles every program and makes every lazy device allocation. Must run before ANY
        trace capture in the process (allocating while a trace exists can be overwritten by its replays; 06:30 hang)."""
        assert self.traces is None and self.vtraces is None and getattr(self, "mtraces", None) is None
        self.reset()
        self.verify(tokens or [198] * self.T)
        self.commit(self.T)
        if self.qsa_on:                                  # the QSA flavour too (compiles + allocates its programs)
            self.force_qsa = True
            self.verify(tokens or [198] * self.T)
            self.force_qsa = False
        self.reset()

    def capture_v(self, tokens=None, warm=True):
        """Capture the verify traces (after warm_v; pass warm=False when it already ran)."""
        if warm:
            self.warm_v(tokens)
        tr = {}
        def cap(name, fn, *args):
            if os.environ.get("P100A_VSYNC") == "3":
                print(f"  [cap] {name} {time.strftime('%H:%M:%S')}", flush=True)
            tid = ttnn.begin_trace_capture(self.dev, cq_id=0)
            out = fn(*args)
            ttnn.end_trace_capture(self.dev, tid, cq_id=0)
            tr[name] = (tid, out)
            return out
        for L in range(self.cfg.n_layers):
            a = cap(f"va{L}", self.vseg_a, L)
            if self.qsa_on and is_attention(L):
                a1 = cap(f"vaq1{L}", self.vseg_aq1, L)
                cap(f"vaq2{L}", self.vseg_aq2, L, a1, a)
            b1 = cap(f"vb1_{L}", self.vseg_b1, L, a)
            cap(f"vb2_{L}", self.vseg_b2, L, a, b1)
        cap("vfinal", self.vseg_final)
        cap("vrows", self.vseg_rows)
        cap("vcommit", self.vcommit)
        self.vtraces = tr
        self.reset()
