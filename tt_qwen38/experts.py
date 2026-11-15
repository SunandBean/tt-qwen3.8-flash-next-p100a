"""routed experts resident on the P100a in BFP4 (the hot ones), the rest on the host CPU (hostexp).

Resident set: the top-N (layer, expert) pairs of Strata's routing profile (data/expert-profile-coder.prof, a ranking)
with N = budget / 2,764,800 B (one BFP4 expert: gate + up + down, 3 x 1600 tiles x 576 B).
Per layer the resident experts are stacked: gate_up [1, n_L, 2560, 1280] ([gate | up] on N), down [1, n_L, 640, 2560],
and run with ttnn.sparse_matmul, the token's routing weights over the n_L slots as the sparsity mask (zeros = skipped).

  python -m tt_qwen38.experts --prepare --gb 20     # dequantize + BFP4-pack, ~10 min, no card
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from tt_qwen38.ref.gguf_weights import Weights
from tt_qwen38.expert_profile import read as read_profile

from tt_qwen38.sources import card_cache, lotto_pool

CACHE = card_cache()
PROFILE = pathlib.Path(__file__).resolve().parent / "data" / (
    "expert-profile.prof" if os.environ.get("P100A_MODEL") == "full" else "expert-profile-coder.prof")
BYTES_PER_EXPERT = {"bfp4": 3 * 1600 * 576, "bfp8": 3 * 1600 * 1088}
H, F = 2560, 640


RES_RANK = os.environ.get("P100A_RES_RANK", "strata")   # "trace" = rank residents by data/route-counts-*.npy


def ranking(kind: str = "strata"):
    """(layer, expert) pairs hottest first: Strata's profile, or (kind "trace") the step-7 *decode* routing counts
    (verify-window rows only: Claude Code + Codex replays, a Korean suite, a decode mix) with ties and unseen
    experts in Strata's order. Prompt rows route differently (corr 0.43) and are kept out."""
    strata = read_profile(str(PROFILE))["ranked"]
    if kind != "trace":
        return strata
    f = pathlib.Path(__file__).resolve().parent.parent / "data" / (
        "route-counts-full-decode.npy" if os.environ.get("P100A_MODEL") == "full" else "route-counts-coder-decode.npy")
    if not f.exists():
        return strata
    C = np.load(f)
    prior = np.full(C.shape, len(strata), np.int64)
    prior[strata[:, 0], strata[:, 1]] = np.arange(len(strata))
    order = np.lexsort((prior.reshape(-1), -C.reshape(-1)))
    return np.stack([order // C.shape[1], order % C.shape[1]], 1)


def resident_sets(gb: float, fmt: str = "bfp4", n_layers=48) -> dict[int, list[int]]:
    ranked = ranking(RES_RANK)
    n = min(int(gb * 1e9 // BYTES_PER_EXPERT[fmt]), len(ranked))
    sets = {L: [] for L in range(n_layers)}
    for L, e in ranked[:n].tolist():
        sets[L].append(e)
    return {L: sorted(v) for L, v in sets.items()}


def tag(gb: float, fmt: str = "bfp4") -> str:
    """cache dir name; trace-ranked sets (P100A_RES_RANK=trace) get their own '-trace' suffix (never reuse a name)"""
    return f"res{gb:g}gb" + ("" if fmt == "bfp4" else f"-{fmt}")





class LottoPool:
    """Lottolabs GPTQ-BFP4 rows (Qwen3.8-Flash-Next-Coder-P150 @ 69e738b): L{LL}.b4.npy uint8 [768, 921600], one row per
    (expert, matrix) as W^T [in, out] bfp4_b tiles; L{LL}.b4idx.npy int32 [768, 2] = (expert, matrix 0 gate/1 up/2 down)."""

    def __init__(self, root=None):
        root = root or lotto_pool()
        self.root = pathlib.Path(root)

    def layer(self, L):
        rows = np.load(self.root / f"L{L:02d}.b4.npy", mmap_mode="r")
        idx = np.load(self.root / f"L{L:02d}.b4idx.npy")
        where = {(int(e), int(m)): i for i, (e, m) in enumerate(idx)}
        return rows, where

    @staticmethod
    def matrix(rows, where, e, m):
        """-> float32 W^T [in, out]. Row layout (found by decoding, checked against the GGUF): planar, not tiled:
        in*out/16 exponent bytes ([in, out/16], one per 16 output channels), then in*out/2 bytes of 4-bit
        sign|mantissa codes ([in, out], low nibble first); value = (-1)^s * (m / 4) * 2^(e - 127)."""
        raw = np.asarray(rows[where[(e, m)]])
        r, c = (H, F) if m < 2 else (F, H)
        ne = r * c // 16
        exp = raw[:ne].astype(np.int32).reshape(r, c // 16)
        d = raw[ne:].reshape(r, c // 2)
        nib = np.stack([d & 15, d >> 4], -1).reshape(r, c)
        v = np.where(nib & 8, -1.0, 1.0) * (nib & 7) / 4.0
        return (v.reshape(r, c // 16, 16) * np.ldexp(1.0, exp - 127)[:, :, None]).reshape(r, c).astype(np.float32)


def prepare(gb: float, fmt: str = "bfp4", threads: int = 12, source: str = "gguf"):
    import ttnn
    dt = {"bfp4": ttnn.bfloat4_b, "bfp8": ttnn.bfloat8_b}[fmt]
    W = Weights()
    sets = resident_sets(gb, fmt)
    d = CACHE / (tag(gb, fmt) + ("" if source == "gguf" else f"-{source}") + ("-dtrace" if RES_RANK == "trace" else ""))
    d.mkdir(parents=True, exist_ok=True)
    json.dump({str(k): v for k, v in sets.items()}, open(d / "sets.json", "w"))
    pool = ThreadPoolExecutor(threads)
    t0 = time.time()
    for L, ids in sets.items():
        fgu, fd = d / f"blk.{L}.gate_up.tensorbin", d / f"blk.{L}.down.tensorbin"
        if not ids or (fgu.exists() and fd.exists()):
            continue
        gu = torch.empty(1, len(ids), H, 2 * F)
        dn = torch.empty(1, len(ids), F, H)
        if source == "lotto":                                              # already W^T [in, out], exact bfp4 values
            rows, where = LottoPool().layer(L)
            for i, e in enumerate(ids):
                gu[0, i, :, :F] = torch.from_numpy(LottoPool.matrix(rows, where, e, 0))
                gu[0, i, :, F:] = torch.from_numpy(LottoPool.matrix(rows, where, e, 1))
                dn[0, i] = torch.from_numpy(LottoPool.matrix(rows, where, e, 2))
        else:
            names = [f"blk.{L}.ffn_{p}_exps.weight" for p in ("gate", "up", "down")]
            jobs = [[pool.submit(W.expert, n, e) for n in names] for e in ids]
            for i, (jg, ju, jd) in enumerate(jobs):
                gu[0, i, :, :F] = torch.from_numpy(jg.result().T)      # gate [640, 2560] -> [2560, 640]
                gu[0, i, :, F:] = torch.from_numpy(ju.result().T)
                dn[0, i] = torch.from_numpy(jd.result().T)               # down [2560, 640] -> [640, 2560]
        ttnn.dump_tensor(str(fgu), ttnn.from_torch(gu, dtype=dt, layout=ttnn.TILE_LAYOUT))
        ttnn.dump_tensor(str(fd), ttnn.from_torch(dn, dtype=dt, layout=ttnn.TILE_LAYOUT))
        print(f"layer {L}: {len(ids)} experts, {time.time() - t0:.0f} s", flush=True)
    print(f"prepared {sum(map(len, sets.values()))} {fmt} experts ({gb} GB budget) in {time.time() - t0:.0f} s -> {d}")


def _sparse_cfg(ttnn, n):
    """sparse_matmul program config (qwen36 tt/moe/decode.py): largest divisor of n tiles on <= 8x8 cores."""
    n_tiles = math.ceil(n / 32)
    best, bx, by = 1, 1, 1
    for c in range(1, min(65, n_tiles + 1)):
        if n_tiles % c:
            continue
        for cy in range(1, 9):
            if c % cy == 0 and c // cy <= 8 and c > best:
                best, bx, by = c, c // cy, cy
                break
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(bx, by), in0_block_w=1, out_subblock_h=1, out_subblock_w=1,
        out_block_h=1, out_block_w=n_tiles // best, per_core_M=1, per_core_N=n_tiles // best, fuse_batch=False,
        fused_activation=None, mcast_in0=True)


class DeviceExperts:
    def __init__(self, device, gb: float, fmt: str = "bfp4", source: str = "gguf"):
        import ttnn
        self.ttnn = ttnn
        self.dev = device
        self.fmt = fmt
        d = CACHE / (tag(gb, fmt) + ("" if source == "gguf" else f"-{source}") + ("-dtrace" if RES_RANK == "trace" else ""))
        self.sets = {int(k): v for k, v in json.load(open(d / "sets.json")).items()}
        self.slot = {L: {e: i for i, e in enumerate(v)} for L, v in self.sets.items()}
        self.gu, self.dn = {}, {}
        t0 = time.time()
        for L, ids in self.sets.items():
            if ids:
                self.gu[L] = ttnn.to_device(ttnn.load_tensor(str(d / f"blk.{L}.gate_up.tensorbin")), device,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
                self.dn[L] = ttnn.to_device(ttnn.load_tensor(str(d / f"blk.{L}.down.tensorbin")), device,
                                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        self.load_s = time.time() - t0
        self.cfg_gu = _sparse_cfg(ttnn, 2 * F)
        self.cfg_dn = _sparse_cfg(ttnn, H)
        self.n_resident = sum(map(len, self.sets.values()))
        fid = getattr(ttnn.MathFidelity, os.environ.get("P100A_EXPERT_FIDELITY", "HiFi2"))
        self.ck = ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                                   packer_l1_acc=True)

    def split(self, L, ids, wts):
        """-> (resident [(slot, w)], cold ids, cold wts)."""
        s = self.slot.get(L, {})
        res = [(s[int(e)], float(w)) for e, w in zip(ids, wts) if int(e) in s]
        cold = [(int(e), float(w)) for e, w in zip(ids, wts) if int(e) not in s]
        return res, [e for e, _ in cold], [w for _, w in cold]

    def forward_sp(self, L, x, sp, wsel=None, mask=None, wsel2=None, sel_lo=None, sel_hi=None):
        """Trace-friendly: sp is a persistent row-major bf16 [1,1,1,n_L] tensor of routing weights (0 = skip);
        nnz is left to the op (read from sp on the device). wsel (optional, TILE bf16 [1,1,32,32*n_L], row 0 holds w_e
        at column 32*e) turns the weighted sum over slots into one matmul (2.3 ms -> 0.2 ms per layer at n_L=200).
        Multi-row (M7 verify): rows 0..T-1 are tokens, sp marks the union of their slots, wsel row r holds w_{r,e} at
        column 32*e + r, and `mask` (bf16 [1,1,32,H], 1 on the valid rows) replaces the row-0 mask."""
        ttnn = self.ttnn
        n = len(self.sets[L])
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        if getattr(self, "_row0", None) is None:
            m = torch.zeros(1, 1, 32, H)
            m[0, 0, 0] = 1
            self._row0 = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
            self._zero = ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                         device=self.dev)
        x = ttnn.where(self._row0 if mask is None else mask, x, self._zero)
        gu = ttnn.sparse_matmul(x, self.gu[L], sparsity=sp, nnz=None, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                output_tile=ttnn.Tile([32, 32]), program_config=self.cfg_gu, dtype=ttnn.bfloat16,
                                compute_kernel_config=self.ck)
        gu = ttnn.reshape(gu, (1, n, 32, 2 * F))
        h = ttnn.multiply(ttnn.silu(ttnn.slice(gu, [0, 0, 0, 0], [1, n, 32, F])),
                          ttnn.slice(gu, [0, 0, 0, F], [1, n, 32, 2 * F]))
        if wsel2 is not None:
            # with is_input_a_sparse, sparse_matmul is wrong as soon as more than 16 rows of the 32-row tile are
            # non-zero (rows 0 and >= 16 off by 60-130%; tt_qwen38/sparse_rows_test.py). Run the down projection twice:
            # rows 0..15 as they are, rows 16..31 moved up to 0..15 by an exact 0/1 selection matmul.
            lo = ttnn.matmul(ttnn.repeat(sel_lo, ttnn.Shape([1, n, 1, 1])), h, dtype=ttnn.bfloat16)
            hi = ttnn.matmul(ttnn.repeat(sel_hi, ttnn.Shape([1, n, 1, 1])), h, dtype=ttnn.bfloat16)
            y = None
            for part, ws in ((lo, wsel), (hi, wsel2)):
                d = ttnn.sparse_matmul(part, self.dn[L], sparsity=sp, nnz=None, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                       output_tile=ttnn.Tile([32, 32]), program_config=self.cfg_dn,
                                       is_input_a_sparse=True, dtype=ttnn.bfloat16, compute_kernel_config=self.ck)
                t = ttnn.matmul(ws, ttnn.reshape(d, (1, 1, n * 32, H)), dtype=ttnn.float32, compute_kernel_config=self.ck)
                y = t if y is None else ttnn.add(y, t)
            return y
        dn = ttnn.sparse_matmul(h, self.dn[L], sparsity=sp, nnz=None, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                output_tile=ttnn.Tile([32, 32]), program_config=self.cfg_dn, is_input_a_sparse=True,
                                dtype=ttnn.bfloat16, compute_kernel_config=self.ck)
        if wsel is not None:
            return ttnn.matmul(wsel, ttnn.reshape(dn, (1, 1, n * 32, H)), dtype=ttnn.float32,
                               compute_kernel_config=self.ck)
        dn = ttnn.reshape(dn, (1, n, 32, H))
        wt = ttnn.typecast(ttnn.to_layout(ttnn.reshape(sp, [1, n, 1, 1]), ttnn.TILE_LAYOUT), ttnn.float32)
        return ttnn.sum(ttnn.multiply(ttnn.typecast(dn, ttnn.float32), wt), dim=1, keepdim=True)

    def forward_ix(self, L, x, sp, idx, wsel, mask):
        """M8 gather path (trace-friendly): only the A slots listed in idx (UINT16 RM [1,1,1,A], padded with any valid
        slot whose weight is 0) are computed, so the cost follows the active experts, not the resident count.
        wsel: TILE bf16 [1,1,32,32*A], row r holds w_{r,i} at column 32*i + r. sp: required operand, unread."""
        ttnn = self.ttnn
        A = idx.shape[-1]
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        x = ttnn.where(mask, x, self._zero_ix())
        gu = ttnn.sparse_matmul(x, self.gu[L], sparsity=sp, indices=idx, is_input_a_sparse=False,
                                is_input_b_sparse=True, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                program_config=self.cfg_gu, dtype=ttnn.bfloat16, compute_kernel_config=self.ck)
        gu = ttnn.reshape(gu, (1, A, 32, 2 * F))
        h = ttnn.multiply(ttnn.silu(ttnn.slice(gu, [0, 0, 0, 0], [1, A, 32, F])),
                          ttnn.slice(gu, [0, 0, 0, F], [1, A, 32, 2 * F]))
        dn = ttnn.sparse_matmul(h, self.dn[L], sparsity=sp, indices=idx, is_input_a_sparse=True,
                                is_input_b_sparse=True, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                program_config=self.cfg_dn, dtype=ttnn.bfloat16, compute_kernel_config=self.ck)
        return ttnn.matmul(wsel, ttnn.reshape(dn, (1, 1, A * 32, H)), dtype=ttnn.float32,
                           compute_kernel_config=self.ck)

    def _zero_ix(self):
        if getattr(self, "_zero", None) is None:
            self._zero = self.ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=self.ttnn.bfloat16,
                                              layout=self.ttnn.TILE_LAYOUT, device=self.dev)
        return self._zero

    def forward(self, L, x, res):
        """x: [1,1,32,2560] bf16 tile (row 0); res: [(slot, weight)] -> [1,1,32,2560] fp32 (row 0) or None."""
        ttnn = self.ttnn
        if not res:
            return None
        n = len(self.sets[L])
        w = torch.zeros(1, 1, 1, n)
        for i, wt in res:
            w[0, 0, 0, i] = wt
        sp = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.dev)
        if x.dtype != ttnn.bfloat16:
            x = ttnn.typecast(x, ttnn.bfloat16)
        # rows 1..31 must be zero: sparse_matmul's row 0 is not independent of the tile's other rows, and tilize after
        # a row-major reshape leaves garbage in the padding (found in 25-99% error until this)
        if getattr(self, "_row0", None) is None:
            m = torch.zeros(1, 1, 32, H)
            m[0, 0, 0] = 1
            self._row0 = ttnn.from_torch(m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.dev)
            self._zero = ttnn.from_torch(torch.zeros(1, 1, 32, H), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                         device=self.dev)
        x = ttnn.where(self._row0, x, self._zero)
        gu = ttnn.sparse_matmul(x, self.gu[L], sparsity=sp, nnz=len(res), memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                output_tile=ttnn.Tile([32, 32]), program_config=self.cfg_gu, dtype=ttnn.bfloat16,
                                compute_kernel_config=self.ck)
        gu = ttnn.reshape(gu, (1, n, 32, 2 * F))
        h = ttnn.multiply(ttnn.silu(ttnn.slice(gu, [0, 0, 0, 0], [1, n, 32, F])),
                          ttnn.slice(gu, [0, 0, 0, F], [1, n, 32, 2 * F]))
        dn = ttnn.sparse_matmul(h, self.dn[L], sparsity=sp, nnz=len(res), memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                output_tile=ttnn.Tile([32, 32]), program_config=self.cfg_dn, is_input_a_sparse=True,
                                dtype=ttnn.bfloat16, compute_kernel_config=self.ck)
        dn = ttnn.reshape(dn, (1, n, 32, H))
        wt = ttnn.typecast(ttnn.to_layout(ttnn.reshape(sp, [1, n, 1, 1]), ttnn.TILE_LAYOUT), ttnn.float32)  # [1,n,1,1]
        y = ttnn.sum(ttnn.multiply(ttnn.typecast(dn, ttnn.float32), wt), dim=1, keepdim=True)
        return y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--gb", type=float, default=20)
    ap.add_argument("--fmt", default="bfp4", choices=["bfp4", "bfp8"])
    ap.add_argument("--source", default="gguf", choices=["gguf", "lotto"])
    ap.add_argument("--check-lotto", type=int, default=-1, help="compare layer N of the Lottolabs pool with the GGUF")
    a = ap.parse_args()
    if a.check_lotto >= 0:
        import ttnn
        L, W = a.check_lotto, Weights()
        rows, where = LottoPool().layer(L)
        for e in (0, 7, 100):
            for m, p in enumerate(("gate", "up", "down")):
                lw = LottoPool.matrix(rows, where, e, m)                       # [in, out]
                gw = W.expert(f"blk.{L}.ffn_{p}_exps.weight", e).T             # [in, out]
                rt = ttnn.to_torch(ttnn.from_torch(torch.from_numpy(lw)[None, None], dtype=ttnn.bfloat4_b,
                                                   layout=ttnn.TILE_LAYOUT)).float()[0, 0].numpy()
                print(f"L{L} e{e} {p}: rel diff vs GGUF {np.linalg.norm(lw - gw) / np.linalg.norm(gw):.3f}, "
                      f"re-pack exact {np.array_equal(rt, lw)}")
    elif a.prepare:
        prepare(a.gb, a.fmt, source=a.source)
