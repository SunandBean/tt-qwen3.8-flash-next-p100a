"""Python side of libhostexp.so: routed experts on the host CPU from the mmapped GGUF (see hostexp.cpp)."""
from __future__ import annotations

import ctypes
import pathlib

import numpy as np

from tt_qwen38.ref.gguf_weights import Weights

def _lib_path() -> pathlib.Path:
    """Q38_HOSTEXP_LIB, else the tt-qwen38-native package (pip-installed, or the in-tree native/ build)"""
    import os
    if os.environ.get("Q38_HOSTEXP_LIB"):
        return pathlib.Path(os.environ["Q38_HOSTEXP_LIB"])
    try:
        import tt_qwen38_native
    except ImportError:
        import sys
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "native"))
        import tt_qwen38_native
    return tt_qwen38_native.lib_path()
_GGML_TYPE = {"F32": 0, "F16": 1, "Q8_0": 8, "Q4_K": 12, "Q5_K": 13, "Q6_K": 14, "IQ2_XXS": 16, "IQ2_XS": 17,
              "IQ3_XXS": 18, "IQ4_NL": 20, "IQ3_S": 21, "IQ2_S": 22, "IQ4_XS": 23, "BF16": 30, "Q2_0": 42}


class HostExperts:
    def __init__(self, W: Weights, threads: int = 8):
        self.W = W
        self.lib = ctypes.CDLL(str(_lib_path()))
        self.lib.hx_init.argtypes = [ctypes.c_int]
        self.lib.hx_moe.argtypes = [ctypes.c_int] + [ctypes.POINTER(ctypes.c_void_p)] * 3 + [ctypes.c_int] * 3 + \
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.hx_moe_rows.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.POINTER(ctypes.c_void_p)] * 3 + \
            [ctypes.c_int] * 3 + [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.threads = self.lib.hx_init(threads)
        self._base = {}
        for i, (g, _, mm) in enumerate(W.files):
            arr = np.frombuffer(mm, dtype=np.uint8)
            self._base[i] = (arr, arr.ctypes.data + g.data_start)
        self._layer = {}

    def _layer_info(self, L: int):
        if L not in self._layer:
            out = []
            for p in ("gate", "up", "down"):
                fi, t = self.W.index[f"blk.{L}.ffn_{p}_exps.weight"]
                n_in, n_out, n_e = t.shape
                per = t.expected_bytes() // n_e
                out.append((self._base[fi][1] + t.offset, per, _GGML_TYPE[t.type_name]))
            self._layer[L] = out
        return self._layer[L]

    def moe(self, L: int, ids, wts, x: np.ndarray, H: int = 2560, F: int = 640) -> np.ndarray:
        """Weighted sum of experts `ids` (weights `wts`) of layer L for one token x [H] -> [H] float32."""
        (gb, gp, gt), (ub, up, ut), (db, dp, dt) = self._layer_info(L)
        n = len(ids)
        P = ctypes.c_void_p * n
        g = P(*[gb + int(e) * gp for e in ids])
        u = P(*[ub + int(e) * up for e in ids])
        d = P(*[db + int(e) * dp for e in ids])
        x = np.ascontiguousarray(x, dtype=np.float32)
        w = np.ascontiguousarray(wts, dtype=np.float32)
        y = np.empty(H, dtype=np.float32)
        rc = self.lib.hx_moe(n, g, u, d, gt, ut, dt, w.ctypes.data, x.ctypes.data, H, F, y.ctypes.data)
        if rc:
            raise RuntimeError(f"hx_moe failed for layer {L} (types {gt}/{ut}/{dt})")
        return y

    def moe_rows(self, L: int, rows, X: np.ndarray, H: int = 2560, F: int = 640) -> np.ndarray:
        """rows: per row r a (ids, wts) list; X [R, H]. Each unique expert is read once for all rows."""
        return moe_rows_ptrs(self.lib, self._layer_info(L), rows, X, H, F)


def moe_rows_ptrs(lib, info, rows, X: np.ndarray, H: int, F: int, up_off: int = 0) -> np.ndarray:
    """info: ((gate base, per-expert bytes, type), (up ...), (down ...)); up_off: byte offset of `up` inside the gate
    blob when gate/up are fused (MTP)."""
    (gb, gp, gt), (ub, up, ut), (db, dp, dt) = info
    by_e = {}
    for r, (ids, wts) in enumerate(rows):
        for e, w in zip(ids, wts):
            by_e.setdefault(int(e), []).append((r, float(w)))
    R = len(rows)
    if not by_e:
        return np.zeros((R, H), np.float32)
    us = sorted(by_e)
    n = len(us)
    P = ctypes.c_void_p * n
    g = P(*[gb + e * gp for e in us])
    u = P(*[ub + e * up + up_off for e in us])
    d = P(*[db + e * dp for e in us])
    off, prow, pw = [0], [], []
    for e in us:
        for r, w in by_e[e]:
            prow.append(r)
            pw.append(w)
        off.append(len(prow))
    off = np.asarray(off, np.int32)
    prow = np.asarray(prow, np.int32)
    pw = np.asarray(pw, np.float32)
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.empty((R, H), np.float32)
    rc = lib.hx_moe_rows(R, n, g, u, d, gt, ut, dt, off.ctypes.data, prow.ctypes.data, pw.ctypes.data,
                         X.ctypes.data, H, F, Y.ctypes.data)
    if rc:
        raise RuntimeError(f"hx_moe_rows failed (types {gt}/{ut}/{dt})")
    return Y

