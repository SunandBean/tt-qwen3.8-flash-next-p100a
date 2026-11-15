"""Weights of the qwen4exp Coder GGUF as float32 numpy arrays (mmap, dequantized on demand).

Shapes are returned in numpy / torch `nn.Linear` order: a GGUF tensor [ne0=in, ne1=out] comes back as [out, in];
a routed-expert tensor [in, out, E] gives one expert at a time as [out, in].

Dequantization: gguf-py 0.19 (`gguf.quants.dequantize`) for the ggml types, our own for `Q2_0` (ggml type 42, the
GSQ-RCO routed-expert encoding: per 64 values an fp16 scale d and 16 bytes of 2-bit codes, value = (code - 1) * d,
codes little-endian within each byte; transcribed from Strata tools/gguf_writer.py `dequantize_q2_0`).
"""
from __future__ import annotations

import mmap
import pathlib
from functools import lru_cache

import numpy as np

import tt_qwen38.third_party  # noqa: F401  (vendored gguf-py on sys.path)
from gguf import GGMLQuantizationType as QT
from gguf import quants as gq

from .third_party.strata_gguf_reader import BLOCK_GEOMETRY, GGUFFile

import os

# P100A_MODEL = coder (default: the Coder IQ1_M GGUF) | full (the base model, all 512 experts, IQ3_XXS GGUF)
MODEL = os.environ.get("P100A_MODEL", "coder")
_MODEL_FILES = {"coder": ("coder-IQ1_M", "IQ1_M"), "full": ("IQ3_XXS", "IQ3_XXS")}[MODEL]
# the two shards of the selected model: pinned, downloaded / checked on first use (tt_qwen38/sources.py)


def dequant_q2_0(raw: np.ndarray) -> np.ndarray:
    """raw: uint8 [..., nblocks*18] -> float32 [..., nblocks*64]."""
    lead = raw.shape[:-1]
    b = raw.reshape(-1, 18)
    d = b[:, :2].copy().view(np.float16).astype(np.float32)          # [n, 1]
    qs = b[:, 2:]                                                    # [n, 16]
    codes = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=-1).reshape(-1, 64)  # j = byte*4 + shift/2
    out = (codes.astype(np.float32) - 1.0) * d
    return out.reshape(*lead, -1)


def dequant(raw: np.ndarray, type_name: str) -> np.ndarray:
    """raw: uint8 [rows, row_bytes] -> float32 [rows, row_elems]."""
    if type_name == "F32":
        return raw.view(np.float32).copy()
    if type_name == "F16":
        return raw.view(np.float16).astype(np.float32)
    if type_name == "Q2_0":
        return dequant_q2_0(raw)
    return gq.dequantize(raw, QT[type_name]).astype(np.float32, copy=False)


class Weights:
    def __init__(self, shards=None):
        if shards is None:
            from tt_qwen38.sources import gguf_shards
            shards = gguf_shards()
        self.files = []
        self.index = {}
        for p in shards:
            g = GGUFFile(pathlib.Path(p))
            fh = open(p, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            self.files.append((g, fh, mm))
            for t in g.tensors:
                self.index[t.name] = (len(self.files) - 1, t)
        self.meta = self.files[0][0].metadata

    # ---- metadata
    def kv(self, key: str, default=None):
        return self.meta.get(key, default)

    def info(self, name: str):
        return self.index[name][1]

    def has(self, name: str) -> bool:
        return name in self.index

    def _raw(self, name: str, byte_start: int = 0, nbytes: int | None = None) -> np.ndarray:
        fi, t = self.index[name]
        g, _, mm = self.files[fi]
        total = t.expected_bytes()
        if nbytes is None:
            nbytes = total - byte_start
        off = g.data_start + t.offset + byte_start
        return np.frombuffer(mm, dtype=np.uint8, count=nbytes, offset=off)

    def _row_bytes(self, t) -> int:
        be, bb = BLOCK_GEOMETRY[t.type_name]
        assert t.shape[0] % be == 0, (t.name, t.shape, t.type_name)
        return t.shape[0] // be * bb

    # ---- tensors
    def tensor(self, name: str) -> np.ndarray:
        """Whole tensor, float32, numpy order (reversed GGUF dims)."""
        t = self.info(name)
        if len(t.shape) == 1:
            return dequant(self._raw(name)[None, :], t.type_name)[0]
        rb = self._row_bytes(t)
        rows = int(np.prod(t.shape[1:]))
        out = dequant(self._raw(name).reshape(rows, rb), t.type_name)
        return out.reshape(*reversed(t.shape))

    def rows(self, name: str, row_ids) -> np.ndarray:
        """Selected rows of a 2-D tensor [ne0, ne1] -> [len(row_ids), ne0] (e.g. embedding lookups)."""
        t = self.info(name)
        rb = self._row_bytes(t)
        out = []
        for r in np.asarray(row_ids, dtype=np.int64).ravel():
            out.append(dequant(self._raw(name, int(r) * rb, rb)[None, :], t.type_name)[0])
        return np.stack(out)

    def expert(self, name: str, e: int) -> np.ndarray:
        """One expert of a [in, out, E] tensor -> float32 [out, in]."""
        t = self.info(name)
        n_in, n_out, n_e = t.shape
        rb = self._row_bytes(t)
        per = rb * n_out
        raw = self._raw(name, e * per, per).reshape(n_out, rb)
        return dequant(raw, t.type_name)


@lru_cache(maxsize=1)
def default_weights() -> Weights:
    return Weights()
