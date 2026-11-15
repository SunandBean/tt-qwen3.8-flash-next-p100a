"""Dense weights of the Coder GGUF converted for the P100a and cached as ttnn tensor files.

`python -m tt_qwen38.weights --prepare` (no device needed) dequantizes every dense
tensor from the GGUF, lays it out for ttnn.matmul ([in, out], TILE), converts to BFP8 (BF16 for small/precision
-critical ones) and writes `<card cache>/<name>.tensorbin` (tt_qwen38/sources.py card_cache).  A device session then only loads files (`DeviceWeights`),
so the card lease is held for device work only.

Layout notes (decode, one token = row 0 of a 32-row tile):
  - hyper-connection weights are split per branch b (the residual is 4 tensors of [1,1,32,2560]):
    down_b [2560, 320], up_b [320, 2560], inject_b [2560, 32] (4 used columns), norm_b [1,1,1,2560] (stored 1+w).
  - GDN: qkv [2560, 10240], z [2560, 6144], ab [2560, 128] (alpha 0..47, beta 48..95), out [6144, 2560];
    conv taps as 4 rows [1,1,1,10240]; dt_bias, ssm_a [1,1,1,48]; ssm_norm [1,1,1,128].
  - attention: q [2560, 12288] (per head [q|gate]), kv [2560, 1024] (k 0..511, v 512..1023), out [6144, 2560],
    q_norm/k_norm [1,1,1,256]; rope permutation [256, 256] (rotate-half on dims 0..63) built at load.
  - shared expert: gu [2560, 1280] (gate 0..639, up 640..1279), down [640, 2560], gate vector [2560, 32] (col 0).
  - LM head [2560, 248320]; final mixer = output_hc_* like a GR read.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import time

import torch

from tt_qwen38.ref.gguf_weights import Weights

from tt_qwen38.sources import card_cache

CACHE = card_cache()
H, HC, LR = 2560, 4, 320


def _t(W, name):
    return torch.from_numpy(W.tensor(name).copy())


def tensors_for_layer(W: Weights, L: int, attention: bool):
    """Yield (key, torch tensor, kind) with kind in {'bfp8', 'bf16', 'f32row'}."""
    p = f"blk.{L}."
    for blk in ("attn", "ffn"):
        norm = _t(W, f"{p}hc_{blk}_norm.weight")                 # [10240]
        down = _t(W, f"{p}hc_{blk}_down.weight")                 # [320, 10240]
        up = _t(W, f"{p}hc_{blk}_up.weight")                     # [10240, 320]
        inj = _t(W, f"{p}hc_{blk}_inject.weight")                # [4, 10240]
        for b in range(HC):
            s = slice(b * H, (b + 1) * H)
            yield f"{p}hc_{blk}.norm{b}", norm[s].reshape(1, 1, 1, H), "f32row"
            yield f"{p}hc_{blk}.down{b}", down[:, s].T.contiguous(), "bf16"
            yield f"{p}hc_{blk}.up{b}", up[s, :].T.contiguous(), "bf16"
            ij = torch.zeros(H, 32)
            ij[:, :HC] = inj[:, s].T
            yield f"{p}hc_{blk}.inj{b}", ij, "bf16"
    if attention:
        yield p + "attn.q", _t(W, p + "attn_q.weight").T.contiguous(), "bfp8"
        kv = torch.cat([_t(W, p + "attn_k.weight"), _t(W, p + "attn_v.weight")], 0)   # [1024, 2560]
        yield p + "attn.kv", kv.T.contiguous(), "bfp8"
        yield p + "attn.out", _t(W, p + "attn_output.weight").T.contiguous(), "bfp8"
        yield p + "attn.q_norm", _t(W, p + "attn_q_norm.weight").reshape(1, 1, 1, -1), "f32row"
        yield p + "attn.k_norm", _t(W, p + "attn_k_norm.weight").reshape(1, 1, 1, -1), "f32row"
        # QSA indexer (4 heads x 128; keys pooled over blocks of 4 tokens)
        yield p + "idx.q", _t(W, p + "indexer.q_proj.weight").T.contiguous(), "bf16"          # [2560, 512]
        yield p + "idx.k", _t(W, p + "indexer.k_proj.weight").T.contiguous(), "bf16"          # [2560, 128]
        yield p + "idx.q_norm", _t(W, p + "indexer.q_norm.weight").reshape(1, 1, 1, -1), "f32row"
        yield p + "idx.k_norm", _t(W, p + "indexer.k_norm.weight").reshape(1, 1, 1, -1), "f32row"
    else:
        yield p + "gdn.qkv", _t(W, p + "attn_qkv.weight").T.contiguous(), "bfp8"
        yield p + "gdn.z", _t(W, p + "attn_gate.weight").T.contiguous(), "bfp8"
        ab = torch.zeros(H, 128)
        ab[:, :48] = _t(W, p + "ssm_alpha.weight").T
        ab[:, 48:96] = _t(W, p + "ssm_beta.weight").T
        yield p + "gdn.ab", ab, "bf16"
        yield p + "gdn.out", _t(W, p + "ssm_out.weight").T.contiguous(), "bfp8"
        conv = _t(W, p + "ssm_conv1d.weight")                    # [10240, 4]
        for k in range(4):
            yield f"{p}gdn.conv{k}", conv[:, k].reshape(1, 1, 1, -1).contiguous(), "f32row"
        yield p + "gdn.dt", _t(W, p + "ssm_dt.bias").reshape(1, 1, 1, -1), "f32row"
        yield p + "gdn.a", _t(W, p + "ssm_a").reshape(1, 1, 1, -1), "f32row"
        yield p + "gdn.norm", _t(W, p + "ssm_norm.weight").reshape(1, 1, 1, -1), "f32row"
    gu = torch.cat([_t(W, p + "ffn_gate_shexp.weight"), _t(W, p + "ffn_up_shexp.weight")], 0)  # [1280, 2560]
    yield p + "sh.gu", gu.T.contiguous(), "bfp8"
    yield p + "sh.down", _t(W, p + "ffn_down_shexp.weight").T.contiguous(), "bfp8"
    g = torch.zeros(H, 32)
    g[:, 0] = _t(W, p + "ffn_gate_inp_shexp.weight")
    yield p + "sh.gate", g, "bf16"


def tensors_global(W: Weights):
    norm = _t(W, "output_hc_norm.weight")
    down = _t(W, "output_hc_down.weight")
    up = _t(W, "output_hc_up.weight")
    for b in range(HC):
        s = slice(b * H, (b + 1) * H)
        yield f"out_hc.norm{b}", norm[s].reshape(1, 1, 1, H), "f32row"
        yield f"out_hc.down{b}", down[:, s].T.contiguous(), "bf16"
        yield f"out_hc.up{b}", up[s, :].T.contiguous(), "bf16"
    yield "lm_head", _t(W, "output.weight").T.contiguous(), "bfp8"


def is_attention(L: int) -> bool:
    return (L + 1) % 4 == 0


def all_tensors(W: Weights, layers=range(48)):
    for L in layers:
        yield from tensors_for_layer(W, L, is_attention(L))
    yield from tensors_global(W)


def _ttnn_dtype(ttnn, kind):
    return {"bfp8": ttnn.bfloat8_b, "bf16": ttnn.bfloat16, "f32row": ttnn.float32}[kind]


def prepare(layers=range(48), only_missing=True):
    import ttnn
    CACHE.mkdir(parents=True, exist_ok=True)
    W = Weights()
    t0 = time.time()
    n = 0
    for key, t, kind in all_tensors(W, layers):
        f = CACHE / f"{key}.tensorbin"
        if only_missing and f.exists():
            continue
        x = t.to(torch.bfloat16) if kind == "bf16" else t.float()
        if x.ndim == 2:
            x = x.reshape(1, 1, *x.shape)
        tt = ttnn.from_torch(x, dtype=_ttnn_dtype(ttnn, kind), layout=ttnn.TILE_LAYOUT)
        ttnn.dump_tensor(str(f), tt)
        n += 1
    print(f"prepared {n} tensors in {time.time() - t0:.0f} s -> {CACHE}")


class DeviceWeights:
    """Loads cached tensors onto the device on first use."""

    def __init__(self, device):
        import ttnn
        self.ttnn = ttnn
        self.dev = device
        self._d = {}

    def __getitem__(self, key):
        if key not in self._d:
            f = CACHE / f"{key}.tensorbin"
            if not f.exists():
                raise FileNotFoundError(f"{f}: run `python -m tt_qwen38.weights --prepare` first")
            t = self.ttnn.load_tensor(str(f))
            self._d[key] = self.ttnn.to_device(t, self.dev, memory_config=self.ttnn.DRAM_MEMORY_CONFIG)
        return self._d[key]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--layers", default="0-47")
    a = ap.parse_args()
    lo, hi = map(int, a.layers.split("-"))
    if a.prepare:
        prepare(range(lo, hi + 1))
