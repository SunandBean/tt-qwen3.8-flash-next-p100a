"""prefix cache. Agent loops and multi-turn chats resend the whole history; this keeps the model state at chosen
prompt positions and resumes from the longest cached one.

State at position b (b tokens read) = GDN recurrent state S[L] + conv history hist3[L] (36 layers, ~160 MB), the
attention K/V rows < b, the QSA indexer K rows < b, the MTP K/V cells < b (12 + 12 + 1 caches), and the host-side
PLE state. Snapshots live in fixed slots on the card (allocated before the traces are captured; a slot is a full
copy of those buffers, ~1.1 GB at 32k context), copied with ttnn.copy (device to device). LRU over slots.

Exactness: the prompt is always read in 28-row chunks aligned at absolute positions 0, 28, 56, ... (as before), and
snapshots are taken only at chunk ends, so a resumed run computes exactly the same chunks as a cold one: the outputs
are bit-identical with or without the cache (checked with agent-trace replays during the port).

Boundaries: the last chunk end <= n-1 (n = prompt length; at least one token is always read, it gives the first
logits) and the chunk end at or below the end of the system turn (shared by every request of an agent / client).
A snapshot at b is keyed by the tokens [0, b] (b+1 tokens): MTP cell b-1 is built from token b.

K/V shortcut: the live caches still hold the rows of the previous prompt (`live`); when a snapshot's key is a prefix
of it, only the ~160 MB GDN part is copied back.
"""
from __future__ import annotations

import copy
import time

import ttnn


class Slot:
    def __init__(self, bufs):
        self.bufs = bufs          # name -> device tensor (same shapes as the live buffers)
        self.key = None           # token ids [0, b]
        self.b = 0
        self.ple = None
        self.used = 0.0


def state_tensors(model):
    """(gdn, kv): name -> live device tensor. gdn = recurrent part (always copied), kv = positional caches."""
    gdn, kv = {}, {}
    for L in model.S:
        gdn[f"S{L}"] = model.S[L]
        gdn[f"h{L}"] = model.hist3[L]
    for L in model.kc:
        kv[f"k{L}"], kv[f"v{L}"] = model.kc[L], model.vc[L]
    for L in getattr(model, "ikc", {}) or {}:
        kv[f"i{L}"] = model.ikc[L]
    if getattr(model, "mkc", None) is not None:
        kv["mk"], kv["mv"] = model.mkc, model.mvc
    return gdn, kv


def alloc_like(tensors: dict, dev) -> dict:
    """uninitialized device copies (call before any trace capture)"""
    return {name: ttnn.allocate_tensor_on_device(t.shape, t.dtype, t.layout, dev, ttnn.DRAM_MEMORY_CONFIG)
            for name, t in tensors.items()}


class PrefixCache:
    def __init__(self, model, slots: int, unit: int):
        self.m, self.unit = model, unit
        self.gdn, self.kv = state_tensors(model)    # gdn: always restored; kv: only when the live rows differ
        self.slots = [Slot(alloc_like({**self.gdn, **self.kv}, model.dev)) for _ in range(slots)]
        self.slot_bytes = sum(_nbytes(t) for t in list(self.gdn.values()) + list(self.kv.values()))
        self.live = []            # token ids whose K/V rows (and MTP cells) are in the live caches
        self.stats = {"lookups": 0, "hits": 0, "hit_tokens": 0, "saves": 0, "kv_restores": 0,
                      "restore_s": 0.0, "save_s": 0.0}

    # ---- planning (host only)
    def system_end(self, ids) -> int:
        """position after the first <|im_end|>\\n when the prompt starts with a system turn, else 0"""
        if len(ids) > 1 and ids[0] == 248045:                     # <|im_start|>
            try:
                return ids.index(248046) + 2                       # <|im_end|> "\n"
            except ValueError:
                return 0
        return 0

    def plan(self, ids) -> set:
        n, u = len(ids), self.unit
        pts = {((n - 1) // u) * u, (min(self.system_end(ids), n - 1) // u) * u}
        return {b for b in pts if b > 0}

    def lookup(self, ids):
        """-> (slot, b) for the longest usable snapshot, or (None, 0)"""
        best, bb = None, 0
        n = len(ids)
        for s in self.slots:
            if s.key is None or s.b > n - 1 or s.b <= bb:
                continue
            if ids[:s.b + 1] == s.key:
                best, bb = s, s.b
        return best, bb

    # ---- device copies
    def restore(self, s: Slot):
        t0 = time.perf_counter()
        for name, t in self.gdn.items():
            ttnn.copy(s.bufs[name], t)
        if self.live[:s.b + 1] != s.key:
            for name, t in self.kv.items():
                ttnn.copy(s.bufs[name], t)
            self.stats["kv_restores"] += 1
            self.live = list(s.key)
        s.used = time.time()
        self.stats["restore_s"] += time.perf_counter() - t0
        return copy.deepcopy(s.ple)

    def save(self, ids, b: int, ple):
        """the live state is at position b (prompt tokens ids[:b] read, MTP cells < b built)"""
        key = list(ids[:b + 1])
        for s in self.slots:
            if s.key == key:
                s.used = time.time()
                return
        if not self.slots:
            return
        t0 = time.perf_counter()
        s = min(self.slots, key=lambda x: x.used)
        for name, t in list(self.gdn.items()) + list(self.kv.items()):
            ttnn.copy(t, s.bufs[name])
        s.key, s.b, s.ple, s.used = key, b, copy.deepcopy(ple), time.time()
        self.stats["saves"] += 1
        self.stats["save_s"] += time.perf_counter() - t0


def _nbytes(t) -> int:
    n = 1
    for d in t.shape:
        n *= d
    return n * (2 if t.dtype == ttnn.bfloat16 else 4)
