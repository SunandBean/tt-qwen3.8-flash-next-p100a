"""several requests at once on one card, by time slicing.

One scheduler thread owns the card. Each request is a SpecModel.steps() generator (it yields after every prompt chunk
and every decode round). Up to `max_conc` requests are active; the scheduler runs one for a time quantum (default
0.5 s), then switches to the next in round-robin order; more requests wait in FIFO order (they get pings).

Switching = copying the live state of the running request (GDN state, conv history, K/V, indexer and MTP caches:
~1.06 GB at 32k, ~7 ms each way on the card) into a free slot and the next request's slot back, plus the host
bookkeeping (position, tokens, PLE state, which prompt the live K/V rows hold). N active requests need N slots
(the running one is parked before the next one leaves its slot); one request at a time needs none.

Why not batched decode rows: the verify pass reads T rows of one sequence; rows of two sequences in one pass would
need per-sequence recurrent state in the fused GDN core / commit / conv kernels, per-sequence K/V caches and commits
(not done for this model).
"""
from __future__ import annotations

import collections
import threading
import time

import ttnn

from .prefix import alloc_like, state_tensors


class Session:
    def __init__(self, ids, max_new, kwargs, on_token):
        self.ids, self.max_new, self.kwargs, self.on_token = ids, max_new, kwargs, on_token
        self.result = {}
        self.gen = None
        self.slot = None              # slot index while suspended
        self.host = None              # host bookkeeping while suspended
        self.cancelled = False
        self.done = threading.Event()
        self.error = None
        self.t_submit = time.time()
        self.t_admit = None


class Scheduler:
    def __init__(self, model, max_conc: int = 2, quantum_s: float = 0.5):
        self.m, self.max_conc, self.quantum = model, max(1, max_conc), quantum_s
        self.gdn, self.kv = state_tensors(model)
        self.tensors = {**self.gdn, **self.kv}
        # a switch parks the running request before the next one leaves its slot: N active -> N slots
        self.slots = [alloc_like(self.tensors, model.dev) for _ in range(self.max_conc if self.max_conc > 1 else 0)]
        self.free = list(range(len(self.slots)))
        self.wait = collections.deque()
        self.active: list[Session] = []
        self.owner: Session | None = None
        self.cv = threading.Condition()
        self.rr = 0
        self.closed = False
        self.stats = {"switches": 0, "switch_s": 0.0, "admitted": 0, "finished": 0, "max_active": 0}
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    # ---- API for the HTTP threads
    def submit(self, sess: Session):
        with self.cv:
            self.wait.append(sess)
            self.cv.notify()

    def busy(self) -> bool:
        with self.cv:
            return bool(self.wait or self.active)

    def queue_pos(self, sess) -> int:
        with self.cv:
            return list(self.wait).index(sess) + 1 if sess in self.wait else 0

    # ---- host bookkeeping of the live state
    def _host_get(self):
        m, pre, pc = self.m, self.m.pre, self.m.pcache
        return (m.pos, list(m.tokens), m.ple_state,
                (pre.pos, list(pre.tokens), pre.ple_state) if pre is not None else None,
                list(pc.live) if pc is not None else None)

    def _host_set(self, h):
        m, pre, pc = self.m, self.m.pre, self.m.pcache
        m.pos, m.tokens, m.ple_state = h[0], list(h[1]), h[2]
        if pre is not None:
            pre.pos, pre.tokens, pre.ple_state = h[3][0], list(h[3][1]), h[3][2]
        if pc is not None:
            pc.live = list(h[4])

    def _make_live(self, s: Session):
        if self.owner is s:
            return
        t0 = time.perf_counter()
        o = self.owner
        if o is not None and o in self.active:                 # park the running request
            o.slot = self.free.pop()
            for name, t in self.tensors.items():
                ttnn.copy(t, self.slots[o.slot][name])
            o.host = self._host_get()
        if s.slot is not None:                                 # bring back a parked one
            for name, t in self.tensors.items():
                ttnn.copy(self.slots[s.slot][name], t)
            self._host_set(s.host)
            self.free.append(s.slot)
            s.slot, s.host = None, None
        self.owner = s
        self.stats["switches"] += 1
        self.stats["switch_s"] += time.perf_counter() - t0

    def _finish(self, s: Session, err=None):
        if s in self.active:
            self.active.remove(s)
        if s.slot is not None:
            self.free.append(s.slot)
            s.slot = None
        if self.owner is s:
            self.owner = None
        s.error = err
        self.stats["finished"] += 1
        s.done.set()

    # ---- the card loop
    def _loop(self):
        while True:
            with self.cv:
                while not (self.wait or self.active):
                    self.cv.wait()
                while self.wait and len(self.active) < self.max_conc:
                    s = self.wait.popleft()
                    s.t_admit = time.time()
                    self.active.append(s)
                    self.stats["admitted"] += 1
                self.stats["max_active"] = max(self.stats["max_active"], len(self.active))
                self.rr %= len(self.active)
                s = self.active[self.rr]
                self.rr += 1
            if s.cancelled:
                if s.gen is not None:
                    s.gen.close()
                self._finish(s)
                continue
            try:
                self._make_live(s)
                if s.gen is None:
                    s.gen = self.m.steps(s.ids, s.max_new, on_token=s.on_token, result=s.result, **s.kwargs)
                t0 = time.perf_counter()
                while True:
                    try:
                        next(s.gen)
                    except StopIteration:
                        self._finish(s)
                        break
                    if s.cancelled:
                        s.gen.close()
                        self._finish(s)
                        break
                    if time.perf_counter() - t0 >= self.quantum:
                        with self.cv:
                            alone = len(self.active) == 1 and not self.wait
                        if not alone:
                            break
                        t0 = time.perf_counter()
            except Exception as e:  # noqa: BLE001  (surfaced to the request's HTTP thread)
                if s.gen is not None:
                    try:
                        s.gen.close()
                    except Exception:  # noqa: BLE001
                        pass
                self.owner = None                              # live state unknown: nobody owns it
                self._finish(s, e)
