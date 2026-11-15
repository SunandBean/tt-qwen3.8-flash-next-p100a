"""OpenAI- and Anthropic-compatible server for Qwen3.8-Flash-Next on one Tenstorrent Blackhole p100a + the host CPU.

    python -m tt_qwen38.server [--profile coder|coder-fast|full] [--host 127.0.0.1] [--port 8000] [--thinking off]

Profiles: coder (default; the ISTA-DASLab Coder IQ1_M GGUF, 256 experts, 8 GB of Lottolabs GPTQ-BFP4 experts resident on
the card, the rest on the CPU), coder-fast (16 GB resident), full (the base model, 512 experts, IQ3_XXS GGUF, 8 GB
resident from the GGUF). MTP speculative decoding (3 drafts), 28-row prompt passes, 32k context (sparse attention past
2,051 tokens), prefix cache, time slicing for several requests. Endpoints in tt_qwen38/api.py.

First start: downloads the pinned weights (tt_qwen38/sources.py) and builds the card caches (dense weights, the
resident expert set, the MTP layer) on this machine; later starts load them. Host: 64 GB RAM (the CPU experts are read
from the memory-mapped GGUF), 8+ cores (CPU experts use 8 OpenMP threads).
"""
from __future__ import annotations

import tt_qwen38.wallclock  # noqa: F401  (self-exit on a hang; P100A_WALL, 0 = off)

import argparse
import json
import os
import queue
import sys
import threading
import time
import types
from http.server import ThreadingHTTPServer

EOS = {248046, 248044}
PROFILES = {   # model, resident GB, expert source, prefix slots, max concurrent (card DRAM: see PORTING_NOTES.md)
    "coder": ("coder", 8, "lotto", 4, 3),
    "coder-fast": ("coder", 16, "lotto", 2, 1),
    "full": ("full", 8, "gguf", 4, 3),
}


def render(messages, enable_thinking: bool) -> str:
    """The steps 1-5 prompt for user/system text messages (kept for the test and profile scripts; the server itself
    renders with the GGUF chat template, tt_qwen38/chat.py, which gives the same string for these messages)."""
    out = []
    for m in messages:
        c = m.get("content") or ""
        if isinstance(c, list):
            c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
        out.append(f"<|im_start|>{m['role']}\n{c.strip()}<|im_end|>\n")
    out.append("<|im_start|>assistant\n" + ("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"))
    return "".join(out)


class Engine:
    """The card behind tt_qwen38.api (see its docstring for the generate() event protocol). One request at a time."""

    def __init__(self, model, spec: bool = False, name: str = "qwen3.8-flash-next-p100a-coder", sched=None):
        self.m = model
        self.spec = spec
        self.sched = sched                                  # tt_qwen38.sessions.Scheduler (spec path)
        from tt_qwen38.ref.tokenizer import tokenizer

        from .chat import Template
        self.tok = tokenizer(model.W)
        self.template = Template(model.W.meta["tokenizer.chat_template"])
        self.name = name
        self.lock = threading.Lock()
        T = getattr(model, "T", 1)
        self.ctx_limit = model.s_max - T - 1

    def _acquire(self):
        while not self.lock.acquire(timeout=10):          # queued behind another request: keep the client alive
            yield {"type": "ping"}

    def busy(self) -> bool:
        return self.sched.busy() if self.sched is not None else self.lock.locked()

    def generate(self, ids, max_tokens: int, sampling=None):
        if self.spec and self.sched is not None:
            yield from self._generate_spec(ids, max_tokens, sampling or {})
            return
        yield from self._acquire()
        try:
            if self.spec:
                yield from self._generate_spec(ids, max_tokens, sampling or {})
            else:
                yield from self._generate_plain(ids, max_tokens)
        finally:
            self.lock.release()

    def _generate_plain(self, ids, max_tokens: int):
        """step-2 TracedModel path (greedy, one token per step)."""
        m = self.m
        m.reset()
        if len(ids) + max_tokens > m.s_max:
            max_tokens = max(1, m.s_max - len(ids))
        yield {"type": "start", "prompt_tokens": len(ids), "cached_tokens": 0}
        t0 = time.perf_counter()
        for t in ids[:-1]:
            m.step(t)
        nxt = m.step(ids[-1], greedy=True)
        t1 = time.perf_counter()
        gen, sent, steps = [], "", 0
        finish = "length"
        while True:
            if nxt in EOS:
                finish = "stop"
                break
            gen.append(nxt)
            text = self.tok.decode(gen)
            if not text.endswith("\ufffd") and len(text) > len(sent):
                yield text[len(sent):]
                sent = text
            if len(gen) >= max_tokens:
                break
            nxt = m.step(nxt, greedy=True)
            steps += 1
        t2 = time.perf_counter()
        text = self.tok.decode(gen)
        if len(text) > len(sent):
            yield text[len(sent):]
        pm, dm = (t1 - t0) * 1e3, (t2 - t1) * 1e3
        yield {"type": "end", "finish_reason": finish, "stop_hit": None,
               "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen), "total_tokens": len(ids) + len(gen)},
               "timings": {"prompt_n": len(ids), "prompt_ms": round(pm, 1),
                           "prompt_per_second": round(len(ids) / (pm / 1e3), 3),
                           "predicted_n": steps, "predicted_ms": round(dm, 1),
                           "predicted_per_second": round(steps / (dm / 1e3), 3) if steps else None}}

    def _generate_spec(self, ids, max_tokens: int, sampling: dict):
        """SpecModel.generate in a worker thread, tokens streamed; sampling = {temperature, top_p, top_k,
        seed, stop}; temperature 0 = greedy. If the client goes away the worker is told to stop and joined before the
        model is released (never two generations on the card at once)."""
        stops = sampling.get("stop") or []
        stops = [stops] if isinstance(stops, str) else [x for x in stops if x]
        m = self.m
        if len(ids) + max_tokens + m.T > m.s_max:
            max_tokens = max(1, m.s_max - len(ids) - m.T)
        q: queue.Queue = queue.Queue()
        res = {}
        halt = threading.Event()

        def on_token(t):
            q.put(t)
            return halt.is_set()

        kw = dict(eos=EOS, spec=m.K > 0, temperature=float(sampling.get("temperature", 0.0) or 0.0),
                  top_p=float(sampling.get("top_p", 1.0) or 1.0), top_k=int(sampling.get("top_k", 0) or 0),
                  seed=sampling.get("seed"))
        sess = None

        def work():
            try:
                res["out"] = m.generate(ids, max_tokens, on_token=on_token, **kw)
            except Exception as e:          # surfaced to the client below
                res["err"] = e
            q.put(None)
        cached = m.cached_prefix(ids) if hasattr(m, "cached_prefix") else 0
        yield {"type": "start", "prompt_tokens": len(ids), "cached_tokens": cached}
        if self.sched is not None:          # the scheduler thread runs it, interleaved with other requests
            from .sessions import Session
            sess = Session(ids, max_tokens, kw, on_token)

            th = types.SimpleNamespace(join=sess.done.wait)     # stands in for the worker thread below
            threading.Thread(target=lambda: (sess.done.wait(), q.put(None)), daemon=True).start()
            self.sched.submit(sess)
        else:
            th = threading.Thread(target=work, daemon=True)
            th.start()
        try:
            gen, sent, hit, hit_s = [], "", None, None
            while True:
                try:
                    t = q.get(timeout=10)
                except queue.Empty:
                    yield {"type": "ping"}
                    continue
                if t is None:
                    break
                if t in EOS or hit is not None:
                    continue
                gen.append(t)
                text = self.tok.decode(gen)
                for st in stops:
                    j = text.find(st, max(0, len(sent) - len(st)))
                    if j >= 0 and (hit is None or j < hit):
                        hit, hit_s = j, st
                if hit is not None:
                    halt.set()
                    if hit > len(sent):
                        yield text[len(sent):hit]
                    sent = text[:hit]
                    continue
                hold = max((len(st) - 1 for st in stops), default=0)   # don't stream a possible stop prefix yet
                safe = len(text) - hold
                if not text.endswith("\ufffd") and safe > len(sent):
                    yield text[len(sent):safe]
                    sent = text[:safe]
        finally:
            halt.set()
            if sess is not None:
                sess.cancelled = True       # no-op when it already finished; else the scheduler drops it
            th.join()
        if sess is not None:
            if sess.error is not None:
                raise sess.error
            res = sess.result
        if "err" in res:
            raise res["err"]
        toks, info = res["out"]
        text = self.tok.decode(gen)
        if hit is None and len(text) > len(sent):
            yield text[len(sent):]
        finish = "stop" if (hit is not None or (toks and toks[-1] in EOS)) else "length"
        pm, dm, steps = info["prompt_s"] * 1e3, info["decode_s"] * 1e3, max(0, len(toks) - 1)
        acc = info["accepted"]
        cached = info.get("cached", 0)
        yield {"type": "end", "finish_reason": finish, "stop_hit": hit_s, "cached_tokens": cached,
               "usage": {"prompt_tokens": len(ids), "completion_tokens": len(gen), "total_tokens": len(ids) + len(gen),
                         "prompt_tokens_details": {"cached_tokens": cached}},
               "timings": {"prompt_n": len(ids) - cached, "cache_n": cached, "prompt_ms": round(pm, 1),
                           "prompt_per_second": round((len(ids) - cached) / (pm / 1e3), 3),
                           "predicted_n": steps, "predicted_ms": round(dm, 1),
                           "predicted_per_second": round(steps / (dm / 1e3), 3) if steps else None,
                           "draft_n": sum(info["proposed"]), "draft_n_accepted": sum(acc)}}


def idle_watch(engine, srv, idle_min: float):
    """--idle-exit: stop the server (and free the card) after this many idle minutes"""
    last = time.time()
    while True:
        time.sleep(5)
        if engine.busy():
            last = time.time()
            continue
        if time.time() - last > idle_min * 60:
            print(f"idle for {idle_min:g} min: exiting", flush=True)
            srv.shutdown()
            return


def writable_cwd():
    """tt-metal writes run files (generated/) under the current directory: move to a writable one if needed"""
    import tempfile
    if os.access(os.getcwd(), os.W_OK):
        return
    for d in (os.environ.get("TT_METAL_CACHE"), os.environ.get("HOME"), tempfile.gettempdir()):
        if d and os.path.isdir(d) and os.access(d, os.W_OK):
            print(f"working directory {os.getcwd()} is read-only: using {d}", flush=True)
            os.chdir(d)
            return


def prepare(gb: float, source: str, spec: int):
    """First start: the card caches, built on this machine from the pinned downloads (markers make later starts skip)."""
    from .sources import card_cache
    cache = card_cache()
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if not (cache / ".dense-done").exists():
        print("prepare: dense weights -> card format (first start only)", flush=True)
        from .weights import prepare as dense
        dense()
        (cache / ".dense-done").write_text(time.strftime("%F %T"))
    if gb:
        from .experts import prepare as experts, tag
        d = cache / (tag(gb) + ("" if source == "gguf" else f"-{source}"))
        if not (d / ".done").exists():
            print(f"prepare: {gb:g} GB resident expert set from {source} (first start only)", flush=True)
            experts(gb, "bfp4", int(os.environ.get("Q38_PREP_THREADS", "12")), source)
            (d / ".done").write_text(time.strftime("%F %T"))
    if spec and not (cache / ".mtp-done").exists():
        print("prepare: MTP draft layer (first start only)", flush=True)
        from .spec import prepare_mtp
        prepare_mtp()
        (cache / ".mtp-done").write_text(time.strftime("%F %T"))
    if time.time() - t0 > 5:
        print(f"prepare: done in {time.time() - t0:.0f} s -> {cache}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", default="coder", choices=list(PROFILES))
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 inside a container)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--resident-gb", type=float, default=None, help="experts resident on the card (profile default)")
    ap.add_argument("--source", default=None, choices=["gguf", "lotto"], help="resident experts from (profile default)")
    ap.add_argument("--spec", type=int, default=3, help="MTP drafts per round (0 = plain decode)")
    ap.add_argument("--min-p", type=float, default=0.0, help="propose MTP drafts only while their prob >= this")
    ap.add_argument("--prefix-slots", type=int, default=None, help="prefix cache slots on the card (profile default)")
    ap.add_argument("--max-concurrent", type=int, default=None, help="time-sliced requests at once (profile default)")
    ap.add_argument("--quantum", type=float, default=0.5, help="seconds a request runs before the next one's turn")
    ap.add_argument("--model-name", default=None, help="id reported by /v1/models (default qwen3.8-flash-next-p100a-<profile>)")
    ap.add_argument("--thinking", default="request", choices=["request", "off"],
                    help="off: never think, whatever the request asks (agents: much faster turns)")
    ap.add_argument("--idle-exit", type=float, default=0.0, help="exit after this many idle minutes (0 = never)")
    ap.add_argument("--prepare-only", action="store_true", help="download + build the caches, then exit (no card)")
    ap.add_argument("--log-requests", default="", help="write every request (headers + body) to this directory")
    a = ap.parse_args(argv)
    model, gb, source, slots, conc = PROFILES[a.profile]
    os.environ["P100A_MODEL"] = model          # read at import by sources / ref.gguf_weights: set before importing them
    # CPU experts: 8 OpenMP threads pinned to cores spread over the CCDs (+12-20 % decode at 8 GB); set before
    # libgomp loads (the host-expert library is loaded later)
    for k, v in (("OMP_NUM_THREADS", "8"), ("OMP_PROC_BIND", "spread"), ("OMP_PLACES", "cores"), ("MKL_NUM_THREADS", "8")):
        os.environ.setdefault(k, v)
    gb = gb if a.resident_gb is None else a.resident_gb
    source = a.source or source
    slots = slots if a.prefix_slots is None else a.prefix_slots
    conc = conc if a.max_concurrent is None else a.max_concurrent
    name = a.model_name or f"qwen3.8-flash-next-p100a-{a.profile}"
    if os.environ.get("P100A_WALL") is None:   # an interactive server runs until stopped
        import faulthandler
        faulthandler.cancel_dump_traceback_later()
    import signal

    def _term(*_):                             # docker stop / tt-model stop: close the device cleanly
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    writable_cwd()
    if a.prepare_only and not os.path.exists("/dev/tenstorrent") and not os.environ.get("TT_METAL_MOCK_CLUSTER_DESC_PATH"):
        # no card: ttnn's host-side BFP packing still wants a cluster description -> tt-metal's P100 descriptor
        rel = "tt_metal/third_party/tt-cluster-descriptors/blackhole/p100_cluster_desc/p100_cluster_desc.yaml"
        for root in (os.environ.get("TT_METAL_HOME", ""), "/tt-metal", "/opt/tt-metal"):
            if root and os.path.exists(os.path.join(root, rel)):
                os.environ["TT_METAL_MOCK_CLUSTER_DESC_PATH"] = os.path.join(root, rel)
                break
    prepare(gb, source, a.spec)
    if a.prepare_only:
        return 0
    import ttnn

    from .api import make_handler
    dev = ttnn.open_device(device_id=0, l1_small_size=24576, trace_region_size=600 * 1024 * 1024)
    try:
        t0 = time.time()
        ex = None
        if gb:
            from .experts import DeviceExperts
            ex = DeviceExperts(dev, gb, "bfp4", source)
        from .sessions import Scheduler
        from .spec import SpecModel
        model = SpecModel(dev, experts=ex, drafts=a.spec, min_p=a.min_p, prefix_slots=slots)
        sched = Scheduler(model, conc, a.quantum)   # its state slots are allocated before the capture
        model.capture_all()
        print(f"model ready in {time.time() - t0:.0f} s", flush=True)
        engine = Engine(model, spec=True, sched=sched, name=name)
        engine.think_mode = a.thinking
        srv = ThreadingHTTPServer((a.host, a.port), make_handler(engine, a.log_requests))
        if a.idle_exit:
            threading.Thread(target=idle_watch, args=(engine, srv, a.idle_exit), daemon=True).start()
        print(json.dumps({"event": "ready", "model": name, "profile": a.profile, "host": a.host, "port": a.port}), flush=True)
        print(f"serving on http://{a.host}:{a.port}/v1", flush=True)
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ttnn.close_device(dev)
    return 0


if __name__ == "__main__":
    sys.exit(main())
