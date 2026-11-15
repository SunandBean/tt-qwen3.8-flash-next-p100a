"""the HTTP layer of the P100a server (no ttnn here, so it also runs on the CPU with a scripted FakeEngine).

  POST /v1/chat/completions       OpenAI chat (stream or not; tools / tool_choice / tool_calls; reasoning_content)
  POST /v1/messages               Anthropic Messages (stream = Anthropic SSE events; system, multi-turn, tool_use /
                                  tool_result, thinking blocks, stop_sequences, stop_reason, usage)
  POST /v1/messages/count_tokens  Anthropic token count
  GET  /v1/models, /health

Engine interface (tt_qwen38.server.Engine, FakeEngine below):
  tok (encode/decode), template (tt_qwen38.chat.Template), ctx_limit (max prompt tokens), name, lock,
  generate(ids, max_tokens, sampling) yields {"type":"start", prompt_tokens, cached_tokens}, {"type":"ping"},
  text pieces (str), and finally {"type":"end", finish_reason, stop_hit, usage, timings}.
"""
from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler

from .chat import OutParser, TemplateError, from_anthropic, from_openai, from_responses, new_id


class ApiError(Exception):
    def __init__(self, code, kind, msg):
        super().__init__(msg)
        self.code, self.kind, self.msg = code, kind, msg


def _prep(engine, msgs, tools, prefill, thinking, effort=None):
    try:
        prompt = engine.template.render(msgs, tools, enable_thinking=thinking, reasoning_effort=effort)
    except TemplateError as e:
        raise ApiError(400, "invalid_request_error", f"chat template: {e}")
    if thinking:
        prefill = ""                                    # forced tool use is not combined with thinking (as Anthropic)
    ids = engine.tok.encode(prompt + prefill, parse_special=True)
    if len(ids) > engine.ctx_limit:
        raise ApiError(400, "invalid_request_error",
                       f"prompt is too long: {len(ids)} tokens > {engine.ctx_limit} maximum")
    return ids, prefill


def _sampling(req, anthropic=False):
    stop = req.get("stop_sequences") if anthropic else req.get("stop")
    return {"temperature": req.get("temperature", 1.0), "top_p": req.get("top_p", 1.0),
            "top_k": req.get("top_k", 0), "seed": req.get("seed"), "stop": stop}


def _anthropic_thinking(req) -> bool:
    th = req.get("thinking") or {}
    return th.get("type") in ("enabled", "adaptive")


def _think(engine, want: bool) -> bool:
    """server --thinking off overrides the request (agents are much faster without the reasoning pass)"""
    return want and getattr(engine, "think_mode", "request") != "off"


def _openai_thinking(req) -> bool:
    think = (req.get("chat_template_kwargs") or {}).get("enable_thinking", True)
    if req.get("reasoning_effort") == "none":
        think = False
    return bool(think)


def make_handler(engine, log_dir: str = ""):
    counter = {"n": 0}
    clock = threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def _err(self, anthropic, code, kind, msg):
            if anthropic:
                return self._json(code, {"type": "error", "error": {"type": kind, "message": msg}})
            return self._json(code, {"error": {"message": msg, "type": kind, "code": code}})

        def _models(self):
            m = {"id": engine.name, "object": "model", "type": "model", "display_name": engine.name,
                 "created_at": "2026-11-12T00:00:00Z", "owned_by": "p100a"}
            return {"object": "list", "data": [m], "has_more": False, "first_id": engine.name, "last_id": engine.name}

        def do_HEAD(self):
            self.send_response(200)
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0].rstrip("/")
            if path in ("/health", "/v1/models", "/models", ""):
                self._json(200, self._models())
            elif path.startswith("/v1/models/"):
                self._json(200, self._models()["data"][0])
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            path = self.path.split("?")[0].rstrip("/")
            anthropic = path.startswith("/v1/messages")
            try:
                raw = self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
                req = json.loads(raw)
            except ValueError:
                return self._err(anthropic, 400, "invalid_request_error", "body is not JSON")
            if log_dir:
                with clock:
                    counter["n"] += 1
                    n = counter["n"]
                os.makedirs(log_dir, exist_ok=True)
                json.dump({"path": self.path, "headers": dict(self.headers), "body": req},
                          open(os.path.join(log_dir, f"{time.strftime('%H%M%S')}-{n:04d}.json"), "w"),
                          ensure_ascii=False, indent=1)
            try:
                if path == "/v1/chat/completions" or path == "/chat/completions":
                    return self._openai(req)
                if path == "/v1/messages":
                    return self._anthropic(req)
                if path in ("/v1/responses", "/responses"):
                    return self._responses(req)
                if path == "/v1/messages/count_tokens":
                    msgs, tools, pre = from_anthropic(req)
                    ids, _ = _prep(engine, msgs, tools, pre, _think(engine, _anthropic_thinking(req)))
                    return self._json(200, {"input_tokens": len(ids)})
                return self._err(anthropic, 404, "not_found_error", f"no route {path}")
            except ApiError as e:
                return self._err(anthropic, e.code, e.kind, e.msg)
            except (BrokenPipeError, ConnectionResetError):
                return None

        # ---------------------------------------------------------------- OpenAI
        def _openai(self, req):
            msgs, tools, pre = from_openai(req)
            think = _think(engine, _openai_thinking(req))
            ids, pre = _prep(engine, msgs, tools, pre, think, None)
            mt = int(req.get("max_completion_tokens") or req.get("max_tokens") or 512)
            gen = engine.generate(ids, mt, _sampling(req))
            par = OutParser(think, tools, pre)
            cid, created, model = f"chatcmpl-{int(time.time() * 1000)}", int(time.time()), req.get("model") or engine.name
            ncalls = [0]

            def tc_obj(c):
                i = ncalls[0]
                ncalls[0] += 1
                return {"index": i, "id": new_id("call_"), "type": "function",
                        "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
            if not req.get("stream"):
                text, reas, calls, tail = [], [], [], None
                for p in gen:
                    evs = par.feed(p) if isinstance(p, str) else (par.finish() if p.get("type") == "end" else [])
                    if isinstance(p, dict) and p.get("type") == "end":
                        tail = p
                    for k, v in evs:
                        (text.append(v) if k == "text" else reas.append(v) if k == "think" else calls.append(tc_obj(v)))
                msg = {"role": "assistant", "content": "".join(text) if (text or not calls) else None}
                if reas:
                    msg["reasoning_content"] = "".join(reas)
                if calls:
                    for c in calls:
                        c.pop("index")
                    msg["tool_calls"] = calls
                fin = "tool_calls" if calls else tail["finish_reason"]
                return self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": model,
                                        "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
                                        "usage": tail["usage"], "timings": tail["timings"]})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def send(delta, fin=None, extra=None):
                o = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model,
                     "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}
                o.update(extra or {})
                self.wfile.write(b"data: " + json.dumps(o, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
            send({"role": "assistant", "content": ""})
            try:
                for p in gen:
                    if isinstance(p, dict) and p.get("type") == "ping":
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        continue
                    end = isinstance(p, dict) and p.get("type") == "end"
                    if isinstance(p, dict) and not end:
                        continue
                    for k, v in (par.finish() if end else par.feed(p)):
                        if k == "text":
                            send({"content": v})
                        elif k == "think":
                            send({"reasoning_content": v})
                        else:
                            send({"tool_calls": [tc_obj(v)]})
                    if end:
                        send({}, "tool_calls" if ncalls[0] else p["finish_reason"],
                             {"usage": p["usage"], "timings": p["timings"]})
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            finally:
                gen.close()

        # ---------------------------------------------------------------- Anthropic
        def _anthropic(self, req):
            msgs, tools, pre = from_anthropic(req)
            think = _think(engine, _anthropic_thinking(req))
            ids, pre = _prep(engine, msgs, tools, pre, think, None)
            mt = int(req.get("max_tokens") or 4096)
            gen = engine.generate(ids, mt, _sampling(req, anthropic=True))
            par = OutParser(think, tools, pre)
            mid, model = new_id("msg_"), req.get("model") or engine.name

            def stop_reason(tail, ncall):
                if ncall:
                    return "tool_use", None
                if tail.get("stop_hit") is not None:
                    return "stop_sequence", tail["stop_hit"]
                return ("max_tokens" if tail["finish_reason"] == "length" else "end_turn"), None

            def usage(tail_or_start, out_tokens):
                cached = tail_or_start.get("cached_tokens", 0) or 0
                n = tail_or_start.get("prompt_tokens", 0)
                return {"input_tokens": n - cached, "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": cached, "output_tokens": out_tokens}

            if not req.get("stream"):
                blocks, start, tail = [], {}, None
                for p in gen:
                    if isinstance(p, dict):
                        if p.get("type") == "start":
                            start = p
                        if p.get("type") != "end":
                            continue
                        tail = p
                    for k, v in (par.finish() if isinstance(p, dict) else par.feed(p)):
                        if k == "tool":
                            blocks.append({"type": "tool_use", "id": new_id("toolu_"), "name": v["name"],
                                           "input": v["arguments"]})
                        elif blocks and blocks[-1]["type"] == ("text" if k == "text" else "thinking"):
                            blocks[-1]["text" if k == "text" else "thinking"] += v
                        else:
                            blocks.append({"type": "text", "text": v} if k == "text" else
                                          {"type": "thinking", "thinking": v, "signature": ""})
                ncall = sum(b["type"] == "tool_use" for b in blocks)
                sr, ss = stop_reason(tail, ncall)
                st = dict(start, cached_tokens=start.get("cached_tokens", 0))
                return self._json(200, {"id": mid, "type": "message", "role": "assistant", "model": model,
                                        "content": blocks, "stop_reason": sr, "stop_sequence": ss,
                                        "usage": usage(st, tail["usage"]["completion_tokens"])})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def ev(name, obj):
                self.wfile.write(f"event: {name}\ndata: ".encode() + json.dumps(obj, ensure_ascii=False).encode()
                                 + b"\n\n")
                self.wfile.flush()
            cur = {"kind": None, "index": -1}
            ncall = [0]

            def close():
                if cur["kind"] is None:
                    return
                if cur["kind"] == "thinking":
                    ev("content_block_delta", {"type": "content_block_delta", "index": cur["index"],
                                               "delta": {"type": "signature_delta", "signature": ""}})
                ev("content_block_stop", {"type": "content_block_stop", "index": cur["index"]})
                cur["kind"] = None

            def open_(block):
                close()
                cur["index"] += 1
                cur["kind"] = block["type"]
                ev("content_block_start", {"type": "content_block_start", "index": cur["index"], "content_block": block})
            try:
                for p in gen:
                    if isinstance(p, dict):
                        t = p.get("type")
                        if t == "start":
                            ev("message_start", {"type": "message_start", "message": {
                                "id": mid, "type": "message", "role": "assistant", "model": model, "content": [],
                                "stop_reason": None, "stop_sequence": None, "usage": usage(p, 1)}})
                            ev("ping", {"type": "ping"})
                            continue
                        if t == "ping":
                            ev("ping", {"type": "ping"})
                            continue
                    evs = par.finish() if isinstance(p, dict) else par.feed(p)
                    for k, v in evs:
                        if k == "tool":
                            open_({"type": "tool_use", "id": new_id("toolu_"), "name": v["name"], "input": {}})
                            ev("content_block_delta", {"type": "content_block_delta", "index": cur["index"], "delta": {
                                "type": "input_json_delta",
                                "partial_json": json.dumps(v["arguments"], ensure_ascii=False)}})
                            close()
                            ncall[0] += 1
                            continue
                        kind = "text" if k == "text" else "thinking"
                        if cur["kind"] != kind:
                            open_({"type": "text", "text": ""} if kind == "text" else
                                  {"type": "thinking", "thinking": "", "signature": ""})
                        ev("content_block_delta", {"type": "content_block_delta", "index": cur["index"], "delta":
                                                   {"type": "text_delta", "text": v} if kind == "text" else
                                                   {"type": "thinking_delta", "thinking": v}})
                    if isinstance(p, dict):                            # end
                        close()
                        sr, ss = stop_reason(p, ncall[0])
                        ev("message_delta", {"type": "message_delta", "delta": {"stop_reason": sr, "stop_sequence": ss},
                                             "usage": {"output_tokens": p["usage"]["completion_tokens"]}})
                        ev("message_stop", {"type": "message_stop"})
            except Exception as e:                                    # headers are out: report in-stream
                if not isinstance(e, (BrokenPipeError, ConnectionResetError)):
                    try:
                        ev("error", {"type": "error", "error": {"type": "api_error", "message": repr(e)}})
                    except OSError:
                        pass
                raise
            finally:
                gen.close()
        # ---------------------------------------------------------------- OpenAI Responses (Codex CLI)
        def _responses(self, req):
            msgs, tools, pre, think, effort = from_responses(req)
            think = _think(engine, think)
            ids, pre = _prep(engine, msgs, tools, pre, think, effort if think else None)
            mt = int(req.get("max_output_tokens") or 8192)
            gen = engine.generate(ids, mt, _sampling(req))
            par = OutParser(think, tools, pre)
            rid, model, created = new_id("resp_"), req.get("model") or engine.name, int(time.time())
            stream = bool(req.get("stream"))
            out_items, seq = [], [0]
            cur = {"item": None}

            def resp(status, usage=None):
                r = {"id": rid, "object": "response", "created_at": created, "status": status, "model": model,
                     "output": out_items if status == "completed" else [], "parallel_tool_calls": True,
                     "tool_choice": req.get("tool_choice", "auto"), "tools": req.get("tools") or []}
                if usage:
                    r["usage"] = usage
                return r

            def ev(name, obj):
                if not stream:
                    return
                obj = dict(obj, type=name, sequence_number=seq[0])
                seq[0] += 1
                self.wfile.write(f"event: {name}\ndata: ".encode() + json.dumps(obj, ensure_ascii=False).encode()
                                 + b"\n\n")
                self.wfile.flush()

            def close():
                it = cur["item"]
                if it is None:
                    return
                idx = len(out_items)
                if it["type"] == "message":
                    txt = it["content"][0]["text"]
                    ev("response.output_text.done", {"item_id": it["id"], "output_index": idx, "content_index": 0,
                                                     "text": txt})
                    ev("response.content_part.done", {"item_id": it["id"], "output_index": idx, "content_index": 0,
                                                      "part": it["content"][0]})
                it["status"] = "completed"
                ev("response.output_item.done", {"output_index": idx, "item": it})
                out_items.append(it)
                cur["item"] = None

            def open_(it):
                close()
                cur["item"] = it
                idx = len(out_items)
                shown = dict(it, status="in_progress")
                if it["type"] == "message":
                    shown["content"] = []
                ev("response.output_item.added", {"output_index": idx, "item": shown})
                if it["type"] == "message":
                    ev("response.content_part.added", {"item_id": it["id"], "output_index": idx, "content_index": 0,
                                                       "part": {"type": "output_text", "text": "", "annotations": []}})

            def handle(k, v):
                idx = len(out_items)
                if k == "tool":
                    it = {"type": "function_call", "id": new_id("fc_"), "call_id": new_id("call_"), "name": v["name"],
                          "arguments": json.dumps(v["arguments"], ensure_ascii=False)}
                    open_(it)
                    ev("response.function_call_arguments.delta", {"item_id": it["id"], "output_index": idx,
                                                                  "delta": it["arguments"]})
                    ev("response.function_call_arguments.done", {"item_id": it["id"], "output_index": idx,
                                                                 "arguments": it["arguments"]})
                    close()
                elif k == "think":
                    it = cur["item"]
                    if it is None or it["type"] != "reasoning":
                        it = {"type": "reasoning", "id": new_id("rs_"), "summary": [],
                              "content": [{"type": "reasoning_text", "text": ""}], "encrypted_content": None}
                        open_(it)
                    it["content"][0]["text"] += v
                    ev("response.reasoning_text.delta", {"item_id": it["id"], "output_index": idx, "content_index": 0,
                                                         "delta": v})
                else:
                    it = cur["item"]
                    if it is None or it["type"] != "message":
                        it = {"type": "message", "id": new_id("msg_"), "role": "assistant",
                              "content": [{"type": "output_text", "text": "", "annotations": []}]}
                        open_(it)
                    it["content"][0]["text"] += v
                    ev("response.output_text.delta", {"item_id": it["id"], "output_index": idx, "content_index": 0,
                                                      "delta": v})
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
            try:
                for p in gen:
                    if isinstance(p, dict):
                        t = p.get("type")
                        if t == "start":
                            ev("response.created", {"response": resp("in_progress")})
                            ev("response.in_progress", {"response": resp("in_progress")})
                            continue
                        if t == "ping":
                            if stream:
                                self.wfile.write(b": ping\n\n")
                                self.wfile.flush()
                            continue
                    for k, v in (par.finish() if isinstance(p, dict) else par.feed(p)):
                        handle(k, v)
                    if isinstance(p, dict):
                        close()
                        u = p["usage"]
                        usage = {"input_tokens": u["prompt_tokens"],
                                 "input_tokens_details": {"cached_tokens": p.get("cached_tokens", 0) or 0},
                                 "output_tokens": u["completion_tokens"],
                                 "output_tokens_details": {"reasoning_tokens": 0},
                                 "total_tokens": u["total_tokens"]}
                        final = resp("completed", usage)
                        if p["finish_reason"] == "length" and not any(i["type"] == "function_call" for i in out_items):
                            final["status"] = "incomplete"
                            final["incomplete_details"] = {"reason": "max_output_tokens"}
                        if stream:
                            ev("response.completed" if final["status"] == "completed" else "response.incomplete",
                               {"response": final})
                        else:
                            return self._json(200, final)
            finally:
                gen.close()

    return H


class FakeEngine:
    """CPU stand-in for the card: replies with scripted texts in turn (or echoes the last prompt line)."""

    def __init__(self, tok, template, replies=(), name="p100a-fake", ctx_limit=32000, delay=0.0):
        self.tok, self.template, self.name, self.ctx_limit = tok, template, name, ctx_limit
        self.replies, self.i, self.delay = list(replies), 0, delay
        self.lock = threading.Lock()
        self.prompts = []

    def generate(self, ids, max_tokens, sampling):
        with self.lock:
            self.prompts.append(ids)
            yield {"type": "start", "prompt_tokens": len(ids), "cached_tokens": 0}
            if self.replies:
                text = self.replies[self.i % len(self.replies)]
                self.i += 1
            else:
                text = "echo: " + self.tok.decode(ids[-40:])
            out = self.tok.encode(text, parse_special=True)[:max_tokens]
            stops = sampling.get("stop") or []
            stops = [stops] if isinstance(stops, str) else stops
            sent, hit = "", None
            for k in range(1, len(out) + 1):
                s = self.tok.decode(out[:k])
                for st in stops:
                    if st and st in s:
                        hit = st
                        s = s[:s.find(st)]
                if len(s) > len(sent):
                    yield s[len(sent):]
                    sent = s
                if hit:
                    break
                if self.delay:
                    time.sleep(self.delay)
            fin = "stop" if hit or len(out) < max_tokens else "length"
            yield {"type": "end", "finish_reason": fin, "stop_hit": hit,
                   "usage": {"prompt_tokens": len(ids), "completion_tokens": len(out),
                             "total_tokens": len(ids) + len(out)}, "timings": {}}


def main():
    """CPU test server:  .venv/bin/python -m tt_qwen38.api --port 8091 [--replies FILE.json] [--log DIR]"""
    import argparse
    from http.server import ThreadingHTTPServer

    from tt_qwen38.ref.gguf_weights import Weights
    from tt_qwen38.ref.tokenizer import tokenizer

    from .chat import Template
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--replies", default="", help="JSON list of reply texts (model format, e.g. with <tool_call>)")
    ap.add_argument("--log", default="", help="write every request (headers + body) to this directory")
    a = ap.parse_args()
    W = Weights()
    eng = FakeEngine(tokenizer(W), Template(W.meta["tokenizer.chat_template"]),
                     json.load(open(a.replies)) if a.replies else ())
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(eng, a.log))
    print(f"fake server on 127.0.0.1:{a.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
