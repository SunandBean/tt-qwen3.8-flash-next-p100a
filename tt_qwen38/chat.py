"""chat formatting and tool calls for the OpenAI and Anthropic endpoints (CPU only; no ttnn, no card).

render():       the GGUF's own chat template (Qwen3.8 + Unsloth fixes, read from tokenizer.chat_template) through jinja2,
                as llama.cpp --jinja / HF apply_chat_template run it. For user/system/assistant text with thinking off it
                gives exactly the step-1..5 prompt (checked in tt_qwen38/test_chat.py).
from_openai():  OpenAI chat messages (+ tools, tool_choice) -> template messages, tools, assistant prefill.
from_anthropic(): Anthropic Messages request -> the same.
OutParser:      incremental parser of the model's output: <think>..</think> reasoning, plain text, and the model's tool calls
                  <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n</tool_call>
                (Qwen3-Coder XML format; parameter values converted with the tool's JSON schema like vLLM's
                qwen3_coder parser). Text that might be the start of a tag is held back until it is decided.
"""
from __future__ import annotations

import json
import re
import secrets

from jinja2.sandbox import ImmutableSandboxedEnvironment


# ------------------------------------------------------------------ template
class TemplateError(ValueError):
    pass


def _raise(msg):
    raise TemplateError(msg)


def _tojson(x, indent=None, separators=None, sort_keys=False, ensure_ascii=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class Template:
    def __init__(self, source: str):
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        self.t = env.from_string(source)

    def render(self, messages, tools=None, enable_thinking: bool = False, reasoning_effort=None,
               add_generation_prompt: bool = True) -> str:
        kw = {"messages": messages, "add_generation_prompt": add_generation_prompt, "enable_thinking": enable_thinking}
        if tools:
            kw["tools"] = tools
        if reasoning_effort:
            kw["reasoning_effort"] = reasoning_effort
        return self.t.render(**kw)


# ------------------------------------------------------------------ request mapping
def _text_of(content) -> str:
    """OpenAI / Anthropic content (string or parts) -> plain text; images and documents become a short placeholder
    (the model is text-only here)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    out = []
    for p in content:
        if not isinstance(p, dict):
            continue
        t = p.get("type")
        if t in (None, "text", "input_text", "output_text"):
            out.append(p.get("text", ""))
        elif t in ("image", "image_url", "input_image"):
            out.append("[image omitted: this model reads text only]")
        elif t == "document":
            src = p.get("source") or {}
            out.append(src.get("data", "") if src.get("type") == "text" else "[document omitted]")
        elif t == "tool_result":                      # nested (rare) -> its text
            out.append(_text_of(p.get("content")))
    return "\n".join(x for x in out if x)


def _tool_fn(t: dict):
    """OpenAI {"type":"function","function":{...}} or Anthropic {"name","description","input_schema"} -> OpenAI form
    (what the template prints). Anthropic server tools (no input_schema, e.g. web_search) are dropped."""
    if t.get("type") == "function" and "function" in t:
        f = dict(t["function"])
        f.setdefault("parameters", {"type": "object", "properties": {}})
        return {"type": "function", "function": f}
    if "input_schema" in t:
        f = {"name": t["name"], "description": t.get("description", ""), "parameters": t["input_schema"]}
        return {"type": "function", "function": f}
    return None


def _prefill(choice_name: str | None, any_tool: bool) -> str:
    """Forced tool use: start the answer with the call header (the model then fills in the arguments)."""
    if choice_name:
        return f"<tool_call>\n<function={choice_name}>\n"
    if any_tool:
        return "<tool_call>\n<function="
    return ""


def _args_obj(a):
    if isinstance(a, dict):
        return a
    if isinstance(a, str):
        if not a.strip():
            return {}
        try:
            v = json.loads(a)
            return v if isinstance(v, dict) else {"value": v}
        except ValueError:
            return {"arguments": a}
    return {}


def _fix_system_order(msgs: list) -> list:
    """The template allows system messages only at the start: later ones (Claude Code sends one with the environment
    after the first user turn) become user messages with the same text."""
    out, lead = [], True
    for m in msgs:
        if m["role"] in ("system", "developer"):
            if lead:
                out.append(m)
                continue
            m = {"role": "user", "content": _text_of(m.get("content"))}   # same text, whatever the content form
        lead = False
        out.append(m)
    return out


def from_openai(req: dict):
    """-> (messages, tools, prefill)"""
    msgs = []
    for m in req.get("messages") or []:
        r = m.get("role")
        if r == "assistant":
            a = {"role": "assistant", "content": _text_of(m.get("content"))}
            rc = m.get("reasoning_content") or m.get("reasoning")
            if isinstance(rc, str):
                a["reasoning_content"] = rc
            tcs = m.get("tool_calls") or []
            if tcs:
                a["tool_calls"] = [{"name": (tc.get("function") or {}).get("name") or tc.get("name"),
                                    "arguments": _args_obj((tc.get("function") or {}).get("arguments", tc.get("arguments")))}
                                   for tc in tcs]
            msgs.append(a)
        elif r == "tool" or r == "function":
            msgs.append({"role": "tool", "content": _text_of(m.get("content"))})
        else:
            msgs.append({"role": r or "user", "content": _text_of(m.get("content"))})
    tools = [x for x in (_tool_fn(t) for t in (req.get("tools") or [])) if x]
    tc = req.get("tool_choice")
    name, anyt = None, False
    if tc == "none":
        tools = []
    elif tc == "required":
        anyt = True
    elif isinstance(tc, dict):
        name = (tc.get("function") or {}).get("name") or tc.get("name")
    return _fix_system_order(msgs), tools, _prefill(name, anyt) if tools else ""


def from_anthropic(req: dict):
    """-> (messages, tools, prefill)"""
    msgs = []
    sysm = req.get("system")
    if sysm:
        st = sysm if isinstance(sysm, str) else "\n\n".join(b.get("text", "") for b in sysm if isinstance(b, dict))
        if st.strip():
            msgs.append({"role": "system", "content": st})
    for m in req.get("messages") or []:
        r, c = m.get("role"), m.get("content")
        if isinstance(c, str):
            msgs.append({"role": r, "content": c})
            continue
        if r == "assistant":
            text, think, calls = [], [], []
            for b in c or []:
                t = b.get("type")
                if t == "text":
                    text.append(b.get("text", ""))
                elif t == "thinking":
                    think.append(b.get("thinking", ""))
                elif t in ("tool_use", "server_tool_use"):
                    calls.append({"name": b.get("name"), "arguments": _args_obj(b.get("input"))})
            a = {"role": "assistant", "content": "".join(text)}
            if think:
                a["reasoning_content"] = "\n".join(think)
            if calls:
                a["tool_calls"] = calls
            msgs.append(a)
            continue
        pend = []                                       # user blocks: tool_result -> tool messages, rest -> user text

        def flush():
            if pend:
                msgs.append({"role": "user", "content": "\n".join(pend)})
                pend.clear()
        for b in c or []:
            if b.get("type") == "tool_result":
                flush()
                body = _text_of(b.get("content"))
                if b.get("is_error"):
                    body = "Error: " + body
                msgs.append({"role": "tool", "content": body})
            else:
                s = _text_of([b])
                if s:
                    pend.append(s)
        flush()
    tools = [x for x in (_tool_fn(t) for t in (req.get("tools") or [])) if x]
    tc = req.get("tool_choice") or {}
    name, anyt = None, False
    if tc.get("type") == "none":
        tools = []
    elif tc.get("type") == "any":
        anyt = True
    elif tc.get("type") == "tool":
        name = tc.get("name")
    return _fix_system_order(msgs), tools, _prefill(name, anyt) if tools else ""


def from_responses(req: dict):
    """OpenAI Responses API request (Codex CLI) -> (messages, tools, prefill, thinking, effort).
    instructions + developer/system messages -> system; function_call / custom_tool_call items -> assistant tool calls
    (consecutive ones merged into one assistant turn); *_output items -> tool messages; reasoning items -> the next
    assistant turn's reasoning_content. Namespace and hosted tools (web_search, ...) are not offered to the model."""
    msgs = []
    if req.get("instructions"):
        msgs.append({"role": "system", "content": req["instructions"]})
    items = req.get("input") or []
    if isinstance(items, str):
        items = [{"type": "message", "role": "user", "content": items}]
    reasoning = []

    def asst():
        if not (msgs and msgs[-1]["role"] == "assistant" and msgs[-1].get("_open")):
            msgs.append({"role": "assistant", "content": "", "_open": True})
            if reasoning:
                msgs[-1]["reasoning_content"] = "\n".join(reasoning)
                reasoning.clear()
        return msgs[-1]
    for it in items:
        t = it.get("type", "message")
        if t == "message":
            r = it.get("role", "user")
            if r == "assistant":
                a = asst()
                a["content"] += _text_of(it.get("content"))
                continue
            msgs.append({"role": "system" if r in ("developer", "system") else "user", "content": _text_of(it.get("content"))})
        elif t == "reasoning":
            txt = _text_of(it.get("content")) or "\n".join(x.get("text", "") for x in it.get("summary") or [])
            if txt:
                reasoning.append(txt)
        elif t in ("function_call", "custom_tool_call"):
            args = _args_obj(it.get("arguments")) if t == "function_call" else {"input": it.get("input", "")}
            asst().setdefault("tool_calls", []).append({"name": it.get("name"), "arguments": args})
        elif t in ("function_call_output", "custom_tool_call_output"):
            out = it.get("output")
            msgs.append({"role": "tool", "content": out if isinstance(out, str) else _text_of(out)})
        elif t == "local_shell_call":
            asst().setdefault("tool_calls", []).append({"name": "shell", "arguments": it.get("action") or {}})
        elif t == "local_shell_call_output":
            msgs.append({"role": "tool", "content": str(it.get("output", ""))})
        if t != "message" or it.get("role") != "assistant":
            if msgs and msgs[-1]["role"] != "assistant":
                for m in msgs:
                    m.pop("_open", None)
    for m in msgs:
        m.pop("_open", None)
    tools = []
    for x in req.get("tools") or []:
        if x.get("type") == "function":
            tools.append({"type": "function", "function": {"name": x["name"], "description": x.get("description", ""),
                                                           "parameters": x.get("parameters") or {"type": "object", "properties": {}}}})
    tc = req.get("tool_choice")
    name, anyt = None, False
    if tc == "none":
        tools = []
    elif tc == "required":
        anyt = True
    elif isinstance(tc, dict) and tc.get("name"):
        name = tc["name"]
    eff = (req.get("reasoning") or {}).get("effort")
    thinking = eff not in ("none", "minimal")
    effort = {"low": "low", "medium": "medium"}.get(eff)
    return _fix_system_order(msgs), tools, (_prefill(name, anyt) if tools else ""), thinking, effort


# ------------------------------------------------------------------ output parsing
_TC_OPEN, _TC_CLOSE, _FN_OPEN = "<tool_call>", "</tool_call>", "<function="
_TH_CLOSE = "</think>"


def _conv(v: str, schema: dict | None):
    t = (schema or {}).get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t == "string":
        return v
    if v.strip().lower() == "null" and t != "string":
        return None
    if t == "integer":
        try:
            return int(v.strip())
        except ValueError:
            pass
    if t == "number":
        try:
            f = float(v.strip())
            return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
        except ValueError:
            pass
    if t == "boolean":
        s = v.strip().lower()
        if s in ("true", "false"):
            return s == "true"
    if t in ("object", "array") or t is None:
        try:
            return json.loads(v)
        except ValueError:
            pass
        if t is None:
            return v
    return v


def parse_call(body: str, tools: list | None) -> dict | None:
    """body = text between <tool_call> and </tool_call> (or a bare <function=..>...</function>)."""
    m = re.search(r"<function=([^>\n]+)>", body)
    if not m:
        return None
    name = m.group(1).strip()
    props = {}
    for t in tools or []:
        f = t.get("function", t)
        if f.get("name") == name:
            props = (f.get("parameters") or {}).get("properties") or {}
    rest = body[m.end():]
    end = rest.find("</function>")
    if end >= 0:
        rest = rest[:end]
    args = {}
    for pm in re.finditer(r"<parameter=([^>\n]+)>(.*?)(?:</parameter>|(?=<parameter=)|$)", rest, re.S):
        v = pm.group(2)
        if v.startswith("\n"):
            v = v[1:]
        if v.endswith("\n"):
            v = v[:-1]
        k = pm.group(1).strip()
        args[k] = _conv(v, props.get(k))
    return {"name": name, "arguments": args}


def new_id(prefix: str) -> str:
    return prefix + secrets.token_hex(12)


class OutParser:
    """feed(text) / finish() -> list of events: ("think", s), ("text", s), ("tool", {"name", "arguments"})."""

    def __init__(self, thinking: bool, tools: list | None, prefill: str = ""):
        self.state = "think" if thinking else "text"
        self.tools = tools or []
        self.buf = ""
        self.lead = False                     # strip whitespace right after </think>
        self.ncalls = 0
        self.ev: list = []
        if prefill:
            self.feed(prefill)

    def _emit(self, kind, s):
        if not s:
            return
        if self.ev and self.ev[-1][0] == kind and kind != "tool":
            self.ev[-1] = (kind, self.ev[-1][1] + s)
        else:
            self.ev.append((kind, s))

    @staticmethod
    def _hold(s: str, tags) -> int:
        """length of the longest suffix of s that is a proper prefix of one of tags"""
        best = 0
        for tag in tags:
            for k in range(min(len(tag) - 1, len(s)), 0, -1):
                if s.endswith(tag[:k]):
                    best = max(best, k)
                    break
        return best

    def feed(self, s: str):
        self.buf += s
        while True:
            if self.state == "think":
                j = self.buf.find(_TH_CLOSE)
                if j < 0:
                    h = self._hold(self.buf, (_TH_CLOSE,))
                    self._emit("think", self.buf[:len(self.buf) - h])
                    self.buf = self.buf[len(self.buf) - h:]
                    break
                self._emit("think", self.buf[:j])
                self.buf = self.buf[j + len(_TH_CLOSE):]
                self.state, self.lead = "text", True
                continue
            if self.state == "text":
                if self.lead:
                    self.buf = self.buf.lstrip()
                    if not self.buf:
                        break
                    self.lead = False
                tags = (_TC_OPEN, _FN_OPEN) if self.tools else ()
                js = [(self.buf.find(t), t) for t in tags]
                js = [x for x in js if x[0] >= 0]
                if not js:
                    h = self._hold(self.buf, tags)
                    out = self.buf[:len(self.buf) - h]
                    if tags:                                      # trailing whitespace waits: it may precede a call
                        h += len(out) - len(out.rstrip())
                        out = out.rstrip()
                    self._emit("text", out)
                    self.buf = self.buf[len(self.buf) - h:]
                    break
                j, t = min(js)
                pre = self.buf[:j]
                if self.ncalls:
                    pre = pre if pre.strip() else ""
                else:
                    pre = pre.rstrip()
                self._emit("text", pre)
                self.buf = self.buf[j + (len(t) if t == _TC_OPEN else 0):]
                self.state = "call"
                continue
            if self.state == "call":
                j = self.buf.find(_TC_CLOSE)
                if j < 0:
                    # normally a call ends at </tool_call>; a bare <function=..> call (no wrapper) ends at
                    # </function> once the text after it is clearly not </tool_call>
                    k = self.buf.find("</function>")
                    if k < 0:
                        break
                    rest = self.buf[k + len("</function>"):].lstrip()
                    if rest == "" or _TC_CLOSE.startswith(rest[:len(_TC_CLOSE)]):
                        break                                     # wait: </tool_call> probably follows
                    j = k + len("</function>")
                    body, self.buf = self.buf[:j], self.buf[j:]
                else:
                    body, self.buf = self.buf[:j], self.buf[j + len(_TC_CLOSE):]
                c = parse_call(body, self.tools)
                if c:
                    self.ncalls += 1
                    self.ev.append(("tool", c))
                else:
                    self._emit("text", _TC_OPEN + body + _TC_CLOSE)
                self.state = "text"
                continue
            break
        ev, self.ev = self.ev, []
        return ev

    def finish(self):
        """end of output: flush held text; a truncated / unclosed call is parsed if it has a name."""
        if self.state == "call":
            c = parse_call(self.buf, self.tools)
            if c:
                self.ncalls += 1
                self.ev.append(("tool", c))
            else:
                self._emit("text", self.buf)
        elif self.state == "think":
            self._emit("think", self.buf)
        else:
            out = self.buf
            if self.ncalls and not out.strip():
                out = ""
            self._emit("text", out)
        self.buf = ""
        ev, self.ev = self.ev, []
        return ev
