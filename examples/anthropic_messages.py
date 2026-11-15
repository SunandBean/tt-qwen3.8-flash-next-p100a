"""One Anthropic Messages request against the server (standard library only).

    python examples/anthropic_messages.py [URL]        # default http://127.0.0.1:8000
"""
import json
import sys
import time
import urllib.request

url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
req = {
    "model": "qwen3.8-flash-next-p100a-coder",
    "max_tokens": 300,
    "temperature": 0,
    "system": "Answer briefly.",
    "messages": [{"role": "user", "content": "Name three things a tokenizer has to get right."}],
}
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(url + "/v1/messages", json.dumps(req).encode(),
                                                  {"Content-Type": "application/json",
                                                   "anthropic-version": "2023-06-01"}), timeout=600)
out = json.load(r)
print("".join(b.get("text", "") for b in out["content"] if b.get("type") == "text"))
print(json.dumps({"stop_reason": out.get("stop_reason"), "usage": out.get("usage"),
                  "wall_s": round(time.time() - t0, 2)}))
