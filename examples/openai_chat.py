"""One OpenAI chat completion against the server (standard library only).

    python examples/openai_chat.py [URL]        # default http://127.0.0.1:8000
"""
import json
import sys
import time
import urllib.request

url = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")
req = {
    "model": "qwen3.8-flash-next-p100a-coder",
    "messages": [{"role": "user", "content": "In two sentences: why does a mixture-of-experts model run fast on a "
                                                "small accelerator?"}],
    "max_tokens": 300,
    "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(req).encode(),
                                                  {"Content-Type": "application/json"}), timeout=600)
out = json.load(r)
print(out["choices"][0]["message"]["content"])
print(json.dumps({"finish_reason": out["choices"][0]["finish_reason"], "usage": out.get("usage"),
                  "timings": out.get("timings"), "wall_s": round(time.time() - t0, 2)}))
