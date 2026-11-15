"""The GGUF's byte-level BPE tokenizer via Strata's reference tokenizer (MIT, vendored in third_party/)."""
from __future__ import annotations

from functools import lru_cache

from .gguf_weights import Weights
from .third_party.strata_tokenizer import Tokenizer


@lru_cache(maxsize=1)
def tokenizer(W: Weights | None = None) -> Tokenizer:
    md = (W or Weights()).meta
    special = {k: int(md[k]) for k in md if k.startswith("tokenizer.ggml.") and k.endswith("_token_id")}
    return Tokenizer(list(md["tokenizer.ggml.tokens"]), list(md["tokenizer.ggml.merges"]),
                     list(md.get("tokenizer.ggml.token_type") or []) or None, md.get("tokenizer.ggml.pre", "qwen35"),
                     special)
