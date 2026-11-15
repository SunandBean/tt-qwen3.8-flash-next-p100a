# tt-qwen3.8-flash-next-p100a

[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) on one Tenstorrent Blackhole **p100a** plus the
host CPU, behind an OpenAI- and Anthropic-compatible server. Model card, numbers and the container package:
[sunandbean/qwen3.8-flash-next-p100a](https://huggingface.co/sunandbean/qwen3.8-flash-next-p100a).

- Dense layers, attention, the LM head, the MTP draft layer and the hottest experts on the card; routing and the
  remaining experts on the CPU (ggml kernels on the memory-mapped GGUF).
- Profiles: `coder` (default; the ISTA-DASLab Coder GGUF, 8 GB of Lottolabs GPTQ-BFP4 experts on the card),
  `coder-fast` (16 GB on the card), `full` (the base model, 512 experts).
- ~10-17 tok/s decode with MTP speculative decoding, 50/50 on GSM8K-50, 32k context, prefix cache, tool calls,
  streaming; works with Claude Code (`/v1/messages`) and OpenAI clients.
- **Needs a 64 GB host.** Weights (Qwen Community License 1.0) are downloaded at first start from their publishers at
  pinned revisions and converted on your machine; nothing is redistributed. Read `NOTICE` for the license terms.

## Run it

With [tt-model](https://github.com/tenstorrent/tt-model-manager) and Docker:

```bash
tt-model pull  sunandbean/qwen3.8-flash-next-p100a --with-weights
tt-model serve sunandbean/qwen3.8-flash-next-p100a                 # or --profile coder-fast / full
```

From source, in a tt-metal `58648d4` environment (e.g. the `tt-metalium-ubuntu-22.04-release-models` container) with
a p100a, git, cmake and a C++ compiler:

```bash
pip install ./native -e .                         # the native CPU-expert library (builds ggml), then this package
python -m tt_qwen38.server --profile coder --port 8000
python examples/openai_chat.py http://127.0.0.1:8000
python examples/anthropic_messages.py http://127.0.0.1:8000
```

The first start downloads ~96 GB (Coder GGUF 58.4 GB, Lottolabs pool 32 GB, the MTP layer's tensors 5.2 GB) and
builds ~14 GB of card caches; `--prepare-only` does that without the card. Weight sources and overrides:
`tt_qwen38/sources.py`.

## Layout

| path | what |
|---|---|
| `tt_qwen38/server.py` | the server (`python -m tt_qwen38.server`), profiles, first-start preparation |
| `tt_qwen38/api.py`, `chat.py` | HTTP layer: OpenAI / Anthropic endpoints, chat template, tool-call parsing |
| `tt_qwen38/model.py`, `traced.py`, `verify.py`, `spec.py` | the model on the card, multi-row verify passes, MTP speculative decoding |
| `tt_qwen38/kgdn.py`, `kmoe.py`, `khc.py`, `kkv.py`, `kernels/` | fused gated-delta-net, MoE, hyper-connection and KV kernels |
| `tt_qwen38/experts.py`, `hostexp/` | card-resident expert sets; CPU experts through the native library |
| `tt_qwen38/sources.py`, `weights.py` | pinned weight sources and checksums; dense card cache |
| `tt_qwen38/ref/` | GGUF reader, tokenizer, float32 reference model |
| `native/` | the CPU-expert library (`hostexp.cpp` + ggml at llama.cpp aa75ff32), x86-64-v3 / v4 builds |
| `tt-model.yaml` | the tt-model container package |

## License

Apache-2.0 for this code (`LICENSE`, `NOTICE`). Vendored code and its licences: `THIRD-PARTY.md`. The model
weights are under the Qwen Community License 1.0 and are not part of this repository.
