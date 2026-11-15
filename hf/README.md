---
tags:
- blackhole
- p100a
- tenstorrent
- ttnn
- tt-metal
- tt-nn
- text-generation
- moe
- qwen3.8
pipeline_tag: text-generation
license: apache-2.0
base_model:
- Qwen/Qwen3.8-Flash-Next
- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF
- ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
base_model_relation: quantized
---

# qwen3.8-flash-next-p100a

Qwen3.8-Flash-Next on one Tenstorrent Blackhole p100a plus the host CPU, with an OpenAI- and Anthropic-compatible API.
Weights: [ISTA-DASLab Coder GGUF](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF) (`5348543e`),
[Lottolabs GPTQ-BFP4 pool](https://huggingface.co/Lottolabs/Qwen3.8-Flash-Next-Coder-P150) (`69e738b2`), the MTP layer of
[Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) (`de4b8e4d`), all downloaded at first start,
none redistributed · Port: [SunandBean/tt-qwen3.8-flash-next-p100a](https://github.com/SunandBean/tt-qwen3.8-flash-next-p100a)

Runs on **p100a** (mesh 1×1) **and a 64 GB host**. Dense layers (gated delta net, sparse attention,
hyper-connections), the LM head, the MTP draft layer and the hottest experts are on the card. Routing and the
remaining experts run on the CPU, the experts read from the memory-mapped GGUF through ggml's CPU kernels.

**What this port adds**: a single-card ttnn build of the model; fused gated-delta-net, MoE, KV and hyper-connection
kernels; card-resident experts placed by Strata's routing profile; host CPU experts through ggml; MTP speculative
decoding; 28-row prompt passes; a prefix cache; time slicing for several users; and a first start that fetches and
converts everything on your machine.
**What it builds on**: the ttnn op recipes of
[sjettTT/tt-qwen-3.8-flash-next](https://github.com/sjettTT/tt-qwen-3.8-flash-next) and
[tt-metal](https://github.com/tenstorrent/tt-metal) (Apache-2.0); Strata's tokenizer, GGUF reader, expert profiles
and MTP pipeline, and llama.cpp's ggml and gguf-py (MIT); the GSQ-RCO quantizations by ISTA-DASLab and the GPTQ-BFP4
expert pool by Lottolabs.

**~10-17 tok/s decode, 50/50 on GSM8K-50, 32k context, on one card and a desktop CPU.**

## Quickstart (container)

```bash
tt-model pull  sunandbean/qwen3.8-flash-next-p100a --with-weights
tt-model serve sunandbean/qwen3.8-flash-next-p100a                    # profile coder (default)
tt-model serve --profile full sunandbean/qwen3.8-flash-next-p100a     # the base model
```

`pull --with-weights` downloads the image and the Coder GGUF (58.4 GB). On its first start the server also
fetches the Lottolabs expert pool (32 GB) and the MTP layer's tensors (5.2 GB, packed to 0.9 GB), checks every file
against pinned SHA-256 sums, and builds the card caches (~14 GB): dense weights, the resident expert set and the MTP
layer. Later starts load them. The server listens on port 20000 (or the next free one) and logs
`{"event": "ready", ...}` when ready.

| profile | model | experts on the card | first-start downloads |
|---|---|---|---|
| `coder` (default) | Qwen3.8-Flash-Next Coder (256 experts, IQ1_M GGUF) | 8 GB, Lottolabs GPTQ-BFP4 | ~96 GB |
| `coder-fast` | same | 16 GB, Lottolabs GPTQ-BFP4 | ~96 GB |
| `full` | base Qwen3.8-Flash-Next (512 experts, IQ3_XXS GGUF) | 8 GB, from the GGUF | ~81 GB (GGUF 75.8 + MTP 5.2); `tt-model serve` also fetches the Coder GGUF (the manifest's weights pointer) unless started with `--no-weights` |

```bash
curl -s localhost:20000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next-p100a-coder", "max_tokens": 300,
  "messages": [{"role": "user", "content": "Write a Python function that parses ISO-8601 durations."}]}'

ANTHROPIC_BASE_URL=http://127.0.0.1:20000 ANTHROPIC_AUTH_TOKEN=dummy claude    # Claude Code; serve with --thinking off
```

## Host requirements

| | |
|---|---|
| **RAM** | **64 GB.** The CPU experts are read from the memory-mapped GGUF and must stay in the page cache (37.8 GB if every expert of `full` is touched). With less RAM, decode falls to disk speed. |
| **CPU** | x86-64 with AVX2 (8+ cores). The host experts use 8 OpenMP threads pinned to cores (`OMP_PROC_BIND=spread`, `OMP_PLACES=cores`: +12-20 % decode). AVX-512 CPUs get the x86-64-v4 build of ggml automatically. |
| **Disk** | ~110 GB for `coder` (downloads ~96 GB + caches ~14 GB); `full` adds ~90 GB (GGUF 75.8 GB + caches ~13 GB). |
| **Card** | one p100a (mesh 1×1). |

## Quickstart (Python)

Prerequisite: a tt-metal / ttnn environment at `58648d4` (ttnn is not on PyPI), e.g. tt-metal's
`tt-metalium-ubuntu-22.04-release-models` container, plus git, cmake and a C++ compiler for the native library.

```bash
hf download sunandbean/qwen3.8-flash-next-p100a --include "code/*" --local-dir qwen38 && cd qwen38/code
pip install jinja2 huggingface_hub regex pyyaml ./native     # builds the native CPU-expert library
python -m tt_qwen38.server --profile coder --port 8000         # first start: downloads + builds the caches
```

| | |
|---|---|
| **API** | `POST /v1/chat/completions` (OpenAI; tools, `tool_choice`, streaming, `reasoning_content`), `POST /v1/responses`, `POST /v1/messages` (Anthropic; tools, thinking blocks, streaming), `POST /v1/messages/count_tokens`, `GET /v1/models`, `GET /health`. Responses carry llama.cpp-style `timings`. |
| **Sampling** | `temperature` (default 1.0, as in the OpenAI API; 0 = greedy), `top_p`, `top_k`, `seed`, `stop` / `stop_sequences`. Exact with speculative decoding. |
| **Thinking** | OpenAI: on unless `chat_template_kwargs.enable_thinking` is false or `reasoning_effort` is `"none"`; Anthropic: on when `thinking` is enabled. `--thinking off` disables it for every request (agents). |
| **Weights** | pinned in `tt_qwen38/sources.py`; `Q38_GGUF_DIR`, `Q38_LOTTO_POOL`, `Q38_MTP_GGUF` point at local copies; `Q38_CACHE` moves the converted caches. `--prepare-only` downloads and converts without the card. |

## Performance

Measured 2026-11-12 on one p100a with a Ryzen 9 7900 host (8 CPU threads pinned), through the HTTP server, greedy,
thinking off, MTP speculative decoding with 3 drafts, medians of 3 runs.

| profile | decode tok/s, chat / code / doc | plain decode (no MTP) | prompt reading |
|---|---:|---:|---:|
| `coder` (8 GB on the card, default) | **10.4 / 14.5 / 13.7** | 8.3 / 8.3 / 8.4 | 31-35 tok/s |
| `coder-fast` (16 GB on the card) | **12.4 / 16.4 / 16.6** | 8.8 / 8.9 / 9.0 | 44-58 tok/s |
| `full` (base model, 8 GB on the card) | **10.9 / 14.1 / 12.2** | 8.0 / 8.2 / 8.0 | 31-34 tok/s |

| | |
|---|---|
| Long context (32k; sparse attention past 2,051 tokens) | needles found at 8k and 30k tokens; at 30k prompt reading 29.2 tok/s, decode 9.65 tok/s |
| Several users | up to 3 requests time-sliced (4 prefix-cache slots) at 8 GB; two clients replaying an agent trace at once got byte-identical outputs to one client |
| Claude Code (`full`, `--thinking off`) | a 7-turn task (add a function, write and run tests) in 5 min 21 s; first turn 207 s (4k-token prompt), later turns started after 8-34 s (prefix cache) |

## Accuracy

Against a float32 CPU run of the same GGUF weights (517 teacher-forced positions; port fidelity, not the
quantization itself):

| profile | KL | top-1 | greedy 16 tokens, 2 prompts | GSM8K-50 @ 2,048 tokens |
|---|---:|---:|---|---|
| `coder` | 0.0029 | 0.983 | 16/16, 16/16 | **50/50** (1,466 s) |
| `coder-fast` | 0.0046 | 0.979 | 16/16, 16/16 | **50/50** (1,175 s) |
| `full` | 0.0029 | 0.983 | 16/16, 16/16 | **50/50** (1,541 s) |

Korean (4 prompts): the `full` model answers cleanly; the Coder drifts into English or loops, and so does the
float32 Coder itself (a property of the pruned model, not of the port).

## Caveats

- **64 GB host.** Below that, decode speed collapses to disk reads.
- **The CPU sets the pace**: ~10-17 tok/s decode, ~30-60 tok/s prompt reading.
- **First start downloads ~96 GB and converts on your machine.** Inside the container `tt-model serve` shows no
  progress for these steps; `tt-model logs -f` does.
- **The Coder is weak in Korean**; use `full`.
- **32k context**, one card, no batched decode (extra requests wait their turn).
- `tt-model serve` refuses while another container holds `/dev/tenstorrent`.
- The RTX 5070 Ti hybrid tier of the development tree (warm experts on a GPU) is not part of this package.

## Licensing

- **Weights: Qwen Community License 1.0**, Copyright (c) 2026 Qwen. They are downloaded from ISTA-DASLab, Lottolabs
  and Qwen at pinned revisions and are not part of this repository. Two conditions to know (read the full text in
  `third_party_licenses/Qwen-Community-License-1.0.txt`; this is not legal advice):
  1. Keep the copyright and permission notice. A commercial product or service with more than 100,000,000 monthly
     active users or US$ 20,000,000 monthly revenue must display the model name prominently in its user interface.
  2. **If you or an affiliate run a Model-as-a-Service business** (giving third parties access to model inference,
     e.g. through an API or hosted endpoint like this server, with meaningful control over inputs or parameters)
     **or an AI-Work-Assistant business** (an independent AI product primarily for AI-assisted coding or office
     productivity), **you need a separate license from Qwen before any commercial use.** Internal use that makes
     neither the software, its outputs nor its capabilities available to any third party is exempt.
- Port code (`code/`): Apache-2.0, from [SunandBean/tt-qwen3.8-flash-next-p100a](https://github.com/SunandBean/tt-qwen3.8-flash-next-p100a).
- Vendored code: Strata (MIT), llama.cpp's ggml and gguf-py (MIT); libgomp (GCC Runtime Library Exception);
  `code/models/common` and `code/models/experimental/gated_attention_gated_deltanet` from tt-metal (Apache-2.0). `NOTICE` and `code/THIRD-PARTY.md` list each with its licence.
