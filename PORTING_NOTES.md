# Porting notes: Qwen3.8-Flash-Next on one p100a + the host CPU

## The model and the split

Qwen3.8-Flash-Next: 48 layers (36 gated-delta-net (GDN), 12 attention with a sparse indexer "QSA" past 2,051
tokens), hidden 2560, 4-branch hyper-connections, a hashed n-gram embedding (PLE, layer 1), an MoE block in every layer (top-10 of
512 routed experts, expert FFN 640, gated shared expert), an MTP draft layer. ISTA-DASLab's GSQ-RCO GGUFs store it in
IQ-type quantizations; the Coder keeps 256 of the 512 routed experts (IQ1_M file, 58.4 GB), the base model all 512
(IQ3_XXS, 75.8 GB).

The experts do not fit the card's 27.8 GiB. So:
- **On the card:** every dense weight (GDN, attention + indexer, hyper-connections, norms, LM head), converted once
  from the GGUF to BFP8/BF16 tile tensors (~5.3 GB), the MTP layer, and a **resident set of experts**: the hottest
  ones by Strata's routing profile, 8 GB by default (16 GB in `coder-fast`). For the Coder they come from Lottolabs'
  GPTQ-BFP4 pool (exact BFP4 values); for the base model they are rounded from the GGUF.
- **On the CPU:** the router (fp32), top-10 and softmax, and every expert that is not resident, computed by ggml's
  CPU dot-product kernels straight on the memory-mapped GGUF (`native/hostexp.cpp`). 8 OpenMP threads pinned to cores
  spread over the CCDs: +12-20 % decode against unpinned.

A decode round reads a 4-row verify window (the token + 3 MTP drafts) through all 48 layers; per layer the card's
residents and the CPU's cold experts run at the same time and their outputs are combined.

## What runs on the card

- Fused programs (one ttnn generic op each): the GDN chunk core (one head per core), the GDN causal conv, the GDN
  commit of an accepted prefix, the hyper-connection write, the KV row writer, and the MoE over the resident experts.
- Prompts are read in 28-row passes; every pass is a captured trace.
- MTP: the model's own draft layer proposes 3 tokens per round; acceptance 0.47 (chat) to 0.74 (code) as measured
  on this card.
- Prefix cache: model state snapshots at 28-row chunk ends in card slots (4 at 8 GB residents, 2 at 16 GB);
  agent loops skip their shared history.
- Several requests: time slicing (state slots on the card), up to 3 at 8 GB.

## The CPU library

`native/` builds ggml's CPU backend from llama.cpp at a pinned commit and `libhostexp.so` on top of it, twice:
x86-64-v4 (AVX-512) and x86-64-v3 (AVX2); `tt_qwen38_native` picks one from the CPU flags. On a Ryzen 9 7900
(Zen 4) the v4 build matches a `-march=native` build (1.18 vs 1.18 ms per layer of 4 rows x 7 experts; v3 1.20-1.27).
libgomp is copied beside it, so no system OpenMP runtime is needed.

## First start

`tt_qwen38/sources.py` pins every source (repo, revision, size, SHA-256): the GGUF shards, the Lottolabs pool (96
files) and the MTP layer, which no GGUF contains: its 31 BF16 tensors are read by byte range from the pinned Qwen
checkpoint and packed to q2_0 with Strata's own tools, exactly as Strata's setup does (the packed file is checked
against a pinned SHA-256). `server.py prepare()` then builds the card caches once.

## Measured

See the model card. Quality is measured against a float32 CPU reference of the same GGUF weights (port fidelity):
KL 0.0029 / top-1 0.983 at 8 GB residents, greedy output identical for 16 tokens on two prompts.
