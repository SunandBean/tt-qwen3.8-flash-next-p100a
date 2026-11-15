# Third-party code and data

`tt_qwen38/` and `native/hostexp.cpp` are this port (Apache-2.0) unless listed here. `examples/`, `tt-model.yaml`,
`pyproject.toml` and `native/{build.sh,setup.py,pyproject.toml,tt_qwen38_native/}` are packaging.

| path | upstream | licence | licence text |
| --- | --- | --- | --- |
| `tt_qwen38/ref/third_party/strata_gguf_reader.py`, `strata_tokenizer.py` | Strata `tools/gguf_reader.py`, `tools/strata_tokenizer.py` | MIT | `third_party_licenses/Strata-MIT.txt` |
| `tt_qwen38/third_party/strata_mtp/` | Strata `tools/mtp_fetch.py`, `tools/mtp_pack.py`, `strata_control/tls.py` (dbb2d168c; import paths changed) | MIT | `third_party_licenses/Strata-MIT.txt` |
| `tt_qwen38/data/expert-profile*.prof` | Strata `data/expert-profile*.bin` (renamed: tt-model's packaging drops `*.bin`) | MIT | `third_party_licenses/Strata-MIT.txt` |
| `tt_qwen38/third_party/gguf_py/gguf/` | llama.cpp `gguf-py/gguf` (aa75ff32) | MIT | `third_party_licenses/llama.cpp-ggml-MIT.txt` |
| ggml, built by `native/build.sh` (not in git) | llama.cpp `ggml/` (aa75ff32) | MIT | `third_party_licenses/llama.cpp-ggml-MIT.txt` |
| `libgomp.so.1`, copied by `native/build.sh` (not in git) | GCC | GPL-3.0 + GCC Runtime Library Exception | the system's GCC |
| `code/models/common`, `code/models/experimental/gated_attention_gated_deltanet` (model repository / container image only; the latter's delta-rule ops are imported) | [tt-metal](https://github.com/tenstorrent/tt-metal) `58648d4` | Apache-2.0 | tt-metal's `LICENSE` |

Design and numerics credits without copied code (see `NOTICE`): sjettTT/tt-qwen-3.8-flash-next (Apache-2.0).

`tt_qwen38/data/lotto_pool_sha256.txt` lists SHA-256 sums of the Lottolabs pool files at the pinned revision; the
files themselves are downloaded, never redistributed. The weights are under the Qwen Community License 1.0
(`third_party_licenses/Qwen-Community-License-1.0.txt`).
