"""Where the weights come from, pinned, and where the converted caches go.

Nothing here is redistributed: every file is downloaded from its publisher at a fixed revision and checked against the
sizes and SHA-256 sums below (or, for the files the Hub verifies itself, the pinned revision).

| what | repo @ revision | size | used by |
|---|---|---:|---|
| Coder GGUF (IQ1_M, 2 shards) | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF @ 5348543e | 58.4 GB | coder, coder-fast |
| base GGUF (IQ3_XXS, 2 shards) | ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF @ ed59f920 | 75.8 GB | full |
| GPTQ-BFP4 expert pool (48 layers) | Lottolabs/Qwen3.8-Flash-Next-Coder-P150 @ 69e738b2 | 32 GB | coder, coder-fast |
| MTP layer: 31 BF16 tensors, byte ranges of 28 shards | Qwen/Qwen3.8-Flash-Next @ de4b8e4d | 5.2 GB read | all (packed to 0.89 GB) |

All of them are under the Qwen Community License 1.0 (Qwen's weights and quantizations of them).

Environment overrides (local copies instead of downloads): Q38_GGUF_DIR (the two shards of the selected model),
Q38_LOTTO_POOL (the pool's quant/store/pool directory), Q38_MTP_GGUF (a packed mtp-q2_0.gguf). Q38_CACHE: where the
converted caches go (default: $TT_DIT_CACHE_DIR inside a tt-model container, else ~/.cache/tt-qwen38).
P100A_MODEL = coder (default) | full selects the model; P100A_WCACHE overrides the card-cache directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

MODEL = os.environ.get("P100A_MODEL", "coder")
if MODEL not in ("coder", "full"):
    raise ValueError(f"P100A_MODEL={MODEL!r}: expected coder or full")

GGUF = {
    "coder": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF", "5348543e0147355ac9cbcb031184a3546350988e", {
        "IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf":
            (29608446496, "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
        "IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf":
            (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113")}),
    "full": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "ed59f92082b1e93c0e96d60a8b11aab089b52f09", {
        "IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf":
            (47039860096, "219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"),
        "IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf":
            (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113")}),
}
LOTTO_REPO = "Lottolabs/Qwen3.8-Flash-Next-Coder-P150"
LOTTO_REVISION = "69e738b2c8602ca93097ba0261a7649e4e71df04"
LOTTO_SHA_FILE = pathlib.Path(__file__).resolve().parent / "data" / "lotto_pool_sha256.txt"
MTP_SHA256 = "a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d"     # packed mtp-q2_0.gguf
MTP_BYTES = 889017216
_HERE = pathlib.Path(__file__).resolve().parent


def log(msg: str):
    print(f"weights: {msg}", flush=True)


def cache_root() -> pathlib.Path:
    if os.environ.get("Q38_CACHE"):
        return pathlib.Path(os.environ["Q38_CACHE"])
    if os.environ.get("TT_DIT_CACHE_DIR"):                  # tt-model's per-model writable cache mount
        return pathlib.Path(os.environ["TT_DIT_CACHE_DIR"])
    return pathlib.Path.home() / ".cache" / "tt-qwen38"


def card_cache() -> pathlib.Path:
    """converted card tensors (dense weights, resident expert sets) of the selected model"""
    if os.environ.get("P100A_WCACHE"):
        return pathlib.Path(os.environ["P100A_WCACHE"])
    return cache_root() / f"wcache-{MODEL}"


def _snapshot(repo, revision, patterns):
    from huggingface_hub import snapshot_download
    return pathlib.Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _checked(path: pathlib.Path, size: int, sha: str, full: bool) -> None:
    """size always; sha256 when `full` (downloads through the Hub are already sha256-checked by huggingface_hub),
    remembered per (path, size, mtime) so it runs once"""
    st = path.stat()
    if st.st_size != size:
        raise RuntimeError(f"{path}: {st.st_size} bytes, expected {size} (incomplete or a different file)")
    if not full:
        return
    db = cache_root() / "verified.json"
    key = f"{path.resolve()}:{st.st_size}:{int(st.st_mtime)}"
    try:
        seen = json.loads(db.read_text())
    except (OSError, ValueError):
        seen = {}
    if seen.get(key) == sha:
        return
    t0 = time.time()
    got = _sha256(path)
    if got != sha:
        raise RuntimeError(f"{path}: sha256 {got} != pinned {sha}")
    log(f"sha256 ok {path.name} ({time.time() - t0:.0f} s)")
    seen[key] = sha
    try:
        db.parent.mkdir(parents=True, exist_ok=True)
        db.write_text(json.dumps(seen))
    except OSError:
        pass


def gguf_shards(model: str | None = None) -> list:
    """the selected model's two GGUF shards (downloaded on first use; ~58 / ~76 GB)"""
    model = model or MODEL
    repo, rev, files = GGUF[model]
    if os.environ.get("Q38_GGUF_DIR"):
        d, local = pathlib.Path(os.environ["Q38_GGUF_DIR"]), True
    else:
        rev = os.environ.get("TT_MODEL_WEIGHTS_REVISION") if model == "coder" and os.environ.get(
            "TT_MODEL_WEIGHTS_REVISION") else rev
        log(f"{repo}@{rev[:8]} ({sum(s for s, _ in files.values()) / 1e9:.1f} GB, Hugging Face cache)")
        d, local = _snapshot(repo, rev, list(files)), False
    out = []
    for name, (size, sha) in files.items():          # repo paths "<quant>/<file>"; a local dir holds the files flat
        p = d / (name.split("/")[-1] if local else name)
        _checked(p, size, sha, full=local and os.environ.get("Q38_VERIFY", "1") != "0")
        out.append(p)
    return out


def lotto_pool() -> pathlib.Path:
    """the Lottolabs GPTQ-BFP4 expert pool (quant/store/pool: L<nn>.b4.npy + L<nn>.b4idx.npy for 48 layers, 32 GB)"""
    if os.environ.get("Q38_LOTTO_POOL"):
        d, local = pathlib.Path(os.environ["Q38_LOTTO_POOL"]), True
    else:
        log(f"{LOTTO_REPO}@{LOTTO_REVISION[:8]} quant/store/pool (32 GB, Hugging Face cache)")
        d = _snapshot(LOTTO_REPO, LOTTO_REVISION, ["quant/store/pool/*", "LICENSE", "README.md"]) / "quant/store/pool"
        local = False
    sums = {}
    for line in LOTTO_SHA_FILE.read_text().splitlines():
        if line.strip():
            sha, name = line.split()
            sums[name] = sha
    for name, sha in sums.items():
        p = d / name
        if not p.exists():
            raise FileNotFoundError(f"{p}: missing from the Lottolabs pool")
        if local and os.environ.get("Q38_VERIFY", "1") != "0":
            _checked(p, p.stat().st_size, sha, full=True)
    return d


def mtp_gguf() -> pathlib.Path:
    """the MTP draft layer as mtp-q2_0.gguf: fetched as BF16 byte ranges of the pinned Qwen checkpoint and packed to
    q2_0 with Strata's tools (third_party/strata_mtp), exactly as Strata's setup builds it; checked by sha256"""
    if os.environ.get("Q38_MTP_GGUF"):
        p = pathlib.Path(os.environ["Q38_MTP_GGUF"])
        _checked(p, MTP_BYTES, MTP_SHA256, full=True)
        return p
    d = cache_root() / "mtp"
    p = d / "mtp-q2_0.gguf"
    if p.exists():
        _checked(p, MTP_BYTES, MTP_SHA256, full=True)
        return p
    d.mkdir(parents=True, exist_ok=True)
    tools = _HERE / "third_party" / "strata_mtp"
    t0 = time.time()
    log("MTP draft layer: fetching 31 BF16 tensors (5.2 GB) from Qwen/Qwen3.8-Flash-Next@de4b8e4d")
    subprocess.run([sys.executable, str(tools / "mtp_fetch.py"), "fetch", "--out", str(d)], check=True)
    log(f"MTP draft layer: packing to q2_0 ({time.time() - t0:.0f} s so far)")
    subprocess.run([sys.executable, str(tools / "mtp_pack.py"), "--src", str(d), "--experts", "q2_0", "--out", str(p)],
                   check=True)
    _checked(p, MTP_BYTES, MTP_SHA256, full=True)
    for t in (d / "tensors").glob("*.bin"):                 # the 5.2 GB of BF16 inputs are not needed any more
        t.unlink()
    log(f"MTP draft layer ready in {time.time() - t0:.0f} s")
    return p
