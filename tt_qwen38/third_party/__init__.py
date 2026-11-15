"""Vendored third-party code (see THIRD-PARTY.md). Importing this puts the vendored llama.cpp gguf-py (aa75ff3,
which knows ggml type Q2_0 = 42) first on sys.path, so `import gguf` gets that version."""
import pathlib
import sys

_GGUF_PY = str(pathlib.Path(__file__).resolve().parent / "gguf_py")
if _GGUF_PY not in sys.path:
    sys.path.insert(0, _GGUF_PY)
