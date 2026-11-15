"""Locates libhostexp.so for this CPU: lib/v4 (AVX-512) when the CPU has it, else lib/v3 (AVX2).
Q38_NATIVE_VARIANT=v3|v4 forces one."""
from __future__ import annotations

import os
import pathlib

LIB = pathlib.Path(__file__).resolve().parent / "lib"
V4_FLAGS = {"avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl", "avx512_vnni", "avx512_bf16", "avx512vbmi"}


def cpu_flags() -> set:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("flags"):
                return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def variant() -> str:
    v = os.environ.get("Q38_NATIVE_VARIANT")
    if v:
        return v
    return "v4" if V4_FLAGS <= cpu_flags() else "v3"


def lib_path(name: str = "libhostexp.so") -> pathlib.Path:
    p = LIB / variant() / name
    if not p.exists():
        raise FileNotFoundError(f"{p}: tt-qwen38-native was installed without its {variant()} build")
    return p
