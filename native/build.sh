#!/usr/bin/env bash
# Build the host CPU-expert library (libhostexp.so) and the ggml CPU backend it calls, in two CPU variants:
#   v4: x86-64-v4 (AVX-512: Zen 4, Sapphire Rapids, ...)   v3: x86-64-v3 (AVX2 + FMA: Haswell / Zen 1 and newer)
# tt_qwen38_native picks one at run time from the CPU flags. Output: tt_qwen38_native/lib/{v3,v4}/ with
# libhostexp.so, libggml-base.so.0, libggml-cpu.so.0 and libgomp.so.1 (rpath $ORIGIN, so nothing else is needed).
# ggml comes from llama.cpp at a pinned commit: LLAMA_CPP_SRC (a local checkout at that commit) or a git clone.
# Needs: git, cmake, ninja or make, g++ with OpenMP.
set -eu
cd "$(dirname "$0")"
LLAMA_CPP_COMMIT=aa75ff3239f9cc675b081697deda9e1b112345c6
OUT=tt_qwen38_native/lib
WORK=${BUILD_DIR:-build}
mkdir -p "$WORK"
SRC=${LLAMA_CPP_SRC:-}
if [ -z "$SRC" ]; then
  SRC=$WORK/llama.cpp
  if [ ! -d "$SRC/.git" ]; then
    git clone --filter=blob:none -q https://github.com/ggml-org/llama.cpp "$SRC"
  fi
  git -C "$SRC" -c advice.detachedHead=false checkout -q "$LLAMA_CPP_COMMIT"
fi
[ "$(git -C "$SRC" rev-parse HEAD)" = "$LLAMA_CPP_COMMIT" ] || { echo "llama.cpp is not at $LLAMA_CPP_COMMIT" >&2; exit 1; }
GEN=Ninja; command -v ninja >/dev/null || GEN="Unix Makefiles"
for V in v3 v4; do
  case $V in
    v3) FLAGS="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON"; ARCH=x86-64-v3 ;;
    v4) FLAGS="-DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON"; ARCH=x86-64-v4 ;;
  esac
  B=$WORK/ggml-$V
  cmake -S "$SRC" -B "$B" -G "$GEN" -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF $FLAGS -DGGML_OPENMP=ON \
    -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DCMAKE_SKIP_BUILD_RPATH=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON -DCMAKE_INSTALL_RPATH='$ORIGIN' -DBUILD_SHARED_LIBS=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TOOLS=OFF >/dev/null
  cmake --build "$B" --target ggml-cpu ggml-base -j "${JOBS:-8}" >/dev/null
  mkdir -p "$OUT/$V"
  cp -L "$B/bin/libggml-base.so.0" "$B/bin/libggml-cpu.so.0" "$OUT/$V/"
  g++ -O3 -march=$ARCH -fopenmp -shared -fPIC -std=c++17 hostexp.cpp -I"$SRC/ggml/include" \
    -L"$OUT/$V" -l:libggml-cpu.so.0 -l:libggml-base.so.0 -Wl,-rpath,'$ORIGIN' -o "$OUT/$V/libhostexp.so"
  cp -L "$(g++ -print-file-name=libgomp.so.1)" "$OUT/$V/libgomp.so.1"
  for f in "$OUT/$V"/libggml-*.so.0; do patchelf --set-rpath '$ORIGIN' "$f" 2>/dev/null || true; done
  echo "built $V: $(ls "$OUT/$V" | tr '\n' ' ')"
done
