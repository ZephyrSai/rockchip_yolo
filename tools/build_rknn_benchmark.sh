#!/usr/bin/env bash
#
# tools/build_rknn_benchmark.sh — build Rockchip's own `rknn_benchmark`
# tool (from airockchip/rknn-toolkit2, rknpu2/examples/rknn_benchmark)
# WITHOUT cloning the multi-GB SDK repo: the six source files it needs are
# fetched individually from GitHub raw and compiled with a plain g++ /
# NDK clang++ command.
#
#   ./tools/build_rknn_benchmark.sh linux   [outdir]   # native build on the board (needs g++, zlib1g-dev)
#   ./tools/build_rknn_benchmark.sh android [outdir]   # needs ANDROID_NDK_PATH (any recent NDK, r25+ fine)
#
# Output:
#   <outdir>/rknn_benchmark            (linux)   — links against the board's librknnrt.so
#   <outdir>/rknn_benchmark_android    (android) — links against the SDK's Android arm64-v8a librknnrt.so
#   <outdir>/lib/librknnrt.so          — the runtime the binary was linked with (for LD_LIBRARY_PATH)
#
# rknn_benchmark usage (from Rockchip's README):
#   ./rknn_benchmark model.rknn [input_data] [loop_count] [core_mask]
#   core_mask (RK3588): 0=auto 1=core0 2=core1 4=core2 3=core0+1 7=core0+1+2
#   prints:  "rknn_api/rknnrt version: X, driver version: Y"
#            "Avg Time 12.34ms, Avg FPS = 81.037"

set -euo pipefail

MODE="${1:-linux}"
OUT="${2:-$PWD/rknn_benchmark_build}"
RAW="https://raw.githubusercontent.com/airockchip/rknn-toolkit2/master"
mkdir -p "$OUT/src/cnpy" "$OUT/3rdparty/stb" "$OUT/include" "$OUT/lib"

fetch() { # url dest
  [ -s "$2" ] || curl -fsSL -o "$2" "$1" || { echo "download failed: $1"; exit 1; }
}
echo "Fetching rknn_benchmark sources into $OUT ..."
fetch "$RAW/rknpu2/examples/rknn_benchmark/src/rknn_benchmark.cpp" "$OUT/src/rknn_benchmark.cpp"
fetch "$RAW/rknpu2/examples/rknn_benchmark/src/cnpy/cnpy.cpp"      "$OUT/src/cnpy/cnpy.cpp"
fetch "$RAW/rknpu2/examples/rknn_benchmark/src/cnpy/cnpy.h"        "$OUT/src/cnpy/cnpy.h"
fetch "$RAW/rknpu2/examples/3rdparty/stb/stb_image.h"              "$OUT/3rdparty/stb/stb_image.h"
fetch "$RAW/rknpu2/examples/3rdparty/stb/stb_image_resize.h"       "$OUT/3rdparty/stb/stb_image_resize.h"
fetch "$RAW/rknpu2/examples/3rdparty/stb/stb_image_write.h"        "$OUT/3rdparty/stb/stb_image_write.h"
fetch "$RAW/rknpu2/runtime/Linux/librknn_api/include/rknn_api.h"   "$OUT/include/rknn_api.h"

case "$MODE" in
  linux)
    # Prefer the board's installed runtime so the binary reports the same
    # librknnrt version the Python path uses; fall back to the SDK copy.
    RT=""
    for c in /usr/lib/librknnrt.so /usr/lib/aarch64-linux-gnu/librknnrt.so /usr/local/lib/librknnrt.so; do
      [ -f "$c" ] && { RT="$c"; break; }
    done
    if [ -z "$RT" ]; then
      RT=$(ldconfig -p 2>/dev/null | awk '/librknnrt.so/{print $NF; exit}')
    fi
    if [ -z "$RT" ]; then
      echo "No librknnrt.so on this system — downloading the SDK copy (v2.3.2 line)"
      fetch "$RAW/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so" "$OUT/lib/librknnrt.so"
      RT="$OUT/lib/librknnrt.so"
    else
      cp -f "$RT" "$OUT/lib/librknnrt.so"
    fi
    echo "Linking against: $RT"
    command -v g++ >/dev/null || { echo "g++ missing: sudo apt install build-essential zlib1g-dev"; exit 1; }
    g++ -std=c++14 -O2 -I"$OUT/src" -I"$OUT/3rdparty" -I"$OUT/include" \
        "$OUT/src/rknn_benchmark.cpp" "$OUT/src/cnpy/cnpy.cpp" \
        -L"$(dirname "$RT")" -lrknnrt -lz -ldl -lpthread \
        -Wl,-rpath,"$(dirname "$RT")" -o "$OUT/rknn_benchmark" \
      || { echo "build failed (is zlib1g-dev installed?)"; exit 1; }
    echo "Built: $OUT/rknn_benchmark"
    ;;
  android)
    : "${ANDROID_NDK_PATH:?set ANDROID_NDK_PATH to your NDK root (e.g. ~/Android/Sdk/ndk/27.2.12479018)}"
    HOST_TAG=$(ls "$ANDROID_NDK_PATH/toolchains/llvm/prebuilt/" | head -1)
    CXX="$ANDROID_NDK_PATH/toolchains/llvm/prebuilt/$HOST_TAG/bin/aarch64-linux-android24-clang++"
    [ -x "$CXX" ] || { echo "NDK clang++ not found at $CXX"; exit 1; }
    fetch "$RAW/rknpu2/runtime/Android/librknn_api/arm64-v8a/librknnrt.so" "$OUT/lib/librknnrt.so"
    "$CXX" -std=c++14 -O2 -static-libstdc++ -I"$OUT/src" -I"$OUT/3rdparty" -I"$OUT/include" \
        "$OUT/src/rknn_benchmark.cpp" "$OUT/src/cnpy/cnpy.cpp" \
        "$OUT/lib/librknnrt.so" -lz -ldl -o "$OUT/rknn_benchmark_android" \
      || { echo "android build failed"; exit 1; }
    echo "Built: $OUT/rknn_benchmark_android  (push with lib/librknnrt.so, or use /vendor/lib64/librknnrt.so on the device)"
    ;;
  *)
    echo "usage: $0 linux|android [outdir]"; exit 1 ;;
esac
