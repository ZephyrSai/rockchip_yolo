#!/usr/bin/env bash
#
# android/test_yolo_rockchip_android.sh — run ON A HOST PC with adb, against
# an RK3588 / RK3576 device running Android (USB or network adb).
#
# What it does:
#   0. Device identity (ro.board.platform, model, Android version), NPU stack
#      presence (/vendor/lib64/librknnrt.so + version, rknn_server), NNAPI
#      vendor HALs (Rockchip ships none for the NPU — NNAPI = CPU reference)
#   1. Pushes Google's prebuilt TFLite benchmark_model (android_aarch64) and
#      runs each .tflite model on CPU (XNNPACK, 4 threads), GPU delegate
#      (OpenCL on Mali) and NNAPI — parses "Inference (avg)" and the
#      delegate-applied line.
#   2. If an Android rknn_benchmark binary is available (build with
#      tools/build_rknn_benchmark.sh android, needs an NDK), pushes it and
#      runs every <model>_<soc>_{fp16,int8}_rknn_model/*.rknn on the NPU
#      with core mask auto and all-cores, using the device's own
#      /vendor/lib64/librknnrt.so.
#   3. Summary table + CSV (same schema as the Linux suite).
#
# Usage:
#   ./android/test_yolo_rockchip_android.sh --models-dir ~/yolo-rockchip-export \
#        [--serial <adb serial>] [--rknn-benchmark path/to/rknn_benchmark_android]
#
# --models-dir: directory holding the outputs of export_rknn_host.sh
#   (*_saved_model/*.tflite and *_<soc>_*_rknn_model/). Run that with
#   --tflite on an x86 PC first.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$HOME/yolo-rockchip-export"
SERIAL=""
RKNN_BENCH=""
WORKDIR="$HOME/yolo-rockchip-android-test"
DEV_DIR="/data/local/tmp/yolo_bench"
BM_URL="https://storage.googleapis.com/tensorflow-nightly-public/prod/tensorflow/release/lite/tools/nightly/latest/android_aarch64_benchmark_model"

while [ $# -gt 0 ]; do
  case "$1" in
    --models-dir) MODELS_DIR="$2"; shift ;;
    --serial) SERIAL="$2"; shift ;;
    --rknn-benchmark) RKNN_BENCH="$2"; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
  esac; shift
done
mkdir -p "$WORKDIR/logs"; LOGDIR="$WORKDIR/logs"
RESULTS_FILE="$WORKDIR/results_summary.txt"; BENCH_CSV="$WORKDIR/benchmark_results.csv"
: > "$RESULTS_FILE"; echo "backend,model,stage,metric,value_ms_or_fps" > "$BENCH_CSV"
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
section() { echo -e "\n${BLUE}==================================================================${NC}\n${BLUE}  $1${NC}\n${BLUE}==================================================================${NC}"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; echo "[PASS] $1" >> "$RESULTS_FILE"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; echo "[FAIL] $1" >> "$RESULTS_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; echo "[WARN] $1" >> "$RESULTS_FILE"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
ADB="adb${SERIAL:+ -s $SERIAL}"
ash() { $ADB shell "$@"; }

section "STEP 0: Device identity + NPU stack"
command -v adb >/dev/null || { fail "adb not found (sudo apt install adb / Android platform-tools)"; exit 1; }
$ADB get-state >/dev/null 2>&1 || { fail "No adb device (adb devices; enable USB debugging; pass --serial)"; exit 1; }
PLATFORM=$(ash getprop ro.board.platform | tr -d '\r'); MODEL=$(ash getprop ro.product.model | tr -d '\r'); AVER=$(ash getprop ro.build.version.release | tr -d '\r')
HW=$(ash getprop ro.hardware | tr -d '\r'); ABI=$(ash getprop ro.product.cpu.abi | tr -d '\r')
echo "platform=$PLATFORM model=$MODEL android=$AVER hardware=$HW abi=$ABI" | tee "$LOGDIR/device.log"
case "$PLATFORM" in
  rk3588*) SOC=rk3588; NPU_CORES=3; pass "RK3588 Android device: $MODEL (Android $AVER)" ;;
  rk3576*) SOC=rk3576; NPU_CORES=2; pass "RK3576 Android device: $MODEL (Android $AVER)" ;;
  *) COMPAT=$(ash "cat /proc/device-tree/compatible 2>/dev/null | tr '\0' ' '" | tr -d '\r')
     if echo "$COMPAT" | grep -q rk3588; then SOC=rk3588; NPU_CORES=3; elif echo "$COMPAT" | grep -q rk3576; then SOC=rk3576; NPU_CORES=2; else SOC=rk3588; NPU_CORES=3; fi
     warn "ro.board.platform='$PLATFORM' not recognised; guessing $SOC from device-tree ($COMPAT)" ;;
esac
[ "$ABI" = "arm64-v8a" ] || warn "ABI $ABI is not arm64-v8a — prebuilt binaries below are aarch64"

if ash "ls /vendor/lib64/librknnrt.so" 2>/dev/null | grep -q librknnrt; then
  $ADB pull /vendor/lib64/librknnrt.so "$WORKDIR/librknnrt.so" >/dev/null 2>&1
  RTV=$(strings "$WORKDIR/librknnrt.so" 2>/dev/null | grep -oE "librknnrt version: [0-9.]+" | awk '{print $3}' | head -1)
  pass "/vendor/lib64/librknnrt.so present (version ${RTV:-unknown})"; RKNN_RT=1
else
  warn "No /vendor/lib64/librknnrt.so — this firmware was built without the RKNN stack (BOARD_RKNN_SUPPORT=false); NPU unusable from Android"; RKNN_RT=0
fi
ash "ls /vendor/bin/rknn_server" 2>/dev/null | grep -q rknn_server && info "rknn_server present ($(ash 'ps -A | grep -c rknn_server' | tr -d '\r') running) — only needed for PC-side rknn-toolkit2 target mode" || info "no rknn_server (fine for on-device inference)"
NNHAL=$(ash "ls /vendor/bin/hw 2>/dev/null | grep -i neuralnetworks" | tr -d '\r')
[ -n "$NNHAL" ] && pass "NNAPI vendor HAL(s): $NNHAL" || warn "No NNAPI vendor HAL — Rockchip ships none for the NPU, so --use_nnapi runs on nnapi-reference (CPU)"
GPUOCL=$(ash "ls /vendor/lib64/libOpenCL.so /vendor/lib64/egl/libGLES_mali.so 2>/dev/null" | tr -d '\r' | tr '\n' ' ')
[ -n "$GPUOCL" ] && pass "Mali GPU libs: $GPUOCL" || warn "No /vendor/lib64/libOpenCL.so — TFLite GPU delegate will fall back to OpenGL or fail"

section "STEP 1: TFLite benchmark_model — CPU / GPU / NNAPI"
BM="$WORKDIR/android_aarch64_benchmark_model"
[ -s "$BM" ] || curl -fsSL -o "$BM" "$BM_URL" || { fail "Could not download benchmark_model from $BM_URL"; }
ash "mkdir -p $DEV_DIR" >/dev/null
if [ -s "$BM" ]; then
  $ADB push "$BM" "$DEV_DIR/benchmark_model" >/dev/null && ash "chmod +x $DEV_DIR/benchmark_model" && pass "benchmark_model pushed to $DEV_DIR"
  TFL_FILES=$(ls "$MODELS_DIR"/*_saved_model/*_float32.tflite "$MODELS_DIR"/*_saved_model/*_full_integer_quant.tflite 2>/dev/null)
  [ -n "$TFL_FILES" ] || warn "No .tflite models under $MODELS_DIR/*_saved_model — run export_rknn_host.sh --tflite on the PC first"
  run_bm() { # label file extra_flags
    local label="$1" f="$2" flags="$3" m; m=$(basename "$f" .tflite | sed 's/_.*//')
    local OUT; OUT=$(ash "cd $DEV_DIR && ./benchmark_model --graph=$DEV_DIR/$(basename "$f") --num_runs=50 --warmup_runs=5 $flags" 2>&1 | tr -d '\r')
    echo "$OUT" > "$LOGDIR/${label}_$(basename "$f" .tflite).log"
    local avg init first; avg=$(echo "$OUT" | grep -oE "Inference \(avg\): [0-9.]+" | grep -oE "[0-9.]+$"); init=$(echo "$OUT" | grep -oE "Init: [0-9.]+" | grep -oE "[0-9.]+"); first=$(echo "$OUT" | grep -oE "First inference: [0-9.]+" | grep -oE "[0-9.]+")
    [ -z "$avg" ] && avg=$(echo "$OUT" | grep -oE "Inference: [0-9.]+" | grep -oE "[0-9.]+")   # legacy format
    if [ -n "$avg" ]; then
      local ms fps; ms=$(awk "BEGIN{printf \"%.2f\", $avg/1000}"); fps=$(awk "BEGIN{printf \"%.2f\", 1000000/$avg}")
      local dl; dl=$(echo "$OUT" | grep -oE "(completely|partially) executed by the delegate[^.]*|will not be executed by the delegate" | head -1)
      pass "[$label/$(basename "$f")] ${ms}ms avg (${fps} FPS) | init $(awk "BEGIN{printf \"%.1f\", ${init:-0}/1000}")ms | first $(awk "BEGIN{printf \"%.1f\", ${first:-0}/1000}")ms ${dl:+| $dl}"
      { echo "$label,$m,steady_avg,ms,$ms"; echo "$label,$m,video,fps,$fps"; echo "$label,$m,first_infer,ms,$(awk "BEGIN{printf \"%.1f\", ${first:-0}/1000}")"; } >> "$BENCH_CSV"
    else
      warn "[$label/$(basename "$f")] failed: $(echo "$OUT" | grep -iE "error|fail" | head -1 | cut -c1-140)"
    fi
  }
  for f in $TFL_FILES; do
    $ADB push "$f" "$DEV_DIR/" >/dev/null || { warn "push failed: $f"; continue; }
    run_bm "android_tflite_cpu" "$f" "--num_threads=4 --use_xnnpack=true"
    run_bm "android_tflite_gpu" "$f" "--use_gpu=true --gpu_precision_loss_allowed=true"
    run_bm "android_tflite_nnapi" "$f" "--use_nnapi=true"
  done
fi

section "STEP 2: RKNN NPU via rknn_benchmark (Android build)"
if [ "$RKNN_RT" -eq 1 ]; then
  [ -z "$RKNN_BENCH" ] && [ -x "$SCRIPT_DIR/../rknn_benchmark_build/rknn_benchmark_android" ] && RKNN_BENCH="$SCRIPT_DIR/../rknn_benchmark_build/rknn_benchmark_android"
  if [ -z "$RKNN_BENCH" ] && [ -n "${ANDROID_NDK_PATH:-}" ]; then
    info "Building rknn_benchmark for Android with NDK at $ANDROID_NDK_PATH ..."
    bash "$SCRIPT_DIR/../tools/build_rknn_benchmark.sh" android "$WORKDIR/rknn_benchmark_build" > "$LOGDIR/build_rknn_benchmark_android.log" 2>&1 && RKNN_BENCH="$WORKDIR/rknn_benchmark_build/rknn_benchmark_android"
  fi
  if [ -n "$RKNN_BENCH" ] && [ -f "$RKNN_BENCH" ]; then
    $ADB push "$RKNN_BENCH" "$DEV_DIR/rknn_benchmark" >/dev/null && ash "chmod +x $DEV_DIR/rknn_benchmark"
    ALLMASK=7; [ "$NPU_CORES" -eq 2 ] && ALLMASK=3
    found=0
    for d in "$MODELS_DIR"/*_"${SOC}"_fp16_rknn_model "$MODELS_DIR"/*_"${SOC}"_int8_rknn_model; do
      f=$(ls "$d"/*.rknn 2>/dev/null | head -1); [ -n "$f" ] || continue; found=1
      m=$(basename "$d" | sed "s/_${SOC}_.*//"); p=$(basename "$d" | grep -oE "fp16|int8")
      $ADB push "$f" "$DEV_DIR/" >/dev/null
      for mask in 0 $ALLMASK; do
        OUT=$(ash "cd $DEV_DIR && LD_LIBRARY_PATH=/vendor/lib64 ./rknn_benchmark $DEV_DIR/$(basename "$f") '' 50 $mask" 2>&1 | tr -d '\r')
        echo "$OUT" > "$LOGDIR/rknn_benchmark_${p}_${m}_mask${mask}.log"
        avg=$(echo "$OUT" | grep -oE "Avg Time [0-9.]+ms" | grep -oE "[0-9.]+" | head -1); fps=$(echo "$OUT" | grep -oE "Avg FPS = [0-9.]+" | grep -oE "[0-9.]+")
        if [ -n "$avg" ]; then
          pass "[android_rknn_npu_${p}_mask${mask}/$m] ${avg}ms avg, ${fps} FPS — $(echo "$OUT" | grep -oE 'rknn_api/rknnrt version: [^,]+, driver version: [^ ]+' | head -1)"
          { echo "android_rknn_npu_${p}_mask${mask},$m,steady_avg,ms,$avg"; echo "android_rknn_npu_${p}_mask${mask},$m,video,fps,$fps"; } >> "$BENCH_CSV"
        else
          fail "[android_rknn_npu_${p}_mask${mask}/$m] $(echo "$OUT" | grep -iE 'error|fail|version' | head -1 | cut -c1-160)"
        fi
      done
    done
    [ "$found" -eq 1 ] || warn "No *_${SOC}_{fp16,int8}_rknn_model dirs under $MODELS_DIR — run export_rknn_host.sh --target $SOC on the PC"
  else
    warn "No Android rknn_benchmark binary. Build one: export ANDROID_NDK_PATH=...; ./tools/build_rknn_benchmark.sh android  (then re-run, or pass --rknn-benchmark)"
  fi
else
  warn "Skipping NPU: firmware has no librknnrt.so"
fi

section "SUMMARY"
echo -e "${GREEN}Passed: $(grep -c '^\[PASS\]' "$RESULTS_FILE")${NC}  ${YELLOW}Warnings: $(grep -c '^\[WARN\]' "$RESULTS_FILE")${NC}  ${RED}Failed: $(grep -c '^\[FAIL\]' "$RESULTS_FILE")${NC}\n"; cat "$RESULTS_FILE"
echo -e "\n${BLUE}  INFERENCE-ONLY THROUGHPUT (benchmark_model / rknn_benchmark; no pre/post-processing, no NMS)${NC}"
printf "%-34s %-10s %12s %12s\n" "BACKEND" "MODEL" "MS" "FPS"
grep ",steady_avg,ms," "$BENCH_CSV" | sort -t, -k2,2 -k1,1 | while IFS=, read -r b m s met v; do
  fps=$(grep "^$b,$m,video,fps," "$BENCH_CSV" | cut -d, -f5); printf "%-34s %-10s %12s %12s\n" "$b" "$m" "$v" "${fps:-}"; done
echo -e "\nCSV: $BENCH_CSV | logs: $LOGDIR"
