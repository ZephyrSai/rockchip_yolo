#!/usr/bin/env bash
#
# test_yolo_rockchip.sh — run ON THE BOARD (RK3588 / RK3588S / RK3576, Linux)
#
# Full diagnostic + benchmark suite for Ultralytics YOLO (YOLO11 + YOLO26)
# on Rockchip RK3588 / RK3576 SoCs: the RKNPU (vendor rknpu driver +
# librknnrt via rknn-toolkit-lite2, or the mainline "rocket" driver via
# Mesa Teflon), the Mali GPU (NCNN Vulkan, OpenCV DNN OpenCL), and the
# CPU (PyTorch, NCNN, ONNX Runtime, TFLite/XNNPACK). It tells you what
# actually works on your board/kernel and how fast it runs.
#
#   SoC      NPU               GPU              Vendor kernel   Mainline
#   RK3588   3 cores, 6 TOPS   Mali-G610 MP4    5.10 / 6.1      rocket NPU (6.18+), panthor GPU (6.10+)
#   RK3576   2 cores, 6 TOPS   Mali-G52 MC3     6.1             no NPU yet; panfrost GPU
#
# Model coverage: YOLO11 + YOLO26, nano/small/medium (--quick = nano only),
# RKNN in FP16 and INT8, NCNN, ONNX, TFLite fp32/int8.
#
# RKNN models must be exported on an x86 Linux PC with export_rknn_host.sh
# (Ultralytics: "exporting on Rockchip-based devices (ARM64) is not
# supported"). Pass the tarball with --models. If nothing is passed, this
# script tries export_rknn_host.sh locally in a separate venv (rknn-toolkit2
# ships aarch64 wheels since 2.3.0) and warns if that fails.
#
# Tests, in order:
#   0. System checks: SoC/board from device-tree, kernel (vendor vs mainline),
#      rknpu module + version (debugfs/dmesg), NPU devfreq/governor,
#      librknnrt.so version match, rocket/accel node, GPU driver
#      (bifrost_kbase/panthor/panfrost), libmali/OpenCL (clinfo), Vulkan
#      (vulkaninfo), render node permissions, Python version constraints
#   1. Python env (Python <= 3.12 for rknn-toolkit-lite2) + assets/models
#   2. Runtime visibility: rknnlite, ncnn (Vulkan device count), OpenCV OpenCL,
#      TFLite runtime, Teflon delegate
#   3. Models: unpack --models tarball, or export locally (NCNN/ONNX always;
#      RKNN best-effort)
#   4. FULL BENCHMARK (Ultralytics end-to-end, with NMS): pytorch_cpu,
#      rknn_npu_fp16, rknn_npu_int8, ncnn_cpu, ncnn_vulkan, onnxrt_cpu, tflite_cpu
#   5. Raw NPU characterisation per core mask (rknn-toolkit-lite2, no NMS)
#   6. Rockchip's rknn_benchmark tool (built on the fly from 6 source files)
#   7. TFLite raw: XNNPACK CPU, Teflon NPU delegate (mainline), OpenCV DNN OpenCL
#   8. NPU diagnosis with the exact fix
#   9. Summary tables + env file
#
# Usage:
#   ./test_yolo_rockchip.sh --models rknn_models_rk3588.tar.gz   # models from the PC export
#   ./test_yolo_rockchip.sh --quick                              # nano only
#   ./test_yolo_rockchip.sh --skip-install
#   ./test_yolo_rockchip.sh --python python3.12                  # pick the interpreter for the venv
#
# Everything lives under ~/yolo-rockchip-test (venv, assets, logs, CSV).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$HOME/yolo-rockchip-test"
VENV_DIR="$WORKDIR/venv"
LOGDIR="$WORKDIR/logs"
RESULTS_FILE="$WORKDIR/results_summary.txt"
BENCH_CSV="$WORKDIR/benchmark_results.csv"
SKIP_INSTALL=0
QUICK_MODE=0
MODELS_IN=""
PYBIN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --skip-install) SKIP_INSTALL=1 ;;
    --quick) QUICK_MODE=1 ;;
    --models) MODELS_IN="$2"; shift ;;
    --python) PYBIN="$2"; shift ;;
    -h|--help) sed -n '2,55p' "$0"; exit 0 ;;
  esac
  shift
done

mkdir -p "$WORKDIR" "$LOGDIR"
: > "$RESULTS_FILE"
echo "backend,model,stage,metric,value_ms_or_fps" > "$BENCH_CSV"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
section() { echo -e "\n${BLUE}==================================================================${NC}"; echo -e "${BLUE}  $1${NC}"; echo -e "${BLUE}==================================================================${NC}"; }
pass() { echo -e "${GREEN}[PASS]${NC} $1"; echo "[PASS] $1" >> "$RESULTS_FILE"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; echo "[FAIL] $1" >> "$RESULTS_FILE"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; echo "[WARN] $1" >> "$RESULTS_FILE"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
sudo_cat() { cat "$1" 2>/dev/null || sudo -n cat "$1" 2>/dev/null; }

# ==================================================================
section "STEP 0: System / driver sanity checks"
# ==================================================================

echo "Kernel: $(uname -r)  arch: $(uname -m)" | tee "$LOGDIR/kernel_version.log"
[ "$(uname -m)" = "aarch64" ] || fail "Not aarch64 — this script runs on the Rockchip board itself"

COMPAT=$(tr '\0' '\n' < /proc/device-tree/compatible 2>/dev/null)
BOARD=$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)
echo "Board: ${BOARD:-unknown}" | tee "$LOGDIR/board.log"
echo "$COMPAT" >> "$LOGDIR/board.log"
SOC="unknown"; NPU_CORES=0; GPU_NAME="unknown"
if echo "$COMPAT" | grep -qE "^rockchip,rk3588s?$"; then
  SOC="rk3588"; NPU_CORES=3; GPU_NAME="Mali-G610 MP4"
  echo "$COMPAT" | grep -q "rk3588s" && SOC_VARIANT="rk3588s" || SOC_VARIANT="rk3588"
elif echo "$COMPAT" | grep -q "^rockchip,rk3576$"; then
  SOC="rk3576"; NPU_CORES=2; GPU_NAME="Mali-G52 MC3"; SOC_VARIANT="rk3576"
fi
if [ "$SOC" != "unknown" ]; then
  pass "SoC: ${SOC_VARIANT} (${NPU_CORES} NPU cores, ${GPU_NAME}) — board: ${BOARD:-?}"
else
  fail "Could not identify RK3588/RK3576 from device-tree compatible: $(echo "$COMPAT" | tr '\n' ' ')"
  SOC="rk3588"; SOC_VARIANT="rk3588"; NPU_CORES=3
fi
# Ultralytics is_rockchip() takes the LAST compatible field and matches it
# against {rk3588, rk3576, ...}; mainline DTs for RK3588S boards end in
# "rockchip,rk3588s", which fails that check.
LAST_COMPAT=$(echo "$COMPAT" | tail -1 | cut -d, -f2 | cut -d- -f1)
FORCE_RK=""
if echo "$LAST_COMPAT" | grep -qE "^(rk3588|rk3576|rk3566|rk3568|rk3562)$"; then
  pass "Ultralytics is_rockchip() will pass (last compatible = $LAST_COMPAT)"
else
  warn "Ultralytics is_rockchip() will FAIL (last compatible = '$LAST_COMPAT', e.g. mainline rk3588s DT) — harness will bypass it with --force-rockchip"
  FORCE_RK="--force-rockchip"
fi

echo -e "\nKernel flavour:"
KVER=$(uname -r | grep -oE '^[0-9]+\.[0-9]+'); KMAJ=${KVER%%.*}; KMIN=${KVER##*.}
kver_ge() { [ "$KMAJ" -gt "$1" ] || { [ "$KMAJ" -eq "$1" ] && [ "$KMIN" -ge "$2" ]; }; }
if [ -d /sys/module/rknpu ] || lsmod | grep -qw rknpu || modinfo rknpu >/dev/null 2>&1; then
  KFLAVOUR="vendor"; pass "Vendor (BSP) kernel $KVER with rknpu driver present"
elif [ -d /sys/module/rocket ] || lsmod | grep -qw rocket || modinfo rocket >/dev/null 2>&1; then
  KFLAVOUR="mainline-rocket"; pass "Mainline kernel $KVER with the rocket NPU driver"
else
  KFLAVOUR="mainline-nonpu"; warn "Kernel $KVER has neither rknpu (vendor) nor rocket (mainline 6.18+) — NPU unavailable on this kernel"
fi

echo -e "\nVendor NPU driver (rknpu):"
RKNPU_DRV_VER=""
if [ "$KFLAVOUR" = "vendor" ]; then
  v=$(sudo_cat /sys/kernel/debug/rknpu/version || sudo_cat /proc/rknpu/version)
  [ -z "$v" ] && v=$(dmesg 2>/dev/null | grep -oE "Initialized rknpu: v[0-9.]+" | head -1)
  RKNPU_DRV_VER=$(echo "$v" | grep -oE "v[0-9]+\.[0-9]+\.[0-9]+" | head -1)
  if [ -n "$RKNPU_DRV_VER" ]; then
    pass "rknpu driver version: $RKNPU_DRV_VER"
  else
    warn "rknpu present but version unreadable (debugfs needs root: sudo cat /sys/kernel/debug/rknpu/version)"
  fi
  load=$(sudo_cat /sys/kernel/debug/rknpu/load || sudo_cat /proc/rknpu/load)
  [ -n "$load" ] && info "NPU load now: $load"
  [ -e /dev/rknpu ] && info "/dev/rknpu present (DMA_HEAP build)" || info "no /dev/rknpu (DRM_GEM build exposes a render node instead — normal)"
  NPU_DEVFREQ=""
  for d in /sys/class/devfreq/*; do
    [ -e "$d/device/of_node/compatible" ] || continue
    if tr -d '\0' < "$d/device/of_node/compatible" | grep -qi rknpu; then NPU_DEVFREQ="$d"; fi
  done
  if [ -n "$NPU_DEVFREQ" ]; then
    pass "NPU devfreq: $NPU_DEVFREQ governor=$(cat "$NPU_DEVFREQ/governor" 2>/dev/null) cur=$(cat "$NPU_DEVFREQ/cur_freq" 2>/dev/null)Hz max=$(cat "$NPU_DEVFREQ/max_freq" 2>/dev/null)Hz"
    info "For peak numbers: echo performance | sudo tee $NPU_DEVFREQ/governor"
  else
    warn "No NPU devfreq node found (frequency scaling not visible)"
  fi
fi

echo -e "\nlibrknnrt.so (RKNN user-space runtime):"
LIBRKNNRT=""
for c in /usr/lib/librknnrt.so /usr/lib/aarch64-linux-gnu/librknnrt.so /usr/local/lib/librknnrt.so; do [ -f "$c" ] && { LIBRKNNRT="$c"; break; }; done
[ -z "$LIBRKNNRT" ] && LIBRKNNRT=$(ldconfig -p 2>/dev/null | awk '/librknnrt.so/{print $NF; exit}')
LIBRKNNRT_VER=""
if [ -n "$LIBRKNNRT" ]; then
  LIBRKNNRT_VER=$(strings "$LIBRKNNRT" 2>/dev/null | grep -oE "librknnrt version: [0-9.]+" | head -1 | awk '{print $3}')
  pass "librknnrt: $LIBRKNNRT (version ${LIBRKNNRT_VER:-unknown})"
  if [ -n "$RKNPU_DRV_VER" ] && [ -n "$LIBRKNNRT_VER" ]; then
    info "runtime $LIBRKNNRT_VER on driver $RKNPU_DRV_VER — rknn-toolkit-lite2 must be the same 2.x minor as librknnrt (SDK 2.3.2 needs driver >= 0.9.6)"
  fi
else
  if [ "$KFLAVOUR" = "vendor" ]; then
    warn "librknnrt.so not installed — Radxa: sudo apt install rknpu2-rk3588 (or rknpu2-rk356x); others: copy rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so from airockchip/rknn-toolkit2 to /usr/lib"
  else
    info "librknnrt.so not installed (only needed on vendor kernels)"
  fi
fi

echo -e "\nMainline NPU driver (rocket / Teflon):"
if [ "$KFLAVOUR" = "mainline-rocket" ]; then
  ls /dev/accel/accel* >/dev/null 2>&1 && pass "accel node: $(ls /dev/accel/accel* | tr '\n' ' ')" || warn "rocket loaded but no /dev/accel/accel0 — board DT must enable rknn_core_0/1/2 (status=okay)"
  [ "$SOC" = "rk3576" ] && warn "rocket supports RK3588 only today; RK3576 mainline NPU is work-in-progress"
fi
TEFLON_LIB="${TEFLON_LIB:-}"
for c in /usr/lib/aarch64-linux-gnu/libteflon.so /usr/local/lib/libteflon.so /usr/lib/libteflon.so "$HOME/mesa/build/src/gallium/targets/teflon/libteflon.so"; do
  [ -z "$TEFLON_LIB" ] && [ -f "$c" ] && TEFLON_LIB="$c"
done
[ -n "$TEFLON_LIB" ] && pass "Teflon TFLite delegate found: $TEFLON_LIB" || info "libteflon.so not found (Mesa 25.3+ built with -Dgallium-drivers=rocket -Dteflon=true); set TEFLON_LIB=/path to use it"

echo -e "\nGPU driver / OpenCL / Vulkan:"
GPU_KMOD="none"
for m in bifrost_kbase mali_kbase panthor panfrost; do lsmod | grep -qw "$m" && GPU_KMOD="$m"; done
[ "$GPU_KMOD" != "none" ] && pass "GPU kernel driver: $GPU_KMOD" || warn "No Mali kernel driver loaded (bifrost_kbase / panthor / panfrost)"
RENDER_NODES=$(ls /dev/dri/renderD* 2>/dev/null | tr '\n' ' ')
[ -n "$RENDER_NODES" ] && pass "render nodes: $RENDER_NODES" || warn "no /dev/dri/renderD* nodes"
for rn in $RENDER_NODES; do [ -r "$rn" ] && [ -w "$rn" ] || warn "$rn not accessible by $USER — sudo usermod -aG render,video \$USER"; done
groups | grep -qE '\b(render|video)\b' && pass "User in render/video group" || warn "User not in render/video group"
dpkg -l 2>/dev/null | grep -qE "^ii\s+libmali" && pass "libmali package: $(dpkg -l | awk '/^ii\s+libmali/{print $2}' | head -1)" || info "no libmali apt package (OpenCL needs the Rockchip libmali blob — see README)"
ls /etc/OpenCL/vendors/*.icd >/dev/null 2>&1 && pass "OpenCL ICD registered: $(cat /etc/OpenCL/vendors/*.icd | tr '\n' ' ')" || info "no OpenCL ICD in /etc/OpenCL/vendors"
OPENCL_OK=0
if command -v clinfo >/dev/null 2>&1; then
  clinfo > "$LOGDIR/clinfo.log" 2>&1
  if grep -qiE "Device Name.*Mali" "$LOGDIR/clinfo.log"; then
    pass "clinfo: $(grep -m1 -iE 'Device Name' "$LOGDIR/clinfo.log" | sed 's/.*Device Name[[:space:]]*//') $(grep -m1 -iE 'Device Version' "$LOGDIR/clinfo.log" | sed 's/.*Device Version[[:space:]]*//')"
    OPENCL_OK=1
  else
    warn "clinfo lists no Mali OpenCL device (see $LOGDIR/clinfo.log)"
  fi
else
  info "clinfo not installed (sudo apt install clinfo)"
fi
VULKAN_OK=0
if command -v vulkaninfo >/dev/null 2>&1; then
  vulkaninfo --summary > "$LOGDIR/vulkaninfo.log" 2>&1
  if grep -qiE "deviceName.*(Mali|panvk|G610|G52)" "$LOGDIR/vulkaninfo.log"; then
    pass "Vulkan: $(grep -m1 -iE 'deviceName' "$LOGDIR/vulkaninfo.log" | sed 's/.*= *//')"
    VULKAN_OK=1
  else
    warn "vulkaninfo shows no Mali device (see $LOGDIR/vulkaninfo.log) — mainline needs panthor/panfrost + Mesa PanVK; libmali blobs are OpenCL/GLES only"
  fi
else
  info "vulkaninfo not installed (sudo apt install vulkan-tools) — ncnn will still probe Vulkan itself"
fi

# ==================================================================
section "STEP 1: Python environment setup"
# ==================================================================
# rknn-toolkit-lite2 has wheels for CPython 3.7-3.12 only.
if [ -z "$PYBIN" ]; then
  for c in python3 python3.12 python3.11 python3.10; do
    command -v "$c" >/dev/null 2>&1 || continue
    v=$("$c" -c 'import sys;print(sys.version_info.minor)')
    if [ "$v" -le 12 ] && [ "$v" -ge 8 ]; then PYBIN="$c"; break; fi
  done
fi
if [ -z "$PYBIN" ]; then
  PYBIN=python3
  warn "No Python 3.8-3.12 found — rknn-toolkit-lite2 has no wheel for $(python3 --version); RKNN backend will be unavailable. Install python3.12 and pass --python python3.12"
fi
info "Using $PYBIN ($($PYBIN --version 2>&1))"

if [ "$SKIP_INSTALL" -eq 0 ]; then
  [ -d "$VENV_DIR" ] || "$PYBIN" -m venv "$VENV_DIR" || { fail "venv creation failed — sudo apt install python3-venv"; exit 1; }
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  pip install -q --upgrade pip
  info "Installing ultralytics (pulls torch aarch64 CPU wheels; needs Python >= 3.10 and glibc >= 2.28)..."
  pip install -q ultralytics 2>"$LOGDIR/pip_ultralytics.log" && pass "ultralytics installed" || fail "ultralytics install failed — see $LOGDIR/pip_ultralytics.log"
  info "Installing rknn-toolkit-lite2 (NPU runtime, Python)..."
  pip install -q rknn-toolkit-lite2 2>"$LOGDIR/pip_rknnlite.log" && pass "rknn-toolkit-lite2 installed" || warn "rknn-toolkit-lite2 install failed (Python > 3.12?) — see $LOGDIR/pip_rknnlite.log"
  info "Installing ncnn, onnxruntime, TFLite runtime..."
  pip install -q ncnn 2>"$LOGDIR/pip_ncnn.log" && pass "ncnn installed" || warn "ncnn install failed"
  pip install -q onnxruntime 2>"$LOGDIR/pip_ort.log" && pass "onnxruntime installed" || warn "onnxruntime install failed"
  pip install -q ai-edge-litert 2>"$LOGDIR/pip_tflite.log" || pip install -q tflite-runtime 2>>"$LOGDIR/pip_tflite.log" && pass "TFLite runtime installed" || warn "No TFLite runtime wheel for this Python — TFLite rows will be skipped"
else
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  info "Skipping installs (--skip-install)"
fi
echo -e "\nInstalled versions:"
python3 - <<'PYEOF' 2>&1 | tee "$LOGDIR/versions.log"
import importlib
for m in ("torch", "ultralytics", "rknnlite.api", "ncnn", "onnxruntime", "ai_edge_litert", "tflite_runtime", "cv2", "numpy"):
    try:
        mod = importlib.import_module(m)
        print(f"  {m:<16} {getattr(mod, '__version__', 'ok')}")
    except Exception as e:
        print(f"  {m:<16} NOT AVAILABLE ({str(e)[:60]})")
PYEOF

TEST_IMG="$WORKDIR/bus.jpg"
[ -f "$TEST_IMG" ] || curl -sL -o "$TEST_IMG" https://raw.githubusercontent.com/ultralytics/assets/main/im/bus.jpg
[ -s "$TEST_IMG" ] && pass "Test image ready" || fail "Test image download failed"
TEST_VIDEO="$WORKDIR/solutions_ci_demo.mp4"
[ -f "$TEST_VIDEO" ] || curl -sL -o "$TEST_VIDEO" https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4
[ -s "$TEST_VIDEO" ] && { pass "Test video ready"; VIDEO_ARG="$TEST_VIDEO"; } || { fail "Test video download failed"; VIDEO_ARG=""; rm -f "$TEST_VIDEO"; }

declare -a BENCH_MODEL_NAMES=("yolo11n" "yolo11s" "yolo11m" "yolo26n" "yolo26s" "yolo26m")
[ "$QUICK_MODE" -eq 1 ] && BENCH_MODEL_NAMES=("yolo11n" "yolo26n")
for m in "${BENCH_MODEL_NAMES[@]}"; do
  [ -f "$WORKDIR/$m.pt" ] || ( cd "$WORKDIR" && python3 -c "from ultralytics import YOLO; YOLO('$m.pt')" >"$LOGDIR/download_$m.log" 2>&1 )
  [ -f "$WORKDIR/$m.pt" ] && pass "weights: $m.pt" || fail "could not download $m.pt"
done

# ==================================================================
section "STEP 2: Runtime visibility — rknnlite, ncnn/Vulkan, OpenCV OpenCL, TFLite"
# ==================================================================
RKNN_OK=0
OUT=$(python3 - <<'PYEOF' 2>&1
try:
    from rknnlite.api import RKNNLite
    import rknnlite
    print(f"RKNNLITE_VERSION={getattr(rknnlite, '__version__', 'unknown')}")
    masks = [m for m in ("NPU_CORE_AUTO","NPU_CORE_0","NPU_CORE_1","NPU_CORE_2","NPU_CORE_0_1","NPU_CORE_0_1_2") if hasattr(RKNNLite, m)]
    print(f"RKNNLITE_MASKS={','.join(masks)}")
    print("RKNNLITE_OK=1")
except Exception as e:
    print(f"RKNNLITE_ERROR={e}")
PYEOF
)
echo "$OUT" > "$LOGDIR/rknnlite_probe.log"
if echo "$OUT" | grep -q "RKNNLITE_OK=1"; then
  RKNNLITE_VER=$(echo "$OUT" | grep RKNNLITE_VERSION | cut -d= -f2)
  pass "rknn-toolkit-lite2 importable (version $RKNNLITE_VER; masks: $(echo "$OUT" | grep RKNNLITE_MASKS | cut -d= -f2))"
  if [ "$KFLAVOUR" = "vendor" ] && [ -n "$LIBRKNNRT" ]; then RKNN_OK=1; else warn "rknnlite importable but no vendor rknpu driver + librknnrt on this kernel — RKNN runs will fail"; fi
  if [ -n "$LIBRKNNRT_VER" ] && [ "${RKNNLITE_VER%.*}" != "${LIBRKNNRT_VER%.*}" ]; then
    warn "rknn-toolkit-lite2 $RKNNLITE_VER vs librknnrt $LIBRKNNRT_VER minor mismatch — models built with toolkit2 $RKNNLITE_VER may refuse to load; align librknnrt (see README)"
  fi
else
  warn "rknn-toolkit-lite2 not importable: $(echo "$OUT" | grep RKNNLITE_ERROR | cut -d= -f2- | cut -c1-120)"
fi

NCNN_GPU=0
OUT=$(python3 -c "
import ncnn
print('NCNN_VERSION=' + getattr(ncnn, '__version__', '?'))
try:
    n = ncnn.get_gpu_count()
    print(f'NCNN_GPU_COUNT={n}')
    if n > 0:
        print('NCNN_GPU_NAME=' + ncnn.get_gpu_info(0).device_name())
except Exception as e:
    print(f'NCNN_GPU_ERROR={e}')
" 2>&1); echo "$OUT" > "$LOGDIR/ncnn_probe.log"
if echo "$OUT" | grep -q "NCNN_GPU_COUNT=[1-9]"; then
  pass "ncnn sees a Vulkan GPU: $(echo "$OUT" | grep NCNN_GPU_NAME | cut -d= -f2-)"; NCNN_GPU=1
elif echo "$OUT" | grep -q "NCNN_VERSION"; then
  warn "ncnn importable but no Vulkan device (ncnn $(echo "$OUT" | grep NCNN_VERSION | cut -d= -f2)) — needs Mesa PanVK (mainline panthor/panfrost) or a Vulkan-capable ICD; NCNN will run on CPU only"
else
  warn "ncnn not importable"
fi

OPENCV_OCL=$(python3 -c "import cv2; print(int(cv2.ocl.haveOpenCL()))" 2>/dev/null || echo 0)
[ "$OPENCV_OCL" = "1" ] && pass "OpenCV sees OpenCL (cv2.ocl.haveOpenCL)" || info "OpenCV has no OpenCL device (needs libmali ICD) — OpenCV DNN will use CPU"

TFLITE_OK=$(python3 -c "
try:
    import ai_edge_litert.interpreter; print(1)
except Exception:
    try:
        import tflite_runtime.interpreter; print(1)
    except Exception: print(0)" 2>/dev/null || echo 0)
[ "$TFLITE_OK" = "1" ] && pass "TFLite runtime importable" || warn "No TFLite runtime — TFLite rows skipped"

# ==================================================================
section "STEP 3: Models — unpack --models tarball or export locally"
# ==================================================================
if [ -n "$MODELS_IN" ]; then
  if [ -f "$MODELS_IN" ]; then
    tar -C "$WORKDIR" -xzf "$MODELS_IN" && pass "Unpacked $MODELS_IN into $WORKDIR" || fail "Could not unpack $MODELS_IN"
  elif [ -d "$MODELS_IN" ]; then
    cp -r "$MODELS_IN"/* "$WORKDIR"/ && pass "Copied models from $MODELS_IN"
  else
    fail "--models path not found: $MODELS_IN"
  fi
fi

have_rknn_models() { ls -d "$WORKDIR"/*_"${SOC}"_fp16_rknn_model >/dev/null 2>&1; }
if ! have_rknn_models; then
  warn "No ${SOC} RKNN models in $WORKDIR. Trying on-board export via export_rknn_host.sh (separate venv; Ultralytics documents this as x86-only, so it may fail)..."
  bash "$SCRIPT_DIR/export_rknn_host.sh" --target "$SOC" --workdir "$WORKDIR" --no-tflite $([ "$QUICK_MODE" -eq 1 ] && echo --quick) > "$LOGDIR/onboard_export.log" 2>&1
  if have_rknn_models; then
    pass "On-board RKNN export worked (aarch64 rknn-toolkit2)"
  else
    warn "On-board RKNN export failed (see $LOGDIR/onboard_export.log). Run ./export_rknn_host.sh --target $SOC on an x86 Linux PC and pass the tarball with --models"
  fi
fi

# NCNN + ONNX exports are architecture-independent: do them here if missing.
( cd "$WORKDIR" && for m in "${BENCH_MODEL_NAMES[@]}"; do
  [ -f "$m.pt" ] || continue
  [ -d "${m}_ncnn_model" ] || python3 -c "from ultralytics import YOLO; YOLO('$m.pt').export(format='ncnn', imgsz=640)" >"$LOGDIR/export_ncnn_$m.log" 2>&1
  [ -f "${m}.onnx" ] || python3 -c "from ultralytics import YOLO; YOLO('$m.pt').export(format='onnx', imgsz=640, dynamic=False, simplify=True)" >"$LOGDIR/export_onnx_$m.log" 2>&1
done )
for m in "${BENCH_MODEL_NAMES[@]}"; do
  [ -d "$WORKDIR/${m}_ncnn_model" ] && pass "NCNN: ${m}_ncnn_model" || warn "NCNN export missing for $m (see $LOGDIR/export_ncnn_$m.log)"
  [ -f "$WORKDIR/${m}.onnx" ] && pass "ONNX: ${m}.onnx" || warn "ONNX export missing for $m"
  for p in fp16 int8; do
    d="$WORKDIR/${m}_${SOC}_${p}_rknn_model"
    ls "$d"/*.rknn >/dev/null 2>&1 && pass "RKNN ${p^^}: $(basename "$d")/$(ls "$d"/*.rknn | head -1 | xargs basename)" || warn "RKNN ${p^^} model missing for $m ($SOC)"
  done
  ls "$WORKDIR/${m}_saved_model"/*.tflite >/dev/null 2>&1 && pass "TFLite: $(ls "$WORKDIR/${m}_saved_model"/*.tflite | xargs -n1 basename | tr '\n' ' ')" || info "no TFLite files for $m (export with --tflite on the PC)"
done

# ==================================================================
section "STEP 4: Full benchmark — every model x every working backend (end-to-end, with NMS)"
# ==================================================================
run_benchmark() {
  local backend_label="$1" device_arg="$2" model_name="$3" model_path="$4" severity="${5:-fail}"
  [ -e "$model_path" ] || { warn "Model $model_path missing, skipping ${backend_label}/${model_name}"; return; }
  info "Benchmarking [$backend_label] model=$model_name ..."
  local logfile="$LOGDIR/bench_${backend_label}_${model_name}.log"
  local OUT
  OUT=$(python3 "$SCRIPT_DIR/bench_harness.py" --model "$model_path" ${device_arg:+--device "$device_arg"} \
        --image "$TEST_IMG" ${VIDEO_ARG:+--video "$VIDEO_ARG"} --backend-label "$backend_label" \
        --img-runs 15 --video-max-frames 300 $FORCE_RK 2>&1)
  echo "$OUT" > "$logfile"
  if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
    local cold first steady_avg steady_p95 vfps vms
    cold=$(echo "$OUT" | grep "^COLD_LOAD_MS=" | cut -d= -f2); first=$(echo "$OUT" | grep "^FIRST_INFER_MS=" | cut -d= -f2)
    steady_avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2); steady_p95=$(echo "$OUT" | grep "^STEADY_P95_MS=" | cut -d= -f2)
    vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2); vms=$(echo "$OUT" | grep "^VIDEO_AVG_MS=" | cut -d= -f2)
    pass "[$backend_label/$model_name] cold-load ${cold}ms | first-infer ${first}ms | steady ${steady_avg}ms (p95 ${steady_p95}ms) | video ${vfps:-N/A} FPS (${vms:-N/A}ms/frame) | dets $(echo "$OUT" | grep '^STEADY_DETECTIONS=' | cut -d= -f2)"
    {
      echo "$backend_label,$model_name,cold_load,ms,$cold"; echo "$backend_label,$model_name,first_infer,ms,$first"
      echo "$backend_label,$model_name,steady_avg,ms,$steady_avg"; echo "$backend_label,$model_name,steady_p95,ms,$steady_p95"
      [ -n "$vfps" ] && echo "$backend_label,$model_name,video,fps,$vfps"; [ -n "$vms" ] && echo "$backend_label,$model_name,video,ms_per_frame,$vms"
    } >> "$BENCH_CSV"
  else
    local err; err=$(echo "$OUT" | grep -E "_ERROR=" | head -1 | cut -c1-180)
    [ "$severity" = "warn" ] && warn "[$backend_label/$model_name] did not run (${err:-see $logfile})" || fail "[$backend_label/$model_name] failed — ${err:-see $logfile}"
  fi
}

for m in "${BENCH_MODEL_NAMES[@]}"; do run_benchmark "pytorch_cpu" "cpu" "$m" "$WORKDIR/$m.pt"; done
if [ "$RKNN_OK" -eq 1 ]; then
  for p in fp16 int8; do for m in "${BENCH_MODEL_NAMES[@]}"; do
    run_benchmark "rknn_npu_${p}" "" "$m" "$WORKDIR/${m}_${SOC}_${p}_rknn_model"
  done; done
else
  warn "Skipping RKNN NPU benchmarks — runtime/driver not usable (Step 2)"
fi
for m in "${BENCH_MODEL_NAMES[@]}"; do run_benchmark "ncnn_cpu" "cpu" "$m" "$WORKDIR/${m}_ncnn_model"; done
if [ "$NCNN_GPU" -eq 1 ]; then
  for m in "${BENCH_MODEL_NAMES[@]}"; do run_benchmark "ncnn_vulkan" "vulkan:0" "$m" "$WORKDIR/${m}_ncnn_model" warn; done
else
  warn "Skipping ncnn Vulkan (Mali GPU) benchmarks — no Vulkan device visible to ncnn"
fi
for m in "${BENCH_MODEL_NAMES[@]}"; do run_benchmark "onnxrt_cpu" "cpu" "$m" "$WORKDIR/${m}.onnx" warn; done
if [ "$TFLITE_OK" = "1" ]; then
  for m in "${BENCH_MODEL_NAMES[@]}"; do
    f=$(ls "$WORKDIR/${m}_saved_model"/*_float32.tflite 2>/dev/null | head -1)
    [ -n "$f" ] && run_benchmark "tflite_cpu_fp32" "cpu" "$m" "$f" warn
    f=$(ls "$WORKDIR/${m}_saved_model"/*_full_integer_quant.tflite 2>/dev/null | head -1)
    [ -n "$f" ] && run_benchmark "tflite_cpu_int8" "cpu" "$m" "$f" warn
  done
fi

# ==================================================================
section "STEP 5: Raw NPU characterisation per core mask (rknn-toolkit-lite2, no NMS)"
# ==================================================================
if [ "$RKNN_OK" -eq 1 ]; then
  MASKS="auto 0 0_1"; [ "$NPU_CORES" -ge 3 ] && MASKS="auto 0 0_1 0_1_2"
  for p in fp16 int8; do for m in yolo11n yolo26n; do
    d="$WORKDIR/${m}_${SOC}_${p}_rknn_model"; ls "$d"/*.rknn >/dev/null 2>&1 || continue
    for mask in $MASKS; do
      label="rknn_raw_${p}_core${mask}"
      OUT=$(python3 "$SCRIPT_DIR/rknn_raw_bench.py" --model "$d" --core-mask "$mask" ${VIDEO_ARG:+--video "$VIDEO_ARG"} --runs 15 --video-max-frames 300 2>&1)
      echo "$OUT" > "$LOGDIR/${label}_${m}.log"
      if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
        avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2); init=$(echo "$OUT" | grep "^INIT_RUNTIME_MS=" | cut -d= -f2); vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
        pass "[$label/$m] init_runtime ${init}ms | raw infer ${avg}ms | video ${vfps:-N/A} FPS $(echo "$OUT" | grep '^NPU_LOAD_SAMPLE=' | cut -d= -f2- | sed 's/^/| /')"
        { echo "$label,$m,first_infer,ms,$init"; echo "$label,$m,steady_avg,ms,$avg"; [ -n "$vfps" ] && echo "$label,$m,video,fps,$vfps"; } >> "$BENCH_CSV"
        [ -z "$(echo "$OUT" | grep '^SDK_VERSION=')" ] || info "$(echo "$OUT" | grep '^SDK_VERSION=')"
      else
        fail "[$label/$m] $(echo "$OUT" | grep -E '_ERROR=|^ERROR=' | head -1 | cut -c1-180)"
      fi
    done
  done; done
else
  warn "Skipping raw NPU bench"
fi

# ==================================================================
section "STEP 6: Rockchip rknn_benchmark tool"
# ==================================================================
if [ "$RKNN_OK" -eq 1 ]; then
  RB="$WORKDIR/rknn_benchmark_build/rknn_benchmark"
  [ -x "$RB" ] || bash "$SCRIPT_DIR/tools/build_rknn_benchmark.sh" linux "$WORKDIR/rknn_benchmark_build" > "$LOGDIR/build_rknn_benchmark.log" 2>&1
  if [ -x "$RB" ]; then
    pass "rknn_benchmark built: $RB"
    ALLMASK=7; [ "$NPU_CORES" -eq 2 ] && ALLMASK=3
    for p in fp16 int8; do for m in yolo11n yolo26n; do
      f=$(ls "$WORKDIR/${m}_${SOC}_${p}_rknn_model"/*.rknn 2>/dev/null | head -1); [ -n "$f" ] || continue
      for mask in 0 $ALLMASK; do
        OUT=$("$RB" "$f" "" 50 "$mask" 2>&1); echo "$OUT" > "$LOGDIR/rknn_benchmark_${p}_${m}_mask${mask}.log"
        avg=$(echo "$OUT" | grep -oE "Avg Time [0-9.]+ms" | grep -oE "[0-9.]+" | head -1); fps=$(echo "$OUT" | grep -oE "Avg FPS = [0-9.]+" | grep -oE "[0-9.]+")
        if [ -n "$avg" ]; then
          pass "[rknn_benchmark/${p}/mask${mask}/$m] ${avg}ms avg, ${fps} FPS — $(echo "$OUT" | grep -oE 'rknn_api/rknnrt version: [^,]+, driver version: [^ ]+' | head -1)"
          { echo "rknn_benchmark_${p}_mask${mask},$m,steady_avg,ms,$avg"; echo "rknn_benchmark_${p}_mask${mask},$m,video,fps,$fps"; } >> "$BENCH_CSV"
        else
          fail "rknn_benchmark failed for $m ($p, mask $mask) — see $LOGDIR/rknn_benchmark_${p}_${m}_mask${mask}.log"
        fi
      done
    done; done
  else
    warn "rknn_benchmark build failed (needs g++ + zlib1g-dev: sudo apt install build-essential zlib1g-dev) — see $LOGDIR/build_rknn_benchmark.log"
  fi
else
  warn "Skipping rknn_benchmark"
fi

# ==================================================================
section "STEP 7: TFLite raw (XNNPACK / Teflon NPU) + OpenCV DNN OpenCL"
# ==================================================================
run_tflite() { # label model delegate
  local label="$1" f="$2" del="$3"
  local OUT; OUT=$(python3 "$SCRIPT_DIR/tflite_bench.py" --model "$f" --threads 4 ${del:+--delegate "$del"} ${VIDEO_ARG:+--video "$VIDEO_ARG"} --runs 15 --video-max-frames 300 2>&1)
  echo "$OUT" > "$LOGDIR/${label}_$(basename "$f" .tflite).log"
  if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
    avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2); init=$(echo "$OUT" | grep "^INTERP_INIT_MS=" | cut -d= -f2); vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
    pass "[$label/$(basename "$f")] init ${init}ms | raw infer ${avg}ms | video ${vfps:-N/A} FPS (no NMS)"
    { echo "$label,$(basename "$f" .tflite | sed 's/_.*//'),steady_avg,ms,$avg"; [ -n "$vfps" ] && echo "$label,$(basename "$f" .tflite | sed 's/_.*//'),video,fps,$vfps"; } >> "$BENCH_CSV"
  else
    warn "[$label/$(basename "$f")] $(echo "$OUT" | grep -E '_ERROR=|^ERROR=' | head -1 | cut -c1-160)"
  fi
}
if [ "$TFLITE_OK" = "1" ]; then
  for m in yolo11n yolo26n; do
    for f in "$WORKDIR/${m}_saved_model"/*_float32.tflite "$WORKDIR/${m}_saved_model"/*_full_integer_quant.tflite; do
      [ -f "$f" ] || continue
      run_tflite "tflite_raw_cpu" "$f" ""
      if [ -n "$TEFLON_LIB" ] && [[ "$f" == *integer_quant* ]]; then
        run_tflite "tflite_raw_teflon_npu" "$f" "$TEFLON_LIB"
        info "Teflon offloads INT8 convolutions only; YOLO's other ops fall back to CPU, so expect partial gains at best"
      fi
    done
  done
fi

if [ "$OPENCV_OCL" = "1" ] || true; then
  for m in yolo11n yolo26n; do
    [ -f "$WORKDIR/$m.onnx" ] || continue
    for tgt in cpu opencl opencl_fp16; do
      [ "$tgt" != "cpu" ] && [ "$OPENCV_OCL" != "1" ] && continue
      OUT=$(python3 - "$WORKDIR/$m.onnx" "$tgt" ${VIDEO_ARG:-} <<'PYEOF' 2>&1
import sys, time, statistics, cv2, numpy as np
onnx, tgt, video = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
net = cv2.dnn.readNetFromONNX(onnx)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget({"cpu": cv2.dnn.DNN_TARGET_CPU, "opencl": cv2.dnn.DNN_TARGET_OPENCL, "opencl_fp16": cv2.dnn.DNN_TARGET_OPENCL_FP16}[tgt])
blob = np.random.rand(1, 3, 640, 640).astype(np.float32)
net.setInput(blob); t0 = time.perf_counter(); net.forward(); print(f"FIRST_INFER_MS={(time.perf_counter()-t0)*1000:.1f}")
ts = []
for _ in range(15):
    net.setInput(blob); t0 = time.perf_counter(); net.forward(); ts.append((time.perf_counter()-t0)*1000)
print(f"STEADY_AVG_MS={statistics.mean(ts):.2f}")
if video:
    cap = cv2.VideoCapture(video); n = 0; t0 = time.perf_counter()
    while n < 300:
        ok, fr = cap.read()
        if not ok: break
        net.setInput(cv2.dnn.blobFromImage(fr, 1/255.0, (640, 640), swapRB=True)); net.forward(); n += 1
    if n: print(f"VIDEO_AVG_FPS={n/(time.perf_counter()-t0):.2f}")
PYEOF
)
      echo "$OUT" > "$LOGDIR/opencv_dnn_${tgt}_$m.log"
      if echo "$OUT" | grep -q "^STEADY_AVG_MS="; then
        avg=$(echo "$OUT" | grep "^STEADY_AVG_MS=" | cut -d= -f2); vfps=$(echo "$OUT" | grep "^VIDEO_AVG_FPS=" | cut -d= -f2)
        pass "[opencv_dnn_${tgt}/$m] raw infer ${avg}ms | video ${vfps:-N/A} FPS (no NMS)"
        { echo "opencv_dnn_${tgt},$m,steady_avg,ms,$avg"; [ -n "$vfps" ] && echo "opencv_dnn_${tgt},$m,video,fps,$vfps"; } >> "$BENCH_CSV"
      else
        warn "[opencv_dnn_${tgt}/$m] failed: $(echo "$OUT" | grep -iE 'error' | head -1 | cut -c1-140)"
      fi
    done
  done
fi

# ==================================================================
section "STEP 8: NPU diagnosis"
# ==================================================================
if grep -q "^\[PASS\] \[rknn_npu_" "$RESULTS_FILE"; then
  pass "NPU usable end-to-end via Ultralytics RKNN backend on $SOC_VARIANT (driver ${RKNPU_DRV_VER:-?}, librknnrt ${LIBRKNNRT_VER:-?}, lite2 ${RKNNLITE_VER:-?})"
elif [ "$KFLAVOUR" = "mainline-nonpu" ]; then
  warn "ROOT CAUSE: mainline kernel $KVER without an NPU driver. Options: (a) vendor BSP kernel (5.10/6.1, e.g. Armbian 'vendor' or Radxa/Orange Pi images) for the full RKNN stack; (b) kernel >= 6.18 + Mesa 25.3 Teflon for the open rocket driver (RK3588 only, conv-only offload)"
elif [ "$KFLAVOUR" = "mainline-rocket" ]; then
  warn "rocket (mainline) NPU: RKNN/librknnrt does not work on it. Use TFLite + Teflon delegate (Step 7). Expect YOLO to run mostly on CPU (only INT8 convs are offloaded today)"
elif [ -z "$LIBRKNNRT" ]; then
  warn "ROOT CAUSE: vendor rknpu driver present but librknnrt.so missing — install rknpu2 runtime (Radxa: apt install rknpu2-rk3588 / rknpu2-rk356x; else copy from airockchip/rknn-toolkit2 rknpu2/runtime/Linux/librknn_api/aarch64/)"
elif ! echo "$(cat "$LOGDIR/rknnlite_probe.log")" | grep -q RKNNLITE_OK; then
  warn "ROOT CAUSE: rknn-toolkit-lite2 not installed/importable — needs CPython 3.8-3.12 (you have $($PYBIN --version 2>&1)); re-run with --python python3.12"
elif ! have_rknn_models; then
  warn "ROOT CAUSE: no RKNN models for $SOC — export on an x86 PC: ./export_rknn_host.sh --target $SOC, then --models rknn_models_${SOC}.tar.gz"
elif grep -qE "^\[FAIL\] \[rknn_npu_.*(init_runtime|mismatch|version)" "$RESULTS_FILE"; then
  warn "ROOT CAUSE: runtime/driver version mismatch — librknnrt ${LIBRKNNRT_VER:-?} vs driver ${RKNPU_DRV_VER:-?} vs toolkit ${RKNNLITE_VER:-?}. Update librknnrt.so to the SDK release matching your rknn-toolkit2 (2.3.2 line) and, for driver < 0.9.6, update the kernel/rknpu (dkms: github.com/bmilde/rknpu-driver-dkms)"
else
  warn "NPU path failed for another reason — check $LOGDIR/bench_rknn_npu_*.log and $LOGDIR/rknn_raw_*.log"
fi
if grep -q "^\[PASS\] \[ncnn_vulkan" "$RESULTS_FILE"; then pass "Mali GPU usable via NCNN Vulkan"; else info "Mali GPU compute not used for YOLO — needs Mesa PanVK (mainline) for ncnn Vulkan, or libmali OpenCL for OpenCV DNN"; fi

# ==================================================================
section "SUMMARY"
# ==================================================================
PASS_COUNT=$(grep -c "^\[PASS\]" "$RESULTS_FILE" || true); FAIL_COUNT=$(grep -c "^\[FAIL\]" "$RESULTS_FILE" || true); WARN_COUNT=$(grep -c "^\[WARN\]" "$RESULTS_FILE" || true)
echo -e "\nFull results: $RESULTS_FILE | logs: $LOGDIR | CSV: $BENCH_CSV"
echo -e "${GREEN}Passed: $PASS_COUNT${NC}  ${YELLOW}Warnings: $WARN_COUNT${NC}  ${RED}Failed: $FAIL_COUNT${NC}\n"
cat "$RESULTS_FILE"
table() {
  echo -e "\n${BLUE}--------------------------------------------------------------${NC}\n${BLUE}  $2${NC}\n${BLUE}--------------------------------------------------------------${NC}"
  printf "%-32s %-10s %12s\n" "BACKEND" "MODEL" "$3"
  grep ",$1," "$BENCH_CSV" | sort -t, -k2,2 -k1,1 | while IFS=, read -r b m s met v; do printf "%-32s %-10s %12s\n" "$b" "$m" "$v"; done
}
table "video,fps" "VIDEO FPS (end-to-end for pytorch_/rknn_npu_/ncnn_/onnxrt_/tflite_cpu_; raw for *_raw_*/rknn_benchmark_/opencv_dnn_) — higher is better" "VIDEO_FPS"
table "steady_avg,ms" "STEADY-STATE LATENCY (ms/frame) — lower is better" "MS/FRAME"
table "first_infer,ms" "FIRST INFERENCE / RUNTIME INIT (ms)" "MS"
table "cold_load,ms" "COLD LOAD (ms)" "MS"

ENV_FILE="$WORKDIR/yolo_rockchip_env.sh"
{ echo "# Auto-generated by test_yolo_rockchip.sh on $(date) — $SOC_VARIANT, kernel $(uname -r) ($KFLAVOUR)"; } > "$ENV_FILE"
BEST=$(grep -E "^(pytorch_cpu|rknn_npu_(fp16|int8)|ncnn_cpu|ncnn_vulkan|onnxrt_cpu|tflite_cpu_(fp32|int8)),yolo(11|26)n,video,fps," "$BENCH_CSV" | sort -t, -k5,5 -gr | head -1)
if [ -n "$BEST" ]; then
  bb=$(echo "$BEST" | cut -d, -f1); bm=$(echo "$BEST" | cut -d, -f2); bf=$(echo "$BEST" | cut -d, -f5)
  echo -e "\n${GREEN}RECOMMENDATION:${NC} fastest end-to-end nano backend: $bb ($bm, $bf FPS)"
  case "$bb" in
    rknn_npu_*) p=${bb#rknn_npu_}; echo "export YOLO_MODEL=$WORKDIR/${bm}_${SOC}_${p}_rknn_model   # yolo predict model=\$YOLO_MODEL source=img.jpg" >> "$ENV_FILE" ;;
    ncnn_vulkan) echo "export YOLO_MODEL=$WORKDIR/${bm}_ncnn_model; export YOLO_DEVICE=vulkan:0" >> "$ENV_FILE" ;;
    ncnn_cpu) echo "export YOLO_MODEL=$WORKDIR/${bm}_ncnn_model; export YOLO_DEVICE=cpu" >> "$ENV_FILE" ;;
    onnxrt_cpu) echo "export YOLO_MODEL=$WORKDIR/${bm}.onnx" >> "$ENV_FILE" ;;
    tflite_cpu_*) echo "export YOLO_MODEL=\$(ls $WORKDIR/${bm}_saved_model/*.tflite | head -1)" >> "$ENV_FILE" ;;
    *) echo "export YOLO_MODEL=$WORKDIR/${bm}.pt; export YOLO_DEVICE=cpu" >> "$ENV_FILE" ;;
  esac
  [ -n "${NPU_DEVFREQ:-}" ] && echo "# peak NPU clocks: echo performance | sudo tee $NPU_DEVFREQ/governor" >> "$ENV_FILE"
else
  echo -e "\n${YELLOW}RECOMMENDATION:${NC} no benchmark completed — fix the FAIL items above first."
fi
echo -e "${BLUE}Env file: $ENV_FILE${NC}"; echo "---"; cat "$ENV_FILE"; echo "---"
