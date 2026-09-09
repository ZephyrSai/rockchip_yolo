#!/usr/bin/env bash
#
# export_rknn_host.sh — export YOLO11 / YOLO26 to every format the Rockchip
# suite benchmarks, and pack them into one tarball for the board.
#
# Ultralytics documents RKNN export as x86-64 Linux only ("exporting on
# Rockchip-based devices (ARM64) is not supported"), so the normal flow is:
#
#   PC (x86 Linux):   ./export_rknn_host.sh --target rk3588,rk3576
#                     -> ~/yolo-rockchip-export/rknn_models_rk3588.tar.gz (+ rk3576)
#   board:            ./test_yolo_rockchip.sh --models rknn_models_rk3588.tar.gz
#
# rknn-toolkit2 >= 2.3.0 does publish aarch64 wheels, so test_yolo_rockchip.sh
# will *try* this script on the board when no models were supplied, in a
# separate venv so the toolkit's pinned deps can't disturb the benchmark venv.
#
# Formats produced per model (yolo11n/s/m, yolo26n/s/m; --quick = nano only):
#   <m>_<target>_fp16_rknn_model/   RKNN FP16                 (quantize=16)
#   <m>_<target>_int8_rknn_model/   RKNN INT8, coco8 calib    (quantize=8)
#   <m>_ncnn_model/                 NCNN (CPU + Vulkan on Mali)
#   <m>.onnx                        ONNX (onnxruntime CPU, OpenCV DNN OpenCL)
#   <m>_saved_model/*.tflite        LiteRT/TFLite fp32 + full-int8 (--tflite; default on x86)
#
# Usage:
#   ./export_rknn_host.sh [--target rk3588,rk3576] [--quick] [--tflite|--no-tflite]
#                         [--workdir DIR] [--python python3.12]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$HOME/yolo-rockchip-export"
TARGETS=""
QUICK=0
TFLITE="auto"
PYBIN=""

while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGETS="$2"; shift ;;
    --quick) QUICK=1 ;;
    --tflite) TFLITE=1 ;;
    --no-tflite) TFLITE=0 ;;
    --workdir) WORKDIR="$2"; shift ;;
    --python) PYBIN="$2"; shift ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
  esac
  shift
done

ARCH=$(uname -m)
if [ -z "$TARGETS" ]; then
  if [ "$ARCH" = "x86_64" ]; then
    TARGETS="rk3588,rk3576"
  else
    COMPAT=$(tr '\0' '\n' < /proc/device-tree/compatible 2>/dev/null || true)
    if echo "$COMPAT" | grep -q "rockchip,rk3588"; then TARGETS="rk3588"
    elif echo "$COMPAT" | grep -q "rockchip,rk3576"; then TARGETS="rk3576"
    else TARGETS="rk3588"; fi
  fi
fi
[ "$TFLITE" = "auto" ] && { [ "$ARCH" = "x86_64" ] && TFLITE=1 || TFLITE=0; }

mkdir -p "$WORKDIR/logs"
LOGDIR="$WORKDIR/logs"
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
fail() { echo -e "${RED}[FAIL]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
info() { echo -e "${BLUE}[INFO]${NC} $1"; }

info "Host arch: $ARCH | targets: $TARGETS | tflite: $TFLITE | workdir: $WORKDIR"
[ "$ARCH" != "x86_64" ] && warn "Not x86-64: Ultralytics documents RKNN export as x86-only; trying anyway with the aarch64 rknn-toolkit2 wheel (may fail)."

# ---- Python: rknn-toolkit2 wheels exist for cp36-cp312 only ----
if [ -z "$PYBIN" ]; then
  for c in python3.12 python3.11 python3.10 python3; do
    command -v "$c" >/dev/null 2>&1 || continue
    v=$("$c" -c 'import sys;print(sys.version_info.minor)')
    if [ "$v" -le 12 ] && [ "$v" -ge 8 ]; then PYBIN="$c"; break; fi
  done
fi
[ -n "$PYBIN" ] || { fail "No Python 3.8-3.12 found (rknn-toolkit2 has no 3.13+ wheels). Install python3.12 (e.g. deadsnakes PPA / pyenv) and pass --python"; exit 1; }
info "Using $PYBIN ($($PYBIN --version 2>&1))"

VENV="$WORKDIR/venv-export"
if [ ! -d "$VENV" ]; then
  "$PYBIN" -m venv "$VENV" || { fail "venv creation failed (install python3-venv)"; exit 1; }
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q --upgrade pip
info "Installing ultralytics + rknn-toolkit2 (pins onnx<1.19, setuptools<82 as Ultralytics' exporter requires)..."
pip install -q ultralytics "rknn-toolkit2>=2.3.2" "onnx>=1.16.1,<1.19.0" "setuptools<82" onnxslim 2>"$LOGDIR/pip_export.log" \
  && pass "export venv ready" \
  || { fail "pip install failed — see $LOGDIR/pip_export.log"; exit 1; }
if [ "$TFLITE" -eq 1 ]; then
  pip install -q "tensorflow>=2.13" onnx2tf sng4onnx onnx_graphsurgeon ai-edge-litert 2>>"$LOGDIR/pip_export.log" \
    && pass "tflite export deps installed" \
    || { warn "TFLite export deps failed to install — skipping TFLite (see $LOGDIR/pip_export.log)"; TFLITE=0; }
fi

MODELS=("yolo11n" "yolo11s" "yolo11m" "yolo26n" "yolo26s" "yolo26m")
[ "$QUICK" -eq 1 ] && MODELS=("yolo11n" "yolo26n")

cd "$WORKDIR" || exit 1
for m in "${MODELS[@]}"; do
  [ -f "$m.pt" ] || python3 -c "from ultralytics import YOLO; YOLO('$m.pt')" >"$LOGDIR/download_$m.log" 2>&1
  [ -f "$m.pt" ] && pass "weights: $m.pt" || { fail "could not download $m.pt"; continue; }
done

# One python helper that tries the new (quantize=) then legacy (half/int8)
# export kwargs, and moves the output to a precision/target-tagged dir.
cat > "$WORKDIR/_export.py" <<'PYEOF'
import os, shutil, sys
from ultralytics import YOLO
m, fmt, target, prec, outdir = sys.argv[1:6]
model = YOLO(f"{m}.pt")
kw = dict(format=fmt, imgsz=640, batch=1)
if fmt == "rknn":
    kw["name"] = target
elif fmt == "onnx":
    kw.update(dynamic=False, simplify=True)
tries = []
if prec == "fp16":
    tries = [dict(quantize=16), dict(half=True)]
elif prec == "int8":
    tries = [dict(quantize=8, data="coco8.yaml"), dict(int8=True, data="coco8.yaml")]
else:
    tries = [dict()]
out = None
for extra in tries:
    try:
        out = model.export(**kw, **extra)
        print(f"EXPORT_ARGS={extra}")
        break
    except (TypeError, SyntaxError, KeyError) as e:
        print(f"args {extra} rejected: {e}")
if not out:
    raise SystemExit("EXPORT_ERROR=all arg variants failed")
out = str(out)
if outdir:
    src = out if os.path.isdir(out) else os.path.dirname(out) if fmt in ("rknn", "ncnn", "openvino") else out
    if os.path.abspath(src) != os.path.abspath(outdir):
        if os.path.exists(outdir):
            shutil.rmtree(outdir) if os.path.isdir(outdir) else os.remove(outdir)
        shutil.move(src, outdir)
    print(f"EXPORTED={outdir}")
else:
    print(f"EXPORTED={out}")
PYEOF

run_export() { # model fmt target prec outdir label
  local m="$1" fmt="$2" target="$3" prec="$4" outdir="$5" label="$6"
  if [ -n "$outdir" ] && [ -e "$outdir" ]; then info "exists: $outdir"; return 0; fi
  info "Exporting $label ..."
  python3 "$WORKDIR/_export.py" "$m" "$fmt" "$target" "$prec" "$outdir" >"$LOGDIR/export_${label//\//_}.log" 2>&1
  if grep -q "^EXPORTED=" "$LOGDIR/export_${label//\//_}.log"; then
    pass "$label -> $(grep '^EXPORTED=' "$LOGDIR/export_${label//\//_}.log" | cut -d= -f2-)"
  else
    fail "$label failed — see $LOGDIR/export_${label//\//_}.log ($(grep -m1 -iE 'error' "$LOGDIR/export_${label//\//_}.log" | cut -c1-140))"
  fi
}

IFS=',' read -r -a TARGET_ARR <<< "$TARGETS"
for m in "${MODELS[@]}"; do
  [ -f "$m.pt" ] || continue
  for t in "${TARGET_ARR[@]}"; do
    run_export "$m" rknn "$t" fp16 "$WORKDIR/${m}_${t}_fp16_rknn_model" "$m/rknn/$t/fp16"
    run_export "$m" rknn "$t" int8 "$WORKDIR/${m}_${t}_int8_rknn_model" "$m/rknn/$t/int8"
  done
  run_export "$m" ncnn "" fp32 "$WORKDIR/${m}_ncnn_model" "$m/ncnn"
  run_export "$m" onnx "" fp32 "$WORKDIR/${m}.onnx" "$m/onnx"
  if [ "$TFLITE" -eq 1 ]; then
    # fp32 + int8 both land in <m>_saved_model/; run fp32 first, then int8 adds *_full_integer_quant.tflite
    run_export "$m" tflite "" fp32 "" "$m/tflite/fp32"
    run_export "$m" tflite "" int8 "" "$m/tflite/int8"
    ls "$WORKDIR/${m}_saved_model"/*.tflite >/dev/null 2>&1 && pass "tflite files: $(ls "$WORKDIR/${m}_saved_model"/*.tflite | xargs -n1 basename | tr '\n' ' ')"
  fi
done

# ---- pack per target: RKNN dirs for that target + the shared formats ----
for t in "${TARGET_ARR[@]}"; do
  TAR="$WORKDIR/rknn_models_${t}.tar.gz"
  items=()
  for m in "${MODELS[@]}"; do
    for d in "${m}_${t}_fp16_rknn_model" "${m}_${t}_int8_rknn_model" "${m}_ncnn_model" "${m}.onnx" "${m}_saved_model"; do
      [ -e "$WORKDIR/$d" ] && items+=("$d")
    done
  done
  if [ ${#items[@]} -gt 0 ]; then
    tar -C "$WORKDIR" -czf "$TAR" "${items[@]}" && pass "Packed ${#items[@]} items -> $TAR ($(du -h "$TAR" | cut -f1))"
  else
    warn "Nothing to pack for $t"
  fi
done
echo ""
info "Copy the tarball to the board and run:  ./test_yolo_rockchip.sh --models rknn_models_<target>.tar.gz"
