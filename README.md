# YOLO on Rockchip RK3588 / RK3576 — NPU + Mali GPU + CPU Diagnostic & Benchmark Suite (Linux + Android)

A full test suite for running [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO (YOLO11 and YOLO26) on Rockchip RK3588 / RK3588S / RK3576 boards: the **RKNPU** (vendor `rknpu` driver + `librknnrt` through `rknn-toolkit-lite2`, or the mainline `rocket` driver through Mesa Teflon), the **Mali GPU** (NCNN Vulkan, OpenCV DNN OpenCL), and the **CPU** (PyTorch, NCNN, ONNX Runtime, TFLite/XNNPACK). On **Android** it drives the same board over `adb` with Google's `benchmark_model` and Rockchip's `rknn_benchmark`. It tells you what actually works on your board, kernel and firmware, and how fast it runs.

Sibling suites with the same metrics and CSV schema: [ryzen_yolo](https://github.com/ZephyrSai/ryzen_yolo) (AMD) and [openvino_yolo](https://github.com/ZephyrSai/openvino_yolo) (Intel).

## Supported hardware

| SoC | NPU | GPU | Vendor (BSP) kernel | Mainline |
|---|---|---|---|---|
| RK3588 / RK3588S | 3 cores, 6 TOPS INT8 | Mali-G610 MP4 | 5.10 (rkr8), 6.1 (rkr7) — `rknpu` 0.9.6–0.9.8 | GPU: `panthor` (6.10+) + Mesa PanVK Vulkan. NPU: `rocket` (6.18+) + Mesa 25.3 Teflon, conv-only |
| RK3576 | 2 cores, 6 TOPS INT8 | Mali-G52 MC3 | 6.1 — `rknpu` 0.9.8 | GPU: `panfrost` + PanVK. NPU: not in mainline yet |

Boards: Radxa ROCK 5A/5B/5C/5T, ROCK 4D, Orange Pi 5/5 Plus/5 Max, FriendlyELEC NanoPC-T6 / NanoPi M5, Khadas Edge 2, ArmSoM Sige5, Firefly ROC-RK3588, ASUS Tinker Board 3, Luckfox Omni3576, and any other board exposing `rockchip,rk3588[s]` / `rockchip,rk3576` in its device tree.

## The one thing to know first: RKNN export happens on a PC

Ultralytics documents RKNN export as x86-64 Linux only. So the flow has two halves:

```bash
# 1. On an x86 Linux PC (or WSL2): export everything and pack a tarball per SoC
./export_rknn_host.sh --target rk3588,rk3576        # add --quick for nano only
#    -> ~/yolo-rockchip-export/rknn_models_rk3588.tar.gz  (and rk3576)

# 2. Copy the tarball to the board and run the suite
scp ~/yolo-rockchip-export/rknn_models_rk3588.tar.gz board:
./test_yolo_rockchip.sh --models rknn_models_rk3588.tar.gz
```

`rknn-toolkit2` 2.3.0+ does ship aarch64 wheels, so if you skip step 1 the board script tries the export locally in a separate venv and warns if it fails. NCNN and ONNX exports always happen on the board.

## What it does

### Linux, on the board (`test_yolo_rockchip.sh`)

1. **System checks** — SoC and board from the device tree, whether the kernel is vendor (BSP) or mainline, `rknpu` driver version (debugfs / dmesg), NPU devfreq governor and clocks, `librknnrt.so` version and whether it matches the driver and toolkit, `rocket` / `/dev/accel` for mainline, Mali kernel driver (`bifrost_kbase` / `panthor` / `panfrost`), libmali + OpenCL ICD + `clinfo`, `vulkaninfo`, render-node permissions, and whether Ultralytics' own `is_rockchip()` check will pass on this device tree.
2. **Environment setup** — an isolated venv with Python ≤ 3.12 (a hard requirement of `rknn-toolkit-lite2`), installing Ultralytics (aarch64 CPU torch), `rknn-toolkit-lite2`, `ncnn`, `onnxruntime`, and a TFLite runtime. Downloads the test image, the six models, and the same public test video the other suites use.
3. **Runtime visibility** — `rknnlite` import and available core masks, ncnn Vulkan device count, OpenCV OpenCL, TFLite runtime, Teflon delegate.
4. **Models** — unpacks the tarball (or exports locally), then confirms RKNN FP16/INT8, NCNN, ONNX and TFLite files per model.
5. **Full benchmark**, every model × every backend that works, through Ultralytics end-to-end (preprocess + inference + NMS): `pytorch_cpu`, `rknn_npu_fp16`, `rknn_npu_int8`, `ncnn_cpu`, `ncnn_vulkan`, `onnxrt_cpu`, `tflite_cpu_fp32`, `tflite_cpu_int8`. For each: cold-load, first-inference (runtime init / NPU program load), steady-state avg/min/max/p95/stdev over 15 runs, real video FPS over up to 300 frames.
6. **Raw NPU per core mask** — `rknn-toolkit-lite2` directly, no NMS: `auto`, single core, two cores, three cores (RK3588), with an NPU-load sample from debugfs mid-run.
7. **Rockchip's `rknn_benchmark`** — built on the fly from six source files (no SDK clone) against the board's own `librknnrt.so`, run with single-core and all-core masks. This is the number Rockchip's own tables use.
8. **TFLite raw + OpenCV DNN** — XNNPACK CPU, the Teflon NPU delegate when `libteflon.so` is present (mainline), and OpenCV DNN on CPU / OpenCL / OpenCL-FP16 when libmali OpenCL is set up.
9. **NPU diagnosis** — one verdict with the fix: mainline kernel without NPU, `librknnrt` missing, Python too new for `rknn-toolkit-lite2`, models not exported, or runtime/driver version mismatch.
10. **Summary** — pass/fail counts, comparison tables (video FPS, steady-state latency, first-inference, cold-load), and an env file pointing at the fastest model/device.

### Android, from a host PC (`android/test_yolo_rockchip_android.sh`)

1. Device identity via `getprop`, presence and version of `/vendor/lib64/librknnrt.so`, `rknn_server`, NNAPI vendor HALs (Rockchip ships none for the NPU, so NNAPI is the CPU reference implementation), Mali OpenCL libs.
2. Google's prebuilt TFLite `benchmark_model` on every exported `.tflite`: CPU (XNNPACK, 4 threads), GPU delegate (OpenCL on Mali), NNAPI. Parses inference average, init and first-inference times, and whether the delegate took the whole graph.
3. Rockchip's `rknn_benchmark` (Android build via `tools/build_rknn_benchmark.sh android`, needs an NDK) on every RKNN model with single-core and all-core masks, using the device's own runtime.
4. Summary table + CSV.

## Requirements

**Board (Linux):** an aarch64 Debian/Ubuntu-style image with Python 3.10–3.12 and `python3-venv`. For the NPU: a vendor BSP kernel with `rknpu` and `librknnrt.so` installed (Radxa: `sudo apt install rknpu2-rk3588` or `rknpu2-rk356x`; other distros: copy `rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so` from [airockchip/rknn-toolkit2](https://github.com/airockchip/rknn-toolkit2) to `/usr/lib`). For `rknn_benchmark`: `build-essential zlib1g-dev`. For GPU tests: see [GPU setup](#mali-gpu-setup).

**PC (export):** x86-64 Linux with Python 3.8–3.12. `--tflite` (default on x86) also installs TensorFlow for the LiteRT export.

**Android:** `adb` on the PC, USB debugging on the device, firmware built with the RKNN stack (`/vendor/lib64/librknnrt.so` present). `ANDROID_NDK_PATH` if you want the NPU numbers.

## Usage

```bash
# PC
./export_rknn_host.sh --target rk3588,rk3576 [--quick] [--no-tflite]

# Board (Linux)
chmod +x test_yolo_rockchip.sh
./test_yolo_rockchip.sh --models rknn_models_rk3588.tar.gz   # full run
./test_yolo_rockchip.sh --quick --models ...                 # nano only
./test_yolo_rockchip.sh --skip-install --models ...          # reuse the venv
./test_yolo_rockchip.sh --python python3.12 --models ...     # pick the interpreter

# Android (from the PC, device on adb)
./android/test_yolo_rockchip_android.sh --models-dir ~/yolo-rockchip-export [--serial XXXX]
```

Everything on the board lives under `~/yolo-rockchip-test/` (venv, assets, logs, `benchmark_results.csv`, `results_summary.txt`, `yolo_rockchip_env.sh`). Android runs write to `~/yolo-rockchip-android-test/` on the PC.

## Output and backend labels

`benchmark_results.csv` uses the shared schema `backend,model,stage,metric,value`.

| Label | Meaning |
|---|---|
| `pytorch_cpu` | Ultralytics on `.pt`, end-to-end |
| `rknn_npu_{fp16,int8}` | Ultralytics RKNN backend (`rknn-toolkit-lite2`, core mask auto), end-to-end |
| `ncnn_cpu`, `ncnn_vulkan` | Ultralytics NCNN backend, CPU or `device=vulkan:0` (Mali via PanVK), end-to-end |
| `onnxrt_cpu` | Ultralytics ONNX backend (onnxruntime CPU), end-to-end |
| `tflite_cpu_{fp32,int8}` | Ultralytics TFLite backend, end-to-end |
| `rknn_raw_{fp16,int8}_core{auto,0,0_1,0_1_2}` | raw `rknn-toolkit-lite2`, no NMS |
| `rknn_benchmark_{fp16,int8}_mask{0,3,7}` | Rockchip's tool; mask 0 = auto, 3 = two cores, 7 = three cores |
| `tflite_raw_cpu`, `tflite_raw_teflon_npu` | raw TFLite interpreter, XNNPACK / Mesa Teflon delegate |
| `opencv_dnn_{cpu,opencl,opencl_fp16}` | raw OpenCV DNN forward on the ONNX model |
| `android_tflite_{cpu,gpu,nnapi}` | `benchmark_model` on Android |
| `android_rknn_npu_{fp16,int8}_mask{0,3,7}` | `rknn_benchmark` on Android |

Only the end-to-end rows are apples-to-apples with the AMD / Intel suites. Raw rows are device ceilings.

## NPU notes

- **Version triangle.** The kernel `rknpu` driver, `/usr/lib/librknnrt.so`, and the `rknn-toolkit2` used to build the `.rknn` must be compatible: SDK 2.3.x needs driver ≥ 0.9.6, and the runtime should be the same 2.x minor as the toolkit. The script prints all three and flags mismatches. Fix the runtime by replacing `librknnrt.so` with the SDK copy; fix an old driver with a newer BSP kernel or the [DKMS driver](https://github.com/bmilde/rknpu-driver-dkms).
- **Python ≤ 3.12.** `rknn-toolkit-lite2` has no wheels for 3.13+. Debian 13 and Ubuntu 25.x images need a `python3.12` interpreter (`--python python3.12`).
- **Core masks.** RK3588 has three cores; RK3576 has two. Ultralytics uses `NPU_CORE_AUTO`. Multi-core masks help large models most and can hurt nano models.
- **Clocks.** For peak numbers: `echo performance | sudo tee /sys/class/devfreq/<npu>/governor` (the script prints the exact path).
- **Mainline (rocket + Teflon)** is RK3588-only, INT8-only, offloads convolutions only, and runs at a fixed clock. YOLO mostly falls back to CPU there today; it's benchmarked so you can see the actual gain.
- **Ultralytics `is_rockchip()`** matches the last device-tree compatible string against `rk3588`, `rk3576`, …; mainline DTs for RK3588S boards end in `rockchip,rk3588s` and fail it. The harness bypasses this with `--force-rockchip` when the script detects that case.
- **YOLO26 INT8 on RKNN** has had reported issues (no detections / segfault with `end2end` graphs). The suite reports detections per run so you can spot it.

## Mali GPU setup

Two mutually exclusive routes:

**A. Rockchip libmali blob (OpenCL; vendor kernels).** Enables OpenCV DNN OpenCL. Radxa: `sudo apt install libmali-valhall-g610-g24p0-x11-wayland-gbm` (RK3588) or `libmali-bifrost-g52-g13p0-x11-wayland-gbm` (RK3576), then blacklist `panfrost`. Generic:

```bash
cd /usr/lib && sudo wget https://github.com/JeffyCN/mirrors/raw/libmali/lib/aarch64-linux-gnu/libmali-valhall-g610-g6p0-x11-wayland-gbm.so
cd /lib/firmware && sudo wget https://github.com/JeffyCN/mirrors/raw/libmali/firmware/g610/mali_csffw.bin
sudo apt install ocl-icd-libopencl1 clinfo
sudo mkdir -p /etc/OpenCL/vendors && echo /usr/lib/libmali-valhall-g610-g6p0-x11-wayland-gbm.so | sudo tee /etc/OpenCL/vendors/mali.icd
clinfo | grep -E "Device Name|Device Version"
```

**B. Mainline Mesa (Vulkan; `panthor` / `panfrost`).** Enables NCNN Vulkan. Needs kernel ≥ 6.10 (RK3588) and Mesa ≥ 25.0 with PanVK; `sudo apt install mesa-vulkan-drivers vulkan-tools` then `vulkaninfo --summary`. No OpenCL on this route.

## Android notes

- Only firmware built with `BOARD_RKNN_SUPPORT` ships `librknnrt.so`; without it the NPU is unreachable from Android.
- NNAPI on Rockchip Android has no NPU HAL. `android_tflite_nnapi` is effectively a CPU number and is kept as a reference.
- The GPU delegate needs `/vendor/lib64/libOpenCL.so`; without it TFLite falls back to OpenGL ES.

## Files in this repo

- `export_rknn_host.sh` — PC-side (x86) export: RKNN FP16/INT8 per target, NCNN, ONNX, TFLite; packs one tarball per SoC
- `test_yolo_rockchip.sh` — board-side Linux diagnostic + benchmark driver
- `bench_harness.py` — Ultralytics end-to-end harness (shared with the other suites), plus `--force-rockchip`
- `rknn_raw_bench.py` — raw `rknn-toolkit-lite2` benchmark per core mask
- `tflite_bench.py` — raw TFLite benchmark with optional external delegate (Teflon)
- `tools/build_rknn_benchmark.sh` — builds Rockchip's `rknn_benchmark` for Linux (native) or Android (NDK) from six fetched source files
- `android/test_yolo_rockchip_android.sh` — host-side adb driver for Android

## Caveats

- Verified so far: the harness scripts on a non-Rockchip machine (CPU paths, TFLite FP32/uint8 handling). The RKNN, Vulkan, Teflon, and Android paths need real hardware; the driver scripts degrade to warnings rather than aborting.
- INT8 RKNN export calibrates on `coco8`, which is tiny. Real deployments should calibrate on a few hundred representative images.
- Raw rows (rknn_raw, rknn_benchmark, tflite_raw, opencv_dnn, android_*) exclude letterboxing and NMS.
