# Rockchip RK3588 / RK3576 — the stack this suite needs, and the traps in it

`test_yolo_rockchip.sh` benchmarks *on top of* an existing RKNPU stack. This file
records what that stack has to be, how to build it, and the specific ways it goes
wrong. On Rockchip almost every failure is a **version-matching** failure, and
almost every disappointing number is a **core-count** or **quantisation**
failure.

> **Verification status.** The *suite fixes* (one ONNX Runtime distribution,
> AutoUpdate off, explicit export dependencies, reporting what actually ran) were
> reproduced and fixed on AMD hardware, where the same bugs existed. The
> *Rockchip install steps* below come from Rockchip's `rknn-toolkit2`
> documentation and board-vendor BSP notes, checked 23 Sep 2026, and have **not**
> been run on an RK3588/RK3576 by the author of this document. §6 is the
> authority: it measures your board.

---

## 1. What has to be true

| Layer | What you need | Check |
|---|---|---|
| Kernel | vendor BSP 5.10 (rkr8) or 6.1 (rkr7) for the full NPU; mainline 6.18+ only gives the conv-only `rocket` driver | `uname -r` |
| NPU driver | `rknpu` 0.9.6–0.9.8 | `cat /sys/kernel/debug/rknpu/version` or `dmesg \| grep -i rknpu` |
| NPU runtime | `librknnrt.so` — **version must match the toolkit that converted the model** | §3 |
| Python runtime | `rknn-toolkit-lite2` 2.3.2 | `pip show rknn-toolkit-lite2` |
| Conversion | `rknn-toolkit2` on an **x86-64 Linux** host (aarch64 wheels exist but are second-class) | §2.1 |
| GPU (optional) | Mali: `panthor` + Mesa PanVK (mainline) or vendor `bifrost` blob | `vulkaninfo --summary` |
| Permissions | access to `/dev/dri/render*` and the DMA heaps | `ls -l /dev/dri/render* /dev/dma_heap/*` |

---

## 2. Procedure

### 2.1 Convert models on an x86 host

Ultralytics documents RKNN export as x86-64 Linux only. The suite splits the work
accordingly:

```bash
# on an x86-64 Linux PC (or WSL2)
./export_rknn_host.sh --target rk3588,rk3576        # --quick for nano only
scp ~/yolo-rockchip-export/rknn_models_rk3588.tar.gz board:
```

### 2.2 On the board

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip
python3 -m venv ~/yolo-rockchip-test/venv && source ~/yolo-rockchip-test/venv/bin/activate
pip install -U pip
pip install ultralytics "onnx>=1.12,<2" onnxslim ncnn onnxruntime rknn-toolkit-lite2
./test_yolo_rockchip.sh --models rknn_models_rk3588.tar.gz
```

Note `onnx`, `onnxslim` and `ncnn` are installed **explicitly**. The suite now
sets `YOLO_AUTOINSTALL=False` (§4), so Ultralytics will no longer pull export
dependencies in on its own — which also means it can no longer pull the wrong
ones in on its own.

---

## 3. The Rockchip trap: `librknnrt.so` must match the toolkit

This is the single most common RKNN failure, and it does not always announce
itself: a model converted with `rknn-toolkit2` 2.3.x and run against an older
`librknnrt.so` on the board may refuse to load, or load and produce wrong
output.

```bash
# the board's runtime library
strings /usr/lib/librknnrt.so | grep -i "librknnrt version" | head -1
# the board's NPU driver
sudo cat /sys/kernel/debug/rknpu/version
# the RKNN server, if you use it
strings /usr/bin/rknn_server | grep -i build
# what converted the model, on the x86 host
python3 -c "import rknn; print(rknn.__version__)"
```

They must agree. Rockchip's own guidance when they do not is to **replace the
board's runtime**, not to downgrade the toolkit — copy from the matching
`rknn-toolkit2` checkout:

```bash
sudo cp rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64/librknnrt.so /usr/lib/
```

The driver is a third version to keep in mind: `rknpu` 0.9.6 vs 0.9.8 differ in
what runtime versions they accept. The suite prints all three in STEP 0.

---

## 4. The trap that silently fakes your results

`onnxruntime` and any accelerated ONNX Runtime build unpack into the **same
`onnxruntime/` directory**, so installing two means one silently shadows the
other. And Ultralytics' exporter checks for a distribution literally named
`onnxruntime`; when it does not find one it **AutoUpdates the plain CPU wheel
mid-export**.

On Rockchip the ONNX path is CPU anyway, so the damage is smaller than on the
Intel/Qualcomm/AMD siblings — but the same mechanism previously meant this suite
depended on AutoUpdate to install `onnx` and `ncnn` at export time, which is a
silent, unpinned install of build dependencies in the middle of a benchmark run.
Both are now explicit, and the suite sets:

```bash
export YOLO_AUTOINSTALL=False
pip list | grep -ci '^onnxruntime'     # must be exactly 1
```

`assert_ort_runtime` re-checks this after the install phase.

---

## 5. Why your NPU numbers may disappoint

Three causes, in the order they usually apply:

1. **You are using one NPU core out of three.** RK3588's NPU is 3 cores / 6 TOPS,
   but a model runs on core 0 unless you say otherwise. With
   `rknn-toolkit-lite2`:

   ```python
   from rknnlite.api import RKNNLite
   rk = RKNNLite()
   rk.load_rknn("yolo11n.rknn")
   rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)   # not the default
   ```

   For a single stream this mainly helps large models; for multiple concurrent
   streams, pinning one model per core is usually better than splitting one.

2. **The model is not INT8.** The RKNPU is an integer engine. Convert with
   `do_quantization=True` and a representative `dataset.txt`; an FP16 RKNN model
   will run but throws away most of the hardware.

3. **Pre/post-processing dominates.** A YOLO nano model's NPU inference can be a
   few milliseconds while letterboxing and NMS in Python cost more. The suite
   separates cold-load, first-inference, steady-state and end-to-end video FPS so
   this is visible instead of hidden in one number.

---

## 6. First run on a new board — the short version

```bash
uname -r; cat /proc/device-tree/compatible | tr '\0' '\n'
dmesg | grep -i rknpu | tail -3
cat /sys/kernel/debug/rknpu/version 2>/dev/null
strings /usr/lib/librknnrt.so | grep -i "librknnrt version" | head -1
pip show rknn-toolkit-lite2 | head -2
ls -l /dev/dri/render* ; ls /dev/dma_heap/ 2>/dev/null
pip list | grep -i '^onnxruntime'        # exactly one line
python3 lib/accel_verify.py --list
./test_yolo_rockchip.sh --models rknn_models_rk3588.tar.gz
```

---

## 7. Verifying the silicon actually ran

`lib/accel_verify.py` reads kernel counters instead of trusting the runtime:

```bash
python3 lib/accel_verify.py --list
python3 lib/accel_verify.py --pid <pid> --seconds 5
```

Rockchip is the best-instrumented of the four platforms in this family:

- **NPU** — `/sys/kernel/debug/rknpu/load` reports per-core load
  (`Core0: 42%, Core1: 0%, Core2: 0%`), which also shows you at a glance whether
  §5.1 applies to you.
- **Mali GPU** — `/sys/class/devfreq/*.gpu/load`.

Both are device-wide rather than per-process, so they cannot separate your job
from anything else on the device — read them on an otherwise idle board.

---

## 8. Measuring fairly

`lib/fair_compare.py` compares every backend present (PyTorch CPU, ONNX Runtime,
RKNN NPU when an `.rknn` sits beside the model), and `lib/fairness.py` holds the
methodology: rounds, medians of medians, reported spread, and a loud flag when
the spread exceeds 15%.

It exists because the naive approach produced, on the AMD sibling box, a 46%
swing on one workload and a confidently wrong 1.8× where the true figure was
1.1–1.3×. On a passively cooled SBC this matters more: sustained load throttles,
and the numbers from the first thirty seconds are not the numbers you will live
with.

```bash
python3 lib/fair_compare.py --model yolo11n --workdir ~/yolo-rockchip-test
python3 lib/fair_compare.py --model yolo11n --mode interleaved
```

---

## 9. Mainline versus vendor kernel

| | Vendor BSP (5.10 rkr8 / 6.1 rkr7) | Mainline 6.18+ |
|---|---|---|
| NPU | full `rknpu` + `librknnrt`, all ops | `rocket` driver + Mesa Teflon, **convolution only** |
| GPU | vendor Bifrost blob | `panthor` + Mesa PanVK (good) |
| Practical verdict for YOLO | use this | not yet — the NPU path cannot run a whole YOLO graph |

The suite detects which of the two you are on and adjusts what it expects, so a
mainline board reports a clear "NPU present but conv-only" rather than a pile of
failures.

---

## 10. Files

- `INSTALL_ROCKCHIP.md` — this document
- `lib/accel_verify.py` — kernel-counter accelerator verification
- `lib/fairness.py`, `lib/fair_compare.py` — measurement methodology
- `test_yolo_rockchip.sh` — the on-board suite
- `export_rknn_host.sh` — x86 host-side conversion
- `rknn_raw_bench.py`, `tflite_bench.py`, `bench_harness.py` — per-stage harnesses
- `android/` — the same board driven over `adb`

---

## 11. Provenance of the version claims

The install steps here come from Rockchip and board-vendor documentation,
checked on **23 Sep 2026**, and have **not** been run on an RK3588/RK3576 board.

| Claim | Source |
|---|---|
| `librknnrt.so` must match the toolkit; fix by replacing the runtime from `rknpu2/runtime/Linux/librknn_api/aarch64/`; version checks via `strings` and `/sys/kernel/debug/rknpu/version` | Radxa RKNN Toolkit2 docs; FriendlyELEC NPU wiki; rockchip-linux/rknn-toolkit2 issue tracker |
| `rknn-toolkit-lite2` 2.3.2 | [PyPI](https://pypi.org/project/rknn-toolkit-lite2/) |
| RKNN export is x86-64 Linux only | [Ultralytics Rockchip RKNN docs](https://docs.ultralytics.com/integrations/rockchip-rknn) |
| RK3588 NPU is 3 cores and defaults to one unless `core_mask` is set | Rockchip RKNPU2 API documentation |
| Mainline `rocket` driver is convolution-only via Mesa Teflon | Mesa release notes, as already stated in this repo's README |
| The ONNX Runtime shadowing behaviour and Ultralytics AutoUpdate (§4) | reproduced and fixed on AMD hardware in [ryzen_yolo](https://github.com/ZephyrSai/ryzen_yolo) |

§6 prints what your board actually has. Where it disagrees, believe the board.
