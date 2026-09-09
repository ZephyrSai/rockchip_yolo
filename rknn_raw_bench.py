#!/usr/bin/env python3
"""
rknn_raw_bench.py — raw RKNN NPU benchmark through rknn-toolkit-lite2
(no Ultralytics wrapper, no NMS), used by test_yolo_rockchip.sh to
characterise the NPU per core mask.

Usage:
  python3 rknn_raw_bench.py --model <dir_rknn_model | model.rknn> \
      --core-mask <auto|0|1|2|0_1|0_1_2> [--video path.mp4] \
      [--runs 15] [--video-max-frames 300]

Output (KEY=VALUE lines):
  HOST                 - SoC from /proc/device-tree/compatible
  SDK_VERSION          - librknnrt api version / driver version (if exposed)
  LOAD_MS              - load_rknn() (parse + weights)
  INIT_RUNTIME_MS      - init_runtime(core_mask=...) (NPU program upload)
  INPUT_SHAPE          - as taken from the video path (NHWC uint8 640x640)
  FIRST_INFER_MS
  STEADY_AVG_MS / MIN / MAX / P95
  VIDEO_FRAMES / VIDEO_AVG_FPS / VIDEO_AVG_MS  - decode + resize + inference
  NPU_LOAD_SAMPLE      - one sample of /sys/kernel/debug/rknpu/load taken
                          mid-run (only if readable; needs root/debugfs)
"""

import argparse
import os
import statistics
import sys
import time

import numpy as np


def host_soc():
    try:
        with open("/proc/device-tree/compatible", "rb") as f:
            parts = [p.decode(errors="ignore") for p in f.read().split(b"\0") if p]
        return ",".join(parts)
    except Exception:
        return "unknown"


def read_npu_load():
    for p in ("/sys/kernel/debug/rknpu/load", "/proc/rknpu/load"):
        try:
            with open(p) as f:
                return f.read().strip().replace("\n", " ")
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--core-mask", default="auto")
    ap.add_argument("--video", default=None)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--video-max-frames", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    try:
        from rknnlite.api import RKNNLite
    except Exception as e:
        print(f"ERROR=rknn-toolkit-lite2 import failed: {e}")
        sys.exit(1)

    model = args.model
    if os.path.isdir(model):
        rknns = [f for f in os.listdir(model) if f.endswith(".rknn")]
        if not rknns:
            print(f"ERROR=no .rknn in {model}")
            sys.exit(1)
        model = os.path.join(model, sorted(rknns)[0])
    print(f"MODEL={model}")
    print(f"HOST={host_soc()}")

    masks = {
        "auto": RKNNLite.NPU_CORE_AUTO,
        "0": RKNNLite.NPU_CORE_0,
        "1": RKNNLite.NPU_CORE_1,
        "2": getattr(RKNNLite, "NPU_CORE_2", None),
        "0_1": RKNNLite.NPU_CORE_0_1,
        "0_1_2": getattr(RKNNLite, "NPU_CORE_0_1_2", None),
    }
    if args.core_mask not in masks or masks[args.core_mask] is None:
        print(f"ERROR=unknown/unsupported core mask {args.core_mask}")
        sys.exit(1)
    print(f"CORE_MASK={args.core_mask}")

    rk = RKNNLite(verbose=False)
    t0 = time.perf_counter()
    if rk.load_rknn(model) != 0:
        print("ERROR=load_rknn failed")
        sys.exit(1)
    print(f"LOAD_MS={(time.perf_counter() - t0) * 1000:.1f}")

    t0 = time.perf_counter()
    ret = rk.init_runtime(core_mask=masks[args.core_mask])
    if ret != 0:
        print(f"INIT_RUNTIME_ERROR=init_runtime returned {ret} (driver/runtime mismatch or NPU not accessible)")
        sys.exit(1)
    print(f"INIT_RUNTIME_MS={(time.perf_counter() - t0) * 1000:.1f}")
    try:
        v = rk.get_sdk_version()
        print(f"SDK_VERSION={str(v).strip().replace(chr(10), ' | ')}")
    except Exception:
        pass

    # Ultralytics RKNN export configures mean 0 / std 255, i.e. the model
    # takes raw uint8 RGB NHWC; that is what RKNNLite.inference expects.
    h = w = args.imgsz
    dummy = np.random.randint(0, 255, (1, h, w, 3), dtype=np.uint8)
    print(f"INPUT_SHAPE=[1, {h}, {w}, 3] uint8 NHWC")

    t0 = time.perf_counter()
    try:
        out = rk.inference(inputs=[dummy])
    except Exception as e:
        print(f"FIRST_INFER_ERROR={str(e).splitlines()[0][:300]}")
        sys.exit(1)
    if out is None:
        print("FIRST_INFER_ERROR=inference returned None")
        sys.exit(1)
    print(f"FIRST_INFER_MS={(time.perf_counter() - t0) * 1000:.2f}")
    print(f"NUM_OUTPUTS={len(out)}")

    times = []
    for i in range(args.runs):
        t0 = time.perf_counter()
        rk.inference(inputs=[dummy])
        times.append((time.perf_counter() - t0) * 1000)
        if i == args.runs // 2:
            s = read_npu_load()
            if s:
                print(f"NPU_LOAD_SAMPLE={s}")
    ts = sorted(times)
    p95_idx = max(0, int(len(ts) * 0.95) - 1)
    print(f"STEADY_AVG_MS={statistics.mean(times):.2f}")
    print(f"STEADY_MIN_MS={min(times):.2f}")
    print(f"STEADY_MAX_MS={max(times):.2f}")
    print(f"STEADY_P95_MS={ts[p95_idx]:.2f}")

    if args.video:
        try:
            import cv2
        except Exception as e:
            print(f"VIDEO_ERROR=opencv not available: {e}")
            rk.release()
            return
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"VIDEO_ERROR=could not open {args.video}")
            rk.release()
            return
        n = 0
        ft = []
        t_start = time.perf_counter()
        while n < args.video_max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            x = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB)[np.newaxis]
            t0 = time.perf_counter()
            rk.inference(inputs=[x])
            ft.append((time.perf_counter() - t0) * 1000)
            n += 1
        t_total = time.perf_counter() - t_start
        cap.release()
        if n:
            print(f"VIDEO_FRAMES={n}")
            print(f"VIDEO_TOTAL_S={t_total:.2f}")
            print(f"VIDEO_AVG_MS={statistics.mean(ft):.2f}")
            print(f"VIDEO_AVG_FPS={n / t_total:.2f}")
    rk.release()


if __name__ == "__main__":
    main()
