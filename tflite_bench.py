#!/usr/bin/env python3
"""
tflite_bench.py — raw TensorFlow Lite benchmark (no Ultralytics, no NMS)
with an optional external delegate, used to drive accelerator delegates
that Ultralytics cannot select itself:
  - Mesa Teflon (libteflon.so)            -> mainline Rockchip NPU driver
  - Qualcomm QNN TFLite delegate (libQnnTFLiteDelegate.so) -> Hexagon NPU / Adreno
  - TFLite GPU delegate (libtensorflowlite_gpu_delegate.so) -> Mali / Adreno OpenCL
  - none                                  -> XNNPACK CPU baseline

Usage:
  python3 tflite_bench.py --model model.tflite [--delegate /path/lib.so]
      [--delegate-opt key=value ...] [--threads 4] [--video path.mp4]
      [--runs 15] [--video-max-frames 300]

Output (KEY=VALUE lines):
  RUNTIME              - tflite_runtime | tensorflow | ai_edge_litert
  DELEGATE             - path or 'none'
  DELEGATE_LOAD_MS     - load_delegate() time (accelerator lib init)
  INTERP_INIT_MS       - Interpreter() + allocate_tensors() (graph partition,
                          NPU compile for delegates that compile at init)
  INPUT_SHAPE / INPUT_DTYPE
  FIRST_INFER_MS       - first invoke() (lazy compile lands here for some delegates)
  STEADY_AVG_MS / MIN / MAX / P95
  VIDEO_FRAMES / VIDEO_AVG_FPS / VIDEO_AVG_MS  - decode + resize + invoke
Quantised (uint8/int8) inputs are fed correctly via the input tensor's
quantization params; FP32 inputs get [0,1] normalised NHWC.
"""

import argparse
import statistics
import sys
import time

import numpy as np


def load_interpreter_cls():
    for mod, name in (("ai_edge_litert.interpreter", "ai_edge_litert"),
                      ("tflite_runtime.interpreter", "tflite_runtime"),
                      ("tensorflow.lite", "tensorflow")):
        try:
            m = __import__(mod, fromlist=["Interpreter", "load_delegate"])
            return m.Interpreter, m.load_delegate, name
        except Exception:
            continue
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--delegate", default=None)
    ap.add_argument("--delegate-opt", action="append", default=[], help="key=value passed to the delegate")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--video", default=None)
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--video-max-frames", type=int, default=300)
    args = ap.parse_args()

    Interpreter, load_delegate, runtime = load_interpreter_cls()
    if Interpreter is None:
        print("ERROR=no TFLite runtime importable (pip install ai-edge-litert or tflite-runtime)")
        sys.exit(1)
    print(f"RUNTIME={runtime}")
    print(f"MODEL={args.model}")
    print(f"DELEGATE={args.delegate or 'none'}")

    delegates = []
    if args.delegate:
        opts = dict(kv.split("=", 1) for kv in args.delegate_opt)
        t0 = time.perf_counter()
        try:
            delegates = [load_delegate(args.delegate, opts)]
        except Exception as e:
            print(f"DELEGATE_LOAD_ERROR={str(e).splitlines()[0][:300]}")
            sys.exit(1)
        print(f"DELEGATE_LOAD_MS={(time.perf_counter() - t0) * 1000:.1f}")
        if opts:
            print(f"DELEGATE_OPTS={opts}")

    t0 = time.perf_counter()
    try:
        kw = dict(model_path=args.model, experimental_delegates=delegates or None)
        if args.threads:
            kw["num_threads"] = args.threads
        interp = Interpreter(**kw)
        interp.allocate_tensors()
    except Exception as e:
        print(f"INTERP_INIT_ERROR={str(e).splitlines()[0][:300]}")
        sys.exit(1)
    print(f"INTERP_INIT_MS={(time.perf_counter() - t0) * 1000:.1f}")

    inp = interp.get_input_details()[0]
    shape = [int(x) for x in inp["shape"]]
    dtype = inp["dtype"]
    scale, zero = inp.get("quantization", (0.0, 0))
    print(f"INPUT_SHAPE={shape}")
    print(f"INPUT_DTYPE={np.dtype(dtype).name}")

    def make_input(frame_rgb01):  # frame as float32 [0,1] NHWC
        if dtype == np.float32:
            return frame_rgb01.astype(np.float32)
        if scale and scale > 0:
            q = np.round(frame_rgb01 / scale + zero)
        else:
            q = frame_rgb01 * 255.0
        info = np.iinfo(dtype)
        return np.clip(q, info.min, info.max).astype(dtype)

    dummy = make_input(np.random.rand(*shape).astype(np.float32))

    t0 = time.perf_counter()
    try:
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
    except Exception as e:
        print(f"FIRST_INFER_ERROR={str(e).splitlines()[0][:300]}")
        sys.exit(1)
    print(f"FIRST_INFER_MS={(time.perf_counter() - t0) * 1000:.2f}")

    times = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], dummy)
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)
    ts = sorted(times)
    p95_idx = max(0, int(len(ts) * 0.95) - 1)
    print(f"STEADY_AVG_MS={statistics.mean(times):.2f}")
    print(f"STEADY_MIN_MS={min(times):.2f}")
    print(f"STEADY_MAX_MS={max(times):.2f}")
    print(f"STEADY_P95_MS={ts[p95_idx]:.2f}")

    if not args.video:
        return
    try:
        import cv2
    except Exception as e:
        print(f"VIDEO_ERROR=opencv not available: {e}")
        return
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"VIDEO_ERROR=could not open {args.video}")
        return
    # NHWC [1,H,W,3] expected (Ultralytics TFLite export); handle NCHW defensively
    nhwc = shape[-1] == 3
    h, w = (shape[1], shape[2]) if nhwc else (shape[2], shape[3])
    n = 0
    ft = []
    t_start = time.perf_counter()
    while n < args.video_max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(cv2.resize(frame, (w, h)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = rgb[np.newaxis] if nhwc else rgb.transpose(2, 0, 1)[np.newaxis]
        x = make_input(x)
        t0 = time.perf_counter()
        interp.set_tensor(inp["index"], x)
        interp.invoke()
        ft.append((time.perf_counter() - t0) * 1000)
        n += 1
    t_total = time.perf_counter() - t_start
    cap.release()
    if n:
        print(f"VIDEO_FRAMES={n}")
        print(f"VIDEO_TOTAL_S={t_total:.2f}")
        print(f"VIDEO_AVG_MS={statistics.mean(ft):.2f}")
        print(f"VIDEO_AVG_FPS={n / t_total:.2f}")


if __name__ == "__main__":
    main()
