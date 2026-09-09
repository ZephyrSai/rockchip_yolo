#!/usr/bin/env python3
"""
bench_harness.py — reusable end-to-end benchmark harness (Ultralytics
wrapper, includes preprocess + NMS) called by the test_yolo_*.sh driver
for every backend/model combination.

Usage:
  python3 bench_harness.py --model <path.pt | dir_rknn_model | dir_ncnn_model | model.tflite | model.onnx> [--force-rockchip] \
      [--device cpu] --image <path.jpg> --video <path.mp4> --backend-label <str> \
      [--img-runs 15] [--video-max-frames 300] [--imgsz 640] [--half] [--int8]

--device is optional: exported backends (RKNN, NCNN, TFLite, ONNX) pick
their runtime from the model format and ignore it; PyTorch weights use it
("cpu" here — these SoCs have no PyTorch GPU/NPU device).

Prints machine-readable KEY=VALUE lines (one per line) so the calling
bash script can grep them out. Measures, in order:
  1. COLD_LOAD_MS      - YOLO(model) constructor (imports + weight load).
                          For RKNN/NCNN/TFLite the runtime and accelerator
                          are initialised lazily on the first predict(), so
                          that cost lands in (2).
  2. FIRST_INFER_MS    - first .predict() call: runtime init, NPU model
                          load, GPU shader/kernel compile all live here.
  3. STEADY_AVG_MS / STEADY_MIN_MS / STEADY_MAX_MS / STEADY_P95_MS /
     STEADY_STDEV_MS   - N repeated inferences on the same image after
                          warmup (preprocess + inference + NMS).
  4. VIDEO_FRAMES, VIDEO_TOTAL_S, VIDEO_AVG_FPS, VIDEO_AVG_MS,
     VIDEO_MIN_FPS, VIDEO_MAX_FPS
                        - full video decode+inference loop, capped at
                          --video-max-frames frames: real end-to-end FPS.
"""

import argparse
import statistics
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--image", required=True)
    ap.add_argument("--video", default=None)
    ap.add_argument("--backend-label", required=True)
    ap.add_argument("--img-runs", type=int, default=15)
    ap.add_argument("--video-max-frames", type=int, default=300)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--force-rockchip", action="store_true",
                    help="bypass ultralytics' is_rockchip() check (mainline DTs ending in 'rockchip,rk3588s' fail it)")
    args = ap.parse_args()

    if args.force_rockchip:
        try:
            import ultralytics.utils.checks as _chk
            _chk.is_rockchip = lambda: True
            try:
                import ultralytics.nn.backends.rknn as _rk
                _rk.is_rockchip = lambda: True
            except Exception:
                pass
            print("FORCE_ROCKCHIP=1")
        except Exception as e:
            print(f"FORCE_ROCKCHIP_ERROR={e}")

    kw = dict(imgsz=args.imgsz, verbose=False)
    if args.device:
        kw["device"] = args.device
    if args.half:
        kw["half"] = True
    if args.int8:
        kw["int8"] = True

    print(f"BACKEND={args.backend_label}")
    print(f"MODEL={args.model}")
    print(f"DEVICE={args.device or 'auto(by-format)'}")

    # ---------- 1. Cold load ----------
    t0 = time.perf_counter()
    try:
        from ultralytics import YOLO
        model = YOLO(args.model)
    except Exception as e:
        print(f"COLD_LOAD_ERROR={e}")
        sys.exit(1)
    print(f"COLD_LOAD_MS={(time.perf_counter() - t0) * 1000:.1f}")

    # ---------- 2. First inference (runtime init / NPU load / GPU compile) ----------
    t0 = time.perf_counter()
    try:
        results = model.predict(args.image, **kw)
    except Exception as e:
        print(f"FIRST_INFER_ERROR={str(e).splitlines()[0][:300]}")
        sys.exit(1)
    print(f"FIRST_INFER_MS={(time.perf_counter() - t0) * 1000:.1f}")
    print(f"FIRST_RUN_DETECTIONS={len(results[0].boxes)}")

    # ---------- 3. Steady-state image inference ----------
    times = []
    for _ in range(args.img_runs):
        t0 = time.perf_counter()
        results = model.predict(args.image, **kw)
        times.append((time.perf_counter() - t0) * 1000)
    ts = sorted(times)
    p95_idx = max(0, int(len(ts) * 0.95) - 1)
    print(f"STEADY_AVG_MS={statistics.mean(times):.2f}")
    print(f"STEADY_MIN_MS={min(times):.2f}")
    print(f"STEADY_MAX_MS={max(times):.2f}")
    print(f"STEADY_P95_MS={ts[p95_idx]:.2f}")
    print(f"STEADY_STDEV_MS={statistics.pstdev(times):.2f}")
    print(f"STEADY_DETECTIONS={len(results[0].boxes)}")

    # ---------- 4. Full video benchmark (real end-to-end FPS) ----------
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
    frame_times = []
    n = 0
    t_start = time.perf_counter()
    while n < args.video_max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        model.predict(frame, **kw)
        frame_times.append((time.perf_counter() - t0) * 1000)
        n += 1
    t_total = time.perf_counter() - t_start
    cap.release()
    if n == 0:
        print("VIDEO_ERROR=zero frames read")
        return
    per_frame_fps = [1000.0 / t for t in frame_times if t > 0]
    print(f"VIDEO_FRAMES={n}")
    print(f"VIDEO_TOTAL_S={t_total:.2f}")
    print(f"VIDEO_AVG_MS={statistics.mean(frame_times):.2f}")
    print(f"VIDEO_AVG_FPS={n / t_total:.2f}")
    print(f"VIDEO_MIN_FPS={min(per_frame_fps):.2f}")
    print(f"VIDEO_MAX_FPS={max(per_frame_fps):.2f}")


if __name__ == "__main__":
    main()
