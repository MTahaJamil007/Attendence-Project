"""Phase 1a: pick the largest pose model and input size that holds real-time.

The predecessor shipped yolo26n-pose at imgsz=768 on a CPU-only torch build. On
this hardware that is roughly 3-8 FPS, and its 0.85s-hold / 0.28s-gap rule cannot
be satisfied at that frame rate - which is the direct cause of its ~30% recall.

This tool measures, rather than guesses, what the GPU can actually sustain, and
reports the largest configuration that clears the target frame rate.

Benchmark on a REAL classroom frame whenever possible:

    python tools/benchmark.py --source classroom.jpg

A synthetic frame measures raw throughput but not the detection load of 40 people,
and detection count affects tracker and post-processing cost.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_MODELS = [
    "yolo11n-pose.pt",
    "yolo11s-pose.pt",
    "yolo11m-pose.pt",
    "yolo26n-pose.pt",
    "yolo26s-pose.pt",
    "yolo26m-pose.pt",
]
DEFAULT_IMGSZ = [960, 1280]


def load_frame(source: str, cfg) -> tuple[np.ndarray, str]:
    """Get one frame from an image, a video, a camera index, or synthesize one."""
    import cv2

    if source == "synthetic":
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8), "synthetic 1920x1080"

    p = Path(source)
    if p.exists():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            img = cv2.imread(str(p))
            if img is None:
                raise SystemExit(f"could not read image: {p}")
            return img, f"image {p.name} {img.shape[1]}x{img.shape[0]}"
        cap = cv2.VideoCapture(str(p))
        ok, img = cap.read()
        cap.release()
        if not ok or img is None:
            raise SystemExit(f"could not read a frame from video: {p}")
        return img, f"video {p.name} {img.shape[1]}x{img.shape[0]}"

    from camera import discover_cameras, open_camera

    try:
        idx = int(source)
    except ValueError:
        raise SystemExit(f"source not understood: {source}")

    if idx < 0:
        cams = discover_cameras(cfg)
        if not cams:
            print("  no camera found; falling back to a synthetic frame")
            return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8), "synthetic 1920x1080"
        idx = cams[0].index

    cap, info = open_camera(idx, cfg)
    if cap is None:
        print(f"  camera {idx} unavailable; falling back to a synthetic frame")
        return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8), "synthetic 1920x1080"
    # Let auto-exposure settle before grabbing the benchmark frame.
    img = None
    for _ in range(10):
        ok, f = cap.read()
        if ok and f is not None:
            img = f
    cap.release()
    if img is None:
        raise SystemExit(f"camera {idx} opened but produced no frame")
    return img, f"{info.label()}"


def bench_one(
    weights: str, imgsz: int, frame, half: bool, device: str, conf: float, iters: int
) -> dict:
    import torch
    from ultralytics import YOLO

    row = {
        "weights": weights,
        "imgsz": imgsz,
        "ok": False,
        "fps": 0.0,
        "ms": 0.0,
        "p95_ms": 0.0,
        "people": 0,
        "vram_mb": 0.0,
        "pre_ms": 0.0,
        "inf_ms": 0.0,
        "post_ms": 0.0,
        "error": "",
    }
    try:
        model = YOLO(weights)
    except Exception as exc:
        row["error"] = f"unavailable: {str(exc).splitlines()[0][:70]}"
        return row

    try:
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()

        # ultralytics 8.4 replaced half=True with quantize=16.
        kw = dict(
            imgsz=imgsz,
            device=device,
            quantize=16 if half else None,
            conf=conf,
            classes=[0],
            verbose=False,
        )
        # Warm-up: first calls pay for kernel selection and cuDNN autotuning.
        for _ in range(5):
            model.predict(frame, **kw)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

        times: list[float] = []
        people = 0
        pre = inf = post = 0.0
        for _ in range(iters):
            t0 = time.perf_counter()
            res = model.predict(frame, **kw)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
            if res:
                if res[0].boxes is not None:
                    people = max(people, len(res[0].boxes))
                # Ultralytics reports its own stage split. Without it the totals
                # are uninterpretable: on an empty frame, preprocessing dominates
                # and the ranking between models becomes meaningless noise.
                sp = getattr(res[0], "speed", None) or {}
                pre += float(sp.get("preprocess") or 0.0)
                inf += float(sp.get("inference") or 0.0)
                post += float(sp.get("postprocess") or 0.0)

        times.sort()
        row["ok"] = True
        row["ms"] = sum(times) / len(times)
        row["p95_ms"] = times[int(len(times) * 0.95) - 1]
        row["fps"] = 1000.0 / row["ms"]
        row["people"] = people
        row["pre_ms"] = pre / iters
        row["inf_ms"] = inf / iters
        row["post_ms"] = post / iters
        if device.startswith("cuda"):
            row["vram_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:70]}"
    finally:
        try:
            del model
            if device.startswith("cuda"):
                import torch as _t

                _t.cuda.empty_cache()
        except Exception:
            pass
    return row


def main(argv: list[str] | None = None) -> int:
    from config import Config
    from runtime_check import check_runtime

    cfg = Config.load(ROOT / "config.json")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="-1", help="image / video path, camera index, or 'synthetic'")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--imgsz", default=",".join(str(i) for i in DEFAULT_IMGSZ))
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--target-fps", type=float, default=25.0)
    ap.add_argument("--fp32", action="store_true", help="disable FP16 even if supported")
    args = ap.parse_args(argv)

    print("=" * 84)
    print("POSE MODEL BENCHMARK")
    print("=" * 84)
    rep = check_runtime(require_cuda=True, allow_cpu_fallback=True)
    print("  " + rep.summary())
    if not rep.ok:
        return 1
    device = "cpu" if rep.using_cpu else cfg.get("model.device", "cuda:0")
    half = (not args.fp32) and rep.fp16_ok and not rep.using_cpu
    if rep.using_cpu:
        print("  WARNING: measuring CPU throughput. Live sessions must run on GPU.")

    frame, desc = load_frame(args.source, cfg)
    print(f"  benchmark frame: {desc}")
    print(f"  precision: {'FP16' if half else 'FP32'} | iters per config: {args.iters}")
    if args.source == "synthetic" or desc.startswith("synthetic"):
        print("  NOTE: a synthetic frame has no people in it. Re-run with a real")
        print("        classroom image for numbers that include detection load.")
    print()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    sizes = [int(s) for s in args.imgsz.split(",") if s.strip()]
    conf = float(cfg.get("model.person_conf", 0.3))

    print(
        f"  {'model':<20} {'imgsz':>6} {'FPS':>7} {'ms':>7} {'p95ms':>7} "
        f"{'pre':>6} {'infer':>6} {'post':>6} {'VRAM':>6}  {'ppl':>4}  note"
    )
    print("  " + "-" * 96)
    rows: list[dict] = []
    for weights in models:
        for imgsz in sizes:
            row = bench_one(weights, imgsz, frame, half, device, conf, args.iters)
            rows.append(row)
            if row["ok"]:
                print(
                    f"  {row['weights']:<20} {row['imgsz']:>6} {row['fps']:>7.1f} "
                    f"{row['ms']:>7.1f} {row['p95_ms']:>7.1f} "
                    f"{row['pre_ms']:>6.1f} {row['inf_ms']:>6.1f} {row['post_ms']:>6.1f} "
                    f"{row['vram_mb']:>6.0f}  {row['people']:>4}"
                )
            else:
                print(
                    f"  {weights:<20} {imgsz:>6} {'-':>7} {'-':>7} {'-':>7} "
                    f"{'-':>6} {'-':>6} {'-':>6} {'-':>6}  {'-':>4}  {row['error']}"
                )

    ok_rows = [r for r in rows if r["ok"]]
    if not ok_rows:
        print("\n  no configuration ran successfully")
        return 1

    # --- validity warnings -------------------------------------------------
    # Numbers from a frame with nobody in it are an upper bound, not a forecast.
    # Say so loudly, because a confident-looking table is worse than no table.
    warnings: list[str] = []
    if max(r["people"] for r in ok_rows) == 0:
        warnings.append(
            "NO PEOPLE DETECTED in the benchmark frame. These figures exclude "
            "keypoint decoding and tracker cost for a full class, so treat them "
            "as an upper bound. Re-run with --source <classroom photo>."
        )
    # Inference time must grow with input size. If it does not, fixed overhead
    # (letterbox resize, host-device copy, framework) dominates and the ranking
    # between models is noise.
    by_model: dict[str, list[dict]] = {}
    for r in ok_rows:
        by_model.setdefault(r["weights"], []).append(r)
    inverted = [
        m
        for m, rs in by_model.items()
        if len(rs) > 1
        and sorted(rs, key=lambda r: r["imgsz"])[0]["fps"]
        < sorted(rs, key=lambda r: r["imgsz"])[-1]["fps"]
    ]
    if inverted:
        warnings.append(
            f"{len(inverted)} model(s) measured FASTER at a larger imgsz "
            f"({', '.join(inverted[:3])}). That is impossible for GPU compute, so "
            "fixed per-call overhead is dominating and model ranking here is not "
            "meaningful. Benchmark a real classroom frame."
        )
    mean_inf = sum(r["inf_ms"] for r in ok_rows) / len(ok_rows)
    mean_pre = sum(r["pre_ms"] for r in ok_rows) / len(ok_rows)
    if mean_inf > 0 and mean_pre > mean_inf:
        warnings.append(
            f"preprocessing ({mean_pre:.1f} ms) exceeds inference ({mean_inf:.1f} ms) "
            "on average - the pipeline is input-bound, not GPU-bound."
        )
    if warnings:
        print()
        print("  " + "!" * 78)
        for w in warnings:
            print("  ! " + w)
        print("  " + "!" * 78)

    # Recommend the heaviest configuration that still clears the target, ranking
    # input size above model size: back-row students are small, and pixels on
    # target matter more for keypoint accuracy than parameter count.
    def weight_rank(name: str) -> int:
        for i, tag in enumerate(["n-", "s-", "m-", "l-", "x-"]):
            if tag in name:
                return i
        return 0

    passing = [r for r in ok_rows if r["fps"] >= args.target_fps]
    print()
    print("=" * 84)
    if passing:
        best = sorted(passing, key=lambda r: (r["imgsz"], weight_rank(r["weights"])))[-1]
        print(f"  RECOMMENDED: {best['weights']} at imgsz={best['imgsz']} "
              f"({best['fps']:.1f} FPS, {best['vram_mb']:.0f} MB VRAM)")
        print()
        print("  Apply with:")
        print(f'    config.json -> model.weights = "{best["weights"]}"')
        print(f"    config.json -> model.imgsz   = {best['imgsz']}")
    else:
        fastest = max(ok_rows, key=lambda r: r["fps"])
        print(f"  NOTHING cleared {args.target_fps:.0f} FPS. Fastest was "
              f"{fastest['weights']} at imgsz={fastest['imgsz']} ({fastest['fps']:.1f} FPS).")
        print("  Lower --target-fps, drop imgsz, or check that the GPU is really in use.")
    print("=" * 84)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
