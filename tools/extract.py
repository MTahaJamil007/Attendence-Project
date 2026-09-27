"""Run pose inference once over a recording and cache the keypoints.

This is what makes calibration tractable. Inference costs ~40 ms per frame, so a
5-minute recording at 15 FPS is about three minutes of GPU time. A sweep over a few
thousand threshold combinations that re-ran the model each time would take days.

Scoring and temporal filtering are pure Python over keypoints that do not change
when a threshold changes, so the model runs once per (model, imgsz) pair and every
subsequent sweep reads the cache in seconds.

    python tools/extract.py --name classroom-1
    python tools/extract.py --name classroom-1 --weights yolo11s-pose.pt --imgsz 960

Writes data/calib/<name>/poses_<model>_<imgsz>.npz, which tools/calibrate.py reads.
Extracting with several models lets the sweep compare them on identical footage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import Config  # noqa: E402
from runtime_check import check_runtime  # noqa: E402


def cache_path(dir_: Path, weights: str, imgsz: int) -> Path:
    stem = Path(weights).stem.replace("-", "_")
    return dir_ / f"poses_{stem}_{imgsz}.npz"


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default=cfg.get("model.weights", "yolo11m-pose.pt"))
    ap.add_argument("--imgsz", type=int, default=int(cfg.get("model.imgsz", 1280)))
    ap.add_argument("--conf", type=float, default=float(cfg.get("model.person_conf", 0.3)))
    ap.add_argument("--tracker", default=cfg.get("model.tracker", "botsort.yaml"))
    ap.add_argument("--force", action="store_true", help="re-extract even if cached")
    args = ap.parse_args(argv)

    rec_dir = ROOT / "data" / "calib" / args.name
    video = rec_dir / "video.avi"
    labels_path = rec_dir / "labels.json"
    if not video.exists() or not labels_path.exists():
        print(f"no recording at {rec_dir}")
        print("record one first:  python tools/record.py --name " + args.name)
        return 1

    out = cache_path(rec_dir, args.weights, args.imgsz)
    if out.exists() and not args.force:
        print(f"already extracted: {out.name}  (use --force to redo)")
        return 0

    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    frame_times = labels.get("frame_times") or []

    rep = check_runtime(require_cuda=True, allow_cpu_fallback=True)
    print("  " + rep.summary())
    if not rep.ok:
        return 1
    device = "cpu" if rep.using_cpu else cfg.get("model.device", "cuda:0")
    quantize = 16 if (rep.fp16_ok and not rep.using_cpu) else None

    # Resolve a project-relative tracker yaml the same way VideoWorker does.
    tracker = args.tracker
    cand = ROOT / tracker
    if cand.exists():
        tracker = str(cand)

    from ultralytics import YOLO

    print(f"  loading {args.weights} @ imgsz {args.imgsz}")
    model = YOLO(args.weights)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"could not open {video}")
        return 1

    rows_frame: list[int] = []
    rows_t: list[float] = []
    rows_tid: list[int] = []
    rows_bbox: list[list[float]] = []
    rows_kp: list[np.ndarray] = []

    n = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        # Use the timestamp recorded live, not frame_index / nominal_fps. The
        # camera on this machine delivers 15 FPS with jitter, and the temporal
        # filter is driven by real elapsed time - feeding it idealized timestamps
        # would calibrate against a frame rate that never existed.
        t = frame_times[n] if n < len(frame_times) else (n / 15.0)

        res = model.track(
            frame, persist=True, tracker=tracker, imgsz=args.imgsz, device=device,
            quantize=quantize, conf=args.conf, classes=[0],
            max_det=int(cfg.get("model.max_det", 60)), verbose=False,
        )
        n += 1
        if res and res[0].boxes is not None and res[0].keypoints is not None \
                and len(res[0].boxes):
            r = res[0]
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids = r.boxes.id
            ids = ids.cpu().numpy().astype(int) if ids is not None else None
            kps = r.keypoints.data.cpu().numpy()
            if kps.ndim == 3 and kps.shape[2] == 2:
                kps = np.concatenate(
                    [kps, np.ones((*kps.shape[:2], 1), dtype=kps.dtype)], axis=2
                )
            for i in range(len(xyxy)):
                rows_frame.append(n - 1)
                rows_t.append(float(t))
                rows_tid.append(int(ids[i]) if ids is not None else -1)
                rows_bbox.append([float(v) for v in xyxy[i]])
                rows_kp.append(kps[i].astype(np.float32))

        if n % 100 == 0:
            el = time.perf_counter() - t0
            print(f"    {n} frames, {len(rows_frame)} detections, "
                  f"{n / max(el, 1e-6):.1f} FPS", flush=True)

    cap.release()
    elapsed = time.perf_counter() - t0

    if not rows_frame:
        print("  no people detected anywhere in the recording.")
        print("  Check camera placement and lighting before calibrating thresholds -")
        print("  no threshold can recover a subject the detector never sees.")

    np.savez_compressed(
        out,
        frame=np.array(rows_frame, dtype=np.int32),
        t=np.array(rows_t, dtype=np.float64),
        track_id=np.array(rows_tid, dtype=np.int32),
        bbox=np.array(rows_bbox, dtype=np.float32).reshape(-1, 4),
        kp=np.array(rows_kp, dtype=np.float32).reshape(-1, 17, 3),
        meta=np.array([json.dumps({
            "name": args.name,
            "weights": args.weights,
            "imgsz": args.imgsz,
            "conf": args.conf,
            "tracker": args.tracker,
            "device": device,
            "quantize": quantize,
            "frames": n,
            "detections": len(rows_frame),
            "extract_fps": round(n / max(elapsed, 1e-6), 2),
        })]),
    )

    print()
    print(f"  {out}")
    print(f"  {n} frames, {len(rows_frame)} detections, "
          f"{len(set(rows_tid))} distinct track ids, {elapsed:.0f}s")
    print()
    print("next:  python tools/calibrate.py --name " + args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
