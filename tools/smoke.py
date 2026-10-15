"""Headless end-to-end pipeline check. No GUI, no window - safe over RDP.

Exercises the whole Phase 1 stack together: runtime gate -> camera -> pose model
-> raise scoring -> temporal tracker -> gating -> liveness watchdog, and reports
what each stage actually produced.

    python tools/smoke.py --seconds 10

Stand in front of the camera and raise a hand to see the score climb. With nobody
in frame it still validates the pipeline: frames flowing, FPS, and liveness.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config  # noqa: E402
from gating import AttendanceGate  # noqa: E402
from raise_detector import TemporalRaiseTracker  # noqa: E402
from runtime_check import check_runtime  # noqa: E402
from video_worker import VideoWorker  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--startup-timeout", type=float, default=180.0)
    ap.add_argument("--camera", default=None,
                    help="local camera index or phone address, e.g. 192.168.43.1:8080")
    args = ap.parse_args(argv)

    cfg = Config.load(ROOT / "config.json")

    print("=" * 72)
    print("PIPELINE SMOKE TEST")
    print("=" * 72)

    rep = check_runtime(require_cuda=True)
    print("  " + rep.summary())
    if not rep.ok:
        for line in rep.report_lines():
            print("  " + line)
        return 1

    worker = VideoWorker(cfg, runtime_report=rep)
    if args.camera:
        worker.set_source(args.camera)
    tracker = TemporalRaiseTracker(cfg)
    gate = AttendanceGate(cfg)

    print(f"  model {cfg.require('model.weights')} @ imgsz {cfg.require('model.imgsz')}")
    print("  starting pipeline...")
    worker.start()

    deadline = time.monotonic() + args.startup_timeout
    last_stage = ""
    while time.monotonic() < deadline:
        st = worker.status()
        if st["stage"] != last_stage:
            print(f"    [{st['stage']}]")
            last_stage = st["stage"]
        if st["ready"]:
            break
        if st["error"]:
            print(f"  ERROR: {st['error']}")
            worker.stop()
            return 1
        time.sleep(0.1)
    else:
        print("  TIMEOUT waiting for the pipeline")
        worker.stop()
        return 1

    st = worker.status()
    print(f"    camera : {st['camera']}")
    print(f"    fp16   : {worker.half} (quantize={worker.quantize})")
    print(f"    device : {worker.infer_device}")
    print(f"    tracker: {worker.tracker}")
    print()
    print(f"  sampling for {args.seconds:.0f}s ...")

    # --- sample -----------------------------------------------------------
    t_end = time.monotonic() + args.seconds
    last_index = -1
    fps_samples: list[float] = []
    infer_samples: list[float] = []
    people_max = 0
    frames = 0
    best_score = 0.0
    best_ema = 0.0
    confirmed_seen = 0
    stalls = 0
    track_ids: set[int] = set()
    untracked = 0

    tracker.arm(time.monotonic())
    while time.monotonic() < t_end:
        snap = worker.snapshot()
        now = time.monotonic()
        if snap.frame is not None and snap.frame_index != last_index:
            last_index = snap.frame_index
            frames += 1
            obs = {}
            for p in snap.people:
                if p.track_id is None:
                    untracked += 1
                    continue
                track_ids.add(p.track_id)
                obs[p.track_id] = (p.score, p.bbox)
                best_score = max(best_score, p.score.score)
            tracker.update(snap.at, obs)
            people_max = max(people_max, len(snap.people))
            if snap.fps > 0:
                fps_samples.append(snap.fps)
            infer_samples.append(snap.infer_ms)
            for stt in tracker.states.values():
                best_ema = max(best_ema, stt.ema)
            conf = tracker.confirmed()
            if conf:
                confirmed_seen = max(confirmed_seen, len(conf))
                d = gate.evaluate(conf, "SMOKE-TEST")
                print(f"      {d.verdict.value.upper()}: {d.message}")
        if not worker.is_live(now):
            stalls += 1
        time.sleep(0.005)

    status = worker.status()
    worker.stop()

    # --- report -----------------------------------------------------------
    print()
    print("  results")
    print("  " + "-" * 60)
    print(f"    frames processed      {frames}")
    if fps_samples:
        print(f"    FPS  mean / min / max {statistics.mean(fps_samples):.1f} / "
              f"{min(fps_samples):.1f} / {max(fps_samples):.1f}")
    if infer_samples:
        print(f"    inference ms mean/p95 {statistics.mean(infer_samples):.1f} / "
              f"{sorted(infer_samples)[int(len(infer_samples) * 0.95) - 1]:.1f}")
    print(f"    max people in frame   {people_max}")
    print(f"    distinct track ids    {len(track_ids)}")
    print(f"    detections w/o an id  {untracked}")
    print(f"    best raw raise score  {best_score:.3f}")
    print(f"    best smoothed (ema)   {best_ema:.3f}")
    print(f"    confirmed raises      {confirmed_seen}")
    print(f"    liveness gaps seen    {stalls}")
    print(f"    final stage           {status['stage']}")
    if status["error"]:
        print(f"    last error            {status['error']}")

    # --- verdict ----------------------------------------------------------
    #
    # Judging on an absolute FPS number is wrong: when the camera is the limit,
    # no amount of GPU will raise it, and failing the run would send you chasing
    # a model problem that does not exist. What actually matters is whether
    # inference can keep up with whatever the camera delivers.
    print()
    problems: list[str] = []
    notes: list[str] = []
    mean_fps = statistics.mean(fps_samples) if fps_samples else 0.0
    mean_infer = statistics.mean(infer_samples) if infer_samples else 0.0
    infer_capacity = 1000.0 / mean_infer if mean_infer > 0 else 0.0
    cap_fps = status.get("capture_fps") or 0.0

    print(f"    camera delivers       {cap_fps:.1f} FPS")
    print(f"    inference capacity    {infer_capacity:.1f} FPS ({mean_infer:.1f} ms/frame)")
    print()

    if frames == 0:
        problems.append("no frames were processed")
    if people_max > 0 and untracked > frames:
        problems.append("most detections had no track id - tracking is not working")
    if status["error"]:
        problems.append(f"pipeline reported an error: {status['error']}")

    # Inference must not be the bottleneck. Allow 15% slack for jitter.
    if cap_fps > 0 and infer_capacity > 0 and infer_capacity < cap_fps * 0.85:
        problems.append(
            f"inference ({infer_capacity:.1f} FPS) cannot keep up with the camera "
            f"({cap_fps:.1f} FPS) - reduce imgsz or use a smaller model"
        )
    elif cap_fps > 0:
        notes.append(
            f"inference keeps up with the camera ({infer_capacity:.1f} >= {cap_fps:.1f} FPS)"
        )

    # Short runs measure a cold GPU. Clocks take several seconds of load to ramp
    # (300 MHz idle -> ~1530 MHz), so a 5-second sample can report barely half the
    # real inference capacity and fail for no reason.
    if args.seconds < 8.0:
        notes.append(
            f"--seconds {args.seconds:.0f} is short: the GPU needs several seconds "
            "of load to reach full clocks, so inference capacity is understated "
            "here. Use 10s or more for a figure you can act on."
        )

    # A slow camera is not a failure, but it is worth knowing about.
    if 0 < cap_fps < 20.0:
        notes.append(
            f"the camera caps at {cap_fps:.0f} FPS. The detector is frame-rate "
            "independent by design so this still works, but a 30 FPS 1080p USB "
            "webcam would roughly halve the confirmation latency."
        )

    print("=" * 72)
    if problems:
        for p in problems:
            print(f"  PROBLEM: {p}")
        for n in notes:
            print(f"  note: {n}")
        return 1
    print("  pipeline OK")
    for n in notes:
        print(f"  note: {n}")
    if people_max == 0:
        print("  (nobody was in frame, so detection and scoring were not exercised -")
        print("   re-run standing in view and raising a hand to check those)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
