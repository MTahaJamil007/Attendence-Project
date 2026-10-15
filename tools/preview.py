"""Live detector preview - the Phase 1 validation tool.

Run this in the actual classroom, from the actual camera position, before trusting
any threshold. It shows every person's smoothed score against the enter threshold,
so you can see at a glance whether the back row is scoring 0.8 or 0.4.

    python tools/preview.py

Keys:
    SPACE  arm a response window (simulates a roll number being called)
    r      reset all tracker state
    s      save the current annotated frame to logs/
    n      switch to the next camera
    q/ESC  quit

What to look for:
    * FPS >= 25 in the header. Below that, fix the model/imgsz first.
    * Every seat you care about gets a skeleton. If the back row has no skeleton,
      that is a camera placement problem and no threshold will fix it.
    * A genuine raise crosses the white tick on the score bar within ~1 second.
    * Stretching, scratching and leaning stay left of the tick.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from annotate import annotate  # noqa: E402
from config import Config  # noqa: E402
from gating import AttendanceGate  # noqa: E402
from raise_detector import TemporalRaiseTracker  # noqa: E402
from runtime_check import check_runtime  # noqa: E402
from video_worker import VideoWorker  # noqa: E402

WINDOW = "Hand-Raise Preview  [SPACE arm] [r reset] [s save] [n next cam] [q quit]"


def main() -> int:
    cfg = Config.load(ROOT / "config.json")

    print("checking runtime...")
    rep = check_runtime(require_cuda=True, allow_cpu_fallback=True)
    print("  " + rep.summary())
    if not rep.ok:
        for line in rep.report_lines():
            print("  " + line)
        return 1
    if rep.using_cpu:
        print("  WARNING: no GPU. Frame rate will be too low for a real session.")

    worker = VideoWorker(cfg, runtime_report=rep)
    if len(sys.argv) > 1:
        # run.bat preview 192.168.43.1:8080 - preview a phone camera
        worker.set_source(sys.argv[1])
    tracker = TemporalRaiseTracker(cfg)
    gate = AttendanceGate(cfg)

    print("starting camera and model (first run downloads the weights)...")
    worker.start()

    # Wait for the pipeline to come up, surfacing whatever stage it is on.
    deadline = time.monotonic() + 180.0
    last_stage = ""
    while time.monotonic() < deadline:
        st = worker.status()
        if st["stage"] != last_stage:
            print(f"  [{st['stage']}]")
            last_stage = st["stage"]
        if st["ready"]:
            break
        if st["error"]:
            print(f"  ERROR: {st['error']}")
            worker.stop()
            return 1
        time.sleep(0.15)
    else:
        print("  timed out waiting for the pipeline")
        worker.stop()
        return 1

    print(f"  {worker.status()['camera']}")
    print(f"  model {worker.model_name} | FP16={worker.half}")
    print("  ready - press SPACE to arm a window, q to quit")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)

    window_sec = float(cfg.get("session.response_window_sec", 6.0))
    armed_until = 0.0
    last_index = -1
    saved = 0

    try:
        while True:
            snap = worker.snapshot()
            now = time.monotonic()

            # Fold in each frame exactly once. Updating the EMA twice for the same
            # observation would distort the time constant the whole design relies on.
            if snap.frame is not None and snap.frame_index != last_index:
                last_index = snap.frame_index
                obs = {
                    p.track_id: (p.score, p.bbox)
                    for p in snap.people
                    if p.track_id is not None
                }
                tracker.update(snap.at, obs)

            live = worker.is_live(now)
            confirmed = tracker.confirmed()
            armed = now < armed_until

            bits = [
                f"FPS {snap.fps:5.1f}",
                f"{snap.infer_ms:5.1f}ms",
                f"people {len(snap.people):2d}",
                f"tracks {len(tracker.states):2d}",
                f"raised {len(confirmed):2d}",
            ]
            if armed:
                bits.append(f"WINDOW {armed_until - now:4.1f}s")
            bits.append("LIVE" if live else f"STALLED {worker.stall_age(now):.1f}s")
            header = "   ".join(bits)

            if snap.frame is None:
                time.sleep(0.02)
                continue

            view = annotate(snap.frame, snap, cfg, track_states=tracker.states, header=header)

            # The watchdog, made visible. In a session this state blocks any
            # automatic Absent instead of letting the timer run on regardless.
            if not live:
                h, w = view.shape[:2]
                cv2.rectangle(view, (0, 32), (w, 76), (25, 25, 120), -1)
                cv2.putText(
                    view,
                    "PIPELINE STALLED - marking is suspended",
                    (12, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if armed and confirmed:
                decision = gate.evaluate(confirmed, "PREVIEW")
                h, w = view.shape[:2]
                cv2.rectangle(view, (0, 78), (w, 118), (30, 90, 30), -1)
                cv2.putText(
                    view,
                    f"{decision.verdict.value.upper()}: {decision.message}",
                    (12, 106),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                tracker.arm(time.monotonic())
                armed_until = time.monotonic() + window_sec
                print(f"  armed for {window_sec:.1f}s")
            elif key == ord("r"):
                tracker.reset()
                gate.reset_session()
                armed_until = 0.0
                print("  tracker + gate reset")
            elif key == ord("s"):
                out = ROOT / "logs" / f"preview_{int(time.time())}_{saved}.jpg"
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), view)
                saved += 1
                print(f"  saved {out}")
            elif key == ord("n"):
                worker.request_next_camera()
                print("  switching camera...")
    finally:
        cv2.destroyAllWindows()
        worker.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
