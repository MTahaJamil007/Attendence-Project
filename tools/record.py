"""Record labelled calibration footage using a scripted protocol.

The tool tells the subject what to do and when, and writes the intervals it asked
for as ground truth. Nobody has to label video by hand afterwards, and the
distractor actions - scratching, stretching, waving, raising a phone - get labelled
just as precisely as the genuine raises.

That matters more than it sounds. Recall on genuine raises is easy to measure
informally ("it saw my hand"). Precision against distractors is not, and a detector
that fires on someone scratching their head will quietly mark absent students
present all term.

    python tools/record.py --name classroom-1
    python tools/record.py --name back-row --reps 3 --hold 5

Outputs to data/calib/<name>/:
    video.avi     MJPG, so re-extraction with a different model is possible
    labels.json   segments, per-frame timestamps, camera details

Record from the real camera position, with the real class if you can. A recording
made at your desk calibrates your desk.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402

from camera import discover_cameras, open_camera  # noqa: E402
from config import Config  # noqa: E402

# (label, on-screen prompt, expect_raise)
# Distractors outnumber genuine raises deliberately: false positives are the
# failure mode that is hard to notice and expensive to get wrong.
PROTOCOL: list[tuple[str, str, bool]] = [
    ("idle", "SIT NORMALLY", False),
    ("raise", "RAISE YOUR HAND", True),
    ("idle", "PUT YOUR HAND DOWN", False),
    ("scratch", "SCRATCH YOUR HEAD", False),
    ("idle", "SIT NORMALLY", False),
    ("stretch", "STRETCH YOUR ARMS OUT", False),
    ("idle", "SIT NORMALLY", False),
    ("raise", "RAISE YOUR HAND", True),
    ("idle", "PUT YOUR HAND DOWN", False),
    ("wave", "WAVE AT THE CAMERA", False),
    ("idle", "SIT NORMALLY", False),
    ("phone", "HOLD A PHONE UP TO YOUR FACE", False),
    ("idle", "SIT NORMALLY", False),
    ("lean", "LEAN ON YOUR HAND / ELBOW ON DESK", False),
    ("idle", "SIT NORMALLY", False),
    ("raise_high", "RAISE YOUR HAND HIGH AND STRAIGHT", True),
    ("idle", "PUT YOUR HAND DOWN", False),
    ("raise_low", "RAISE YOUR HAND JUST ABOVE YOUR SHOULDER", True),
    ("idle", "SIT NORMALLY", False),
]

GREEN = (110, 220, 130)
RED = (110, 110, 250)
AMBER = (60, 190, 250)
WHITE = (255, 255, 255)
DARK = (28, 28, 32)


def draw_prompt(frame, prompt: str, remaining: float, total: float, expect: bool,
                seg_i: int, seg_n: int, recording: bool):
    h, w = frame.shape[:2]
    view = frame.copy()

    # Dim the image so the instruction is unmissable from across a room.
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, view, 0.65, 0, view)

    colour = GREEN if expect else AMBER
    scale = max(0.9, w / 900.0)
    (tw, th), _ = cv2.getTextSize(prompt, cv2.FONT_HERSHEY_SIMPLEX, scale * 1.4, 3)
    cv2.putText(view, prompt, ((w - tw) // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 1.4, colour, 3, cv2.LINE_AA)

    # Countdown bar.
    frac = 0.0 if total <= 0 else max(0.0, min(1.0, remaining / total))
    cv2.rectangle(view, (0, 0), (w, 14), DARK, -1)
    cv2.rectangle(view, (0, 0), (int(w * frac), 14), colour, -1)

    tag = "EXPECT: RAISE" if expect else "EXPECT: NO RAISE"
    cv2.putText(view, tag, (12, 44), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6,
                colour, 2, cv2.LINE_AA)
    cv2.putText(view, f"{seg_i + 1}/{seg_n}   {remaining:0.1f}s",
                (12, h - 18), cv2.FONT_HERSHEY_SIMPLEX, scale * 0.6, WHITE, 2, cv2.LINE_AA)
    if recording:
        cv2.circle(view, (w - 28, 40), 10, RED, -1)
        cv2.putText(view, "REC", (w - 92, 47), cv2.FONT_HERSHEY_SIMPLEX,
                    scale * 0.6, RED, 2, cv2.LINE_AA)
    return view


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="recording name, e.g. classroom-1")
    ap.add_argument("--hold", type=float, default=4.0, help="seconds per segment")
    ap.add_argument("--idle", type=float, default=2.5, help="seconds per idle segment")
    ap.add_argument("--reps", type=int, default=1, help="repeat the protocol N times")
    ap.add_argument("--camera", type=int, default=None, help="camera index (default: auto)")
    ap.add_argument("--lead-in", type=float, default=4.0, help="countdown before recording")
    args = ap.parse_args(argv)

    cfg = Config.load()
    out_dir = ROOT / "data" / "calib" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = args.camera
    if idx is None:
        cams = discover_cameras(cfg)
        if not cams:
            print("no camera found")
            return 1
        idx = cams[0].index
    cap, info = open_camera(idx, cfg)
    if cap is None or info is None:
        print(f"could not open camera {idx}")
        return 1
    print(f"recording from {info.label()}")

    # MJPG/AVI rather than mp4v: it needs no extra codec on Windows and is only
    # lightly compressed, so re-extracting poses later sees almost the same pixels
    # the detector would have seen live.
    video_path = out_dir / "video.avi"
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(video_path), fourcc, 15.0, (info.width, info.height))
    if not writer.isOpened():
        print("could not open the video writer")
        cap.release()
        return 1

    segments: list[tuple[str, str, bool, float]] = []
    for _ in range(max(1, args.reps)):
        for label, prompt, expect in PROTOCOL:
            secs = args.idle if label == "idle" else args.hold
            segments.append((label, prompt, expect, secs))

    win = "Calibration recording - Q to abort"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1100, 640)

    print(f"{len(segments)} segments, about {sum(s[3] for s in segments):.0f}s total")
    print("follow the on-screen prompts; press Q to abort")

    frame_times: list[float] = []
    records: list[dict] = []
    aborted = False
    t_zero = None

    try:
        # Lead-in so the subject can get into position before anything is recorded.
        lead_end = time.monotonic() + args.lead_in
        while time.monotonic() < lead_end:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            remaining = lead_end - time.monotonic()
            view = draw_prompt(frame, "GET READY", remaining, args.lead_in, False,
                               0, len(segments), recording=False)
            cv2.imshow(win, view)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                aborted = True
                break

        if not aborted:
            t_zero = time.monotonic()
            for i, (label, prompt, expect, secs) in enumerate(segments):
                seg_start = time.monotonic() - t_zero
                seg_end_at = time.monotonic() + secs
                while time.monotonic() < seg_end_at:
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    now = time.monotonic() - t_zero
                    writer.write(frame)
                    frame_times.append(round(now, 4))
                    view = draw_prompt(frame, prompt, seg_end_at - time.monotonic(),
                                       secs, expect, i, len(segments), recording=True)
                    cv2.imshow(win, view)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        aborted = True
                        break
                if aborted:
                    break
                records.append({
                    "kind": label,
                    "prompt": prompt,
                    "expect_raise": expect,
                    "start": round(seg_start, 3),
                    "end": round(time.monotonic() - t_zero, 3),
                })
    finally:
        writer.release()
        cap.release()
        cv2.destroyAllWindows()

    if aborted and not records:
        print("aborted before anything was recorded")
        try:
            video_path.unlink()
        except Exception:
            pass
        return 1

    duration = frame_times[-1] if frame_times else 0.0
    labels = {
        "name": args.name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "aborted": aborted,
        "camera": info.label(),
        "width": info.width,
        "height": info.height,
        "frames": len(frame_times),
        "duration_sec": round(duration, 2),
        "measured_fps": round(len(frame_times) / duration, 2) if duration > 0 else 0.0,
        "frame_times": frame_times,
        "segments": records,
    }
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    genuine = sum(1 for r in records if r["expect_raise"])
    print()
    print(f"  video     {video_path}")
    print(f"  labels    {out_dir / 'labels.json'}")
    print(f"  frames    {len(frame_times)} over {duration:.1f}s "
          f"({labels['measured_fps']:.1f} FPS)")
    print(f"  segments  {len(records)}  ({genuine} genuine raises, "
          f"{len(records) - genuine} negatives)")
    if aborted:
        print("  NOTE: recording was aborted early; partial data saved")
    print()
    print("next:")
    print(f"  python tools/extract.py --name {args.name}")
    print(f"  python tools/calibrate.py --name {args.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
