"""Audible check of the roll-call voice. Needs speakers; no camera or GPU.

Speaks the first few calls exactly as a session would and times each one. The
timing is the point: pyttsx3's silent-repeat bug (see src/speech.py) reported
every utterance after the first as "spoken" in ~0.15 s without playing anything,
and it went unnoticed because nothing failed. An utterance that finishes much
faster than its text could be read aloud is flagged here as SILENT.

    python tools/voice_check.py
    python tools/voice_check.py --roster "C:\\path\\to\\roster.csv" --count 5
    python tools/voice_check.py --mode short_unique
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config  # noqa: E402
from roster import Student, assign_calls, load_csv  # noqa: E402
from speech import SpeechWorker  # noqa: E402

# Generous lower bound on speaking speed. Real speech at 175 wpm is ~2.9 words/s;
# anything above this rate means the audio cannot actually have played.
MAX_PLAUSIBLE_WORDS_PER_SEC = 8.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", default=None, help="roster CSV (default: sample rolls)")
    ap.add_argument("--count", type=int, default=3, help="how many calls to speak")
    ap.add_argument("--mode", default=None, choices=("full", "short_unique"),
                    help="override speech.mode")
    args = ap.parse_args(argv)

    cfg = Config.load()
    mode = args.mode or cfg.get("speech.mode", "full")
    if args.roster:
        students = load_csv(args.roster)
    else:
        students = [Student(roll=f"SP25-BCS-{i:03d}") for i in (1, 7, 12, 20)]
    assign_calls(students, mode)
    students = students[: max(1, args.count)]

    worker = SpeechWorker(cfg)
    worker.start()
    t0 = time.monotonic()
    while not worker.available and worker.error is None and time.monotonic() - t0 < 15:
        time.sleep(0.05)
    if not worker.available:
        print(f"FAIL  speech unavailable: {worker.error or 'timed out starting'}")
        return 1
    print(f"engine {worker.backend} | voice {worker.voice_name or '?'} | "
          f"rate {worker.rate} wpm | mode {mode}")
    print(f"warm-up announcement took {time.monotonic() - t0:.2f}s\n")

    failures = 0
    for token, st in enumerate(students, start=1):
        start = time.monotonic()
        worker.speak(st.spoken_call, token)
        ev = None
        while ev is None and time.monotonic() - start < 25:
            ev = next((e for e in worker.poll() if e.token == token), None)
            time.sleep(0.02)
        took = time.monotonic() - start
        words = len(st.spoken_call.replace(",", " ").split())
        floor = words / MAX_PLAUSIBLE_WORDS_PER_SEC
        if ev is None or ev.kind != "spoken":
            verdict = f"FAIL ({ev.kind if ev else 'no completion event'})"
            failures += 1
        elif took < floor:
            verdict = f"SILENT (finished in {took:.2f}s, needs >= {floor:.2f}s)"
            failures += 1
        else:
            verdict = "ok"
        print(f"  {st.roll:<16} \"{st.spoken_call}\"  {took:5.2f}s  {verdict}")

    worker.stop()
    print()
    if failures:
        print(f"{failures} call(s) did not play")
        return 1
    print("every call played at full length")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
