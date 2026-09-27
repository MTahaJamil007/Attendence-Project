"""Application entry point.

Runs the Phase 0 runtime gate before building any UI, and refuses to start a live
session on CPU. The predecessor shipped a CPU-only build that ran pose inference at
3-8 FPS, which made its decision rule unsatisfiable and produced roughly 30%
recall; the whole point of the gate is that this can no longer happen silently.

    python src/main.py              normal run
    python src/main.py --allow-cpu  degraded, for UI work without a GPU
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import Config  # noqa: E402
from gating import AttendanceGate  # noqa: E402
from raise_detector import TemporalRaiseTracker  # noqa: E402
from runtime_check import check_runtime  # noqa: E402
from session import Session  # noqa: E402
from speech import SpeechWorker  # noqa: E402
from video_worker import VideoWorker  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="start even without CUDA (too slow for a real session)")
    ap.add_argument("--config", default=None, help="path to an alternative config.json")
    ap.add_argument("--roster", default=None, help="load this roster CSV at startup")
    args = ap.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config.load()

    print("checking runtime...")
    rep = check_runtime(require_cuda=True, allow_cpu_fallback=args.allow_cpu)
    for line in rep.report_lines():
        print("  " + line)
    print(rep.summary())
    if not rep.ok:
        # Show the failure in a dialog too: this is launched from a shortcut as
        # often as from a terminal, and a silent console exit looks like a crash.
        try:
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "Runtime check failed",
                rep.summary()
                + "\n\nInstall CUDA PyTorch with:\n"
                  "pip install torch torchvision "
                  "--index-url https://download.pytorch.org/whl/cu126",
            )
            root.destroy()
        except Exception:
            pass
        return 1

    video = VideoWorker(cfg, runtime_report=rep)
    tracker = TemporalRaiseTracker(cfg)
    gate = AttendanceGate(cfg)
    speech = SpeechWorker(cfg)

    exporter = None
    try:  # Phase 3; absent for now, and the console degrades gracefully.
        from export import Exporter

        exporter = Exporter(cfg)
    except Exception:
        exporter = None

    evidence = None
    try:  # Phase 3
        from evidence import EvidenceStore

        evidence = EvidenceStore(cfg)
    except Exception:
        evidence = None

    session = Session(cfg, video, tracker, gate, speech, evidence=evidence)

    if args.roster:
        from roster import load_csv

        try:
            session.set_students(load_csv(args.roster))
            print(f"loaded {len(session.students)} students from {args.roster}")
        except Exception as exc:
            print(f"could not load roster: {exc}")

    # Seat zones persist across sessions, so Rule C can flag a student answering
    # from a very different place than last time.
    seatmap = Path(cfg.get("output.dir") or ROOT.parent / "data") / "seatmap.json"
    try:
        n = gate.load_zones(seatmap)
        if n:
            print(f"loaded {n} seat zones")
    except Exception:
        pass

    from ui.console import Console

    root = tk.Tk()
    console = Console(root, cfg, video, tracker, gate, speech, session, exporter=exporter)
    try:
        root.mainloop()
    finally:
        try:
            gate.save_zones(seatmap)
        except Exception:
            pass
        try:
            video.stop()
        except Exception:
            pass
        try:
            speech.stop()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
