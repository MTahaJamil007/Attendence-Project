"""UI construction and rendering test. Opens real windows briefly, then closes them.

Does not need a camera, a GPU or a model: the video pipeline is stubbed, so this
runs in about a second and exercises every rendering path - including the phases
that are hard to reach by hand, like STALLED and AMBIGUOUS.

The point is to catch tkinter and layout errors (bad geometry, missing canvas item,
a font that will not scale) before they surface in front of a class.

Run:  .venv\\Scripts\\python.exe tools\\test_ui.py
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from config import Config  # noqa: E402
from gating import AttendanceGate  # noqa: E402
from raise_detector import TemporalRaiseTracker  # noqa: E402
from roster import Student, assign_short_unique_calls  # noqa: E402
from session import Phase, Session  # noqa: E402
from video_worker import FrameResult  # noqa: E402

CFG = Config.load(ROOT / "config.json")

_results: list[tuple[bool | None, str, str]] = []


def check(cond: bool, name: str, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


def section(title: str) -> None:
    _results.append((None, title, ""))


# --- stubs ----------------------------------------------------------------
class StubVideo:
    """Enough of VideoWorker for the console to render against."""

    def __init__(self, with_frame: bool = True):
        self.half = True
        self.quantize = 16
        self.tracker = "stub"
        self.model_name = "stub-pose"
        self._frame = (
            np.full((480, 640, 3), 40, dtype=np.uint8) if with_frame else None
        )
        self.live = True
        self.started = False
        self.switched = 0

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def snapshot(self):
        return FrameResult(frame=self._frame, people=[], at=0.0, infer_ms=12.0,
                           fps=30.0, frame_index=1)

    def status(self):
        return {
            "stage": "running", "ready": True, "error": None, "fps": 30.0,
            "capture_fps": 30.0, "infer_ms": 12.0, "model": "stub-pose",
            "half": True, "camera": "Stub 640x480", "cameras": ["Stub 640x480"],
            "live": self.live, "stall_age": 0.0 if self.live else 9.9,
        }

    def may_record_absence(self, now):
        return (True, "") if self.live else (False, "perception pipeline stalled")

    def is_live(self, now=None):
        return self.live

    def stall_age(self, now=None):
        return 0.0 if self.live else 9.9

    def request_next_camera(self):
        self.switched += 1


class StubSpeech:
    available = True
    error = None

    def start(self):
        pass

    def stop(self):
        pass

    def speak(self, text, token):
        pass

    def poll(self):
        return []

    def clear(self):
        pass


def make_students(n=6):
    st = [Student(roll=f"SP25-BCS-{i + 1:03d}", name=f"Student {i + 1}") for i in range(n)]
    assign_short_unique_calls(st)
    return st


# =========================================================================
def test_roll_display_all_phases() -> None:
    section("student display renders in every phase")
    from ui.roll_display import RollDisplay

    root = tk.Tk()
    root.withdraw()
    disp = RollDisplay(root, CFG)
    check(disp.alive, "display window created")

    video = StubVideo()
    session = Session(CFG, video, TemporalRaiseTracker(CFG), AttendanceGate(CFG), StubSpeech())
    session.set_students(make_students())

    rendered = []
    for phase, result, remaining, hold, score in (
        (Phase.IDLE, "", 0.0, 0.0, 0.0),
        (Phase.CALLING, "", 0.0, 0.0, 0.0),
        (Phase.RESPONSE, "", 6.0, 0.0, 0.0),
        (Phase.RESPONSE, "", 3.2, 0.0, 0.30),   # partial, amber bar
        (Phase.RESPONSE, "", 1.1, 0.65, 0.90),  # holding, green bar
        (Phase.STALLED, "", 4.0, 0.0, 0.0),
        (Phase.AMBIGUOUS, "TWO HANDS", 0.0, 0.0, 0.9),
        (Phase.RESULT, "PRESENT", 0.0, 0.0, 0.9),
        (Phase.RESULT, "ABSENT", 0.0, 0.0, 0.0),
        (Phase.PAUSED, "", 0.0, 0.0, 0.0),
        (Phase.FINISHED, "COMPLETE", 0.0, 0.0, 0.0),
    ):
        session.phase = phase
        session.current = 2
        session.result_text = result
        v = session.view(0.0)
        v.remaining = remaining
        v.window = 6.0 if remaining else 0.0
        v.hold_progress = hold
        v.best_score = score
        try:
            disp.update(v)
            root.update()
            rendered.append(phase.value)
        except Exception as exc:
            check(False, f"render {phase.value}", f"{type(exc).__name__}: {exc}")
    check(len(rendered) == 11, "all 11 phase renders succeeded", f"{len(rendered)}/11")

    # Resizing must relayout without error - the projector is a different size
    # from the laptop screen, and font scaling runs on <Configure>.
    ok = True
    for w, h in ((520, 340), (1920, 1080), (800, 1200)):
        try:
            disp.win.geometry(f"{w}x{h}")
            root.update()
            disp._layout()
            disp.update(session.view(0.0))
            root.update()
        except Exception as exc:
            ok = False
            check(False, f"relayout at {w}x{h}", f"{type(exc).__name__}: {exc}")
    check(ok, "relayouts cleanly at projector, laptop and portrait sizes")

    disp.close()
    root.update()
    check(not disp.alive, "closes cleanly")
    root.destroy()


def test_console_builds_and_ticks() -> None:
    section("teacher console builds and runs its loop")
    from ui.console import Console

    root = tk.Tk()
    video = StubVideo()
    tracker = TemporalRaiseTracker(CFG)
    gate = AttendanceGate(CFG)
    speech = StubSpeech()
    session = Session(CFG, video, tracker, gate, speech)
    session.set_students(make_students(8))

    try:
        console = Console(root, CFG, video, tracker, gate, speech, session)
        check(True, "console constructed")
    except Exception as exc:
        check(False, "console constructed", f"{type(exc).__name__}: {exc}")
        root.destroy()
        return

    # Run the real after() loop for a while.
    errors: list[str] = []
    for _ in range(20):
        try:
            root.update()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    check(not errors, "20 UI ticks with no exception", errors[0] if errors else "")
    check(
        "UI error" not in console.lbl_status.cget("text"),
        "status bar reports no UI error",
        console.lbl_status.cget("text")[:70],
    )

    # A stalled pipeline must render, not crash.
    video.live = False
    session.phase = Phase.RESPONSE
    session.start(0.0)
    for _ in range(10):
        root.update()
    check("UI error" not in console.lbl_status.cget("text"), "renders while stalled")
    video.live = True

    # Roster panel actually populated.
    rows = console.tree.get_children()
    check(len(rows) == 8, "roster shows every student", f"{len(rows)} rows")

    # Manual marking through the UI path.
    session.current = 0
    console.manual(True)
    check(session.students[0].status == "Present", "Present button marks the student",
          session.students[0].status)
    console.undo()
    check(session.students[0].status == "Pending", "Undo button reverts it",
          session.students[0].status)

    # Camera switch request reaches the worker.
    console._devices_started = True
    console.switch_camera()
    check(video.switched == 1, "Switch Camera reaches the video worker")

    # No-frame path (camera not started yet) must not crash the renderer.
    console.video = StubVideo(with_frame=False)
    for _ in range(5):
        root.update()
    check("UI error" not in console.lbl_status.cget("text"),
          "renders with no camera frame",
          console.lbl_status.cget("text")[:70])

    try:
        root.destroy()
    except Exception:
        pass


def main() -> int:
    for fn in (test_roll_display_all_phases, test_console_builds_and_ticks):
        try:
            fn()
        except Exception as exc:
            check(False, f"{fn.__name__} crashed", f"{type(exc).__name__}: {exc}")

    print()
    print("=" * 72)
    passed = failed = 0
    for ok, name, detail in _results:
        if ok is None:
            print(f"\n  -- {name} --")
            continue
        suffix = f"   [{detail}]" if detail else ""
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{suffix}")
        if ok:
            passed += 1
        else:
            failed += 1
    print("=" * 72)
    print(f"  {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
