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

    def request_source(self, source):
        self.requested = source

    def set_source(self, source):
        self.source = source


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


def test_phone_dialog_and_seat_flag() -> None:
    section("phone camera dialog, GPU status, and a seat flag on the video")
    import time as _time

    from gating import Verdict
    from ui import console as console_mod
    from ui.console import Console

    # Keep the test from touching the real "last phone" file.
    console_mod.LAST_CAMERA_FILE = ROOT / "logs" / "_test_last_camera.json"

    root = tk.Tk()
    root.withdraw()
    video = StubVideo()
    tracker = TemporalRaiseTracker(CFG)
    gate = AttendanceGate(CFG)
    session = Session(CFG, video, tracker, gate, StubSpeech())
    session.set_students(make_students())
    con = Console(root, CFG, video, tracker, gate, StubSpeech(), session)
    root.update()

    names = []

    def walk(w):
        for c in w.winfo_children():
            try:
                names.append(str(c.cget("text")))
            except Exception:
                pass
            walk(c)

    walk(root)
    check("Phone Camera" in names, "toolbar has a Phone Camera button")

    con.phone_camera()
    root.update()
    dlg = con._phone_dialog
    check(dlg is not None and dlg.win.winfo_exists(), "dialog opens")
    dlg.addr.set("192.168.43.1:8080")
    dlg.connect()
    check(getattr(video, "source", None) == "192.168.43.1:8080" and video.started,
          "Connect before Start Camera: sets the source and starts the pipeline",
          str(getattr(video, "source", None)))
    check(con._last_phone() == "192.168.43.1:8080", "the address is remembered for next time")
    con._connect_phone("10.0.0.7")
    check(getattr(video, "requested", None) == "10.0.0.7",
          "Connect while running: switches source live", str(getattr(video, "requested", None)))
    dlg.close()

    # A REUSED_SEAT flag rendered on the operator video, without a UI error.
    from raise_detector import TrackState

    def seated(tid):
        st = TrackState(track_id=tid, ema=0.9, latched=True, latched_at=0.0,
                        confirmed=True, peak_ema=0.9)
        st.bbox = st.best_bbox = (280.0, 150.0, 360.0, 420.0)
        st.anchor = (320.0, 250.0, 60.0)
        return st

    gate.commit(session.students[0].roll, seated(1), now=1.0)
    session.current = 3
    session.phase = Phase.AMBIGUOUS
    session.flag = gate.evaluate([seated(9)], session.students[3].roll)
    tracker.states[9] = seated(9)
    for _ in range(3):
        con._tick()
        root.update()
    status = con.lbl_status.cget("text")
    check("UI error" not in status, "renders the seat flag without error", status[:90])
    check(session.flag is not None and session.flag.verdict is Verdict.REUSED_SEAT,
          "session holds the flag for the overlay")
    if con.gpu.available:
        check("gpu" in status, "status bar shows live GPU load (NVML)", status[-40:])
    con.close()
    try:
        console_mod.LAST_CAMERA_FILE.unlink()
    except Exception:
        pass


def test_layout_is_stable() -> None:
    section("layout: long messages cannot move or hide anything; Roll Screen stays out of the way")
    import time as _time

    from ui.console import Console

    root = tk.Tk()
    video = StubVideo()
    video._frame = np.full((720, 1280, 3), 60, dtype=np.uint8)
    tracker = TemporalRaiseTracker(CFG)
    gate = AttendanceGate(CFG)
    session = Session(CFG, video, tracker, gate, StubSpeech())
    session.set_students(make_students(40))
    con = Console(root, CFG, video, tracker, gate, StubSpeech(), session)
    con._devices_started = True
    root.geometry("1400x860")

    def settle(n=12):
        for _ in range(n):
            root.update()
            _time.sleep(0.02)

    settle()

    def boxes():
        return (con.video_canvas.winfo_width(), con.video_canvas.winfo_height(),
                con.tree.winfo_width())

    before = boxes()
    req_before = root.winfo_reqwidth()

    # The longest teacher message the app produces, and then some.
    session.current = 3
    session.phase = Phase.AMBIGUOUS
    session.blocked_reason = session.message = (
        "2 hands raised (#57 at SP25-BCS-009's seat, #58) - confirm manually; "
        "hand #57 is at the seat already marked for SP25-BCS-009 (0.05 shoulder-widths "
        "away, clothing match 97%) - confirm manually")
    settle(30)
    after = boxes()
    check(after == before, "video and roster keep their size with a long message",
          f"{before} -> {after}")
    check(root.winfo_reqwidth() <= max(req_before, 1200),
          "the window does not ask to grow", f"{req_before} -> {root.winfo_reqwidth()}")

    # Every manual control is on screen inside the window.
    rx, ry = root.winfo_rootx(), root.winfo_rooty()
    rw, rh = root.winfo_width(), root.winfo_height()
    found = []

    def walk(w):
        for c in w.winfo_children():
            try:
                if str(c.cget("text")).startswith(("Present", "Absent", "Undo")):
                    found.append(c)
            except Exception:
                pass
            walk(c)

    walk(root)
    inside = all(
        c.winfo_ismapped()
        and c.winfo_rootx() >= rx and c.winfo_rooty() >= ry
        and c.winfo_rootx() + c.winfo_width() <= rx + rw
        and c.winfo_rooty() + c.winfo_height() <= ry + rh
        for c in found
    )
    check(len(found) == 3 and inside, "Present / Absent / Undo are fully visible",
          f"{len(found)} found")
    detail = con.lbl_detail
    check(detail.winfo_rootx() + detail.winfo_width() <= rx + rw,
          "the message wraps inside the panel instead of running off it")

    # Start Attendance with one screen: the student display opens behind the
    # console and the console keeps the keyboard.
    session.phase = Phase.IDLE
    session.current = -1
    con.start_attendance()
    settle(20)
    check(con.display is not None and con.display.alive, "student display is opened")
    if con.display is not None and not con.display.on_projector:
        focus = root.focus_get()
        check(focus is not None and focus.winfo_toplevel() is root,
              "the console keeps keyboard focus (P / A / U keep working)",
              str(focus))
        order = root.tk.call("wm", "stackorder", ".")
        disp = str(con.display.win)
        check(list(order).index(disp) < list(order).index(".") if disp in order else True,
              "the student display is behind the console, not over the video",
              str(order))
    con.close()
    con.close()  # closing twice (window X, then Ctrl+C) must be harmless
    check(True, "close() is safe to call twice")


def main() -> int:
    for fn in (test_roll_display_all_phases, test_console_builds_and_ticks,
               test_phone_dialog_and_seat_flag, test_layout_is_stable):
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
