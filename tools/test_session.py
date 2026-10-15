"""Tests for the roll-call state machine. No camera, GPU, TTS or window needed.

The real TemporalRaiseTracker and AttendanceGate are used; only the video pipeline
and the speech engine are faked, because those are the two things a test cannot
supply. Time is driven explicitly, so a 50-second stall costs no wall clock.

Run:  .venv\\Scripts\\python.exe tools\\test_session.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config  # noqa: E402
from gating import AttendanceGate  # noqa: E402
from raise_detector import PersonScore, TemporalRaiseTracker  # noqa: E402
from roster import (  # noqa: E402
    ABSENT,
    PENDING,
    PRESENT,
    Student,
    assign_calls,
    assign_short_unique_calls,
    full_call,
    parse_pasted,
)
from session import Phase, Session  # noqa: E402
from speech import SpeechEvent  # noqa: E402

CFG = Config.load(ROOT / "config.json")
BBOX = (200.0, 100.0, 260.0, 360.0)


# --- fakes ----------------------------------------------------------------
class FakeVideo:
    """Stands in for VideoWorker's liveness contract."""

    def __init__(self, live: bool = True):
        self.live = live

    def may_record_absence(self, now: float):
        if self.live:
            return True, ""
        return False, "perception pipeline stalled"


class FakeSpeech:
    def __init__(self, available: bool = True, auto_complete: bool = True):
        self.available = available
        self.auto_complete = auto_complete
        self.spoken: list[tuple[int, str]] = []
        self._events: list[SpeechEvent] = []

    def speak(self, text: str, token: int) -> None:
        self.spoken.append((token, text))
        if self.auto_complete:
            self._events.append(SpeechEvent(token=token, kind="spoken"))

    def poll(self):
        out, self._events = self._events, []
        return out

    def clear(self) -> None:
        self.cleared += 1

    cleared = 0

    def finish(self) -> None:
        """Report the most recent utterance as spoken, for slow-speech tests."""
        if self.spoken:
            self._events.append(SpeechEvent(token=self.spoken[-1][0], kind="spoken"))


def make_session(n: int = 3, live: bool = True, speech: FakeSpeech | None = None, cfg=None):
    students = [Student(roll=f"SP25-BCS-{i + 1:03d}") for i in range(n)]
    assign_short_unique_calls(students)
    video = FakeVideo(live=live)
    cfg = cfg or CFG
    tracker = TemporalRaiseTracker(cfg)
    gate = AttendanceGate(cfg)
    sp = speech if speech is not None else FakeSpeech()
    sess = Session(cfg, video, tracker, gate, sp)
    sess.set_students(students)
    return sess, students, video, tracker, gate, sp


def drive(sess, tracker, t0: float, until: float, score=None, track_id: int = 1,
          step: float = 0.05, on_tick=None):
    """Advance time, feeding a raise score into the tracker each step.

    `score` may be a float, a callable(t) -> float|None, or None for no detection.
    """
    t = t0
    view = None
    while t < until:
        val = score(t) if callable(score) else score
        if val is not None:
            ps = PersonScore()
            ps.score = float(val)
            ps.torso_px = 100.0
            tracker.update(t, {track_id: (ps, BBOX)})
        else:
            tracker.update(t, {})
        view = sess.tick(t)
        if on_tick is not None:
            on_tick(t, view)
        t += step
    return view, t


_results: list[tuple[bool | None, str, str]] = []


def check(cond: bool, name: str, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


def section(title: str) -> None:
    _results.append((None, title, ""))


# =========================================================================
def test_happy_path() -> None:
    section("happy path: raise -> Present -> advance")
    sess, students, _v, tracker, _g, sp = make_session(3)
    check(sess.start(0.0), "start() succeeds with a roster")
    check(sess.phase is Phase.CALLING, "phase is CALLING first", sess.phase)
    check(sp.spoken and sp.spoken[0][1] == "S P two five, B C S, zero zero one",
          "reads the full registration number aloud", str(sp.spoken[:1]))

    view, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    check(sess.phase is Phase.RESPONSE, "window opens after speech + reaction delay", sess.phase)

    view, t = drive(sess, tracker, t, t + 2.0, score=1.0)
    check(students[0].status == PRESENT, "student marked Present", students[0].status)
    check(students[0].track_id == 1, "credited track recorded", str(students[0].track_id))
    check(students[0].score > 0.5, "score recorded", f"{students[0].score}")

    # Result hold, then it should move to the next student.
    view, t = drive(sess, tracker, t, t + 2.0, score=None)
    check(sess.current == 1, "advanced to the second student", f"current={sess.current}")


def test_timeout_absent() -> None:
    section("no raise -> Absent")
    sess, students, _v, tracker, _g, _sp = make_session(2)
    sess.start(0.0)
    view, t = drive(sess, tracker, 0.0, 9.0, score=0.0)
    check(students[0].status == ABSENT, "marked Absent after the window", students[0].status)
    check("Timeout" in students[0].evidence, "evidence explains why", students[0].evidence)


def test_stall_blocks_absence() -> None:
    section("STALL must never become an Absent")
    sess, students, video, tracker, _g, _sp = make_session(2)
    sess.start(0.0)
    view, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    check(sess.phase is Phase.RESPONSE, "window open before the stall")

    # Reproduce the predecessor's 52-second inference freeze.
    video.live = False
    view, t = drive(sess, tracker, t, t + 52.0, score=None, step=0.25)
    check(sess.phase is Phase.STALLED, "phase becomes STALLED", sess.phase)
    check(
        students[0].status == PENDING,
        "student is STILL PENDING after a 52s stall (the predecessor marked Absent)",
        students[0].status,
    )
    check("stalled" in view.blocked_reason, "reason is surfaced", view.blocked_reason)

    # Frames return. The student is owed a full window from this moment.
    video.live = True
    view, t = drive(sess, tracker, t, t + 0.5, score=0.0)
    check(sess.phase is Phase.RESPONSE, "recovers to RESPONSE", sess.phase)
    check(students[0].status == PENDING, "still pending right after recovery")
    check(view.remaining > 4.0, "window was extended by the stall duration",
          f"{view.remaining:.1f}s left")

    # And a raise after recovery is credited normally.
    view, t = drive(sess, tracker, t, t + 2.0, score=1.0)
    check(students[0].status == PRESENT, "raise after recovery still counts",
          students[0].status)


def test_stalled_raise_still_credited() -> None:
    section("a raise during recovery is not lost")
    sess, students, video, tracker, _g, _sp = make_session(1)
    sess.start(0.0)
    _v, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    video.live = False
    _v, t = drive(sess, tracker, t, t + 3.0, score=None, step=0.25)
    video.live = True
    _v, t = drive(sess, tracker, t, t + 2.0, score=1.0)
    check(students[0].status == PRESENT, "Present after stall + raise", students[0].status)


def test_ambiguous_blocks() -> None:
    section("two hands -> AMBIGUOUS, waits for the teacher")
    sess, students, _v, tracker, _g, _sp = make_session(2)
    sess.start(0.0)
    _view, t = drive(sess, tracker, 0.0, 1.0, score=0.0)

    # Two different tracks both raise. One update per frame carrying BOTH tracks,
    # which is how VideoWorker actually feeds the tracker - calling update() once
    # per track would make each look momentarily unseen to the other.
    tt = t
    while tt < t + 2.0:
        obs = {}
        for tid in (1, 2):
            ps = PersonScore()
            ps.score = 1.0
            ps.torso_px = 100.0
            obs[tid] = (ps, BBOX)
        tracker.update(tt, obs)
        view = sess.tick(tt)
        tt += 0.05

    check(sess.phase is Phase.AMBIGUOUS, "phase is AMBIGUOUS", sess.phase)
    check(students[0].status == PENDING, "nothing auto-marked on ambiguity")
    check(view.needs_teacher, "view flags that a teacher is needed")

    # It must stay blocked, not time out into a guess.
    view, tt = drive(sess, tracker, tt, tt + 12.0, score=None)
    check(students[0].status == PENDING, "still unmarked 12s later - never guesses")

    # Teacher resolves it.
    sess.mark_manual(True, tt)
    check(students[0].status == PRESENT, "teacher override marks Present", students[0].status)
    check("Manual" in students[0].evidence, "evidence records the override",
          students[0].evidence)


def test_staggered_hands_still_ambiguous() -> None:
    section("staggered hands: the near-miss that Rule A originally failed")
    # Two students raising 0.4 s apart - far more realistic than the same frame.
    # Before contenders were counted, the earlier hand confirmed first and was
    # credited outright, so a friend answering slightly ahead of the real student
    # would have been accepted silently. This is the regression guard.
    # The arbitration horizon is emergent, not configured. Track 1 latches once its
    # EMA crosses the enter threshold (~0.15 s at tau=0.18) and confirms hold_sec
    # (0.60 s) later, so it is credited ~0.75 s after raising. A second hand is
    # counted as competing if it has latched by then, i.e. raised within about
    # 0.60 s of the first. Right at that edge the outcome is quantized by the frame
    # interval, so the cases below stay comfortably inside it.
    #
    # Past the horizon the called student has already answered promptly and
    # correctly, and a later raise should not retroactively void it - Rule B stops
    # that person being credited for anybody else anyway.
    for offset in (0.05, 0.20, 0.35, 0.45):
        sess, students, _v, tracker, _g, _sp = make_session(1)
        sess.start(0.0)
        _x, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
        second_at = t + offset
        tt = t
        while tt < t + 3.0 and sess.phase is Phase.RESPONSE:
            obs = {}
            for tid, start in ((1, t), (2, second_at)):
                if tt >= start:
                    ps = PersonScore()
                    ps.score = 1.0
                    ps.torso_px = 100.0
                    obs[tid] = (ps, BBOX)
            tracker.update(tt, obs)
            sess.tick(tt)
            tt += 0.05
        ok = sess.phase is Phase.AMBIGUOUS and students[0].status == PENDING
        check(ok, f"second hand {offset:.2f}s later still triggers AMBIGUOUS",
              f"phase={sess.phase.value} status={students[0].status}")

    # The documented boundary: a hand that goes up only after the first is already
    # confirmed does NOT void a prompt, correct answer.
    sess, students, _v, tracker, _g, _sp = make_session(1)
    sess.start(0.0)
    _x, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    late = t + 1.2
    tt = t
    while tt < t + 3.0:
        obs = {}
        for tid, start in ((1, t), (2, late)):
            if tt >= start:
                ps = PersonScore()
                ps.score = 1.0
                ps.torso_px = 100.0
                obs[tid] = (ps, BBOX)
        tracker.update(tt, obs)
        sess.tick(tt)
        tt += 0.05
    check(students[0].status == PRESENT,
          "a hand raised after confirmation does not void a prompt answer",
          students[0].status)


def test_rule_b_blocks_second_answer() -> None:
    section("Rule B: one person cannot answer twice")
    sess, students, _v, tracker, _g, _sp = make_session(2)
    sess.start(0.0)
    _v2, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    _v2, t = drive(sess, tracker, t, t + 2.0, score=1.0)
    check(students[0].status == PRESENT, "first student Present")

    # Advance to student 2; the same track raises again.
    _v2, t = drive(sess, tracker, t, t + 2.0, score=None)
    check(sess.current == 1, "on the second student", f"current={sess.current}")
    _v2, t = drive(sess, tracker, t, t + 1.0, score=0.0)
    view, t = drive(sess, tracker, t, t + 2.5, score=1.0)
    check(
        sess.phase is Phase.AMBIGUOUS or students[1].status == PENDING,
        "same person answering again is blocked, not credited",
        f"phase={sess.phase} status={students[1].status}",
    )
    check("already marked" in (view.blocked_reason or ""), "reason names the conflict",
          view.blocked_reason)


def test_window_extension() -> None:
    section("adaptive window extension")
    sess, students, _v, tracker, _g, _sp = make_session(1)
    sess.start(0.0)
    _v2, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    window = float(CFG.require("session.response_window_sec"))
    above = float(CFG.require("session.extend_if_score_above"))

    # A student halfway up as the deadline arrives, who then commits. The score
    # function switches at the deadline so the raise happens INSIDE the extension -
    # driving past the extension first would simply time out again.
    partial = (above + float(CFG.require("temporal.enter_threshold"))) / 2.0
    deadline = t + window
    view, t = drive(
        sess, tracker, t, deadline + 1.2,
        score=lambda tt: partial if tt < deadline else 1.0,
    )
    check(sess._extended, "window was extended on a partial raise")
    check(students[0].status == PRESENT, "credited within the extension",
          students[0].status)

    # Control: a partial raise that never improves must still end as Absent.
    sess2, students2, _v2, tracker2, _g2, _sp2 = make_session(1)
    sess2.start(0.0)
    _x, t2 = drive(sess2, tracker2, 0.0, 1.0, score=0.0)
    _x, t2 = drive(sess2, tracker2, t2, t2 + window + 4.0, score=partial)
    check(students2[0].status == ABSENT,
          "a partial raise that never completes is Absent, not Present",
          students2[0].status)


def test_speech_unavailable_and_hung() -> None:
    section("speech must never block the roll call")
    # No engine at all.
    sess, students, _v, tracker, _g, _sp = make_session(1, speech=FakeSpeech(available=False))
    sess.start(0.0)
    _v2, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    check(sess.phase is Phase.RESPONSE, "window opens with no TTS engine", sess.phase)

    # Engine accepts work but never reports completion.
    hung = FakeSpeech(available=True, auto_complete=False)
    sess2, students2, _v3, tracker2, _g2, _sp2 = make_session(1, speech=hung)
    sess2.start(0.0)
    wait = float(CFG.get("session.max_speech_wait_sec", 4.0))
    _v4, t2 = drive(sess2, tracker2, 0.0, wait + 1.0, score=0.0)
    check(sess2.phase is Phase.RESPONSE, "window opens after the speech timeout",
          sess2.phase)


def test_undo() -> None:
    section("undo")
    sess, students, _v, tracker, gate, _sp = make_session(2)
    sess.start(0.0)
    _v2, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    _v2, t = drive(sess, tracker, t, t + 2.0, score=1.0)
    check(students[0].status == PRESENT, "marked Present before undo")
    check(1 in gate.credited, "track credit was taken")
    sess.undo()
    check(students[0].status == PENDING, "undo restores Pending", students[0].status)
    check(1 not in gate.credited, "undo releases the track credit")


def test_finishes() -> None:
    section("completion")
    sess, students, _v, tracker, _g, _sp = make_session(2)
    sess.start(0.0)
    view, t = drive(sess, tracker, 0.0, 30.0, score=0.0, step=0.1)
    check(sess.phase is Phase.FINISHED, "reaches FINISHED", sess.phase)
    check(all(s.status != PENDING for s in students), "every student is marked")
    check(view.stats["absent"] == 2, "stats are right", str(view.stats))


def test_pause_resume() -> None:
    section("pause and resume")
    sess, students, _v, tracker, _g, sp = make_session(2)
    sess.start(0.0)
    _v2, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    before = sess.view(t).remaining
    sess.pause(t)
    check(sess.phase is Phase.PAUSED, "pauses", sess.phase)
    check(sp.cleared >= 1, "pausing silences the voice")
    _v2, t = drive(sess, tracker, t, t + 20.0, score=0.0, step=0.25)
    check(students[0].status == PENDING, "paused sessions never auto-mark",
          students[0].status)
    sess.start(t)
    check(sess.phase is Phase.RESPONSE, "resumes into RESPONSE", sess.phase)
    # The 20 s pause is far longer than the window. Before the fix, deadlines were
    # absolute, so the student was marked Absent on the very next tick.
    after = sess.view(t).remaining
    check(abs(after - before) < 0.3, "the window resumes with the time it had left",
          f"{before:.2f}s before pause, {after:.2f}s after")
    sess.tick(t + 0.05)
    check(students[0].status == PENDING, "not marked Absent the moment it resumes",
          students[0].status)
    _v2, t = drive(sess, tracker, t, t + after + 0.5, score=0.0)
    check(students[0].status == ABSENT, "times out normally once the rest has elapsed",
          students[0].status)


def test_pause_during_call_rereads() -> None:
    section("pause while the number is being read")
    slow = FakeSpeech(available=True, auto_complete=False)
    sess, students, _v, tracker, _g, sp = make_session(1, speech=slow)
    sess.start(0.0)
    _v2, t = drive(sess, tracker, 0.0, 1.0, score=0.0)
    sess.pause(t)
    _v2, t = drive(sess, tracker, t, t + 5.0, score=0.0)
    sess.start(t)
    check(sess.phase is Phase.CALLING, "resumes by calling the number again", sess.phase)
    check(len(sp.spoken) == 2 and sp.spoken[0][1] == sp.spoken[1][1],
          "the interrupted number is read again in full", str([x[1] for x in sp.spoken]))
    check(sp.spoken[1][0] != sp.spoken[0][0], "with a new token, so a stale event cannot open the window")


def test_call_modes() -> None:
    section("spoken call modes")
    check(full_call("SP25-BCS-007") == "S P two five, B C S, zero zero seven",
          "full: every part, digits read individually", full_call("SP25-BCS-007"))
    check(full_call(" FA24_BSE_120 ") == "F A two four, B S E, one two zero",
          "full: trims and splits on any separator", full_call(" FA24_BSE_120 "))
    check(full_call("12345") == "one two three four five", "full: bare numeric roll")
    students = parse_pasted("SP25-BCS-001 SP25-BCS-007")
    assign_calls(students, "short_unique")
    check([x.spoken_call for x in students] == ["one", "seven"], "short_unique still available",
          str([x.spoken_call for x in students]))
    try:
        assign_calls(students, "loud")
        check(False, "an unknown mode is rejected")
    except ValueError:
        check(True, "an unknown mode is rejected")

    short_cfg = CFG.override({"speech": {"mode": "short_unique"}})
    sess, _st, _v, _t, _g, sp = make_session(2, cfg=short_cfg)
    sess.start(0.0)
    check(sp.spoken and sp.spoken[0][1] == "one", "session honours speech.mode",
          str(sp.spoken[:1]))


def test_raise_during_long_call_credited() -> None:
    section("raise while the full number is still being read")
    slow = FakeSpeech(available=True, auto_complete=False)
    sess, students, _v, tracker, _g, sp = make_session(2, speech=slow)
    sess.start(0.0)
    # The student reads their number off the screen and raises at once, while
    # the voice is still reading it (4.4 s for a full registration number).
    _v2, t = drive(sess, tracker, 0.0, 4.4, score=1.0)
    check(sess.phase is Phase.CALLING, "still calling while the voice speaks", sess.phase)
    slow.finish()
    _v2, t = drive(sess, tracker, t, t + 2.0, score=1.0)
    check(students[0].status == PRESENT,
          "credited: a raise during the call is fresh, not a parked hand",
          students[0].status)


def main() -> int:
    for fn in (
        test_happy_path,
        test_timeout_absent,
        test_stall_blocks_absence,
        test_stalled_raise_still_credited,
        test_ambiguous_blocks,
        test_staggered_hands_still_ambiguous,
        test_rule_b_blocks_second_answer,
        test_window_extension,
        test_speech_unavailable_and_hung,
        test_undo,
        test_finishes,
        test_pause_resume,
        test_pause_during_call_rereads,
        test_call_modes,
        test_raise_during_long_call_credited,
    ):
        fn()

    print()
    print("=" * 74)
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
    print("=" * 74)
    print(f"  {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
