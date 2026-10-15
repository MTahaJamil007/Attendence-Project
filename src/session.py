"""Roll-call state machine.

Drives one student at a time: speak the number, open a response window, watch for a
confirmed raise, record a result, advance.

Three behaviours here exist specifically because the predecessor lacked them:

* **An absence is never recorded while the pipeline is stalled.** Every automatic
  Absent goes through ``video.may_record_absence``. While stalled the phase
  becomes STALLED and the deadline is pushed forward by the stall duration, so the
  student still gets their full window once frames resume. The predecessor's state
  machine ran on its own timer and wrote "Timeout: no valid hand" through 52-second
  inference freezes, ruining three of its eight recorded runs.

* **Ambiguity is escalated, never guessed.** Two raised hands, or a hand belonging
  to someone already marked - by tracker ID or by seat - stops the machine and
  asks the teacher, with the seat in question drawn on the operator view. Rule A
  only means something if the system actually refuses to decide.

* **Speech is advisory.** The window opens when the utterance completes *or* when
  a timeout expires, whichever comes first. A hung SAPI engine cannot freeze the
  roll call.

The machine is driven by the UI thread calling ``tick(now)``; it owns no threads of
its own, which keeps all tkinter interaction on the main thread.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from gating import Verdict
from roster import ABSENT, PENDING, PRESENT, Student, assign_calls, summarize


class Phase(str, Enum):
    IDLE = "IDLE"
    CALLING = "CALLING"
    RESPONSE = "RESPONSE"
    AMBIGUOUS = "AMBIGUOUS"
    STALLED = "STALLED"
    RESULT = "RESULT"
    PAUSED = "PAUSED"
    FINISHED = "FINISHED"


@dataclass
class SessionView:
    """Everything the two windows need to render. Pure data, no behaviour."""

    phase: Phase = Phase.IDLE
    roll: str = ""
    name: str = ""
    spoken: str = ""
    prompt: str = "Load a roll list"
    message: str = ""
    result: str = ""
    remaining: float = 0.0
    window: float = 0.0
    hold_progress: float = 0.0
    best_score: float = 0.0
    index: int = 0
    total: int = 0
    stats: dict = field(default_factory=dict)
    blocked_reason: str = ""
    needs_teacher: bool = False
    # Why the machine stopped for the teacher (a gating Verdict value), if it did.
    flag: str = ""

    @property
    def window_fraction(self) -> float:
        if self.window <= 0:
            return 0.0
        return max(0.0, min(1.0, self.remaining / self.window))


class Session:
    def __init__(self, cfg, video, tracker, gate, speech, evidence=None):
        self.cfg = cfg
        self.video = video
        self.tracker = tracker
        self.gate = gate
        self.speech = speech
        self.evidence = evidence

        s = cfg.section("session")
        self.reaction_delay = float(s["reaction_delay_sec"])
        self.window_sec = float(s["response_window_sec"])
        self.extension_sec = float(s["window_extension_sec"])
        self.extend_above = float(s["extend_if_score_above"])
        self.result_hold = float(s["result_hold_sec"])
        self.max_speech_wait = float(s.get("max_speech_wait_sec", 8.0))
        self.hold_sec = float(cfg.get("temporal.hold_sec", 0.6))
        self.call_mode = str(cfg.get("speech.mode", "full"))

        self.students: list[Student] = []
        self.phase: Phase = Phase.IDLE
        self.current: int = -1

        self._token = 0
        self._speech_done_at: float | None = None
        self._call_started: float = 0.0
        self._response_ends: float = 0.0
        self._result_until: float = 0.0
        self._extended: bool = False
        self._last_tick: float = 0.0
        self._resume_phase: Phase | None = None
        self._paused_at: float = 0.0

        self.message: str = ""
        self.result_text: str = ""
        self.blocked_reason: str = ""
        # The gating decision that stopped the current call for the teacher. The
        # console draws its seats and hands; telemetry records it with the mark.
        self.flag = None
        self.last_marked: int = -1
        # One record per decision, exported as session_*.json. Without this the
        # CSV cannot answer questions like "was the same tracked person credited
        # twice?", which is exactly the kind of thing that needs auditing.
        self.telemetry: list[dict] = []

    # -- roster ------------------------------------------------------------
    def set_students(self, students: list[Student]) -> None:
        assign_calls(students, self.call_mode)
        self.students = students
        self.current = -1
        self.phase = Phase.IDLE
        self.last_marked = -1
        self.flag = None
        self.telemetry.clear()
        self.message = f"{len(students)} students loaded"
        self.tracker.reset()
        self.gate.reset_session()

    def reset_marks(self) -> None:
        for s in self.students:
            s.status = PENDING
            s.marked_at = ""
            s.evidence = ""
            s.track_id = None
            s.score = 0.0
            s.evidence_image = ""
        self.current = -1
        self.last_marked = -1
        self.phase = Phase.IDLE
        self.flag = None
        self.telemetry.clear()
        self.tracker.reset()
        self.gate.reset_session()
        self.message = "all marks cleared"

    # -- control -----------------------------------------------------------
    def start(self, now: float) -> bool:
        if not self.students:
            self.message = "load or paste roll numbers first"
            return False
        if self.phase is Phase.PAUSED:
            self._resume(now)
            return True
        nxt = self._next_pending()
        if nxt is None:
            self.phase = Phase.FINISHED
            self.message = "every student is already marked"
            return False
        self._last_tick = now
        self._begin_call(nxt, now)
        return True

    def pause(self, now: float | None = None) -> None:
        if self.phase in (Phase.IDLE, Phase.FINISHED, Phase.PAUSED):
            return
        self._resume_phase = self.phase
        self._paused_at = self._last_tick if now is None else now
        self.phase = Phase.PAUSED
        self.speech.clear()
        self.message = "paused"

    def _resume(self, now: float) -> None:
        """Continue exactly where the pause left off.

        The clock does not run while paused. Before this, deadlines were absolute,
        so pausing mid-window for longer than the time left marked the student
        Absent the instant the teacher resumed.
        """
        resume = self._resume_phase or Phase.RESPONSE
        self._resume_phase = None
        paused_for = max(0.0, now - self._paused_at)
        self._last_tick = now
        if resume is Phase.CALLING:
            # Pausing cut the utterance off, so read the number again in full.
            self._begin_call(self.current, now)
            self.message = "resumed"
            return
        self._response_ends += paused_for
        self._result_until += paused_for
        self.phase = resume
        self.message = "resumed"

    def stop(self) -> None:
        self.phase = Phase.IDLE
        self.current = -1
        self.speech.clear()

    # -- the tick ----------------------------------------------------------
    def tick(self, now: float) -> SessionView:
        dt = max(0.0, now - self._last_tick) if self._last_tick else 0.0
        self._last_tick = now

        if self.phase is Phase.CALLING:
            self._tick_calling(now)
        elif self.phase in (Phase.RESPONSE, Phase.STALLED):
            self._tick_response(now, dt)
        elif self.phase is Phase.RESULT:
            if now >= self._result_until:
                self._advance(now)
        # AMBIGUOUS, PAUSED, IDLE and FINISHED are all inert: they wait for the
        # teacher or for start().
        return self.view(now)

    # -- phases ------------------------------------------------------------
    def _begin_call(self, index: int, now: float) -> None:
        self.current = index
        st = self.students[index]
        self.phase = Phase.CALLING
        self.result_text = ""
        self.blocked_reason = ""
        self.flag = None
        self._extended = False
        self._speech_done_at = None
        self._call_started = now
        self._token += 1
        self.speech.speak(st.spoken_call or st.roll, self._token)
        self.message = f"calling {st.roll}"

    def _tick_calling(self, now: float) -> None:
        # Consume speech completions. Only the current token matters; a stale
        # event from a previous call must not open this window early.
        for ev in self.speech.poll():
            if ev.token == self._token:
                self._speech_done_at = now

        waited = now - self._call_started
        speech_settled = (
            self._speech_done_at is not None
            or not self.speech.available
            or waited >= self.max_speech_wait
        )
        if not speech_settled:
            return
        base = self._speech_done_at if self._speech_done_at is not None else self._call_started
        if now - base >= self.reaction_delay:
            self._open_response(now)

    def _open_response(self, now: float) -> None:
        # Freshness counts from when the number was first shown and spoken, not
        # from now: reading a full registration number takes 4-5 s, and a student
        # who raised as soon as they recognised it must not be treated as a hand
        # left over from the previous call.
        self.tracker.arm(now, since=self._call_started)
        self.phase = Phase.RESPONSE
        self._response_ends = now + self.window_sec
        st = self.students[self.current]
        self.message = f"waiting for {st.roll}"

    def _tick_response(self, now: float, dt: float) -> None:
        # 1. Liveness gate, before anything else can resolve the call.
        allowed, reason = self.video.may_record_absence(now)
        if not allowed:
            if self.phase is not Phase.STALLED:
                self.phase = Phase.STALLED
            self.blocked_reason = reason
            # Hold the deadline open: the student is owed a full window from the
            # moment the system can actually see them again.
            self._response_ends += dt
            return
        if self.phase is Phase.STALLED:
            self.phase = Phase.RESPONSE
            self.blocked_reason = ""

        st = self.students[self.current]

        # 2. Has anyone been confirmed?
        confirmed = self.tracker.confirmed()
        # Contenders (every hand currently up) are what the single-hand rule counts,
        # so a second raiser who has not yet held long enough still registers as a
        # competing hand instead of being silently ignored.
        decision = self.gate.evaluate(confirmed, st.roll, self.tracker.contenders())

        if decision.is_present:
            self._mark(PRESENT, "Hand raised", now, decision)
            return
        if decision.needs_teacher:
            self.phase = Phase.AMBIGUOUS
            self.flag = decision
            self.message = decision.message
            self.blocked_reason = decision.message
            self.result_text = {
                Verdict.AMBIGUOUS: "TWO HANDS",
                Verdict.REUSED_SEAT: "SAME SEAT",
                Verdict.REUSED_TRACK: "ALREADY MARKED",
            }.get(decision.verdict, "CHECK")
            return

        # 3. Deadline.
        if now >= self._response_ends:
            partial = self.tracker.best_partial()
            if not self._extended and partial >= self.extend_above:
                # Someone is visibly on the way up. Give them the extra time
                # rather than recording an absence we can already see is wrong.
                self._extended = True
                self._response_ends = now + self.extension_sec
                self.message = f"{st.roll}: partial raise seen, extending"
                return
            self._mark(ABSENT, "Timeout: no hand raised", now, None)

    # -- marking -----------------------------------------------------------
    def _mark(self, status: str, evidence: str, now: float, decision) -> None:
        st = self.students[self.current]
        st.status = status
        st.marked_at = datetime.now().isoformat(timespec="seconds")
        st.evidence = evidence
        track = getattr(decision, "track", None) if decision else None
        if track is not None:
            st.track_id = track.track_id
            st.score = round(track.peak_ema, 3)
            self.gate.commit(st.roll, track, now)
            if self.evidence is not None:
                try:
                    st.evidence_image = self.evidence.capture(st, track, self.video)
                except Exception:
                    st.evidence_image = ""
        self.last_marked = self.current
        self.result_text = status.upper()
        self.phase = Phase.RESULT
        self._result_until = now + self.result_hold
        self.message = f"{st.roll}: {status}" + (f" ({evidence})" if evidence else "")

        vstat = {}
        try:
            vstat = self.video.status()
        except Exception:
            pass
        self.telemetry.append({
            "roll": st.roll,
            "status": status,
            "evidence": evidence,
            "marked_at": st.marked_at,
            "track_id": st.track_id,
            "score": st.score,
            "call_started": round(self._call_started, 3),
            "decided_at": round(now, 3),
            "seconds_to_decide": round(now - self._call_started, 3),
            "window_extended": self._extended,
            "contenders": [t.track_id for t in self.tracker.contenders()],
            "confirmed": [t.track_id for t in self.tracker.confirmed()],
            "fps": round(vstat.get("fps") or 0.0, 1),
            "capture_fps": round(vstat.get("capture_fps") or 0.0, 1),
            "evidence_image": st.evidence_image,
            "seat": self._seat_summary(st.roll),
            "flag": self._flag_summary(),
        })
        self.flag = None

    def _seat_summary(self, roll: str) -> dict | None:
        rec = getattr(self.gate, "seats", {}).get(roll)
        if rec is None:
            return None
        return {"x": round(rec.x, 1), "y": round(rec.y, 1), "scale": round(rec.scale, 1)}

    def _flag_summary(self) -> dict | None:
        """What stopped this call for the teacher, kept with the teacher's answer.

        Lets an audit separate "the system flagged it and the teacher overrode"
        from "the system was sure", which the CSV's evidence column cannot.
        """
        d = self.flag
        if d is None:
            return None
        out = {"verdict": d.verdict.value, "message": d.message}
        seats = getattr(d, "seat_matches", None) or {}
        if seats:
            out["seat_matches"] = [
                {
                    "track_id": m.track_id,
                    "seat_of": m.record.roll,
                    "distance_shoulders": round(m.distance, 3),
                    "appearance": None if m.appearance is None else round(m.appearance, 3),
                }
                for m in seats.values()
            ]
        return out

    def mark_manual(self, present: bool, now: float) -> None:
        """Teacher override, also used to resolve an AMBIGUOUS call."""
        if self.current < 0 or self.current >= len(self.students):
            return
        if self.phase in (Phase.IDLE, Phase.FINISHED):
            return
        decision = None
        if present:
            # Credit the strongest confirmed track so Rule B still consumes it -
            # otherwise a manually confirmed student leaves their track free to
            # answer for somebody else. When the call was stopped over one
            # specific hand (a reused seat or track), that is the hand the
            # teacher just looked at and confirmed.
            confirmed = self.tracker.confirmed()
            flagged = getattr(self.flag, "track", None)
            if flagged is not None and flagged.track_id in self.tracker.states:
                confirmed = [self.tracker.states[flagged.track_id]]
            if confirmed:
                best = max(confirmed, key=lambda s: s.peak_ema)

                class _D:
                    track = best

                decision = _D()
        self._mark(PRESENT if present else ABSENT, "Manual (teacher)", now, decision)

    def undo(self) -> None:
        if self.last_marked < 0:
            self.message = "nothing to undo"
            return
        st = self.students[self.last_marked]
        self.gate.release(st.roll)
        st.status = PENDING
        st.marked_at = ""
        st.evidence = ""
        st.track_id = None
        st.score = 0.0
        st.evidence_image = ""
        for i in range(len(self.telemetry) - 1, -1, -1):
            if self.telemetry[i].get("roll") == st.roll:
                self.telemetry.pop(i)
                break
        self.current = self.last_marked
        self.last_marked = -1
        self.phase = Phase.IDLE
        self.result_text = ""
        self.message = f"undid {st.roll} - press Start to call again"

    # -- navigation --------------------------------------------------------
    def _next_pending(self, after: int | None = None) -> int | None:
        start = 0 if after is None else after + 1
        for i in range(start, len(self.students)):
            if self.students[i].is_pending:
                return i
        # Wrap, in case an Undo left an earlier student pending.
        for i in range(0, min(start, len(self.students))):
            if self.students[i].is_pending:
                return i
        return None

    def _advance(self, now: float) -> None:
        nxt = self._next_pending(self.current)
        if nxt is None:
            self.phase = Phase.FINISHED
            self.result_text = "COMPLETE"
            stats = summarize(self.students)
            self.message = (
                f"complete - {stats['present']} present, {stats['absent']} absent"
            )
            return
        self._begin_call(nxt, now)

    # -- view --------------------------------------------------------------
    def view(self, now: float) -> SessionView:
        v = SessionView(phase=self.phase, total=len(self.students))
        v.stats = summarize(self.students)
        v.message = self.message
        v.result = self.result_text
        v.blocked_reason = self.blocked_reason
        v.needs_teacher = self.phase is Phase.AMBIGUOUS
        if self.flag is not None:
            v.flag = self.flag.verdict.value

        if 0 <= self.current < len(self.students):
            st = self.students[self.current]
            v.roll, v.name, v.spoken = st.roll, st.name, st.spoken_call
            v.index = self.current + 1

        if self.phase in (Phase.RESPONSE, Phase.STALLED):
            v.window = self.window_sec
            v.remaining = max(0.0, self._response_ends - now)

        # Hold progress and best score drive the student-facing feedback: seeing
        # "HOLD" fill up is what tells a student they have been noticed, which
        # matters more for real-world hit rate than any threshold tweak.
        best = 0.0
        prog = 0.0
        for stt in self.tracker.states.values():
            best = max(best, stt.ema)
            if stt.latched and self.hold_sec > 0:
                prog = max(prog, min(1.0, stt.held_for(now) / self.hold_sec))
        v.best_score = best
        v.hold_progress = prog
        v.prompt = self._prompt(v)
        return v

    def _prompt(self, v: SessionView) -> str:
        if self.phase is Phase.IDLE:
            return "Ready" if self.students else "Load a roll list"
        if self.phase is Phase.CALLING:
            return "GET READY"
        if self.phase is Phase.STALLED:
            return "CAMERA PROBLEM - PLEASE WAIT"
        if self.phase is Phase.RESPONSE:
            if v.hold_progress > 0.0:
                return "HAND SEEN - KEEP IT UP"
            if v.best_score > 0.25:
                return "RAISE HIGHER"
            return "RAISE YOUR HAND"
        if self.phase is Phase.AMBIGUOUS:
            if v.flag in (Verdict.REUSED_SEAT.value, Verdict.REUSED_TRACK.value):
                return "ALREADY MARKED - TEACHER WILL CHECK"
            return "TOO MANY HANDS - WAIT"
        if self.phase is Phase.RESULT:
            return "PRESENT" if v.result == PRESENT.upper() else "ABSENT"
        if self.phase is Phase.PAUSED:
            return "PAUSED"
        if self.phase is Phase.FINISHED:
            return "ATTENDANCE COMPLETE"
        return ""
