"""Tests for seat memory (gating Rule D). No torch, camera or GPU needed.

The failure this guards against, as reported from a real class: a student
answered for their own roll number (9), and several calls later raised again for
someone else's (31) from the same seat - and was credited for both. Rule B
(one credit per tracker ID) should have stopped it, but tracker IDs are not
permanent: a moment's occlusion gives the same person a new ID, and Rule B then
sees a stranger.

Everything here runs through the real RaiseScorer, TemporalRaiseTracker,
AttendanceGate and Session with synthetic but anatomically proportioned
skeletons. The seat layout uses measured proportions: a seated student's shoulder
width ~0.67 torso lengths, neighbours about one shoulder width apart.

Run:  .venv\\Scripts\\python.exe tools\\test_seats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from appearance import similarity, torso_signature  # noqa: E402
from config import Config  # noqa: E402
from gating import AttendanceGate, SeatRecord, Verdict  # noqa: E402
from geometry import seat_anchor  # noqa: E402
from raise_detector import RaiseScorer, TemporalRaiseTracker, TrackState  # noqa: E402
from roster import ABSENT, PENDING, PRESENT, Student, assign_short_unique_calls  # noqa: E402
from session import Phase, Session  # noqa: E402
from speech import SpeechEvent  # noqa: E402

CFG = Config.load(ROOT / "config.json")
SCORER = RaiseScorer(CFG)

SW = 80.0       # shoulder width, px
TORSO = 120.0   # shoulder-to-hip, px
ROW_Y = 300.0   # shoulder line of the row


# --- synthetic people -----------------------------------------------------
def person_kp(cx: float, sy: float = ROW_Y, sw: float = SW, torso: float = TORSO,
              raised: bool = False, conf: float = 0.9):
    """COCO-17 keypoints for a seated student facing the camera.

    The person's right side is on the image's left. A raise lifts the right arm
    straight up; at rest it hangs by the side.
    """
    kp = [(0.0, 0.0, 0.0)] * 17
    rsx, lsx = cx - sw / 2, cx + sw / 2
    kp[0] = (cx, sy - 0.40 * torso, conf)                    # nose
    kp[1] = (cx + 0.10 * sw, sy - 0.45 * torso, conf)        # left eye
    kp[2] = (cx - 0.10 * sw, sy - 0.45 * torso, conf)        # right eye
    kp[3] = (cx + 0.22 * sw, sy - 0.42 * torso, conf)        # left ear
    kp[4] = (cx - 0.22 * sw, sy - 0.42 * torso, conf)        # right ear
    kp[5] = (lsx, sy, conf)                                  # left shoulder
    kp[6] = (rsx, sy, conf)                                  # right shoulder
    kp[7] = (lsx + 4, sy + 0.55 * torso, conf)               # left elbow (rest)
    kp[9] = (lsx + 8, sy + 1.05 * torso, conf)               # left wrist (rest)
    if raised:
        kp[8] = (rsx + 2, sy - 0.50 * torso, conf)           # right elbow up
        kp[10] = (rsx + 4, sy - 1.05 * torso, conf)          # right wrist up
    else:
        kp[8] = (rsx - 4, sy + 0.55 * torso, conf)
        kp[10] = (rsx - 8, sy + 1.05 * torso, conf)
    kp[11] = (cx + 0.40 * sw, sy + torso, conf)              # hips
    kp[12] = (cx - 0.40 * sw, sy + torso, conf)
    for i in (13, 14, 15, 16):
        kp[i] = (cx, sy + 1.6 * torso, 0.05)                 # legs hidden by desks
    return kp


def bbox_of(kp):
    pts = [(x, y) for x, y, c in kp if c >= 0.3]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10)


def observe(cx, raised=True, sy=ROW_Y, sw=SW, torso=TORSO, signature=None):
    kp = person_kp(cx, sy, sw, torso, raised)
    box = bbox_of(kp)
    ps = SCORER.score(kp, box[3] - box[1])
    ps.signature = signature
    return ps, box


def shirt(bgr) -> np.ndarray:
    """Signature of a person wearing a plain shirt of colour *bgr*."""
    frame = np.full((600, 800, 3), 40, np.uint8)
    frame[ROW_Y_INT:ROW_Y_INT + 140, 300:500] = bgr
    rng = np.random.default_rng(int(sum(bgr)))
    noise = rng.integers(-6, 7, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return torso_signature(frame, person_kp(400.0), 0.3)


ROW_Y_INT = int(ROW_Y)
RED, BLUE, WHITE, BLACK = (40, 40, 200), (200, 60, 30), (235, 235, 235), (20, 20, 20)


def track(tid, cx, sw=SW, sig=None, ema=0.95, sy=ROW_Y) -> TrackState:
    """A confirmed track sitting at (cx, sy) with a smoothed anchor."""
    ps, box = observe(cx, sy=sy, sw=sw, torso=sw * TORSO / SW, signature=sig)
    st = TrackState(track_id=tid, ema=ema, latched=True, latched_at=0.0,
                    confirmed=True, confirmed_at=0.6, peak_ema=ema)
    st.bbox = st.best_bbox = box
    st.last_score = st.best_score = ps
    st.anchor = ps.anchor
    st.signature = sig
    return st


# --- harness --------------------------------------------------------------
_results: list[tuple[bool | None, str, str]] = []


def check(cond: bool, name: str, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


def section(title: str) -> None:
    _results.append((None, title, ""))


# =========================================================================
def test_anchor_ignores_the_raised_arm() -> None:
    section("seat anchor: shoulders, not the box - a raised arm must not move it")
    rest = person_kp(400.0, raised=False)
    up = person_kp(400.0, raised=True)
    a_rest = seat_anchor(rest, TORSO, 0.3)
    a_up = seat_anchor(up, TORSO, 0.3)
    shift = ((a_rest[0] - a_up[0]) ** 2 + (a_rest[1] - a_up[1]) ** 2) ** 0.5 / a_up[2]
    check(shift < 0.01, "anchor moves < 0.01 shoulder widths when the arm goes up", f"{shift:.4f}")
    b_rest, b_up = bbox_of(rest), bbox_of(up)
    c_rest = ((b_rest[0] + b_rest[2]) / 2, (b_rest[1] + b_rest[3]) / 2)
    c_up = ((b_up[0] + b_up[2]) / 2, (b_up[1] + b_up[3]) / 2)
    bshift = ((c_rest[0] - c_up[0]) ** 2 + (c_rest[1] - c_up[1]) ** 2) ** 0.5 / a_up[2]
    check(bshift > 0.4, "...whereas the box centre the old seat map used moves > 0.4",
          f"{bshift:.2f} shoulder widths - nearly half the way to the next seat")
    check(abs(a_up[2] - SW) < 1e-6, "scale is the shoulder width when facing the camera", f"{a_up[2]}")

    turned = person_kp(400.0, sw=SW * 0.4)  # shoulders foreshortened by a turn
    a_turned = seat_anchor(turned, TORSO, 0.3)
    check(a_turned[2] >= TORSO / 1.5 - 1e-6, "turning sideways cannot shrink the unit below torso/1.5",
          f"{a_turned[2]:.1f}")

    one = list(person_kp(400.0))
    one[5] = (one[5][0], one[5][1], 0.05)  # left shoulder hidden
    a_one = seat_anchor(one, TORSO, 0.3)
    check(a_one is not None and abs(a_one[0] - 400.0) < 1.0,
          "one shoulder hidden: centred under the face", f"{a_one}")
    none = [(x, y, 0.0) for x, y, _ in person_kp(400.0)]
    check(seat_anchor(none, TORSO, 0.3) is None, "no shoulders: no anchor (never guessed)")

    ps, _box = observe(400.0, raised=True)
    check(ps.score > 0.55, "the synthetic raise really scores as a raise", f"{ps.score:.3f}")
    check(ps.anchor is not None, "RaiseScorer fills PersonScore.anchor")


def test_appearance_signature() -> None:
    section("clothing signature (no face involved)")
    red_a, red_b, blue = shirt(RED), shirt(RED), shirt(BLUE)
    white, black = shirt(WHITE), shirt(BLACK)
    s_same = similarity(red_a, red_b)
    check(s_same is not None and s_same > 0.9, "same shirt -> similarity > 0.9", f"{s_same:.3f}")
    check(similarity(red_a, blue) < 0.5, "red vs blue -> < 0.5", f"{similarity(red_a, blue):.3f}")
    check(similarity(white, black) < 0.5, "white vs black -> < 0.5 (value axis)",
          f"{similarity(white, black):.3f}")
    check(similarity(None, red_a) is None, "missing signature -> None, not 0")
    hidden = list(person_kp(400.0))
    hidden[6] = (hidden[6][0], hidden[6][1], 0.1)
    frame = np.zeros((600, 800, 3), np.uint8)
    check(torso_signature(frame, hidden, 0.3) is None, "needs both shoulders")


def test_same_seat_new_track_is_flagged() -> None:
    section("Rule D: same seat, new tracker ID -> flagged, not credited")
    gate = AttendanceGate(CFG)
    first = track(1, 400.0, sig=shirt(RED))
    d = gate.evaluate([first], "SP25-BCS-009")
    check(d.verdict is Verdict.PRESENT, "roll 9 credited normally", d.verdict.value)
    gate.commit("SP25-BCS-009", d.track, now=10.0)
    check("SP25-BCS-009" in gate.seats, "seat recorded for roll 9")

    # The same person, after the tracker lost and re-acquired them (new ID),
    # shifted a little in their seat.
    again = track(7, 400.0 + 0.2 * SW, sig=shirt(RED))
    d = gate.evaluate([again], "SP25-BCS-031")
    check(d.verdict is Verdict.REUSED_SEAT, "roll 31 from the same seat -> REUSED_SEAT", d.verdict.value)
    check(d.needs_teacher, "it needs the teacher")
    check(not d.is_present, "it is NOT credited")
    check(d.seat is not None and d.seat.record.roll == "SP25-BCS-009",
          "the decision names the seat's owner", d.message)
    check("SP25-BCS-009" in d.message and "#7" in d.message, "message names both", d.message)
    check(abs(d.seat.distance - 0.2) < 0.02, "distance reported in shoulder widths",
          f"{d.seat.distance:.3f}")

    # Even with completely different clothing, sitting in the same place is enough.
    jacket_off = track(8, 400.0 + 0.1 * SW, sig=shirt(BLUE))
    d = gate.evaluate([jacket_off], "SP25-BCS-031")
    check(d.verdict is Verdict.REUSED_SEAT, "inside the tight radius, clothing is not needed",
          d.verdict.value)

    # Without colour at all (signature not measured).
    d = gate.evaluate([track(9, 400.0 + 0.3 * SW, sig=None)], "SP25-BCS-031")
    check(d.verdict is Verdict.REUSED_SEAT, "no signature: position alone still flags", d.verdict.value)


def test_neighbours_are_not_flagged() -> None:
    section("Rule D: the student in the next seat is NOT the same seat")
    gate = AttendanceGate(CFG)
    gate.commit("SP25-BCS-009", track(1, 400.0, sig=shirt(WHITE)), now=1.0)
    for spacing in (1.0, 1.05, 1.2):
        nb = track(20, 400.0 + spacing * SW, sig=shirt(WHITE))  # same white shirt!
        d = gate.evaluate([nb], "SP25-BCS-031")
        check(d.verdict is Verdict.PRESENT,
              f"neighbour {spacing:.2f} shoulder-widths away, same white shirt -> Present",
              d.verdict.value)
    nb = track(21, 400.0 - 1.0 * SW, sig=shirt(WHITE))
    check(gate.evaluate([nb], "SP25-BCS-031").is_present, "neighbour on the other side -> Present")


def test_appearance_widens_only_for_a_match() -> None:
    section("Rule D: a lean (0.5-0.8 shoulder widths) needs matching clothes")
    gate = AttendanceGate(CFG)
    gate.commit("SP25-BCS-009", track(1, 400.0, sig=shirt(RED)), now=1.0)
    lean = 0.65 * SW
    d = gate.evaluate([track(5, 400.0 + lean, sig=shirt(RED))], "SP25-BCS-031")
    check(d.verdict is Verdict.REUSED_SEAT, "0.65 away + same clothes -> REUSED_SEAT", d.verdict.value)
    d = gate.evaluate([track(6, 400.0 + lean, sig=shirt(BLUE))], "SP25-BCS-031")
    check(d.verdict is Verdict.PRESENT, "0.65 away + different clothes -> Present", d.verdict.value)
    d = gate.evaluate([track(7, 400.0 + lean, sig=None)], "SP25-BCS-031")
    check(d.verdict is Verdict.PRESENT, "0.65 away + clothes unknown -> Present (no guessing)",
          d.verdict.value)


def test_row_behind_is_a_different_seat() -> None:
    section("Rule D: the student directly behind (smaller on screen) is a different seat")
    gate = AttendanceGate(CFG)
    gate.commit("SP25-BCS-009", track(1, 400.0, sig=shirt(RED)), now=1.0)
    # Two rows back: 0.55x the size, shoulders appearing just above the front
    # student's, horizontally aligned.
    behind = track(30, 400.0, sw=SW * 0.55, sy=ROW_Y - 0.3 * SW, sig=shirt(RED))
    d = gate.evaluate([behind], "SP25-BCS-031")
    check(d.verdict is Verdict.PRESENT, "different apparent size -> different seat", d.verdict.value)


def test_lifecycle() -> None:
    section("Rule D lifecycle: re-call, Undo, new session, switch off")
    gate = AttendanceGate(CFG)
    gate.commit("SP25-BCS-009", track(1, 400.0), now=1.0)
    d = gate.evaluate([track(2, 400.0)], "SP25-BCS-009")
    check(d.is_present, "the SAME roll called again (after Undo) is not reuse", d.verdict.value)
    gate.release("SP25-BCS-009")
    check("SP25-BCS-009" not in gate.seats, "release() (Undo) frees the seat")
    check(gate.evaluate([track(3, 400.0)], "SP25-BCS-031").is_present, "...so the seat can answer")

    gate.commit("SP25-BCS-009", track(4, 400.0), now=2.0)
    gate.reset_session()
    check(not gate.seats, "a new session starts with no seats")
    check(gate.evaluate([track(5, 400.0)], "SP25-BCS-031").is_present, "...and credits normally")

    off = AttendanceGate(CFG.override({"gating": {"seat_memory": False}}))
    off.commit("SP25-BCS-009", track(1, 400.0), now=1.0)
    check(off.evaluate([track(7, 400.0)], "SP25-BCS-031").is_present,
          "seat_memory=false restores the old behaviour")


def test_interaction_with_rules_a_and_b() -> None:
    section("Rule D with Rules A and B")
    gate = AttendanceGate(CFG)
    gate.commit("SP25-BCS-009", track(1, 400.0), now=1.0)

    # The flagged person and a genuine student both raise for 31.
    at_seat = track(7, 400.0)
    genuine = track(8, 400.0 + 3 * SW)
    d = gate.evaluate([at_seat, genuine], "SP25-BCS-031")
    check(d.verdict is Verdict.AMBIGUOUS, "two hands, one at a credited seat -> AMBIGUOUS", d.verdict.value)
    check("SP25-BCS-009's seat" in d.message, "the message says which hand is at whose seat", d.message)
    check(7 in d.seat_matches and 8 not in d.seat_matches, "seat_matches identifies the hand")

    # A hand at the credited seat that is merely latched (not held long enough)
    # alongside a genuine confirmed raise still makes it ambiguous...
    latched = track(9, 400.0)
    latched.confirmed = False
    d = gate.evaluate([genuine], "SP25-BCS-031", contenders=[latched, genuine])
    check(d.verdict is Verdict.AMBIGUOUS, "latched hand at the seat still counts for Rule A", d.verdict.value)
    # ...but on its own it is not yet anything.
    d = gate.evaluate([], "SP25-BCS-031", contenders=[latched])
    check(d.verdict is Verdict.NONE, "unconfirmed hand alone -> wait (NONE)", d.verdict.value)

    # Rule B still fires first when the tracker kept the ID.
    d = gate.evaluate([track(1, 400.0)], "SP25-BCS-031")
    check(d.verdict is Verdict.REUSED_TRACK, "same ID -> REUSED_TRACK as before", d.verdict.value)
    check(d.track is not None and d.track.track_id == 1, "REUSED_TRACK now carries the track to draw")

    # A teacher override must not erase who a track first answered for.
    gate.commit("SP25-BCS-031", track(1, 400.0), now=5.0)
    check(gate.credited[1] == "SP25-BCS-009", "a track keeps its first roll after an override",
          gate.credited[1])


def test_tracker_smooths_anchor() -> None:
    section("TemporalRaiseTracker carries a smoothed anchor and signature")
    tr = TemporalRaiseTracker(CFG)
    t = 0.0
    sig = shirt(RED)
    rng = np.random.default_rng(3)
    while t < 1.5:
        jitter = float(rng.normal(0.0, 3.0))
        ps, box = observe(400.0 + jitter, signature=sig)
        tr.update(t, {1: (ps, box)})
        t += 1 / 30
    st = tr.states[1]
    check(st.anchor is not None, "track has an anchor")
    check(abs(st.anchor[0] - 400.0) < 2.5, "smoothed anchor sits on the seat despite 3 px jitter",
          f"{st.anchor[0]:.2f}")
    check(similarity(st.signature, sig) > 0.99, "track signature follows the observations")


# --- session-level: the reported scenario ---------------------------------
class FakeVideo:
    def __init__(self):
        self.live = True

    def may_record_absence(self, now):
        return (True, "") if self.live else (False, "stalled")

    def snapshot(self):
        return None

    def status(self):
        return {}


class FakeSpeech:
    available = True
    error = ""

    def __init__(self):
        self._ev = []
        self.n = 0

    def speak(self, text, token):
        self._ev.append(SpeechEvent(token=token, kind="spoken"))

    def poll(self):
        out, self._ev = self._ev, []
        return out

    def clear(self):
        pass


def make_session(n: int, cfg=CFG):
    students = [Student(roll=f"SP25-BCS-{i:03d}") for i in range(1, n + 1)]
    assign_short_unique_calls(students)
    tracker = TemporalRaiseTracker(cfg)
    gate = AttendanceGate(cfg)
    sess = Session(cfg, FakeVideo(), tracker, gate, FakeSpeech())
    sess.set_students(students)
    return sess, students, tracker, gate


def run(sess, tracker, t, until, people: dict[int, tuple[float, bool]], sig=None):
    """Advance at 30 FPS. people: track_id -> (seat x, hand up?)."""
    view = None
    while t < until:
        obs = {}
        for tid, (cx, up) in people.items():
            obs[tid] = observe(cx, raised=up, signature=sig)
        tracker.update(t, obs)
        view = sess.tick(t)
        t += 1 / 30
    return view, t


def run_until(sess, tracker, t, people, cond, limit=30.0):
    """Step at 30 FPS until cond() holds; fail loudly instead of drifting past it."""
    end = t + limit
    while t < end:
        if cond():
            return t
        _v, t = run(sess, tracker, t, t + 1 / 30, people)
    raise AssertionError(f"condition not reached ({sess.phase}, current={sess.current})")


def wait_for_window(sess, tracker, t, people):
    return run_until(sess, tracker, t, people, lambda: sess.phase is Phase.RESPONSE)


def test_reported_scenario() -> None:
    section("the reported case: answers for 9, skips some calls, then raises for 31")
    # Rolls 009..031 stand in for "9" and "31"; we call 009 first, let a few
    # absent students pass, then call 031.
    sess, students, tracker, gate = make_session(5)
    seat_a, seat_b = 400.0, 400.0 + 1.1 * SW
    roll9, roll31 = students[0], students[4]

    sess.start(0.0)
    t = wait_for_window(sess, tracker, 0.0, {1: (seat_a, False), 2: (seat_b, False)})
    _v, t = run(sess, tracker, t, t + 2.0, {1: (seat_a, True), 2: (seat_b, False)})
    check(roll9.status == PRESENT and roll9.track_id == 1, "roll 9 present via track #1",
          f"{roll9.status} #{roll9.track_id}")

    # Three calls go by with nobody answering (they are absent).
    idle = {1: (seat_a, False), 2: (seat_b, False)}
    for i in (1, 2, 3):
        t = run_until(sess, tracker, t, idle, lambda i=i: students[i].status != PENDING)
        check(students[i].status == ABSENT, f"call {i + 1}: nobody raises -> Absent", students[i].status)

    # Somewhere in there the tracker lost student A for a moment: they come back
    # as track #57. Rule B alone no longer knows who they are.
    back = {57: (seat_a + 4.0, False), 2: (seat_b, False)}
    t = run_until(sess, tracker, t, back,
                  lambda: sess.current == 4 and sess.phase is Phase.RESPONSE)
    check(sess.students[sess.current] is roll31, "now calling roll 31", sess.students[sess.current].roll)
    view, t = run(sess, tracker, t, t + 2.0, {57: (seat_a + 4.0, True), 2: (seat_b, False)})

    check(roll31.status == PENDING, "roll 31 is NOT auto-credited to the person at seat 9",
          roll31.status)
    check(sess.phase is Phase.AMBIGUOUS, "the call stops for the teacher", sess.phase)
    check(view.flag == Verdict.REUSED_SEAT.value, "flag is REUSED_SEAT", view.flag)
    check(sess.result_text == "SAME SEAT", "console shows SAME SEAT", sess.result_text)
    check(view.prompt.startswith("ALREADY MARKED"), "student screen says why", view.prompt)
    check(roll9.roll in view.blocked_reason, "the reason names roll 9", view.blocked_reason)

    # It must not time out into a guess.
    _v, t = run(sess, tracker, t, t + 12.0, {57: (seat_a + 4.0, False), 2: (seat_b, False)})
    check(roll31.status == PENDING, "still waiting for the teacher 12 s later")

    sess.mark_manual(False, t)
    check(roll31.status == ABSENT, "teacher marks 31 Absent")
    rec = sess.telemetry[-1]
    check(rec["flag"] and rec["flag"]["verdict"] == "reused_seat", "telemetry keeps the flag",
          str(rec.get("flag")))
    check(rec["flag"]["seat_matches"][0]["seat_of"] == roll9.roll,
          "telemetry records whose seat it was")
    check(sess.telemetry[0]["seat"] is not None, "telemetry records each credited seat")


def test_without_seat_memory_the_bug_returns() -> None:
    section("control: with seat memory off, the same person IS credited twice (the bug)")
    cfg = CFG.override({"gating": {"seat_memory": False}})
    sess, students, tracker, _gate = make_session(2, cfg)
    sess.start(0.0)
    t = wait_for_window(sess, tracker, 0.0, {1: (400.0, False)})
    _v, t = run(sess, tracker, t, t + 2.0, {1: (400.0, True)})
    t = run_until(sess, tracker, t, {1: (400.0, False)},
                  lambda: sess.current == 1 and sess.phase is Phase.RESPONSE)
    _v, t = run(sess, tracker, t, t + 2.0, {57: (404.0, True)})
    check(students[0].status == PRESENT and students[1].status == PRESENT,
          "old behaviour: one person, new tracker ID, two Presents",
          f"{students[0].status} / {students[1].status}")


def test_neighbour_answers_normally() -> None:
    section("the student in the next seat answering for themselves is unaffected")
    sess, students, tracker, gate = make_session(2)
    seat_a, seat_b = 400.0, 400.0 + 1.0 * SW
    sess.start(0.0)
    t = wait_for_window(sess, tracker, 0.0, {1: (seat_a, False), 2: (seat_b, False)})
    _v, t = run(sess, tracker, t, t + 2.0, {1: (seat_a, True), 2: (seat_b, False)})
    check(students[0].status == PRESENT, "student A present")
    _v, t = run(sess, tracker, t, t + 1.5, {1: (seat_a, False), 2: (seat_b, False)})
    t = wait_for_window(sess, tracker, t, {1: (seat_a, False), 2: (seat_b, False)})
    _v, t = run(sess, tracker, t, t + 2.0, {1: (seat_a, False), 2: (seat_b, True)})
    check(students[1].status == PRESENT, "neighbour one shoulder-width away -> Present, no flag",
          f"{students[1].status} / {sess.phase}")


def test_teacher_override_after_flag() -> None:
    section("teacher confirms a flagged hand: credited, evidence is that hand")
    sess, students, tracker, gate = make_session(2)
    sess.start(0.0)
    t = wait_for_window(sess, tracker, 0.0, {1: (400.0, False)})
    _v, t = run(sess, tracker, t, t + 2.0, {1: (400.0, True)})
    _v, t = run(sess, tracker, t, t + 1.5, {1: (400.0, False)})
    t = wait_for_window(sess, tracker, t, {1: (400.0, False)})
    _v, t = run(sess, tracker, t, t + 2.0, {44: (400.0, True), 45: (700.0, False)})
    check(sess.phase is Phase.AMBIGUOUS, "flagged", sess.phase)
    sess.mark_manual(True, t)
    check(students[1].status == PRESENT, "teacher P -> Present")
    check(students[1].track_id == 44, "credited to the flagged hand, not some other track",
          str(students[1].track_id))


def test_annotation_points_at_the_seat() -> None:
    section("operator view draws the credited seat and the flagged hand")
    from annotate import MAGENTA, annotate
    from video_worker import FrameResult, PersonObservation

    gate = AttendanceGate(CFG)
    gate.commit("SP25-BCS-009", track(1, 400.0), now=1.0)
    st = track(7, 400.0)
    d = gate.evaluate([st], "SP25-BCS-031")
    kp = person_kp(400.0, raised=True)
    ps, box = observe(400.0)
    res = FrameResult(frame=np.zeros((600, 800, 3), np.uint8),
                      people=[PersonObservation(7, box, 0.9, kp, ps)])
    out = annotate(res.frame, res, CFG, track_states={7: st},
                   seats=list(gate.seats.values()), flag=d)
    mask = np.all(out == np.array(MAGENTA, np.uint8), axis=2)
    check(mask.sum() > 200, "magenta seat ring / hand box drawn", f"{int(mask.sum())} px")
    ring_y = int(gate.seats["SP25-BCS-009"].y)
    ring_x = int(gate.seats["SP25-BCS-009"].x + SW * 0.6)
    near = mask[ring_y - 3:ring_y + 4, ring_x - 3:ring_x + 4].any()
    check(bool(near), "the ring is centred on roll 9's seat")


def main() -> int:
    tests = [
        test_anchor_ignores_the_raised_arm,
        test_appearance_signature,
        test_same_seat_new_track_is_flagged,
        test_neighbours_are_not_flagged,
        test_appearance_widens_only_for_a_match,
        test_row_behind_is_a_different_seat,
        test_lifecycle,
        test_interaction_with_rules_a_and_b,
        test_tracker_smooths_anchor,
        test_reported_scenario,
        test_without_seat_memory_the_bug_returns,
        test_neighbour_answers_normally,
        test_teacher_override_after_flag,
        test_annotation_points_at_the_seat,
    ]
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # a crash is a failure, not an abort
            import traceback

            traceback.print_exc()
            check(False, f"{fn.__name__} raised", f"{type(exc).__name__}: {exc}")

    print("=" * 72)
    print("SEAT MEMORY TESTS (gating Rule D)")
    print("=" * 72)
    passed = failed = 0
    for ok, name, detail in _results:
        if ok is None:
            print(f"\n  -- {name} --")
            continue
        tag = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok
        print(f"  {tag}  {name}" + (f"   [{detail}]" if detail else ""))
    print()
    print(f"  {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
