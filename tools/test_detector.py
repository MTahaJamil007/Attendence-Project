"""Logic tests for the perception core. Runs without torch, CUDA or a camera.

These cover the two failures that made the predecessor inaccurate, so they are
regression tests in the literal sense:

* distractor rejection - a head-scratch or a stretch must not read as a raise;
* frame-rate independence - identical footage must produce an identical verdict
  at 5 FPS and at 40 FPS, including with dropped detections.

Run:  .venv\\Scripts\\python.exe tools\\test_detector.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config  # noqa: E402
from gating import AttendanceGate, Verdict  # noqa: E402
from geometry import (  # noqa: E402
    LEFT_ELBOW,
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    NOSE,
    RIGHT_ELBOW,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
)
from raise_detector import PersonScore, RaiseScorer, TemporalRaiseTracker, TrackState  # noqa: E402

CFG = Config.load(ROOT / "config.json")

# --- synthetic skeleton ---------------------------------------------------
# Right shoulder at (220, 200), hips at y=300 -> torso = 100 px, nose at y=160.
# y grows downward, so "above" means a smaller y.
SHOULDER_Y = 200.0
RIGHT_SHOULDER_X = 220.0
HIP_Y = 300.0
NOSE_XY = (218.0, 160.0)

POSES: dict[str, tuple[float, float, float, float]] = {
    # name: (elbow_x, elbow_y, wrist_x, wrist_y)
    "straight_up_raise": (222, 150, 224, 95),
    "forearm_up_raise": (225, 200, 230, 150),
    "scratch_head": (250, 195, 205, 165),
    "sideways_stretch": (290, 190, 360, 185),
    "at_rest": (225, 260, 230, 310),
    "phone_raised": (240, 215, 250, 170),
}


def make_person(pose: str, conf: float = 0.9, nose_visible: bool = True):
    """Build a 17-point COCO skeleton with the right arm in *pose*."""
    ex, ey, wx, wy = POSES[pose]
    kp = [(0.0, 0.0, 0.0)] * 17
    kp[NOSE] = (NOSE_XY[0], NOSE_XY[1], conf if nose_visible else 0.0)
    kp[LEFT_SHOULDER] = (180.0, SHOULDER_Y, conf)
    kp[RIGHT_SHOULDER] = (RIGHT_SHOULDER_X, SHOULDER_Y, conf)
    kp[LEFT_HIP] = (185.0, HIP_Y, conf)
    kp[RIGHT_HIP] = (215.0, HIP_Y, conf)
    # Left arm parked at rest so it is gated out and never wins.
    kp[LEFT_ELBOW] = (175.0, 260.0, conf)
    kp[LEFT_WRIST] = (170.0, 310.0, conf)
    kp[RIGHT_ELBOW] = (float(ex), float(ey), conf)
    kp[RIGHT_WRIST] = (float(wx), float(wy), conf)
    return kp


BBOX_H = 260.0  # plausible seated-person box height


# --- tiny test runner -----------------------------------------------------
_results: list[tuple[bool, str, str]] = []


def check(cond: bool, name: str, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


def section(title: str) -> None:
    _results.append((None, title, ""))  # type: ignore[arg-type]


# =========================================================================
# 1. Pose discrimination
# =========================================================================
def test_pose_discrimination() -> None:
    section("pose discrimination")
    scorer = RaiseScorer(CFG)
    enter = float(CFG.require("temporal.enter_threshold"))
    scores = {}
    for name in POSES:
        ps = scorer.score(make_person(name), BBOX_H)
        scores[name] = ps.score

    for name, s in scores.items():
        print(f"      {name:<20s} score = {s:.3f}")

    check(scores["at_rest"] == 0.0, "at_rest scores exactly 0 (hard gate)", f"{scores['at_rest']:.3f}")
    check(scores["straight_up_raise"] > 0.90, "straight-up raise > 0.90", f"{scores['straight_up_raise']:.3f}")
    check(scores["forearm_up_raise"] > enter, f"forearm-up raise passes enter ({enter})", f"{scores['forearm_up_raise']:.3f}")
    check(scores["scratch_head"] < enter, "head-scratch rejected", f"{scores['scratch_head']:.3f}")
    check(scores["sideways_stretch"] < enter, "sideways stretch rejected", f"{scores['sideways_stretch']:.3f}")
    check(scores["phone_raised"] < enter, "raised phone at face height rejected", f"{scores['phone_raised']:.3f}")
    margin = scores["forearm_up_raise"] - scores["scratch_head"]
    check(margin > 0.30, "raise/distractor margin > 0.30", f"{margin:.3f}")


def test_occluded_nose_still_rejects_distractor() -> None:
    section("distractor rejection with nose occluded")
    scorer = RaiseScorer(CFG)
    enter = float(CFG.require("temporal.enter_threshold"))
    scratch = scorer.score(make_person("scratch_head", nose_visible=False), BBOX_H).score
    raise_ = scorer.score(make_person("forearm_up_raise", nose_visible=False), BBOX_H).score
    print(f"      scratch_head (no nose)     score = {scratch:.3f}")
    print(f"      forearm_up_raise (no nose) score = {raise_:.3f}")
    # This is the case that broke when straightness was merely a weighted term:
    # losing the nose dropped vertical_extension and pushed a head-scratch to 0.61.
    check(scratch < enter, "head-scratch still rejected without nose keypoint", f"{scratch:.3f}")
    check(raise_ > enter, "genuine raise still accepted without nose keypoint", f"{raise_:.3f}")


def test_scale_invariance() -> None:
    section("scale invariance (front row vs back row)")
    scorer = RaiseScorer(CFG)
    base = scorer.score(make_person("straight_up_raise"), BBOX_H).score

    # Same pose, uniformly scaled to a quarter size around the origin: a student
    # four times further away. Torso normalization should make this identical.
    kp = make_person("straight_up_raise")
    small = [(x * 0.25, y * 0.25, c) for (x, y, c) in kp]
    small_score = scorer.score(small, BBOX_H * 0.25).score
    print(f"      full size score = {base:.3f}   quarter size score = {small_score:.3f}")
    check(abs(base - small_score) < 0.02, "score invariant to subject scale", f"delta {abs(base - small_score):.4f}")


def test_confidence_floor() -> None:
    section("confidence handling")
    scorer = RaiseScorer(CFG)
    floor = float(CFG.require("model.keypoint_conf_floor"))
    noisy = scorer.score(make_person("straight_up_raise", conf=floor * 0.5), BBOX_H)
    check(noisy.score == 0.0, "keypoints below floor abstain rather than guess", noisy.reason)
    crisp = scorer.score(make_person("straight_up_raise", conf=0.95), BBOX_H).score
    mid = scorer.score(make_person("straight_up_raise", conf=0.22), BBOX_H).score
    print(f"      conf 0.95 -> {crisp:.3f}   conf 0.22 -> {mid:.3f}")
    check(mid < crisp, "lower keypoint confidence yields a lower score", f"{mid:.3f} < {crisp:.3f}")


# =========================================================================
# 2. Temporal filter - the frame-rate regression tests
# =========================================================================
def _run_stream(fps: float, duration: float, score_of, drop_every: int = 0):
    """Feed a synthetic stream into the tracker; return (confirm_time, tracker)."""
    tracker = TemporalRaiseTracker(CFG)
    tracker.arm(0.0)
    dt = 1.0 / fps
    n = int(duration / dt)
    confirm_t: float | None = None
    for i in range(n + 1):
        t = i * dt
        obs = {}
        if not (drop_every and i % drop_every == 0 and i > 0):
            ps = PersonScore()
            ps.score = score_of(t)
            ps.torso_px = 100.0
            obs[1] = (ps, (200.0, 100.0, 260.0, 360.0))
        confirmed = tracker.update(t, obs)
        if confirmed and confirm_t is None:
            confirm_t = t
    return confirm_t, tracker


def test_fps_independence() -> None:
    section("frame-rate independence")
    step = lambda t: 1.0 if t >= 0.0 else 0.0  # noqa: E731
    tau = float(CFG.require("temporal.ema_tau_sec"))
    hold = float(CFG.require("temporal.hold_sec"))
    enter = float(CFG.require("temporal.enter_threshold"))
    ideal = tau * math.log(1.0 / (1.0 - enter)) + hold

    times = {}
    for fps in (5.0, 10.0, 20.0, 40.0, 60.0):
        t, _ = _run_stream(fps, 3.0, step)
        times[fps] = t
        print(f"      {fps:5.0f} FPS -> confirmed at {t:.3f}s")
    print(f"      continuous-time ideal      {ideal:.3f}s")

    check(all(t is not None for t in times.values()), "confirms at every frame rate")
    spread = max(times.values()) - min(times.values())
    # Quantization is bounded by the slowest stream's frame interval (1/5 s).
    check(spread <= 0.21, "confirmation time spread <= one slow-frame interval", f"{spread:.3f}s")
    check(all(abs(t - ideal) < 0.25 for t in times.values()), "all within 0.25s of the continuous ideal")


def test_dropout_tolerance() -> None:
    section("detection dropout tolerance")
    step = lambda t: 1.0  # noqa: E731
    # 5 FPS with every 3rd detection missing. Frame interval 0.2s, so each gap is
    # 0.2s. The predecessor tolerated gaps of only 0.28s while requiring 0.85s of
    # continuous detection, so this pattern reset its chain repeatedly and logged
    # a held raise as "Timeout: no valid hand".
    t, tracker = _run_stream(5.0, 4.0, step, drop_every=3)
    print(f"      5 FPS, 1-in-3 detections dropped -> confirmed at {t}")
    check(t is not None, "still confirms despite periodic dropouts")
    check(t is not None and t < 1.6, "confirms promptly (< 1.6s)", f"{t}")


def test_hysteresis_no_chatter() -> None:
    section("hysteresis")
    enter = float(CFG.require("temporal.enter_threshold"))
    exit_ = float(CFG.require("temporal.exit_threshold"))
    # Park the input between the two thresholds and jitter it. A single-threshold
    # design flickers here; a Schmitt trigger must not.
    mid = (enter + exit_) / 2.0
    tracker = TemporalRaiseTracker(CFG)
    tracker.arm(0.0)
    flips = 0
    prev = False
    for i in range(200):
        t = i * 0.025
        ps = PersonScore()
        ps.score = mid + (0.06 if i % 2 else -0.06)
        ps.torso_px = 100.0
        tracker.update(t, {1: (ps, (200.0, 100.0, 260.0, 360.0))})
        cur = tracker.states[1].latched
        if cur != prev:
            flips += 1
        prev = cur
    print(f"      latch transitions while input jitters mid-band: {flips}")
    check(flips <= 1, "no latch chatter in the dead band", f"{flips} transitions")


def test_brief_raise_not_credited() -> None:
    section("hold requirement")
    hold = float(CFG.require("temporal.hold_sec"))
    # A hand up for well under the hold time - a wave, or an arm passing through
    # frame - must not be credited.
    brief = lambda t: 1.0 if 0.2 <= t <= 0.2 + hold * 0.4 else 0.0  # noqa: E731
    t, _ = _run_stream(30.0, 3.0, brief)
    print(f"      raise held {hold * 0.4:.2f}s (hold={hold}s) -> confirmed at {t}")
    check(t is None, "a raise shorter than hold_sec is not credited")


def test_fresh_raise_requirement() -> None:
    section("fresh-raise requirement")
    tracker = TemporalRaiseTracker(CFG)
    # Hand already up and confirmed before the window opens.
    tracker.arm(0.0)
    t = 0.0
    while t < 2.0:
        ps = PersonScore()
        ps.score = 1.0
        ps.torso_px = 100.0
        tracker.update(t, {1: (ps, (200.0, 100.0, 260.0, 360.0))})
        t += 0.033
    check(len(tracker.confirmed()) == 1, "baseline: held hand is confirmed")

    # New roll number called while that hand stays up.
    tracker.arm(t)
    for _ in range(40):
        ps = PersonScore()
        ps.score = 1.0
        ps.torso_px = 100.0
        tracker.update(t, {1: (ps, (200.0, 100.0, 260.0, 360.0))})
        t += 0.033
    stale = tracker.confirmed()
    print(f"      confirmed tracks after re-arming with the hand still up: {len(stale)}")
    # The latch predates the new window, so it must not count for the new call.
    check(len(stale) == 0, "a hand parked up cannot answer a second roll number")

    # ...but a student who lowers and re-raises is credited normally.
    for _ in range(30):  # drop below the exit threshold to unlatch
        ps = PersonScore()
        ps.score = 0.0
        ps.torso_px = 100.0
        tracker.update(t, {1: (ps, (200.0, 100.0, 260.0, 360.0))})
        t += 0.033
    for _ in range(60):
        ps = PersonScore()
        ps.score = 1.0
        ps.torso_px = 100.0
        tracker.update(t, {1: (ps, (200.0, 100.0, 260.0, 360.0))})
        t += 0.033
    check(len(tracker.confirmed()) == 1, "lowering and re-raising is credited")


def test_eager_raise_within_lookback() -> None:
    section("eager raise (lookback tolerance)")
    lookback = float(CFG.require("temporal.fresh_lookback_sec"))
    tracker = TemporalRaiseTracker(CFG)
    # A student raises the moment they hear their number, a little BEFORE the
    # response window formally opens. Without a lookback this cooperative
    # behaviour would be discarded as a stale latch.
    t = 0.0
    for _ in range(25):  # ~0.8s of raised hand before arming
        ps = PersonScore()
        ps.score = 1.0
        ps.torso_px = 100.0
        tracker.update(t, {1: (ps, (200.0, 100.0, 260.0, 360.0))})
        t += 0.033
    tracker.arm(t)  # window opens now, after the raise already began
    for _ in range(10):
        ps = PersonScore()
        ps.score = 1.0
        ps.torso_px = 100.0
        tracker.update(t, {1: (ps, (200.0, 100.0, 260.0, 360.0))})
        t += 0.033
    n = len(tracker.confirmed())
    print(f"      raise began {0.8:.1f}s before arming (lookback {lookback}s) -> confirmed: {n}")
    check(n == 1, "a raise just before the window opens still counts")


# =========================================================================
# 3. Gating rules
# =========================================================================
def _track(tid: int, ema: float = 0.9, bbox=(200.0, 100.0, 260.0, 360.0)) -> TrackState:
    st = TrackState(track_id=tid)
    st.ema = ema
    st.peak_ema = ema
    st.confirmed = True
    st.bbox = bbox
    st.best_bbox = bbox
    ps = PersonScore()
    ps.score = ema
    ps.torso_px = 100.0
    st.best_score = ps
    st.last_score = ps
    return st


def test_gating_rules() -> None:
    section("gating rules")
    gate = AttendanceGate(CFG)

    d = gate.evaluate([], "SP25-BCS-001")
    check(d.verdict is Verdict.NONE, "no raise -> NONE")

    d = gate.evaluate([_track(1)], "SP25-BCS-001")
    check(d.verdict is Verdict.PRESENT, "single raise -> PRESENT", d.message)
    gate.commit("SP25-BCS-001", d.track)

    # Rule A
    d = gate.evaluate([_track(2), _track(3)], "SP25-BCS-002")
    check(d.verdict is Verdict.AMBIGUOUS, "two raises -> AMBIGUOUS (Rule A)", d.message)

    # Rule B - track 1 was already credited to 001.
    d = gate.evaluate([_track(1)], "SP25-BCS-002")
    check(d.verdict is Verdict.REUSED_TRACK, "already-credited person -> REUSED_TRACK (Rule B)", d.message)

    # A different person answering 002 is fine.
    d = gate.evaluate([_track(4)], "SP25-BCS-002")
    check(d.verdict is Verdict.PRESENT, "a fresh person can answer the next roll", d.message)

    # Undo frees the credit again.
    gate.commit("SP25-BCS-002", d.track)
    gate.release("SP25-BCS-002")
    d = gate.evaluate([_track(4)], "SP25-BCS-002")
    check(d.verdict is Verdict.PRESENT, "release() frees the track for a re-run", d.message)


def test_seat_zone_lock() -> None:
    section("seat zone lock (Rule C)")
    cfg = CFG.override({"gating": {"seat_zone_lock": True}})
    gate = AttendanceGate(cfg)

    near = (200.0, 100.0, 260.0, 360.0)
    far = (1400.0, 100.0, 1460.0, 360.0)

    d = gate.evaluate([_track(1, bbox=near)], "SP25-BCS-007")
    gate.commit("SP25-BCS-007", d.track)
    check(d.verdict is Verdict.PRESENT, "first answer establishes the seat zone")

    gate.reset_session()  # new period; zones persist, credits do not
    d = gate.evaluate([_track(9, bbox=near)], "SP25-BCS-007")
    check(d.verdict is Verdict.PRESENT, "same seat next session -> PRESENT", d.message)

    gate.reset_session()
    d = gate.evaluate([_track(10, bbox=far)], "SP25-BCS-007")
    check(d.verdict is Verdict.ZONE_MISMATCH, "far-away answer -> ZONE_MISMATCH", d.message)
    check(d.needs_teacher, "zone mismatch escalates to the teacher rather than auto-marking")


# =========================================================================
def main() -> int:
    tests = [
        test_pose_discrimination,
        test_occluded_nose_still_rejects_distractor,
        test_scale_invariance,
        test_confidence_floor,
        test_fps_independence,
        test_dropout_tolerance,
        test_hysteresis_no_chatter,
        test_brief_raise_not_credited,
        test_fresh_raise_requirement,
        test_eager_raise_within_lookback,
        test_gating_rules,
        test_seat_zone_lock,
    ]
    for fn in tests:
        fn()

    print()
    print("=" * 70)
    passed = failed = 0
    for ok, name, detail in _results:
        if ok is None:
            print(f"\n  -- {name} --")
            continue
        tag = "PASS" if ok else "FAIL"
        suffix = f"   [{detail}]" if detail else ""
        print(f"  {tag}  {name}{suffix}")
        if ok:
            passed += 1
        else:
            failed += 1
    print("=" * 70)
    print(f"  {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
