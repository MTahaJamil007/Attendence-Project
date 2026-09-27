"""Tests for the liveness watchdog. No torch, camera or GPU required.

This covers the predecessor's worst failure mode. Its saved CSVs show gaps of 52 s,
40 s, 27 s and 22 s against a nominal 5 s cadence: inference stalled while the
state machine kept running on its own timer and wrote "Timeout: no valid hand" for
every remaining student. A missed detection loses one mark; a stall that is allowed
to resolve calls loses a whole session and looks clean in the log.

Run:  .venv\\Scripts\\python.exe tools\\test_liveness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from liveness import LivenessMonitor  # noqa: E402

_results: list[tuple[bool | None, str, str]] = []


def check(cond: bool, name: str, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


def section(title: str) -> None:
    _results.append((None, title, ""))


def test_before_first_frame() -> None:
    section("cold start")
    m = LivenessMonitor(0.75, "inference")
    # A fresh monitor must NOT read as alive. If it did, a session that began
    # marking before the first frame arrived would record absences against a
    # pipeline that had never run.
    check(not m.is_live(0.0), "a monitor with no output is not live")
    check(m.age(0.0) == float("inf"), "age is infinite before the first frame")
    allowed, reason = m.may_record_absence(0.0)
    check(not allowed, "absence refused before the first frame", reason)


def test_live_window() -> None:
    section("live window")
    m = LivenessMonitor(0.75, "inference")
    m.mark(10.0)
    check(m.is_live(10.0), "live at the instant of output")
    check(m.is_live(10.5), "live 0.50s later (inside 0.75s)")
    check(m.is_live(10.75), "live exactly at the threshold")
    check(not m.is_live(10.76), "stalled just past the threshold")
    check(abs(m.age(10.3) - 0.3) < 1e-9, "age is measured correctly", f"{m.age(10.3):.3f}")


def test_absence_gate_blocks_during_stall() -> None:
    section("absence gate during a stall")
    m = LivenessMonitor(0.75, "inference")
    m.mark(100.0)

    allowed, _ = m.may_record_absence(100.2)
    check(allowed, "absence permitted while the pipeline is live")

    # Reproduce the predecessor's 52-second stall.
    allowed, reason = m.may_record_absence(152.0)
    check(not allowed, "absence REFUSED during a 52s stall", reason)
    check("stalled" in reason, "reason names the stall", reason)

    # Recovery: once frames resume, marking is permitted again.
    m.mark(152.5)
    allowed, _ = m.may_record_absence(152.6)
    check(allowed, "absence permitted again after recovery")


def test_stall_would_have_saved_the_predecessor_runs() -> None:
    section("replay of the predecessor's observed gaps")
    # Real inter-mark gaps from Documents/SmartRollCall Attendance/*.csv.
    # Anything beyond the 0.75s threshold must block an automatic Absent.
    observed_gaps = [5.0, 6.0, 22.0, 27.0, 40.0, 52.0]
    m = LivenessMonitor(0.75, "inference")
    blocked = 0
    for gap in observed_gaps:
        m.reset()
        m.mark(0.0)
        allowed, _ = m.may_record_absence(gap)
        if not allowed:
            blocked += 1
    print(f"      gaps tested: {observed_gaps}")
    print(f"      blocked from auto-marking Absent: {blocked}/{len(observed_gaps)}")
    check(blocked == len(observed_gaps), "every observed stall blocks auto-marking")


def test_status_text() -> None:
    section("status text")
    m = LivenessMonitor(0.75, "inference")
    check("no output yet" in m.status(0.0), "cold status is explicit", m.status(0.0))
    m.mark(5.0)
    check("live" in m.status(5.1), "live status", m.status(5.1))
    check("STALLED" in m.status(9.0), "stalled status is loud", m.status(9.0))


def test_counter_and_reset() -> None:
    section("bookkeeping")
    m = LivenessMonitor(0.75)
    for i in range(5):
        m.mark(float(i))
    check(m.count == 5, "counts outputs", str(m.count))
    m.reset()
    check(m.count == 0 and not m.is_live(0.0), "reset returns to the cold state")


def test_rejects_bad_threshold() -> None:
    section("construction")
    for bad in (0.0, -1.0):
        try:
            LivenessMonitor(bad)
            check(False, f"rejects stall_threshold={bad}")
        except ValueError:
            check(True, f"rejects stall_threshold={bad}")


def main() -> int:
    for fn in (
        test_before_first_frame,
        test_live_window,
        test_absence_gate_blocks_during_stall,
        test_stall_would_have_saved_the_predecessor_runs,
        test_status_text,
        test_counter_and_reset,
        test_rejects_bad_threshold,
    ):
        fn()

    print()
    print("=" * 70)
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
    print("=" * 70)
    print(f"  {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
