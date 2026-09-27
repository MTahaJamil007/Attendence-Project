"""Tests for the calibration replay and sweep. No camera, GPU or recording needed.

Builds a synthetic recording - a pose cache plus labels, in exactly the format
record.py and extract.py produce - and runs the real calibration code over it.

The synthetic footage is clean by construction, so a correct implementation must
score close to perfect on it. That is the point: if the replay cannot get 100%
recall on data where every raise is unambiguous, any number it reports on real
footage is meaningless.

Run:  .venv\\Scripts\\python.exe tools\\test_calibrate.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from config import Config  # noqa: E402

import calibrate as C  # noqa: E402
from test_detector import BBOX_H, POSES, make_person  # noqa: E402

CFG = Config.load(ROOT / "config.json")
FPS = 15.0

_results: list[tuple[bool | None, str, str]] = []


def check(cond: bool, name: str, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


def section(title: str) -> None:
    _results.append((None, title, ""))


# --- synthetic recording --------------------------------------------------
SCRIPT = [
    ("idle", "at_rest", False, 3.0),
    ("raise", "straight_up_raise", True, 4.0),
    ("idle", "at_rest", False, 3.0),
    ("scratch", "scratch_head", False, 4.0),
    ("idle", "at_rest", False, 3.0),
    ("raise", "forearm_up_raise", True, 4.0),
    ("idle", "at_rest", False, 3.0),
    ("stretch", "sideways_stretch", False, 4.0),
    ("idle", "at_rest", False, 3.0),
    ("phone", "phone_raised", False, 4.0),
    ("idle", "at_rest", False, 3.0),
    ("raise_high", "straight_up_raise", True, 4.0),
    ("idle", "at_rest", False, 3.0),
]


def build_recording(conf: float = 0.9, track_id: int = 1, script=None):
    """Return (labels dict, npz-shaped arrays) for a synthetic recording."""
    script = script or SCRIPT
    frames, ts, tids, bboxes, kps = [], [], [], [], []
    segments = []
    t = 0.0
    fi = 0
    for kind, pose, expect, secs in script:
        start = t
        n = int(secs * FPS)
        for _ in range(n):
            kp = make_person(pose, conf=conf)
            frames.append(fi)
            ts.append(round(t, 4))
            tids.append(track_id)
            # A box tall enough to clear min_person_height_px.
            bboxes.append([150.0, 120.0, 260.0, 120.0 + BBOX_H])
            kps.append(np.array(kp, dtype=np.float32))
            fi += 1
            t += 1.0 / FPS
        segments.append({
            "kind": kind, "prompt": kind, "expect_raise": expect,
            "start": round(start, 3), "end": round(t, 3),
        })
    labels = {
        "name": "synthetic", "duration_sec": round(t, 2), "measured_fps": FPS,
        "camera": "synthetic", "frames": fi,
        "frame_times": ts, "segments": segments,
    }
    data = {
        "frame": np.array(frames, dtype=np.int32),
        "t": np.array(ts, dtype=np.float64),
        "track_id": np.array(tids, dtype=np.int32),
        "bbox": np.array(bboxes, dtype=np.float32).reshape(-1, 4),
        "kp": np.array(kps, dtype=np.float32).reshape(-1, 17, 3),
    }
    return labels, data


# =========================================================================
def test_replay_on_clean_footage() -> None:
    section("replay on clean synthetic footage")
    labels, data = build_recording()
    frames, small, untracked = C.group_by_frame(data, float(CFG.require("model.min_person_height_px")))
    check(len(frames) > 0, "frames grouped", f"{len(frames)} frames")
    check(small == 0 and untracked == 0, "nothing skipped", f"small={small} untracked={untracked}")

    scored = C.score_frames(frames, CFG)
    m = C.replay(scored, labels["segments"], CFG)
    print(f"      TP={m.tp} FN={m.fn} FP={m.fp} TN={m.tn}  "
          f"recall={m.recall:.1%} precision={m.precision:.1%}")
    check(m.tp + m.fn == 3, "three genuine raises evaluated", f"{m.tp + m.fn}")
    check(m.recall == 1.0, "every genuine raise detected", f"{m.recall:.1%}")
    check(m.fp == 0, "no distractor fired", f"fp={m.fp} {m.fp_by_kind}")
    check(m.precision == 1.0, "precision 1.0 on clean footage", f"{m.precision:.1%}")


def test_metrics_arithmetic() -> None:
    section("metrics")
    m = C.Metrics(tp=9, fn=1, fp=2, tn=8, fp_by_kind={})
    check(abs(m.recall - 0.9) < 1e-9, "recall", f"{m.recall}")
    check(abs(m.precision - 9 / 11) < 1e-9, "precision", f"{m.precision:.4f}")
    check(abs(m.fp_rate - 0.2) < 1e-9, "false-positive rate", f"{m.fp_rate}")
    check(0.8 < m.f1 < 0.9, "f1 between precision and recall", f"{m.f1:.4f}")
    empty = C.Metrics()
    check(empty.recall == 0.0 and empty.f1 == 0.0, "empty metrics do not divide by zero")


def test_grid_sweep_runs_and_ranks() -> None:
    section("grid sweep")
    labels, data = build_recording()
    frames, _s, _u = C.group_by_frame(data, float(CFG.require("model.min_person_height_px")))
    grid = {
        "temporal.enter_threshold": [0.45, 0.55, 0.70],
        "temporal.hold_sec": [0.4, 0.8],
        "raise.straightness_floor": [0.2, 0.4],
    }
    rows = C.evaluate_grid(CFG, frames, labels["segments"], grid, exit_ratio=0.64)
    check(len(rows) == 12, "every combination evaluated", f"{len(rows)}")
    rows.sort(key=lambda r: (r[1].f1, r[1].recall), reverse=True)
    best_combo, best = rows[0]
    print(f"      best: recall={best.recall:.1%} precision={best.precision:.1%} "
          f"{C.fmt(best_combo)}")
    check(best.f1 >= 0.99, "a near-perfect operating point exists on clean data",
          f"F1={best.f1:.3f}")
    worst = rows[-1][1]
    check(best.f1 >= worst.f1, "ranking is ordered by F1", f"{best.f1:.3f} >= {worst.f1:.3f}")


def test_threshold_sensitivity_is_visible() -> None:
    section("sensitivity")
    labels, data = build_recording()
    frames, _s, _u = C.group_by_frame(data, float(CFG.require("model.min_person_height_px")))
    # An absurdly high enter threshold must destroy recall; the sweep has to be
    # able to see that, or it cannot distinguish a good setting from a bad one.
    high = CFG.override({"temporal": {"enter_threshold": 0.99, "exit_threshold": 0.9}})
    m_high = C.replay(C.score_frames(frames, high), labels["segments"], high)
    low = CFG.override({"temporal": {"enter_threshold": 0.05, "exit_threshold": 0.02}})
    m_low = C.replay(C.score_frames(frames, low), labels["segments"], low)
    print(f"      enter=0.99 -> recall {m_high.recall:.1%}, fp {m_high.fp}")
    print(f"      enter=0.05 -> recall {m_low.recall:.1%}, fp {m_low.fp}")
    check(m_high.recall < 1.0, "an unreachable threshold loses recall", f"{m_high.recall:.1%}")
    check(m_low.fp > 0, "a permissive threshold admits distractors", f"fp={m_low.fp}")
    check(m_low.fp_by_kind, "false positives are attributed to an action",
          str(m_low.fp_by_kind))


def test_low_confidence_keypoints_degrade() -> None:
    section("degraded keypoints")
    floor = float(CFG.require("model.keypoint_conf_floor"))
    labels, data = build_recording(conf=floor * 0.5)
    frames, _s, _u = C.group_by_frame(data, float(CFG.require("model.min_person_height_px")))
    m = C.replay(C.score_frames(frames, CFG), labels["segments"], CFG)
    check(m.recall == 0.0, "keypoints below the floor detect nothing, rather than guessing",
          f"{m.recall:.1%}")
    check(m.fp == 0, "and invent no false positives either", f"fp={m.fp}")


def test_small_and_untracked_filtered() -> None:
    section("detection filtering mirrors production")
    labels, data = build_recording()
    # Too short to judge.
    data2 = dict(data)
    data2["bbox"] = np.array(
        [[150.0, 120.0, 260.0, 150.0]] * len(data["frame"]), dtype=np.float32
    )
    frames, small, _u = C.group_by_frame(data2, 90.0)
    check(len(frames) == 0 and small > 0, "detections below min height are dropped",
          f"skipped {small}")

    # No track id: production cannot use these either.
    data3 = dict(data)
    data3["track_id"] = np.full(len(data["frame"]), -1, dtype=np.int32)
    frames, _s, untracked = C.group_by_frame(data3, 90.0)
    check(len(frames) == 0 and untracked > 0, "untracked detections are dropped",
          f"skipped {untracked}")


def test_one_update_per_frame() -> None:
    section("multi-person frames")
    # Two people in the same frame must arrive in ONE tracker update. Feeding them
    # separately makes each look momentarily unseen to the other - the bug that
    # broke Rule A in Phase 2.
    labels, data = build_recording()
    n = len(data["frame"])
    data2 = {
        "frame": np.concatenate([data["frame"], data["frame"]]),
        "t": np.concatenate([data["t"], data["t"]]),
        "track_id": np.concatenate([data["track_id"], np.full(n, 2, dtype=np.int32)]),
        "bbox": np.concatenate([data["bbox"], data["bbox"]]),
        "kp": np.concatenate([data["kp"], data["kp"]]),
    }
    frames, _s, _u = C.group_by_frame(data2, 90.0)
    check(all(len(f["people"]) == 2 for f in frames),
          "both people land in the same frame group",
          f"{set(len(f['people']) for f in frames)}")


def test_overrides_and_split() -> None:
    section("parameter plumbing")
    nested = C.overrides_from({"temporal.enter_threshold": 0.6, "raise.straightness_floor": 0.3})
    check(nested["temporal"]["enter_threshold"] == 0.6, "dotted keys become nested config")
    check(nested["raise"]["straightness_floor"] == 0.3, "across sections")
    s, t = C.split_params({
        "raise.straightness_floor": 0.3,
        "temporal.hold_sec": 0.6,
        "model.keypoint_conf": 0.3,
    })
    check(set(s) == {"raise.straightness_floor", "model.keypoint_conf"},
          "scorer params identified", str(sorted(s)))
    check(set(t) == {"temporal.hold_sec"}, "temporal params identified", str(sorted(t)))

    cfg2 = CFG.override(nested)
    check(cfg2.get("temporal.enter_threshold") == 0.6, "override applies")
    check(CFG.get("temporal.enter_threshold") != 0.6, "base config is untouched")


def test_missing_recording_is_reported() -> None:
    section("missing inputs")
    try:
        C.load_recording("definitely-not-a-recording", "yolo11m-pose.pt", 1280)
        check(False, "missing recording raises")
    except SystemExit as exc:
        check("record.py" in str(exc), "missing recording tells you how to make one",
              str(exc).splitlines()[-1][:60])

    tmp = Path(tempfile.mkdtemp())
    try:
        rec = ROOT / "data" / "calib" / "_test_no_poses"
        rec.mkdir(parents=True, exist_ok=True)
        (rec / "labels.json").write_text(json.dumps({"segments": []}), encoding="utf-8")
        try:
            C.load_recording("_test_no_poses", "yolo11m-pose.pt", 1280)
            check(False, "missing pose cache raises")
        except SystemExit as exc:
            check("extract.py" in str(exc), "missing pose cache tells you to extract",
                  str(exc).splitlines()[1][:60] if len(str(exc).splitlines()) > 1 else "")
        shutil.rmtree(rec, ignore_errors=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    for fn in (
        test_replay_on_clean_footage,
        test_metrics_arithmetic,
        test_grid_sweep_runs_and_ranks,
        test_threshold_sensitivity_is_visible,
        test_low_confidence_keypoints_degrade,
        test_small_and_untracked_filtered,
        test_one_update_per_frame,
        test_overrides_and_split,
        test_missing_recording_is_reported,
    ):
        try:
            fn()
        except Exception as exc:
            check(False, f"{fn.__name__} crashed", f"{type(exc).__name__}: {exc}")

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
