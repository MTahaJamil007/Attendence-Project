"""Sweep detection thresholds against labelled footage and report recall/precision.

This is the step that turns the thresholds from defensible-looking defaults into
measured choices. Everything before it was validated against synthetic poses, which
prove the maths is right but say nothing about whether 0.55 is the correct enter
threshold for a real room.

    python tools/calibrate.py --name classroom-1              # grid sweep
    python tools/calibrate.py --name classroom-1 --mode coord # one-at-a-time
    python tools/calibrate.py --name classroom-1 --apply      # write the winner

The replay mirrors production exactly: detections are grouped per frame and fed to
the tracker in a single update (feeding one track at a time would make each look
momentarily unseen to the others), the tracker is armed at each segment boundary
just as it is when a roll number is called, and real recorded timestamps are used
so the time-constant filter behaves as it did live.

Scorer parameters and temporal parameters are swept in two levels: scores are
computed once per distinct scorer configuration and reused across every temporal
combination. Recomputing them would multiply the work by the size of the temporal
grid for no benefit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from config import Config  # noqa: E402
from raise_detector import RaiseScorer, TemporalRaiseTracker  # noqa: E402

# Swept by default. Kept small on purpose: a grid nobody waits for is a grid
# nobody runs.
DEFAULT_GRID = {
    "temporal.enter_threshold": [0.45, 0.50, 0.55, 0.60, 0.65],
    "temporal.hold_sec": [0.40, 0.60, 0.80],
    "temporal.ema_tau_sec": [0.12, 0.18, 0.25],
    "raise.straightness_floor": [0.20, 0.30, 0.40],
    # How far below nose height a hand held clear of the face may be and still
    # count. Higher accepts slighter raises; the sweep weighs that against the
    # distractors in the recording.
    "raise.beside_head_allowance": [0.15, 0.30],
}
SCORER_KEYS = ("raise.", "model.keypoint_conf", "model.keypoint_conf_floor")


@dataclass
class Metrics:
    tp: int = 0
    fn: int = 0
    fp: int = 0
    tn: int = 0
    fp_by_kind: dict | None = None

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else (1.0 if self.tp == 0 and self.fp == 0 else 0.0)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def fp_rate(self) -> float:
        d = self.fp + self.tn
        return self.fp / d if d else 0.0


def load_recording(name: str, weights: str, imgsz: int):
    rec = ROOT / "data" / "calib" / name
    labels_path = rec / "labels.json"
    if not labels_path.exists():
        raise SystemExit(f"no recording at {rec}\n"
                         f"record one first:  python tools/record.py --name {name}")
    labels = json.loads(labels_path.read_text(encoding="utf-8"))

    from extract import cache_path

    npz_path = cache_path(rec, weights, imgsz)
    if not npz_path.exists():
        available = sorted(p.name for p in rec.glob("poses_*.npz"))
        raise SystemExit(
            f"no pose cache {npz_path.name}\n"
            f"extract first:  python tools/extract.py --name {name}"
            + (f"\navailable caches: {', '.join(available)}" if available else "")
        )
    data = np.load(npz_path, allow_pickle=False)
    return labels, data, npz_path


def group_by_frame(data, min_height: float):
    """Rebuild per-frame detection groups, exactly as VideoWorker publishes them."""
    frame = data["frame"]
    t = data["t"]
    tid = data["track_id"]
    bbox = data["bbox"]
    kp = data["kp"]

    frames: dict[int, dict] = {}
    skipped_small = 0
    skipped_untracked = 0
    for i in range(len(frame)):
        h = float(bbox[i][3] - bbox[i][1])
        if h < min_height:
            skipped_small += 1
            continue
        if int(tid[i]) < 0:
            skipped_untracked += 1
            continue
        f = int(frame[i])
        entry = frames.setdefault(f, {"t": float(t[i]), "people": []})
        entry["people"].append((int(tid[i]), tuple(float(v) for v in bbox[i]), kp[i]))
    ordered = [frames[k] for k in sorted(frames)]
    return ordered, skipped_small, skipped_untracked


def score_frames(frames, cfg):
    """Score every detection once for a given scorer configuration."""
    scorer = RaiseScorer(cfg)
    out = []
    for fr in frames:
        people = []
        for tid, bbox, kp in fr["people"]:
            kps = [(float(a), float(b), float(c)) for a, b, c in kp]
            ps = scorer.score(kps, bbox[3] - bbox[1])
            people.append((tid, ps, bbox))
        out.append({"t": fr["t"], "people": people})
    return out


def replay(scored, segments, cfg) -> Metrics:
    """Replay scored frames through the temporal filter, segment by segment."""
    tracker = TemporalRaiseTracker(cfg)
    m = Metrics(fp_by_kind={})

    idx = 0
    n = len(scored)
    for seg in segments:
        start, end = float(seg["start"]), float(seg["end"])
        # Arm exactly as the session does when a roll number is called.
        tracker.arm(start)
        detected = False
        while idx < n and scored[idx]["t"] < start:
            # Keep feeding frames so tracker state stays continuous across the
            # gap, as it would live.
            fr = scored[idx]
            tracker.update(fr["t"], {tid: (ps, bb) for tid, ps, bb in fr["people"]})
            idx += 1
        j = idx
        while j < n and scored[j]["t"] <= end:
            fr = scored[j]
            confirmed = tracker.update(
                fr["t"], {tid: (ps, bb) for tid, ps, bb in fr["people"]}
            )
            if confirmed:
                detected = True
            j += 1

        if seg["expect_raise"]:
            if detected:
                m.tp += 1
            else:
                m.fn += 1
        else:
            if detected:
                m.fp += 1
                kind = seg.get("kind", "?")
                m.fp_by_kind[kind] = m.fp_by_kind.get(kind, 0) + 1
            else:
                m.tn += 1
    return m


def overrides_from(flat: dict) -> dict:
    nested: dict = {}
    for dotted, value in flat.items():
        node = nested
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return nested


def split_params(combo: dict):
    scorer = {k: v for k, v in combo.items() if k.startswith(SCORER_KEYS)}
    temporal = {k: v for k, v in combo.items() if not k.startswith(SCORER_KEYS)}
    return scorer, temporal


def evaluate_grid(base_cfg, frames, segments, grid: dict, exit_ratio: float):
    """Two-level sweep: score once per scorer config, reuse across temporal ones."""
    keys = list(grid)
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))]

    # Bucket combinations by their scorer-affecting half.
    buckets: dict[tuple, list[dict]] = {}
    for c in combos:
        s, _ = split_params(c)
        buckets.setdefault(tuple(sorted(s.items())), []).append(c)

    rows = []
    done = 0
    for skey, group in buckets.items():
        scfg = base_cfg.override(overrides_from(dict(skey)))
        scored = score_frames(frames, scfg)
        for combo in group:
            flat = dict(combo)
            enter = flat.get("temporal.enter_threshold",
                             base_cfg.require("temporal.enter_threshold"))
            flat.setdefault("temporal.exit_threshold", round(enter * exit_ratio, 3))
            cfg = base_cfg.override(overrides_from(flat))
            rows.append((combo, replay(scored, segments, cfg)))
            done += 1
            if done % 25 == 0:
                print(f"    {done}/{len(combos)} combinations", flush=True)
    return rows


def coord_sweep(base_cfg, frames, segments, grid: dict, exit_ratio: float):
    """Vary one parameter at a time around the current defaults."""
    rows = []
    for key, values in grid.items():
        for v in values:
            flat = {key: v}
            enter = flat.get("temporal.enter_threshold",
                             base_cfg.require("temporal.enter_threshold"))
            if key == "temporal.enter_threshold":
                flat["temporal.exit_threshold"] = round(enter * exit_ratio, 3)
            cfg = base_cfg.override(overrides_from(flat))
            scored = score_frames(frames, cfg)
            rows.append(({key: v}, replay(scored, segments, cfg)))
    return rows


def fmt(combo: dict) -> str:
    short = {
        "temporal.enter_threshold": "enter",
        "temporal.exit_threshold": "exit",
        "temporal.hold_sec": "hold",
        "temporal.ema_tau_sec": "tau",
        "raise.straightness_floor": "straight",
        "raise.beside_head_allowance": "beside",
        "model.keypoint_conf": "kpconf",
    }
    return "  ".join(f"{short.get(k, k)}={v}" for k, v in sorted(combo.items()))


def main(argv: list[str] | None = None) -> int:
    cfg = Config.load()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--weights", default=cfg.get("model.weights", "yolo11m-pose.pt"))
    ap.add_argument("--imgsz", type=int, default=int(cfg.get("model.imgsz", 1280)))
    ap.add_argument("--mode", choices=("grid", "coord"), default="grid")
    ap.add_argument("--exit-ratio", type=float, default=0.64,
                    help="exit_threshold = enter_threshold * this")
    ap.add_argument("--target-recall", type=float, default=0.98)
    ap.add_argument("--max-fp-rate", type=float, default=0.01)
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--apply", action="store_true", help="write the winner into config.json")
    args = ap.parse_args(argv)

    labels, data, npz_path = load_recording(args.name, args.weights, args.imgsz)
    segments = labels.get("segments") or []
    if not segments:
        raise SystemExit("the recording has no labelled segments")

    min_h = float(cfg.get("model.min_person_height_px", 90))
    frames, small, untracked = group_by_frame(data, min_h)

    genuine = sum(1 for s in segments if s["expect_raise"])
    print("=" * 78)
    print("THRESHOLD CALIBRATION")
    print("=" * 78)
    print(f"  recording   {args.name}  ({labels.get('duration_sec', 0):.0f}s, "
          f"{labels.get('measured_fps', 0):.1f} FPS, {labels.get('camera', '?')})")
    print(f"  poses       {npz_path.name}")
    print(f"  frames      {len(frames)} with usable detections")
    print(f"  segments    {len(segments)}  ({genuine} genuine raises, "
          f"{len(segments) - genuine} negatives)")
    if small:
        print(f"  skipped     {small} detections below min_person_height_px={min_h:.0f}")
    if untracked:
        print(f"  skipped     {untracked} detections with no track id")

    if not frames:
        print()
        print("  No usable detections. Nothing can be calibrated from this recording -")
        print("  fix camera placement, framing or lighting first.")
        return 1
    if genuine == 0:
        print()
        print("  No genuine-raise segments, so recall cannot be measured.")
        return 1

    # Baseline: how the shipped configuration performs on this footage.
    base_scored = score_frames(frames, cfg)
    base = replay(base_scored, segments, cfg)
    print()
    print(f"  baseline (current config.json):  recall {base.recall:6.1%}   "
          f"precision {base.precision:6.1%}   F1 {base.f1:6.3f}   "
          f"FP {base.fp}/{base.fp + base.tn}")

    grid = DEFAULT_GRID
    print()
    print(f"  sweeping {args.mode}: " + ", ".join(f"{k.split('.')[-1]}x{len(v)}"
                                                  for k, v in grid.items()))
    if args.mode == "grid":
        rows = evaluate_grid(cfg, frames, segments, grid, args.exit_ratio)
    else:
        rows = coord_sweep(cfg, frames, segments, grid, args.exit_ratio)

    # Rank by F1, then recall: among equally balanced settings prefer the one that
    # misses fewer real students, since a false absence is the costlier error.
    rows.sort(key=lambda r: (r[1].f1, r[1].recall), reverse=True)

    print()
    print(f"  {'recall':>7} {'prec':>7} {'F1':>7} {'TP':>4} {'FN':>4} {'FP':>4} {'TN':>4}   settings")
    print("  " + "-" * 74)
    for combo, m in rows[: args.top]:
        print(f"  {m.recall:7.1%} {m.precision:7.1%} {m.f1:7.3f} "
              f"{m.tp:4d} {m.fn:4d} {m.fp:4d} {m.tn:4d}   {fmt(combo)}")

    qualified = [
        (c, m) for c, m in rows
        if m.recall >= args.target_recall and m.fp_rate <= args.max_fp_rate
    ]
    print()
    print("=" * 78)
    if qualified:
        combo, m = qualified[0]
        print(f"  RECOMMENDED  recall {m.recall:.1%}, precision {m.precision:.1%}, "
              f"FP rate {m.fp_rate:.1%}")
        print(f"               {fmt(combo)}")
    else:
        combo, m = rows[0]
        print(f"  NOTHING met recall >= {args.target_recall:.0%} with FP rate "
              f"<= {args.max_fp_rate:.0%}.")
        print(f"  Best available: recall {m.recall:.1%}, precision {m.precision:.1%}")
        print(f"               {fmt(combo)}")
        print("  More footage, better camera placement, or relaxed targets are the")
        print("  honest options here - not a threshold that only looks good.")

    if m.fp_by_kind:
        print()
        print("  false positives by action: " + ", ".join(
            f"{k}x{v}" for k, v in sorted(m.fp_by_kind.items(), key=lambda kv: -kv[1])))

    if len(segments) < 20:
        print()
        print(f"  CAUTION: only {len(segments)} segments. With this few, one flipped")
        print("  decision moves recall by several percent. Record more reps before")
        print("  trusting these numbers (--reps on record.py).")

    if args.apply:
        flat = dict(combo)
        enter = flat.get("temporal.enter_threshold", cfg.require("temporal.enter_threshold"))
        flat.setdefault("temporal.exit_threshold", round(enter * args.exit_ratio, 3))
        path = ROOT / "config.json"
        shutil.copy2(path, path.with_suffix(".json.bak"))
        data_cfg = json.loads(path.read_text(encoding="utf-8"))
        for dotted, value in flat.items():
            node = data_cfg
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = value
        path.write_text(json.dumps(data_cfg, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"  applied to config.json (previous saved as {path.name}.bak)")
        for k, v in sorted(flat.items()):
            print(f"    {k} = {v}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
