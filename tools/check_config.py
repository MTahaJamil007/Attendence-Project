"""Verify code and config.json agree. Needs no torch, camera or GPU.

Two directions, both of which have already bitten this project once:

* a key the code reads but config.json does not define (``require_fresh_raise``
  was referenced before it existed, and only a hand-written test caught it);
* a key config.json defines that no code ever reads - a silent typo, where the
  operator tunes a value and nothing changes.

The AST pass finds dotted ``cfg.get`` / ``cfg.require`` / ``cfg.section`` lookups.
Because sections are then indexed with plain ``["..."]`` subscripts that static
analysis cannot follow reliably, the second pass simply constructs every
config-consuming class for real, which turns any missing subscript into a
KeyError here rather than mid-session.

Run:  .venv\\Scripts\\python.exe tools\\check_config.py
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

CONFIG_METHODS = {"get", "require", "section"}
# Keys that exist for humans, not code.
IGNORE_PREFIXES = ("_comment",)


CONFIG_NAMES = {"cfg", "config", "CFG", "conf_cfg"}


def _receiver_is_config(fn: ast.Attribute, config_vars: set[str]) -> bool:
    """Is this ``.get`` being called on a config object rather than any old dict?

    Needed because plain ``dict.get("key")`` is everywhere - ``status.get(...)``,
    ultralytics' ``speed.get(...)`` - and counting those produced four phantom
    "missing key" reports. A checker that cries wolf gets ignored, which is worse
    than having no checker.
    """
    recv = fn.value
    if isinstance(recv, ast.Name):
        return recv.id in config_vars
    if isinstance(recv, ast.Attribute):
        # self.cfg.get(...) / obj.config.get(...)
        return recv.attr in CONFIG_NAMES
    if isinstance(recv, ast.Call):
        # Config.load(...).get(...) / cfg.override(...).get(...)
        inner = recv.func
        return isinstance(inner, ast.Attribute) and inner.attr in {"load", "override", "section"}
    return False


def collect_dotted_keys(paths: list[Path]) -> dict[str, list[str]]:
    """Map dotted config key -> source locations that read it."""
    found: dict[str, list[str]] = {}
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            print(f"  SYNTAX ERROR in {path.name}: {exc}")
            continue

        # Pass 1: which local names hold a Config or a config section?
        config_vars = set(CONFIG_NAMES)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute):
                continue
            if value.func.attr not in {"section", "load", "override"}:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    config_vars.add(t.id)

        # Pass 2: collect the key literals.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not isinstance(fn, ast.Attribute) or fn.attr not in CONFIG_METHODS:
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant):
                continue
            key = node.args[0].value
            if not isinstance(key, str):
                continue
            # `require` and `section` exist only on Config, so they always count.
            # `get` is ambiguous and needs the receiver check.
            if fn.attr == "get" and not _receiver_is_config(fn, config_vars):
                continue
            found.setdefault(key, []).append(f"{path.name}:{node.lineno}")
    return found


def flatten(data: dict, prefix: str = "") -> set[str]:
    out: set[str] = set()
    for k, v in data.items():
        if k.startswith(IGNORE_PREFIXES):
            continue
        dotted = f"{prefix}{k}"
        out.add(dotted)
        if isinstance(v, dict):
            out |= flatten(v, dotted + ".")
    return out


def main() -> int:
    cfg_path = ROOT / "config.json"
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    defined = flatten(data)

    sources = sorted((ROOT / "src").glob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    referenced = collect_dotted_keys(sources)

    print("=" * 72)
    print("CONFIG CONTRACT CHECK")
    print("=" * 72)
    print(f"  config.json keys: {len(defined)}")
    print(f"  dotted lookups in code: {len(referenced)}")
    print()

    failures = 0

    # 1. read but not defined.
    #
    # A bare key (no dot) is almost always dict.get on an already-extracted
    # section - `cam.get("buffer_size", 1)` after `cam = cfg.section("camera")`.
    # The AST pass matches on method name alone and cannot distinguish that from
    # Config.get, so resolve bare keys against the leaf name under any section.
    # Genuine typos still fail, because a typo matches no leaf anywhere.
    leaf_names = {k.rsplit(".", 1)[-1] for k in defined}

    def resolves(key: str) -> bool:
        if key in defined:
            return True
        return "." not in key and key in leaf_names

    missing = sorted(k for k in referenced if not resolves(k))
    bare = sorted(k for k in referenced if k not in defined and resolves(k))
    if missing:
        print("  MISSING from config.json (code reads these):")
        for k in missing:
            print(f"    {k:<44} <- {', '.join(referenced[k])}")
        failures += len(missing)
    else:
        print("  OK  every config lookup resolves")
    if bare:
        print("  (resolved as section-relative keys: " + ", ".join(bare) + ")")

    # 2. defined but never read. Only leaves matter: a section is "read" via
    #    section() or by its children, so reporting containers would be noise.
    leaves = {k for k in defined if not any(o.startswith(k + ".") for o in defined)}
    read_prefixes = set(referenced)
    orphans = sorted(
        k
        for k in leaves
        if k not in read_prefixes
        and not any(k.startswith(p + ".") for p in read_prefixes)
    )
    print()
    if orphans:
        print("  NOT READ via a dotted lookup (expected for section()-indexed keys):")
        for k in orphans:
            print(f"    {k}")
    else:
        print("  OK  no unreferenced leaf keys")

    # 3. construct the real consumers - catches ["..."] subscripts the AST pass
    #    cannot follow.
    print()
    print("  constructing config consumers:")
    from config import Config

    cfg = Config.load(cfg_path)
    consumers = []
    try:
        from raise_detector import RaiseScorer, TemporalRaiseTracker

        consumers += [("RaiseScorer", RaiseScorer), ("TemporalRaiseTracker", TemporalRaiseTracker)]
    except Exception as exc:
        print(f"    FAIL import raise_detector: {exc}")
        failures += 1
    try:
        from gating import AttendanceGate

        consumers.append(("AttendanceGate", AttendanceGate))
    except Exception as exc:
        print(f"    FAIL import gating: {exc}")
        failures += 1

    for name, klass in consumers:
        try:
            klass(cfg)
            print(f"    OK   {name}")
        except Exception as exc:
            print(f"    FAIL {name}: {type(exc).__name__}: {exc}")
            failures += 1

    # VideoWorker needs cv2/numpy but not a camera or CUDA, so attempt it and
    # report an unmet import as a skip rather than a failure.
    try:
        from video_worker import VideoWorker

        VideoWorker(cfg, runtime_report=None)
        print("    OK   VideoWorker")
    except ImportError as exc:
        print(f"    SKIP VideoWorker (dependency not installed yet: {exc.name})")
    except Exception as exc:
        print(f"    FAIL VideoWorker: {type(exc).__name__}: {exc}")
        failures += 1

    # 4. internal consistency of the tuned values
    print()
    print("  sanity checks on values:")
    checks: list[tuple[str, bool, str]] = []
    t = cfg.section("temporal")
    r = cfg.section("raise")
    s = cfg.section("session")
    checks.append(
        ("exit_threshold < enter_threshold (hysteresis)", t["exit_threshold"] < t["enter_threshold"],
         f"{t['exit_threshold']} < {t['enter_threshold']}")
    )
    checks.append(
        ("hold_sec < response_window_sec", t["hold_sec"] < s["response_window_sec"],
         f"{t['hold_sec']} < {s['response_window_sec']}")
    )
    checks.append(
        ("track_grace_sec < track_expire_sec", t["track_grace_sec"] < t["track_expire_sec"],
         f"{t['track_grace_sec']} < {t['track_expire_sec']}")
    )
    wsum = sum(r["weights"].values())
    checks.append(("raise weights sum to 1.0", abs(wsum - 1.0) < 1e-6, f"{wsum}"))
    checks.append(
        ("every weight has a matching ramp",
         set(r["weights"]) <= set(r["ramps"]),
         f"missing ramps: {sorted(set(r['weights']) - set(r['ramps'])) or 'none'}")
    )
    checks.append(
        ("arm_straightness has a ramp but no weight (it is a multiplier)",
         "arm_straightness" in r["ramps"] and "arm_straightness" not in r["weights"], "")
    )
    checks.append(
        ("lateral_penalty_start < lateral_penalty_end",
         r["lateral_penalty_start"] < r["lateral_penalty_end"],
         f"{r['lateral_penalty_start']} < {r['lateral_penalty_end']}")
    )
    checks.append(
        ("0 < straightness_floor < 1", 0.0 < r["straightness_floor"] < 1.0, f"{r['straightness_floor']}")
    )
    checks.append(
        ("keypoint_conf_floor < keypoint_conf",
         cfg.require("model.keypoint_conf_floor") < cfg.require("model.keypoint_conf"),
         f"{cfg.require('model.keypoint_conf_floor')} < {cfg.require('model.keypoint_conf')}")
    )
    checks.append(
        ("stall_threshold_sec < response_window_sec",
         s["stall_threshold_sec"] < s["response_window_sec"], "")
    )
    checks.append(
        ("imgsz is a multiple of 32", int(cfg.require("model.imgsz")) % 32 == 0,
         str(cfg.require("model.imgsz")))
    )

    for name, ok, detail in checks:
        tag = "OK  " if ok else "FAIL"
        suffix = f"   [{detail}]" if detail else ""
        print(f"    {tag} {name}{suffix}")
        if not ok:
            failures += 1

    print()
    print("=" * 72)
    if failures:
        print(f"  {failures} problem(s) found")
        return 1
    print("  config contract OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
