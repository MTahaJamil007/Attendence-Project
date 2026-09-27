"""Attendance gating: turning "a hand went up" into "this student is present".

The predecessor counted hands frame-wide (``raised_count``), so *anyone* raising a
hand marked the called student present. One person could answer for the whole
class and the log would look clean. These rules close that hole using the
tracker IDs that come free with ``model.track(persist=True)``:

* **Rule A - single hand.** Two or more confirmed raises during one call is not a
  vote, it is an ambiguity. The system refuses to guess and asks the teacher.
* **Rule B - one credit per track per session.** A person already credited for
  one roll number cannot be credited for another. This is the rule that makes
  proxy attendance structurally impossible rather than merely discouraged.
* **Rule C - seat zone lock (optional).** A student's credited position is
  remembered. If the same roll number is later answered from a very different
  part of the room, that is flagged. Over a few sessions this also accumulates a
  usable seat map.

Every refusal is surfaced with a reason rather than silently downgraded to
Absent. An unexplained absence is what made the predecessor's output impossible
to trust or debug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from raise_detector import TrackState


class Verdict(str, Enum):
    NONE = "none"
    PRESENT = "present"
    AMBIGUOUS = "ambiguous"
    REUSED_TRACK = "reused_track"
    ZONE_MISMATCH = "zone_mismatch"


@dataclass
class Decision:
    verdict: Verdict = Verdict.NONE
    track: TrackState | None = None
    competing: list[TrackState] = field(default_factory=list)
    message: str = ""

    @property
    def is_present(self) -> bool:
        return self.verdict is Verdict.PRESENT

    @property
    def needs_teacher(self) -> bool:
        """True when the system has evidence but refuses to decide alone."""
        return self.verdict in (
            Verdict.AMBIGUOUS,
            Verdict.REUSED_TRACK,
            Verdict.ZONE_MISMATCH,
        )


@dataclass
class SeatZone:
    cx: float
    cy: float
    torso: float

    def distance_in_torsos(self, cx: float, cy: float) -> float:
        ref = self.torso if self.torso > 1e-6 else 1.0
        return ((cx - self.cx) ** 2 + (cy - self.cy) ** 2) ** 0.5 / ref


def _centroid(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


class AttendanceGate:
    def __init__(self, cfg):
        g = cfg.section("gating")
        self.require_single_hand: bool = bool(g["require_single_hand"])
        self.one_credit_per_track: bool = bool(g["one_credit_per_track"])
        self.seat_zone_lock: bool = bool(g["seat_zone_lock"])
        self.zone_tolerance: float = float(g["seat_zone_tolerance_torso"])

        # track_id -> roll number already credited to that person this session
        self.credited: dict[int, str] = {}
        # roll number -> where that student sat when last credited
        self.zones: dict[str, SeatZone] = {}

    # -- session lifecycle -------------------------------------------------
    def reset_session(self) -> None:
        """Clear per-session credits. Seat zones deliberately persist."""
        self.credited.clear()

    # -- seat map persistence ---------------------------------------------
    def load_zones(self, path: Path | str) -> int:
        path = Path(path)
        if not path.exists():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        self.zones = {
            roll: SeatZone(float(v["cx"]), float(v["cy"]), float(v["torso"]))
            for roll, v in data.items()
            if isinstance(v, dict) and {"cx", "cy", "torso"} <= set(v)
        }
        return len(self.zones)

    def save_zones(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            roll: {"cx": z.cx, "cy": z.cy, "torso": z.torso}
            for roll, z in self.zones.items()
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # -- the decision ------------------------------------------------------
    def evaluate(
        self,
        confirmed: list[TrackState],
        roll: str,
        contenders: list[TrackState] | None = None,
    ) -> Decision:
        """Decide what a set of confirmed raises means for the called *roll*.

        Args:
            confirmed: tracks that have held a raise long enough to be credited.
            contenders: tracks whose hand is up at all, used for the single-hand
                rule. Counting only *confirmed* tracks made Rule A a race the
                marginally faster hand always won, because two people never raise
                in the same frame. Defaults to ``confirmed`` when not supplied.
        """
        if not confirmed:
            return Decision(verdict=Verdict.NONE, message="no confirmed raise")

        # One entry per track, confirmed tracks included even if a caller passed a
        # contender list that somehow omits them.
        pool: dict[int, TrackState] = {st.track_id: st for st in (contenders or [])}
        for st in confirmed:
            pool[st.track_id] = st
        confirmed_ids = {st.track_id for st in confirmed}

        # Rule B - split into those still eligible for credit and those belonging
        # to someone already marked.
        blocked: list[TrackState] = []
        eligible: list[TrackState] = []
        for st in pool.values():
            prior = self.credited.get(st.track_id) if self.one_credit_per_track else None
            if prior is not None and prior != roll:
                blocked.append(st)
            else:
                eligible.append(st)

        # Only a confirmed track can actually be credited; a contender that is
        # merely latched is evidence of ambiguity, not a candidate.
        creditable = [st for st in eligible if st.track_id in confirmed_ids]

        if not creditable:
            if blocked:
                others = ", ".join(sorted({self.credited[st.track_id] for st in blocked}))
                return Decision(
                    verdict=Verdict.REUSED_TRACK,
                    competing=blocked,
                    message=f"the only raised hand is already marked for {others}",
                )
            return Decision(verdict=Verdict.NONE, message="no creditable raise")

        # Rule A - more than one eligible hand is an ambiguity, not a vote.
        if self.require_single_hand and len(eligible) > 1:
            ids = ", ".join(f"#{st.track_id}" for st in sorted(eligible, key=lambda s: s.track_id))
            return Decision(
                verdict=Verdict.AMBIGUOUS,
                competing=eligible,
                message=f"{len(eligible)} hands raised ({ids}) - confirm manually",
            )

        candidate = max(creditable, key=lambda st: st.peak_ema)

        # Rule C - optional seat consistency check.
        if self.seat_zone_lock:
            zone = self.zones.get(roll)
            bbox = candidate.best_bbox or candidate.bbox
            if zone is not None and bbox is not None:
                cx, cy = _centroid(bbox)
                gap = zone.distance_in_torsos(cx, cy)
                if gap > self.zone_tolerance:
                    return Decision(
                        verdict=Verdict.ZONE_MISMATCH,
                        track=candidate,
                        competing=eligible,
                        message=(
                            f"raise came from {gap:.1f} torso-widths away from where "
                            f"{roll} last answered - confirm manually"
                        ),
                    )

        return Decision(
            verdict=Verdict.PRESENT,
            track=candidate,
            competing=creditable,
            message=f"track #{candidate.track_id} raised (score {candidate.peak_ema:.2f})",
        )

    # -- committing --------------------------------------------------------
    def commit(self, roll: str, track: TrackState | None) -> None:
        """Record that *roll* was credited, optionally to a specific track.

        Called for automatic marks and for teacher overrides alike, so that a
        manually confirmed student still consumes their track's one credit.
        """
        if track is None:
            return
        if self.one_credit_per_track:
            self.credited[track.track_id] = roll
        bbox = track.best_bbox or track.bbox
        if bbox is not None:
            cx, cy = _centroid(bbox)
            torso = 0.0
            if track.best_score is not None:
                torso = track.best_score.torso_px
            elif track.last_score is not None:
                torso = track.last_score.torso_px
            self.zones[roll] = SeatZone(cx, cy, torso)

    def release(self, roll: str) -> None:
        """Undo a credit, for the teacher's Undo action."""
        for tid, r in list(self.credited.items()):
            if r == roll:
                del self.credited[tid]
