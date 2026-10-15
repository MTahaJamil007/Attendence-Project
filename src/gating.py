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
* **Rule D - seat memory: one credit per seat per session.** Rule B is keyed on
  the tracker's ID, and tracker IDs are not permanent: a student hidden for a
  moment by someone walking past, or who leans out of view, comes back with a new
  ID. Rule B then sees a stranger, so the student who answered for roll 9 could
  answer for roll 31 a few calls later and be credited for both. Rule D remembers
  *where* every credited student sat - the shoulder midpoint, which does not move
  when an arm goes up - and flags a raise from a seat that already answered for
  someone else, whatever ID the tracker has given it.

  Distances are measured in shoulder widths, the natural seat unit: two people
  side by side cannot have shoulder midpoints much closer than one shoulder width
  without overlapping, while the same person shifting in their seat stays well
  inside half of one. A clothing-colour match (appearance.py - no face is used)
  may widen the radius a little for someone who leaned while raising, never as
  far as a neighbour could be. A seat at a clearly different depth (a student
  directly behind another) is told apart by its apparent size.

Every refusal is surfaced with a reason rather than silently downgraded to
Absent. An unexplained absence is what made the predecessor's output impossible
to trust or debug.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from appearance import similarity as appearance_similarity
from raise_detector import TrackState


class Verdict(str, Enum):
    NONE = "none"
    PRESENT = "present"
    AMBIGUOUS = "ambiguous"
    REUSED_TRACK = "reused_track"
    REUSED_SEAT = "reused_seat"
    ZONE_MISMATCH = "zone_mismatch"


@dataclass
class SeatRecord:
    """Where a student credited this session was sitting when credited."""

    roll: str
    x: float
    y: float
    scale: float               # one shoulder width in px (geometry.seat_anchor)
    track_id: int | None = None
    signature: object | None = None
    marked_at: float = 0.0     # monotonic time of the credit


@dataclass
class SeatMatch:
    """A raised hand at (or very near) a seat already credited to ``record.roll``."""

    track_id: int
    record: SeatRecord
    distance: float            # in shoulder widths
    appearance: float | None   # clothing-colour similarity, None if not measured

    def describe(self) -> str:
        bits = [f"{self.distance:.2f} shoulder-widths away"]
        if self.appearance is not None:
            bits.append(f"clothing match {self.appearance:.0%}")
        return ", ".join(bits)


@dataclass
class Decision:
    verdict: Verdict = Verdict.NONE
    track: TrackState | None = None
    competing: list[TrackState] = field(default_factory=list)
    message: str = ""
    # Rule D: every raised hand found at a seat already credited this session.
    seat_matches: dict[int, SeatMatch] = field(default_factory=dict)

    @property
    def is_present(self) -> bool:
        return self.verdict is Verdict.PRESENT

    @property
    def needs_teacher(self) -> bool:
        """True when the system has evidence but refuses to decide alone."""
        return self.verdict in (
            Verdict.AMBIGUOUS,
            Verdict.REUSED_TRACK,
            Verdict.REUSED_SEAT,
            Verdict.ZONE_MISMATCH,
        )

    @property
    def seat(self) -> SeatMatch | None:
        """The seat match for this decision's own track, if there is one."""
        if self.track is None:
            return None
        return self.seat_matches.get(self.track.track_id)


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


def _position(track: TrackState) -> tuple[float, float] | None:
    """The seat anchor when known; otherwise the box centre.

    The box centre is a poor position - a raised arm grows the box upward and
    drags its centre with it - so it is only a fallback for a track whose
    shoulders were never seen.
    """
    if track.anchor is not None:
        return track.anchor[0], track.anchor[1]
    bbox = track.best_bbox or track.bbox
    return _centroid(bbox) if bbox is not None else None


class AttendanceGate:
    def __init__(self, cfg):
        g = cfg.section("gating")
        self.require_single_hand: bool = bool(g["require_single_hand"])
        self.one_credit_per_track: bool = bool(g["one_credit_per_track"])
        self.seat_zone_lock: bool = bool(g["seat_zone_lock"])
        self.zone_tolerance: float = float(g["seat_zone_tolerance_torso"])

        self.seat_memory: bool = bool(g["seat_memory"])
        self.seat_radius: float = float(g["seat_radius_shoulders"])
        self.seat_radius_appearance: float = float(g["seat_appearance_radius_shoulders"])
        self.seat_min_similarity: float = float(g["seat_appearance_min_similarity"])
        self.seat_scale_ratio_max: float = float(g["seat_scale_ratio_max"])

        # track_id -> roll number already credited to that person this session
        self.credited: dict[int, str] = {}
        # roll number -> where that student sat when last credited (Rule C, persists)
        self.zones: dict[str, SeatZone] = {}
        # roll number -> where that student sat when credited this session (Rule D)
        self.seats: dict[str, SeatRecord] = {}

    # -- session lifecycle -------------------------------------------------
    def reset_session(self) -> None:
        """Clear per-session credits and seats. Rule C's zones deliberately persist."""
        self.credited.clear()
        self.seats.clear()

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

    # -- Rule D ------------------------------------------------------------
    def match_seat(self, track: TrackState, roll: str) -> SeatMatch | None:
        """Is *track* sitting where somebody other than *roll* already answered?

        Returns the closest such seat, or None. A seat is the same seat when the
        anchors are within ``seat_radius_shoulders``, or within the wider
        ``seat_appearance_radius_shoulders`` when the clothing also matches.
        """
        if not self.seat_memory or track.anchor is None or not self.seats:
            return None
        x, y, scale = track.anchor
        best: SeatMatch | None = None
        for rec in self.seats.values():
            if rec.roll == roll:
                continue  # the same roll called again (after an Undo) is not reuse
            lo, hi = sorted((max(scale, 1e-6), max(rec.scale, 1e-6)))
            if hi / lo > self.seat_scale_ratio_max:
                continue  # different apparent size: a different row
            unit = (scale + rec.scale) * 0.5
            d = ((x - rec.x) ** 2 + (y - rec.y) ** 2) ** 0.5 / max(unit, 1e-6)
            if d > max(self.seat_radius, self.seat_radius_appearance):
                continue
            sim = appearance_similarity(track.signature, rec.signature)
            same = d <= self.seat_radius or (
                sim is not None and sim >= self.seat_min_similarity
            )
            if same and (best is None or d < best.distance):
                best = SeatMatch(track_id=track.track_id, record=rec,
                                 distance=d, appearance=sim)
        return best

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

        # Rule D - hands at a seat that already answered for someone else. They
        # stay in `eligible`, so they still count as a competing hand for Rule A,
        # but they are never credited automatically: position is strong evidence,
        # not proof, so the teacher makes the call.
        seat_matches: dict[int, SeatMatch] = {}
        for st in eligible:
            m = self.match_seat(st, roll)
            if m is not None:
                seat_matches[st.track_id] = m

        # Only a confirmed track can actually be credited; a contender that is
        # merely latched is evidence of ambiguity, not a candidate.
        creditable = [
            st for st in eligible
            if st.track_id in confirmed_ids and st.track_id not in seat_matches
        ]

        if not creditable:
            reused = [st for st in eligible
                      if st.track_id in confirmed_ids and st.track_id in seat_matches]
            if reused:
                st = min(reused, key=lambda t: seat_matches[t.track_id].distance)
                m = seat_matches[st.track_id]
                return Decision(
                    verdict=Verdict.REUSED_SEAT,
                    track=st,
                    competing=eligible,
                    seat_matches=seat_matches,
                    message=(
                        f"hand #{st.track_id} is at the seat already marked for "
                        f"{m.record.roll} ({m.describe()}) - confirm manually"
                    ),
                )
            if blocked:
                others = ", ".join(sorted({self.credited[st.track_id] for st in blocked}))
                return Decision(
                    verdict=Verdict.REUSED_TRACK,
                    track=blocked[0],
                    competing=blocked,
                    message=f"the only raised hand is already marked for {others}",
                )
            return Decision(verdict=Verdict.NONE, message="no creditable raise")

        # Rule A - more than one eligible hand is an ambiguity, not a vote.
        if self.require_single_hand and len(eligible) > 1:
            def tag(st: TrackState) -> str:
                m = seat_matches.get(st.track_id)
                return f"#{st.track_id}" + (f" at {m.record.roll}'s seat" if m else "")

            ids = ", ".join(tag(st) for st in sorted(eligible, key=lambda s: s.track_id))
            return Decision(
                verdict=Verdict.AMBIGUOUS,
                competing=eligible,
                seat_matches=seat_matches,
                message=f"{len(eligible)} hands raised ({ids}) - confirm manually",
            )

        candidate = max(creditable, key=lambda st: st.peak_ema)

        # Rule C - optional seat consistency check.
        if self.seat_zone_lock:
            zone = self.zones.get(roll)
            pos = _position(candidate)
            if zone is not None and pos is not None:
                gap = zone.distance_in_torsos(*pos)
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
    def commit(self, roll: str, track: TrackState | None, now: float = 0.0) -> None:
        """Record that *roll* was credited, optionally to a specific track.

        Called for automatic marks and for teacher overrides alike, so that a
        manually confirmed student still consumes their track's one credit and
        their seat. A track keeps the *first* roll it was credited for: a teacher
        overriding a flag to mark a second roll from the same person must not
        erase the record that this person already answered once.
        """
        if track is None:
            return
        if self.one_credit_per_track:
            self.credited.setdefault(track.track_id, roll)
        if track.anchor is not None:
            ax, ay, scale = track.anchor
            self.seats[roll] = SeatRecord(
                roll=roll, x=ax, y=ay, scale=scale, track_id=track.track_id,
                signature=track.signature, marked_at=now,
            )
        pos = _position(track)
        if pos is not None:
            cx, cy = pos
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
        self.seats.pop(roll, None)
