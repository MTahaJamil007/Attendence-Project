"""Hand-raise scoring and temporal confirmation.

This module replaces the two mechanisms that made the predecessor inaccurate.

**1. Boolean AND over raw thresholds -> a graded score.**
The old rule was ``above_shoulder and wrist_above_elbow and elbow_not_low and
hand_high and plausible``. Every term was a hard threshold, so a subject sitting
near any one of them flickered on and off with pixel-level keypoint noise, and a
single failing term discarded all other evidence. Here each term becomes a ramp
in [0, 1] and they are combined as a weighted mean, so evidence accumulates
instead of being vetoed. Only one physical necessity stays hard: a raised hand
has its wrist above its shoulder.

**2. A frame-counting hold chain -> a time-constant filter.**
The old rule needed 0.85 s of continuous detection while tolerating gaps of only
0.28 s. At the 3-8 FPS its CPU-only torch build actually delivered, one dropped
frame produced a 0.4 s gap, broke the chain, and logged a genuine raise as
"Timeout: no valid hand". Here smoothing uses ``alpha = 1 - exp(-dt / tau)``,
which makes the filter's response identical at 10 FPS or 40 FPS, brief detection
misses are absorbed by a time-based grace period, and a Schmitt trigger
(separate enter/exit thresholds) removes threshold chatter entirely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from geometry import (
    NOSE,
    SIDE_JOINTS,
    angle_at_deg,
    ramp,
    torso_length,
)

Keypoints = list[tuple[float, float, float]]  # 17 x (x, y, conf)


# --------------------------------------------------------------------------
# Frame-level scoring (stateless)
# --------------------------------------------------------------------------

@dataclass
class SideScore:
    """Scored evidence for one arm, with the full feature breakdown kept.

    The breakdown is not decoration: tools/calibrate.py sweeps thresholds against
    recorded footage and needs per-feature values to explain why a frame scored
    the way it did.
    """

    side: str
    score: float = 0.0
    reason: str = ""
    features: dict[str, float] = field(default_factory=dict)
    raw: dict[str, float] = field(default_factory=dict)


@dataclass
class PersonScore:
    score: float = 0.0
    side: str = ""
    reason: str = ""
    torso_px: float = 0.0
    left: SideScore | None = None
    right: SideScore | None = None

    @property
    def best(self) -> SideScore | None:
        if self.side == "left":
            return self.left
        if self.side == "right":
            return self.right
        return None


class RaiseScorer:
    """Stateless per-frame scorer. One instance per session; no mutable state."""

    def __init__(self, cfg):
        raise_cfg = cfg.section("raise")
        self.ramps: dict[str, list[float]] = raise_cfg["ramps"]
        self.weights: dict[str, float] = raise_cfg["weights"]
        self.lateral_start: float = raise_cfg["lateral_penalty_start"]
        self.lateral_end: float = raise_cfg["lateral_penalty_end"]
        self.straightness_floor: float = raise_cfg["straightness_floor"]
        self.min_torso_px: float = raise_cfg["min_torso_px"]

        model_cfg = cfg.section("model")
        self.kp_conf: float = model_cfg["keypoint_conf"]
        self.kp_floor: float = model_cfg["keypoint_conf_floor"]

    # -- per-side ----------------------------------------------------------
    def _score_side(
        self, kp: Keypoints, side: str, torso: float
    ) -> SideScore:
        out = SideScore(side=side)
        s_i, e_i, w_i, _h_i = SIDE_JOINTS[side]
        sx, sy, sc = kp[s_i]
        ex, ey, ec = kp[e_i]
        wx, wy, wc = kp[w_i]

        # Shoulder, elbow and wrist are all required to say anything at all.
        # Below the floor the coordinates are noise, and scoring noise is worse
        # than abstaining.
        worst = min(sc, ec, wc)
        if worst < self.kp_floor:
            out.reason = "arm keypoints below confidence floor"
            return out

        # Hard physical gate. Everything else is graded, but a hand raised for
        # attendance has its wrist above its shoulder - no weighting of other
        # evidence should overturn that.
        lift = (sy - wy) / torso
        if lift <= 0.0:
            out.reason = "wrist not above shoulder"
            out.raw["wrist_lift"] = lift
            return out

        raw: dict[str, float] = {}
        feats: dict[str, float] = {}

        # Primary signal: how high the wrist sits above the shoulder.
        raw["wrist_lift"] = lift
        feats["wrist_lift"] = ramp(lift, *self.ramps["wrist_lift"])

        # A raised forearm points up. Separates a real raise from an arm draped
        # on a desk or a head propped on a hand.
        wae = (ey - wy) / torso
        raw["wrist_above_elbow"] = wae
        feats["wrist_above_elbow"] = ramp(wae, *self.ramps["wrist_above_elbow"])

        # The elbow should not be hanging at hip level; that is a flicked wrist,
        # not a raised hand.
        elbow_lift = (sy - ey) / torso
        raw["elbow_lift"] = elbow_lift
        feats["elbow_lift"] = ramp(elbow_lift, *self.ramps["elbow_lift"])

        # Wrist above the head is strong confirmation. The nose is often missing
        # for students facing away, so this feature drops out rather than being
        # faked from a proxy - the weights renormalize below.
        nx, ny, nc = kp[NOSE]
        if nc >= self.kp_conf:
            vext = (ny - wy) / torso
            raw["vertical_extension"] = vext
            feats["vertical_extension"] = ramp(vext, *self.ramps["vertical_extension"])

        # Weighted mean over whichever features were available. Renormalizing by
        # the weights actually present is what lets a partially occluded student
        # still be scored fairly instead of penalized for a missing joint.
        total_w = sum(self.weights[k] for k in feats)
        if total_w <= 0.0:
            out.reason = "no usable features"
            return out
        score = sum(feats[k] * self.weights[k] for k in feats) / total_w

        # Elbow extension is applied as a MULTIPLIER, not as another weighted
        # term, because it is the decisive distractor discriminator and averaging
        # dilutes it. Measured on synthetic poses: as a 0.14-weight term, a
        # head-scratch scored 0.61 - a false positive above the 0.55 enter
        # threshold, and 0.61 again with the nose occluded. As a multiplier with
        # a 0.30 floor the same pose scores 0.20 while a legitimate forearm-up
        # raise (elbow ~96 deg) still scores 0.73.
        #
        # The ramp runs 50->110 deg rather than 70->150 deg on purpose: a
        # forearm-up raise sits near 90 deg and must pass, while a hand tucked by
        # the head sits near 45 deg and must not. A floor above zero keeps this
        # graded rather than a veto.
        angle = angle_at_deg(ex, ey, sx, sy, wx, wy)
        raw["arm_straightness"] = angle
        straightness = self.straightness_floor + (1.0 - self.straightness_floor) * ramp(
            angle, *self.ramps["arm_straightness"]
        )
        # Kept in `raw`, not `feats`: `feats` holds only terms that carry an entry
        # in self.weights, and anything recomputing total_w from it would KeyError.
        raw["straightness_factor"] = straightness
        score *= straightness

        # Penalty for an arm flung far out sideways: stretching and waving look
        # like a raise vertically but sit far from the shoulder horizontally.
        lateral = abs(wx - sx) / torso
        raw["lateral"] = lateral
        penalty = 1.0 - ramp(lateral, self.lateral_start, self.lateral_end)
        score *= penalty

        # Scale by keypoint confidence so a shaky skeleton scores lower than a
        # crisp one, rather than both being treated as equally trustworthy.
        conf_factor = ramp(worst, self.kp_floor, self.kp_conf) * 0.4 + 0.6
        score *= min(1.0, conf_factor)

        out.score = max(0.0, min(1.0, score))
        out.features = feats
        out.raw = raw
        out.reason = "ok"
        return out

    # -- per-person --------------------------------------------------------
    def score(self, kp: Keypoints, bbox_height: float) -> PersonScore:
        """Score one person's keypoints. Returns the stronger of the two arms."""
        out = PersonScore()
        torso = torso_length(kp, bbox_height, self.kp_conf, self.min_torso_px)
        out.torso_px = torso
        if torso <= 0.0:
            out.reason = "subject too small to judge"
            return out

        out.left = self._score_side(kp, "left", torso)
        out.right = self._score_side(kp, "right", torso)

        if out.right.score >= out.left.score:
            out.score, out.side, out.reason = out.right.score, "right", out.right.reason
        else:
            out.score, out.side, out.reason = out.left.score, "left", out.left.reason
        return out


# --------------------------------------------------------------------------
# Temporal confirmation (stateful, per track)
# --------------------------------------------------------------------------

@dataclass
class TrackState:
    track_id: int
    ema: float = 0.0
    latched: bool = False
    latched_at: float | None = None
    confirmed: bool = False
    confirmed_at: float | None = None

    peak_ema: float = 0.0
    last_update: float = 0.0
    last_seen: float = 0.0

    last_score: PersonScore | None = None
    best_score: PersonScore | None = None
    bbox: tuple[float, float, float, float] | None = None
    best_bbox: tuple[float, float, float, float] | None = None

    def held_for(self, now: float) -> float:
        if not self.latched or self.latched_at is None:
            return 0.0
        return max(0.0, now - self.latched_at)


class TemporalRaiseTracker:
    """EMA + Schmitt trigger + wall-clock hold, keyed by tracker ID.

    Why a time constant instead of a frame count: with ``alpha = 1 - exp(-dt/tau)``
    the filter reaches the same fraction of a step input after the same number of
    *seconds* regardless of frame rate. The predecessor's fixed frame budget made
    its behaviour a function of GPU speed, which is why its results swung between
    0/10 and 7/20 across runs on identical footage.
    """

    def __init__(self, cfg):
        t = cfg.section("temporal")
        self.tau: float = float(t["ema_tau_sec"])
        self.enter: float = float(t["enter_threshold"])
        self.exit: float = float(t["exit_threshold"])
        self.hold: float = float(t["hold_sec"])
        self.grace: float = float(t["track_grace_sec"])
        self.expire: float = float(t["track_expire_sec"])
        self.require_fresh: bool = bool(t.get("require_fresh_raise", True))
        self.fresh_lookback: float = float(t.get("fresh_lookback_sec", 1.5))

        self.states: dict[int, TrackState] = {}
        self._armed_at: float = 0.0

    # -- call lifecycle ----------------------------------------------------
    def _is_fresh(self, st: TrackState) -> bool:
        """Did this track's current latch begin close enough to the open window?

        A strict "strictly after arming" rule would punish the most cooperative
        students: someone who raises the instant they hear their number can latch
        a fraction of a second before the response window formally opens.
        ``fresh_lookback_sec`` admits that, while a hand parked in the air across
        several calls latched many seconds earlier and is still excluded.
        """
        if not self.require_fresh:
            return True
        if st.latched_at is None:
            return False
        return st.latched_at >= self._armed_at - self.fresh_lookback

    def arm(self, now: float) -> None:
        """Open a new response window.

        Also retires confirmations carried over from the previous call. Without
        this, a track that was confirmed once stayed confirmed for the rest of the
        session - the ``not st.confirmed`` guard in update() meant it never
        re-entered the freshness check - so one raised hand answered every
        subsequent roll number.
        """
        self._armed_at = now
        if not self.require_fresh:
            return
        for st in self.states.values():
            if st.confirmed and not self._is_fresh(st):
                st.confirmed = False
                st.confirmed_at = None

    def reset(self) -> None:
        self.states.clear()
        self._armed_at = 0.0

    # -- per-frame ---------------------------------------------------------
    def _alpha(self, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        if self.tau <= 1e-6:
            return 1.0
        return 1.0 - math.exp(-dt / self.tau)

    def update(
        self,
        now: float,
        observations: dict[int, tuple[PersonScore, tuple[float, float, float, float]]],
    ) -> list[TrackState]:
        """Fold one frame of per-track scores into the temporal state.

        Args:
            now: monotonic timestamp of the frame.
            observations: track_id -> (person score, bbox xyxy). Tracks absent
                from this mapping were not detected on this frame.

        Returns:
            Every track currently in the confirmed state.
        """
        # Seen tracks: pull the EMA toward the new observation.
        for tid, (ps, bbox) in observations.items():
            st = self.states.get(tid)
            if st is None:
                st = TrackState(track_id=tid, last_update=now, last_seen=now)
                self.states[tid] = st
            a = self._alpha(now - st.last_update)
            st.ema += a * (ps.score - st.ema)
            st.last_update = now
            st.last_seen = now
            st.last_score = ps
            st.bbox = bbox
            if st.ema > st.peak_ema:
                st.peak_ema = st.ema
                st.best_score = ps
                st.best_bbox = bbox

        # Unseen tracks: hold briefly, then decay. Freezing during the grace
        # window is what makes a one- or two-frame detection miss harmless -
        # the failure mode that dominated the predecessor's false absences.
        for tid, st in self.states.items():
            if tid in observations:
                continue
            unseen = now - st.last_seen
            if unseen <= self.grace:
                st.last_update = now
                continue
            a = self._alpha(now - st.last_update)
            st.ema += a * (0.0 - st.ema)
            st.last_update = now

        # Schmitt trigger + hold, applied to every live track.
        for st in self.states.values():
            if st.latched:
                if st.ema <= self.exit:
                    st.latched = False
                    st.latched_at = None
                    st.confirmed = False
                    st.confirmed_at = None
            else:
                if st.ema >= self.enter:
                    st.latched = True
                    st.latched_at = now
            if st.latched and not st.confirmed:
                if st.held_for(now) >= self.hold and self._is_fresh(st):
                    st.confirmed = True
                    st.confirmed_at = now

        # Drop long-gone tracks so the dict cannot grow without bound.
        stale = [tid for tid, st in self.states.items() if now - st.last_seen > self.expire]
        for tid in stale:
            del self.states[tid]

        return [st for st in self.states.values() if st.confirmed]

    # -- queries -----------------------------------------------------------
    def confirmed(self) -> list[TrackState]:
        return [st for st in self.states.values() if st.confirmed]

    def contenders(self) -> list[TrackState]:
        """Every track whose hand is currently up, confirmed or not yet.

        Needed for the single-hand rule. Two people never raise in the same video
        frame: measured on a synthetic two-hand case, one track latched 0.05 s
        before the other and therefore *confirmed* 0.05 s earlier too. Counting
        only confirmed tracks made the rule a race that the marginally faster hand
        always won, so a second raised hand was silently ignored. Counting latched
        tracks catches the competing hand at the moment of decision, with no added
        latency - it is already latched by then.
        """
        out = [st for st in self.states.values() if st.latched]
        if self.require_fresh:
            out = [st for st in out if self._is_fresh(st)]
        return out

    def best_partial(self) -> float:
        """Highest smoothed score among unconfirmed tracks.

        Drives the adaptive window extension: if someone is visibly on the way up
        as the deadline arrives, the session grants extra time instead of logging
        an absence.
        """
        vals = [st.ema for st in self.states.values() if not st.confirmed]
        return max(vals) if vals else 0.0
