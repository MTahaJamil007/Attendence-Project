"""COCO-17 keypoint constants and scale-free geometric primitives.

All image coordinates follow OpenCV convention: x grows right, y grows DOWN.
So "the wrist is above the shoulder" means ``wrist_y < shoulder_y``, and a
positive "lift" is computed as ``shoulder_y - wrist_y``.

Every measurement produced here is normalized by the subject's torso length so
that a student in the back row and a student in the front row yield comparable
numbers. That normalization is what lets one threshold work across the room.
"""

from __future__ import annotations

import math

# --- COCO-17 keypoint indices (Ultralytics pose output order) --------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Per-side joint lookup: side -> (shoulder, elbow, wrist, hip)
SIDE_JOINTS = {
    "left": (LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST, LEFT_HIP),
    "right": (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST, RIGHT_HIP),
}

# Upper-body skeleton edges, for annotation only.
SKELETON = [
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW),
    (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW),
    (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP),
    (RIGHT_SHOULDER, RIGHT_HIP),
    (LEFT_HIP, RIGHT_HIP),
    (NOSE, LEFT_SHOULDER),
    (NOSE, RIGHT_SHOULDER),
]


def ramp(value: float, lo: float, hi: float) -> float:
    """Map *value* onto [0, 1] linearly, clamped outside ``[lo, hi]``.

    This is the graded replacement for the predecessor's boolean thresholds.
    A boolean test has a cliff edge: a subject sitting exactly at the threshold
    flickers between raised and not-raised as keypoint noise jitters by a pixel.
    A ramp turns that cliff into a slope, so the temporal filter downstream has
    a continuous signal to smooth instead of a square wave.

    Supports a descending ramp (``lo > hi``), which yields 1 for small values.
    """
    if hi == lo:
        return 1.0 if value >= hi else 0.0
    t = (value - lo) / (hi - lo)
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t


def smoothstep(t: float) -> float:
    """Hermite ease on [0, 1]; softens a ramp's corners."""
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def angle_at_deg(
    vx: float, vy: float, ax: float, ay: float, bx: float, by: float
) -> float:
    """Interior angle in degrees at vertex ``v`` between rays ``v->a`` and ``v->b``.

    Used for the elbow angle: 180 degrees is a fully extended arm, ~45 degrees is
    a hand tucked near the head. Returns 0.0 when a limb segment is degenerate.
    """
    v1x, v1y = ax - vx, ay - vy
    v2x, v2y = bx - vx, by - vy
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_t = (v1x * v2x + v1y * v2y) / (n1 * n2)
    cos_t = max(-1.0, min(1.0, cos_t))
    return math.degrees(math.acos(cos_t))


def forearm_from_vertical_deg(ex: float, ey: float, wx: float, wy: float) -> float:
    """Angle in degrees between the elbow->wrist vector and straight up.

    0 is a forearm pointing at the ceiling, 90 is horizontal, 180 points at the
    floor. Unlike the elbow's interior angle this does not care where the elbow
    is, which is what lets a relaxed raise - elbow down by the ribs, forearm up -
    be told apart from a head-scratch, whose forearm slants in toward the head.
    Returns 180.0 for a degenerate forearm so it can never read as "up".
    """
    dx, dy = wx - ex, wy - ey
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return 180.0
    cos_t = max(-1.0, min(1.0, -dy / n))  # "up" is -y in image coordinates
    return math.degrees(math.acos(cos_t))


def face_reference(
    kp: list[tuple[float, float, float]], keypoint_conf: float
) -> tuple[float | None, float | None, str]:
    """Position of the face, from the best keypoints that are visible.

    The nose is preferred. When it is missing the eyes, then the ears, stand in:
    all three sit within a few pixels of each other vertically, so a hand held
    above any of them is above face height, and a pair of them is centred on the
    face horizontally. These are real keypoints, not a proxy invented from the
    shoulders - when none of the face is visible the caller gets ``None`` and
    must drop the feature rather than guess.

    Returns ``(x, y, source)`` where *source* names the keypoint group used.
    """
    for name, idxs in (("nose", (NOSE,)), ("eyes", (LEFT_EYE, RIGHT_EYE)),
                       ("ears", (LEFT_EAR, RIGHT_EAR))):
        pts = [kp[i] for i in idxs if kp[i][2] >= keypoint_conf]
        if pts:
            n = len(pts)
            return sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n, name
    return None, None, ""


def torso_length(
    kp: list[tuple[float, float, float]],
    bbox_height: float,
    keypoint_conf: float,
    min_torso_px: float,
) -> float:
    """Best available estimate of torso length in pixels, with fallbacks.

    Torso length is the scale denominator for every other measurement, so it has
    to degrade gracefully rather than return garbage:

    1. shoulder-to-hip vertical span (most reliable when hips are visible),
    2. shoulder width x 1.5 (seated students often have hips occluded by desks -
       this is the common case in a classroom),
    3. 30% of the detection box height (last resort).

    Returns 0.0 when even the fallback is below *min_torso_px*, which the caller
    must treat as "too small to judge" rather than as a negative.
    """
    def ok(i: int) -> bool:
        return kp[i][2] >= keypoint_conf

    # 1. shoulder midpoint to hip midpoint
    if ok(LEFT_SHOULDER) and ok(RIGHT_SHOULDER) and ok(LEFT_HIP) and ok(RIGHT_HIP):
        sy = (kp[LEFT_SHOULDER][1] + kp[RIGHT_SHOULDER][1]) * 0.5
        hy = (kp[LEFT_HIP][1] + kp[RIGHT_HIP][1]) * 0.5
        span = abs(hy - sy)
        if span >= min_torso_px:
            return span

    # 1b. single-sided shoulder to hip
    for s, h in ((LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP)):
        if ok(s) and ok(h):
            span = abs(kp[h][1] - kp[s][1])
            if span >= min_torso_px:
                return span

    # 2. shoulder width proxy - the usual path for desk-occluded students
    if ok(LEFT_SHOULDER) and ok(RIGHT_SHOULDER):
        width = dist(
            kp[LEFT_SHOULDER][0], kp[LEFT_SHOULDER][1],
            kp[RIGHT_SHOULDER][0], kp[RIGHT_SHOULDER][1],
        )
        span = width * 1.5
        if span >= min_torso_px:
            return span

    # 3. detection-box proxy
    span = bbox_height * 0.30
    return span if span >= min_torso_px else 0.0
