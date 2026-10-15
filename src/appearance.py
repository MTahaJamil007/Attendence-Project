"""A clothing-colour signature for telling the same seated person from a neighbour.

No face is looked at or stored. The signature is a coarse colour histogram of the
chest - the region between the shoulders and a little below them - which within
one class period is a strong cue: the same person keeps the same shirt and the same
lighting, while the person beside them usually does not.

It is deliberately only *supporting* evidence for the seat check in gating.py.
Position decides; colour can only widen the "same seat" radius a little, from
``seat_radius_shoulders`` to ``seat_appearance_radius_shoulders``, for someone who
leaned while raising. It never overrides position, because in many classrooms
half the room wears white or black and colour alone would confuse them.

``similarity`` is the Bhattacharyya coefficient of two normalized histograms:
1.0 for identical colour distributions, near 0 for disjoint ones.
"""

from __future__ import annotations

import math

import numpy as np

from geometry import LEFT_SHOULDER, RIGHT_SHOULDER

# Hue x saturation x value bins. Value carries most of the information for
# white / grey / black clothing, whose hue is noise, so it gets its own axis
# instead of being folded into hue.
BINS = (12, 4, 4)
_MAX_SIDE = 48  # downsample the patch first: colour statistics need few pixels


def torso_signature(
    frame: np.ndarray,
    kp: list[tuple[float, float, float]],
    keypoint_conf: float,
) -> np.ndarray | None:
    """Normalized HSV histogram of the chest patch, or None when it cannot be seen."""
    import cv2

    ls, rs = kp[LEFT_SHOULDER], kp[RIGHT_SHOULDER]
    if ls[2] < keypoint_conf or rs[2] < keypoint_conf:
        return None
    sw = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
    if sw < 12.0:
        return None
    h, w = frame.shape[:2]
    x_lo, x_hi = sorted((ls[0], rs[0]))
    inset = (x_hi - x_lo) * 0.12
    top = max(ls[1], rs[1]) + sw * 0.15
    x1 = int(max(0, x_lo + inset))
    x2 = int(min(w, x_hi - inset))
    y1 = int(max(0, top))
    y2 = int(min(h, top + sw * 0.75))
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    patch = frame[y1:y2, x1:x2]
    ph, pw = patch.shape[:2]
    if max(ph, pw) > _MAX_SIDE:
        s = _MAX_SIDE / max(ph, pw)
        patch = cv2.resize(patch, (max(2, int(pw * s)), max(2, int(ph * s))),
                           interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, list(BINS), [0, 180, 0, 256, 0, 256])
    hist = hist.astype(np.float32).ravel()
    total = float(hist.sum())
    if total <= 0.0:
        return None
    return hist / total


def blend(old: np.ndarray | None, new: np.ndarray | None, weight: float = 0.3) -> np.ndarray | None:
    """Running average of signatures, so one odd frame cannot redefine a person."""
    if new is None:
        return old
    if old is None or getattr(old, "shape", None) != getattr(new, "shape", None):
        return new.copy()
    out = old * (1.0 - weight) + new * weight
    s = float(out.sum())
    return out / s if s > 0 else out


def similarity(a, b) -> float | None:
    """Bhattacharyya coefficient in [0, 1]; None when either side is missing."""
    if a is None or b is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return None
    return float(np.clip(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None)).sum(), 0.0, 1.0))
