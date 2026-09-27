"""Frame annotation for the operator view and for evidence images.

Deliberately shows the *graded* score, not just a raised/not-raised badge. During
calibration you need to see how close each student is to the threshold: a room
full of 0.50s means the threshold is wrong, and a binary overlay would hide that
completely.
"""

from __future__ import annotations

import cv2
import numpy as np

from geometry import SKELETON

# BGR
GRAY = (150, 150, 150)
AMBER = (60, 190, 250)
GREEN = (110, 220, 130)
RED = (110, 110, 250)
WHITE = (255, 255, 255)
DARK = (35, 35, 35)


def score_color(score: float, enter: float, exit_: float) -> tuple[int, int, int]:
    if score >= enter:
        return GREEN
    if score >= exit_:
        return AMBER
    return GRAY


def _draw_skeleton(
    frame: np.ndarray,
    kp: list[tuple[float, float, float]],
    conf_min: float,
    color: tuple[int, int, int],
) -> None:
    for a, b in SKELETON:
        if kp[a][2] < conf_min or kp[b][2] < conf_min:
            continue
        cv2.line(
            frame,
            (int(kp[a][0]), int(kp[a][1])),
            (int(kp[b][0]), int(kp[b][1])),
            color,
            2,
            cv2.LINE_AA,
        )
    for x, y, c in kp:
        if c >= conf_min:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


def _label(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
) -> None:
    """Text with a filled backing box, so it stays readable over any background."""
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x, y = org
    cv2.rectangle(frame, (x, y - th - base - 3), (x + tw + 6, y + 2), DARK, -1)
    cv2.putText(
        frame, text, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA
    )


def _score_bar(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    score: float,
    color: tuple[int, int, int],
    enter: float,
) -> None:
    x1, y1, x2, _y2 = (int(v) for v in bbox)
    w = max(40, x2 - x1)
    top = max(2, y1 - 8)
    cv2.rectangle(frame, (x1, top), (x1 + w, top + 5), DARK, -1)
    fill = int(w * max(0.0, min(1.0, score)))
    if fill > 0:
        cv2.rectangle(frame, (x1, top), (x1 + fill, top + 5), color, -1)
    # Threshold tick, so the operator can see how far off a student is.
    tick = x1 + int(w * enter)
    cv2.line(frame, (tick, top - 2), (tick, top + 7), WHITE, 1)


def annotate(
    frame: np.ndarray,
    result,
    cfg,
    track_states: dict | None = None,
    header: str | None = None,
    copy: bool = True,
) -> np.ndarray:
    """Draw detections, scores and status onto a frame.

    Args:
        result: a FrameResult from VideoWorker.
        track_states: optional ``track_id -> TrackState`` for smoothed values.
        header: optional status line drawn across the top.
    """
    out = frame.copy() if copy else frame
    enter = float(cfg.get("temporal.enter_threshold", 0.55))
    exit_ = float(cfg.get("temporal.exit_threshold", 0.35))
    kp_conf = float(cfg.get("model.keypoint_conf", 0.30))

    for p in result.people:
        st = None
        if track_states and p.track_id is not None:
            st = track_states.get(p.track_id)
        shown = st.ema if st is not None else p.score.score
        confirmed = bool(st.confirmed) if st is not None else False

        color = GREEN if confirmed else score_color(shown, enter, exit_)
        x1, y1, x2, y2 = (int(v) for v in p.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2 if confirmed else 1)
        _draw_skeleton(out, p.keypoints, kp_conf, color)
        _score_bar(out, p.bbox, shown, color, enter)

        tid = f"#{p.track_id}" if p.track_id is not None else "#?"
        bits = [tid, f"{shown:.2f}"]
        if p.score.side:
            bits.append(p.score.side[0].upper())
        if confirmed:
            bits.append("RAISED")
        _label(out, " ".join(bits), (x1, max(14, y1 - 10)), color)

    if header:
        h, w = out.shape[:2]
        cv2.rectangle(out, (0, 0), (w, 30), DARK, -1)
        cv2.putText(
            out, header, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1, cv2.LINE_AA
        )
    return out


def crop_person(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_ratio: float = 0.18,
) -> np.ndarray:
    """Crop a padded box around one person, for the evidence image."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pw = (x2 - x1) * pad_ratio
    ph = (y2 - y1) * pad_ratio
    x1 = int(max(0, x1 - pw))
    y1 = int(max(0, y1 - ph))
    x2 = int(min(w, x2 + pw))
    y2 = int(min(h, y2 + ph))
    if x2 <= x1 or y2 <= y1:
        return frame.copy()
    return frame[y1:y2, x1:x2].copy()
