"""Evidence images: a cropped photo of the student at the moment they were credited.

The predecessor recorded the string "Hand only" and nothing else. When a student
disputes an absence, or an administrator asks how a mark was arrived at, a text
label proves nothing and cannot be checked. A cropped frame showing the raised
hand, captioned with the roll number, score and timestamp, can be looked at by a
human and either accepted or rejected.

The crop is deliberately **not** drawn over with the pose skeleton. The skeleton is
useful for tuning and is shown live in the operator view, but it obscures exactly
the thing an evidence image exists to show.

Privacy: these are photographs of students. Where they are stored, who can read
them, and how long they are kept are decisions for the institution, not for this
program. `output.save_evidence_images: false` disables capture entirely, and the
folder is written next to the attendance files so a retention policy can cover
both together.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from annotate import crop_person

CAPTION_BG = (24, 24, 28)
CAPTION_FG = (240, 240, 245)
ACCENT = (110, 220, 130)


class EvidenceStore:
    def __init__(self, cfg, session_id: str | None = None):
        out = cfg.section("output")
        self.enabled: bool = bool(out.get("save_evidence_images", True))
        self.quality: int = int(out.get("evidence_jpeg_quality", 88))
        self.min_width: int = int(out.get("evidence_min_width", 240))

        from export import resolve_output_dir  # shared with the exporter

        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.base = resolve_output_dir(cfg)
        self.root = self.base / "evidence" / self.session_id
        self.count = 0

    # -- capture -----------------------------------------------------------
    def capture(self, student, track, video) -> str:
        """Save a captioned crop of *track* from the current frame.

        Returns the path as a string, or "" when nothing could be captured. Never
        raises: an attendance mark must not fail because a disk write failed.
        """
        if not self.enabled or track is None or video is None:
            return ""
        try:
            snap = video.snapshot()
        except Exception:
            return ""
        frame = getattr(snap, "frame", None)
        if frame is None:
            return ""

        bbox = getattr(track, "best_bbox", None) or getattr(track, "bbox", None)
        if bbox is None:
            return ""

        try:
            crop = crop_person(frame, bbox, pad_ratio=0.22)
            if crop is None or crop.size == 0:
                return ""
            crop = self._upscale_small(crop)
            crop = self._caption(crop, student, track)

            self.root.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in student.roll)
            path = self.root / f"{safe}.jpg"
            ok = cv2.imwrite(
                str(path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            )
            if not ok:
                return ""
            self.count += 1
            # Recorded relative to the output folder, so the whole folder - CSV,
            # workbook and images together - can be moved or archived without the
            # references breaking. session_*.json records the absolute base, and
            # the workbook's Summary sheet names it too, so the full path is always
            # recoverable.
            try:
                return str(path.relative_to(self.base))
            except ValueError:
                return str(path)
        except Exception:
            return ""

    # -- helpers -----------------------------------------------------------
    def _upscale_small(self, crop):
        """Enlarge a tiny back-row crop so a person can actually see it.

        A student 90 px tall is legitimate for detection but useless as evidence at
        native size. Upscaling does not add information, but it makes the image
        viewable without the reader having to zoom.
        """
        h, w = crop.shape[:2]
        if w >= self.min_width or w == 0:
            return crop
        scale = min(4.0, self.min_width / float(w))
        return cv2.resize(
            crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
        )

    def _caption(self, crop, student, track):
        """Burn roll number, score and timestamp into a strip under the image.

        Burned in rather than kept only in the CSV, so the image stays
        self-describing if it is copied out of the folder on its own.
        """
        h, w = crop.shape[:2]
        strip = max(28, int(h * 0.12))
        out = cv2.copyMakeBorder(
            crop, 0, strip, 0, 0, cv2.BORDER_CONSTANT, value=CAPTION_BG
        )
        scale = max(0.42, min(0.9, w / 520.0))
        thickness = 1 if scale < 0.7 else 2

        left = student.roll
        if student.name:
            left += f"  {student.name}"
        cv2.putText(
            out, left, (10, h + int(strip * 0.62)),
            cv2.FONT_HERSHEY_SIMPLEX, scale, CAPTION_FG, thickness, cv2.LINE_AA,
        )

        score = getattr(track, "peak_ema", 0.0)
        right = f"{score:.2f}  #{getattr(track, 'track_id', '?')}  " \
                f"{datetime.now().strftime('%H:%M:%S')}"
        (tw, _), _ = cv2.getTextSize(right, cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, thickness)
        cv2.putText(
            out, right, (max(10, w - tw - 10), h + int(strip * 0.62)),
            cv2.FONT_HERSHEY_SIMPLEX, scale * 0.8, ACCENT, thickness, cv2.LINE_AA,
        )
        return out
