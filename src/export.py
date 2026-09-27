"""Attendance export: CSV, XLSX and a session telemetry file.

The CSV keeps the predecessor's exact first five columns - Roll Number, Name,
Status, Marked At, Evidence - so anything already consuming those files keeps
working, and adds the columns that make a mark checkable: which tracked person was
credited, how strong the raise was, and where the evidence image is.

Two deliberate choices:

* **utf-8-sig.** Excel misreads a plain UTF-8 CSV, which is why the predecessor's
  own output carried a BOM. Roster loading reads utf-8-sig for the same reason.
* **Pending is never silently exported as Absent.** One of the predecessor's eight
  saved runs is twenty rows of `Pending` with empty timestamps, written by an
  export that did not check. Here an unfinished roster is reported to the caller,
  which asks before continuing, and the count is recorded in the session file.

`session_*.json` carries the full config, runtime details and a record per mark.
That is what makes Phase 4 threshold tuning possible after the fact, and it is
also what answers questions like "did the same tracked person get credited twice?"
- which cannot be reconstructed from the CSV alone.
"""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime
from pathlib import Path

from roster import ABSENT, PENDING, PRESENT, summarize

CSV_HEADER = [
    "Roll Number", "Name", "Status", "Marked At", "Evidence",
    "Track ID", "Score", "Evidence Image",
]


def resolve_output_dir(cfg) -> Path:
    """Where attendance files are written.

    Defaults to the predecessor's location so an existing habit keeps working.
    """
    configured = (cfg.get("output.dir") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / "Documents" / "SmartRollCall Attendance"


class ExportError(Exception):
    pass


class Exporter:
    def __init__(self, cfg, session_id: str | None = None):
        self.cfg = cfg
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = resolve_output_dir(cfg)
        self.want_xlsx = bool(cfg.get("output.write_xlsx", True))
        self.want_session = bool(cfg.get("output.write_session_json", True))

    # -- public ------------------------------------------------------------
    def export(self, students, telemetry=None, runtime=None, path: Path | None = None
               ) -> list[Path]:
        """Write every configured output. Returns the paths written."""
        if not students:
            raise ExportError("there is nothing to export")

        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = path.with_suffix("") if path else (self.dir / f"attendance_{stamp}")

        written = [self.write_csv(students, base.with_suffix(".csv"))]

        if self.want_xlsx:
            try:
                written.append(self.write_xlsx(students, base.with_suffix(".xlsx")))
            except ImportError:
                pass  # openpyxl absent: CSV is still written, which is the important one
        if self.want_session:
            written.append(
                self.write_session(
                    students, base.with_name(f"session_{stamp}").with_suffix(".json"),
                    telemetry=telemetry, runtime=runtime,
                )
            )
        return written

    def pending_count(self, students) -> int:
        return sum(1 for s in students if s.status == PENDING)

    # -- csv ---------------------------------------------------------------
    def write_csv(self, students, path: Path) -> Path:
        with open(path, "w", encoding="utf-8-sig", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(CSV_HEADER)
            for s in students:
                w.writerow([
                    s.roll,
                    s.name,
                    s.status,
                    s.marked_at,
                    s.evidence,
                    "" if s.track_id is None else s.track_id,
                    f"{s.score:.3f}" if s.score else "",
                    s.evidence_image,
                ])
        return path

    # -- xlsx --------------------------------------------------------------
    def write_xlsx(self, students, path: Path) -> Path:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        ws = wb.active
        ws.title = "Attendance"

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill("solid", fgColor="2F4F7A")
        fills = {
            PRESENT: PatternFill("solid", fgColor="D8F3DC"),
            ABSENT: PatternFill("solid", fgColor="FFD6DD"),
            PENDING: PatternFill("solid", fgColor="EDEDED"),
        }

        ws.append(CSV_HEADER)
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")

        for s in students:
            ws.append([
                s.roll, s.name, s.status, s.marked_at, s.evidence,
                s.track_id if s.track_id is not None else "",
                round(s.score, 3) if s.score else "",
                s.evidence_image,
            ])
            fill = fills.get(s.status)
            if fill is not None:
                ws.cell(row=ws.max_row, column=3).fill = fill

        widths = (18, 22, 11, 21, 26, 9, 8, 46)
        for i, width in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = width
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = f"A1:{get_column_letter(len(CSV_HEADER))}{ws.max_row}"

        # A summary sheet, because the first thing anyone asks a register is "how
        # many were present?" and counting coloured rows by eye is not an answer.
        stats = summarize(students)
        s2 = wb.create_sheet("Summary")
        s2.append(["Generated", datetime.now().isoformat(timespec="seconds")])
        s2.append(["Session", self.session_id])
        s2.append(["Folder", str(self.dir)])
        s2.append(["Evidence paths", "relative to Folder"])
        for key in ("total", "present", "absent", "pending"):
            s2.append([key.capitalize(), stats[key]])
        if stats["total"]:
            rate = stats["present"] / stats["total"] * 100.0
            s2.append(["Attendance %", round(rate, 1)])
        if stats["pending"]:
            s2.append(["WARNING", f"{stats['pending']} student(s) were never called"])
        s2.column_dimensions["A"].width = 16
        s2.column_dimensions["B"].width = 42

        wb.save(path)
        return path

    # -- session telemetry --------------------------------------------------
    def write_session(self, students, path: Path, telemetry=None, runtime=None) -> Path:
        stats = summarize(students)
        payload = {
            "session_id": self.session_id,
            "generated": datetime.now().isoformat(timespec="seconds"),
            "host": {
                "python": platform.python_version(),
                "machine": platform.machine(),
                "system": f"{platform.system()} {platform.release()}",
            },
            "output_dir": str(self.dir),
            "evidence_paths": "relative to output_dir",
            "runtime": runtime or {},
            "config": self.cfg.as_dict(),
            "summary": stats,
            "students": [
                {
                    "roll": s.roll,
                    "name": s.name,
                    "status": s.status,
                    "marked_at": s.marked_at,
                    "evidence": s.evidence,
                    "track_id": s.track_id,
                    "score": s.score,
                    "evidence_image": s.evidence_image,
                }
                for s in students
            ],
            "marks": list(telemetry or []),
        }
        # Track-reuse audit. Rule B should make this impossible, so if it ever
        # appears it means tracker IDs are churning and the anti-proxy guarantee is
        # weaker than designed. Recording it is how that gets noticed at all.
        seen: dict[int, list[str]] = {}
        for s in students:
            if s.track_id is not None:
                seen.setdefault(s.track_id, []).append(s.roll)
        payload["track_reuse"] = {
            str(tid): rolls for tid, rolls in seen.items() if len(rolls) > 1
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
