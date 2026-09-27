"""Roster loading and shortest-unique spoken calls.

The spoken-call idea is taken from the predecessor, which had it right: the screen
shows the full roll number (`SP25-BCS-007`) while the speaker says only what is
needed to identify it uniquely (`seven`). Reading the whole string aloud -
"ess pee two five bee see ess zero zero seven" - takes about four seconds per
student, which is longer than the response window itself and would make a 60-person
class unbearable.

The logic here picks the shortest form that is still unambiguous across the loaded
roster, escalating only when it has to:

    level 0   number only            "seven"
    level 1   program + number       "B C S seven"
    level 2   batch + program        "S P two five B C S seven"
    level 3   whole roll, spelled    (fallback)

A single-section roster lands on level 0. A roster mixing `SP25-BCS-001` with
`FA24-BCS-001` is forced to level 2, automatically.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

# --- number to words -------------------------------------------------------
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def number_words(n: int) -> str:
    """Spell an integer in English words. Handles 0..9999, which covers roll numbers."""
    if n < 0:
        return "minus " + number_words(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (" " + _ONES[rest] if rest else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        out = _ONES[hundreds] + " hundred"
        return out + (" " + number_words(rest) if rest else "")
    thousands, rest = divmod(n, 1000)
    out = number_words(thousands) + " thousand"
    return out + (" " + number_words(rest) if rest else "")


def letters_spoken(text: str) -> str:
    """Spell letters individually so TTS does not mangle an acronym.

    "BCS" read as a word comes out as an unintelligible syllable; "B C S" is clear.
    """
    return " ".join(ch.upper() for ch in text if ch.isalnum())


def spoken_component(part: str) -> str:
    """Speak one roll component, mixing letter-spelling and number words.

    "SP25" -> "S P two five": the digits are spoken individually rather than as
    "twenty five", because a batch code is read as a sequence, not a quantity.
    """
    out: list[str] = []
    for chunk in re.findall(r"[A-Za-z]+|\d+", part):
        if chunk.isdigit():
            out.extend(_ONES[int(d)] for d in chunk)
        else:
            out.append(letters_spoken(chunk))
    return " ".join(x for x in out if x)


# --- roll parsing ----------------------------------------------------------
_SPLIT = re.compile(r"[-_/\s]+")
_TRAILING_NUM = re.compile(r"(\d+)\s*$")


@dataclass
class RollParts:
    raw: str
    parts: list[str] = field(default_factory=list)
    number: int | None = None
    program: str = ""
    batch: str = ""


def parse_roll(roll: str) -> RollParts:
    """Split a roll number into batch / program / trailing number."""
    raw = roll.strip()
    parts = [p for p in _SPLIT.split(raw) if p]
    out = RollParts(raw=raw, parts=parts)

    m = _TRAILING_NUM.search(raw)
    if m:
        try:
            out.number = int(m.group(1))
        except ValueError:
            out.number = None

    # Everything before the trailing-number component is prefix. For
    # SP25-BCS-001 that gives batch="SP25", program="BCS".
    prefix = parts[:-1] if (len(parts) > 1 and _TRAILING_NUM.search(parts[-1])) else parts
    if prefix:
        out.program = prefix[-1]
        out.batch = prefix[0] if len(prefix) > 1 else ""
    return out


# --- student record --------------------------------------------------------
PENDING = "Pending"
PRESENT = "Present"
ABSENT = "Absent"


@dataclass
class Student:
    roll: str
    name: str = ""
    status: str = PENDING
    marked_at: str = ""
    evidence: str = ""
    spoken_call: str = ""
    track_id: int | None = None
    score: float = 0.0
    evidence_image: str = ""

    @property
    def is_pending(self) -> bool:
        return self.status == PENDING


def assign_short_unique_calls(students: list[Student]) -> int:
    """Give every student the shortest spoken form unique within this roster.

    Returns the escalation level that was needed (0-3), which the UI reports so the
    operator can see why calls are long when they are.
    """
    if not students:
        return 0

    parsed = {s.roll: parse_roll(s.roll) for s in students}

    def candidate(roll: str, level: int) -> str:
        p = parsed[roll]
        num = number_words(p.number) if p.number is not None else spoken_component(p.parts[-1] if p.parts else roll)
        if level == 0:
            return num
        if level == 1:
            return f"{spoken_component(p.program)} {num}".strip() if p.program else num
        if level == 2:
            bits = [spoken_component(p.batch), spoken_component(p.program), num]
            return " ".join(b for b in bits if b)
        return spoken_component(roll)

    for level in (0, 1, 2, 3):
        forms = {s.roll: candidate(s.roll, level) for s in students}
        if len(set(forms.values())) == len(students):
            for s in students:
                s.spoken_call = forms[s.roll]
            return level

    # Unreachable for distinct rolls, but never leave the field empty.
    for s in students:
        s.spoken_call = spoken_component(s.roll)
    return 3


# --- loading ---------------------------------------------------------------
_ROLL_HINTS = ("roll", "reg", "id", "student", "number", "no")
_NAME_HINTS = ("name", "student name", "full name")


class RosterError(Exception):
    pass


def _looks_like_header(row: list[str]) -> bool:
    joined = " ".join(row).lower()
    return any(h in joined for h in _ROLL_HINTS + _NAME_HINTS) and not any(
        _TRAILING_NUM.search(c) and c.strip().isdigit() for c in row
    )


def parse_rows(rows: list[list[str]]) -> list[Student]:
    """Turn raw CSV rows into students, detecting an optional header."""
    rows = [[(c or "").strip() for c in r] for r in rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        raise RosterError("no roll numbers found")

    roll_idx, name_idx = 0, 1
    start = 0
    if _looks_like_header(rows[0]):
        header = [c.lower() for c in rows[0]]
        for i, h in enumerate(header):
            if any(k in h for k in _ROLL_HINTS):
                roll_idx = i
                break
        name_idx = -1
        for i, h in enumerate(header):
            if any(k in h for k in _NAME_HINTS):
                name_idx = i
                break
        start = 1

    students: list[Student] = []
    seen: dict[str, int] = {}
    dupes: list[str] = []
    for r in rows[start:]:
        if roll_idx >= len(r):
            continue
        roll = r[roll_idx].strip()
        if not roll:
            continue
        name = ""
        if 0 <= name_idx < len(r):
            name = r[name_idx].strip()
        key = roll.upper()
        if key in seen:
            dupes.append(roll)
            continue
        seen[key] = len(students)
        students.append(Student(roll=roll, name=name))

    if not students:
        raise RosterError("no roll numbers found")
    if dupes:
        shown = ", ".join(sorted(set(dupes))[:6])
        raise RosterError(f"duplicate roll number(s): {shown}")

    assign_short_unique_calls(students)
    return students


def load_csv(path: Path | str) -> list[Student]:
    """Load a roster CSV.

    Read as utf-8-sig because Excel writes a BOM, which would otherwise become
    part of the first roll number and make it never match. Every cell is stripped
    - the sample roster on this machine has a leading space on one row.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        delimiter = ","
        try:
            delimiter = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except Exception:
            pass
        rows = list(csv.reader(fh, delimiter=delimiter))
    return parse_rows(rows)


def parse_pasted(text: str) -> list[Student]:
    """Parse roll numbers pasted as free text, split on commas/semicolons/whitespace."""
    tokens = [t for t in re.split(r"[\s,;]+", text or "") if t.strip()]
    return parse_rows([[t] for t in tokens])


# --- summary ---------------------------------------------------------------
def summarize(students: list[Student]) -> dict:
    present = sum(1 for s in students if s.status == PRESENT)
    absent = sum(1 for s in students if s.status == ABSENT)
    pending = sum(1 for s in students if s.status == PENDING)
    return {
        "total": len(students),
        "present": present,
        "absent": absent,
        "pending": pending,
    }
