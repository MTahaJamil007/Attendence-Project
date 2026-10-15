"""Student-facing display: the window that goes on the projector.

Kept from the predecessor's design, which was right to have a separate fullscreen
window showing a huge roll number. Two things are added, and both are about the
students rather than the teacher:

* **A countdown**, so nobody has to guess how long they have.
* **Live feedback** - "HAND SEEN - KEEP IT UP" with a filling hold bar, then
  "PRESENT". The predecessor showed hold progress only on the operator's screen,
  which meant a student had no way to tell whether they had been noticed. A student
  who can see they are being detected holds their hand up long enough; one who
  cannot drops it early and is marked absent. This costs nothing and is probably
  the largest real-world recall gain in the whole UI.

The countdown is a full-width bar rather than a ring around the number. A ring
looks better in a screenshot, but it caps the roll number at roughly a quarter of
the screen width - and a student in the back row reading "SP25-BCS-007" from
fifteen metres is the entire point of this window. The bar frees the centre so the
number can span most of the display.

Canvas items are created once and then reconfigured, rather than the canvas being
cleared and redrawn each frame, because a full redraw at 30 Hz flickers.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont

from session import Phase

BG = "#0d0d10"
FG = "#f2f2f5"
DIM = "#6b6b76"
TRACK = "#26262e"

PHASE_COLOR = {
    Phase.IDLE: "#6b6b76",
    Phase.CALLING: "#facc15",
    Phase.RESPONSE: "#60a5fa",
    Phase.AMBIGUOUS: "#c084fc",
    Phase.STALLED: "#fb923c",
    Phase.RESULT: "#4ade80",
    Phase.PAUSED: "#94a3b8",
    Phase.FINISHED: "#4ade80",
}
PRESENT_COLOR = "#4ade80"
ABSENT_COLOR = "#fb7185"
PARTIAL_COLOR = "#facc15"


def monitors() -> list[tuple[tuple[int, int, int, int], bool]]:
    """Every monitor as ((left, top, right, bottom), is_primary). Windows only.

    Returns an empty list where the query is not available, which callers
    treat as "one screen".
    """
    try:
        import ctypes
        from ctypes import wintypes

        class MONITORINFO(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                        ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

        user32 = ctypes.windll.user32
        found: list[tuple[tuple[int, int, int, int], bool]] = []
        proc = ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC,
                                  ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)

        def cb(hmon, _hdc, _rect, _data):
            info = MONITORINFO()
            info.cbSize = ctypes.sizeof(MONITORINFO)
            if user32.GetMonitorInfoW(hmon, ctypes.byref(info)):
                r = info.rcMonitor
                found.append(((r.left, r.top, r.right, r.bottom), bool(info.dwFlags & 1)))
            return 1

        user32.EnumDisplayMonitors(None, None, proc(cb), 0)
        return found
    except Exception:
        return []


def secondary_monitor() -> tuple[int, int, int, int] | None:
    """The first non-primary monitor (a projector), or None with one screen."""
    for rect, primary in monitors():
        if not primary:
            return rect
    return None


class RollDisplay:
    """A Toplevel showing the current roll number, countdown and feedback."""

    def __init__(self, root: tk.Misc, cfg, on_close=None):
        self.cfg = cfg
        self._on_close = on_close
        self.win = tk.Toplevel(root)
        self.win.title("Roll Number - Student Display")
        self.win.geometry("1100x650")
        self.win.configure(bg=BG)
        self.win.minsize(480, 320)
        self._fullscreen = False
        self._w, self._h = 1100, 650
        self.on_projector = False
        self._roll_shown = None

        self.canvas = tk.Canvas(self.win, bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        # Countdown bar across the top.
        self.time_bg = self.canvas.create_rectangle(0, 0, 1, 1, fill=TRACK, width=0)
        self.time_fill = self.canvas.create_rectangle(0, 0, 1, 1, fill="#60a5fa", width=0)

        self._roll_font = tkfont.Font(family="Segoe UI", size=120, weight="bold")
        self.roll_text = self.canvas.create_text(
            0, 0, text="", fill=FG, font=self._roll_font, justify="center"
        )
        self.name_text = self.canvas.create_text(0, 0, text="", fill=DIM,
                                                 font=("Segoe UI", 26))
        self.prompt_text = self.canvas.create_text(0, 0, text="Load a roll list",
                                                   fill=FG, font=("Segoe UI", 40, "bold"))
        self.secs_text = self.canvas.create_text(0, 0, text="", fill=DIM,
                                                 font=("Segoe UI", 22), anchor="w")
        self.hold_bg = self.canvas.create_rectangle(0, 0, 1, 1, fill=TRACK, width=0)
        self.hold_fill = self.canvas.create_rectangle(0, 0, 1, 1, fill=PRESENT_COLOR, width=0)
        self.counter_text = self.canvas.create_text(0, 0, text="", fill=DIM,
                                                    font=("Segoe UI", 18), anchor="e")

        self.win.bind("<F11>", lambda _e: self.toggle_fullscreen())
        self.win.bind("<Escape>", lambda _e: self.exit_fullscreen())
        self.win.protocol("WM_DELETE_WINDOW", self._closed)
        self.canvas.bind("<Configure>", lambda _e: self._layout())
        self._layout()

    # -- window ------------------------------------------------------------
    @property
    def alive(self) -> bool:
        try:
            return bool(self.win.winfo_exists())
        except Exception:
            return False

    def toggle_fullscreen(self) -> None:
        self._fullscreen = not self._fullscreen
        try:
            self.win.attributes("-fullscreen", self._fullscreen)
        except Exception:
            pass

    def exit_fullscreen(self) -> None:
        self._fullscreen = False
        try:
            self.win.attributes("-fullscreen", False)
        except Exception:
            pass

    def place_on_secondary_monitor(self) -> bool:
        """Move the window onto a second monitor (the projector), if there is one.

        Returns True when it was moved. Only the position changes; F11 still
        toggles fullscreen, now on the projector instead of the laptop screen.
        """
        mon = secondary_monitor()
        self.on_projector = mon is not None
        if mon is None:
            return False
        x1, y1, x2, y2 = mon
        w, h = min(self._w, x2 - x1), min(self._h, y2 - y1)
        x = x1 + max(0, (x2 - x1 - w) // 2)
        y = y1 + max(0, (y2 - y1 - h) // 2)
        try:
            self.win.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            self.on_projector = False
        return self.on_projector

    def send_behind(self, other: tk.Misc) -> None:
        """Open without covering *other* (the teacher's console)."""
        try:
            self.win.lower(other)
        except Exception:
            pass

    def focus(self) -> None:
        try:
            self.win.deiconify()
            self.win.lift()
        except Exception:
            pass

    def _closed(self) -> None:
        if self._on_close is not None:
            self._on_close()
        try:
            self.win.destroy()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self.win.destroy()
        except Exception:
            pass

    # -- layout ------------------------------------------------------------
    def _layout(self) -> None:
        if not self.alive:
            return
        w = max(1, self.canvas.winfo_width())
        h = max(1, self.canvas.winfo_height())
        self._w, self._h = w, h
        cx = w * 0.5

        # Countdown bar, full width across the top.
        bar_h = max(10.0, h * 0.035)
        self._time_bar = (0.0, 0.0, float(w), bar_h)
        self.canvas.coords(self.time_bg, *self._time_bar)
        self.canvas.coords(self.time_fill, 0, 0, float(w), bar_h)

        self.canvas.coords(self.secs_text, 18, bar_h + h * 0.055)
        self.canvas.coords(self.counter_text, w - 20, bar_h + h * 0.055)

        self.canvas.coords(self.roll_text, cx, h * 0.44)
        self.canvas.coords(self.name_text, cx, h * 0.63)
        self.canvas.coords(self.prompt_text, cx, h * 0.78)

        hb_w = w * 0.62
        hb_h = max(10.0, h * 0.028)
        hb_y = h * 0.90
        self._hold_bar = (cx - hb_w / 2, hb_y, cx + hb_w / 2, hb_y + hb_h)
        self.canvas.coords(self.hold_bg, *self._hold_bar)
        self.canvas.coords(self.hold_fill, self._hold_bar[0], hb_y, self._hold_bar[0], hb_y + hb_h)

        base = min(w, h)
        self.canvas.itemconfigure(self.name_text, font=("Segoe UI", max(10, int(base * 0.042))))
        self.canvas.itemconfigure(self.prompt_text, font=("Segoe UI", max(12, int(base * 0.062)), "bold"))
        self.canvas.itemconfigure(self.secs_text, font=("Segoe UI", max(10, int(base * 0.034))))
        self.canvas.itemconfigure(self.counter_text, font=("Segoe UI", max(9, int(base * 0.028))))

        self._roll_shown = None  # force a refit at the new size
        self._fit_roll(self.canvas.itemcget(self.roll_text, "text"))

    def _fit_roll(self, text: str) -> None:
        """Size the roll number to fill the width without overflowing it.

        Measured rather than estimated, because roll numbers vary a lot in length
        ("7" versus "SP25-BCS-007") and a fixed size either clips the long ones or
        wastes most of the screen on the short ones.
        """
        if text == self._roll_shown:
            return
        self._roll_shown = text
        if not text:
            return
        limit = self._w * 0.88
        size = max(16, int(self._h * 0.30))
        self._roll_font.configure(size=size)
        while size > 16 and self._roll_font.measure(text) > limit:
            size -= 2
            self._roll_font.configure(size=size)

    # -- render ------------------------------------------------------------
    def update(self, view) -> None:
        if not self.alive:
            return

        colour = PHASE_COLOR.get(view.phase, FG)
        if view.phase is Phase.RESULT:
            colour = PRESENT_COLOR if view.result == "PRESENT" else ABSENT_COLOR

        roll = view.roll or ("--" if view.total else "")
        self._fit_roll(roll)
        self.canvas.itemconfigure(self.roll_text, text=roll, fill=FG)
        self.canvas.itemconfigure(self.name_text, text=view.name or "")
        self.canvas.itemconfigure(self.prompt_text, text=view.prompt, fill=colour)

        # Countdown bar.
        x0, y0, x1, y1 = self._time_bar
        if view.phase in (Phase.RESPONSE, Phase.STALLED) and view.window > 0:
            fill_x = x0 + (x1 - x0) * view.window_fraction
            self.canvas.coords(self.time_fill, x0, y0, fill_x, y1)
            self.canvas.itemconfigure(self.time_fill, state="normal", fill=colour)
            self.canvas.itemconfigure(self.time_bg, state="normal")
            # During a stall the bar is frozen on purpose: the clock genuinely is
            # not running, and animating it would misrepresent what happens next.
            label = (
                "paused - camera problem"
                if view.phase is Phase.STALLED
                else f"{view.remaining:0.1f}s"
            )
            self.canvas.itemconfigure(self.secs_text, text=label, fill=colour)
        else:
            self.canvas.itemconfigure(self.time_fill, state="hidden")
            self.canvas.itemconfigure(self.time_bg, state="hidden")
            self.canvas.itemconfigure(self.secs_text, text="")

        # Hold bar: green once a raise is being held, amber while something is
        # detected but not yet enough, so "not seen" and "not enough" look different.
        hx0, hy0, hx1, hy1 = self._hold_bar
        show_bg = view.phase in (Phase.RESPONSE, Phase.STALLED)
        if view.hold_progress > 0.0 and show_bg:
            frac, fill = max(0.0, min(1.0, view.hold_progress)), PRESENT_COLOR
        elif show_bg and view.best_score > 0.05:
            frac, fill = max(0.0, min(1.0, view.best_score)), PARTIAL_COLOR
        else:
            frac, fill = 0.0, PRESENT_COLOR
        if frac > 0.0:
            self.canvas.coords(self.hold_fill, hx0, hy0, hx0 + (hx1 - hx0) * frac, hy1)
            self.canvas.itemconfigure(self.hold_fill, state="normal", fill=fill)
        else:
            self.canvas.itemconfigure(self.hold_fill, state="hidden")
        self.canvas.itemconfigure(self.hold_bg, state="normal" if show_bg else "hidden")

        stats = view.stats or {}
        if view.total:
            self.canvas.itemconfigure(
                self.counter_text,
                text=f"{view.index}/{view.total}    "
                     f"present {stats.get('present', 0)}    absent {stats.get('absent', 0)}",
            )
        else:
            self.canvas.itemconfigure(self.counter_text, text="")
