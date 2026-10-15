"""Phone camera dialog: find or type the phone's address and connect to it.

Discovery and connection tests run on a worker thread - a network sweep takes a
few seconds and the video must keep running meanwhile - and report back through a
queue that this window polls, so every tkinter call stays on the UI thread.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk
from typing import Callable

DARK = "#15151a"
PANEL = "#1d1d24"
TEXT = "#e8e8ee"
MUTED = "#8b8b98"

HELP = (
    "1.  On the phone, install a free camera app:\n"
    "      Android:  IP Webcam  or  DroidCam        iPhone:  DroidCam\n"
    "2.  Put the phone and this laptop on the same network. No internet is needed:\n"
    "      - turn on the phone's hotspot and join it from the laptop, or\n"
    "      - turn on the laptop's Mobile Hotspot and join it from the phone, or\n"
    "      - USB cable: Android 'USB tethering' (or USB debugging + adb),\n"
    "        iPhone 'Personal Hotspot' with the cable plugged in.\n"
    "3.  Start the server in the app and stand the phone where it sees the class.\n"
    "4.  Press Find phone, or type the address the app shows (e.g. 192.168.43.1:8080)."
)


class PhoneDialog:
    def __init__(
        self,
        root: tk.Misc,
        initial: str,
        on_connect: Callable[[str], None],
        on_local: Callable[[], None],
    ):
        self.on_connect = on_connect
        self.on_local = on_local
        self._q: queue.Queue = queue.Queue()
        self._busy = False
        self._found: list = []

        self.win = tk.Toplevel(root)
        self.win.title("Phone camera")
        self.win.configure(bg=DARK)
        self.win.transient(root)
        self.win.resizable(False, False)

        frame = ttk.Frame(self.win, padding=14)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text=HELP, bg=DARK, fg=TEXT, justify="left",
                 font=("Segoe UI", 10)).pack(anchor="w")

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=(12, 6))
        ttk.Label(row, text="Address:").pack(side="left")
        self.addr = tk.StringVar(value=initial)
        entry = ttk.Entry(row, textvariable=self.addr, width=42)
        entry.pack(side="left", padx=6)
        entry.bind("<Return>", lambda _e: self.connect())
        self.btn_connect = ttk.Button(row, text="Connect", command=self.connect)
        self.btn_connect.pack(side="left")

        row2 = ttk.Frame(frame)
        row2.pack(fill="x", pady=(0, 6))
        self.btn_find = ttk.Button(row2, text="Find phone", command=self.find)
        self.btn_find.pack(side="left")
        ttk.Button(row2, text="Use laptop / USB webcam", command=self.use_local).pack(side="left", padx=6)
        ttk.Button(row2, text="Close", command=self.close).pack(side="right")

        self.results = tk.Listbox(frame, height=5, bg=PANEL, fg=TEXT, borderwidth=0,
                                  highlightthickness=0, selectbackground="#2f4f7a",
                                  font=("Consolas", 10))
        self.results.pack(fill="x", pady=(4, 6))
        self.results.bind("<Double-Button-1>", lambda _e: self._pick())
        self.results.bind("<<ListboxSelect>>", lambda _e: self._select())

        self.status = tk.Label(frame, text="", bg=DARK, fg=MUTED, anchor="w", justify="left",
                               font=("Segoe UI", 10), wraplength=620)
        self.status.pack(fill="x")

        entry.focus_set()
        self.win.after(100, self._poll)

    # -- actions -----------------------------------------------------------
    def connect(self) -> None:
        text = self.addr.get().strip()
        if not text:
            self._say("Type the phone's address, or press Find phone.")
            return
        self.on_connect(text)
        self._say(f"Connecting to {text} - watch the status bar of the main window.")
        self.win.after(1200, self.close)

    def use_local(self) -> None:
        self.on_local()
        self.close()

    def find(self) -> None:
        if self._busy:
            return
        self._busy = True
        self.btn_find.configure(state="disabled")
        self.results.delete(0, "end")
        self._found = []
        self._say("Searching... (a few seconds)")
        threading.Thread(target=self._find_worker, name="phone-find", daemon=True).start()

    def close(self) -> None:
        try:
            self.win.destroy()
        except Exception:
            pass

    # -- worker ------------------------------------------------------------
    def _find_worker(self) -> None:
        try:
            from netcam import adb_forward, find_adb, find_phone_cameras

            if find_adb():
                ok, msg = adb_forward()
                self._q.put(("say", f"USB (adb): {msg}"))
            found = find_phone_cameras(progress=lambda m: self._q.put(("say", m)))
            self._q.put(("found", found))
        except Exception as exc:
            self._q.put(("say", f"Search failed: {type(exc).__name__}: {exc}"))
            self._q.put(("found", []))

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "say":
                    self._say(payload)
                elif kind == "found":
                    self._busy = False
                    self.btn_find.configure(state="normal")
                    self._found = payload
                    for cam in payload:
                        self.results.insert("end", cam.label())
                    if payload:
                        self.results.selection_set(0)
                        self.addr.set(payload[0].url)
                        self._say(f"Found {len(payload)} camera(s). Double-click one, or press Connect.")
                    else:
                        self._say(
                            "No phone camera found. Check that the app's server is started, that "
                            "both devices are on the same hotspot / Wi-Fi, and that the phone's "
                            "screen did not lock. You can also type the address the app shows."
                        )
        except queue.Empty:
            pass
        try:
            if self.win.winfo_exists():
                self.win.after(100, self._poll)
        except Exception:
            pass

    def _select(self) -> None:
        sel = self.results.curselection()
        if sel and sel[0] < len(self._found):
            self.addr.set(self._found[sel[0]].url)

    def _pick(self) -> None:
        self._select()
        self.connect()

    def _say(self, text: str) -> None:
        try:
            self.status.configure(text=text)
        except Exception:
            pass
