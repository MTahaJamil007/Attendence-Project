"""Teacher console: live video, roster, controls, status.

The UI thread owns everything. VideoWorker runs its own capture and inference
threads but the console only ever reads an immutable snapshot from it, and the
session state machine is driven from this window's ``after`` loop. That keeps all
tkinter calls on the main thread, which is the only way tkinter is safe.

Manual Present / Absent / Undo are deliberately always available. A classroom tool
that cannot be overridden by the person responsible for the register is not usable,
and the ambiguity rule depends on there being a human to defer to.
"""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from annotate import annotate
from roster import PRESENT, RosterError, load_csv, parse_pasted, summarize
from session import Phase, Session
from ui.roll_display import RollDisplay

APP_TITLE = "Smart Roll Call - Hand Raise Attendance"

DARK = "#15151a"
PANEL = "#1d1d24"
TEXT = "#e8e8ee"
MUTED = "#8b8b98"

PHASE_TINT = {
    Phase.IDLE: MUTED,
    Phase.CALLING: "#facc15",
    Phase.RESPONSE: "#60a5fa",
    Phase.AMBIGUOUS: "#c084fc",
    Phase.STALLED: "#fb923c",
    Phase.RESULT: "#4ade80",
    Phase.PAUSED: "#94a3b8",
    Phase.FINISHED: "#4ade80",
}


class Console:
    def __init__(self, root: tk.Tk, cfg, video, tracker, gate, speech, session: Session,
                 exporter=None):
        self.root = root
        self.cfg = cfg
        self.video = video
        self.tracker = tracker
        self.gate = gate
        self.speech = speech
        self.session = session
        self.exporter = exporter

        self.display: RollDisplay | None = None
        self._photo = None
        self._last_frame_index = -1
        self._roster_sig = None
        self._devices_started = False

        root.title(APP_TITLE)
        root.geometry("1560x920")
        root.minsize(1100, 700)
        root.configure(bg=DARK)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_style()
        self._build_ui()
        self._bind_keys()
        self.root.after(33, self._tick)

    # -- chrome ------------------------------------------------------------
    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=DARK, foreground=TEXT, fieldbackground=PANEL)
        style.configure("TFrame", background=DARK)
        style.configure("TLabelframe", background=DARK, foreground=MUTED)
        style.configure("TLabelframe.Label", background=DARK, foreground=MUTED)
        style.configure("TLabel", background=DARK, foreground=TEXT)
        style.configure("Muted.TLabel", background=DARK, foreground=MUTED)
        style.configure("TButton", padding=(10, 6))
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=TEXT, rowheight=24, borderwidth=0)
        style.configure("Treeview.Heading", background=DARK, foreground=MUTED)
        style.map("Treeview", background=[("selected", "#2f4f7a")])

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        # --- toolbar ------------------------------------------------------
        bar = ttk.Frame(outer)
        bar.pack(fill="x", pady=(0, 10))
        self.btn_load = ttk.Button(bar, text="Load CSV", command=self.load_csv)
        self.btn_load.pack(side="left")
        ttk.Button(bar, text="Paste Rolls", command=self.paste_rolls).pack(side="left", padx=(6, 0))
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.btn_devices = ttk.Button(bar, text="Start Camera", command=self.start_devices)
        self.btn_devices.pack(side="left")
        ttk.Button(bar, text="Switch Camera", command=self.switch_camera).pack(side="left", padx=(6, 0))
        ttk.Button(bar, text="Roll Screen", command=self.open_display).pack(side="left", padx=(6, 0))
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        self.btn_start = ttk.Button(bar, text="Start Attendance", command=self.start_attendance)
        self.btn_start.pack(side="left")
        self.btn_pause = ttk.Button(bar, text="Pause", command=self.pause)
        self.btn_pause.pack(side="left", padx=(6, 0))
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(bar, text="Export", command=self.export).pack(side="left")
        ttk.Button(bar, text="Reset Marks", command=self.reset_marks).pack(side="left", padx=(6, 0))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        # --- left: video + current call -----------------------------------
        left = ttk.Frame(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        self.video_label = tk.Label(left, bg="#000000", text="Camera not started",
                                    fg=MUTED, font=("Segoe UI", 14))
        self.video_label.grid(row=0, column=0, sticky="nsew")

        call = ttk.Labelframe(left, text="Current call", padding=10)
        call.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        self.lbl_roll = tk.Label(call, text="--", bg=DARK, fg=TEXT,
                                 font=("Segoe UI", 30, "bold"))
        self.lbl_roll.pack(side="left")
        mid = ttk.Frame(call)
        mid.pack(side="left", padx=18)
        self.lbl_phase = tk.Label(mid, text="IDLE", bg=DARK, fg=MUTED,
                                  font=("Segoe UI", 14, "bold"), anchor="w")
        self.lbl_phase.pack(anchor="w")
        self.lbl_detail = tk.Label(mid, text="", bg=DARK, fg=MUTED,
                                   font=("Segoe UI", 11), anchor="w", justify="left")
        self.lbl_detail.pack(anchor="w")

        btns = ttk.Frame(call)
        btns.pack(side="right")
        ttk.Button(btns, text="Present", command=lambda: self.manual(True)).pack(side="left")
        ttk.Button(btns, text="Absent", command=lambda: self.manual(False)).pack(side="left", padx=6)
        ttk.Button(btns, text="Undo", command=self.undo).pack(side="left")

        # --- right: roster -------------------------------------------------
        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        cols = ("roll", "name", "status", "evidence")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", selectmode="browse")
        for key, text, width, anchor in (
            ("roll", "Roll Number", 150, "w"),
            ("name", "Name", 130, "w"),
            ("status", "Status", 80, "center"),
            ("evidence", "Evidence", 190, "w"),
        ):
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, anchor=anchor)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.tag_configure("Present", foreground="#4ade80")
        self.tree.tag_configure("Absent", foreground="#fb7185")
        self.tree.tag_configure("Pending", foreground=MUTED)
        self.tree.tag_configure("current", background="#2a2a36")

        self.lbl_summary = tk.Label(right, text="no roster loaded", bg=DARK, fg=MUTED,
                                    font=("Segoe UI", 11))
        self.lbl_summary.grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        # --- status bar ----------------------------------------------------
        self.lbl_status = tk.Label(outer, text="ready", bg=PANEL, fg=MUTED, anchor="w",
                                   font=("Consolas", 10), padx=8, pady=5)
        self.lbl_status.pack(fill="x", pady=(10, 0))

    def _bind_keys(self) -> None:
        # Single keys for the two actions a teacher needs mid-call without looking.
        self.root.bind("<p>", lambda _e: self.manual(True))
        self.root.bind("<a>", lambda _e: self.manual(False))
        self.root.bind("<u>", lambda _e: self.undo())
        self.root.bind("<space>", lambda _e: self.toggle_run())
        self.root.bind("<F11>", lambda _e: self.display and self.display.toggle_fullscreen())

    # -- actions -----------------------------------------------------------
    def load_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Load roll list",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            students = load_csv(Path(path))
        except (RosterError, OSError) as exc:
            messagebox.showerror("Roll list", str(exc))
            return
        self.session.set_students(students)
        self._roster_sig = None
        self._flash(f"loaded {len(students)} students from {Path(path).name}")

    def paste_rolls(self) -> None:
        text = simpledialog.askstring(
            "Paste roll numbers",
            "Paste roll numbers separated by commas, spaces or new lines:",
            parent=self.root,
        )
        if not text:
            return
        try:
            students = parse_pasted(text)
        except RosterError as exc:
            messagebox.showerror("Roll list", str(exc))
            return
        self.session.set_students(students)
        self._roster_sig = None
        self._flash(f"loaded {len(students)} pasted roll numbers")

    def start_devices(self) -> None:
        if self._devices_started:
            return
        self._devices_started = True
        self.btn_devices.configure(text="Starting...", state="disabled")
        self.video.start()
        self.speech.start()
        self._flash("starting camera, model and speech")

    def switch_camera(self) -> None:
        if not self._devices_started:
            messagebox.showinfo("Camera", "Press Start Camera first.")
            return
        self.video.request_next_camera()
        self._flash("switching camera")

    def open_display(self) -> None:
        if self.display is not None and self.display.alive:
            self.display.focus()
            return
        self.display = RollDisplay(self.root, self.cfg, on_close=self._display_closed)
        self._flash("student display opened - F11 for fullscreen on the projector")

    def _display_closed(self) -> None:
        self.display = None

    def start_attendance(self) -> None:
        if not self.session.students:
            messagebox.showwarning("Attendance", "Load or paste roll numbers first.")
            return
        if not self._devices_started:
            messagebox.showwarning("Attendance", "Press Start Camera first.")
            return
        if not self.video.status()["ready"]:
            messagebox.showwarning("Attendance", "The camera and model are still starting.")
            return
        if self.display is None or not self.display.alive:
            self.open_display()
        self.session.start(time.monotonic())

    def pause(self) -> None:
        self.session.pause()

    def toggle_run(self) -> None:
        if self.session.phase in (Phase.IDLE, Phase.PAUSED, Phase.FINISHED):
            self.start_attendance()
        else:
            self.pause()

    def manual(self, present: bool) -> None:
        if self.session.current < 0:
            return
        self.session.mark_manual(present, time.monotonic())

    def undo(self) -> None:
        self.session.undo()

    def reset_marks(self) -> None:
        if not self.session.students:
            return
        if not messagebox.askyesno("Reset", "Clear every mark and start over?"):
            return
        self.session.reset_marks()
        self._roster_sig = None

    def export(self) -> None:
        if self.exporter is None:
            messagebox.showinfo("Export", "Export is not available in this build.")
            return
        stats = summarize(self.session.students)
        if not self.session.students:
            messagebox.showwarning("Export", "Nothing to export.")
            return
        if stats["pending"]:
            if not messagebox.askyesno(
                "Export",
                f"{stats['pending']} student(s) are still Pending and will be exported "
                "as Pending. Continue?",
            ):
                return
        try:
            runtime = {}
            try:
                st = self.video.status()
                runtime = {
                    "model": st.get("model"),
                    "camera": st.get("camera"),
                    "fps": round(st.get("fps") or 0.0, 1),
                    "capture_fps": round(st.get("capture_fps") or 0.0, 1),
                    "half": st.get("half"),
                }
            except Exception:
                pass
            paths = self.exporter.export(
                self.session.students,
                telemetry=self.session.telemetry,
                runtime=runtime,
            )
        except Exception as exc:
            messagebox.showerror("Export", str(exc))
            return
        messagebox.showinfo("Export", "Saved:\n" + "\n".join(str(p) for p in paths))

    def close(self) -> None:
        try:
            self.video.stop()
        except Exception:
            pass
        try:
            self.speech.stop()
        except Exception:
            pass
        if self.display is not None:
            self.display.close()
        self.root.destroy()

    # -- the loop ----------------------------------------------------------
    def _tick(self) -> None:
        now = time.monotonic()
        try:
            snap = self.video.snapshot()

            # Fold each frame into the tracker exactly once. Updating twice for the
            # same observation would distort the EMA time constant the detector
            # depends on.
            if snap.frame is not None and snap.frame_index != self._last_frame_index:
                self._last_frame_index = snap.frame_index
                obs = {
                    p.track_id: (p.score, p.bbox)
                    for p in snap.people
                    if p.track_id is not None
                }
                self.tracker.update(snap.at, obs)

            view = self.session.tick(now)
            self._render_video(snap)
            self._render_call(view)
            self._render_roster(view)
            self._render_status(now, snap)
            if self.display is not None and self.display.alive:
                self.display.update(view)
        except Exception as exc:  # never let a render error kill the loop
            self.lbl_status.configure(text=f"UI error: {type(exc).__name__}: {exc}")
        finally:
            self.root.after(33, self._tick)

    # -- rendering ---------------------------------------------------------
    def _render_video(self, snap) -> None:
        if snap.frame is None:
            st = self.video.status()
            if st["error"]:
                text = st["error"]
            elif not self._devices_started:
                text = "Press  Start Camera  to begin"
            else:
                text = f"starting: {st['stage']}..."
            self.video_label.configure(image="", text=text)
            return
        lw = max(160, self.video_label.winfo_width())
        lh = max(120, self.video_label.winfo_height())
        view = annotate(snap.frame, snap, self.cfg, track_states=self.tracker.states)
        h, w = view.shape[:2]
        scale = min(lw / w, lh / h)
        if scale < 1.0:
            view = cv2.resize(view, (int(w * scale), int(h * scale)),
                              interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.video_label.configure(image=self._photo, text="")

    def _render_call(self, view) -> None:
        tint = PHASE_TINT.get(view.phase, MUTED)
        self.lbl_roll.configure(text=view.roll or "--")
        label = view.phase.value
        if view.phase in (Phase.RESPONSE, Phase.STALLED) and view.window > 0:
            label = f"{view.phase.value}  {view.remaining:0.1f}s"
        self.lbl_phase.configure(text=label, fg=tint)
        bits = []
        if view.name:
            bits.append(view.name)
        if view.spoken:
            bits.append(f'says "{view.spoken}"')
        if view.best_score > 0.02:
            bits.append(f"score {view.best_score:0.2f}")
        if view.blocked_reason:
            bits.append(view.blocked_reason)
        self.lbl_detail.configure(text="   |   ".join(bits) or view.message)

    def _render_roster(self, view) -> None:
        students = self.session.students
        # Rebuilding 60 rows at 30 Hz is wasteful, so only redraw when something
        # actually changed.
        sig = (
            tuple((s.status, s.evidence) for s in students),
            self.session.current,
            len(students),
        )
        if sig == self._roster_sig:
            return
        self._roster_sig = sig

        self.tree.delete(*self.tree.get_children())
        for i, s in enumerate(students):
            tags = [s.status]
            if i == self.session.current:
                tags.append("current")
            self.tree.insert(
                "", "end", iid=str(i),
                values=(s.roll, s.name, s.status, s.evidence),
                tags=tuple(tags),
            )
        if 0 <= self.session.current < len(students):
            iid = str(self.session.current)
            self.tree.selection_set(iid)
            self.tree.see(iid)

        st = view.stats
        self.lbl_summary.configure(
            text=f"{st['total']} students   |   present {st['present']}   "
                 f"absent {st['absent']}   pending {st['pending']}"
        )

    def _render_status(self, now: float, snap) -> None:
        st = self.video.status()
        # Before the camera is started the pipeline is legitimately silent, and
        # stall_age is infinite. Rendering that as "STALLED inf s" looks like a
        # fault when nothing is wrong yet.
        if not self._devices_started:
            live = "camera not started"
        elif not st["ready"]:
            # Still loading the model or warming the GPU. The pipeline has not
            # produced a frame yet, which is not a stall - saying so would have
            # the operator hunting a fault during normal startup.
            live = "starting"
        elif st["live"]:
            live = "LIVE"
        else:
            age = st["stall_age"]
            live = "STALLED" if age == float("inf") else f"STALLED {age:.1f}s"
        cam_fps = st.get("capture_fps") or 0.0
        bits = [
            f"{st['stage']:<18}",
            f"cam {cam_fps:4.1f}fps",
            f"ai {snap.fps:4.1f}fps",
            f"{snap.infer_ms:5.1f}ms",
            f"people {len(snap.people):2d}",
            f"tracks {len(self.tracker.states):2d}",
            live,
        ]
        if self.speech.available:
            bits.append("tts")
        elif self.speech.error:
            bits.append("tts off")
        if st["error"]:
            bits.append(f"! {st['error']}")
        self.lbl_status.configure(text="   ".join(bits))
        if self._devices_started and st["ready"]:
            self.btn_devices.configure(text="Camera running", state="disabled")

    def _flash(self, text: str) -> None:
        self.session.message = text
