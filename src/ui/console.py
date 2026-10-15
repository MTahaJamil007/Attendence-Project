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

import json
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
from PIL import Image, ImageTk

from annotate import annotate
from gpu_monitor import GpuMonitor
from roster import PRESENT, RosterError, load_csv, parse_pasted, summarize
from session import Phase, Session
from ui.phone_dialog import PhoneDialog
from ui.roll_display import RollDisplay

APP_TITLE = "Smart Roll Call - Hand Raise Attendance"
# The last phone address used, so in class the teacher only has to press Connect.
LAST_CAMERA_FILE = Path(__file__).resolve().parents[2] / "data" / "last_camera.json"

DARK = "#15151a"
PANEL = "#1d1d24"
TEXT = "#e8e8ee"
MUTED = "#8b8b98"
# The right-hand panel (current call + roster) has this fixed width; the video
# takes everything else.
SIDE_WIDTH = 460

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
        self._phone_dialog: PhoneDialog | None = None
        # NVML, not Task Manager: Task Manager's default GPU graphs do not include
        # the CUDA engine, so it reads 0% while the model runs. See gpu_monitor.py.
        self.gpu = GpuMonitor()

        root.title(APP_TITLE)
        # Fit the screen: a window taller than the screen hides its bottom edge
        # (the call controls and status bar) behind the taskbar.
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        w, h = min(1560, sw - 40), min(920, sh - 90)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 2 - 20)}")
        root.minsize(1100, 640)
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
        style.configure("Group.TLabel", background=DARK, foreground=MUTED,
                        font=("Segoe UI", 8, "bold"))
        style.configure("Accent.TButton", foreground="#0d0d10", background="#60a5fa")
        style.map("Accent.TButton", background=[("active", "#93c5fd")])
        style.configure("Present.TButton", foreground="#0d0d10", background="#4ade80")
        style.map("Present.TButton", background=[("active", "#86efac")])
        style.configure("Absent.TButton", foreground="#0d0d10", background="#fb7185")
        style.map("Absent.TButton", background=[("active", "#fda4af")])
        style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                        foreground=TEXT, rowheight=24, borderwidth=0)
        style.configure("Treeview.Heading", background=DARK, foreground=MUTED)
        style.map("Treeview", background=[("selected", "#2f4f7a")])

    def _build_ui(self) -> None:
        """Lay out the window so that nothing can push anything else off screen.

        Measured on the previous layout: one long message in the call panel (the
        seat-memory explanation) grew the window from 1560 to 1920 px wide and
        pushed Present / Absent / Undo out of view - exactly when the teacher was
        being asked to press them - while the video area changed size with every
        status change. Here every region has a size decided by the window, never
        by its contents:

        * the video is a Canvas that requests a token size and draws the frame
          letterboxed into whatever space it is given;
        * the right panel has a fixed width, and its message text wraps inside
          it with a fixed number of lines;
        * the status bar requests no width and clips its text.

        The call panel sits at the top of the right column so the manual
        controls stay visible beside the video, and the video gets the full
        height of the window.
        """
        outer = ttk.Frame(self.root, padding=(10, 8, 10, 8))
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        # --- toolbar: grouped by when in the lesson each is used ------------
        bar = ttk.Frame(outer)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        def group(caption: str) -> ttk.Frame:
            box = ttk.Frame(bar)
            box.pack(side="left", padx=(0, 14))
            ttk.Label(box, text=caption, style="Group.TLabel").pack(anchor="w")
            row = ttk.Frame(box)
            row.pack(anchor="w")
            return row

        g = group("ROLL LIST")
        self.btn_load = ttk.Button(g, text="Load CSV", command=self.load_csv)
        self.btn_load.pack(side="left")
        ttk.Button(g, text="Paste Rolls", command=self.paste_rolls).pack(side="left", padx=(4, 0))

        g = group("CAMERA")
        self.btn_devices = ttk.Button(g, text="Start Camera", command=self.start_devices)
        self.btn_devices.pack(side="left")
        ttk.Button(g, text="Switch Camera", command=self.switch_camera).pack(side="left", padx=(4, 0))
        ttk.Button(g, text="Phone Camera", command=self.phone_camera).pack(side="left", padx=(4, 0))

        g = group("ATTENDANCE")
        self.btn_start = ttk.Button(g, text="Start Attendance", command=self.start_attendance,
                                    style="Accent.TButton")
        self.btn_start.pack(side="left")
        self.btn_pause = ttk.Button(g, text="Pause", command=self.pause)
        self.btn_pause.pack(side="left", padx=(4, 0))
        ttk.Button(g, text="Roll Screen", command=self.open_display).pack(side="left", padx=(4, 0))

        g = group("RECORDS")
        ttk.Button(g, text="Export", command=self.export).pack(side="left")
        ttk.Button(g, text="Reset Marks", command=self.reset_marks).pack(side="left", padx=(4, 0))

        # --- body: video | side panel --------------------------------------
        body = ttk.Frame(outer)
        body.grid(row=1, column=0, sticky="nsew")
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=SIDE_WIDTH)

        # A Canvas, not a Label: a Label holding an image asks for the image's
        # size, so each fitted frame made it a little bigger than the last.
        self.video_canvas = tk.Canvas(body, bg="#000000", highlightthickness=0, bd=0,
                                      width=320, height=180)
        self.video_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self._video_item = self.video_canvas.create_image(0, 0, anchor="center")
        self._video_text = self.video_canvas.create_text(
            0, 0, text="Camera not started", fill=MUTED, font=("Segoe UI", 14),
            justify="center")

        side = ttk.Frame(body, width=SIDE_WIDTH)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)

        # --- current call ---------------------------------------------------
        call = tk.Frame(side, bg=PANEL, padx=14, pady=10)
        call.grid(row=0, column=0, sticky="ew")
        call.columnconfigure(0, weight=1)
        tk.Label(call, text="CURRENT CALL", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        self.lbl_roll = tk.Label(call, text="--", bg=PANEL, fg=TEXT, anchor="w",
                                 font=("Segoe UI", 28, "bold"))
        self.lbl_roll.grid(row=1, column=0, sticky="ew")
        self.lbl_phase = tk.Label(call, text="IDLE", bg=PANEL, fg=MUTED, anchor="w",
                                  font=("Segoe UI", 14, "bold"))
        self.lbl_phase.grid(row=2, column=0, sticky="ew")
        # Fixed height in text lines and wrapped to the panel: a long message
        # adds lines inside this box instead of widening the window.
        self.lbl_detail = tk.Label(call, text="", bg=PANEL, fg=MUTED, anchor="nw",
                                   justify="left", font=("Segoe UI", 10), height=4,
                                   wraplength=SIDE_WIDTH - 40)
        self.lbl_detail.grid(row=3, column=0, sticky="ew", pady=(2, 8))
        call.bind("<Configure>", lambda e: self.lbl_detail.configure(
            wraplength=max(120, e.width - 32)))

        btns = tk.Frame(call, bg=PANEL)
        btns.grid(row=4, column=0, sticky="ew")
        for i in range(3):
            btns.columnconfigure(i, weight=1, uniform="manual")
        ttk.Button(btns, text="Present  (P)", style="Present.TButton",
                   command=lambda: self.manual(True)).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Absent  (A)", style="Absent.TButton",
                   command=lambda: self.manual(False)).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(btns, text="Undo  (U)", command=self.undo).grid(row=0, column=2, sticky="ew")

        # --- roster ---------------------------------------------------------
        roster = ttk.Frame(side)
        roster.grid(row=1, column=0, sticky="nsew", pady=(10, 0))
        roster.rowconfigure(0, weight=1)
        roster.columnconfigure(0, weight=1)

        cols = ("roll", "name", "status", "evidence")
        self.tree = ttk.Treeview(roster, columns=cols, show="headings", selectmode="browse")
        for key, text, width, anchor, stretch in (
            ("roll", "Roll Number", 118, "w", False),
            ("name", "Name", 110, "w", True),
            ("status", "Status", 66, "center", False),
            ("evidence", "Evidence", 120, "w", True),
        ):
            self.tree.heading(key, text=text)
            self.tree.column(key, width=width, minwidth=50, anchor=anchor, stretch=stretch)
        self.tree.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(roster, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        sb.grid(row=0, column=1, sticky="ns")
        self.tree.tag_configure("Present", foreground="#4ade80")
        self.tree.tag_configure("Absent", foreground="#fb7185")
        self.tree.tag_configure("Pending", foreground=MUTED)
        self.tree.tag_configure("current", background="#2a2a36")

        self.lbl_summary = tk.Label(roster, text="no roster loaded", bg=DARK, fg=MUTED,
                                    font=("Segoe UI", 11), anchor="w")
        self.lbl_summary.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        # --- status bar ----------------------------------------------------
        # width=1: request no width at all, so a long status line is clipped at
        # the window edge rather than widening the window.
        self.lbl_status = tk.Label(outer, text="ready", bg=PANEL, fg=MUTED, anchor="w",
                                   font=("Consolas", 10), padx=8, pady=5, width=1)
        self.lbl_status.grid(row=2, column=0, sticky="ew", pady=(8, 0))

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

    def phone_camera(self) -> None:
        if self._phone_dialog is not None:
            try:
                if self._phone_dialog.win.winfo_exists():
                    self._phone_dialog.win.lift()
                    return
            except Exception:
                pass
        initial = self._last_phone() or ""
        self._phone_dialog = PhoneDialog(
            self.root, initial, on_connect=self._connect_phone, on_local=self._use_local_camera
        )

    def _connect_phone(self, address: str) -> None:
        self._save_last_phone(address)
        if self._devices_started:
            self.video.request_source(address)
            self._flash(f"connecting to phone camera {address}")
        else:
            self.video.set_source(address)
            self.start_devices()

    def _use_local_camera(self) -> None:
        if self._devices_started:
            self.video.request_source("usb")
            self._flash("switching to the laptop / USB camera")
        else:
            self.video.set_source("")
            self._flash("the laptop / USB camera will be used")

    @staticmethod
    def _last_phone() -> str:
        try:
            return str(json.loads(LAST_CAMERA_FILE.read_text(encoding="utf-8")).get("phone", ""))
        except Exception:
            return ""

    @staticmethod
    def _save_last_phone(address: str) -> None:
        try:
            LAST_CAMERA_FILE.parent.mkdir(parents=True, exist_ok=True)
            LAST_CAMERA_FILE.write_text(json.dumps({"phone": address}), encoding="utf-8")
        except Exception:
            pass

    def open_display(self) -> None:
        """The Roll Screen button: open the student display, or bring it forward."""
        if self.display is not None and self.display.alive:
            self.display.focus()
            return
        self._create_display()
        if self.display.on_projector:
            self._flash("student display is on the projector - F11 there for fullscreen")
        else:
            self._flash("student display opened - F11 for fullscreen")

    def _create_display(self) -> None:
        self.display = RollDisplay(self.root, self.cfg, on_close=self._display_closed)
        # With a projector connected, the student display goes there.
        self.display.place_on_secondary_monitor()

    def _open_display_for_attendance(self) -> None:
        """Start Attendance opens the student display without covering the console.

        Before, it opened in front of the console - on a single screen right over
        the camera feed - and took keyboard focus, so P / A / U went to the wrong
        window until the teacher clicked back. Now: on a projector if there is
        one; otherwise it is opened behind the console (Roll Screen or the
        taskbar brings it up). Either way the console keeps focus.
        """
        if self.display is not None and self.display.alive:
            return
        self._create_display()
        if self.display.on_projector:
            self._flash("student display is on the projector - F11 there for fullscreen")
        else:
            self.display.send_behind(self.root)
            self._flash("no projector found: student display is behind this window "
                        "(press Roll Screen to show it)")
        self.root.lift()
        self.root.focus_force()

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
            self._open_display_for_attendance()
        self.session.start(time.monotonic())

    def pause(self) -> None:
        self.session.pause(time.monotonic())

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
        if getattr(self, "_closed", False):
            return
        self._closed = True
        try:
            self.video.stop()
        except Exception:
            pass
        try:
            self.speech.stop()
        except Exception:
            pass
        if self.display is not None:
            try:
                self.display.close()
            except Exception:
                pass
        try:
            self.root.destroy()
        except Exception:
            pass

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
            cv = self.video_canvas
            cv.itemconfigure(self._video_item, image="")
            cv.coords(self._video_text, cv.winfo_width() / 2, cv.winfo_height() / 2)
            cv.itemconfigure(self._video_text, text=text,
                             width=max(200, cv.winfo_width() - 60))
            return
        cv = self.video_canvas
        lw = max(160, cv.winfo_width())
        lh = max(120, cv.winfo_height())
        flag = self.session.flag if self.session.phase is Phase.AMBIGUOUS else None
        view = annotate(snap.frame, snap, self.cfg, track_states=self.tracker.states,
                        seats=list(self.gate.seats.values()), flag=flag)
        h, w = view.shape[:2]
        # Shrink to fit, keeping the aspect ratio; never enlarge (enlarging costs
        # a resize and a bigger image transfer every frame for no extra detail).
        # This only changes what is drawn - the canvas's size is set by the window.
        scale = min(lw / w, lh / h)
        if scale < 1.0:
            view = cv2.resize(view, (max(1, int(w * scale)), max(1, int(h * scale))),
                              interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        cv.coords(self._video_item, lw / 2, lh / 2)
        cv.itemconfigure(self._video_item, image=self._photo)
        cv.itemconfigure(self._video_text, text="")

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
            st["camera"],
            f"cam {cam_fps:4.1f}fps",
            f"ai {snap.fps:4.1f}fps",
            f"{snap.infer_ms:5.1f}ms",
            f"people {len(snap.people):2d}",
            f"tracks {len(self.tracker.states):2d}",
            live,
        ]
        if st.get("device"):
            bits.append(st["device"])
        gpu = self.gpu.text(now)
        if gpu:
            bits.append(gpu)
        if self.speech.available:
            bits.append(f"tts {getattr(self.speech, 'backend', '')}".rstrip())
        elif self.speech.error:
            bits.append("tts off")
        if st["error"]:
            bits.append(f"! {st['error']}")
        self.lbl_status.configure(text="   ".join(bits))
        if self._devices_started and st["ready"]:
            self.btn_devices.configure(text="Camera running", state="disabled")
        elif self._devices_started and st["stage"] == "no camera":
            self.btn_devices.configure(text="No camera", state="disabled")

    def _flash(self, text: str) -> None:
        self.session.message = text
