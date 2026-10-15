"""Capture + pose inference pipeline, with liveness reporting.

Two threads, deliberately:

* **Capture** writes each decoded frame into a single-slot buffer, overwriting
  whatever was there. That overwrite *is* the frame-dropping policy: inference
  always runs on the newest view of the room. A queue would instead let
  inference fall behind and score a frame from a second ago, which silently
  shifts every raise outside its own response window.
* **Inference** owns the Ultralytics model exclusively. ``model.track`` carries
  tracker state on the model object and is not thread-safe, so exactly one
  thread ever touches it.

Both threads stamp a monotonic clock. ``is_live()`` exposes those stamps, and the
session state machine is required to consult it before recording an absence. The
predecessor had no such check: when its CPU-only inference stalled for 20-50
seconds, the state machine kept running on its own timer and wrote "Timeout: no
valid hand" for every remaining student. Three of its eight recorded runs were
destroyed that way. An infrastructure failure must never become an attendance
record.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from appearance import torso_signature
from camera import CameraInfo, discover_cameras, open_source
from netcam import is_network_source

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from liveness import LivenessMonitor
from raise_detector import PersonScore, RaiseScorer


@dataclass
class PersonObservation:
    track_id: int | None
    bbox: tuple[float, float, float, float]
    box_conf: float
    keypoints: list[tuple[float, float, float]]
    score: PersonScore


@dataclass
class FrameResult:
    frame: np.ndarray | None = None
    people: list[PersonObservation] = field(default_factory=list)
    at: float = 0.0
    infer_ms: float = 0.0
    fps: float = 0.0
    frame_index: int = 0

    @property
    def raised_count(self) -> int:
        return sum(1 for p in self.people if p.score.score > 0.0)


class VideoWorker:
    """Owns the camera and the pose model; publishes the latest scored frame."""

    def __init__(self, cfg, runtime_report=None):
        self.cfg = cfg
        self.runtime = runtime_report

        mcfg = cfg.section("model")
        self.weights: str = mcfg["weights"]
        self.imgsz: int = int(mcfg["imgsz"])
        self.device: str = mcfg["device"]
        self.person_conf: float = float(mcfg["person_conf"])
        self.min_person_h: float = float(mcfg["min_person_height_px"])
        self.max_det: int = int(mcfg["max_det"])
        # A project-relative tracker yaml (our GMC-free BoT-SORT) must resolve no
        # matter what the working directory is; a bare name like "bytetrack.yaml"
        # is left alone so ultralytics resolves it from its own cfg directory.
        tracker = str(mcfg["tracker"])
        candidate = (PROJECT_ROOT / tracker) if not Path(tracker).is_absolute() else Path(tracker)
        self.tracker: str = str(candidate) if candidate.exists() else tracker
        # Only request FP16 if the runtime check actually proved it works.
        want_half = bool(mcfg["half"])
        self.half: bool = want_half and bool(
            getattr(runtime_report, "fp16_ok", False)
        )
        # ultralytics 8.4 replaced half=True with quantize=16 (8 = INT8,
        # None = FP32). Passing `half` still works but emits a deprecation
        # warning on every single call, which floods the log at 40 FPS.
        self.quantize: int | None = 16 if self.half else None

        self.flip: bool = bool(cfg.get("camera.flip_horizontal", False))
        rot = int(cfg.get("camera.rotate", 0) or 0) % 360
        # A phone may be mounted in portrait; turn its picture upright before
        # inference, which expects people the right way up.
        self.rotate_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }.get(rot)
        # Camera to use: a local index, or a phone address / URL. Empty means
        # camera.index, then automatic discovery of local cameras.
        self.source: str = str(cfg.get("camera.source", "") or "").strip()
        self.kp_conf: float = float(mcfg["keypoint_conf"])
        self.scorer = RaiseScorer(cfg)

        # --- published state (guarded by _lock) ---------------------------
        self._lock = threading.Lock()
        self._latest = FrameResult()
        self._raw_frame: np.ndarray | None = None
        self._raw_at: float = 0.0

        self.stage: str = "idle"
        self.error: str | None = None
        self.ready: bool = False
        self.model_name: str = ""
        # Where inference really runs, read back from the predictor after the
        # warm-up rather than assumed from the config.
        self.infer_device: str = ""
        self._model_loaded: bool = False

        self.last_frame_at: float = 0.0
        self.last_infer_at: float = 0.0
        self.fps: float = 0.0
        self.capture_fps: float = 0.0
        self.infer_ms: float = 0.0

        self.cameras: list[CameraInfo] = []
        self.active: CameraInfo | None = None
        self._switch_to: int | str | None = None
        self._switch_next: bool = False

        self._stop = threading.Event()
        # Separate event so a camera switch can retire the capture thread without
        # signalling the inference thread to exit.
        self._cap_stop = threading.Event()
        self._cap = None
        self._cap_thread: threading.Thread | None = None
        self._inf_thread: threading.Thread | None = None
        self._frame_index = 0

        self.stall_threshold: float = float(
            cfg.get("session.stall_threshold_sec", 0.75)
        )
        # One implementation of the liveness rule, shared with the session state
        # machine, so the two can never disagree about what "stalled" means.
        self.infer_liveness = LivenessMonitor(self.stall_threshold, "inference")
        self.capture_liveness = LivenessMonitor(self.stall_threshold, "capture")
        self.max_consecutive_failures: int = int(
            cfg.get("model.max_consecutive_failures", 10)
        )
        self.warmup_sec: float = float(cfg.get("model.warmup_sec", 3.0))
        self.warmup_iters: int = 0

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if self._inf_thread is not None:
            return
        self._stop.clear()
        self._inf_thread = threading.Thread(
            target=self._infer_loop, name="pose-infer", daemon=True
        )
        self._inf_thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._cap_stop.set()
        for t in (self._cap_thread, self._inf_thread):
            if t is not None and t.is_alive():
                t.join(timeout=3.0)
        self._cap_thread = None
        self._inf_thread = None
        # Release the device, or the camera stays locked until the process exits
        # and the next run reports "no usable camera".
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.ready = False
        self._set_stage("stopped")

    # -- liveness ----------------------------------------------------------
    def is_live(self, now: float | None = None) -> bool:
        """True when inference produced a result recently enough to trust.

        The session must gate every automatic Absent on this.
        """
        now = time.monotonic() if now is None else now
        return self.ready and self.infer_liveness.is_live(now)

    def stall_age(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        return self.infer_liveness.age(now)

    def may_record_absence(self, now: float | None = None) -> tuple[bool, str]:
        """Gate for the session state machine before it writes an Absent."""
        now = time.monotonic() if now is None else now
        if not self.ready:
            return False, f"perception pipeline not ready ({self.stage})"
        return self.infer_liveness.may_record_absence(now)

    # -- camera switching --------------------------------------------------
    def set_source(self, source: int | str | None) -> None:
        """Choose the camera before start(): an index or a phone address/URL.

        None, "" or "usb" mean the local camera (camera.index, else discovery).
        """
        text = "" if source is None else str(source).strip()
        self.source = "" if text.lower() in ("usb", "auto") else text

    def request_camera(self, index: int) -> None:
        self._switch_to = index

    def request_source(self, source: int | str) -> None:
        """Switch to a local index or a phone address while running.

        Before start() this only records the choice. While running, the switch is
        done on the inference thread, and the current camera keeps running until
        the new one has actually delivered a frame - a mistyped phone address
        must not leave the class with no camera at all.
        """
        if self._inf_thread is None:
            self.set_source(source)
            return
        self._switch_to = source

    def request_next_camera(self) -> None:
        self._switch_next = True

    # -- published snapshot ------------------------------------------------
    def snapshot(self) -> FrameResult:
        with self._lock:
            return self._latest

    def status(self) -> dict:
        return {
            "stage": self.stage,
            "ready": self.ready,
            "error": self.error,
            "fps": self.fps,
            "capture_fps": self.capture_fps,
            "infer_ms": self.infer_ms,
            "model": self.model_name,
            "half": self.half,
            "device": self.infer_device,
            "camera": self.active.label() if self.active else "none",
            "camera_source": self.active.source if self.active else "",
            "cameras": [c.label() for c in self.cameras],
            "live": self.is_live(),
            "stall_age": self.stall_age(),
        }

    def _set_stage(self, stage: str) -> None:
        self.stage = stage

    # -- capture thread ----------------------------------------------------
    def _capture_loop(self, cap, cap_stop: threading.Event, network: bool = False) -> None:
        """Decode frames into a single overwritable slot."""
        misses = 0
        down_since = 0.0
        while not (self._stop.is_set() or cap_stop.is_set()):
            ok, frame = cap.read()
            if not ok or frame is None or frame.size == 0:
                misses += 1
                if network:
                    # A phone on weak Wi-Fi drops out and comes back. Never give
                    # up on it: the stream reconnects itself inside read(), and
                    # meanwhile no inference results arrive, so the session sees
                    # a stall and suspends marking instead of recording absences.
                    # A stream read only fails when the link is actually down
                    # (a slow phone just makes read() wait), so say so at once.
                    if down_since == 0.0:
                        down_since = time.monotonic()
                    why = getattr(cap, "last_error", "") or "no frames"
                    self.error = f"phone camera not responding ({why}) - reconnecting"
                    self._set_stage("phone reconnecting")
                    continue
                # Tolerate brief hiccups; give up only on a sustained failure so
                # the inference loop can surface a real camera fault.
                if misses > 60:
                    self.error = "camera stopped delivering frames"
                    self._set_stage("camera lost")
                    return
                time.sleep(0.005)
                continue
            misses = 0
            if down_since:
                down_since = 0.0
                if self.stage == "phone reconnecting":
                    self.error = None
                    self._set_stage("running" if self._model_loaded else self.stage)
            if self.rotate_code is not None:
                frame = cv2.rotate(frame, self.rotate_code)
            if self.flip:
                frame = frame[:, ::-1].copy()
            now = time.monotonic()
            with self._lock:
                self._raw_frame = frame
                self._raw_at = now
            # Capture rate, measured separately from inference rate. Without this
            # you cannot tell a slow GPU from a slow camera - and a 15 FPS webcam
            # looks identical to an overloaded model in a single combined number.
            if self.last_frame_at > 0.0:
                inst = 1.0 / max(1e-6, now - self.last_frame_at)
                self.capture_fps = (
                    inst if self.capture_fps <= 0.0
                    else self.capture_fps * 0.9 + inst * 0.1
                )
            self.last_frame_at = now
            self.capture_liveness.mark(now)

    def _take_frame(self) -> tuple[np.ndarray | None, float]:
        with self._lock:
            frame, at = self._raw_frame, self._raw_at
            self._raw_frame = None  # consume, so we never infer the same frame twice
        return frame, at

    # -- inference thread --------------------------------------------------
    def _infer_loop(self) -> None:
        try:
            # Camera first, model second. Probing camera indices across three
            # Windows backends can take many seconds, and the GPU warm-up inside
            # _load_model only holds its clocks for a moment - doing the slow scan
            # afterwards would let them fall back to idle before the first real
            # frame, wasting the warm-up entirely.
            attached = self._attach_camera()
            no_camera_error = self.error
            model = self._load_model()
            if model is None:
                return
            self._model_loaded = True
            if attached:
                self.ready = True
                self._set_stage("running")
            else:
                # Keep the loaded model and wait for a camera - typically the
                # teacher connecting a phone - instead of ending the pipeline and
                # forcing a restart of the whole application.
                self.error = no_camera_error
                self._set_stage("no camera")
            fps_ema = 0.0
            last_at = 0.0
            consecutive_failures = 0

            while not self._stop.is_set():
                if self._handle_switch_request():
                    continue
                if self._cap is None:
                    time.sleep(0.05)
                    continue

                frame, at = self._take_frame()
                if frame is None:
                    time.sleep(0.002)
                    continue

                t0 = time.perf_counter()
                try:
                    people = self._infer_frame(model, frame)
                    consecutive_failures = 0
                except Exception as exc:
                    # A transient fault - a one-off CUDA OOM, a driver hiccup -
                    # must not end the session. Let it pass and keep going; the
                    # liveness monitor will report the gap, and the session will
                    # suspend marking on its own rather than inventing absences.
                    # Only a sustained failure is fatal.
                    consecutive_failures += 1
                    self.error = f"{type(exc).__name__}: {exc}"
                    if consecutive_failures >= self.max_consecutive_failures:
                        self._set_stage("inference failed")
                        self.ready = False
                        return
                    self._set_stage(f"inference error ({consecutive_failures}), retrying")
                    time.sleep(0.05)
                    continue
                infer_ms = (time.perf_counter() - t0) * 1000.0
                now = time.monotonic()
                if self.stage.startswith("inference error"):
                    self.error = None
                    self._set_stage("running")

                # FPS as an EMA over inter-result intervals.
                if last_at > 0.0:
                    inst = 1.0 / max(1e-6, now - last_at)
                    fps_ema = inst if fps_ema <= 0.0 else fps_ema * 0.85 + inst * 0.15
                last_at = now

                self._frame_index += 1
                result = FrameResult(
                    frame=frame,
                    people=people,
                    at=now,
                    infer_ms=infer_ms,
                    fps=fps_ema,
                    frame_index=self._frame_index,
                )
                with self._lock:
                    self._latest = result
                self.fps = fps_ema
                self.infer_ms = infer_ms
                self.last_infer_at = now
                self.infer_liveness.mark(now)

        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self._set_stage("inference failed")
            self.ready = False

    def _load_model(self):
        self._set_stage("loading pose model")
        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.error = f"cannot import ultralytics: {exc}"
            self._set_stage("model load failed")
            return None
        try:
            model = YOLO(self.weights)
            self.model_name = self.weights
            # Sustained warm-up, not a single call.
            #
            # One inference is enough to pay for CUDA kernel selection and cuDNN
            # autotuning, but not to raise the GPU's clocks. Measured on this
            # hardware: the SM clock idles at 300 MHz and only reaches ~1530 MHz
            # (of a 2100 MHz maximum) after roughly 8 seconds of continuous load,
            # with power going 9 W -> 32 W -> 79 W. A cold GPU runs inference at
            # roughly half speed, and without this loop that penalty would land
            # squarely on the first student called.
            self._set_stage("warming up GPU")
            # Warm up on the camera's real frame shape, through the same track()
            # call the live loop makes. Measured: warming with predict() on a
            # square dummy left the first real 16:9 frame taking 2.06 s - tracker
            # set-up plus kernels for the letterboxed input shape - so the
            # pipeline reported ready and then stalled for two seconds.
            if self.active is not None and self.active.width and self.active.height:
                h, w = self.active.height, self.active.width
                if self.rotate_code in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE):
                    h, w = w, h
            else:
                h = w = self.imgsz
            dummy = np.zeros((h, w, 3), dtype=np.uint8)
            kw = dict(
                persist=True,
                tracker=self.tracker,
                imgsz=self.imgsz,
                device=self.device,
                quantize=self.quantize,
                conf=self.person_conf,
                max_det=self.max_det,
                classes=[0],
                verbose=False,
            )
            deadline = time.monotonic() + self.warmup_sec
            iters = 0
            while True:
                # An empty frame has no detections, so no tracks are created.
                model.track(dummy, **kw)
                iters += 1
                if time.monotonic() >= deadline or self._stop.is_set():
                    break
            self.warmup_iters = iters
            predictor = getattr(model, "predictor", None)
            self.infer_device = str(getattr(predictor, "device", "") or self.device)
            return model
        except Exception as exc:
            self.error = f"cannot load model {self.weights}: {exc}"
            self._set_stage("model load failed")
            return None

    def _attach_camera(self) -> bool:
        if self.source:
            self._set_stage(
                "connecting to phone" if is_network_source(self.source) else "opening camera"
            )
            cap, info, err = open_source(self.source, self.cfg)
            if cap is not None and info is not None:
                if not info.is_network:
                    self.cameras = [info]
                self._begin_capture(cap, info)
                return True
            self.error = err
            self._set_stage("no camera")
            return False

        self._set_stage("scanning cameras")
        want = int(self.cfg.get("camera.index", -1))

        if want >= 0:
            cap, info, err = open_source(want, self.cfg)
            if cap is not None and info is not None:
                self.cameras = [info]
                self._begin_capture(cap, info)
                return True
            self.error = err
            self._set_stage("no camera")
            return False

        self.cameras = discover_cameras(self.cfg)
        if not self.cameras:
            self.error = (
                "the AI runtime is working but Windows returned no usable camera. "
                "Close Camera, Zoom or Teams if they are holding the device, then retry."
            )
            self._set_stage("no camera")
            return False

        info = self.cameras[0]
        from camera import open_camera

        cap, opened = open_camera(info.index, self.cfg, info.backend)
        if cap is None or opened is None:
            self.error = f"camera {info.index} was detected but could not be reopened"
            self._set_stage("no camera")
            return False
        self._begin_capture(cap, opened)
        return True

    def _begin_capture(self, cap, info: CameraInfo) -> None:
        self._cap = cap
        self.active = info
        self.source = info.source or str(info.index)
        self._cap_stop = threading.Event()
        self._cap_thread = threading.Thread(
            target=self._capture_loop,
            args=(cap, self._cap_stop, info.is_network),
            name="camera-capture",
            daemon=True,
        )
        self._cap_thread.start()
        if self._model_loaded:
            self.ready = True
            self.error = None
            self._set_stage("running")

    def _end_capture(self) -> None:
        """Retire the capture thread and release its device."""
        old_cap = self._cap
        self._cap_stop.set()
        if self._cap_thread is not None:
            self._cap_thread.join(timeout=4.0)
        self._cap_thread = None
        self._cap = None
        if old_cap is not None:
            try:
                old_cap.release()
            except Exception:
                pass

    def _handle_switch_request(self) -> bool:
        target: int | str | None = None
        if self._switch_to is not None:
            target = self._switch_to
            self._switch_to = None
        elif self._switch_next:
            self._switch_next = False
            if not self.cameras or (self.active is not None and self.active.is_network):
                # On a phone, or nothing found at start-up: "next" means back to
                # a local camera, so look for them now.
                self._set_stage("scanning cameras")
                self.cameras = discover_cameras(self.cfg)
            if self.cameras:
                idxs = [c.index for c in self.cameras]
                pos = -1
                if self.active is not None and not self.active.is_network:
                    try:
                        pos = idxs.index(self.active.index)
                    except ValueError:
                        pos = -1
                target = idxs[(pos + 1) % len(idxs)]
            elif self.active is None:
                self.error = "no local camera found"
                self._set_stage("no camera")
        if target is None:
            return False
        if isinstance(target, str) and target.strip().lower() in ("usb", "auto"):
            self._set_stage("scanning cameras")
            self.cameras = discover_cameras(self.cfg)
            if not self.cameras:
                self.error = "no local camera found"
                self._set_stage("running" if self._cap is not None else "no camera")
                return False
            target = self.cameras[0].index
        if self.active is not None:
            if self.active.is_network and str(target).strip() == self.active.source:
                return False
            if not self.active.is_network and str(target).strip() == str(self.active.index):
                return False

        prev_stage = self.stage
        self._set_stage(
            "connecting to phone" if is_network_source(target) else "opening camera"
        )
        # Open the new camera first, while the old one keeps running. Only when
        # it has produced a frame is the old one retired - a phone that cannot be
        # reached leaves the current camera in place, with the reason shown.
        cap, info, err = open_source(target, self.cfg)
        if cap is None and not is_network_source(target) and self._cap is not None:
            # Some USB controllers cannot run two webcams at once: free the old
            # one and try again, putting it back if the new one still fails.
            old = self.active
            self._end_capture()
            cap, info, err = open_source(target, self.cfg)
            if cap is None and old is not None:
                back, back_info, _ = open_source(old.source or old.index, self.cfg)
                if back is not None:
                    self._begin_capture(back, back_info)
        if cap is None or info is None:
            self.error = f"could not switch camera - {err}"
            if self._cap is not None:
                self._set_stage(prev_stage if prev_stage != "connecting to phone" else "running")
            else:
                self._set_stage("no camera")
            return False
        if self._cap is not None:
            self._end_capture()
        self._begin_capture(cap, info)
        return True

    # -- scoring one frame -------------------------------------------------
    def _infer_frame(self, model, frame: np.ndarray) -> list[PersonObservation]:
        results = model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            imgsz=self.imgsz,
            device=self.device,
            quantize=self.quantize,
            conf=self.person_conf,
            max_det=self.max_det,
            classes=[0],
            verbose=False,
        )
        if not results:
            return []
        res = results[0]
        if res.boxes is None or res.keypoints is None or len(res.boxes) == 0:
            return []

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        ids = res.boxes.id
        ids = ids.cpu().numpy().astype(int) if ids is not None else None
        kps = res.keypoints.data.cpu().numpy()  # (N, 17, 3) = x, y, visibility
        # Some export paths emit (N, 17, 2) with no visibility channel. Synthesize
        # full confidence rather than crashing on the unpack below; the scorer's
        # confidence floor then simply never trips for this source.
        if kps.ndim == 3 and kps.shape[2] == 2:
            kps = np.concatenate([kps, np.ones((*kps.shape[:2], 1), dtype=kps.dtype)], axis=2)
        if kps.ndim != 3 or kps.shape[1] < 17 or kps.shape[2] < 3:
            return []

        people: list[PersonObservation] = []
        for i in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[i])
            height = y2 - y1
            # Too few pixels on target for keypoints to mean anything. Scoring
            # them anyway is how the predecessor generated noise from the back row.
            if height < self.min_person_h:
                continue
            kp = [(float(a), float(b), float(c)) for a, b, c in kps[i]]
            score = self.scorer.score(kp, height)
            # Clothing colour for the seat check (gating Rule D). Only people with
            # a hand at least partly up are ever compared, so only they pay for it.
            if score.score > 0.0:
                try:
                    score.signature = torso_signature(frame, kp, self.kp_conf)
                except Exception:
                    score.signature = None
            people.append(
                PersonObservation(
                    track_id=int(ids[i]) if ids is not None else None,
                    bbox=(x1, y1, x2, y2),
                    box_conf=float(confs[i]),
                    keypoints=kp,
                    score=score,
                )
            )
        return people
