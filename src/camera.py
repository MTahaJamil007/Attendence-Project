"""Windows camera discovery and opening.

Kept close to the predecessor's approach, which was sound: probe indices across
several OpenCV backends, because no single Windows backend opens every USB
webcam. DirectShow is tried first - it opens in well under a second, whereas
Media Foundation can stall for several seconds per probe, and a slow scan looks
like a hang to the user.

A camera counts as working only when it actually yields a decoded frame.
``isOpened()`` returns True for devices that then fail on every ``read()``,
which is exactly the case that gets misdiagnosed as a model problem.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Silence OpenCV's per-probe warnings for camera indices that do not exist.
# Must be set before cv2 is imported.
os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")

import cv2  # noqa: E402

try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
except Exception:
    pass


@dataclass
class CameraInfo:
    index: int
    backend: str
    width: int
    height: int
    # Empty for a local (USB / built-in) camera; the stream URL for a phone.
    source: str = ""

    @property
    def is_network(self) -> bool:
        return bool(self.source)

    def label(self) -> str:
        if self.source:
            from netcam import describe

            return f"Phone {describe(self.source)} ({self.width}x{self.height})"
        return f"Camera {self.index} ({self.backend}, {self.width}x{self.height})"


_BACKENDS: list[tuple[str, int]] = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("ANY", cv2.CAP_ANY),
]


def _configure(cap: "cv2.VideoCapture", cfg) -> None:
    cam = cfg.section("camera")
    # FOURCC must be set BEFORE the resolution on Windows, or the driver picks an
    # uncompressed format first and then clamps the resolution to fit the USB
    # bandwidth. MJPEG is what lets a USB 2.0 webcam deliver 1080p at 30 FPS at
    # all: uncompressed YUY2 at 1920x1080x30 needs ~1.5 Gbit/s, which USB 2.0
    # cannot carry, so the driver silently drops the frame rate instead.
    fourcc = cam.get("fourcc") or ""
    if fourcc:
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc[:4]))
        except Exception:
            pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam["height"]))
    fps = int(cam.get("fps_request", 0) or 0)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    # A one-frame buffer keeps latency low. Without it the driver queues frames
    # and inference ends up scoring a view of the room from a second ago, which
    # silently shifts every raise outside its own response window.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(cam.get("buffer_size", 1)))
    except Exception:
        pass


def open_camera(
    index: int, cfg, backend_name: str | None = None
) -> tuple["cv2.VideoCapture | None", CameraInfo | None]:
    """Open one camera index, trying each backend until a frame decodes."""
    candidates = _BACKENDS
    if backend_name:
        candidates = [b for b in _BACKENDS if b[0] == backend_name] or _BACKENDS

    for name, api in candidates:
        cap = None
        try:
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                cap.release()
                continue
            _configure(cap, cfg)

            # Some drivers return an empty first frame while they warm up, so
            # give the device a few attempts before writing it off.
            frame = None
            for _ in range(5):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    break
                frame = None
            if frame is None:
                cap.release()
                continue

            h, w = frame.shape[:2]
            return cap, CameraInfo(index=index, backend=name, width=w, height=h)
        except Exception:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
    return None, None


def open_source(source, cfg):
    """Open a local camera index or a phone stream address.

    *source* is an int / digit string (local camera index), or anything a teacher
    might type for a phone: ``192.168.43.1``, ``192.168.43.1:4747`` or a full
    ``http://`` / ``rtsp://`` URL. For a bare address each known phone app's port
    is tried in turn.

    Returns ``(capture, CameraInfo, error)``; capture is None on failure.
    """
    from netcam import candidate_urls, is_usb_source, open_stream

    if is_usb_source(source):
        cap, info = open_camera(int(source), cfg)
        if cap is None:
            return None, None, f"camera {int(source)} could not be opened"
        return cap, info, ""

    timeout = float(cfg.get("camera.network_timeout_sec", 3.0))
    urls = candidate_urls(str(source))
    if not urls:
        return None, None, f"{source!r} is not a camera index or a phone address"
    last_err = ""
    for url in urls:
        stream, frame, err = open_stream(url, timeout=timeout)
        if stream is not None and frame is not None:
            h, w = frame.shape[:2]
            backend = "HTTP" if url.lower().startswith("http") else "FFMPEG"
            return stream, CameraInfo(index=-1, backend=backend, width=w, height=h, source=url), ""
        last_err = f"{url}: {err}"
    return None, None, f"phone camera not reachable - {last_err}"


def discover_cameras(cfg, max_index: int | None = None) -> list[CameraInfo]:
    """Probe indices 0..max_index and return every camera that produced a frame."""
    if max_index is None:
        max_index = int(cfg.get("camera.scan_max_index", 7))

    found: list[CameraInfo] = []
    for idx in range(0, max_index + 1):
        cap, info = open_camera(idx, cfg)
        if cap is not None and info is not None:
            found.append(info)
            cap.release()
    return found


def main() -> int:
    """Standalone camera scan - part of the Phase 0 physical camera test."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import Config

    cfg = Config.load()
    print("scanning cameras (this can take a few seconds per index)...")
    cams = discover_cameras(cfg)
    if not cams:
        print("NO LOCAL CAMERA FOUND.")
        print("Close Camera / Zoom / Teams if they are holding the device, then retry.")
    else:
        print(f"found {len(cams)} local camera(s):")
        for c in cams:
            print("  " + c.label())
    print()
    print("phone cameras on Wi-Fi / hotspot / USB: run.bat phone")
    return 0 if cams else 1


if __name__ == "__main__":
    raise SystemExit(main())
