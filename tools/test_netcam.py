"""Tests for phone cameras (netcam.py). No phone, GPU or internet needed.

Each test starts a real HTTP server on 127.0.0.1 that behaves like the phone apps:
an MJPEG multipart stream (IP Webcam and DroidCam both serve one), a JPEG
snapshot endpoint, a stream without part lengths, and a non-camera web page. The
frame number is encoded in the image itself so a test can tell exactly which
frame the reader returned.

Run:  .venv\\Scripts\\python.exe tools\\test_netcam.py
"""

from __future__ import annotations

import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from config import Config  # noqa: E402
from netcam import (  # noqa: E402
    MjpegStream,
    candidate_urls,
    find_phone_cameras,
    is_network_source,
    is_usb_source,
    open_stream,
    PhoneApp,
)

CFG = Config.load(ROOT / "config.json")
W, H = 320, 240


def frame_jpeg(n: int) -> bytes:
    """A JPEG whose two top-left blocks encode frame number n (0-624)."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:40, :40] = (n % 25) * 10
    img[:40, 40:80] = (n // 25 % 25) * 10
    img[100:140, 100:220] = (30, 160, 240)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    assert ok
    return buf.tobytes()


def frame_id(img: np.ndarray) -> int:
    lo = int(round(float(img[4:36, 4:36].mean()) / 10.0))
    hi = int(round(float(img[4:36, 44:76].mean()) / 10.0))
    return hi * 25 + lo


class PhoneServer:
    """Threaded HTTP server imitating a phone camera app."""

    def __init__(self, port: int = 0, fps: float = 30.0, content_length: bool = True,
                 burst: int = 0):
        self.fps = fps
        self.content_length = content_length
        self.burst = burst          # frames sent back-to-back before pacing starts
        self.counter = 0
        self.alive = True
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a):
                pass

            def do_GET(self):  # noqa: N802
                if self.path.startswith("/video"):
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frameboundary")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    sent = 0
                    try:
                        while outer.alive:
                            outer.counter += 1
                            jpg = frame_jpeg(outer.counter)
                            head = b"--frameboundary\r\nContent-Type: image/jpeg\r\n"
                            if outer.content_length:
                                head += f"Content-Length: {len(jpg)}\r\n".encode()
                            self.wfile.write(head + b"\r\n" + jpg + b"\r\n")
                            self.wfile.flush()
                            sent += 1
                            if sent > outer.burst:
                                time.sleep(1.0 / outer.fps)
                    except (ConnectionError, OSError):
                        pass
                elif self.path.startswith("/shot.jpg"):
                    outer.counter += 1
                    jpg = frame_jpeg(outer.counter)
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                elif self.path.startswith("/page"):
                    body = b"<html>router admin</html>"
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def url(self, path: str = "/video") -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def stop(self) -> None:
        self.alive = False
        self.httpd.shutdown()
        self.httpd.server_close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --- harness --------------------------------------------------------------
_results: list[tuple[bool | None, str, str]] = []


def check(cond: bool, name: str, detail: str = "") -> None:
    _results.append((bool(cond), name, detail))


def section(title: str) -> None:
    _results.append((None, title, ""))


# =========================================================================
def test_source_strings() -> None:
    section("what a teacher types -> what gets tried")
    check(is_usb_source(0) and is_usb_source("1"), "digits are local camera indices")
    check(not is_network_source("0") and is_network_source("192.168.43.1"), "an IP is a phone")
    check(not is_network_source("usb") and not is_network_source(""), "'usb' / '' are not phones")
    c = candidate_urls("192.168.43.1")
    check(c == ["http://192.168.43.1:8080/video", "http://192.168.43.1:4747/video"],
          "bare IP -> IP Webcam then DroidCam", str(c))
    c = candidate_urls("192.168.43.1:4747")
    check(c[0] == "http://192.168.43.1:4747/video", "IP:4747 -> DroidCam's stream first", str(c))
    c = candidate_urls("10.0.0.5:9000")
    check(c[0] == "http://10.0.0.5:9000/video", "unknown port -> /video first", str(c))
    c = candidate_urls("http://u:p@10.0.0.5:8080/video")
    check(c == ["http://u:p@10.0.0.5:8080/video"], "full URL used exactly", str(c))
    check(candidate_urls("rtsp://10.0.0.5:8554/live") == ["rtsp://10.0.0.5:8554/live"], "rtsp kept")


def test_stream_reads_frames() -> None:
    section("MJPEG stream with Content-Length (IP Webcam / DroidCam style)")
    srv = PhoneServer(fps=30)
    try:
        stream, frame, err = open_stream(srv.url(), timeout=2.0)
        check(stream is not None, "opens and decodes a first frame", err)
        if stream is None:
            return
        check(frame.shape[:2] == (H, W), "frame size is right", str(frame.shape))
        check(stream.mode == "stream", "multipart detected as a stream", stream.mode)
        ids = []
        t_end = time.monotonic() + 1.0
        while time.monotonic() < t_end:
            ok, f = stream.read()
            if ok:
                ids.append(frame_id(f))
        check(len(ids) >= 20, "~30 FPS delivered for 1 s", f"{len(ids)} frames")
        check(all(b >= a for a, b in zip(ids, ids[1:])), "frames arrive in order")
        stream.release()
        check(not stream.isOpened(), "release() closes it")
    finally:
        srv.stop()


def test_newest_frame_wins() -> None:
    section("latency: a backlog is skipped, only the newest frame is decoded")
    srv = PhoneServer(fps=5, burst=12)  # 12 frames at once, then slow
    try:
        stream = MjpegStream(srv.url(), timeout=2.0)
        time.sleep(0.5)  # let the burst pile up in the socket
        ok, f = stream.read()
        check(ok, "read succeeds")
        if ok:
            n = frame_id(f)
            check(n >= 10, "returned a frame from the end of the backlog, not the oldest",
                  f"frame {n} of a 12-frame burst")
        stream.release()
    finally:
        srv.stop()


def test_stream_without_content_length() -> None:
    section("MJPEG stream without part lengths (marker scan fallback)")
    srv = PhoneServer(fps=30, content_length=False)
    try:
        stream, frame, err = open_stream(srv.url(), timeout=2.0)
        check(stream is not None and frame is not None, "decodes by JPEG markers", err)
        if stream is not None:
            ok = sum(stream.read()[0] for _ in range(10))
            check(ok == 10, "keeps decoding", f"{ok}/10")
            stream.release()
    finally:
        srv.stop()


def test_snapshot_mode() -> None:
    section("JPEG snapshot endpoint (IP Webcam /shot.jpg)")
    srv = PhoneServer()
    try:
        stream, frame, err = open_stream(srv.url("/shot.jpg"), timeout=2.0)
        check(stream is not None, "opens", err)
        if stream is None:
            return
        check(stream.mode == "snapshot", "detected as snapshots", stream.mode)
        ids = [frame_id(frame)]
        for _ in range(5):
            ok, f = stream.read()
            if ok:
                ids.append(frame_id(f))
        check(len(ids) == 6 and len(set(ids)) == 6, "each read is a new snapshot", str(ids))
        stream.release()
    finally:
        srv.stop()


def test_rejects_non_camera() -> None:
    section("not every open port is a camera")
    srv = PhoneServer()
    try:
        t0 = time.monotonic()
        stream, _f, err = open_stream(srv.url("/page"), timeout=2.0)
        check(stream is None, "an HTML page is refused", err)
        check("Content-Type" in err, "with the reason", err)
        check(time.monotonic() - t0 < 1.5, "refused quickly, without waiting out the timeout")
        stream, _f, err = open_stream(srv.url("/nothing"), timeout=2.0)
        check(stream is None and "404" in err, "a 404 is refused", err)
        port = free_port()
        stream, _f, err = open_stream(f"http://127.0.0.1:{port}/video", timeout=1.0)
        check(stream is None, "nothing listening -> refused", err)
    finally:
        srv.stop()


def test_reconnects_after_wifi_drop() -> None:
    section("weak Wi-Fi: the phone disappears and comes back")
    port = free_port()
    srv = PhoneServer(port=port, fps=30)
    stream = MjpegStream(srv.url(), timeout=1.0)
    ok, _f = stream.read()
    check(ok, "streaming before the drop")
    srv.stop()  # phone walks out of range
    t0 = time.monotonic()
    fails = 0
    while time.monotonic() - t0 < 1.5:
        ok, _f = stream.read()
        if not ok:
            fails += 1
    check(fails > 0, "read() reports no frame while the phone is gone (never hangs)", f"{fails} misses")
    check(time.monotonic() - t0 < 4.0, "each read is bounded by the timeout")
    srv2 = PhoneServer(port=port, fps=30)  # phone is back, same address
    try:
        got = False
        t1 = time.monotonic()
        while time.monotonic() - t1 < 5.0:
            ok, _f = stream.read()
            if ok:
                got = True
                break
        check(got, "frames resume on their own after the phone returns",
              f"{time.monotonic() - t1:.2f}s to recover")
        check(stream.reconnects >= 1, "the reconnect is counted", str(stream.reconnects))
    finally:
        stream.release()
        srv2.stop()


def test_discovery() -> None:
    section("Find phone: sweep, then prove each hit is a real camera")
    cam = PhoneServer(fps=30)
    web = PhoneServer()
    try:
        apps = (
            PhoneApp("Test camera app", cam.port, "/video"),
            PhoneApp("Router page", web.port, "/page"),
        )
        t0 = time.monotonic()
        found = find_phone_cameras(hosts=[], apps=apps, include_localhost=True)
        dt = time.monotonic() - t0
        urls = [f.url for f in found]
        check(cam.url() in urls, "the camera is found", str(urls))
        check(web.url("/page") not in urls, "the web page on an open port is not", str(urls))
        check(found and found[0].width == W, "resolution reported", found[0].label() if found else "")
        check(dt < 5.0, "search is quick", f"{dt:.2f}s")
    finally:
        cam.stop()
        web.stop()


def test_camera_open_source() -> None:
    section("camera.open_source accepts a phone address")
    from camera import open_source

    srv = PhoneServer(fps=30)
    try:
        cap, info, err = open_source(f"127.0.0.1:{srv.port}", CFG)
        check(cap is not None, "IP:port opens", err)
        if cap is not None:
            check(info.is_network and info.source == srv.url(), "CameraInfo records the URL", info.source)
            check(info.label().startswith("Phone"), "labelled as a phone", info.label())
            cap.release()
        cap, info, err = open_source("127.0.0.1:1", CFG)
        check(cap is None and "not reachable" in err, "dead address -> clear error", err)
    finally:
        srv.stop()


def test_video_worker_capture_from_phone() -> None:
    section("VideoWorker captures from a phone stream, rotates, survives a drop")
    from camera import open_source
    from video_worker import VideoWorker

    port = free_port()
    srv = PhoneServer(port=port, fps=30)
    cfg = CFG.override({"camera": {"rotate": 90}})
    worker = VideoWorker(cfg, runtime_report=None)
    try:
        cap, info, err = open_source(srv.url(), cfg)
        check(cap is not None, "opened", err)
        if cap is None:
            return
        worker._begin_capture(cap, info)
        time.sleep(0.5)
        frame, _at = worker._take_frame()
        check(frame is not None, "frames reach the inference slot")
        if frame is not None:
            check(frame.shape[:2] == (W, H), "rotate=90 turns a landscape frame upright",
                  str(frame.shape[:2]))
        check(worker.capture_fps > 15, "capture FPS measured", f"{worker.capture_fps:.1f}")

        srv.stop()
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0 and worker.stage != "phone reconnecting":
            time.sleep(0.05)
        check(worker.stage == "phone reconnecting", "stage says it is reconnecting",
              f"{worker.stage} after {time.monotonic() - t0:.2f}s")
        time.sleep(2.0)
        check(worker._cap_thread is not None and worker._cap_thread.is_alive(),
              "capture thread keeps running through the drop (no 'camera lost')")
        srv = PhoneServer(port=port, fps=30)
        t0 = time.monotonic()
        got = None
        while time.monotonic() - t0 < 6.0:
            got, _ = worker._take_frame()
            if got is not None:
                break
            time.sleep(0.05)
        check(got is not None, "frames resume after the phone returns",
              f"{time.monotonic() - t0:.2f}s")
        check(worker.error is None, "the error clears on recovery", str(worker.error))
    finally:
        worker.stop()
        srv.stop()


def main() -> int:
    tests = [
        test_source_strings,
        test_stream_reads_frames,
        test_newest_frame_wins,
        test_stream_without_content_length,
        test_snapshot_mode,
        test_rejects_non_camera,
        test_reconnects_after_wifi_drop,
        test_discovery,
        test_camera_open_source,
        test_video_worker_capture_from_phone,
    ]
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            import traceback

            traceback.print_exc()
            check(False, f"{fn.__name__} raised", f"{type(exc).__name__}: {exc}")

    print("=" * 72)
    print("PHONE CAMERA TESTS (netcam)")
    print("=" * 72)
    passed = failed = 0
    for ok, name, detail in _results:
        if ok is None:
            print(f"\n  -- {name} --")
            continue
        tag = "PASS" if ok else "FAIL"
        passed += ok
        failed += not ok
        print(f"  {tag}  {name}" + (f"   [{detail}]" if detail else ""))
    print()
    print(f"  {passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
