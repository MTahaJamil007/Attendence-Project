"""Phone cameras over Wi-Fi or a USB cable, with no internet required.

A phone running a camera-server app serves its camera as **MJPEG over HTTP** on
the local network. The phone and the laptop only have to share a network - they
never talk to the internet - so any of these work, including in a building whose
internet is unusable:

* the phone's own hotspot, with the laptop joined to it (mobile data can be off);
* the laptop's Mobile Hotspot, with the phone joined to it;
* any Wi-Fi both are on, or a second phone acting as a router;
* a USB cable: Android **USB tethering** or ``adb forward``, iPhone **Personal
  Hotspot over USB** - each makes a private network over the cable.

Apps this has been written against (both free):

* **IP Webcam** (Android) - port 8080, stream ``/video``, snapshot ``/shot.jpg``;
* **DroidCam** (Android and iPhone) - port 4747, stream ``/video``.

Anything else that serves MJPEG or JPEG snapshots over HTTP works when given its
full URL, and ``rtsp://`` URLs go through OpenCV's FFMPEG backend.

Why a hand-written MJPEG reader rather than ``cv2.VideoCapture(url)``:

* **Latency.** FFMPEG buffers network input. Inference must score the newest view
  of the room - a frame from a second ago shifts every raise out of its own
  response window - so this reader drains whatever has arrived and decodes only
  the newest complete JPEG.
* **Weak Wi-Fi.** A dropped connection is reconnected inside ``read()`` with a
  short backoff, forever. While it is down ``read()`` returns no frame, inference
  stops producing results, and the liveness watchdog suspends the roll call
  (STALLED) rather than marking anyone absent. When the phone comes back the call
  resumes on its own.
* **Timeouts.** Every socket operation is bounded, so a phone that walks out of
  range cannot hang the capture thread.
"""

from __future__ import annotations

import base64
import http.client
import ipaddress
import os
import re
import socket
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

SOI = b"\xff\xd8"
EOI = b"\xff\xd9"
_CONTENT_LENGTH = re.compile(rb"content-length:\s*(\d+)", re.IGNORECASE)
# A buffer this large with no complete frame in it is not a JPEG stream.
_MAX_BUFFER = 16 * 1024 * 1024


@dataclass(frozen=True)
class PhoneApp:
    name: str
    port: int
    stream_path: str
    snapshot_path: str | None = None


KNOWN_APPS: tuple[PhoneApp, ...] = (
    PhoneApp("IP Webcam (Android)", 8080, "/video", "/shot.jpg"),
    PhoneApp("DroidCam (Android / iPhone)", 4747, "/video", None),
)


# --------------------------------------------------------------------------
# Source strings
# --------------------------------------------------------------------------

def is_usb_source(source) -> bool:
    """An integer (or an all-digit string) is a local camera index."""
    if isinstance(source, int):
        return True
    return isinstance(source, str) and source.strip().lstrip("-").isdigit()


def is_network_source(source) -> bool:
    if source is None or is_usb_source(source):
        return False
    text = str(source).strip()
    return bool(text) and text.lower() not in ("usb", "auto")


def candidate_urls(text: str) -> list[str]:
    """Turn what a teacher types into the URLs worth trying, most likely first.

    ``192.168.43.1``            -> each known app's stream on that host
    ``192.168.43.1:4747``       -> that port's known stream path, then ``/video``
    ``http://host:8080/video``  -> exactly that
    """
    text = (text or "").strip()
    if not text:
        return []
    if "://" in text:
        return [text]
    host, _, port_s = text.partition(":")
    port_s, _, path = port_s.partition("/")
    path = "/" + path if path else ""
    if port_s:
        try:
            port = int(port_s)
        except ValueError:
            return []
        if path:
            return [f"http://{host}:{port}{path}"]
        paths = [a.stream_path for a in KNOWN_APPS if a.port == port] or []
        for p in ("/video", "/mjpegfeed", "/"):
            if p not in paths:
                paths.append(p)
        return [f"http://{host}:{port}{p}" for p in paths]
    return [f"http://{host}:{a.port}{a.stream_path}" for a in KNOWN_APPS]


def describe(url: str) -> str:
    """Short human label for a stream URL."""
    try:
        parts = urlsplit(url)
        for app in KNOWN_APPS:
            if parts.port == app.port:
                return f"{app.name.split(' (')[0]} {parts.hostname}"
        return f"{parts.hostname}:{parts.port or 80}"
    except Exception:
        return url


# --------------------------------------------------------------------------
# HTTP MJPEG / snapshot reader
# --------------------------------------------------------------------------

class MjpegStream:
    """``cv2.VideoCapture``-compatible reader for MJPEG-over-HTTP or JPEG snapshots.

    Implements just the surface VideoWorker uses: ``read()``, ``isOpened()``,
    ``release()`` and a no-op ``set()``.
    """

    def __init__(self, url: str, timeout: float = 3.0, connect: bool = True):
        self.url = url
        self.timeout = float(timeout)
        self.mode = ""            # "stream" or "snapshot" once connected
        self.content_type = ""
        self.last_error = ""
        self.reconnects = 0
        self.frames = 0
        self._conn: http.client.HTTPConnection | None = None
        self._resp: http.client.HTTPResponse | None = None
        self._buf = bytearray()
        self._released = False
        self._connected_once = False
        self._backoff = 0.25
        self._retry_at = 0.0
        if connect:
            self._connect()

    # -- cv2.VideoCapture surface -----------------------------------------
    def isOpened(self) -> bool:  # noqa: N802 - mirrors cv2
        return not self._released

    def set(self, *_args) -> bool:
        return False

    def release(self) -> None:
        self._released = True
        self._close()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Return the newest complete frame, reconnecting if the link dropped.

        Blocks for at most about ``timeout`` seconds. Returns ``(False, None)``
        while the phone is unreachable; the caller simply calls again.
        """
        if self._released:
            return False, None
        if self._resp is None:
            now = time.monotonic()
            if now < self._retry_at:
                time.sleep(min(0.25, self._retry_at - now))
                return False, None
            if not self._connect():
                self._retry_at = time.monotonic() + self._backoff
                self._backoff = min(2.0, self._backoff * 2.0)
                return False, None
        try:
            frame = self._read_snapshot() if self.mode == "snapshot" else self._read_stream()
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            frame = None
        if frame is None:
            # Any failure mid-stream: drop the connection and let the next call
            # reconnect. Partial data from a dead connection is never trusted.
            self._close()
            self._retry_at = time.monotonic() + self._backoff
            return False, None
        self._backoff = 0.25
        self.frames += 1
        return True, frame

    # -- connection --------------------------------------------------------
    def _connect(self) -> bool:
        self._close()
        try:
            parts = urlsplit(self.url)
            if parts.scheme not in ("http", "https"):
                self.last_error = f"unsupported scheme {parts.scheme!r}"
                return False
            host = parts.hostname or ""
            port = parts.port or (443 if parts.scheme == "https" else 80)
            klass = http.client.HTTPSConnection if parts.scheme == "https" else http.client.HTTPConnection
            conn = klass(host, port, timeout=self.timeout)
            path = parts.path or "/"
            if parts.query:
                path += "?" + parts.query
            headers = {"User-Agent": "SmartRollCall", "Connection": "keep-alive"}
            if parts.username is not None:
                cred = f"{parts.username}:{parts.password or ''}".encode()
                headers["Authorization"] = "Basic " + base64.b64encode(cred).decode()
            conn.request("GET", path, headers=headers)
            resp = conn.getresponse()
            if resp.status != 200:
                self.last_error = f"HTTP {resp.status} {resp.reason}"
                conn.close()
                return False
            ctype = (resp.getheader("Content-Type") or "").lower()
            self.content_type = ctype
            if ctype.startswith("multipart/"):
                self.mode = "stream"
            elif ctype.startswith("image/"):
                self.mode = "snapshot"
            else:
                self.last_error = f"not a camera stream (Content-Type {ctype or 'missing'})"
                conn.close()
                return False
            if self._connected_once:
                self.reconnects += 1
            self._connected_once = True
            self._conn, self._resp = conn, resp
            self._buf.clear()
            self.last_error = ""
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self._close()
            return False

    def _close(self) -> None:
        for obj in (self._resp, self._conn):
            if obj is not None:
                try:
                    obj.close()
                except Exception:
                    pass
        self._resp = None
        self._conn = None

    # -- stream mode -------------------------------------------------------
    def _next_jpeg(self) -> bytes | None:
        """Pop the next complete JPEG off the buffer, or None if none is complete.

        Uses the part's Content-Length when the server sends one (both apps do),
        which is exact even for JPEGs that embed a thumbnail. Otherwise it falls
        back to scanning for the end-of-image marker.
        """
        buf = self._buf
        soi = buf.find(SOI)
        if soi < 0:
            # Keep one byte in case a marker is split across two reads.
            if len(buf) > 1:
                del buf[:-1]
            return None
        m = None
        for m in _CONTENT_LENGTH.finditer(buf, 0, soi):
            pass
        if m is not None:
            n = int(m.group(1))
            if n > 4 and len(buf) >= soi + n:
                jpeg = bytes(buf[soi:soi + n])
                del buf[:soi + n]
                return jpeg
            if n > 4:
                return None
        eoi = buf.find(EOI, soi + 2)
        if eoi < 0:
            return None
        jpeg = bytes(buf[soi:eoi + 2])
        del buf[:eoi + 2]
        return jpeg

    def _read_stream(self) -> np.ndarray | None:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            latest = None
            while True:
                jpeg = self._next_jpeg()
                if jpeg is None:
                    break
                latest = jpeg  # skip to the newest: older frames are stale
            if latest is not None:
                frame = cv2.imdecode(np.frombuffer(latest, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None and frame.size > 0:
                    return frame
                continue  # a corrupt frame; keep reading
            chunk = self._resp.read1(256 * 1024)
            if not chunk:
                self.last_error = "the phone closed the stream"
                return None
            self._buf += chunk
            if len(self._buf) > _MAX_BUFFER:
                self.last_error = "no JPEG frames in the stream"
                return None
        self.last_error = f"no frame within {self.timeout:.0f}s"
        return None

    # -- snapshot mode -----------------------------------------------------
    def _read_snapshot(self) -> np.ndarray | None:
        # self._resp holds an unread snapshot response: the one from the request
        # that opened the connection, or the one requested at the end of the
        # previous call.
        data = self._resp.read()
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            self.last_error = "snapshot did not decode"
            return None
        # Request the next snapshot on the same keep-alive connection. If that
        # fails, the next read() reconnects rather than this frame being lost.
        try:
            parts = urlsplit(self.url)
            path = (parts.path or "/") + (("?" + parts.query) if parts.query else "")
            self._conn.request("GET", path, headers={"Connection": "keep-alive"})
            self._resp = self._conn.getresponse()
        except Exception:
            self._close()
        return frame


class FfmpegStream:
    """RTSP and anything else OpenCV's FFMPEG backend can open, with reconnect."""

    def __init__(self, url: str, timeout: float = 3.0):
        self.url = url
        self.timeout = float(timeout)
        self.last_error = ""
        self.reconnects = 0
        self._cap = None
        self._released = False
        self._retry_at = 0.0
        # Low-latency demuxing. Must be in the environment before the capture is
        # created; TCP for RTSP because UDP drops frames on weak Wi-Fi.
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
        )
        self._open()

    def _open(self) -> bool:
        try:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout * 1000)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout * 1000)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            if not cap.isOpened():
                cap.release()
                self.last_error = "FFMPEG could not open the stream"
                return False
            self._cap = cap
            return True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False

    def isOpened(self) -> bool:  # noqa: N802
        return not self._released

    def set(self, *_args) -> bool:
        return False

    def release(self) -> None:
        self._released = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._released:
            return False, None
        if self._cap is None:
            if time.monotonic() < self._retry_at:
                time.sleep(0.25)
                return False, None
            if not self._open():
                self._retry_at = time.monotonic() + 1.0
                return False, None
            self.reconnects += 1
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            self._cap.release()
            self._cap = None
            self._retry_at = time.monotonic() + 0.5
            return False, None
        return True, frame


def open_stream(url: str, timeout: float = 3.0):
    """Open *url* and prove it with one decoded frame.

    Returns ``(stream, frame, error)``. *stream* is None when nothing decoded.
    """
    scheme = urlsplit(url).scheme.lower()
    if scheme in ("http", "https"):
        stream = MjpegStream(url, timeout=timeout)
    else:
        stream = FfmpegStream(url, timeout=timeout)
    deadline = time.monotonic() + timeout * 2
    while time.monotonic() < deadline:
        ok, frame = stream.read()
        if ok and frame is not None:
            return stream, frame, ""
        if isinstance(stream, MjpegStream) and stream.mode == "" and stream.last_error:
            # Never connected at all (refused, 404, not a camera): do not wait out
            # the whole deadline on a wrong address.
            break
    err = getattr(stream, "last_error", "") or "no frame received"
    stream.release()
    return None, None, err


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

@dataclass
class FoundCamera:
    url: str
    app: str
    width: int = 0
    height: int = 0

    def label(self) -> str:
        size = f"{self.width}x{self.height}" if self.width else "?"
        return f"{self.app} at {self.url}  ({size})"


def _private_ipv4_interfaces() -> list[ipaddress.IPv4Interface]:
    """The laptop's own private IPv4 addresses, with their networks."""
    out: list[ipaddress.IPv4Interface] = []
    try:
        import psutil

        for addrs in psutil.net_if_addrs().values():
            for a in addrs:
                if a.family != socket.AF_INET or not a.address:
                    continue
                try:
                    iface = ipaddress.IPv4Interface(f"{a.address}/{a.netmask or '255.255.255.0'}")
                except ValueError:
                    continue
                out.append(iface)
    except Exception:
        try:
            for addr in socket.gethostbyname_ex(socket.gethostname())[2]:
                out.append(ipaddress.IPv4Interface(f"{addr}/24"))
        except Exception:
            pass
    return [
        i for i in out
        if i.ip.is_private and not i.ip.is_loopback and not i.ip.is_link_local
    ]


def scan_hosts() -> list[str]:
    """Hosts to probe: every address on each local private network.

    Networks wider than a /24 (a campus LAN) are narrowed to the /24 around the
    laptop's own address. A hotspot's network is never wider than that, and
    sweeping thousands of addresses on someone else's network is not acceptable.
    The likely gateway (``.1``) goes first, because when the laptop is on the
    phone's hotspot the phone *is* the gateway.
    """
    hosts: list[str] = []
    seen: set[str] = set()
    for iface in _private_ipv4_interfaces():
        net = iface.network
        if net.prefixlen < 24:
            net = ipaddress.IPv4Network(f"{iface.ip}/24", strict=False)
        own = str(iface.ip)
        ordered = sorted(net.hosts(), key=lambda h: (int(h) & 0xFF) != 1)
        for h in ordered:
            s = str(h)
            if s != own and s not in seen:
                seen.add(s)
                hosts.append(s)
    return hosts


def _port_open(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def find_adb() -> str | None:
    """adb.exe from PATH, the Android SDK, or a platform-tools folder in the project."""
    from shutil import which

    exe = which("adb")
    if exe:
        return exe
    root = Path(__file__).resolve().parent.parent
    for cand in (
        root / "platform-tools" / "adb.exe",
        root / "tools" / "platform-tools" / "adb.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
    ):
        if cand.is_file():
            return str(cand)
    return None


def adb_forward(ports: tuple[int, ...] | None = None, timeout: float = 5.0) -> tuple[bool, str]:
    """Forward the camera apps' ports from a USB-connected Android phone to localhost.

    Needs USB debugging enabled on the phone and Google's platform-tools (adb) on
    the laptop. After this, the phone's camera is at ``http://127.0.0.1:<port>``
    over the cable, with no Wi-Fi at all.
    """
    adb = find_adb()
    if not adb:
        return False, "adb not found (install Android platform-tools to use a USB cable this way)"
    ports = ports or tuple(a.port for a in KNOWN_APPS)
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        out = subprocess.run([adb, "devices"], capture_output=True, text=True,
                             timeout=timeout, creationflags=flags).stdout
    except Exception as exc:
        return False, f"adb failed: {exc}"
    devices = [ln.split()[0] for ln in out.splitlines()[1:]
               if ln.strip() and ln.split()[-1] == "device"]
    if not devices:
        if "unauthorized" in out:
            return False, "phone connected but not authorized - accept the USB debugging prompt on the phone"
        return False, "no Android phone found over USB (enable USB debugging)"
    for port in ports:
        try:
            subprocess.run([adb, "forward", f"tcp:{port}", f"tcp:{port}"], capture_output=True,
                           timeout=timeout, creationflags=flags)
        except Exception as exc:
            return False, f"adb forward failed: {exc}"
    return True, f"forwarded ports {', '.join(map(str, ports))} from {devices[0]}"


def find_phone_cameras(
    hosts: list[str] | None = None,
    apps: tuple[PhoneApp, ...] = KNOWN_APPS,
    connect_timeout: float = 0.35,
    stream_timeout: float = 2.5,
    include_localhost: bool = True,
    workers: int = 96,
    progress=None,
) -> list[FoundCamera]:
    """Find phones serving a camera on the local network (and via adb on localhost).

    Two stages: a fast TCP connect sweep for the apps' ports, then a real stream
    open on every open port, so a router's admin page on port 8080 is not
    mistaken for a camera.
    """
    if hosts is None:
        hosts = scan_hosts()
    hosts = list(hosts)
    if include_localhost and "127.0.0.1" not in hosts:
        hosts.insert(0, "127.0.0.1")
    targets = [(h, a) for h in hosts for a in apps]
    if progress:
        progress(f"checking {len(hosts)} address(es)...")

    open_targets: list[tuple[str, PhoneApp]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(targets) or 1))) as ex:
        results = ex.map(lambda t: (t, _port_open(t[0], t[1].port, connect_timeout)), targets)
        for (t, ok) in results:
            if ok:
                open_targets.append(t)

    found: list[FoundCamera] = []
    for host, app in open_targets:
        url = f"http://{host}:{app.port}{app.stream_path}"
        if progress:
            progress(f"trying {url}")
        stream, frame, _err = open_stream(url, timeout=stream_timeout)
        if stream is None or frame is None:
            continue
        stream.release()
        h, w = frame.shape[:2]
        found.append(FoundCamera(url=url, app=app.name, width=w, height=h))
    return found
