"""Find and test a phone camera. No GPU or model needed - run it before class.

    run.bat phone                     find phones on every local network (+ adb USB)
    run.bat phone 192.168.43.1        test that address (each known app's port)
    run.bat phone 192.168.43.1:8080   test that exact port
    run.bat phone --seconds 15        measure for longer

Reports resolution, delivered FPS, and how often the link dropped, then prints the
exact command that starts the application on that camera. A weak link shows up
here as a low or uneven FPS long before it shows up as a stalled roll call.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import Config  # noqa: E402
from netcam import (  # noqa: E402
    KNOWN_APPS,
    adb_forward,
    candidate_urls,
    find_adb,
    find_phone_cameras,
    open_stream,
    scan_hosts,
)


def measure(url: str, seconds: float, timeout: float) -> int:
    print(f"\nconnecting to {url} ...")
    stream, frame, err = open_stream(url, timeout=timeout)
    if stream is None:
        print(f"  FAILED: {err}")
        return 1
    h, w = frame.shape[:2]
    print(f"  connected: {w}x{h}, mode {getattr(stream, 'mode', '') or 'ffmpeg'}")
    print(f"  measuring for {seconds:.0f}s ...")
    gaps: list[float] = []
    frames = 0
    misses = 0
    t_end = time.monotonic() + seconds
    last = time.monotonic()
    while time.monotonic() < t_end:
        ok, frame = stream.read()
        now = time.monotonic()
        if ok and frame is not None:
            frames += 1
            gaps.append(now - last)
            last = now
        else:
            misses += 1
    stream.release()
    if not gaps:
        print("  no frames during the measurement")
        return 1
    fps = frames / seconds
    worst = max(gaps)
    p95 = sorted(gaps)[int(len(gaps) * 0.95) - 1] if len(gaps) > 20 else worst
    print(f"  frames            {frames}  ({fps:.1f} FPS)")
    print(f"  frame gap p50/p95 {statistics.median(gaps) * 1000:.0f} / {p95 * 1000:.0f} ms, "
          f"worst {worst * 1000:.0f} ms")
    print(f"  reconnects        {getattr(stream, 'reconnects', 0)}")
    verdict = []
    if fps < 10:
        verdict.append("FPS is low: lower the app's resolution/quality, move the phone closer to "
                       "the hotspot, or use a USB cable")
    if worst > 0.75:
        verdict.append(f"a {worst:.1f}s gap would pause the roll call (stall limit 0.75 s); "
                       "the link is unreliable")
    if w < 1280:
        verdict.append(f"{w}x{h} is small for back rows: set the app to 1280x720 or 1920x1080")
    print()
    if verdict:
        for v in verdict:
            print(f"  note: {v}")
    else:
        print("  link looks good")
    print(f"\n  start the app on this camera with:\n    run.bat app --camera {url}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("address", nargs="?", default="", help="phone address or URL (omit to search)")
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args(argv)
    cfg = Config.load(ROOT / "config.json")
    timeout = float(cfg.get("camera.network_timeout_sec", 3.0))

    print("=" * 72)
    print("PHONE CAMERA CHECK")
    print("=" * 72)

    if args.address:
        urls = candidate_urls(args.address)
        if not urls:
            print(f"  {args.address!r} is not an address")
            return 1
        for url in urls:
            stream, _frame, err = open_stream(url, timeout=timeout)
            if stream is not None:
                stream.release()
                return measure(url, args.seconds, timeout)
            print(f"  {url}: {err}")
        print("\n  nothing answered. Is the app's server started and the phone on this network?")
        return 1

    if find_adb():
        ok, msg = adb_forward()
        print(f"  USB via adb: {msg}")
    else:
        print("  USB via adb: adb not installed (optional - see README)")
    hosts = scan_hosts()
    apps = ", ".join(f"{a.name} :{a.port}" for a in KNOWN_APPS)
    print(f"  searching {len(hosts)} local address(es) + localhost for: {apps}")
    found = find_phone_cameras(hosts=hosts)
    if not found:
        print()
        print("  NO PHONE CAMERA FOUND. Check:")
        print("   - the camera app's server is started (IP Webcam: 'Start server';")
        print("     DroidCam: the app is open and showing its IP)")
        print("   - phone and laptop are on the same hotspot / Wi-Fi, or the cable")
        print("     tethering is on; mobile data and internet are NOT needed")
        print("   - the phone screen has not locked or put the app to sleep")
        print("   - Windows Firewall is not involved: the laptop makes the connection")
        print("  Or pass the address the app shows:  run.bat phone 192.168.x.x:8080")
        return 1
    print(f"\n  found {len(found)}:")
    for cam in found:
        print("    " + cam.label())
    return measure(found[0].url, args.seconds, timeout)


if __name__ == "__main__":
    raise SystemExit(main())
