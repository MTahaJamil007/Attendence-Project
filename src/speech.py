"""Text-to-speech worker for calling roll numbers.

pyttsx3 on Windows drives SAPI through comtypes, which is not safe to use from
several threads and whose ``runAndWait()`` blocks until the utterance finishes.
So exactly one thread owns the engine and pulls work from a queue.

Two failure modes are handled deliberately, because both would otherwise stall a
roll call permanently:

* **The engine cannot be created at all** (no voice installed, COM failure).
  ``available`` stays False and the session falls back to a fixed reaction delay.
* **An utterance never completes.** SAPI can hang. The session does not wait on
  speech indefinitely - it treats the "spoken" event as a hint and proceeds on a
  timeout regardless. Speech is a convenience; attendance must not depend on it.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass


@dataclass
class SpeechEvent:
    token: int
    kind: str  # "spoken" | "error"
    detail: str = ""


class SpeechWorker:
    def __init__(self, cfg):
        s = cfg.section("speech")
        self.enabled: bool = bool(s.get("enabled", True))
        self.rate: int = int(s.get("rate", 155))
        self.volume: float = float(s.get("volume", 1.0))
        self.voice_hint: str = str(s.get("voice", "") or "")
        self.startup_announcement: str = str(s.get("startup_announcement", "Attendance ready") or "")

        self._in: queue.Queue = queue.Queue()
        self._out: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

        self.available: bool = False
        self.error: str | None = None
        self.voice_name: str = ""
        self.warmed: bool = False

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._in.put(None)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self.available = False

    # -- api ---------------------------------------------------------------
    def speak(self, text: str, token: int) -> None:
        """Queue an utterance. Silently ignored when TTS is unavailable."""
        if not self.enabled or not text:
            return
        self._in.put((token, text))

    def poll(self) -> list[SpeechEvent]:
        """Drain completed-speech events without blocking."""
        out: list[SpeechEvent] = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def clear(self) -> None:
        """Drop anything still queued, e.g. when the teacher hits Pause."""
        while True:
            try:
                self._in.get_nowait()
            except queue.Empty:
                break

    # -- worker ------------------------------------------------------------
    def _run(self) -> None:
        # comtypes normally initializes COM itself, but doing it explicitly on
        # this thread avoids an apartment mismatch on some Windows builds.
        try:
            import pythoncom

            pythoncom.CoInitialize()
        except Exception:
            pass

        engine = None
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", max(0.0, min(1.0, self.volume)))
            if self.voice_hint:
                try:
                    for v in engine.getProperty("voices"):
                        if self.voice_hint.lower() in (v.name or "").lower():
                            engine.setProperty("voice", v.id)
                            break
                except Exception:
                    pass
            try:
                cur = engine.getProperty("voice")
                for v in engine.getProperty("voices"):
                    if v.id == cur:
                        self.voice_name = v.name or ""
                        break
            except Exception:
                pass
            # Warm up the engine before the session starts.
            #
            # Measured on this machine: the first utterance takes ~2.5 s while
            # later ones take ~0.25 s, because SAPI initializes its audio device
            # lazily. Without this, that 2 s penalty lands on the first student
            # called. Speaking a real word rather than silence also gives the
            # teacher an audible check that output and volume are working before
            # the class is waiting on it.
            try:
                if self.startup_announcement:
                    engine.say(self.startup_announcement)
                    engine.runAndWait()
                    self.warmed = True
            except Exception:
                pass
            self.available = True
        except Exception as exc:
            self.error = f"text-to-speech unavailable: {exc}"
            self.available = False
            return

        while not self._stop.is_set():
            item = self._in.get()
            if item is None:
                break
            token, text = item
            try:
                engine.say(text)
                engine.runAndWait()
                self._out.put(SpeechEvent(token=token, kind="spoken"))
            except Exception as exc:
                self.error = str(exc)
                self._out.put(SpeechEvent(token=token, kind="error", detail=str(exc)))

        try:
            engine.stop()
        except Exception:
            pass
        try:
            import pythoncom

            pythoncom.CoUninitialize()
        except Exception:
            pass
