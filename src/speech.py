"""Text-to-speech worker for calling registration numbers.

The machine reads each registration number aloud - "S P two five, B C S, zero zero
seven" - and the student answers by raising a hand.

**Why SAPI directly, not pyttsx3.** Measured on this machine with pyttsx3 2.99:
only the *first* ``runAndWait()`` ever plays. Every later call returns in about
0.15 s having produced no audio at all - a sentence that takes 4 s to say
"finished" in 0.24 s. The worker's warm-up announcement was that first call, so
every roll number after it was silent, while "spoken" events kept arriving on
time and nothing looked wrong. (The old README's "later utterances take
0.10-0.18 s" was this bug being measured.) Driving ``SAPI.SpVoice`` through
comtypes - already a dependency, it is what pyttsx3 uses underneath - speaks every
utterance at full length, repeatably. pyttsx3 remains the fallback where SAPI does
not exist.

SAPI is COM, which is not safe to share across threads, so exactly one thread
owns the voice and pulls work from a queue.

Two failure modes are handled deliberately, because both would otherwise stall a
roll call permanently:

* **The engine cannot be created at all** (no voice installed, COM failure).
  ``available`` stays False and the session falls back to a fixed reaction delay.
* **An utterance never completes.** SAPI can hang. The session does not wait on
  speech indefinitely - it treats the "spoken" event as a hint and proceeds on a
  timeout regardless - and the worker abandons any utterance that runs past
  ``MAX_UTTERANCE_SEC`` so a hang cannot back the queue up. Speech is a
  convenience; attendance must not depend on it.
"""

from __future__ import annotations

import math
import queue
import sys
import threading
import time
from dataclasses import dataclass

# SpeechVoiceSpeakFlags
SVSF_ASYNC = 1
SVSF_PURGE_BEFORE_SPEAK = 2

# pyttsx3's words-per-minute -> SAPI Rate mapping (Rate = log_b(wpm / a)), kept
# so that ``speech.rate`` means the same thing whichever backend is speaking.
_SAPI_WPM_AT_ZERO = 156.63
_SAPI_WPM_BASE = 1.11

# No registration number takes anywhere near this long to say. Past it the
# engine is presumed hung and the utterance is abandoned.
MAX_UTTERANCE_SEC = 20.0
_POLL_MS = 50


def sapi_rate(wpm: float) -> int:
    """Convert words per minute to SAPI's -10..10 Rate scale."""
    if wpm <= 0:
        return 0
    rate = round(math.log(wpm / _SAPI_WPM_AT_ZERO, _SAPI_WPM_BASE))
    return max(-10, min(10, rate))


@dataclass
class SpeechEvent:
    token: int
    kind: str  # "spoken" | "cancelled" | "error"
    detail: str = ""


class _SapiVoice:
    """SAPI.SpVoice through comtypes. Asynchronous, so it can be cut off."""

    backend = "sapi"

    def __init__(self, rate_wpm: float, volume: float, voice_hint: str):
        import comtypes.client

        self._v = comtypes.client.CreateObject("SAPI.SpVoice")
        if voice_hint:
            voices = self._v.GetVoices()
            for i in range(voices.Count):
                token = voices.Item(i)
                if voice_hint.lower() in (token.GetDescription() or "").lower():
                    self._v.Voice = token
                    break
        self._v.Rate = sapi_rate(rate_wpm)
        self._v.Volume = int(round(max(0.0, min(1.0, volume)) * 100))
        try:
            self.name = self._v.Voice.GetDescription() or ""
        except Exception:
            self.name = ""

    def begin(self, text: str) -> None:
        self._v.Speak(text, SVSF_ASYNC)

    def wait(self, ms: int) -> bool:
        """True once the current utterance has finished playing."""
        return bool(self._v.WaitUntilDone(ms))

    def cancel(self) -> None:
        self._v.Speak("", SVSF_ASYNC | SVSF_PURGE_BEFORE_SPEAK)

    def close(self) -> None:
        try:
            self.cancel()
        except Exception:
            pass


class _Pyttsx3Voice:
    """Fallback for platforms without SAPI. Blocking, so it cannot be cut off."""

    backend = "pyttsx3"

    def __init__(self, rate_wpm: float, volume: float, voice_hint: str):
        import pyttsx3

        self._e = pyttsx3.init()
        self._e.setProperty("rate", rate_wpm)
        self._e.setProperty("volume", max(0.0, min(1.0, volume)))
        self.name = ""
        try:
            voices = self._e.getProperty("voices")
            if voice_hint:
                for v in voices:
                    if voice_hint.lower() in (v.name or "").lower():
                        self._e.setProperty("voice", v.id)
                        break
            cur = self._e.getProperty("voice")
            self.name = next((v.name or "" for v in voices if v.id == cur), "")
        except Exception:
            pass

    def begin(self, text: str) -> None:
        self._e.say(text)
        self._e.runAndWait()

    def wait(self, ms: int) -> bool:
        return True

    def cancel(self) -> None:
        pass

    def close(self) -> None:
        try:
            self._e.stop()
        except Exception:
            pass


def _com_init() -> None:
    # comtypes only initializes COM on the thread that imports it, so a worker
    # thread must do it explicitly or SAPI object creation fails.
    try:
        import comtypes

        comtypes.CoInitialize()
        return
    except Exception:
        pass
    try:
        import pythoncom

        pythoncom.CoInitialize()
    except Exception:
        pass


def _com_uninit() -> None:
    try:
        import comtypes

        comtypes.CoUninitialize()
    except Exception:
        pass


class SpeechWorker:
    def __init__(self, cfg):
        s = cfg.section("speech")
        self.enabled: bool = bool(s.get("enabled", True))
        self.engine: str = str(s.get("engine", "auto") or "auto").lower()
        self.rate: int = int(s.get("rate", 175))
        self.volume: float = float(s.get("volume", 1.0))
        self.voice_hint: str = str(s.get("voice", "") or "")
        self.startup_announcement: str = str(s.get("startup_announcement", "Attendance ready") or "")

        self._in: queue.Queue = queue.Queue()
        self._out: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        # Bumped by clear(). Every queued item carries the generation it was
        # queued in, so the worker can tell - without a race - that an item, or
        # the utterance it is playing right now, was cancelled.
        self._gen = 0

        self.available: bool = False
        self.error: str | None = None
        self.voice_name: str = ""
        self.backend: str = ""
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
        """Queue an utterance. Silently ignored when TTS is disabled."""
        if not self.enabled or not text:
            return
        self._in.put((token, text, self._gen))

    def poll(self) -> list[SpeechEvent]:
        """Drain completed-speech events without blocking."""
        out: list[SpeechEvent] = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                return out

    def clear(self) -> None:
        """Stop talking: drop anything queued and cut off the current utterance.

        Used when the teacher hits Pause - a voice that keeps reading numbers
        after Pause is confusing to a class that has been told to wait.
        """
        self._gen += 1
        while True:
            try:
                self._in.get_nowait()
            except queue.Empty:
                break

    # -- worker ------------------------------------------------------------
    def _open_voice(self):
        order: list[type] = []
        if self.engine in ("auto", "sapi") and sys.platform == "win32":
            order.append(_SapiVoice)
        if self.engine in ("auto", "pyttsx3"):
            order.append(_Pyttsx3Voice)
        errors = []
        for klass in order:
            try:
                return klass(self.rate, self.volume, self.voice_hint)
            except Exception as exc:
                errors.append(f"{klass.backend}: {exc}")
        raise RuntimeError("; ".join(errors) or f"no speech engine for {self.engine!r}")

    def _say(self, voice, text: str, gen: int | None) -> str:
        """Speak one utterance to completion. Returns "spoken" / "cancelled"."""
        voice.begin(text)
        started = time.monotonic()
        while not voice.wait(_POLL_MS):
            if self._stop.is_set() or (gen is not None and gen != self._gen):
                voice.cancel()
                return "cancelled"
            if time.monotonic() - started > MAX_UTTERANCE_SEC:
                voice.cancel()
                raise TimeoutError(f"utterance did not finish in {MAX_UTTERANCE_SEC:.0f}s")
        return "spoken"

    def _run(self) -> None:
        _com_init()
        try:
            voice = self._open_voice()
        except Exception as exc:
            self.error = f"text-to-speech unavailable: {exc}"
            self.available = False
            _com_uninit()
            return
        self.backend = voice.backend
        self.voice_name = voice.name

        # Warm up the engine before the session starts.
        #
        # The first utterance pays ~2 s of audio-device initialization that later
        # ones do not. Without this, that penalty lands on the first student
        # called. Speaking a real phrase rather than silence also gives the
        # teacher an audible check that output and volume are working before
        # the class is waiting on it.
        try:
            if self.startup_announcement:
                self._say(voice, self.startup_announcement, None)
                self.warmed = True
        except Exception:
            pass
        self.available = True

        while not self._stop.is_set():
            item = self._in.get()
            if item is None:
                break
            token, text, gen = item
            if gen != self._gen:
                self._out.put(SpeechEvent(token=token, kind="cancelled"))
                continue
            try:
                kind = self._say(voice, text, gen)
                self._out.put(SpeechEvent(token=token, kind=kind))
            except Exception as exc:
                self.error = str(exc)
                self._out.put(SpeechEvent(token=token, kind="error", detail=str(exc)))

        voice.close()
        _com_uninit()
