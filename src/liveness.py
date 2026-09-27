"""Liveness tracking for a pipeline stage.

Small on purpose. This holds the single most important safety property in the
system, so it lives in its own module where it can be tested without a camera, a
GPU or torch, and where the Phase 2 session state machine can reuse it directly.

The rule it enforces: **an absence may only be recorded while the perception
pipeline is demonstrably alive.** The predecessor had no such rule. Its saved runs
show gaps of 52 s, 40 s, 27 s and 22 s against a nominal 5 s cadence - inference
stalling while the state machine ran on its own timer and wrote
"Timeout: no valid hand" for every remaining student. Three of its eight recorded
runs were ruined that way, which is worse than a missed detection: it is a
confident false record.
"""

from __future__ import annotations


class LivenessMonitor:
    """Tracks how long ago a stage last produced output.

    Args:
        stall_threshold: seconds after which the stage counts as stalled.
        name: label used in status text.
    """

    def __init__(self, stall_threshold: float, name: str = "stage"):
        if stall_threshold <= 0.0:
            raise ValueError("stall_threshold must be positive")
        self.stall_threshold = float(stall_threshold)
        self.name = name
        self.last_at: float = 0.0
        self.count: int = 0

    def mark(self, now: float) -> None:
        """Record that the stage produced output at *now*."""
        self.last_at = now
        self.count += 1

    def reset(self) -> None:
        self.last_at = 0.0
        self.count = 0

    def age(self, now: float) -> float:
        """Seconds since the last output. ``inf`` before the first one.

        Returning infinity rather than 0 before startup matters: a freshly
        constructed monitor must read as "not yet alive", so a session that starts
        marking before the first frame arrives is blocked rather than allowed.
        """
        if self.last_at <= 0.0:
            return float("inf")
        return max(0.0, now - self.last_at)

    def is_live(self, now: float) -> bool:
        return self.age(now) <= self.stall_threshold

    def status(self, now: float) -> str:
        if self.last_at <= 0.0:
            return f"{self.name}: no output yet"
        age = self.age(now)
        if age <= self.stall_threshold:
            return f"{self.name}: live ({age * 1000:.0f} ms ago)"
        return f"{self.name}: STALLED for {age:.1f}s"

    def may_record_absence(self, now: float) -> tuple[bool, str]:
        """The gate the session state machine must call before writing Absent.

        Returns (allowed, reason). When not allowed, the caller must suspend its
        countdown rather than resolving the call, so that a stall postpones the
        decision instead of fabricating one.
        """
        if self.last_at <= 0.0:
            return False, "perception pipeline has not produced a frame yet"
        age = self.age(now)
        if age > self.stall_threshold:
            return False, f"perception pipeline stalled {age:.1f}s ago"
        return True, ""
