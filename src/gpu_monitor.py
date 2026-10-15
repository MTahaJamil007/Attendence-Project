"""Live GPU telemetry from NVML, for the console status bar.

Why this exists: Windows Task Manager's GPU page graphs the *3D*, *Copy* and
*Video* engines by default. PyTorch runs on the *Compute/CUDA* engine, which is
not one of the default graphs, so a GPU running pose inference flat out can read
"Utilization 0%" there while its dedicated memory shows the model loaded. Measured
on this machine during `run.bat smoke`: Task Manager 0%, NVML 56-59% utilization,
487 MB used, 75 W, SM clock 1.6-1.9 GHz.

Reading NVML directly - the same counters ``nvidia-smi`` prints - shows the real
load inside the app, so nobody has to go digging through Task Manager's engine
drop-downs to find out whether the GPU is doing the work.

Never raises: on a machine without the NVIDIA driver or ``nvidia-ml-py`` the
monitor reports itself unavailable and the status bar simply omits the figures.
"""

from __future__ import annotations

import time


class GpuMonitor:
    def __init__(self, index: int = 0, min_interval: float = 1.0):
        self.available = False
        self.name = ""
        self.error = ""
        self.min_interval = float(min_interval)
        self._nvml = None
        self._handle = None
        self._last_at = 0.0
        self._last: dict | None = None
        try:
            import pynvml  # provided by the nvidia-ml-py package

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(index))
            name = pynvml.nvmlDeviceGetName(self._handle)
            self.name = name.decode() if isinstance(name, bytes) else str(name)
            self._nvml = pynvml
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def sample(self, now: float | None = None) -> dict | None:
        """Utilization %, memory used (MB) and power (W). Cached for min_interval.

        NVML calls are cheap (well under a millisecond), but the UI ticks at 30 Hz
        and a once-a-second figure is all a human can read anyway.
        """
        if not self.available:
            return None
        now = time.monotonic() if now is None else now
        if self._last is not None and now - self._last_at < self.min_interval:
            return self._last
        nv = self._nvml
        out: dict = {"name": self.name}
        try:
            out["util"] = int(nv.nvmlDeviceGetUtilizationRates(self._handle).gpu)
        except Exception:
            out["util"] = None
        try:
            out["mem_mb"] = nv.nvmlDeviceGetMemoryInfo(self._handle).used / (1024 * 1024)
        except Exception:
            out["mem_mb"] = None
        try:
            out["power_w"] = nv.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:
            out["power_w"] = None
        self._last, self._last_at = out, now
        return out

    def text(self, now: float | None = None) -> str:
        s = self.sample(now)
        if not s:
            return ""
        bits = ["gpu"]
        if s.get("util") is not None:
            bits.append(f"{s['util']:3d}%")
        if s.get("mem_mb") is not None:
            bits.append(f"{s['mem_mb'] / 1024:.1f}GB")
        if s.get("power_w") is not None:
            bits.append(f"{s['power_w']:.0f}W")
        return " ".join(bits)
