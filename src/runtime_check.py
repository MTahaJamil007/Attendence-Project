"""Startup runtime gate.

The shipped predecessor to this project bundled a CPU-only PyTorch build and
silently ran pose inference at 3-8 FPS, which made its hold-based decision rule
mathematically unsatisfiable. This module exists so that can never happen again:
check_runtime refuses to report success on CPU unless the caller explicitly opts
in, and the application treats a failed report as fatal.

It also keeps the predecessor's one genuinely good idea - a real torchvision::nms
call - because a mismatched torch/torchvision pair fails there with a confusing
error that is otherwise misdiagnosed as a camera fault.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field


@dataclass
class RuntimeReport:
    ok: bool = False
    stage: str = "not started"
    messages: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    python_version: str = ""
    torch_version: str = ""
    torchvision_version: str = ""
    cuda_build: str = ""
    cuda_available: bool = False
    device_name: str = ""
    capability: str = ""
    vram_gb: float = 0.0
    fp16_ok: bool = False
    using_cpu: bool = False

    def summary(self) -> str:
        if self.ok and not self.using_cpu:
            return (
                f"GPU ready: {self.device_name} "
                f"({self.vram_gb:.1f} GB, sm_{self.capability}) | "
                f"torch {self.torch_version} / cu{self.cuda_build} | "
                f"FP16 {'yes' if self.fp16_ok else 'no'}"
            )
        if self.ok and self.using_cpu:
            return f"CPU MODE (degraded): torch {self.torch_version}"
        return f"RUNTIME FAILED at '{self.stage}': " + "; ".join(self.errors)

    def report_lines(self) -> list[str]:
        lines = [
            f"python           {self.python_version}",
            f"torch            {self.torch_version}",
            f"torchvision      {self.torchvision_version}",
            f"torch cuda build {self.cuda_build or '(cpu-only build)'}",
            f"cuda available   {self.cuda_available}",
        ]
        if self.cuda_available:
            lines.append(f"device           {self.device_name}")
            lines.append(f"capability       sm_{self.capability}")
            lines.append(f"vram             {self.vram_gb:.1f} GB")
            lines.append(f"fp16 matmul      {'ok' if self.fp16_ok else 'FAILED'}")
        lines.extend(self.messages)
        lines.extend(f"ERROR: {e}" for e in self.errors)
        return lines


def check_runtime(require_cuda: bool = True, allow_cpu_fallback: bool = False) -> RuntimeReport:
    """Verify the ML runtime is usable.

    Args:
        require_cuda: fail the report when no CUDA device is present.
        allow_cpu_fallback: when True and CUDA is missing, still return ok=True
            but with using_cpu set. For offline tooling only, never for a live
            attendance session.
    """
    rep = RuntimeReport()
    rep.python_version = f"{platform.python_version()} ({platform.machine()})"

    # --- torch import -----------------------------------------------------
    rep.stage = "import torch"
    try:
        import torch
    except Exception as exc:
        rep.errors.append(f"cannot import torch: {exc}")
        return rep
    rep.torch_version = torch.__version__
    rep.cuda_build = (torch.version.cuda or "").replace(".", "")

    # --- torchvision import ----------------------------------------------
    rep.stage = "import torchvision"
    try:
        import torchvision
    except Exception as exc:
        rep.errors.append(f"cannot import torchvision: {exc}")
        return rep
    rep.torchvision_version = torchvision.__version__

    # --- torchvision::nms operator ---------------------------------------
    # A torch/torchvision version mismatch surfaces here as
    # "operator torchvision::nms does not exist". Catching it now stops the
    # failure being misreported later as a camera problem.
    rep.stage = "torchvision::nms self-test"
    try:
        from torchvision.ops import nms

        boxes = torch.tensor(
            [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [50.0, 50.0, 60.0, 60.0]]
        )
        scores = torch.tensor([0.9, 0.8, 0.7])
        keep = nms(boxes, scores, 0.5)
        if keep.numel() == 0:
            rep.errors.append("torchvision::nms returned no result")
            return rep
        rep.messages.append(f"nms self-test kept {keep.numel()} of 3 boxes")
    except Exception as exc:
        rep.errors.append(
            f"torchvision::nms failed - torch {rep.torch_version} and "
            f"torchvision {rep.torchvision_version} are incompatible: {exc}"
        )
        return rep

    # --- CUDA -------------------------------------------------------------
    rep.stage = "cuda availability"
    rep.cuda_available = bool(torch.cuda.is_available())

    if not rep.cuda_available:
        if require_cuda and not allow_cpu_fallback:
            if not torch.version.cuda:
                hint = (
                    "this is a CPU-only torch build; reinstall with "
                    "--index-url https://download.pytorch.org/whl/cu126"
                )
            else:
                hint = "torch has CUDA support but no device was detected; check the NVIDIA driver"
            rep.errors.append(f"no CUDA device available ({hint})")
            return rep
        rep.using_cpu = True
        rep.messages.append("running on CPU - detection will be too slow for live use")
        rep.ok = True
        rep.stage = "complete (cpu)"
        return rep

    # --- device properties ------------------------------------------------
    rep.stage = "cuda device properties"
    try:
        props = torch.cuda.get_device_properties(0)
        rep.device_name = props.name
        rep.capability = f"{props.major}{props.minor}"
        rep.vram_gb = props.total_memory / (1024**3)
    except Exception as exc:
        rep.errors.append(f"cannot read CUDA device properties: {exc}")
        return rep

    # --- real GPU compute -------------------------------------------------
    # torch.cuda.is_available() can be True while a driver/toolkit mismatch
    # makes the first real kernel launch fail, so actually launch one.
    rep.stage = "cuda matmul self-test"
    try:
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        (a @ b).sum().item()
        torch.cuda.synchronize()
    except Exception as exc:
        rep.errors.append(f"CUDA kernel launch failed: {exc}")
        return rep

    rep.stage = "fp16 self-test"
    try:
        out = (a.half() @ b.half()).float().sum().item()
        torch.cuda.synchronize()
        rep.fp16_ok = out == out  # False only if NaN
    except Exception as exc:
        rep.fp16_ok = False
        rep.messages.append(f"FP16 unavailable, will run FP32: {exc}")

    del a, b
    torch.cuda.empty_cache()

    rep.ok = True
    rep.stage = "complete"
    return rep


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Verify the ML runtime for the attendance system.")
    ap.add_argument("--allow-cpu", action="store_true", help="do not fail when CUDA is missing")
    args = ap.parse_args(argv)

    rep = check_runtime(require_cuda=True, allow_cpu_fallback=args.allow_cpu)
    print("=" * 64)
    print("RUNTIME CHECK")
    print("=" * 64)
    for line in rep.report_lines():
        print("  " + line)
    print("-" * 64)
    print(rep.summary())
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
