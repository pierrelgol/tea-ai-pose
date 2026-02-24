from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from threading import Event, Thread
import time


@dataclass(slots=True)
class GpuProbe:
    index: int
    name: str
    total_vram_mb: int
    compute_capability: str
    driver_version: str
    signature: str


@dataclass(slots=True)
class GpuSampleSummary:
    peak_vram_mb: float | None
    avg_util_percent: float | None


def _parse_device_index(resolved_device: str) -> int:
    rd = str(resolved_device).strip().lower()
    if rd.isdigit():
        return int(rd)
    if rd.startswith("cuda:"):
        try:
            return int(rd.split(":", 1)[1])
        except Exception:
            return 0
    return 0


def signature_from_parts(*, name: str, total_vram_mb: int, compute_capability: str, driver_version: str) -> str:
    raw = f"{name}|{int(total_vram_mb)}|{compute_capability}|{driver_version}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def build_gpu_signature(resolved_device: str) -> GpuProbe:
    idx = _parse_device_index(resolved_device)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    props = torch.cuda.get_device_properties(idx)
    name = str(props.name)
    total_vram_mb = int(props.total_memory // (1024 * 1024))
    compute_capability = f"{props.major}.{props.minor}"

    driver = "unknown"
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
                "--id",
                str(idx),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
        if lines:
            driver = lines[0]
    except Exception:
        pass

    return GpuProbe(
        index=idx,
        name=name,
        total_vram_mb=total_vram_mb,
        compute_capability=compute_capability,
        driver_version=driver,
        signature=signature_from_parts(
            name=name,
            total_vram_mb=total_vram_mb,
            compute_capability=compute_capability,
            driver_version=driver,
        ),
    )


def sample_nvidia_smi_once(index: int) -> tuple[float | None, float | None]:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
                "--id",
                str(index),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
        if not lines:
            return None, None
        first = lines[0]
        u, m = [x.strip() for x in first.split(",")[:2]]
        return float(u), float(m)
    except Exception:
        return None, None


class GpuMonitor:
    def __init__(self, *, index: int, interval_s: float = 0.5) -> None:
        self._index = int(index)
        self._interval_s = float(interval_s)
        self._stop = Event()
        self._thread: Thread | None = None
        self._utils: list[float] = []
        self._mems: list[float] = []

    def start(self) -> None:
        self._stop.clear()

        def _run() -> None:
            while not self._stop.is_set():
                util, mem = sample_nvidia_smi_once(self._index)
                if util is not None:
                    self._utils.append(util)
                if mem is not None:
                    self._mems.append(mem)
                time.sleep(self._interval_s)

        self._thread = Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> GpuSampleSummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        peak = max(self._mems) if self._mems else None
        avg_util = (sum(self._utils) / len(self._utils)) if self._utils else None
        return GpuSampleSummary(peak_vram_mb=peak, avg_util_percent=avg_util)
