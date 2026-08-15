"""Optional memory-footprint sampling for a serving process, keyed by PID.

Only meaningful when the harness runs on the same host as the backend server (or
has /proc access to it, e.g. same container/namespace) -- which is common for local
self-hosted benchmarking (Ollama/vLLM/llama.cpp on the same box as the harness).
Host RSS sampling reads /proc/<pid>/status directly rather than shelling out to `ps`,
so it has no extra dependency and works without permission to send signals to the
process. GPU VRAM sampling shells out to `nvidia-smi` since there's no /proc
equivalent for GPU memory -- for most self-hosted LLM serving on a GPU, VRAM is the
metric that actually constrains deployment, not host RSS.
"""
from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path


def read_rss_mb(pid: int) -> float | None:
    """Reads current resident set size for `pid` in MB from /proc/<pid>/status.

    Returns None if unavailable: non-Linux (no /proc), the process has already
    exited, or permission is denied. This is a point-in-time snapshot, not an
    average -- callers that want peak/mean over a run should sample repeatedly.
    """
    status_path = Path(f"/proc/{pid}/status")
    try:
        with status_path.open(encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    # Format: "VmRSS:    12345 kB"
                    if len(parts) >= 2 and parts[1].isdigit():
                        return int(parts[1]) / 1024
                    return None
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    return None


def read_gpu_vram_mb(pid: int) -> float | None:
    """Reads current GPU VRAM usage for `pid` in MB, summed across every GPU it holds
    compute memory on, via `nvidia-smi --query-compute-apps=pid,used_memory`.

    Returns None if `nvidia-smi` is not installed, the call errors or times out, or the
    PID does not currently appear as a compute process on any GPU -- e.g. no NVIDIA GPU
    present, a CPU-only backend, or the process hasn't allocated GPU memory yet. Like
    `read_rss_mb`, this is a point-in-time snapshot; a PID can legitimately appear more
    than once (one row per GPU it uses), so matching rows are summed rather than
    averaged or taking just the first.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None

    total_mb = 0.0
    matched = False
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            row_pid = int(parts[0])
            used_mb = float(parts[1])
        except ValueError:
            continue
        if row_pid == pid:
            total_mb += used_mb
            matched = True
    return total_mb if matched else None


@dataclass(frozen=True)
class MemoryStats:
    peak_mb: float | None
    mean_mb: float | None
    sample_count: int

    def to_dict(self) -> dict:
        return {
            "peak_mb": round(self.peak_mb, 2) if self.peak_mb is not None else None,
            "mean_mb": round(self.mean_mb, 2) if self.mean_mb is not None else None,
            "sample_count": self.sample_count,
        }


class RssSampler:
    """Polls a PID's RSS on a background thread at a fixed interval until stopped.

    Runs as a daemon thread so a crash mid-benchmark never hangs the CLI waiting
    on the sampler.
    """

    def __init__(self, pid: int, interval_s: float = 0.1):
        self.pid = pid
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = read_rss_mb(self.pid)
            if sample is not None:
                self._samples.append(sample)
            self._stop_event.wait(self.interval_s)

    def stop(self) -> MemoryStats:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 5 + 1.0)
        if not self._samples:
            return MemoryStats(peak_mb=None, mean_mb=None, sample_count=0)
        return MemoryStats(
            peak_mb=max(self._samples),
            mean_mb=sum(self._samples) / len(self._samples),
            sample_count=len(self._samples),
        )


class GpuVramSampler:
    """Polls a PID's GPU VRAM usage on a background thread via `nvidia-smi`, mirroring
    `RssSampler`'s interface and threading model.

    Defaults to a slower poll interval than `RssSampler` (0.5s vs 0.1s) because each
    sample shells out to `nvidia-smi` -- a subprocess launch is orders of magnitude
    slower than a `/proc` read, so polling at RSS's rate would add real overhead to the
    benchmark it's trying to measure.
    """

    def __init__(self, pid: int, interval_s: float = 0.5):
        self.pid = pid
        self.interval_s = interval_s
        self._samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = read_gpu_vram_mb(self.pid)
            if sample is not None:
                self._samples.append(sample)
            self._stop_event.wait(self.interval_s)

    def stop(self) -> MemoryStats:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval_s * 5 + 1.0)
        if not self._samples:
            return MemoryStats(peak_mb=None, mean_mb=None, sample_count=0)
        return MemoryStats(
            peak_mb=max(self._samples),
            mean_mb=sum(self._samples) / len(self._samples),
            sample_count=len(self._samples),
        )
