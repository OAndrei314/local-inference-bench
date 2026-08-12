"""Optional memory-footprint sampling for a serving process, keyed by PID.

Only meaningful when the harness runs on the same host as the backend server (or
has /proc access to it, e.g. same container/namespace) -- which is common for local
self-hosted benchmarking (Ollama/vLLM/llama.cpp on the same box as the harness).
Sampling reads /proc/<pid>/status directly rather than shelling out to `ps`, so it
has no extra dependency and works without permission to send signals to the process.
"""
from __future__ import annotations

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
