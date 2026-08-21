"""Optional memory-footprint sampling for a serving process, keyed by PID.

Only meaningful when the harness runs on the same host as the backend server (or
has /proc access to it, e.g. same container/namespace) -- which is common for local
self-hosted benchmarking (Ollama/vLLM/llama.cpp/MLX on the same box as the harness).
Host RSS sampling reads /proc/<pid>/status directly on Linux, with no extra
dependency and no permission needed to signal the process; on platforms without
/proc (macOS) it falls back to `ps -o rss=`. GPU VRAM sampling shells out to
`nvidia-smi` (NVIDIA) or `rocm-smi` (AMD) since there's no /proc equivalent for GPU
memory -- for most self-hosted LLM serving on a discrete GPU, VRAM is the metric
that actually constrains deployment, not host RSS. Apple Silicon has no discrete-GPU
VRAM sampler because it has no discrete VRAM: its unified memory architecture means
host RSS *is* the GPU-resident footprint there too.
"""
from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


def read_rss_mb(pid: int) -> float | None:
    """Reads current resident set size for `pid` in MB.

    Dispatches on whether /proc exists: Linux reads /proc/<pid>/status directly
    (no subprocess, no extra dependency); everything else -- notably macOS, which
    has no /proc -- falls back to shelling out to `ps`. Returns None if the process
    has already exited or permission is denied. This is a point-in-time snapshot,
    not an average -- callers that want peak/mean over a run should sample
    repeatedly (see RssSampler).
    """
    if Path("/proc").is_dir():
        return _read_rss_mb_via_proc(pid)
    return _read_rss_mb_via_ps(pid)


def _read_rss_mb_via_proc(pid: int) -> float | None:
    """Reads RSS from /proc/<pid>/status. Linux only -- see `read_rss_mb`."""
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


def _read_rss_mb_via_ps(pid: int) -> float | None:
    """Reads RSS for `pid` in MB via `ps -o rss= -p <pid>` (RSS in KB, no header).

    Fallback for platforms without /proc -- chiefly macOS, both Intel and Apple
    Silicon. On Apple Silicon's unified memory architecture, CPU and GPU share one
    physical memory pool, so this host RSS already reflects a serving process's
    GPU-resident weights and KV-cache too; there is no separate "VRAM" pool to
    sample the way `read_gpu_vram_mb`/`read_amd_gpu_vram_mb` do for a discrete
    NVIDIA/AMD GPU, so no Apple-specific VRAM sampler exists or is needed here.

    Returns None if `ps` is missing, the call errors or times out, the PID has
    already exited (empty stdout / nonzero exit), or the output isn't a plain
    integer.
    """
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    output = result.stdout.strip()
    if result.returncode != 0 or not output:
        return None
    try:
        return int(output) / 1024
    except ValueError:
        return None


def read_gpu_vram_mb(pid: int) -> float | None:
    """Reads current NVIDIA GPU VRAM usage for `pid` in MB, summed across every GPU it
    holds compute memory on, via `nvidia-smi --query-compute-apps=pid,used_memory`.

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


def _parse_rocm_showpids_vram_mb(output: str, pid: int) -> float | None:
    """Parses plain-text `rocm-smi --showpids` output for one PID's VRAM usage in MB.

    The table's columns are `PID  PROCESS NAME  GPU(s)  VRAM USED  SDMA USED
    CU OCCUPANCY` -- PROCESS NAME is free text (no fixed width or quoting), so rows are
    parsed from both ends: PID is the first token, and GPU(s)/VRAM USED/SDMA USED/CU
    OCCUPANCY are the last four. `--json` was considered but rocm-smi's JSON mode for
    this specific command collapses the column headers into a single lowercased,
    comma-joined string per PID rather than keyed fields, which is less stable to parse
    than the documented plain-text table.

    Unlike nvidia-smi's `--query-compute-apps` (one row per GPU+PID pair, summed here),
    rocm-smi already reports one aggregated row per PID with VRAM USED as the process's
    total across every GPU it uses, so the first (only) matching row is returned as-is.

    Returns None if the PID has no row, or its VRAM USED is the literal `UNKNOWN` that
    rocm-smi emits when the driver can't attribute usage to that PID.
    """
    header_seen = False
    for line in output.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "PID" and "VRAM" in line.upper():
            header_seen = True
            continue
        if not header_seen or len(tokens) < 6:
            continue
        try:
            row_pid = int(tokens[0])
        except ValueError:
            continue
        if row_pid != pid:
            continue
        vram_token = tokens[-3]
        if vram_token.upper() == "UNKNOWN":
            return None
        try:
            vram_bytes = float(vram_token)
        except ValueError:
            return None
        return vram_bytes / (1024 * 1024)
    return None


def read_amd_gpu_vram_mb(pid: int) -> float | None:
    """Reads current AMD GPU VRAM usage for `pid` in MB via `rocm-smi --showpids`.

    Returns None if `rocm-smi` is not installed, the call errors or times out, or the
    PID does not currently appear in its process table -- e.g. no AMD GPU present, a
    CPU-only backend, or the process hasn't allocated GPU memory yet. Mirrors
    `read_gpu_vram_mb`'s failure handling so both vendors degrade the same way.
    """
    try:
        result = subprocess.run(
            ["rocm-smi", "--showpids"],
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return _parse_rocm_showpids_vram_mb(result.stdout, pid)


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


class _PollingSampler:
    """Shared polling/threading logic: calls `read_fn(pid)` on a background daemon
    thread at a fixed interval until stopped, then reduces the collected samples to
    peak/mean. Runs as a daemon thread so a crash mid-benchmark never hangs the CLI
    waiting on the sampler. Subclasses just fix `read_fn` and a default interval --
    kept as subclasses (rather than one directly-instantiated class) so callers and
    tests keep referring to `RssSampler`/`GpuVramSampler`/`AmdGpuVramSampler` by the
    metric they sample, not by an implementation-detail read function.
    """

    def __init__(self, pid: int, read_fn: Callable[[int], float | None], interval_s: float):
        self.pid = pid
        self.interval_s = interval_s
        self._read_fn = read_fn
        self._samples: list[float] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = self._read_fn(self.pid)
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


class RssSampler(_PollingSampler):
    """Polls a PID's host RSS on a background thread at a fixed interval until stopped."""

    def __init__(self, pid: int, interval_s: float = 0.1):
        super().__init__(pid, read_rss_mb, interval_s)


class GpuVramSampler(_PollingSampler):
    """Polls a PID's NVIDIA GPU VRAM usage via `nvidia-smi`, mirroring `RssSampler`.

    Defaults to a slower poll interval than `RssSampler` (0.5s vs 0.1s) because each
    sample shells out to a subprocess -- orders of magnitude slower than a `/proc`
    read, so polling at RSS's rate would add real overhead to the benchmark it's
    trying to measure.
    """

    def __init__(self, pid: int, interval_s: float = 0.5):
        super().__init__(pid, read_gpu_vram_mb, interval_s)


class AmdGpuVramSampler(_PollingSampler):
    """Polls a PID's AMD GPU VRAM usage via `rocm-smi`, mirroring `GpuVramSampler`
    (same subprocess-per-sample cost, same default 0.5s interval)."""

    def __init__(self, pid: int, interval_s: float = 0.5):
        super().__init__(pid, read_amd_gpu_vram_mb, interval_s)
