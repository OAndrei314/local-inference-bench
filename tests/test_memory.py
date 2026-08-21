"""Tests for RSS and GPU VRAM memory sampling. Uses only local subprocesses and
monkeypatched subprocess calls -- no network access, no GPU required."""
import os
import subprocess
import sys
import time
from pathlib import Path

from inference_bench.memory import (
    AmdGpuVramSampler,
    GpuVramSampler,
    MemoryStats,
    RssSampler,
    _read_rss_mb_via_proc,
    _read_rss_mb_via_ps,
    read_amd_gpu_vram_mb,
    read_gpu_vram_mb,
    read_rss_mb,
)


def test_read_rss_mb_for_current_process_is_positive():
    rss = read_rss_mb(os.getpid())
    assert rss is not None
    assert rss > 0


def test_read_rss_mb_for_nonexistent_pid_returns_none():
    # PID unlikely to exist; if it does, this environment is unusual enough to skip.
    unlikely_pid = 2**22
    assert read_rss_mb(unlikely_pid) is None


def test_rss_sampler_collects_samples_from_real_subprocess():
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; hold = [0] * 2_000_000; time.sleep(0.4)"]
    )
    try:
        sampler = RssSampler(proc.pid, interval_s=0.02)
        sampler.start()
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
    stats = sampler.stop()

    assert stats.sample_count > 0
    assert stats.peak_mb is not None
    assert stats.mean_mb is not None
    assert stats.peak_mb >= stats.mean_mb > 0


def test_rss_sampler_yields_no_samples_when_pid_is_unreadable(monkeypatch):
    import inference_bench.memory as memory_module

    monkeypatch.setattr(memory_module, "read_rss_mb", lambda pid: None)

    sampler = RssSampler(pid=1, interval_s=0.02)
    sampler.start()
    time.sleep(0.06)  # let a couple of poll cycles run
    stats = sampler.stop()

    assert stats == MemoryStats(peak_mb=None, mean_mb=None, sample_count=0)


def test_memory_stats_to_dict_rounds_and_handles_none():
    stats = MemoryStats(peak_mb=12.3456, mean_mb=10.001, sample_count=5)
    assert stats.to_dict() == {"peak_mb": 12.35, "mean_mb": 10.0, "sample_count": 5}

    empty = MemoryStats(peak_mb=None, mean_mb=None, sample_count=0)
    assert empty.to_dict() == {"peak_mb": None, "mean_mb": None, "sample_count": 0}


def test_read_rss_mb_dispatches_to_proc_when_proc_exists(monkeypatch):
    import inference_bench.memory as memory_module

    monkeypatch.setattr(Path, "is_dir", lambda self: True)
    monkeypatch.setattr(memory_module, "_read_rss_mb_via_proc", lambda pid: 111.0)
    monkeypatch.setattr(memory_module, "_read_rss_mb_via_ps", lambda pid: 222.0)
    assert read_rss_mb(1234) == 111.0


def test_read_rss_mb_falls_back_to_ps_when_no_proc(monkeypatch):
    import inference_bench.memory as memory_module

    monkeypatch.setattr(Path, "is_dir", lambda self: False)
    monkeypatch.setattr(memory_module, "_read_rss_mb_via_proc", lambda pid: 111.0)
    monkeypatch.setattr(memory_module, "_read_rss_mb_via_ps", lambda pid: 222.0)
    assert read_rss_mb(1234) == 222.0


def test_read_rss_mb_via_ps_for_current_process_is_positive():
    # Real `ps` subprocess (available on both Linux and macOS runners), bypassing
    # the /proc dispatch to exercise the macOS-fallback code path directly.
    rss = _read_rss_mb_via_ps(os.getpid())
    assert rss is not None
    assert rss > 0


def test_read_rss_mb_via_proc_for_current_process_matches_via_ps_within_tolerance():
    # Cross-check: on this Linux CI host both paths read the same process's RSS,
    # so they should agree closely even though one reads /proc and the other
    # shells out to `ps` (small drift is expected -- they sample microseconds apart).
    proc_rss = _read_rss_mb_via_proc(os.getpid())
    ps_rss = _read_rss_mb_via_ps(os.getpid())
    assert proc_rss is not None and ps_rss is not None
    assert abs(proc_rss - ps_rss) / proc_rss < 0.5


def test_read_rss_mb_via_ps_returns_none_when_ps_missing(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("no ps on PATH")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert _read_rss_mb_via_ps(1234) is None


def test_read_rss_mb_via_ps_returns_none_for_exited_process(monkeypatch):
    # `ps -o rss= -p <gone>` exits nonzero with empty stdout once the PID is gone.
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=1, stdout="", stderr=""),
    )
    assert _read_rss_mb_via_ps(1234) is None


def test_read_rss_mb_via_ps_parses_kb_output_to_mb(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=0, stdout="20480\n", stderr=""),
    )
    assert _read_rss_mb_via_ps(1234) == 20.0


def test_read_rss_mb_via_ps_returns_none_on_unparseable_output(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=0, stdout="garbage\n", stderr=""),
    )
    assert _read_rss_mb_via_ps(1234) is None


def _fake_nvidia_smi(stdout: str, returncode: int = 0):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=returncode, stdout=stdout, stderr="")

    return _run


def test_read_gpu_vram_mb_returns_none_when_nvidia_smi_missing(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("no nvidia-smi on PATH")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert read_gpu_vram_mb(1234) is None


def test_read_gpu_vram_mb_returns_none_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_nvidia_smi("", returncode=1))
    assert read_gpu_vram_mb(1234) is None


def test_read_gpu_vram_mb_returns_none_when_pid_absent_from_output(monkeypatch):
    stdout = "5555, 1024\n6666, 2048\n"
    monkeypatch.setattr(subprocess, "run", _fake_nvidia_smi(stdout))
    assert read_gpu_vram_mb(1234) is None


def test_read_gpu_vram_mb_sums_matching_rows_across_gpus(monkeypatch):
    # Same pid appears twice -- once per GPU it holds compute memory on.
    stdout = "1234, 1024\n5555, 4096\n1234, 2048\n"
    monkeypatch.setattr(subprocess, "run", _fake_nvidia_smi(stdout))
    assert read_gpu_vram_mb(1234) == 3072.0


def test_gpu_vram_sampler_collects_monkeypatched_samples(monkeypatch):
    import inference_bench.memory as memory_module

    values = iter([500.0, 750.0, 250.0])
    monkeypatch.setattr(memory_module, "read_gpu_vram_mb", lambda pid: next(values, None))

    sampler = GpuVramSampler(pid=1, interval_s=0.02)
    sampler.start()
    time.sleep(0.1)
    stats = sampler.stop()

    assert stats.sample_count >= 1
    assert stats.peak_mb is not None
    assert stats.mean_mb is not None
    assert stats.peak_mb >= stats.mean_mb


def test_gpu_vram_sampler_yields_no_samples_when_nvidia_smi_unavailable(monkeypatch):
    import inference_bench.memory as memory_module

    monkeypatch.setattr(memory_module, "read_gpu_vram_mb", lambda pid: None)

    sampler = GpuVramSampler(pid=1, interval_s=0.02)
    sampler.start()
    time.sleep(0.06)
    stats = sampler.stop()

    assert stats == MemoryStats(peak_mb=None, mean_mb=None, sample_count=0)


# --- AMD (rocm-smi) coverage -------------------------------------------------------
#
# `rocm-smi --showpids` output, per the ROCm CLI docs/source: a banner, then a table
# with columns `PID  PROCESS NAME  GPU(s)  VRAM USED  SDMA USED  CU OCCUPANCY`, VRAM
# reported in bytes and one row already aggregated per PID (not per PID+GPU pair like
# nvidia-smi). 2097152000 B == 2000.0 MB and 4294967296 B == 4096.0 MB exactly, chosen
# so the /1048576 conversion is checkable by inspection.
_ROCM_SHOWPIDS_OUTPUT = """\
======================= ROCm System Management Interface =======================
================================== Concurrent Processes ==================================
PID     PROCESS NAME              GPU(s)   VRAM USED    SDMA USED   CU OCCUPANCY
1234    python                    0        2097152000   0           0
5678    vllm                      0,1      4294967296   0           15
9999    ghost                     0        UNKNOWN      0           0
====================================================================================
================================== End of ROCm SMI Log ===================================
"""


def _fake_rocm_smi(stdout: str, returncode: int = 0):
    def _run(*args, **kwargs):
        return subprocess.CompletedProcess(args, returncode=returncode, stdout=stdout, stderr="")

    return _run


def test_parse_rocm_showpids_vram_mb_finds_matching_pid():
    from inference_bench.memory import _parse_rocm_showpids_vram_mb

    assert _parse_rocm_showpids_vram_mb(_ROCM_SHOWPIDS_OUTPUT, 1234) == 2000.0
    assert _parse_rocm_showpids_vram_mb(_ROCM_SHOWPIDS_OUTPUT, 5678) == 4096.0


def test_parse_rocm_showpids_vram_mb_returns_none_for_absent_pid():
    from inference_bench.memory import _parse_rocm_showpids_vram_mb

    assert _parse_rocm_showpids_vram_mb(_ROCM_SHOWPIDS_OUTPUT, 4242) is None


def test_parse_rocm_showpids_vram_mb_returns_none_for_unknown_vram():
    from inference_bench.memory import _parse_rocm_showpids_vram_mb

    # rocm-smi reports the literal string UNKNOWN when the driver can't attribute
    # usage to a PID -- must not be parsed as 0 MB.
    assert _parse_rocm_showpids_vram_mb(_ROCM_SHOWPIDS_OUTPUT, 9999) is None


def test_read_amd_gpu_vram_mb_returns_none_when_rocm_smi_missing(monkeypatch):
    def _raise(*args, **kwargs):
        raise FileNotFoundError("no rocm-smi on PATH")

    monkeypatch.setattr(subprocess, "run", _raise)
    assert read_amd_gpu_vram_mb(1234) is None


def test_read_amd_gpu_vram_mb_returns_none_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_rocm_smi("", returncode=1))
    assert read_amd_gpu_vram_mb(1234) is None


def test_read_amd_gpu_vram_mb_parses_matching_pid_from_subprocess_output(monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_rocm_smi(_ROCM_SHOWPIDS_OUTPUT))
    assert read_amd_gpu_vram_mb(5678) == 4096.0
    assert read_amd_gpu_vram_mb(4242) is None


def test_amd_gpu_vram_sampler_collects_monkeypatched_samples(monkeypatch):
    import inference_bench.memory as memory_module

    values = iter([500.0, 750.0, 250.0])
    monkeypatch.setattr(memory_module, "read_amd_gpu_vram_mb", lambda pid: next(values, None))

    sampler = AmdGpuVramSampler(pid=1, interval_s=0.02)
    sampler.start()
    time.sleep(0.1)
    stats = sampler.stop()

    assert stats.sample_count >= 1
    assert stats.peak_mb is not None
    assert stats.mean_mb is not None
    assert stats.peak_mb >= stats.mean_mb


def test_amd_gpu_vram_sampler_yields_no_samples_when_rocm_smi_unavailable(monkeypatch):
    import inference_bench.memory as memory_module

    monkeypatch.setattr(memory_module, "read_amd_gpu_vram_mb", lambda pid: None)

    sampler = AmdGpuVramSampler(pid=1, interval_s=0.02)
    sampler.start()
    time.sleep(0.06)
    stats = sampler.stop()

    assert stats == MemoryStats(peak_mb=None, mean_mb=None, sample_count=0)
