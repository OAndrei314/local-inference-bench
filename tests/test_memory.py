"""Tests for RSS and GPU VRAM memory sampling. Uses only local subprocesses and
monkeypatched subprocess calls -- no network access, no GPU required."""
import os
import subprocess
import sys
import time

from inference_bench.memory import (
    GpuVramSampler,
    MemoryStats,
    RssSampler,
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
