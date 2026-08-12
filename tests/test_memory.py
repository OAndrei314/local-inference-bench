"""Tests for RSS memory sampling. Uses only local subprocesses -- no network access."""
import os
import subprocess
import sys
import time

from inference_bench.memory import MemoryStats, RssSampler, read_rss_mb


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
