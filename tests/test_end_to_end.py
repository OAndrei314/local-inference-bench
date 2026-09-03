"""End-to-end test using only MockBackend -- no server or network access required."""
import json
import os
import urllib.error

from inference_bench.report import build_report
from inference_bench.runner import load_backend_specs, run_benchmark
from inference_bench.workload import DEFAULT_WORKLOAD


def test_full_pipeline_with_mock_backends(tmp_path):
    specs = load_backend_specs("configs/mock.yaml")
    out_dir = tmp_path / "results"

    run_benchmark(DEFAULT_WORKLOAD, specs, repeats=2, out_dir=out_dir)

    result_files = list(out_dir.glob("*.jsonl"))
    assert len(result_files) == len(specs)

    report = build_report(out_dir)
    assert "mock-fast" in report
    assert "mock-slow" in report
    assert "Research question" in report
    assert "mean TTFT" in report


def test_mock_fast_backend_has_lower_mean_latency_than_mock_slow(tmp_path):
    specs = load_backend_specs("configs/mock.yaml")
    out_dir = tmp_path / "results"
    run_benchmark(DEFAULT_WORKLOAD, specs, repeats=3, out_dir=out_dir)

    import json

    def mean_latency(name):
        path = out_dir / f"{name}.jsonl"
        vals = [json.loads(line)["latency_s"] for line in path.read_text().splitlines()]
        return sum(vals) / len(vals)

    assert mean_latency("mock-fast") < mean_latency("mock-slow")


def test_streaming_pipeline_records_ttft(tmp_path):
    specs = load_backend_specs("configs/mock.yaml")
    out_dir = tmp_path / "results"

    run_benchmark(DEFAULT_WORKLOAD[:1], specs, repeats=1, out_dir=out_dir, stream=True)

    import json

    records = [
        json.loads(line)
        for path in out_dir.glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert records
    assert all(r["stream"] is True for r in records)
    assert all(r["ttft_s"] is not None for r in records)
    assert all(r["decode_tokens_per_s"] is not None for r in records)


def test_pipeline_samples_serving_process_memory_when_pid_configured(tmp_path):
    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        f"""
backends:
  - name: self-mock
    kind: mock
    tokens_per_s: 80
    overhead_s: 0.03
    pid: {os.getpid()}
"""
    )
    specs = load_backend_specs(config_path)
    out_dir = tmp_path / "results"

    run_benchmark(DEFAULT_WORKLOAD, specs, repeats=2, out_dir=out_dir)

    memory_path = out_dir / "self-mock.memory.json"
    assert memory_path.exists()

    import json

    stats = json.loads(memory_path.read_text())
    assert stats["sample_count"] >= 1
    assert stats["peak_mb"] > 0
    assert stats["mean_mb"] > 0

    report = build_report(out_dir)
    assert "Serving process memory (RSS)" in report
    assert "self-mock" in report


def test_pipeline_writes_gpu_vram_file_with_null_stats_when_no_gpu_present(tmp_path):
    """Honest-scope-limit test: this sandbox has no NVIDIA GPU / nvidia-smi, so the
    pipeline should still write a gpu_memory.json file (opted in via `pid`) with null
    stats rather than failing -- mirroring how RSS sampling degrades on non-Linux."""
    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        f"""
backends:
  - name: self-mock
    kind: mock
    tokens_per_s: 80
    overhead_s: 0.03
    pid: {os.getpid()}
"""
    )
    specs = load_backend_specs(config_path)
    out_dir = tmp_path / "results"

    run_benchmark(DEFAULT_WORKLOAD, specs, repeats=2, out_dir=out_dir)

    gpu_memory_path = out_dir / "self-mock.gpu_memory.json"
    assert gpu_memory_path.exists()

    import json

    stats = json.loads(gpu_memory_path.read_text())
    assert stats == {"peak_mb": None, "mean_mb": None, "sample_count": 0, "vendor": "nvidia"}

    # No GPU section in the report when every backend's GPU sample_count is 0 --
    # _load_gpu_memory_stats still returns the entry, so the section header does show
    # up, but with an honest all-null row rather than being silently omitted.
    report = build_report(out_dir)
    assert "Serving process GPU VRAM" in report
    # sample_count is 0, so RSS is labeled the complete memory picture for this
    # backend rather than an incomplete fallback for a metric that failed to collect.
    assert "complete -- no GPU VRAM observed" in report
    assert "supplementary" not in report


def test_pipeline_samples_gpu_vram_when_nvidia_smi_available(tmp_path, monkeypatch):
    """Simulates an nvidia-smi-equipped host via monkeypatch, since this sandbox has no
    real GPU -- verifies the pipeline actually plumbs GPU samples through to the report
    when they are available, not just the null-stats degraded path."""
    import inference_bench.memory as memory_module

    pid = os.getpid()
    values = iter([1000.0, 1500.0, 1200.0, 1800.0])
    monkeypatch.setattr(
        memory_module, "read_gpu_vram_mb", lambda p: next(values, None) if p == pid else None
    )

    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        f"""
backends:
  - name: self-mock
    kind: mock
    tokens_per_s: 80
    overhead_s: 0.03
    pid: {pid}
"""
    )
    specs = load_backend_specs(config_path)
    out_dir = tmp_path / "results"

    run_benchmark(DEFAULT_WORKLOAD, specs, repeats=2, out_dir=out_dir)

    import json

    gpu_memory_path = out_dir / "self-mock.gpu_memory.json"
    stats = json.loads(gpu_memory_path.read_text())
    assert stats["sample_count"] >= 1
    assert stats["peak_mb"] > 0
    assert stats["mean_mb"] > 0
    assert stats["vendor"] == "nvidia"

    report = build_report(out_dir)
    assert "Serving process GPU VRAM" in report
    assert "self-mock" in report
    # sample_count > 0 here, so RSS is labeled supplementary to the real GPU VRAM
    # figure rather than claiming to be the complete memory picture.
    assert "supplementary (see GPU VRAM below)" in report
    assert "complete -- no GPU VRAM observed" not in report


def test_pipeline_samples_amd_gpu_vram_when_rocm_smi_available(tmp_path, monkeypatch):
    """Same as the nvidia-smi pipeline test above, but for the `gpu_vendor: amd` path --
    verifies the config knob actually routes to AmdGpuVramSampler/read_amd_gpu_vram_mb
    end to end, not just that the standalone sampler unit works."""
    import inference_bench.memory as memory_module

    pid = os.getpid()
    values = iter([2048.0, 3072.0, 2560.0])
    monkeypatch.setattr(
        memory_module, "read_amd_gpu_vram_mb", lambda p: next(values, None) if p == pid else None
    )

    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        f"""
backends:
  - name: self-mock-amd
    kind: mock
    tokens_per_s: 80
    overhead_s: 0.03
    pid: {pid}
    gpu_vendor: amd
"""
    )
    specs = load_backend_specs(config_path)
    out_dir = tmp_path / "results"

    run_benchmark(DEFAULT_WORKLOAD, specs, repeats=2, out_dir=out_dir)

    import json

    gpu_memory_path = out_dir / "self-mock-amd.gpu_memory.json"
    stats = json.loads(gpu_memory_path.read_text())
    assert stats["sample_count"] >= 1
    assert stats["peak_mb"] > 0
    assert stats["mean_mb"] > 0
    assert stats["vendor"] == "amd"

    report = build_report(out_dir)
    assert "Serving process GPU VRAM" in report
    assert "self-mock-amd" in report
    assert "amd" in report


def test_pipeline_records_backend_errors_without_aborting_the_run(tmp_path, monkeypatch):
    """A failing real-server request (connection refused, timeout, bad response) must be
    recorded per-run rather than crashing the whole benchmark -- and the serving process's
    memory sampler must still be stopped and its stats file written, since the sampler was
    already running when the request failed."""
    import inference_bench.backend as backend_module

    def _raise(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))

    monkeypatch.setattr(backend_module.urllib.request, "urlopen", _raise)

    pid = os.getpid()
    config_path = tmp_path / "backends.yaml"
    config_path.write_text(
        f"""
backends:
  - name: unreachable
    kind: openai_compat
    base_url: http://127.0.0.1:1
    pid: {pid}
"""
    )
    specs = load_backend_specs(config_path)
    out_dir = tmp_path / "results"

    run_benchmark(DEFAULT_WORKLOAD, specs, repeats=1, out_dir=out_dir)

    records = [
        json.loads(line) for line in (out_dir / "unreachable.jsonl").read_text().splitlines()
    ]
    assert len(records) == len(DEFAULT_WORKLOAD)
    assert all(r["error"] is not None for r in records)
    assert all(r["completion_tokens"] is None for r in records)

    memory_path = out_dir / "unreachable.memory.json"
    assert memory_path.exists()
    stats = json.loads(memory_path.read_text())
    assert stats["sample_count"] >= 1

    report = build_report(out_dir)
    assert "### Errors" in report
    assert "unreachable" in report
    assert "Connection refused" in report
