"""End-to-end test using only MockBackend -- no server or network access required."""
import os

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
