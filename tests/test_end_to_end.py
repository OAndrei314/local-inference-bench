"""End-to-end test using only MockBackend -- no server or network access required."""
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
