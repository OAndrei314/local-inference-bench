from inference_bench.backend import BackendSpec, MockBackend, get_backend


def test_mock_backend_latency_scales_with_max_tokens():
    backend = MockBackend(tokens_per_s=50.0, overhead_s=0.1)
    spec = BackendSpec(name="m", base_url="", model="m")
    short = backend.complete(spec, "hi", max_tokens=10)
    long = backend.complete(spec, "hi", max_tokens=200)
    assert long.latency_s > short.latency_s
    assert short.tokens_per_s == 50.0


def test_mock_backend_reports_requested_completion_tokens():
    backend = MockBackend()
    spec = BackendSpec(name="m", base_url="", model="m")
    result = backend.complete(spec, "hi", max_tokens=42)
    assert result.completion_tokens == 42


def test_get_backend_unknown_kind_raises():
    try:
        get_backend("not_a_real_backend")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_get_backend_mock_passes_kwargs():
    backend = get_backend("mock", tokens_per_s=99.0)
    assert isinstance(backend, MockBackend)
    assert backend.tokens_per_s == 99.0
