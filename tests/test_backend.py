import json
import urllib.error

import inference_bench.backend as backend_module
from inference_bench.backend import (
    BackendSpec,
    MockBackend,
    OpenAICompatBackend,
    _parse_sse_data_line,
    get_backend,
)


def test_mock_backend_latency_scales_with_max_tokens():
    backend = MockBackend(tokens_per_s=50.0, overhead_s=0.1)
    spec = BackendSpec(name="m", base_url="", model="m")
    short = backend.complete(spec, "hi", max_tokens=10)
    long = backend.complete(spec, "hi", max_tokens=200)
    assert long.latency_s > short.latency_s
    assert short.tokens_per_s == 10 / short.latency_s
    assert short.tokens_per_s < 50.0


def test_mock_backend_reports_requested_completion_tokens():
    backend = MockBackend()
    spec = BackendSpec(name="m", base_url="", model="m")
    result = backend.complete(spec, "hi", max_tokens=42)
    assert result.completion_tokens == 42


def test_mock_backend_streaming_reports_ttft_and_decode_rate():
    backend = MockBackend(tokens_per_s=50.0, overhead_s=0.1, ttft_s=0.1)
    spec = BackendSpec(name="m", base_url="", model="m")

    result = backend.complete(spec, "hi", max_tokens=25, stream=True)

    assert result.ttft_s == 0.1
    assert result.decode_tokens_per_s == 50.0
    assert result.tokens_estimated is False


def test_parse_sse_data_line_extracts_json_payload():
    payload = {"choices": [{"delta": {"content": "hello"}}]}
    event = _parse_sse_data_line(f"data: {json.dumps(payload)}\n".encode("utf-8"))

    assert event == payload


def test_parse_sse_data_line_ignores_done_marker():
    assert _parse_sse_data_line(b"data: [DONE]\n") is None


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


def test_openai_compat_backend_reports_connection_error_without_raising(monkeypatch):
    def _raise(req, timeout):
        raise urllib.error.URLError(ConnectionRefusedError("Connection refused"))

    monkeypatch.setattr(backend_module.urllib.request, "urlopen", _raise)
    backend = OpenAICompatBackend()
    spec = BackendSpec(name="b", base_url="http://127.0.0.1:1", model="m")

    result = backend.complete(spec, "hi", max_tokens=8)

    assert result.completion_tokens is None
    assert result.tokens_per_s is None
    assert result.error is not None
    assert "URLError" in result.error
    assert result.latency_s >= 0


def test_openai_compat_backend_formats_http_error_with_status_and_reason(monkeypatch):
    def _raise(req, timeout):
        raise urllib.error.HTTPError(
            url="http://127.0.0.1:1/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr(backend_module.urllib.request, "urlopen", _raise)
    backend = OpenAICompatBackend()
    spec = BackendSpec(name="b", base_url="http://127.0.0.1:1", model="m")

    result = backend.complete(spec, "hi", max_tokens=8)

    assert result.error == "HTTPError: 500 Internal Server Error"


def test_openai_compat_backend_reports_malformed_json_body_as_error(monkeypatch):
    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return b"not json"

    monkeypatch.setattr(
        backend_module.urllib.request, "urlopen", lambda req, timeout: _FakeResponse()
    )
    backend = OpenAICompatBackend()
    spec = BackendSpec(name="b", base_url="http://127.0.0.1:1", model="m")

    result = backend.complete(spec, "hi", max_tokens=8)

    assert result.error is not None
    assert "JSONDecodeError" in result.error


def test_openai_compat_backend_streaming_reports_error_without_raising(monkeypatch):
    def _raise(req, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr(backend_module.urllib.request, "urlopen", _raise)
    backend = OpenAICompatBackend()
    spec = BackendSpec(name="b", base_url="http://127.0.0.1:1", model="m")

    result = backend.complete(spec, "hi", max_tokens=8, stream=True)

    assert result.error is not None
    assert "TimeoutError" in result.error
    assert result.completion_tokens is None
    assert result.ttft_s is None
