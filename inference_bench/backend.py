"""Backend abstraction over local/self-hosted LLM inference servers.

OpenAICompatBackend talks to any server exposing an OpenAI-compatible
/v1/chat/completions endpoint -- which covers Ollama, vLLM, llama.cpp's server mode,
and most self-hosted open-weight model deployments. MockBackend exists so the harness
is runnable and testable with zero servers and zero network access.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BackendSpec:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_s: float = 120.0
    pid: int | None = None  # serving process PID, for optional RSS/GPU sampling
    gpu_vendor: str = "nvidia"  # "nvidia" or "amd" -- picks the GPU VRAM sampler


@dataclass(frozen=True)
class RequestResult:
    latency_s: float
    completion_tokens: int | None
    tokens_per_s: float | None  # None if the server didn't report usage
    ttft_s: float | None = None
    decode_tokens_per_s: float | None = None
    tokens_estimated: bool = False
    error: str | None = None  # set instead of raising when a request fails


# Exceptions a single flaky request against a real server can raise: connection
# refused/reset/DNS failure (URLError), a non-2xx HTTP status (HTTPError, a
# URLError subclass), a client-side timeout, or a response body that isn't valid
# JSON (a crashed/misbehaving server). One bad run shouldn't take down the rest of
# the benchmark, so `OpenAICompatBackend.complete` catches these and reports them
# on the `RequestResult` instead of letting them propagate out of `run_benchmark`.
_REQUEST_ERRORS = (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError)


def _error_result(latency_s: float, exc: Exception) -> RequestResult:
    if isinstance(exc, urllib.error.HTTPError):
        message = f"HTTPError: {exc.code} {exc.reason}"
    else:
        message = f"{type(exc).__name__}: {exc}"
    return RequestResult(
        latency_s=latency_s,
        completion_tokens=None,
        tokens_per_s=None,
        error=message,
    )


class Backend:
    def complete(
        self,
        spec: BackendSpec,
        prompt: str,
        max_tokens: int,
        stream: bool = False,
    ) -> RequestResult:
        raise NotImplementedError


def _estimate_tokens(text: str) -> int | None:
    tokens = [part for part in text.replace("\n", " ").split(" ") if part]
    return len(tokens) or None


def _result_from_measurement(
    latency_s: float,
    completion_tokens: int | None,
    ttft_s: float | None = None,
    tokens_estimated: bool = False,
) -> RequestResult:
    tokens_per_s = completion_tokens / latency_s if completion_tokens and latency_s > 0 else None
    decode_duration_s = latency_s - ttft_s if ttft_s is not None else None
    decode_tokens_per_s = (
        completion_tokens / decode_duration_s
        if completion_tokens and decode_duration_s and decode_duration_s > 0
        else None
    )
    return RequestResult(
        latency_s=latency_s,
        completion_tokens=completion_tokens,
        tokens_per_s=tokens_per_s,
        ttft_s=ttft_s,
        decode_tokens_per_s=decode_tokens_per_s,
        tokens_estimated=tokens_estimated,
    )


def _parse_sse_data_line(raw_line: bytes) -> dict[str, Any] | None:
    line = raw_line.decode("utf-8").strip()
    if not line.startswith("data:"):
        return None
    payload = line.removeprefix("data:").strip()
    if not payload or payload == "[DONE]":
        return None
    return json.loads(payload)


class OpenAICompatBackend(Backend):
    def complete(
        self,
        spec: BackendSpec,
        prompt: str,
        max_tokens: int,
        stream: bool = False,
    ) -> RequestResult:
        api_key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else ""
        payload_obj: dict[str, Any] = {
            "model": spec.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if stream:
            payload_obj["stream"] = True
            payload_obj["stream_options"] = {"include_usage": True}

        payload = json.dumps(payload_obj).encode("utf-8")
        req = urllib.request.Request(
            url=f"{spec.base_url.rstrip('/')}/chat/completions",
            data=payload,
            method="POST",
            headers={
                "Content-Type": "application/json",
                **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
            },
        )
        start = time.monotonic()
        if stream:
            return self._complete_streaming(req, spec.timeout_s, start)

        try:
            with urllib.request.urlopen(req, timeout=spec.timeout_s) as resp:
                body = json.loads(resp.read())
        except _REQUEST_ERRORS as exc:
            return _error_result(time.monotonic() - start, exc)
        latency = time.monotonic() - start

        usage = body.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        return _result_from_measurement(latency, completion_tokens)

    def _complete_streaming(
        self,
        req: urllib.request.Request,
        timeout_s: float,
        start: float,
    ) -> RequestResult:
        first_token_at: float | None = None
        completion_tokens: int | None = None
        content_parts: list[str] = []

        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    event = _parse_sse_data_line(raw_line)
                    if event is None:
                        continue

                    usage = event.get("usage") or {}
                    if usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])

                    for choice in event.get("choices", []):
                        delta = choice.get("delta") or {}
                        content = delta.get("content") or ""
                        if content:
                            if first_token_at is None:
                                first_token_at = time.monotonic()
                            content_parts.append(content)
        except _REQUEST_ERRORS as exc:
            return _error_result(time.monotonic() - start, exc)

        latency = time.monotonic() - start
        tokens_estimated = False
        if completion_tokens is None and content_parts:
            completion_tokens = _estimate_tokens("".join(content_parts))
            tokens_estimated = completion_tokens is not None

        ttft_s = first_token_at - start if first_token_at is not None else None
        return _result_from_measurement(
            latency,
            completion_tokens,
            ttft_s=ttft_s,
            tokens_estimated=tokens_estimated,
        )


class MockBackend(Backend):
    """Deterministic, seeded-by-content stand-in: simulates a fixed per-token generation
    rate plus a fixed request overhead, so the harness has something real to exercise
    end-to-end without a running server. Not a claim about any real backend's speed."""

    def __init__(
        self,
        tokens_per_s: float = 40.0,
        overhead_s: float = 0.05,
        ttft_s: float | None = None,
    ):
        self.tokens_per_s = tokens_per_s
        self.overhead_s = overhead_s
        self.ttft_s = ttft_s

    def complete(
        self,
        spec: BackendSpec,
        prompt: str,
        max_tokens: int,
        stream: bool = False,
    ) -> RequestResult:
        simulated_latency = self.overhead_s + max_tokens / self.tokens_per_s
        ttft_s = None
        if stream:
            ttft_s = min(
                self.ttft_s if self.ttft_s is not None else self.overhead_s,
                simulated_latency,
            )
        return _result_from_measurement(simulated_latency, max_tokens, ttft_s=ttft_s)


def get_backend(kind: str, **kwargs) -> Backend:
    if kind == "openai_compat":
        return OpenAICompatBackend()
    if kind == "mock":
        return MockBackend(**kwargs)
    raise ValueError(f"unknown backend kind {kind!r}; expected 'openai_compat' or 'mock'")
