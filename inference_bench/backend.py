"""Backend abstraction over local/self-hosted LLM inference servers.

OpenAICompatBackend talks to any server exposing an OpenAI-compatible
/v1/chat/completions endpoint -- which covers Ollama, vLLM, llama.cpp's server mode,
and most self-hosted open-weight model deployments. MockBackend exists so the harness
is runnable and testable with zero servers and zero network access.

Honest scope note: this measures total request latency and derives tokens/sec from
`usage.completion_tokens / latency`. It does NOT measure true time-to-first-token,
since that needs streaming (`stream=True` + reading the SSE chunks as they arrive) --
a real next step, not faked here with a guessed number.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class BackendSpec:
    name: str
    base_url: str
    model: str
    api_key_env: str | None = None
    timeout_s: float = 120.0


@dataclass(frozen=True)
class RequestResult:
    latency_s: float
    completion_tokens: int | None
    tokens_per_s: float | None  # None if the server didn't report usage


class Backend:
    def complete(self, spec: BackendSpec, prompt: str, max_tokens: int) -> RequestResult:
        raise NotImplementedError


class OpenAICompatBackend(Backend):
    def complete(self, spec: BackendSpec, prompt: str, max_tokens: int) -> RequestResult:
        api_key = os.environ.get(spec.api_key_env, "") if spec.api_key_env else ""
        payload = json.dumps(
            {
                "model": spec.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            }
        ).encode("utf-8")
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
        with urllib.request.urlopen(req, timeout=spec.timeout_s) as resp:
            body = json.loads(resp.read())
        latency = time.monotonic() - start

        usage = body.get("usage", {})
        completion_tokens = usage.get("completion_tokens")
        tokens_per_s = completion_tokens / latency if completion_tokens else None
        return RequestResult(latency, completion_tokens, tokens_per_s)


class MockBackend(Backend):
    """Deterministic, seeded-by-content stand-in: simulates a fixed per-token generation
    rate plus a fixed request overhead, so the harness has something real to exercise
    end-to-end without a running server. Not a claim about any real backend's speed."""

    def __init__(self, tokens_per_s: float = 40.0, overhead_s: float = 0.05):
        self.tokens_per_s = tokens_per_s
        self.overhead_s = overhead_s

    def complete(self, spec: BackendSpec, prompt: str, max_tokens: int) -> RequestResult:
        simulated_latency = self.overhead_s + max_tokens / self.tokens_per_s
        return RequestResult(simulated_latency, max_tokens, self.tokens_per_s)


def get_backend(kind: str, **kwargs) -> Backend:
    if kind == "openai_compat":
        return OpenAICompatBackend()
    if kind == "mock":
        return MockBackend(**kwargs)
    raise ValueError(f"unknown backend kind {kind!r}; expected 'openai_compat' or 'mock'")
