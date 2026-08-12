# local-inference-bench

*Maintained by: claude-actions-daily-routine · Status: Active*
A small, dependency-light harness for benchmarking self-hosted LLM inference backends
(Ollama, vLLM, llama.cpp server mode, or anything else exposing an OpenAI-compatible
`/v1/chat/completions` endpoint) on latency and throughput, instead of trusting vendor
marketing numbers for your specific hardware and model.

## Why this matters

**Research question:** at fixed hardware and model, does inference-server choice
materially change latency and throughput — enough to justify a migration?

**Practical impact:** self-hosted inference cost is dominated by tokens/sec per GPU-hour.
A backend that's meaningfully faster at the same hardware and output quality directly
lowers $/million-tokens served — which matters more as more AI infrastructure spend shifts
toward self-hosted open-weight models instead of API providers.

## How it works

- `inference_bench/backend.py` — `OpenAICompatBackend` talks to any real
  `/v1/chat/completions` server using only the standard library. `MockBackend` simulates a
  fixed tokens/sec rate + request overhead, so the pipeline is fully testable with zero
  servers and zero network access.
- `inference_bench/workload.py` — a fixed 6-prompt workload spanning short/medium/long
  target output lengths, so results show how latency scales with generation length instead
  of one aggregate number.
- `inference_bench/runner.py` — runs the workload against each configured backend
  `--repeats` times, writes per-run JSONL, and can request streaming responses to measure
  time-to-first-token.
- `inference_bench/memory.py` — if a backend config sets `pid`, samples that process's
  RSS from `/proc/<pid>/status` on a background thread for the duration of its runs and
  writes peak/mean RSS to `<backend>.memory.json`.
- `inference_bench/report.py` — aggregates into mean/p50/p95 latency and mean tokens/sec,
  streaming TTFT, decode throughput, overall results, workload buckets, and (when
  available) serving-process memory.

## Quickstart

```bash
pip install -r requirements.txt

# Zero-setup run against two mock backends (no server needed)
python -m inference_bench.cli run --config configs/mock.yaml --repeats 3 --out results/mock
python -m inference_bench.cli report --results results/mock --out report.md

# Streaming mode adds TTFT/decode metrics when the backend supports SSE streaming
python -m inference_bench.cli run --config configs/mock.yaml --repeats 3 --out results/mock-stream --stream

# Real run: copy configs/backends.example.yaml, point base_url at a running
# Ollama/vLLM/llama.cpp server, then:
python -m inference_bench.cli run --config configs/backends.yaml --repeats 5 --out results/live

# Add `pid: <server PID>` to a backend entry (e.g. `pid: $(pgrep -f vllm)`) to also
# capture its peak/mean RSS for that run -- requires the harness and server on the
# same host.
```

## Honest scope limits

1. **TTFT requires `--stream`.** Non-streaming runs still report total latency and
   throughput only. Streaming runs measure the first content chunk from the server. If the
   server does not return `usage.completion_tokens` in streaming mode, the harness falls
   back to a rough whitespace-token estimate and flags those runs in the report.
2. **The mock numbers below are a pipeline demonstration, not a backend comparison.** They
   come from `MockBackend`'s two hardcoded rate/overhead presets, not real servers:

   | backend | runs | mean latency (s) | p50 (s) | p95 (s) | mean TTFT (s) | mean tokens/s | mean decode tokens/s |
   | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
   | mock-fast | 18 | 1.397 | 0.830 | 3.230 | 0.030 | 72.64 | 80.00 |
   | mock-slow | 18 | 4.523 | 2.710 | 10.390 | 0.150 | 21.76 | 25.00 |

   A real comparison requires actually running two backends on the same hardware and model
   — that's on you (or whoever clones this) to do, not something a mock can substitute for.
3. **Memory sampling is host RSS, not GPU VRAM, and Linux-only.** It reads
   `/proc/<pid>/status`, which requires the harness and the serving process to share a
   host (or /proc namespace) and does not exist on macOS/Windows — `read_rss_mb` returns
   `None` there and the harness just omits the memory section rather than failing. For
   most self-hosted LLM serving (vLLM, Ollama with a GPU backend), the memory pressure
   that actually matters is GPU VRAM, which host RSS does not capture at all — it's most
   useful for CPU-bound backends (e.g. llama.cpp in CPU mode) or for isolating host-side
   overhead.

## Status / next steps

GPU VRAM sampling (via `nvidia-smi --query-compute-apps` keyed by PID, alongside the
existing host RSS sampling) is the natural next step, since it's the memory metric that
actually constrains most self-hosted LLM deployments.

## License

MIT — see [LICENSE](LICENSE).
