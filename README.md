# local-inference-bench

A small, dependency-light harness for benchmarking self-hosted LLM inference backends
(Ollama, vLLM, llama.cpp server mode, or anything else exposing an OpenAI-compatible
`/v1/chat/completions` endpoint) on latency and throughput, instead of trusting vendor
marketing numbers for your specific hardware and model.

## Research + money thesis

**Research question:** at fixed hardware and model, does inference-server choice
materially change latency and throughput — enough to justify a migration?

**Money question:** self-hosted inference cost is dominated by tokens/sec per GPU-hour.
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
  `--repeats` times, writes per-run JSONL.
- `inference_bench/report.py` — aggregates into mean/p50/p95 latency and mean tokens/sec,
  overall and broken down by workload bucket.

## Quickstart

```bash
pip install -r requirements.txt

# Zero-setup run against two mock backends (no server needed)
python -m inference_bench.cli run --config configs/mock.yaml --repeats 3 --out results/mock
python -m inference_bench.cli report --results results/mock --out report.md

# Real run: copy configs/backends.example.yaml, point base_url at a running
# Ollama/vLLM/llama.cpp server, then:
python -m inference_bench.cli run --config configs/backends.yaml --repeats 5 --out results/live
```

## Honest scope limits

1. **This measures total request latency, not true time-to-first-token.** TTFT needs
   streaming (`stream=True`, reading SSE chunks as they arrive) — not implemented here.
   Reported "tokens/s" is `completion_tokens / total_latency`, which is a reasonable
   *throughput* proxy but conflates prefill and decode time, so don't read it as a
   per-decode-step rate.
2. **The mock numbers below are a pipeline demonstration, not a backend comparison.** They
   come from `MockBackend`'s two hardcoded rate/overhead presets, not real servers:

   | backend | runs | mean latency (s) | p50 (s) | p95 (s) | mean tokens/s |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | mock-fast | 18 | 1.397 | 0.830 | 3.230 | 80.00 |
   | mock-slow | 18 | 4.523 | 2.710 | 10.390 | 25.00 |

   A real comparison requires actually running two backends on the same hardware and model
   — that's on you (or whoever clones this) to do, not something a mock can substitute for.

## Status / next steps

Streaming support (for real TTFT) and a memory-footprint measurement (via the serving
process's RSS, where the harness has local access to it) are the natural next steps.

## License

MIT — see [LICENSE](LICENSE).
