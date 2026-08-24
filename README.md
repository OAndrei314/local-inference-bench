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
  writes peak/mean RSS to `<backend>.memory.json`. It also samples GPU VRAM for the same
  PID via `nvidia-smi --query-compute-apps`, on a slower interval since each sample is a
  subprocess call, and writes peak/mean VRAM to `<backend>.gpu_memory.json`.
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
3. **RSS sampling now covers Linux and macOS; GPU VRAM sampling needs `nvidia-smi` or
   `rocm-smi`.** RSS reads `/proc/<pid>/status` directly on Linux (no subprocess); on
   platforms without `/proc` — macOS, both Intel and Apple Silicon — it falls back to
   shelling out to `ps -o rss= -p <pid>`. Either path requires the harness and the
   serving process to share a host (or PID namespace). Windows isn't covered by
   either, so `read_rss_mb` returns `None` there and the harness just omits it from
   the report. GPU VRAM sampling shells out to `nvidia-smi --query-compute-apps`
   (default) or `rocm-smi --showpids` (`gpu_vendor: amd` in the backend config),
   which requires a matching discrete GPU and driver — on any other host, or for a
   CPU-only backend (e.g. llama.cpp in CPU mode), it returns `None` and the harness
   still writes a `sample_count: 0` file rather than failing, so a missing GPU never
   silently reads as "0 MB used." For most self-hosted GPU serving (vLLM, Ollama with
   a discrete GPU backend), VRAM is the memory pressure that actually constrains
   deployment; host RSS is most useful for CPU-bound backends, for isolating
   host-side overhead, or — on Apple Silicon — as the complete memory picture, since
   its unified memory means there's no separate GPU pool to sample (see next item).
   The report's RSS table labels each backend's row `complete -- no GPU VRAM
   observed` or `supplementary (see GPU VRAM below)` based on whether that
   backend's own `gpu_memory.json` has any real samples (`sample_count > 0`), so a
   reader doesn't have to infer from context whether the RSS number is the whole
   memory picture or just host-side overhead alongside a real VRAM figure.
4. **GPU VRAM aggregation differs by vendor, both intentionally end up as one number
   per PID.** `nvidia-smi --query-compute-apps` emits one row per (GPU, PID) pair, so a
   backend sharded across multiple NVIDIA GPUs is summed here. `rocm-smi --showpids`
   already reports one row per PID with VRAM pre-aggregated across the GPUs it uses, so
   that row is taken as-is rather than summed again. Neither backend gives a
   per-device breakdown when a process spans multiple GPUs.
5. **`gpu_vendor: amd` is untested against real ROCm hardware.** The `rocm-smi
   --showpids` column layout and byte units come from the published CLI docs and
   `rocm_smi_lib` source (`rsmi_compute_process_info_by_pid_get`), not a live run — the
   parsing logic is covered by tests against a hand-built sample table (including its
   `UNKNOWN` VRAM case), but confirming the assumption holds across ROCm versions needs
   an actual run against vLLM/Ollama on an AMD GPU box.

## Status / next steps

GPU VRAM sampling covers both major discrete GPU vendors: `nvidia-smi
--query-compute-apps` (default) and `rocm-smi --showpids` (`gpu_vendor: amd` in the
backend config), keyed by the same serving-process PID used for RSS sampling — see
`<backend>.gpu_memory.json` (tagged with a `vendor` field) and the "Serving process
GPU VRAM" report section. Both remain untested against real GPU hardware in this
environment (the CI runner and sandbox have neither an NVIDIA nor an AMD GPU); the
parsing and threading logic are covered by tests with monkeypatched
`nvidia-smi`/`rocm-smi` output built from each tool's documented format, but a real
run against vLLM/Ollama on actual GPU boxes of both vendors is the honest way to
confirm the output-format assumptions hold across driver versions.

RSS sampling — previously Linux-only via `/proc/<pid>/status` — now also covers
macOS (Intel and Apple Silicon) via a `ps -o rss= -p <pid>` fallback, chosen because
`read_rss_mb` already dispatches on whether `/proc` exists, so this is the natural
place to add the second platform rather than a new function callers have to know
about. This directly answers the unified-memory gap noted here previously: Apple
Silicon has no discrete VRAM pool to sample the way `nvidia-smi`/`rocm-smi` do, so
host RSS *is* the complete GPU-resident memory picture for an MLX or llama.cpp
Metal backend there — no third GPU-vendor sampler branch was needed after all. The
`ps` fallback is exercised by a real subprocess in CI (Linux, calling
`_read_rss_mb_via_ps` directly rather than through the `/proc`-preferring dispatch,
and cross-checked against the same process's `/proc` reading to confirm the two
paths agree), but the macOS branch of `read_rss_mb` itself — where `Path("/proc")`
genuinely doesn't exist — has only been exercised via a monkeypatched dispatch test,
not a real Mac; that's the honest gap left in this feature.

The report now labels each backend's RSS row `complete -- no GPU VRAM observed` or
`supplementary (see GPU VRAM below)`, based on whether that backend's
`gpu_memory.json` actually collected any samples (`sample_count > 0`) rather than
just being present. This was possible without any new sampling logic because
`runner.py` already starts the RSS and GPU samplers together whenever a backend
sets `pid` (see `run_benchmark`), so the two result files for a given backend
always exist as a pair — the report just had to read both to decide which label
applies, instead of leaving a reader to infer from the surrounding prose whether a
given RSS number is the whole memory picture or host-side overhead next to a real
VRAM figure. Covered by `tests/test_end_to_end.py`, both the null-GPU-stats and the
monkeypatched-real-GPU-stats cases.

The one remaining honest gap: an actual run on Apple hardware to confirm `ps`'s
column format holds across macOS versions, mirroring the same caveat already
carried by the untested ROCm path above. Both are real-hardware verification gaps
that no amount of additional mocking in this sandbox can close.

## License

MIT — see [LICENSE](LICENSE).
