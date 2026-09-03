"""Aggregates per-backend JSONL run records into a markdown comparison report."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    idx = min(len(sorted_values) - 1, max(0, round(p / 100 * (len(sorted_values) - 1))))
    return sorted_values[idx]


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _load_records(results_dir: str | Path) -> dict[str, list[dict]]:
    results_dir = Path(results_dir)
    by_backend: dict[str, list[dict]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        records = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        if records:
            by_backend[records[0]["backend"]] = records
    return by_backend


def _load_memory_stats(results_dir: str | Path) -> dict[str, dict]:
    results_dir = Path(results_dir)
    by_backend: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.memory.json")):
        backend_name = path.name.removesuffix(".memory.json")
        with path.open(encoding="utf-8") as f:
            by_backend[backend_name] = json.load(f)
    return by_backend


def _load_gpu_memory_stats(results_dir: str | Path) -> dict[str, dict]:
    results_dir = Path(results_dir)
    by_backend: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.gpu_memory.json")):
        backend_name = path.name.removesuffix(".gpu_memory.json")
        with path.open(encoding="utf-8") as f:
            by_backend[backend_name] = json.load(f)
    return by_backend


def build_report(results_dir: str | Path) -> str:
    by_backend = _load_records(results_dir)
    if not by_backend:
        return "# Inference Backend Benchmark Report\n\nNo results found.\n"

    memory_stats = _load_memory_stats(results_dir)
    gpu_memory_stats = _load_gpu_memory_stats(results_dir)

    lines = [
        "# Inference Backend Benchmark Report",
        "",
        "## Research question",
        "",
        "At fixed hardware and model, does inference-server choice materially change",
        "latency and throughput -- enough to justify a migration?",
        "",
        "## Practical impact",
        "",
        "Self-hosted inference cost is dominated by tokens/sec per GPU-hour. A backend",
        "that's meaningfully faster at the same hardware and quality directly lowers",
        "$/million-tokens served.",
        "",
        "## Results",
        "",
        "| backend | runs | errors | mean latency (s) | p50 (s) | p95 (s) | mean TTFT (s) | p95 TTFT (s) | mean tokens/s | mean decode tokens/s | est. token runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    any_errors = False
    for backend, recs in sorted(by_backend.items()):
        ok_recs = [r for r in recs if r.get("error") is None]
        errors = len(recs) - len(ok_recs)
        if errors:
            any_errors = True
        latencies = sorted(r["latency_s"] for r in ok_recs)
        mean_lat = _mean(latencies)
        p50 = _percentile(latencies, 50) if latencies else None
        p95 = _percentile(latencies, 95) if latencies else None
        ttfts = sorted(r["ttft_s"] for r in ok_recs if r.get("ttft_s") is not None)
        mean_ttft = _mean(ttfts)
        p95_ttft = _percentile(ttfts, 95) if ttfts else None
        throughputs = [r["tokens_per_s"] for r in ok_recs if r.get("tokens_per_s")]
        mean_tps = _mean(throughputs)
        decode_tps = [r["decode_tokens_per_s"] for r in ok_recs if r.get("decode_tokens_per_s")]
        mean_decode_tps = _mean(decode_tps)
        estimated_token_runs = sum(1 for r in ok_recs if r.get("tokens_estimated"))
        lines.append(
            f"| {backend} | {len(recs)} | {errors} | {_format_float(mean_lat)} | "
            f"{_format_float(p50)} | {_format_float(p95)} | "
            f"{_format_float(mean_ttft)} | {_format_float(p95_ttft)} | "
            f"{_format_float(mean_tps, 2)} | {_format_float(mean_decode_tps, 2)} | "
            f"{estimated_token_runs} |"
        )

    lines.extend(
        [
            "",
            "TTFT and decode throughput are populated only for runs captured with",
            "`run --stream`. If a server does not report usage in streaming mode,",
            "completion tokens are estimated from streamed text and counted above.",
            "Latency/TTFT/throughput statistics exclude errored runs -- a connection",
            "failure's elapsed time isn't a real response latency, so mixing it in would",
            "distort the comparison rather than inform it.",
        ]
    )

    if any_errors:
        lines.append("")
        lines.append("### Errors")
        lines.append("")
        lines.append(
            "A failed request (connection refused, timeout, non-2xx status, or an"
            " unparseable response body) is recorded rather than aborting the rest of"
            " the benchmark. Distinct error messages observed per backend:"
        )
        lines.append("")
        for backend, recs in sorted(by_backend.items()):
            messages = sorted({r["error"] for r in recs if r.get("error") is not None})
            if messages:
                lines.append(f"- **{backend}**: " + "; ".join(messages))

    lines.append("")
    lines.append("### By workload bucket (short/medium/long)")
    lines.append("")
    buckets = sorted({r["item_id"].split("-")[0] for recs in by_backend.values() for r in recs})
    header = ["backend", *[f"{b} mean latency (s)" for b in buckets]]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for backend, recs in sorted(by_backend.items()):
        by_bucket = defaultdict(list)
        for r in recs:
            by_bucket[r["item_id"].split("-")[0]].append(r["latency_s"])
        row = [backend]
        for b in buckets:
            vals = by_bucket.get(b, [])
            row.append(f"{(sum(vals) / len(vals)):.3f}" if vals else "-")
        lines.append("| " + " | ".join(row) + " |")

    if memory_stats:
        lines.append("")
        lines.append("### Serving process memory (RSS)")
        lines.append("")
        lines.append(
            "Sampled from `/proc/<pid>/status` on the configured server PID while the "
            "backend's runs executed -- requires the harness and server to share a host "
            "(or /proc namespace) and a `pid` set in the backend config."
        )
        lines.append("")
        lines.append("| backend | peak RSS (MB) | mean RSS (MB) | samples | memory picture |")
        lines.append("| --- | ---: | ---: | ---: | --- |")
        any_complete = False
        for backend in sorted(memory_stats):
            stats = memory_stats[backend]
            gpu_stats = gpu_memory_stats.get(backend, {})
            if gpu_stats.get("sample_count", 0) > 0:
                picture = "supplementary (see GPU VRAM below)"
            else:
                picture = "complete -- no GPU VRAM observed"
                any_complete = True
            lines.append(
                f"| {backend} | {_format_float(stats.get('peak_mb'), 1)} | "
                f"{_format_float(stats.get('mean_mb'), 1)} | {stats.get('sample_count', 0)} | "
                f"{picture} |"
            )
        if any_complete:
            lines.append("")
            lines.append(
                '"Complete" means no GPU VRAM samples were collected for that backend -- '
                "either no discrete GPU was present (e.g. Apple Silicon unified memory or a "
                "CPU-only backend) or `nvidia-smi`/`rocm-smi` was unavailable. In that case "
                "host RSS already is the full memory picture, not a fallback standing in for "
                "a metric the harness failed to collect."
            )

    if gpu_memory_stats:
        lines.append("")
        lines.append("### Serving process GPU VRAM")
        lines.append("")
        lines.append(
            "Sampled from `nvidia-smi --query-compute-apps` (NVIDIA) or `rocm-smi "
            "--showpids` (AMD) on the configured server PID while the backend's runs "
            "executed -- requires a matching GPU + CLI tool on PATH and a `pid` set in "
            "the backend config (`gpu_vendor: amd` to select rocm-smi; defaults to "
            "nvidia-smi). `sample_count: 0` means the tool was unavailable or the PID "
            "never showed up as a GPU compute process (e.g. a CPU-only backend), not "
            "that VRAM usage was zero."
        )
        lines.append("")
        lines.append("| backend | vendor | peak VRAM (MB) | mean VRAM (MB) | samples |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for backend in sorted(gpu_memory_stats):
            stats = gpu_memory_stats[backend]
            lines.append(
                f"| {backend} | {stats.get('vendor', 'nvidia')} | "
                f"{_format_float(stats.get('peak_mb'), 1)} | "
                f"{_format_float(stats.get('mean_mb'), 1)} | {stats.get('sample_count', 0)} |"
            )

    return "\n".join(lines) + "\n"
