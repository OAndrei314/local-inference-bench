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


def build_report(results_dir: str | Path) -> str:
    by_backend = _load_records(results_dir)
    if not by_backend:
        return "# Inference Backend Benchmark Report\n\nNo results found.\n"

    lines = [
        "# Inference Backend Benchmark Report",
        "",
        "## Research question",
        "",
        "At fixed hardware and model, does inference-server choice materially change",
        "latency and throughput -- enough to justify a migration?",
        "",
        "## Money question",
        "",
        "Self-hosted inference cost is dominated by tokens/sec per GPU-hour. A backend",
        "that's meaningfully faster at the same hardware and quality directly lowers",
        "$/million-tokens served.",
        "",
        "## Results",
        "",
        "| backend | runs | mean latency (s) | p50 (s) | p95 (s) | mean tokens/s |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for backend, recs in sorted(by_backend.items()):
        latencies = sorted(r["latency_s"] for r in recs)
        mean_lat = sum(latencies) / len(latencies)
        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        throughputs = [r["tokens_per_s"] for r in recs if r.get("tokens_per_s")]
        mean_tps = sum(throughputs) / len(throughputs) if throughputs else float("nan")
        lines.append(
            f"| {backend} | {len(recs)} | {mean_lat:.3f} | {p50:.3f} | {p95:.3f} | {mean_tps:.2f} |"
        )

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

    return "\n".join(lines) + "\n"
