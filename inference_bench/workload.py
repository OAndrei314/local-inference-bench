"""A small fixed workload spanning short/medium/long target output lengths, so results
show how latency and throughput scale with generation length rather than a single point."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadItem:
    id: str
    prompt: str
    max_tokens: int


DEFAULT_WORKLOAD: list[WorkloadItem] = [
    WorkloadItem("short-1", "Reply with just the word 'ready'.", 8),
    WorkloadItem("short-2", "What is 2+2? Answer with a single number.", 8),
    WorkloadItem("medium-1", "Summarize what a Fabry-Perot laser is in 2 sentences.", 64),
    WorkloadItem("medium-2", "List 5 causes of optical link degradation, one per line.", 64),
    WorkloadItem("long-1", "Write a detailed explanation of coherent detection in optical communications.", 256),
    WorkloadItem("long-2", "Explain, step by step, how tabular Q-learning updates its value estimates.", 256),
]


def workload_by_bucket(workload: list[WorkloadItem]) -> dict[str, list[WorkloadItem]]:
    buckets: dict[str, list[WorkloadItem]] = {}
    for item in workload:
        bucket = item.id.split("-")[0]
        buckets.setdefault(bucket, []).append(item)
    return buckets
