"""Runs a workload against a backend `repeats` times each, writes one JSONL record per run."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .backend import BackendSpec, get_backend
from .workload import WorkloadItem


def load_backend_specs(config_path: str | Path) -> list[tuple[str, BackendSpec, dict]]:
    """Returns list of (kind, spec, extra_kwargs) -- extra_kwargs feeds MockBackend's
    tunable knobs (tokens_per_s, overhead_s) when kind == 'mock'."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out = []
    for entry in raw["backends"]:
        kind = entry.get("kind", "openai_compat")
        spec = BackendSpec(
            name=entry["name"],
            base_url=entry.get("base_url", ""),
            model=entry.get("model", entry["name"]),
            api_key_env=entry.get("api_key_env"),
        )
        extra = {
            k: v for k, v in entry.items()
            if k in ("tokens_per_s", "overhead_s")
        }
        out.append((kind, spec, extra))
    return out


def run_benchmark(
    workload: list[WorkloadItem],
    backend_specs: list[tuple[str, BackendSpec, dict]],
    repeats: int,
    out_dir: str | Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind, spec, extra in backend_specs:
        backend = get_backend(kind, **extra)
        out_path = out_dir / f"{spec.name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in workload:
                for rep in range(repeats):
                    result = backend.complete(spec, item.prompt, item.max_tokens)
                    record = {
                        "backend": spec.name,
                        "item_id": item.id,
                        "rep": rep,
                        "max_tokens": item.max_tokens,
                        "latency_s": round(result.latency_s, 4),
                        "completion_tokens": result.completion_tokens,
                        "tokens_per_s": (
                            round(result.tokens_per_s, 2) if result.tokens_per_s else None
                        ),
                    }
                    f.write(json.dumps(record) + "\n")
        print(f"[{spec.name}] wrote {len(workload) * repeats} runs -> {out_path}")
