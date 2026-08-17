"""Runs a workload against a backend `repeats` times each, writes one JSONL record per run."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .backend import BackendSpec, get_backend
from .memory import AmdGpuVramSampler, GpuVramSampler, RssSampler
from .workload import WorkloadItem

_GPU_SAMPLERS = {"nvidia": GpuVramSampler, "amd": AmdGpuVramSampler}


def load_backend_specs(config_path: str | Path) -> list[tuple[str, BackendSpec, dict]]:
    """Returns list of (kind, spec, extra_kwargs) -- extra_kwargs feeds MockBackend's
    tunable knobs (tokens_per_s, overhead_s) when kind == 'mock'."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    out = []
    for entry in raw["backends"]:
        kind = entry.get("kind", "openai_compat")
        pid = entry.get("pid")
        gpu_vendor = entry.get("gpu_vendor", "nvidia")
        if gpu_vendor not in _GPU_SAMPLERS:
            raise ValueError(
                f"backend {entry.get('name')!r}: unknown gpu_vendor {gpu_vendor!r}; "
                f"expected one of {sorted(_GPU_SAMPLERS)}"
            )
        spec = BackendSpec(
            name=entry["name"],
            base_url=entry.get("base_url", ""),
            model=entry.get("model", entry["name"]),
            api_key_env=entry.get("api_key_env"),
            timeout_s=float(entry.get("timeout_s", 120.0)),
            pid=int(pid) if pid is not None else None,
            gpu_vendor=gpu_vendor,
        )
        extra = {
            k: v for k, v in entry.items()
            if k in ("tokens_per_s", "overhead_s", "ttft_s")
        }
        out.append((kind, spec, extra))
    return out


def run_benchmark(
    workload: list[WorkloadItem],
    backend_specs: list[tuple[str, BackendSpec, dict]],
    repeats: int,
    out_dir: str | Path,
    stream: bool = False,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind, spec, extra in backend_specs:
        backend = get_backend(kind, **extra)
        out_path = out_dir / f"{spec.name}.jsonl"

        sampler = RssSampler(spec.pid) if spec.pid is not None else None
        gpu_sampler = (
            _GPU_SAMPLERS[spec.gpu_vendor](spec.pid) if spec.pid is not None else None
        )
        if sampler is not None:
            sampler.start()
        if gpu_sampler is not None:
            gpu_sampler.start()

        with out_path.open("w", encoding="utf-8") as f:
            for item in workload:
                for rep in range(repeats):
                    result = backend.complete(spec, item.prompt, item.max_tokens, stream=stream)
                    record = {
                        "backend": spec.name,
                        "item_id": item.id,
                        "rep": rep,
                        "max_tokens": item.max_tokens,
                        "stream": stream,
                        "latency_s": round(result.latency_s, 4),
                        "completion_tokens": result.completion_tokens,
                        "tokens_estimated": result.tokens_estimated,
                        "tokens_per_s": (
                            round(result.tokens_per_s, 2) if result.tokens_per_s else None
                        ),
                        "ttft_s": round(result.ttft_s, 4) if result.ttft_s is not None else None,
                        "decode_tokens_per_s": (
                            round(result.decode_tokens_per_s, 2)
                            if result.decode_tokens_per_s
                            else None
                        ),
                    }
                    f.write(json.dumps(record) + "\n")

        if sampler is not None:
            stats = sampler.stop()
            memory_path = out_dir / f"{spec.name}.memory.json"
            memory_path.write_text(json.dumps(stats.to_dict(), indent=2) + "\n", encoding="utf-8")
            print(f"[{spec.name}] memory: peak={stats.peak_mb} MB mean={stats.mean_mb} MB "
                  f"({stats.sample_count} samples) -> {memory_path}")

        if gpu_sampler is not None:
            gpu_stats = gpu_sampler.stop()
            gpu_memory_path = out_dir / f"{spec.name}.gpu_memory.json"
            gpu_record = {**gpu_stats.to_dict(), "vendor": spec.gpu_vendor}
            gpu_memory_path.write_text(
                json.dumps(gpu_record, indent=2) + "\n", encoding="utf-8"
            )
            print(f"[{spec.name}] {spec.gpu_vendor} gpu vram: peak={gpu_stats.peak_mb} MB "
                  f"mean={gpu_stats.mean_mb} MB ({gpu_stats.sample_count} samples) -> {gpu_memory_path}")

        print(f"[{spec.name}] wrote {len(workload) * repeats} runs -> {out_path}")
