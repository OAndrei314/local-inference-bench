"""`python -m inference_bench.cli run|report ...`"""
from __future__ import annotations

import argparse

from .report import build_report
from .runner import load_backend_specs, run_benchmark
from .workload import DEFAULT_WORKLOAD


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="inference-bench")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the workload against configured backends")
    run_p.add_argument("--config", required=True, help="path to backends YAML config")
    run_p.add_argument("--repeats", type=int, default=3)
    run_p.add_argument("--out", required=True, help="output directory for result JSONL files")

    report_p = sub.add_parser("report", help="build a markdown report from results")
    report_p.add_argument("--results", required=True, help="results directory (from `run --out`)")
    report_p.add_argument("--out", required=True, help="output markdown file path")

    args = parser.parse_args(argv)

    if args.command == "run":
        specs = load_backend_specs(args.config)
        run_benchmark(DEFAULT_WORKLOAD, specs, args.repeats, args.out)
        return 0

    if args.command == "report":
        report = build_report(args.results)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(report)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
