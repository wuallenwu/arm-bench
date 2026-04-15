"""
scripts/perf_trends.py — Show benchmark performance trends over time.

Reads append-only JSONL result files from results/ and prints a time-series
table of speedup metrics per (problem, isa, model).

Usage:
    # All problems/models:
    python scripts/perf_trends.py

    # Filter to a specific model:
    python scripts/perf_trends.py --model anthropic/claude-sonnet-4-6

    # Filter to a specific problem:
    python scripts/perf_trends.py --problem loop_001

    # Show per-turn tool usage summary from a trace file:
    python scripts/perf_trends.py --trace traces/loop_001_sve_anthropic_claude-sonnet-4-6_2026-04-14T12-00-00Z.json
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "results"


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def print_trends(model_filter: str | None, problem_filter: str | None):
    jsonl_files = sorted(RESULTS_DIR.glob("*.jsonl"))
    if not jsonl_files:
        print("No .jsonl result files found in results/. Run the benchmark with at least one eval first.")
        return

    # Group records by (problem_id, isa, model)
    groups: dict[tuple, list[dict]] = {}
    for path in jsonl_files:
        records = load_jsonl(path)
        for r in records:
            pid = r.get("problem_id", "")
            isa = r.get("isa", "")
            model = r.get("model", "")
            if model_filter and model_filter not in model:
                continue
            if problem_filter and problem_filter not in pid:
                continue
            key = (pid, isa, model)
            groups.setdefault(key, []).append(r)

    if not groups:
        print("No matching records found.")
        return

    for (pid, isa, model), records in sorted(groups.items()):
        # Sort by timestamp
        records.sort(key=lambda r: r.get("timestamp", ""))

        print(f"\n{'─'*72}")
        print(f"  {pid}  |  isa={isa}  |  model={model}")
        print(f"{'─'*72}")
        header = f"  {'Timestamp':<22}  {'Lvl':>3}  {'vs_scalar':>10}  {'vs_autovec':>10}  {'vs_ref':>8}  {'tools':>5}  {'correct'}"
        print(header)
        print(f"  {'-'*68}")

        for r in records:
            ts = r.get("timestamp", "")[:19]
            lvl = r.get("level", "?")
            vs_s = r.get("speedup_vs_scalar")
            vs_a = r.get("speedup_vs_autovec")
            vs_r = r.get("speedup_vs_ref")
            tools = r.get("tool_calls", "?")
            correct = "✓" if r.get("correct") else "✗"

            def fmt(v):
                return f"{v:.2f}×" if v is not None else "    —  "

            print(f"  {ts:<22}  {lvl:>3}  {fmt(vs_s):>10}  {fmt(vs_a):>10}  {fmt(vs_r):>8}  {tools:>5}  {correct}")

        # Trend summary if ≥2 runs
        correct_runs = [r for r in records if r.get("correct")]
        if len(correct_runs) >= 2:
            first, last = correct_runs[0], correct_runs[-1]
            delta = (last.get("speedup_vs_scalar") or 0) - (first.get("speedup_vs_scalar") or 0)
            direction = "▲" if delta > 0 else ("▼" if delta < 0 else "━")
            print(f"\n  Trend (first→last correct run): speedup_vs_scalar {direction} {delta:+.2f}×")

    print(f"\n{'─'*72}")
    print(f"  Total groups: {len(groups)}")


def print_trace(trace_path: Path):
    data = json.loads(trace_path.read_text())
    print(f"\nTrace: {data.get('problem_id')} | isa={data.get('isa')} | model={data.get('model')}")
    print(f"Timestamp: {data.get('timestamp')}")
    print(f"{'─'*72}")

    for entry in data.get("trace", []):
        turn = entry.get("turn")
        tool = entry.get("tool")
        reasoning = (entry.get("reasoning") or "").strip()
        args = entry.get("args", {})
        result = entry.get("result", {})

        print(f"\n[Turn {turn}] → {tool}")

        if reasoning:
            # Print first 400 chars of reasoning
            preview = reasoning[:400] + ("..." if len(reasoning) > 400 else "")
            for line in preview.splitlines():
                print(f"  thinking: {line}")

        # Args summary (skip large code, show just first line + length)
        for k, v in args.items():
            if k == "code":
                first_line = str(v).splitlines()[0][:80] if v else ""
                print(f"  args.code: {first_line!r}… [{len(str(v))} chars]")
            else:
                print(f"  args.{k}: {v!r}")

        # Result summary
        for k, v in result.items():
            print(f"  result.{k}: {v}")


def main():
    parser = argparse.ArgumentParser(description="Show benchmark performance trends over time")
    parser.add_argument("--model", default=None, help="Filter by model substring")
    parser.add_argument("--problem", default=None, help="Filter by problem ID substring")
    parser.add_argument("--trace", default=None, help="Path to a trace JSON file to inspect")
    args = parser.parse_args()

    if args.trace:
        print_trace(Path(args.trace))
    else:
        print_trends(args.model, args.problem)


if __name__ == "__main__":
    main()
