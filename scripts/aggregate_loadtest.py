"""Aggregate multiple Locust load test runs into a single Markdown summary.

Scans a directory for run files named ``u<users>_r<repeat>.json`` together
with matching Prometheus snapshots ``u<users>_r<repeat>_before.txt`` and
``u<users>_r<repeat>_after.txt``.  For every run the script calls
``compute_summary_from_json`` and ``load_server_delta``; it then reconciles
the request counts and rejects any run with failures > 0.

Per user-count it reports the median and (min–max) range of throughput, mean,
p50, p95, p99, and max latency (both client-observed and server-observed), as
well as total requests and total failures across repeats.

Usage:
    .venv\\Scripts\\python.exe -m scripts.aggregate_loadtest \\
        --dir docs/results/loadtest \\
        --out docs/results/loadtest/summary.md

Note:
    Running the file path directly is not supported; invoke via
    -m scripts.aggregate_loadtest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from scripts.summarize_loadtest import (
    compute_summary_from_json,
    load_server_delta,
    reconcile,
)

# ---------------------------------------------------------------------------
# Pattern for run filenames: u<users>_r<repeat>.json
# ---------------------------------------------------------------------------
_RUN_RE = re.compile(r"^u(\d+)_r(\d+)\.json$")

CAVEAT: str = (
    "Measured on one Windows 11 laptop with Docker Desktop (WSL2); "
    "the Locust load generator and the API share the same CPU. "
    "Single uvicorn container. "
    "Latency is client-observed unless marked server. "
    "Percentiles are nearest-rank on Locust's rounded-bucket histogram. "
    "Not a production-throughput claim."
)


# ---------------------------------------------------------------------------
# Stats helpers (operate on plain Python lists; no numpy dependency)
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    """Return the median of a non-empty list using linear interpolation for even n.

    For even n the median is the mean of the two middle values (index n//2-1
    and n//2 of the sorted list).
    """
    if not values:
        raise ValueError("Cannot compute median of empty list")
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _fmt_range(values: list[float], fmt: str = ".2f") -> str:
    """Return 'median (min–max)' string, e.g. '12.34 (10.00–15.00)'."""
    med = _median(values)
    lo = min(values)
    hi = max(values)
    f = f"{{:{fmt}}}"
    return f"{f.format(med)} ({f.format(lo)}–{f.format(hi)})"


# ---------------------------------------------------------------------------
# Run-level data container
# ---------------------------------------------------------------------------


def _load_run(
    json_path: Path,
    before_path: Path,
    after_path: Path,
    users: int,
) -> dict[str, Any]:
    """Parse one run trio and return a dict of per-run metrics.

    Raises:
        ValueError: If reconcile fails, failures > 0, or any required data is
            missing.  The message always names the offending file or metric.
    """
    with json_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    summary = compute_summary_from_json(data)

    if summary["failure_count"] > 0:
        raise ValueError(
            f"Run {json_path.name}: {summary['failure_count']} failure(s) detected "
            f"(failures > 0 is not allowed)"
        )

    count_delta, sum_delta_s = load_server_delta(before_path, after_path)
    reconcile(summary["request_count"], count_delta, users)

    server_mean_ms = (sum_delta_s / count_delta * 1000.0) if count_delta > 0 else 0.0

    return {
        "json_path": json_path,
        "users": users,
        "request_count": summary["request_count"],
        "failure_count": summary["failure_count"],
        "requests_per_s": summary["requests_per_s"],
        "avg_response_time_ms": summary["avg_response_time_ms"],
        "median_response_time_ms": summary["median_response_time_ms"],
        "p95_response_time_ms": summary["p95_response_time_ms"],
        "p99_response_time_ms": summary["p99_response_time_ms"],
        "max_response_time_ms": summary["max_response_time_ms"],
        "server_delta": count_delta,
        "gap": count_delta - summary["request_count"],
        "server_mean_ms": server_mean_ms,
    }


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------


def _per_config_table(runs_by_users: dict[int, list[dict[str, Any]]]) -> str:
    """Build the per-configuration summary table (one row per user count)."""
    header = (
        "| Users | Repeats | Throughput req/s | Mean ms | p50 ms | p95 ms | p99 ms"
        " | Max ms | Server mean ms | Total requests | Total failures |\n"
        "| :--- | ---: | :--- | :--- | :--- | :--- | :--- | :--- | :--- | ---: | ---: |"
    )
    rows: list[str] = [header]
    for users in sorted(runs_by_users):
        runs = runs_by_users[users]
        n = len(runs)
        total_req = sum(r["request_count"] for r in runs)
        total_fail = sum(r["failure_count"] for r in runs)
        rows.append(
            f"| {users} | {n} "
            f"| {_fmt_range([r['requests_per_s'] for r in runs])} "
            f"| {_fmt_range([r['avg_response_time_ms'] for r in runs])} "
            f"| {_fmt_range([r['median_response_time_ms'] for r in runs])} "
            f"| {_fmt_range([r['p95_response_time_ms'] for r in runs])} "
            f"| {_fmt_range([r['p99_response_time_ms'] for r in runs])} "
            f"| {_fmt_range([r['max_response_time_ms'] for r in runs])} "
            f"| {_fmt_range([r['server_mean_ms'] for r in runs])} "
            f"| {total_req:,} | {total_fail:,} |"
        )
    return "\n".join(rows)


def _per_run_table(all_runs: list[dict[str, Any]]) -> str:
    """Build the per-run detail table."""
    header = (
        "| Run | Users | Requests | Failures | Throughput req/s | Mean ms"
        " | p50 ms | p95 ms | p99 ms | Max ms | Server delta | Gap |\n"
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    rows: list[str] = [header]
    for r in sorted(all_runs, key=lambda x: (x["users"], x["json_path"].name)):
        name = r["json_path"].stem
        rows.append(
            f"| {name} | {r['users']} | {r['request_count']:,} | {r['failure_count']:,}"
            f" | {r['requests_per_s']:.2f}"
            f" | {r['avg_response_time_ms']:.2f}"
            f" | {r['median_response_time_ms']:.2f}"
            f" | {r['p95_response_time_ms']:.2f}"
            f" | {r['p99_response_time_ms']:.2f}"
            f" | {r['max_response_time_ms']:.2f}"
            f" | {r['server_delta']:,} | {r['gap']:+d} |"
        )
    return "\n".join(rows)


def format_aggregate_markdown(
    runs_by_users: dict[int, list[dict[str, Any]]],
    all_runs: list[dict[str, Any]],
) -> str:
    """Compose the full aggregate Markdown report."""
    lines = [
        "## Load Test Aggregate Summary: `/predict`",
        "",
        "### Per-Configuration Results",
        "",
        _per_config_table(runs_by_users),
        "",
        "### Per-Run Results",
        "",
        _per_run_table(all_runs),
        "",
        "### Notes",
        "",
        CAVEAT,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Directory scan
# ---------------------------------------------------------------------------


def discover_runs(
    directory: Path,
) -> dict[int, list[tuple[Path, Path, Path]]]:
    """Find all run trios in *directory* and group them by user count.

    Args:
        directory: Directory to scan for ``u<users>_r<repeat>.json`` files.

    Returns:
        Mapping of users -> sorted list of (json_path, before_path, after_path).

    Raises:
        ValueError: Naming the incomplete file if a matching _before.txt or
            _after.txt is missing for any .json file that matches the naming
            pattern.
    """
    runs: dict[int, list[tuple[Path, Path, Path]]] = {}
    for json_path in sorted(directory.glob("u*_r*.json")):
        m = _RUN_RE.match(json_path.name)
        if not m:
            continue
        users = int(m.group(1))
        stem = json_path.stem  # e.g. u5_r1
        before = directory / f"{stem}_before.txt"
        after = directory / f"{stem}_after.txt"
        if not before.is_file():
            raise ValueError(
                f"Incomplete run trio: missing metrics file {before.name} "
                f"(required alongside {json_path.name})"
            )
        if not after.is_file():
            raise ValueError(
                f"Incomplete run trio: missing metrics file {after.name} "
                f"(required alongside {json_path.name})"
            )
        runs.setdefault(users, []).append((json_path, before, after))
    return runs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for aggregating multiple Locust JSON run files."""
    parser = argparse.ArgumentParser(
        description=("Aggregate Locust load test results across multiple runs for /predict.")
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing u<users>_r<repeat>.json run files",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output markdown file path (defaults to stdout)",
    )
    args = parser.parse_args(argv)

    run_dir = Path(args.dir)
    if not run_dir.is_dir():
        print(f"Error: directory not found: {run_dir}", file=sys.stderr)
        return 1

    try:
        trios_by_users = discover_runs(run_dir)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not trios_by_users:
        print(
            f"Error: no u<users>_r<repeat>.json files found in {run_dir}",
            file=sys.stderr,
        )
        return 1

    all_runs: list[dict[str, Any]] = []
    runs_by_users: dict[int, list[dict[str, Any]]] = {}

    for users, trios in sorted(trios_by_users.items()):
        for json_path, before_path, after_path in trios:
            try:
                run_data = _load_run(json_path, before_path, after_path, users)
            except ValueError as exc:
                print(f"Error in run {json_path.name}: {exc}", file=sys.stderr)
                return 1
            all_runs.append(run_data)
            runs_by_users.setdefault(users, []).append(run_data)

    markdown = format_aggregate_markdown(runs_by_users, all_runs)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Aggregate summary written to {out_path}")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
