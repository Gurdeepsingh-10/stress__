"""Extract and format Locust load test performance summary for /predict from JSON.

Parses Locust --json-file export, computes nearest-rank latency percentiles,
active-window throughput, and reconciles request count with server metrics.

Usage:
    .venv\\Scripts\\python.exe -m scripts.summarize_loadtest --json <file> --users N \\
        [--metrics-before F --metrics-after F] [--out F]

Note:
    Running the file path directly is not supported; invoke via -m scripts.summarize_loadtest.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from scripts.read_metrics import compute_metrics_delta, parse_metrics

REQUIRED_KEYS: tuple[str, ...] = (
    "num_requests",
    "num_failures",
    "total_response_time",
    "min_response_time",
    "max_response_time",
    "response_times",
    "start_time",
    "last_request_timestamp",
)


def _nearest_rank_percentile(sorted_buckets: list[tuple[float, int]], n: int, p: float) -> float:
    """Calculate the p-th percentile using nearest-rank method on histogram buckets."""
    target = math.ceil(n * p)
    cumulative = 0
    for val, count in sorted_buckets:
        cumulative += count
        if cumulative >= target:
            return val
    return sorted_buckets[-1][0]


def compute_summary_from_json(data: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract performance metrics for the POST /predict endpoint from Locust JSON data.

    Args:
        data: List of dictionaries parsed from Locust JSON export.

    Returns:
        Dictionary of extracted latency and throughput metrics.

    Raises:
        ValueError: If the POST /predict entry or any required key is missing,
            num_requests is zero, histogram count sum mismatches num_requests,
            or last_request_timestamp <= start_time.
    """
    if not isinstance(data, list):
        raise ValueError("Root JSON structure must be a list")

    predict_entry: dict[str, Any] | None = None
    for entry in data:
        if (
            isinstance(entry, dict)
            and entry.get("name") == "/predict"
            and entry.get("method") == "POST"
        ):
            predict_entry = entry
            break

    if predict_entry is None:
        raise ValueError("No entry with name='/predict' and method='POST' found in JSON data")

    missing_keys = [k for k in REQUIRED_KEYS if k not in predict_entry]
    if missing_keys:
        raise ValueError(f"Missing required key: '{missing_keys[0]}'")

    num_requests = int(predict_entry["num_requests"])
    if num_requests <= 0:
        raise ValueError("num_requests must be greater than 0")

    response_times: Any = predict_entry["response_times"]
    if not isinstance(response_times, dict):
        raise ValueError("response_times must be a dictionary")

    total_histogram_count = sum(int(c) for c in response_times.values())
    if total_histogram_count != num_requests:
        raise ValueError(
            f"Sum of response_times counts ({total_histogram_count}) "
            f"does not match num_requests ({num_requests})"
        )

    start_time = float(predict_entry["start_time"])
    last_request_timestamp = float(predict_entry["last_request_timestamp"])
    if last_request_timestamp <= start_time:
        raise ValueError(
            f"last_request_timestamp ({last_request_timestamp}) "
            f"must be greater than start_time ({start_time})"
        )

    failure_count = int(predict_entry["num_failures"])
    failure_rate_pct = (failure_count / num_requests * 100.0) if num_requests > 0 else 0.0

    total_response_time = float(predict_entry["total_response_time"])
    avg_latency = total_response_time / num_requests
    min_latency = float(predict_entry["min_response_time"])
    max_latency = float(predict_entry["max_response_time"])

    duration = last_request_timestamp - start_time
    throughput = num_requests / duration

    sorted_buckets: list[tuple[float, int]] = sorted(
        ((float(k), int(v)) for k, v in response_times.items()),
        key=lambda x: x[0],
    )

    median_latency = _nearest_rank_percentile(sorted_buckets, num_requests, 0.50)
    p95_latency = _nearest_rank_percentile(sorted_buckets, num_requests, 0.95)
    p99_latency = _nearest_rank_percentile(sorted_buckets, num_requests, 0.99)

    return {
        "name": "/predict",
        "request_count": num_requests,
        "failure_count": failure_count,
        "failure_rate_pct": failure_rate_pct,
        "requests_per_s": throughput,
        "avg_response_time_ms": avg_latency,
        "min_response_time_ms": min_latency,
        "max_response_time_ms": max_latency,
        "median_response_time_ms": median_latency,
        "p95_response_time_ms": p95_latency,
        "p99_response_time_ms": p99_latency,
    }


def reconcile(locust_count: int, server_delta: int, users: int) -> None:
    """Validate that server request delta accounts for all Locust requests within user drift.

    Args:
        locust_count: Total requests recorded by Locust.
        server_delta: Total requests recorded on server metrics.
        users: Number of virtual users.

    Raises:
        ValueError: If server_delta - locust_count is not in [0, users].
    """
    diff = server_delta - locust_count
    if not (0 <= diff <= users):
        raise ValueError(
            f"Reconciliation failed: server_delta ({server_delta}) - locust_count ({locust_count}) "
            f"= {diff}, which is outside expected range [0, {users}] for users={users}"
        )


def load_server_delta(
    before_path: Path, after_path: Path, require_sum: bool = True
) -> tuple[int, float]:
    """Load Prometheus snapshots and return the request-count and latency-sum deltas.

    Args:
        before_path: Path to metrics snapshot taken before the run.
        after_path: Path to metrics snapshot taken after the run.
        require_sum: Whether to require stress_predict_latency_seconds_sum metric.

    Returns:
        (count_delta, sum_delta_seconds) where count_delta is the integer number of
        new requests observed on the server and sum_delta_seconds is the total latency
        added in seconds (stress_predict_latency_seconds_sum delta).

    Raises:
        ValueError: Naming the missing metric key if either
            stress_predict_latency_seconds_count or (if require_sum)
            stress_predict_latency_seconds_sum is absent from the computed delta.
    """
    before_metrics = parse_metrics(before_path.read_text(encoding="utf-8"))
    after_metrics = parse_metrics(after_path.read_text(encoding="utf-8"))
    delta = compute_metrics_delta(before_metrics, after_metrics)

    count_key = "stress_predict_latency_seconds_count"
    if count_key not in delta:
        raise ValueError(f"Missing required metric in delta: '{count_key}'")

    sum_key = "stress_predict_latency_seconds_sum"
    if sum_key not in delta:
        if require_sum:
            raise ValueError(f"Missing required metric in delta: '{sum_key}'")
        sum_delta = 0.0
    else:
        sum_delta = delta[sum_key]

    return int(round(delta[count_key])), sum_delta


FOOTNOTE: str = (
    "Percentiles: nearest-rank on Locust's rounded-bucket histogram; "
    "throughput: requests / active window"
)


def format_markdown(summary: dict[str, Any], server_delta: int | None = None) -> str:
    """Format extracted metrics as a readable Markdown summary table.

    Args:
        summary: Dictionary produced by compute_summary_from_json().
        server_delta: Optional server request count delta from Prometheus.

    Returns:
        Markdown string summarizing the load test performance.
    """
    lines = [
        "## Load Test Summary: `/predict`",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| Endpoint | `{summary['name']}` |",
        f"| Total Requests | {summary['request_count']:,} |",
        f"| Failed Requests | {summary['failure_count']:,} ({summary['failure_rate_pct']:.2f}%) |",
        f"| Throughput | {summary['requests_per_s']:.2f} req/s |",
        f"| Mean Latency | {summary['avg_response_time_ms']:.2f} ms |",
        f"| Median Latency (p50) | {summary['median_response_time_ms']:.2f} ms |",
        f"| p95 Latency | {summary['p95_response_time_ms']:.2f} ms |",
        f"| p99 Latency | {summary['p99_response_time_ms']:.2f} ms |",
        f"| Min Latency | {summary['min_response_time_ms']:.2f} ms |",
        f"| Max Latency | {summary['max_response_time_ms']:.2f} ms |",
    ]
    if server_delta is not None:
        lines.append(f"| Server count delta | {server_delta:,} |")
        lines.append("| Reconcile | OK |")

    lines.extend(
        [
            "",
            FOOTNOTE,
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for summarizing a Locust JSON export file."""
    parser = argparse.ArgumentParser(
        description="Summarize Locust load test results for /predict from JSON."
    )
    parser.add_argument(
        "--json",
        type=str,
        required=True,
        dest="json_path",
        help="Path to Locust JSON export file",
    )
    parser.add_argument(
        "--users",
        type=int,
        required=True,
        help="Number of simulated concurrent users",
    )
    parser.add_argument(
        "--metrics-before",
        type=str,
        default=None,
        help="Path to Prometheus metrics snapshot taken before the run",
    )
    parser.add_argument(
        "--metrics-after",
        type=str,
        default=None,
        help="Path to Prometheus metrics snapshot taken after the run",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output markdown file path (defaults to stdout)",
    )

    args = parser.parse_args(argv)

    if bool(args.metrics_before) != bool(args.metrics_after):
        print(
            "Error: --metrics-before and --metrics-after must be specified together.",
            file=sys.stderr,
        )
        return 1

    json_path = Path(args.json_path)
    if not json_path.is_file():
        print(f"Error: JSON file not found: {json_path}", file=sys.stderr)
        return 1

    try:
        with json_path.open(encoding="utf-8") as f:
            data = json.load(f)
        summary = compute_summary_from_json(data)
    except Exception as exc:
        print(f"Error summarizing load test JSON: {exc}", file=sys.stderr)
        return 1

    server_delta: int | None = None
    if args.metrics_before and args.metrics_after:
        before_path = Path(args.metrics_before)
        after_path = Path(args.metrics_after)
        if not before_path.is_file():
            print(f"Error: File not found: {before_path}", file=sys.stderr)
            return 1
        if not after_path.is_file():
            print(f"Error: File not found: {after_path}", file=sys.stderr)
            return 1

        try:
            count_delta, _sum_delta = load_server_delta(before_path, after_path, require_sum=False)
            server_delta = count_delta
            reconcile(summary["request_count"], server_delta, args.users)
        except ValueError as exc:
            print(f"Error during reconciliation: {exc}", file=sys.stderr)
            return 1

    markdown = format_markdown(summary, server_delta=server_delta)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(markdown, encoding="utf-8")
        print(f"Summary written to {out_path}")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    sys.exit(main())
