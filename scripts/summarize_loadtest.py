"""Extract and format Locust load test performance summary for /predict.

Parses Locust *_stats.csv and extracts latency and throughput metrics
for the /predict inference endpoint into a Markdown table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd


def read_stats_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read a Locust stats CSV file into a pandas DataFrame.

    Args:
        csv_path: Path to the *_stats.csv file.

    Returns:
        DataFrame containing Locust run statistics.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Stats CSV file not found: {path}")
    return pd.read_csv(path)


REQUIRED_COLUMNS: tuple[str, ...] = (
    "Name",
    "Request Count",
    "Failure Count",
    "Requests/s",
    "Average Response Time",
    "Min Response Time",
    "Max Response Time",
    "50%",
    "95%",
    "99%",
)


def compute_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Extract performance metrics for the /predict endpoint from Locust stats.

    Args:
        df: DataFrame containing the raw Locust stats CSV records.

    Returns:
        Dictionary of extracted latency and throughput metrics.

    Raises:
        ValueError: If required columns are missing from the DataFrame,
            or if no row with Name == '/predict' is present.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in stats CSV: {', '.join(missing)}")

    predict_rows = df[df["Name"] == "/predict"]
    if predict_rows.empty:
        raise ValueError("No row with Name == '/predict' found in stats CSV")

    row = predict_rows.iloc[0]

    request_count = int(row["Request Count"])
    failure_count = int(row["Failure Count"])
    requests_per_s = float(row["Requests/s"])
    failures_per_s = float(row.get("Failures/s", 0.0))

    avg_latency = float(row["Average Response Time"])
    min_latency = float(row["Min Response Time"])
    max_latency = float(row["Max Response Time"])
    median_latency = float(row["50%"])
    p95_latency = float(row["95%"])
    p99_latency = float(row["99%"])

    failure_rate = (failure_count / request_count * 100.0) if request_count > 0 else 0.0

    return {
        "name": "/predict",
        "request_count": request_count,
        "failure_count": failure_count,
        "failure_rate_pct": failure_rate,
        "requests_per_s": requests_per_s,
        "failures_per_s": failures_per_s,
        "avg_response_time_ms": avg_latency,
        "min_response_time_ms": min_latency,
        "max_response_time_ms": max_latency,
        "median_response_time_ms": median_latency,
        "p95_response_time_ms": p95_latency,
        "p99_response_time_ms": p99_latency,
    }


def format_markdown(summary: dict[str, Any]) -> str:
    """Format extracted metrics as a readable Markdown summary table.

    Args:
        summary: Dictionary produced by compute_summary().

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
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for summarizing a Locust stats CSV file."""
    parser = argparse.ArgumentParser(description="Summarize Locust load test results for /predict.")
    parser.add_argument("--csv", type=str, required=True, help="Path to Locust *_stats.csv")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output markdown file path (defaults to stdout)",
    )

    args = parser.parse_args(argv)

    try:
        df = read_stats_csv(args.csv)
        summary = compute_summary(df)
        markdown = format_markdown(summary)
    except Exception as exc:
        print(f"Error summarizing load test stats: {exc}", file=sys.stderr)
        return 1

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
