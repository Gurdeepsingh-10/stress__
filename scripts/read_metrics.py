"""Fetch Prometheus metrics from the API and compute latency deltas.

Stdlib-only utility for before/after load test metrics comparison.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


def fetch_metrics(url: str, timeout: float = 10.0) -> str:
    """Fetch raw Prometheus metrics text from an HTTP endpoint.

    Args:
        url: Metrics endpoint URL (e.g. http://127.0.0.1:8000/metrics).
        timeout: Network timeout in seconds.

    Returns:
        Scraped metrics payload as decoded text.
    """
    req = urllib.request.Request(url, headers={"Accept": "text/plain; version=0.0.4"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        content: bytes = response.read()
        return content.decode("utf-8")


def parse_metrics(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition format into a name-value mapping.

    Comments and unparseable lines are ignored. For lines with labels, the full
    token before the value is preserved.

    Args:
        text: Prometheus exposition text.

    Returns:
        Mapping of metric identifiers to float values.
    """
    parsed: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0]
            try:
                val = float(parts[1])
                parsed[key] = val
            except ValueError:
                continue
    return parsed


def compute_metrics_delta(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    """Compute the difference (after - before) for all observed metric keys.

    Args:
        before: Metrics mapping before load test.
        after: Metrics mapping after load test.

    Returns:
        Mapping of metric keys to their numeric deltas.
    """
    delta: dict[str, float] = {}
    all_keys = set(before.keys()).union(after.keys())
    for key in all_keys:
        val_before = before.get(key, 0.0)
        val_after = after.get(key, 0.0)
        delta[key] = val_after - val_before
    return delta


def format_latency_summary(
    before: dict[str, float],
    after: dict[str, float],
) -> str:
    """Format before/after delta specifically for stress_predict_latency_seconds.

    Args:
        before: Metrics mapping before load test.
        after: Metrics mapping after load test.

    Returns:
        Human-readable summary report.
    """
    count_before = before.get("stress_predict_latency_seconds_count", 0.0)
    count_after = after.get("stress_predict_latency_seconds_count", 0.0)
    count_delta = count_after - count_before

    sum_before = before.get("stress_predict_latency_seconds_sum", 0.0)
    sum_after = after.get("stress_predict_latency_seconds_sum", 0.0)
    sum_delta = sum_after - sum_before

    mean_ms = (sum_delta / count_delta * 1000.0) if count_delta > 0 else 0.0

    lines = [
        "============================================================",
        "Prometheus Metrics Delta: stress_predict_latency_seconds",
        "============================================================",
        f"Request Count : {count_before:.0f} -> {count_after:.0f} (delta: {count_delta:+.0f})",
        f"Latency Sum   : {sum_before:.6f}s -> {sum_after:.6f}s (delta: {sum_delta:+.6f}s)",
        f"Mean Latency  : {mean_ms:.3f} ms/req",
        "============================================================",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for reading metrics or computing deltas."""
    parser = argparse.ArgumentParser(
        description="Fetch Prometheus metrics or compute latency deltas."
    )
    parser.add_argument("--url", type=str, default=None, help="URL to scrape metrics from")
    parser.add_argument("--out", type=str, default=None, help="File path to save scraped metrics")
    parser.add_argument(
        "--diff",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Compare two metrics files and report latency delta",
    )

    args = parser.parse_args(argv)

    if args.url:
        try:
            metrics_text = fetch_metrics(args.url)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"Error fetching metrics from {args.url}: {exc}", file=sys.stderr)
            return 1

        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(metrics_text, encoding="utf-8")
            print(f"Metrics saved to {out_path}")
        else:
            print(metrics_text)
        return 0

    if args.diff:
        before_path = Path(args.diff[0])
        after_path = Path(args.diff[1])
        if not before_path.is_file():
            print(f"File not found: {before_path}", file=sys.stderr)
            return 1
        if not after_path.is_file():
            print(f"File not found: {after_path}", file=sys.stderr)
            return 1

        before = parse_metrics(before_path.read_text(encoding="utf-8"))
        after = parse_metrics(after_path.read_text(encoding="utf-8"))
        print(format_latency_summary(before, after))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
