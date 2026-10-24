"""Unit tests for scripts/summarize_loadtest.py."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.summarize_loadtest import (
    compute_summary,
    format_markdown,
    main,
    read_stats_csv,
)


def _make_sample_stats_df() -> pd.DataFrame:
    """Create a synthetic Locust stats DataFrame containing /predict and Aggregated rows."""
    records = [
        {
            "Type": None,
            "Name": "Aggregated",
            "Request Count": 2000,  # Distinct value from /predict
            "Failure Count": 5,
            "Median Response Time": 5.0,
            "Average Response Time": 6.0,
            "Min Response Time": 0.5,
            "Max Response Time": 50.0,
            "Average Content Size": 90.0,
            "Requests/s": 290.0,
            "Failures/s": 0.58,
            "50%": 5.0,
            "95%": 10.0,
            "99%": 18.0,
        },
        {
            "Type": "POST",
            "Name": "/predict",
            "Request Count": 1000,
            "Failure Count": 2,
            "Median Response Time": 3.2,
            "Average Response Time": 3.8,
            "Min Response Time": 0.9,
            "Max Response Time": 42.1,
            "Average Content Size": 85.0,
            "Requests/s": 145.5,
            "Failures/s": 0.29,
            "50%": 3.2,
            "95%": 7.5,
            "99%": 12.8,
        },
    ]
    return pd.DataFrame(records)


def test_read_stats_csv(tmp_path: Path) -> None:
    csv_file = tmp_path / "run_stats.csv"
    df_sample = _make_sample_stats_df()
    df_sample.to_csv(csv_file, index=False)

    df_loaded = read_stats_csv(csv_file)
    assert len(df_loaded) == 2
    assert "Name" in df_loaded.columns


def test_read_stats_csv_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        read_stats_csv("nonexistent_stats.csv")


def test_compute_summary_filters_predict_row() -> None:
    """Load-bearing test: compute_summary MUST select Name == '/predict' and not Aggregated."""
    df = _make_sample_stats_df()
    summary = compute_summary(df)

    assert summary["name"] == "/predict"
    # /predict has 1000 requests; Aggregated has 2000
    assert summary["request_count"] == 1000
    assert summary["failure_count"] == 2
    assert summary["requests_per_s"] == 145.5
    assert summary["avg_response_time_ms"] == 3.8
    assert summary["median_response_time_ms"] == 3.2
    assert summary["p95_response_time_ms"] == 7.5
    assert summary["p99_response_time_ms"] == 12.8
    assert summary["min_response_time_ms"] == 0.9
    assert summary["max_response_time_ms"] == 42.1
    assert pytest.approx(summary["failure_rate_pct"], rel=1e-3) == 0.20


def test_compute_summary_missing_predict_row_raises_error() -> None:
    df = pd.DataFrame(
        [
            {
                "Type": "GET",
                "Name": "/health",
                "Request Count": 10,
                "Failure Count": 0,
                "Requests/s": 1.0,
                "Average Response Time": 1.2,
                "Min Response Time": 0.5,
                "Max Response Time": 2.0,
                "50%": 1.0,
                "95%": 1.5,
                "99%": 1.8,
            }
        ]
    )
    with pytest.raises(ValueError, match="No row with Name == '/predict' found in stats CSV"):
        compute_summary(df)


def test_compute_summary_missing_50_percent_column_raises_error() -> None:
    df = _make_sample_stats_df().drop(columns=["50%"])
    with pytest.raises(ValueError, match="50%") as exc_info:
        compute_summary(df)
    assert "50%" in str(exc_info.value)


def test_compute_summary_missing_95_percent_column_raises_error() -> None:
    df = _make_sample_stats_df().drop(columns=["95%"])
    with pytest.raises(ValueError, match="95%") as exc_info:
        compute_summary(df)
    assert "95%" in str(exc_info.value)


def test_compute_summary_missing_99_percent_column_raises_error() -> None:
    df = _make_sample_stats_df().drop(columns=["99%"])
    with pytest.raises(ValueError, match="99%") as exc_info:
        compute_summary(df)
    assert "99%" in str(exc_info.value)


def test_format_markdown() -> None:
    df = _make_sample_stats_df()
    summary = compute_summary(df)
    md = format_markdown(summary)

    assert "## Load Test Summary: `/predict`" in md
    assert "| Total Requests | 1,000 |" in md
    assert "| Failed Requests | 2 (0.20%) |" in md
    assert "| Throughput | 145.50 req/s |" in md
    assert "| p95 Latency | 7.50 ms |" in md


def test_main_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    csv_file = tmp_path / "test_stats.csv"
    out_file = tmp_path / "summary.md"
    _make_sample_stats_df().to_csv(csv_file, index=False)

    # Test stdout output
    code = main(["--csv", str(csv_file)])
    assert code == 0
    stdout = capsys.readouterr().out
    assert "Load Test Summary: `/predict`" in stdout

    # Test file output
    code = main(["--csv", str(csv_file), "--out", str(out_file)])
    assert code == 0
    assert out_file.is_file()
    assert "Load Test Summary: `/predict`" in out_file.read_text(encoding="utf-8")
