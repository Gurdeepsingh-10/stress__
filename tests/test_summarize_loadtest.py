"""Unit tests for scripts/summarize_loadtest.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.summarize_loadtest import (
    FOOTNOTE,
    REQUIRED_KEYS,
    compute_summary_from_json,
    format_markdown,
    main,
    reconcile,
)


def _make_sample_entry() -> dict[str, Any]:
    """Create a hand-built POST /predict entry with hand-computed percentiles.

    Histogram distribution with n = 105 total requests:
      bucket "10": 52 requests (cumulative: 52)
      bucket "20": 1 request   (cumulative: 53)
      bucket "30": 46 requests (cumulative: 99)
      bucket "40": 1 request   (cumulative: 100)
      bucket "50": 3 requests  (cumulative: 103)
      bucket "60": 1 request   (cumulative: 104)
      bucket "70": 1 request   (cumulative: 105)
    Total sum of counts = 52 + 1 + 46 + 1 + 3 + 1 + 1 = 105.

    Hand calculations for nearest-rank percentiles (target = ceil(n * p)):
    1. p50 (median):
       n * 0.50 = 105 * 0.50 = 52.5
       target = ceil(52.5) = 53
       Cumulative count reaches 53 at bucket 20.0 ms.
       (Note: floor(52.5) = 52, which would incorrectly yield bucket 10.0 ms).

    2. p95:
       n * 0.95 = 105 * 0.95 = 99.75
       target = ceil(99.75) = 100
       Cumulative count reaches 100 at bucket 40.0 ms.
       (Note: floor(99.75) = 99, which would incorrectly yield bucket 30.0 ms).

    3. p99:
       n * 0.99 = 105 * 0.99 = 103.95
       target = ceil(103.95) = 104
       Cumulative count reaches 104 at bucket 60.0 ms.
       (Note: floor(103.95) = 103, which would incorrectly yield bucket 50.0 ms).
    """
    return {
        "name": "/predict",
        "method": "POST",
        "num_requests": 105,
        "num_failures": 0,
        "total_response_time": 2100.0,
        "min_response_time": 10.0,
        "max_response_time": 70.0,
        "start_time": 1000.0,
        "last_request_timestamp": 1010.0,
        "response_times": {
            "10": 52,
            "20": 1,
            "30": 46,
            "40": 1,
            "50": 3,
            "60": 1,
            "70": 1,
        },
    }


def test_compute_summary_from_json_hand_computed_percentiles() -> None:
    data = [
        {"name": "Aggregated", "method": None, "num_requests": 105},
        _make_sample_entry(),
    ]
    summary = compute_summary_from_json(data)

    assert summary["name"] == "/predict"
    assert summary["request_count"] == 105
    assert summary["failure_count"] == 0
    assert summary["failure_rate_pct"] == 0.0
    assert summary["avg_response_time_ms"] == 20.0
    assert summary["min_response_time_ms"] == 10.0
    assert summary["max_response_time_ms"] == 70.0
    # Verified by hand calculation above
    assert summary["median_response_time_ms"] == 20.0
    assert summary["p95_response_time_ms"] == 40.0
    assert summary["p99_response_time_ms"] == 60.0
    assert summary["requests_per_s"] == 10.5  # 105 requests / 10.0 s


def test_compute_summary_missing_entry_raises() -> None:
    data = [{"name": "/other", "method": "GET"}]
    with pytest.raises(ValueError, match="No entry with name='/predict' and method='POST'"):
        compute_summary_from_json(data)


def test_compute_summary_missing_required_key_raises() -> None:
    for key in REQUIRED_KEYS:
        entry = _make_sample_entry()
        del entry[key]
        with pytest.raises(ValueError, match=f"Missing required key: '{key}'"):
            compute_summary_from_json([entry])


def test_compute_summary_zero_requests_raises() -> None:
    entry = _make_sample_entry()
    entry["num_requests"] = 0
    entry["response_times"] = {}
    with pytest.raises(ValueError, match="num_requests must be greater than 0"):
        compute_summary_from_json([entry])


def test_compute_summary_histogram_sum_mismatch_raises() -> None:
    entry = _make_sample_entry()
    entry["response_times"]["10"] = 50  # Total becomes 103 != 105
    with pytest.raises(
        ValueError, match="Sum of response_times counts.*does not match num_requests"
    ):
        compute_summary_from_json([entry])


def test_compute_summary_invalid_timestamps_raises() -> None:
    entry = _make_sample_entry()
    entry["last_request_timestamp"] = entry["start_time"]
    with pytest.raises(ValueError, match="last_request_timestamp.*must be greater than start_time"):
        compute_summary_from_json([entry])


def test_reconcile_at_bounds() -> None:
    users = 5
    locust_count = 100

    # Lower bound: gap = 0 -> OK
    reconcile(locust_count, 100, users)

    # Upper bound: gap = users -> OK
    reconcile(locust_count, 105, users)

    # Below lower bound: gap = -1 -> raises ValueError naming all three numbers
    with pytest.raises(ValueError) as exc_low:
        reconcile(locust_count, 99, users)
    err_low = str(exc_low.value)
    assert "99" in err_low
    assert "100" in err_low
    assert "5" in err_low

    # Above upper bound: gap = users + 1 -> raises ValueError naming all three numbers
    with pytest.raises(ValueError) as exc_high:
        reconcile(locust_count, 106, users)
    err_high = str(exc_high.value)
    assert "106" in err_high
    assert "100" in err_high
    assert "5" in err_high


def test_format_markdown_without_and_with_server_delta() -> None:
    summary = compute_summary_from_json([_make_sample_entry()])

    # Without server delta
    md_basic = format_markdown(summary)
    assert "## Load Test Summary: `/predict`" in md_basic
    assert "| Total Requests | 105 |" in md_basic
    assert FOOTNOTE in md_basic
    assert "Server count delta" not in md_basic

    # With server delta
    md_reconciled = format_markdown(summary, server_delta=107)
    assert "| Server count delta | 107 |" in md_reconciled
    assert "| Reconcile | OK |" in md_reconciled
    assert md_reconciled.count("Reconcile") == 1


def test_cli_success_with_metrics_reconcile(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_file = tmp_path / "run.json"
    json_file.write_text(json.dumps([_make_sample_entry()]), encoding="utf-8")

    m_before = tmp_path / "before.txt"
    m_after = tmp_path / "after.txt"
    m_before.write_text("stress_predict_latency_seconds_count 10.0\n", encoding="utf-8")
    m_after.write_text("stress_predict_latency_seconds_count 117.0\n", encoding="utf-8")
    # server_delta = 107. locust_count = 105. users = 5. diff = 2 in [0, 5].

    exit_code = main(
        [
            "--json",
            str(json_file),
            "--users",
            "5",
            "--metrics-before",
            str(m_before),
            "--metrics-after",
            str(m_after),
        ]
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "| Server count delta | 107 |" in out
    assert "| Reconcile | OK |" in out
    assert FOOTNOTE in out
    assert out.count("Reconcile") == 1


def test_cli_mismatching_metrics_pair_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_file = tmp_path / "run.json"
    json_file.write_text(json.dumps([_make_sample_entry()]), encoding="utf-8")

    m_before = tmp_path / "before.txt"
    m_after = tmp_path / "after.txt"
    m_before.write_text("stress_predict_latency_seconds_count 10.0\n", encoding="utf-8")
    m_after.write_text("stress_predict_latency_seconds_count 130.0\n", encoding="utf-8")
    # server_delta = 120. locust_count = 105. diff = 15 > users (5).

    exit_code = main(
        [
            "--json",
            str(json_file),
            "--users",
            "5",
            "--metrics-before",
            str(m_before),
            "--metrics-after",
            str(m_after),
        ]
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "Reconciliation failed" in err


def test_cli_missing_latency_metric_returns_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    json_file = tmp_path / "run.json"
    json_file.write_text(json.dumps([_make_sample_entry()]), encoding="utf-8")

    m_before = tmp_path / "before.txt"
    m_after = tmp_path / "after.txt"
    m_before.write_text("some_other_metric 10.0\n", encoding="utf-8")
    m_after.write_text("some_other_metric 20.0\n", encoding="utf-8")

    exit_code = main(
        [
            "--json",
            str(json_file),
            "--users",
            "5",
            "--metrics-before",
            str(m_before),
            "--metrics-after",
            str(m_after),
        ]
    )
    assert exit_code == 1
    err = capsys.readouterr().err
    assert "stress_predict_latency_seconds_count" in err


def test_cli_success_without_metrics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    json_file = tmp_path / "run.json"
    json_file.write_text(json.dumps([_make_sample_entry()]), encoding="utf-8")

    exit_code = main(["--json", str(json_file), "--users", "5"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "| Total Requests | 105 |" in out
    assert "Server count delta" not in out


def test_cli_module_execution_help() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    res = subprocess.run(
        [sys.executable, "-m", "scripts.summarize_loadtest", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--json" in res.stdout
