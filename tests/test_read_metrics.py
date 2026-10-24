"""Unit tests for scripts/read_metrics.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from scripts.read_metrics import (
    compute_metrics_delta,
    fetch_metrics,
    format_latency_summary,
    main,
    parse_metrics,
)

_SAMPLE_METRICS = """# HELP stress_http_requests_total Total number of HTTP requests processed.
# TYPE stress_http_requests_total counter
stress_http_requests_total{method="POST",route="/predict",status="200"} 150.0
# HELP stress_predict_latency_seconds Latency of /predict endpoint inference in seconds.
# TYPE stress_predict_latency_seconds histogram
stress_predict_latency_seconds_bucket{le="0.005"} 120.0
stress_predict_latency_seconds_count 150.0
stress_predict_latency_seconds_sum 0.75
"""


def test_parse_metrics() -> None:
    parsed = parse_metrics(_SAMPLE_METRICS)
    assert parsed["stress_predict_latency_seconds_count"] == 150.0
    assert parsed["stress_predict_latency_seconds_sum"] == 0.75
    assert (
        parsed['stress_http_requests_total{method="POST",route="/predict",status="200"}'] == 150.0
    )


def test_compute_metrics_delta() -> None:
    before = {
        "stress_predict_latency_seconds_count": 100.0,
        "stress_predict_latency_seconds_sum": 0.50,
        "old_metric": 5.0,
    }
    after = {
        "stress_predict_latency_seconds_count": 250.0,
        "stress_predict_latency_seconds_sum": 1.25,
        "new_metric": 10.0,
    }
    delta = compute_metrics_delta(before, after)
    assert delta["stress_predict_latency_seconds_count"] == 150.0
    assert pytest.approx(delta["stress_predict_latency_seconds_sum"], rel=1e-5) == 0.75
    assert delta["old_metric"] == -5.0
    assert delta["new_metric"] == 10.0


def test_format_latency_summary() -> None:
    before = {
        "stress_predict_latency_seconds_count": 100.0,
        "stress_predict_latency_seconds_sum": 0.50,
    }
    after = {
        "stress_predict_latency_seconds_count": 200.0,
        "stress_predict_latency_seconds_sum": 0.90,
    }
    summary = format_latency_summary(before, after)
    assert "delta: +100" in summary
    assert "delta: +0.400000s" in summary
    # 0.4s / 100 req = 4 ms/req
    assert "4.000 ms/req" in summary


def test_fetch_metrics_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"# mock metrics data\nmetric_foo 42.0\n"
    mock_resp.__enter__.return_value = mock_resp
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=10.0: mock_resp)

    text = fetch_metrics("http://testserver:8000/metrics")
    assert "metric_foo 42.0" in text


def test_main_diff_mode(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before_file = tmp_path / "before.txt"
    after_file = tmp_path / "after.txt"

    before_file.write_text(
        "stress_predict_latency_seconds_count 10.0\nstress_predict_latency_seconds_sum 0.05\n"
    )
    after_file.write_text(
        "stress_predict_latency_seconds_count 30.0\nstress_predict_latency_seconds_sum 0.15\n"
    )

    exit_code = main(["--diff", str(before_file), str(after_file)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "delta: +20" in out
    assert "5.000 ms/req" in out


def test_main_url_and_out_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mock_resp = MagicMock()
    mock_resp.read.return_value = b"test_metric 123.0\n"
    mock_resp.__enter__.return_value = mock_resp
    monkeypatch.setattr("urllib.request.urlopen", lambda req, timeout=10.0: mock_resp)

    out_file = tmp_path / "metrics.txt"
    exit_code = main(["--url", "http://testserver:8000/metrics", "--out", str(out_file)])
    assert exit_code == 0
    assert out_file.read_text() == "test_metric 123.0\n"
