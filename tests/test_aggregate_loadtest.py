"""Unit tests for scripts/aggregate_loadtest.py.

All fixture datasets use distinct numbers across all metrics (throughput,
mean, p50, p95, p99, max, server delta, gap, and server mean). Hand-computed
calculations for nearest-rank targets, medians, ranges, and reconciliations
are documented in comments directly alongside the data definitions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.aggregate_loadtest import (
    CAVEAT,
    _fmt_range,
    _median,
    discover_runs,
    main,
)

# ---------------------------------------------------------------------------
# Shared Prometheus snapshots
# ---------------------------------------------------------------------------
_BEFORE_COUNT = 10.0
_BEFORE_SUM = 2.0


def _prom_before() -> str:
    return (
        f"stress_predict_latency_seconds_count {_BEFORE_COUNT}\n"
        f"stress_predict_latency_seconds_sum {_BEFORE_SUM}\n"
    )


def _prom_after(count: float, latency_sum: float) -> str:
    return (
        f"stress_predict_latency_seconds_count {count}\n"
        f"stress_predict_latency_seconds_sum {latency_sum}\n"
    )


# ---------------------------------------------------------------------------
# Synthetic Locust entry helper
# ---------------------------------------------------------------------------
def _make_locust_entry(
    num_requests: int,
    total_response_time: float,
    min_response_time: float,
    max_response_time: float,
    response_times: dict[str, int],
    duration_s: float = 10.0,
    num_failures: int = 0,
) -> dict[str, Any]:
    return {
        "name": "/predict",
        "method": "POST",
        "num_requests": num_requests,
        "num_failures": num_failures,
        "total_response_time": total_response_time,
        "min_response_time": min_response_time,
        "max_response_time": max_response_time,
        "start_time": 1000.0,
        "last_request_timestamp": 1000.0 + duration_s,
        "response_times": response_times,
    }


def _write_run_files(
    directory: Path,
    stem: str,
    entry: dict[str, Any],
    after_prom: str,
) -> None:
    (directory / f"{stem}.json").write_text(json.dumps([entry]), encoding="utf-8")
    (directory / f"{stem}_before.txt").write_text(_prom_before(), encoding="utf-8")
    (directory / f"{stem}_after.txt").write_text(after_prom, encoding="utf-8")


# ---------------------------------------------------------------------------
# Hand-computed fixture data & expectations
# ---------------------------------------------------------------------------
#
# RUN: u5_r1 (users=5, repeat 1)
#   num_requests = 100, duration = 10.0s -> throughput = 100 / 10.0 = 10.00 req/s
#   total_response_time = 1500.0 -> mean = 1500.0 / 100 = 15.00 ms
#   buckets: {"10": 60, "20": 36, "30": 3, "40": 1} (sum = 100)
#     nearest-rank p50: ceil(100 * 0.50) = 50 -> bucket "10" (cum 60 >= 50) -> 10.00 ms
#     nearest-rank p95: ceil(100 * 0.95) = 95 -> bucket "20" (cum 96 >= 95) -> 20.00 ms
#     nearest-rank p99: ceil(100 * 0.99) = 99 -> bucket "30" (cum 99 >= 99) -> 30.00 ms
#   min = 10.00 ms, max = 45.00 ms
#   Prometheus: before count=10.0, sum=2.0; after count=112.0, sum=22.4
#     count_delta = 112 - 10 = 102 (rounded int: 102)
#     sum_delta = 22.4 - 2.0 = 20.4 s
#     gap = count_delta - num_requests = 102 - 100 = +2 (reconcile: 2 in [0, 5] -> OK)
#     server_mean = 20.4 / 102 * 1000 = 200.00 ms
#
# RUN: u5_r2 (users=5, repeat 2)
#   num_requests = 200, duration = 10.0s -> throughput = 200 / 10.0 = 20.00 req/s
#   total_response_time = 4200.0 -> mean = 4200.0 / 200 = 21.00 ms
#   buckets: {"15": 110, "25": 81, "35": 8, "50": 1} (sum = 200)
#     nearest-rank p50: ceil(200 * 0.50) = 100 -> bucket "15" (cum 110 >= 100) -> 15.00 ms
#     nearest-rank p95: ceil(200 * 0.95) = 190 -> bucket "25" (cum 191 >= 190) -> 25.00 ms
#     nearest-rank p99: ceil(200 * 0.99) = 198 -> bucket "35" (cum 199 >= 198) -> 35.00 ms
#   min = 12.00 ms, max = 55.00 ms
#   Prometheus: before count=10.0, sum=2.0; after count=213.0, sum=52.75
#     count_delta = 213 - 10 = 203 (rounded int: 203)
#     sum_delta = 52.75 - 2.0 = 50.75 s
#     gap = count_delta - num_requests = 203 - 200 = +3 (reconcile: 3 in [0, 5] -> OK)
#     server_mean = 50.75 / 203 * 1000 = 250.00 ms
#
# USERS=5 MEDIANS (even n=2, mean of middle two: r1 and r2):
#   throughput:  median = (10.00 + 20.00) / 2 = 15.00, range: 15.00 (10.00–20.00)
#   mean:        median = (15.00 + 21.00) / 2 = 18.00, range: 18.00 (15.00–21.00)
#   p50:         median = (10.00 + 15.00) / 2 = 12.50, range: 12.50 (10.00–15.00)
#   p95:         median = (20.00 + 25.00) / 2 = 22.50, range: 22.50 (20.00–25.00)
#   p99:         median = (30.00 + 35.00) / 2 = 32.50, range: 32.50 (30.00–35.00)
#   max:         median = (45.00 + 55.00) / 2 = 50.00, range: 50.00 (45.00–55.00)
#   server_mean: median = (200.00 + 250.00) / 2 = 225.00, range: 225.00 (200.00–250.00)
#   total requests: 100 + 200 = 300, total failures: 0
#
# ---------------------------------------------------------------------------
# RUN: u10_r1 (users=10, repeat 1)
#   num_requests = 100, duration = 4.0s -> throughput = 100 / 4.0 = 25.00 req/s
#   total_response_time = 1800.0 -> mean = 1800.0 / 100 = 18.00 ms
#   buckets: {"12": 55, "22": 41, "32": 3, "60": 1} (sum = 100)
#     nearest-rank p50: ceil(100 * 0.50) = 50 -> bucket "12" (cum 55 >= 50) -> 12.00 ms
#     nearest-rank p95: ceil(100 * 0.95) = 95 -> bucket "22" (cum 96 >= 95) -> 22.00 ms
#     nearest-rank p99: ceil(100 * 0.99) = 99 -> bucket "32" (cum 99 >= 99) -> 32.00 ms
#   min = 11.00 ms, max = 62.00 ms
#   Prometheus: before count=10.0, sum=2.0; after count=114.0, sum=33.2
#     count_delta = 114 - 10 = 104
#     sum_delta = 33.2 - 2.0 = 31.2 s
#     gap = count_delta - num_requests = 104 - 100 = +4 (reconcile: 4 in [0, 10] -> OK)
#     server_mean = 31.2 / 104 * 1000 = 300.00 ms
#
# RUN: u10_r2 (users=10, repeat 2)
#   num_requests = 200, duration = 4.0s -> throughput = 200 / 4.0 = 50.00 req/s
#   total_response_time = 4800.0 -> mean = 4800.0 / 200 = 24.00 ms
#   buckets: {"18": 105, "28": 86, "38": 8, "70": 1} (sum = 200)
#     nearest-rank p50: ceil(200 * 0.50) = 100 -> bucket "18" (cum 105 >= 100) -> 18.00 ms
#     nearest-rank p95: ceil(200 * 0.95) = 190 -> bucket "28" (cum 191 >= 190) -> 28.00 ms
#     nearest-rank p99: ceil(200 * 0.99) = 198 -> bucket "38" (cum 199 >= 198) -> 38.00 ms
#   min = 14.00 ms, max = 72.00 ms
#   Prometheus: before count=10.0, sum=2.0; after count=216.0, sum=84.4
#     count_delta = 216 - 10 = 206
#     sum_delta = 84.4 - 2.0 = 82.4 s
#     gap = count_delta - num_requests = 206 - 200 = +6 (reconcile: 6 in [0, 10] -> OK)
#     server_mean = 82.4 / 206 * 1000 = 400.00 ms
#
# USERS=10 MEDIANS (even n=2, mean of middle two: r1 and r2):
#   throughput:  median = (25.00 + 50.00) / 2 = 37.50, range: 37.50 (25.00–50.00)
#   mean:        median = (18.00 + 24.00) / 2 = 21.00, range: 21.00 (18.00–24.00)
#   p50:         median = (12.00 + 18.00) / 2 = 15.00, range: 15.00 (12.00–18.00)
#   p95:         median = (22.00 + 28.00) / 2 = 25.00, range: 25.00 (22.00–28.00)
#   p99:         median = (32.00 + 38.00) / 2 = 35.00, range: 35.00 (32.00–38.00)
#   max:         median = (62.00 + 72.00) / 2 = 67.00, range: 67.00 (62.00–72.00)
#   server_mean: median = (300.00 + 400.00) / 2 = 350.00, range: 350.00 (300.00–400.00)
#   total requests: 100 + 200 = 300, total failures: 0
#
# ---------------------------------------------------------------------------
# RUN: u5_r3 (users=5, repeat 3 - for odd-n 3-repeat test)
#   num_requests = 300, duration = 10.0s -> throughput = 300 / 10.0 = 30.00 req/s
#   total_response_time = 8100.0 -> mean = 8100.0 / 300 = 27.00 ms
#   buckets: {"18": 160, "28": 126, "38": 12, "60": 2} (sum = 300)
#     nearest-rank p50: ceil(300 * 0.50) = 150 -> bucket "18" (cum 160 >= 150) -> 18.00 ms
#     nearest-rank p95: ceil(300 * 0.95) = 285 -> bucket "28" (cum 286 >= 285) -> 28.00 ms
#     nearest-rank p99: ceil(300 * 0.99) = 297 -> bucket "38" (cum 298 >= 297) -> 38.00 ms
#   min = 15.00 ms, max = 65.00 ms
#   Prometheus: before count=10.0, sum=2.0; after count=314.0, sum=93.2
#     count_delta = 314 - 10 = 304
#     sum_delta = 93.2 - 2.0 = 91.2 s
#     gap = count_delta - num_requests = 304 - 300 = +4 (reconcile: 4 in [0, 5] -> OK)
#     server_mean = 91.2 / 304 * 1000 = 300.00 ms
#
# USERS=5 MEDIANS FOR 3 REPEATS (odd n=3, middle index 3//2 = 1):
#   throughput:  [10.00, 20.00, 30.00] -> median = 20.00, range: 20.00 (10.00–30.00)
#   mean:        [15.00, 21.00, 27.00] -> median = 21.00, range: 21.00 (15.00–27.00)
#   p50:         [10.00, 15.00, 18.00] -> median = 15.00, range: 15.00 (10.00–18.00)
#   p95:         [20.00, 25.00, 28.00] -> median = 25.00, range: 25.00 (20.00–28.00)
#   p99:         [30.00, 35.00, 38.00] -> median = 35.00, range: 35.00 (30.00–38.00)
#   max:         [45.00, 55.00, 65.00] -> median = 55.00, range: 55.00 (45.00–65.00)
#   server_mean: [200.00, 250.00, 300.00] -> median = 250.00, range: 250.00 (200.00–300.00)
#   total requests: 100 + 200 + 300 = 600, total failures: 0

EXPECTED_ROW_U5 = (
    "| 5 | 2 | 15.00 (10.00–20.00) | 18.00 (15.00–21.00) | 12.50 (10.00–15.00) "
    "| 22.50 (20.00–25.00) | 32.50 (30.00–35.00) | 50.00 (45.00–55.00) "
    "| 225.00 (200.00–250.00) | 300 | 0 |"
)

EXPECTED_ROW_U10 = (
    "| 10 | 2 | 37.50 (25.00–50.00) | 21.00 (18.00–24.00) | 15.00 (12.00–18.00) "
    "| 25.00 (22.00–28.00) | 35.00 (32.00–38.00) | 67.00 (62.00–72.00) "
    "| 350.00 (300.00–400.00) | 300 | 0 |"
)

EXPECTED_RUN_U5_R1 = (
    "| u5_r1 | 5 | 100 | 0 | 10.00 | 15.00 | 10.00 | 20.00 | 30.00 | 45.00 | 102 | +2 |"
)
EXPECTED_RUN_U5_R2 = (
    "| u5_r2 | 5 | 200 | 0 | 20.00 | 21.00 | 15.00 | 25.00 | 35.00 | 55.00 | 203 | +3 |"
)
EXPECTED_RUN_U10_R1 = (
    "| u10_r1 | 10 | 100 | 0 | 25.00 | 18.00 | 12.00 | 22.00 | 32.00 | 62.00 | 104 | +4 |"
)
EXPECTED_RUN_U10_R2 = (
    "| u10_r2 | 10 | 200 | 0 | 50.00 | 24.00 | 18.00 | 28.00 | 38.00 | 72.00 | 206 | +6 |"
)

EXPECTED_ROW_U5_3REPEATS = (
    "| 5 | 3 | 20.00 (10.00–30.00) | 21.00 (15.00–27.00) | 15.00 (10.00–18.00) "
    "| 25.00 (20.00–28.00) | 35.00 (30.00–38.00) | 55.00 (45.00–65.00) "
    "| 250.00 (200.00–300.00) | 600 | 0 |"
)
EXPECTED_RUN_U5_R3 = (
    "| u5_r3 | 5 | 300 | 0 | 30.00 | 27.00 | 18.00 | 28.00 | 38.00 | 65.00 | 304 | +4 |"
)


def _write_u5_r1(d: Path) -> None:
    entry = _make_locust_entry(
        num_requests=100,
        total_response_time=1500.0,
        min_response_time=10.0,
        max_response_time=45.0,
        response_times={"10": 60, "20": 36, "30": 3, "40": 1},
        duration_s=10.0,
    )
    _write_run_files(d, "u5_r1", entry, _prom_after(112.0, 22.4))


def _write_u5_r2(d: Path) -> None:
    entry = _make_locust_entry(
        num_requests=200,
        total_response_time=4200.0,
        min_response_time=12.0,
        max_response_time=55.0,
        response_times={"15": 110, "25": 81, "35": 8, "50": 1},
        duration_s=10.0,
    )
    _write_run_files(d, "u5_r2", entry, _prom_after(213.0, 52.75))


def _write_u5_r3(d: Path) -> None:
    entry = _make_locust_entry(
        num_requests=300,
        total_response_time=8100.0,
        min_response_time=15.0,
        max_response_time=65.0,
        response_times={"18": 160, "28": 126, "38": 12, "60": 2},
        duration_s=10.0,
    )
    _write_run_files(d, "u5_r3", entry, _prom_after(314.0, 93.2))


def _write_u10_r1(d: Path) -> None:
    entry = _make_locust_entry(
        num_requests=100,
        total_response_time=1800.0,
        min_response_time=11.0,
        max_response_time=62.0,
        response_times={"12": 55, "22": 41, "32": 3, "60": 1},
        duration_s=4.0,
    )
    _write_run_files(d, "u10_r1", entry, _prom_after(114.0, 33.2))


def _write_u10_r2(d: Path) -> None:
    entry = _make_locust_entry(
        num_requests=200,
        total_response_time=4800.0,
        min_response_time=14.0,
        max_response_time=72.0,
        response_times={"18": 105, "28": 86, "38": 8, "70": 1},
        duration_s=4.0,
    )
    _write_run_files(d, "u10_r2", entry, _prom_after(216.0, 84.4))


def _write_run_trio(
    directory: Path,
    users: int,
    repeat: int,
    num_requests: int = 100,
    after_prom: str | None = None,
    num_failures: int = 0,
) -> None:
    """Helper used by error-handling tests."""
    stem = f"u{users}_r{repeat}"
    entry = _make_locust_entry(
        num_requests=num_requests,
        total_response_time=1500.0,
        min_response_time=10.0,
        max_response_time=45.0,
        response_times={"10": 60, "20": 36, "30": 3, "40": 1},
        duration_s=10.0,
        num_failures=num_failures,
    )
    after = after_prom if after_prom is not None else _prom_after(112.0, 22.4)
    _write_run_files(directory, stem, entry, after)


# ---------------------------------------------------------------------------
# _median unit tests
# ---------------------------------------------------------------------------


def test_median_odd() -> None:
    assert _median([3.0, 1.0, 2.0]) == 2.0


def test_median_even() -> None:
    assert _median([9.0, 1.0, 5.0, 3.0]) == 4.0


def test_median_empty_raises() -> None:
    with pytest.raises(ValueError):
        _median([])


def test_fmt_range_single() -> None:
    result = _fmt_range([5.0])
    assert result == "5.00 (5.00\u20135.00)"


def test_fmt_range_two_values() -> None:
    result = _fmt_range([4.0, 2.0])
    assert result == "3.00 (2.00\u20134.00)"


# ---------------------------------------------------------------------------
# discover_runs
# ---------------------------------------------------------------------------


def test_discover_runs_missing_before(tmp_path: Path) -> None:
    (tmp_path / "u5_r1.json").write_text("[]", encoding="utf-8")
    (tmp_path / "u5_r1_after.txt").write_text("x 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="u5_r1_before.txt"):
        discover_runs(tmp_path)


def test_discover_runs_missing_after(tmp_path: Path) -> None:
    (tmp_path / "u5_r1.json").write_text("[]", encoding="utf-8")
    (tmp_path / "u5_r1_before.txt").write_text("x 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="u5_r1_after.txt"):
        discover_runs(tmp_path)


# ---------------------------------------------------------------------------
# Fixture: 2 user counts x 2 repeats
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_user_two_repeat_dir(tmp_path: Path) -> Path:
    _write_u5_r1(tmp_path)
    _write_u5_r2(tmp_path)
    _write_u10_r1(tmp_path)
    _write_u10_r2(tmp_path)
    return tmp_path


def test_main_success_exit_0(two_user_two_repeat_dir: Path) -> None:
    rc = main(["--dir", str(two_user_two_repeat_dir)])
    assert rc == 0


def test_main_output_contains_headings(
    two_user_two_repeat_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--dir", str(two_user_two_repeat_dir)])
    out = capsys.readouterr().out
    assert "## Load Test Aggregate Summary: `/predict`" in out
    assert "### Per-Configuration Results" in out
    assert "### Per-Run Results" in out
    assert "### Notes" in out
    assert CAVEAT in out


def test_main_per_config_table_complete_rows(
    two_user_two_repeat_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Assert complete per-configuration rows for users=5 and users=10."""
    main(["--dir", str(two_user_two_repeat_dir)])
    out = capsys.readouterr().out
    assert EXPECTED_ROW_U5 in out
    assert EXPECTED_ROW_U10 in out


def test_main_per_run_table_complete_rows(
    two_user_two_repeat_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Assert complete per-run rows for all 4 runs."""
    main(["--dir", str(two_user_two_repeat_dir)])
    out = capsys.readouterr().out
    assert EXPECTED_RUN_U5_R1 in out
    assert EXPECTED_RUN_U5_R2 in out
    assert EXPECTED_RUN_U10_R1 in out
    assert EXPECTED_RUN_U10_R2 in out


def test_main_throughput_median_users5(
    two_user_two_repeat_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--dir", str(two_user_two_repeat_dir)])
    out = capsys.readouterr().out
    assert EXPECTED_ROW_U5 in out


def test_main_throughput_median_users10(
    two_user_two_repeat_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    main(["--dir", str(two_user_two_repeat_dir)])
    out = capsys.readouterr().out
    assert EXPECTED_ROW_U10 in out


# ---------------------------------------------------------------------------
# 3-repeat odd-n test case
# ---------------------------------------------------------------------------


def test_main_odd_repeats_median_three_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Odd n=3 repeats for users=5 exercises the odd-n median path end to end."""
    _write_u5_r1(tmp_path)
    _write_u5_r2(tmp_path)
    _write_u5_r3(tmp_path)

    rc = main(["--dir", str(tmp_path)])
    assert rc == 0

    out = capsys.readouterr().out
    assert EXPECTED_ROW_U5_3REPEATS in out
    assert EXPECTED_RUN_U5_R1 in out
    assert EXPECTED_RUN_U5_R2 in out
    assert EXPECTED_RUN_U5_R3 in out


def test_main_writes_file(two_user_two_repeat_dir: Path) -> None:
    out_file = two_user_two_repeat_dir / "summary.md"
    rc = main(["--dir", str(two_user_two_repeat_dir), "--out", str(out_file)])
    assert rc == 0
    assert out_file.is_file()
    content = out_file.read_text(encoding="utf-8")
    assert EXPECTED_ROW_U5 in content
    assert EXPECTED_ROW_U10 in content


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_main_incomplete_trio_exits_1(tmp_path: Path) -> None:
    _write_run_trio(tmp_path, users=5, repeat=1, num_requests=100)
    (tmp_path / "u5_r2.json").write_text("[]", encoding="utf-8")
    (tmp_path / "u5_r2_before.txt").write_text(_prom_before(), encoding="utf-8")
    # _after.txt deliberately absent

    rc = main(["--dir", str(tmp_path)])
    assert rc == 1


def test_main_incomplete_trio_stderr_names_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "u5_r1.json").write_text("[]", encoding="utf-8")
    (tmp_path / "u5_r1_before.txt").write_text(_prom_before(), encoding="utf-8")

    main(["--dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert "u5_r1_after.txt" in err


def test_main_failing_reconcile_exits_1(tmp_path: Path) -> None:
    huge_after = (
        "stress_predict_latency_seconds_count 9999.0\nstress_predict_latency_seconds_sum 100.0\n"
    )
    _write_run_trio(tmp_path, users=5, repeat=1, num_requests=100, after_prom=huge_after)
    rc = main(["--dir", str(tmp_path)])
    assert rc == 1


def test_main_failing_reconcile_stderr_names_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    huge_after = (
        "stress_predict_latency_seconds_count 9999.0\nstress_predict_latency_seconds_sum 100.0\n"
    )
    _write_run_trio(tmp_path, users=5, repeat=1, num_requests=100, after_prom=huge_after)
    main(["--dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert "u5_r1.json" in err


def test_main_failures_gt_0_exits_1(tmp_path: Path) -> None:
    _write_run_trio(tmp_path, users=5, repeat=1, num_requests=100, num_failures=3)
    rc = main(["--dir", str(tmp_path)])
    assert rc == 1


def test_main_failures_gt_0_stderr_names_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_run_trio(tmp_path, users=5, repeat=1, num_requests=100, num_failures=3)
    main(["--dir", str(tmp_path)])
    err = capsys.readouterr().err
    assert "u5_r1.json" in err


def test_main_no_runs_exits_1(tmp_path: Path) -> None:
    rc = main(["--dir", str(tmp_path)])
    assert rc == 1


def test_main_directory_not_found_exits_1(tmp_path: Path) -> None:
    rc = main(["--dir", str(tmp_path / "nonexistent")])
    assert rc == 1


# ---------------------------------------------------------------------------
# Module invocation via -m
# ---------------------------------------------------------------------------


def test_cli_module_execution_help() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    res = subprocess.run(
        [sys.executable, "-m", "scripts.aggregate_loadtest", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--dir" in res.stdout
