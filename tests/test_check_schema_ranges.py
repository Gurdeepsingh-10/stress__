"""Tests for scripts/check_schema_ranges.py.

All tests use small synthetic DataFrames; no real data is loaded.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.check_schema_ranges import (
    FEATURE_COLS,
    _row_to_dict,
    load,
    summarize,
    validate_rows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_ROW: dict[str, object] = {
    "hrv_mean_hr": 75.0,
    "hrv_hr_std": 5.2,
    "hrv_sdnn": 45.0,
    "hrv_rmssd": 38.0,
    "hrv_pnn50": 0.15,
    "hrv_num_beats": 74,
    "eda_mean": 2.5,
    "eda_std": 0.4,
    "eda_min": 1.8,
    "eda_max": 3.4,
    "eda_slope": 0.002,
    "eda_num_peaks": 3,
    "temp_mean": 32.5,
    "temp_std": 0.15,
    "temp_slope": -0.001,
    "acc_magnitude_mean": 0.98,
    "acc_magnitude_std": 0.04,
}


def _make_df(*rows: dict) -> pd.DataFrame:
    """Build a DataFrame from one or more row dicts."""
    return pd.DataFrame(list(rows))


def _valid_df() -> pd.DataFrame:
    return _make_df(_VALID_ROW)


# ---------------------------------------------------------------------------
# FEATURE_COLS invariant
# ---------------------------------------------------------------------------


def test_feature_cols_derived_from_schema() -> None:
    """FEATURE_COLS must equal the ordered keys from PredictRequest.model_fields."""
    from stress.api.schemas import PredictRequest

    assert FEATURE_COLS == list(PredictRequest.model_fields.keys())
    assert len(FEATURE_COLS) == 17


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


def test_load_reads_csv(tmp_path: Path) -> None:
    """load() returns a DataFrame with the expected columns."""
    csv = tmp_path / "feat.csv"
    df_orig = _valid_df()
    df_orig.to_csv(csv, index=False)
    df = load(csv)
    assert isinstance(df, pd.DataFrame)
    assert set(FEATURE_COLS).issubset(df.columns)


# ---------------------------------------------------------------------------
# _row_to_dict()
# ---------------------------------------------------------------------------


def test_row_to_dict_nan_hrv_becomes_none() -> None:
    """NaN in an HRV nullable field must become Python None, not a float NaN."""
    row_data = dict(_VALID_ROW)
    row_data["hrv_mean_hr"] = float("nan")
    row = pd.Series(row_data)
    d = _row_to_dict(row)
    assert d["hrv_mean_hr"] is None


def test_row_to_dict_int_fields_cast() -> None:
    """hrv_num_beats and eda_num_peaks must be Python int, not numpy int64."""
    row = pd.Series(_VALID_ROW)
    d = _row_to_dict(row)
    assert isinstance(d["hrv_num_beats"], int)
    assert isinstance(d["eda_num_peaks"], int)


def test_row_to_dict_valid_floats_unchanged() -> None:
    """Non-NaN HRV values must be preserved as floats, not turned into None."""
    row = pd.Series(_VALID_ROW)
    d = _row_to_dict(row)
    assert d["hrv_mean_hr"] == 75.0
    assert d["hrv_rmssd"] == 38.0


# ---------------------------------------------------------------------------
# validate_rows()
# ---------------------------------------------------------------------------


def test_validate_rows_all_valid_passes() -> None:
    """An all-valid DataFrame produces zero failing rows."""
    df = _valid_df()
    failing, field_failures = validate_rows(df)
    assert failing == []
    assert field_failures == {}


def test_validate_rows_out_of_range_reports_field() -> None:
    """A row with acc_magnitude_mean > 20 fails and is reported under that field."""
    row = dict(_VALID_ROW)
    row["acc_magnitude_mean"] = 999.0  # far above le=20.0
    df = _make_df(row)
    failing, field_failures = validate_rows(df)
    assert len(failing) == 1
    assert "acc_magnitude_mean" in field_failures
    assert field_failures["acc_magnitude_mean"] == [0]


def test_validate_rows_multiple_bad_rows() -> None:
    """Two failing rows are both captured; one good row is not included."""
    bad1 = dict(_VALID_ROW)
    bad1["temp_mean"] = 5.0  # below ge=15.0
    bad2 = dict(_VALID_ROW)
    bad2["eda_mean"] = -1.0  # below ge=0.0
    good = dict(_VALID_ROW)
    df = _make_df(bad1, good, bad2)
    failing, field_failures = validate_rows(df)
    assert set(failing) == {0, 2}
    assert "temp_mean" in field_failures
    assert "eda_mean" in field_failures
    assert 1 not in failing  # good row untouched


def test_validate_rows_nan_hrv_passes_as_none() -> None:
    """A NaN in an HRV nullable field is converted to None and passes validation."""
    row = dict(_VALID_ROW)
    row["hrv_mean_hr"] = float("nan")
    df = _make_df(row)
    failing, field_failures = validate_rows(df)
    assert failing == []
    assert field_failures == {}


def test_validate_rows_nan_hrv_all_five_pass() -> None:
    """All five HRV nullable fields as NaN in the same row passes validation."""
    row = dict(_VALID_ROW)
    for f in ("hrv_mean_hr", "hrv_hr_std", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"):
        row[f] = float("nan")
    df = _make_df(row)
    failing, field_failures = validate_rows(df)
    assert failing == []
    assert field_failures == {}


def test_validate_rows_failure_count_per_field() -> None:
    """field_failures counts are accurate when multiple rows fail on the same field."""
    bad1 = dict(_VALID_ROW)
    bad1["hrv_mean_hr"] = 5.0  # below ge=30.0
    bad2 = dict(_VALID_ROW)
    bad2["hrv_mean_hr"] = 5.0
    good = dict(_VALID_ROW)
    df = _make_df(bad1, bad2, good)
    failing, field_failures = validate_rows(df)
    assert len(failing) == 2
    assert len(field_failures["hrv_mean_hr"]) == 2


# ---------------------------------------------------------------------------
# summarize() — smoke tests (no assertion on printed text, just no exceptions)
# ---------------------------------------------------------------------------


def test_summarize_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    """summarize() on a valid all-pass frame completes without exception."""
    df = _valid_df()
    # Add label columns so the label section is exercised
    df["label_binary"] = 0
    df["label_3class"] = 1
    failing, field_failures = validate_rows(df)
    summarize(df, failing, field_failures)
    out = capsys.readouterr().out
    assert "SCHEMA RANGE CHECK REPORT" in out
    assert "OBSERVED STATISTICS" in out
    assert "LABEL DISTRIBUTION" in out
    assert "Failing    : 0" in out


def test_summarize_shows_field_failures(capsys: pytest.CaptureFixture[str]) -> None:
    """summarize() lists failing field names and row counts when failures exist."""
    row = dict(_VALID_ROW)
    row["acc_magnitude_mean"] = 50.0
    df = _make_df(row)
    failing, field_failures = validate_rows(df)
    summarize(df, failing, field_failures)
    out = capsys.readouterr().out
    assert "acc_magnitude_mean" in out
    assert "Failing    : 1" in out


# ---------------------------------------------------------------------------
# main() exit-code tests
# ---------------------------------------------------------------------------


def test_main_exits_0_on_valid_csv(tmp_path: Path) -> None:
    """main() returns 0 when every row passes validation."""
    from scripts.check_schema_ranges import main

    csv = tmp_path / "valid.csv"
    _valid_df().to_csv(csv, index=False)
    assert main(["--csv", str(csv)]) == 0


def test_main_exits_1_on_invalid_csv(tmp_path: Path) -> None:
    """main() returns 1 when at least one row fails validation."""
    from scripts.check_schema_ranges import main

    bad_row = dict(_VALID_ROW)
    bad_row["temp_mean"] = 100.0  # well above le=45.0
    csv = tmp_path / "bad.csv"
    _make_df(bad_row).to_csv(csv, index=False)
    assert main(["--csv", str(csv)]) == 1


def test_main_exits_1_on_missing_csv(tmp_path: Path) -> None:
    """main() returns 1 when the CSV path does not exist."""
    from scripts.check_schema_ranges import main

    assert main(["--csv", str(tmp_path / "nonexistent.csv")]) == 1


# ---------------------------------------------------------------------------
# NaN-to-None conversion integrity test (caught by mutation check)
# ---------------------------------------------------------------------------


def test_nan_converted_to_none_not_nan_float() -> None:
    """_row_to_dict must produce None, not a NaN float, for HRV fields.

    This test is the mutation-check target: if NaN→None conversion is removed
    so that NaN is passed directly, validate_rows will raise a ValidationError
    and this test (via validate_rows) will detect a spurious failure.
    """
    row_data = dict(_VALID_ROW)
    row_data["hrv_sdnn"] = float("nan")
    row = pd.Series(row_data)

    d = _row_to_dict(row)

    # Must be exactly None, not float NaN
    assert d["hrv_sdnn"] is None
    assert not isinstance(d["hrv_sdnn"], float)

    # And the full validation path must also pass
    df = _make_df(row_data)
    failing, _ = validate_rows(df)
    assert failing == [], "NaN HRV should pass as None; conversion may be broken"
