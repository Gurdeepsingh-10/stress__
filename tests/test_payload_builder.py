"""Unit tests for stress.api.payload_builder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from stress.api.payload_builder import (
    _HRV_NULLABLE,
    _INT_FIELDS,
    FEATURE_COLS,
    load_payloads,
    row_to_dict,
)
from stress.api.schemas import PredictRequest

_SAMPLE_VALID_ROW: dict[str, Any] = {
    "hrv_mean_hr": 72.5,
    "hrv_hr_std": 4.1,
    "hrv_sdnn": 48.0,
    "hrv_rmssd": 35.0,
    "hrv_pnn50": 0.12,
    "hrv_num_beats": 70,
    "eda_mean": 2.1,
    "eda_std": 0.3,
    "eda_min": 1.5,
    "eda_max": 2.9,
    "eda_slope": 0.001,
    "eda_num_peaks": 4,
    "temp_mean": 33.1,
    "temp_std": 0.1,
    "temp_slope": -0.0005,
    "acc_magnitude_mean": 0.99,
    "acc_magnitude_std": 0.03,
}


def test_feature_cols_matches_schema() -> None:
    assert FEATURE_COLS == list(PredictRequest.model_fields.keys())
    assert len(FEATURE_COLS) == 17


def test_row_to_dict_valid() -> None:
    d = row_to_dict(_SAMPLE_VALID_ROW)
    assert len(d) == 17
    assert isinstance(d["hrv_num_beats"], int)
    assert isinstance(d["eda_num_peaks"], int)
    assert isinstance(d["temp_mean"], float)


def test_row_to_dict_hrv_nullable_nan_becomes_none() -> None:
    row = dict(_SAMPLE_VALID_ROW)
    for col in _HRV_NULLABLE:
        row[col] = float("nan")

    d = row_to_dict(row)
    for col in _HRV_NULLABLE:
        assert d[col] is None


def test_row_to_dict_hrv_float_values_preserved() -> None:
    row = dict(_SAMPLE_VALID_ROW)
    row["hrv_mean_hr"] = 80.5
    d = row_to_dict(row)
    assert d["hrv_mean_hr"] == 80.5


def test_row_to_dict_int_field_nan_raises_value_error() -> None:
    for int_col in _INT_FIELDS:
        row = dict(_SAMPLE_VALID_ROW)
        row[int_col] = float("nan")
        with pytest.raises(ValueError, match=f"column '{int_col}' is NaN") as exc_info:
            row_to_dict(row, row_index=3)
        assert "Row 3:" in str(exc_info.value)


def test_row_to_dict_non_nullable_float_nan_raises_value_error() -> None:
    row = dict(_SAMPLE_VALID_ROW)
    row["temp_mean"] = float("nan")
    with pytest.raises(ValueError, match="column 'temp_mean' is NaN") as exc_info:
        row_to_dict(row, row_index=7)
    assert "Row 7:" in str(exc_info.value)


def test_row_to_dict_no_row_index_raises_value_error() -> None:
    row = dict(_SAMPLE_VALID_ROW)
    row["acc_magnitude_mean"] = float("nan")
    with pytest.raises(ValueError, match="column 'acc_magnitude_mean' is NaN") as exc_info:
        row_to_dict(row)
    assert "Row" not in str(exc_info.value) or "Row None" not in str(exc_info.value)


def test_load_payloads_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_payloads("nonexistent_path_to_csv_file.csv")


def test_load_payloads_missing_columns(tmp_path: Path) -> None:
    csv_file = tmp_path / "missing_cols.csv"
    pd.DataFrame([{"hrv_mean_hr": 70.0}]).to_csv(csv_file, index=False)
    with pytest.raises(ValueError, match="CSV missing required feature columns"):
        load_payloads(csv_file)


def test_load_payloads_success_and_n_limit(tmp_path: Path) -> None:
    csv_file = tmp_path / "valid.csv"
    rows = [dict(_SAMPLE_VALID_ROW) for _ in range(10)]
    pd.DataFrame(rows).to_csv(csv_file, index=False)

    # Load all (n=None or n=0)
    all_payloads = load_payloads(csv_file)
    assert len(all_payloads) == 10

    all_payloads_zero = load_payloads(csv_file, n=0)
    assert len(all_payloads_zero) == 10

    # Load first 3
    first_three = load_payloads(csv_file, n=3)
    assert len(first_three) == 3


def test_load_payloads_invalid_row_raises_with_index(tmp_path: Path) -> None:
    csv_file = tmp_path / "with_nan.csv"
    row1 = dict(_SAMPLE_VALID_ROW)
    row2 = dict(_SAMPLE_VALID_ROW)
    row2["eda_mean"] = float("nan")
    pd.DataFrame([row1, row2]).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Row 1: column 'eda_mean' is NaN"):
        load_payloads(csv_file)
