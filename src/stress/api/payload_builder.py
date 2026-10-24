"""Payload builder module for converting feature rows to API request payloads.

Pure module shared between offline schema validation and load testing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stress.api.schemas import PredictRequest

# HRV interval fields that accept None when peaks are insufficient
_HRV_NULLABLE: frozenset[str] = frozenset(
    {"hrv_mean_hr", "hrv_hr_std", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"}
)

# Integer fields that must be cast from float64
_INT_FIELDS: frozenset[str] = frozenset({"hrv_num_beats", "eda_num_peaks"})

# Derive the 17 feature column names directly from the schema — never hand-typed
FEATURE_COLS: list[str] = list(PredictRequest.model_fields.keys())


def row_to_dict(
    row: pd.Series | dict[str, Any],
    row_index: int | None = None,
) -> dict[str, Any]:
    """Convert one CSV row to a dict suitable for PredictRequest validation.

    Rules:
    - HRV nullable fields: pandas NaN -> Python None; non-NaN converted to float.
    - Integer fields: cast to int; NaN raises ValueError naming row index and column.
    - Non-nullable float fields: pass through as float; NaN raises ValueError
      naming row index and column.

    Args:
        row: Row mapping (Series or dict) containing feature columns.
        row_index: Optional row index (0-based) for precise error reporting.

    Returns:
        Dictionary mapping feature column names to converted values.

    Raises:
        ValueError: If a non-nullable column contains NaN or cannot be converted.
    """
    row_prefix = f"Row {row_index}: " if row_index is not None else ""
    d: dict[str, Any] = {}

    for col in FEATURE_COLS:
        val = row[col]
        if col in _HRV_NULLABLE:
            d[col] = None if pd.isna(val) else float(val)
        elif col in _INT_FIELDS:
            if pd.isna(val):
                raise ValueError(f"{row_prefix}column '{col}' is NaN, expected int")
            try:
                d[col] = int(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{row_prefix}column '{col}' cannot be cast to int: {val}"
                ) from exc
        else:
            if pd.isna(val):
                raise ValueError(f"{row_prefix}column '{col}' is NaN, expected float")
            try:
                d[col] = float(val)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"{row_prefix}column '{col}' cannot be cast to float: {val}"
                ) from exc

    return d


def load_payloads(
    csv_path: str | Path,
    n: int | None = None,
) -> list[dict[str, Any]]:
    """Load the first N feature rows from CSV as PredictRequest payload dictionaries.

    Args:
        csv_path: Path to the processed features CSV file.
        n: Optional maximum number of rows to load. If None or <= 0, all rows are loaded.

    Returns:
        List of payload dictionaries ready for API requests.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required feature columns are missing or if any row is invalid.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    missing_cols = set(FEATURE_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required feature columns: {sorted(missing_cols)}")

    if n is not None and n > 0:
        df = df.iloc[:n]

    payloads: list[dict[str, Any]] = []
    for idx, (_, row) in enumerate(df.iterrows()):
        payloads.append(row_to_dict(row, row_index=idx))

    return payloads
