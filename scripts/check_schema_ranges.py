"""Validate feature rows from the processed CSV against the PredictRequest schema.

Usage:
    python scripts/check_schema_ranges.py [--csv PATH]

Exit code 1 if any row fails validation, 0 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from stress.api.payload_builder import (
    _HRV_NULLABLE,
    _INT_FIELDS,
    FEATURE_COLS,
    row_to_dict,
)
from stress.api.schemas import PredictRequest

__all__ = [
    "FEATURE_COLS",
    "_HRV_NULLABLE",
    "_INT_FIELDS",
    "_row_to_dict",
    "load",
    "summarize",
    "validate_rows",
]

# Default CSV path relative to repo root
_DEFAULT_CSV = Path(__file__).parent.parent / "data" / "processed" / "features_phase2.csv"

# Backward-compatible alias for existing tests and scripts
_row_to_dict = row_to_dict


# ---------------------------------------------------------------------------
# Public functions (also used by tests)
# ---------------------------------------------------------------------------


def load(csv_path: Path) -> pd.DataFrame:
    """Load the feature CSV and return the full DataFrame (all columns)."""
    return pd.read_csv(csv_path)


def validate_rows(
    df: pd.DataFrame,
) -> tuple[list[int], dict[str, list[int]]]:
    """Validate each row against PredictRequest.

    Returns
    -------
    failing_indices : list[int]
        Row indices (0-based in df) that failed validation.
    field_failures : dict[str, list[int]]
        Mapping field name → list of row indices where that field caused a failure.
    """
    failing_indices: list[int] = []
    field_failures: dict[str, list[int]] = {}

    for idx in range(len(df)):
        row = df.iloc[idx]
        try:
            d = _row_to_dict(row, row_index=idx)
            PredictRequest(**d)
        except ValidationError as exc:
            failing_indices.append(idx)
            for err in exc.errors():
                locs = err.get("loc", ())
                if locs:
                    field = str(locs[0])
                    field_failures.setdefault(field, []).append(idx)
        except ValueError as exc:
            failing_indices.append(idx)
            for col in FEATURE_COLS:
                if f"'{col}'" in str(exc):
                    field_failures.setdefault(col, []).append(idx)
                    break

    return failing_indices, field_failures


def _get_schema_bounds(field: str) -> tuple[float | None, float | None]:
    """Return (ge, le) bounds from schema metadata for a given field."""
    fi = PredictRequest.model_fields[field]
    ge = le = None
    for m in fi.metadata:
        if hasattr(m, "ge"):
            ge = m.ge
        if hasattr(m, "le"):
            le = m.le
    return ge, le


def summarize(
    df: pd.DataFrame,
    failing_indices: list[int],
    field_failures: dict[str, list[int]],
) -> None:
    """Print the full validation report to stdout."""
    total = len(df)
    n_fail = len(failing_indices)
    n_pass = total - n_fail

    print("=" * 70)
    print("SCHEMA RANGE CHECK REPORT")
    print("=" * 70)
    print(f"Total rows : {total}")
    print(f"Passing    : {n_pass}")
    print(f"Failing    : {n_fail}")
    print()

    # ── Per-field failure report ────────────────────────────────────────────
    if field_failures:
        print("FIELD FAILURES")
        print("-" * 70)
        header = (
            f"{'Field':<22} {'Fail#':>6}  {'Obs min':>12}  {'Obs max':>12}"
            f"  {'Schema ge':>10}  {'Schema le':>10}"
        )
        print(header)
        print("-" * 70)
        for field, rows in sorted(field_failures.items()):
            obs = df[field] if field in df.columns else pd.Series(dtype=float)
            obs_min = obs.min() if not obs.empty else float("nan")
            obs_max = obs.max() if not obs.empty else float("nan")
            ge, le = _get_schema_bounds(field) if field in FEATURE_COLS else (None, None)
            ge_str = f"{ge}" if ge is not None else "-"
            le_str = f"{le}" if le is not None else "-"
            print(
                f"{field:<22} {len(rows):>6}"
                f"  {obs_min:>12.4f}  {obs_max:>12.4f}"
                f"  {ge_str:>10}  {le_str:>10}"
            )
        print()
    else:
        print("No field failures.\n")

    # ── Observed statistics for all 17 features ─────────────────────────────
    print("OBSERVED STATISTICS FOR ALL 17 FEATURES")
    print("-" * 70)
    header2 = (
        f"{'Field':<22} {'Obs min':>12}  {'Obs max':>12}  {'NaN count':>10}"
        f"  {'Schema ge':>10}  {'Schema le':>10}"
    )
    print(header2)
    print("-" * 70)
    for field in FEATURE_COLS:
        col = df[field]
        obs_min = col.min()
        obs_max = col.max()
        nan_count = int(col.isna().sum())
        ge, le = _get_schema_bounds(field)
        ge_str = f"{ge}" if ge is not None else "-"
        le_str = f"{le}" if le is not None else "-"
        print(
            f"{field:<22} {obs_min:>12.4f}  {obs_max:>12.4f}  {nan_count:>10}"
            f"  {ge_str:>10}  {le_str:>10}"
        )
    print()

    # ── Label distribution ──────────────────────────────────────────────────
    if "label_binary" in df.columns and "label_3class" in df.columns:
        print("LABEL DISTRIBUTION  (label_binary x label_3class)")
        print("-" * 70)
        ct = pd.crosstab(df["label_binary"], df["label_3class"], margins=True)
        ct.index.name = "label_binary"
        print(ct.to_string())
        print()
        print("label_binary counts:")
        print(df["label_binary"].value_counts().sort_index().to_string())
        print()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate feature rows against PredictRequest schema."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_CSV,
        help=f"Path to feature CSV (default: {_DEFAULT_CSV})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the schema range check. Returns exit code."""
    args = _parse_args(argv)
    csv_path: Path = args.csv

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        return 1

    df = load(csv_path)
    failing_indices, field_failures = validate_rows(df)
    summarize(df, failing_indices, field_failures)

    return 1 if failing_indices else 0


if __name__ == "__main__":
    sys.exit(main())
