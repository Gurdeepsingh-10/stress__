"""Build the windowed feature table and run a sanity check on it.

Usage:
    .venv/Scripts/python.exe scripts/extract_features.py
    .venv/Scripts/python.exe scripts/extract_features.py --out data/processed/features_phase2.csv

Exit code is 1 if any sanity check fails (stress must show higher HR and EDA than baseline).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from stress.features.extract import extract_features_all_subjects

# Features shown in the per-class mean table
KEY_FEATURES = ("hrv_mean_hr", "eda_mean", "hrv_num_beats", "acc_magnitude_mean")

# Sanity expectations: (feature, readable name) where stress must exceed baseline
EXPECTED_UP = (
    ("hrv_mean_hr", "mean HR"),
    ("eda_mean", "mean EDA"),
)

BASELINE_LABEL = 1
STRESS_LABEL = 2


def class_means(df: pd.DataFrame, features: tuple[str, ...] = KEY_FEATURES) -> pd.DataFrame:
    """Mean of each key feature per 3-class label.

    Args:
        df: Feature table with a 'label_3class' column.
        features: Feature columns to average.

    Returns:
        DataFrame indexed by label, one column per available feature.
    """
    present = [name for name in features if name in df.columns]
    return df.groupby("label_3class")[present].mean()


def sanity_checks(df: pd.DataFrame) -> list[tuple[str, bool]]:
    """Check that stress windows show higher HR and EDA than baseline windows.

    Args:
        df: Feature table with 'label_3class' and the key feature columns.

    Returns:
        List of (message, passed) pairs, one per expectation.
    """
    means = class_means(df)
    results: list[tuple[str, bool]] = []

    for feature, label in EXPECTED_UP:
        if feature not in means.columns:
            results.append((f"{label}: cannot evaluate (column '{feature}' missing)", False))
            continue
        if BASELINE_LABEL not in means.index or STRESS_LABEL not in means.index:
            results.append((f"{label}: cannot evaluate (baseline or stress class missing)", False))
            continue

        baseline = means.loc[BASELINE_LABEL, feature]
        stress = means.loc[STRESS_LABEL, feature]
        passed = bool(stress > baseline)
        verdict = "PASS" if passed else "FAIL"
        results.append(
            (
                f"{label}: baseline={baseline:8.3f}  stress={stress:8.3f}  "
                f"delta={stress - baseline:+8.3f}  -> {verdict}",
                passed,
            )
        )

    return results


def nan_rates(df: pd.DataFrame) -> pd.Series:
    """NaN rate per feature column, worst first."""
    feature_cols = [
        name
        for name in df.columns
        if name not in ("subject", "window_start_s", "label_3class", "label_binary")
    ]
    return df[feature_cols].isna().mean().sort_values(ascending=False)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract windowed features for all subjects and sanity check the table."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed .npz directory (default: data/processed)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/features_phase2.csv"),
        help="Output path, .csv or .parquet (default: data/processed/features_phase2.csv)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build the feature table, print the summary, return exit code."""
    args = parse_args(argv)

    df = extract_features_all_subjects(str(args.processed_dir), str(args.out))

    if df.empty:
        print(f"ERROR: no windows extracted from {args.processed_dir}", file=sys.stderr)
        return 1

    print(f"\nWrote {args.out}: {len(df)} windows, {df['subject'].nunique()} subjects")

    counts = df["label_3class"].value_counts().sort_index()
    count_str = ", ".join(f"{int(label)}={int(n)}" for label, n in counts.items())
    print(f"Window counts by class (1=baseline, 2=stress, 3=amusement): {count_str}")

    print("\nPer-class feature means:")
    print(class_means(df).round(3).to_string())

    print("\nHighest NaN rates:")
    for name, rate in nan_rates(df).head(8).items():
        print(f"  {name:24s} {rate:6.1%}")

    print("\nSanity checks (expect stress > baseline):")
    results = sanity_checks(df)
    for message, _ in results:
        print(f"  {message}")

    failed = [message for message, passed in results if not passed]
    if failed:
        print(f"\nSANITY CHECK FAILED: {len(failed)} expectation(s) not met")
        return 1

    print("\nSANITY CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
