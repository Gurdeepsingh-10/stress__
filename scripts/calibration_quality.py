"""Exploratory post hoc analysis of calibration window representativeness.

EXPLORATORY, POST HOC ANALYSIS ONLY. NOT PRE-REGISTERED.
Investigates whether subjects whose macro-F1 degraded under Method A (logistic,
group all, k=10) had their first 10 baseline windows unrepresentative of the
remainder of their baseline period in T_eval.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from stress.personalization.splitting import get_calibration_windows, split_calibration_and_test

TARGET_FEATURES: tuple[str, ...] = ("hrv_mean_hr", "eda_mean", "temp_mean")
DISCLAIMER_NOTE: str = "NOTE: The sample is 15 subjects, so correlations are descriptive only."


def compute_feature_gap_z(
    cal_df: pd.DataFrame,
    later_df: pd.DataFrame,
    feature: str,
) -> float:
    """Compute normalized gap between calibration and later baseline windows.

    gap_z = (mean(cal) - mean(later)) / std(later)

    Args:
        cal_df: Calibration baseline windows.
        later_df: Later baseline windows in T_eval.
        feature: Feature column name.

    Returns:
        Normalized gap_z, or NaN if std(later) == 0 or later has insufficient samples.
    """
    if cal_df.empty or later_df.empty:
        return float("nan")

    if feature not in cal_df.columns or feature not in later_df.columns:
        return float("nan")

    cal_vals = cal_df[feature].dropna().to_numpy(dtype=float)
    later_vals = later_df[feature].dropna().to_numpy(dtype=float)

    if len(cal_vals) == 0 or len(later_vals) < 2:
        return float("nan")

    mean_cal = float(np.mean(cal_vals))
    mean_later = float(np.mean(later_vals))
    std_later = float(np.std(later_vals, ddof=1))

    if np.isnan(std_later) or std_later == 0.0:
        return float("nan")

    return (mean_cal - mean_later) / std_later


def extract_subject_baseline_windows(
    s_df: pd.DataFrame,
    *,
    k: int = 10,
    k_max: int = 10,
    target_col: str = "label_3class",
    baseline_val: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extract first k calibration windows and subsequent baseline windows from T_eval.

    Uses get_calibration_windows for calibration and split_calibration_and_test(k_max)
    for T_eval, filtering test_df to true-baseline windows only.

    Args:
        s_df: Single subject DataFrame ordered by time.
        k: Number of calibration windows (default: 10).
        k_max: Max calibration windows determining T_eval buffer (default: 10).
        target_col: Column identifying label condition.
        baseline_val: Value identifying baseline condition (default: 1).

    Returns:
        Tuple of (cal_windows_df, later_baseline_windows_df).
    """
    cal_df = get_calibration_windows(
        s_df,
        k=k,
        target_col=target_col,
        baseline_val=baseline_val,
    )
    split = split_calibration_and_test(
        s_df,
        k=k,
        k_max=k_max,
        target_col=target_col,
        baseline_val=baseline_val,
    )
    later_baseline_df = split.test_df[split.test_df[target_col] == baseline_val].copy()
    return cal_df, later_baseline_df


def compute_subject_gaps(
    s_df: pd.DataFrame,
    features: Sequence[str] = TARGET_FEATURES,
    k: int = 10,
    k_max: int = 10,
) -> dict[str, float]:
    """Compute gap_z for each target feature for a single subject.

    Args:
        s_df: Single subject DataFrame.
        features: Feature names to evaluate.
        k: Calibration window count (default: 10).
        k_max: Max calibration window count for eval set (default: 10).

    Returns:
        Dict mapping feature name to its gap_z score.
    """
    cal_df, later_df = extract_subject_baseline_windows(s_df, k=k, k_max=k_max)
    return {feat: compute_feature_gap_z(cal_df, later_df, feat) for feat in features}


def compute_per_subject_deltas(
    results_df: pd.DataFrame,
    *,
    model: str = "logistic",
    group: str = "all",
    k: int = 10,
) -> pd.DataFrame:
    """Extract paired delta_f1 and delta_stress_recall (Method A k vs Reference k=0).

    Args:
        results_df: Per-subject personalization experiment results.
        model: Target model (default: logistic).
        group: Target feature group (default: all).
        k: Target Method A calibration count (default: 10).

    Returns:
        DataFrame with columns [subject, delta_f1, delta_stress_recall].
    """
    ref_mask = (
        (results_df["method"] == "reference")
        & (results_df["model"] == model)
        & (results_df["group"] == group)
        & (results_df["k"] == 0)
    )
    ref_sub = results_df[ref_mask].set_index("subject")

    method_a_mask = (
        (results_df["method"] == "method_a")
        & (results_df["model"] == model)
        & (results_df["group"] == group)
        & (results_df["k"] == k)
    )
    a_sub = results_df[method_a_mask].set_index("subject")

    common_subjects = sorted(set(ref_sub.index).intersection(set(a_sub.index)))
    rows: list[dict[str, Any]] = []
    for s in common_subjects:
        r_f1 = float(ref_sub.loc[s, "macro_f1"])
        a_f1 = float(a_sub.loc[s, "macro_f1"])
        r_rec = float(ref_sub.loc[s, "recall_stress"])
        a_rec = float(a_sub.loc[s, "recall_stress"])
        rows.append(
            {
                "subject": s,
                "delta_f1": a_f1 - r_f1,
                "delta_stress_recall": a_rec - r_rec,
            }
        )

    return pd.DataFrame(rows)


def analyze_calibration_quality(
    features_df: pd.DataFrame,
    results_df: pd.DataFrame,
    *,
    features: Sequence[str] = TARGET_FEATURES,
    model: str = "logistic",
    group: str = "all",
    k: int = 10,
    k_max: int = 10,
    subject_col: str = "subject",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Run exploratory calibration quality analysis and Spearman correlations.

    Args:
        features_df: Windowed features DataFrame.
        results_df: Per-subject personalization results DataFrame.
        features: Target physiological feature columns.
        model: Target model name (default: logistic).
        group: Target feature group (default: all).
        k: Calibration window count (default: 10).
        k_max: Maximum calibration window count (default: 10).
        subject_col: Column identifying subjects.

    Returns:
        Tuple of (per_subject_diagnostic_df, correlation_records).
    """
    deltas_df = compute_per_subject_deltas(results_df, model=model, group=group, k=k)

    diagnostic_rows: list[dict[str, Any]] = []
    for _, delta_row in deltas_df.iterrows():
        s = delta_row["subject"]
        s_mask = features_df[subject_col] == s
        s_features = features_df[s_mask].copy()

        gaps = compute_subject_gaps(s_features, features=features, k=k, k_max=k_max)
        row_dict: dict[str, Any] = {
            "subject": s,
            **{f"gap_z_{feat}": gaps[feat] for feat in features},
            "delta_f1": delta_row["delta_f1"],
            "delta_stress_recall": delta_row["delta_stress_recall"],
        }
        diagnostic_rows.append(row_dict)

    diag_df = pd.DataFrame(diagnostic_rows)

    # Compute Spearman rank correlations of delta_f1 vs gap_z and vs |gap_z|
    correlations: list[dict[str, Any]] = []
    f1_vals = diag_df["delta_f1"].to_numpy(dtype=float)

    for feat in features:
        col_signed = f"gap_z_{feat}"
        vals_signed = diag_df[col_signed].to_numpy(dtype=float)

        # Signed correlation
        valid_mask = ~np.isnan(vals_signed) & ~np.isnan(f1_vals)
        n_valid = int(valid_mask.sum())
        if n_valid >= 3:
            res_signed = spearmanr(vals_signed[valid_mask], f1_vals[valid_mask])
            r_signed, p_signed = float(res_signed.statistic), float(res_signed.pvalue)
        else:
            r_signed, p_signed = float("nan"), float("nan")

        correlations.append(
            {
                "feature": feat,
                "variant": "gap_z (signed)",
                "spearman_r": r_signed,
                "p_value": p_signed,
                "n": n_valid,
            }
        )

        # Absolute correlation
        vals_abs = np.abs(vals_signed)
        valid_mask_abs = ~np.isnan(vals_abs) & ~np.isnan(f1_vals)
        n_valid_abs = int(valid_mask_abs.sum())
        if n_valid_abs >= 3:
            res_abs = spearmanr(vals_abs[valid_mask_abs], f1_vals[valid_mask_abs])
            r_abs, p_abs = float(res_abs.statistic), float(res_abs.pvalue)
        else:
            r_abs, p_abs = float("nan"), float("nan")

        correlations.append(
            {
                "feature": feat,
                "variant": "|gap_z| (magnitude)",
                "spearman_r": r_abs,
                "p_value": p_abs,
                "n": n_valid_abs,
            }
        )

    return diag_df, correlations


def format_calibration_report(
    diag_df: pd.DataFrame,
    correlations: list[dict[str, Any]],
    *,
    features: Sequence[str] = TARGET_FEATURES,
) -> str:
    """Format calibration quality diagnostic into a plain-ASCII report.

    Args:
        diag_df: Per-subject diagnostic DataFrame.
        correlations: List of correlation dicts.
        features: Target physiological feature columns.

    Returns:
        Formatted ASCII report string.
    """
    banner = "=" * 85
    lines = [
        banner,
        "EXPLORATORY POST HOC ANALYSIS: Calibration Representativeness Diagnostic",
        (
            "Target condition: Logistic Regression, Feature Group 'all', "
            "Method A (k=10) vs Reference (k=0)"
        ),
        banner,
        f"{DISCLAIMER_NOTE}\n",
    ]

    # Per-subject table
    header = (
        f"{'Subject':<9} "
        f"{'gap_z(hr)':<13} "
        f"{'gap_z(eda)':<13} "
        f"{'gap_z(temp)':<13} "
        f"{'delta_f1':<11} "
        f"{'delta_stress_rec':<16}"
    )
    divider = "-" * len(header)
    lines.extend([header, divider])

    for _, row in diag_df.iterrows():
        subj = row["subject"]
        if isinstance(subj, (int, np.integer)):
            s = str(subj)
        else:
            try:
                f_val = float(subj)
                s = str(int(f_val)) if f_val.is_integer() else str(subj)
            except (ValueError, TypeError):
                s = str(subj)
        g_hr = (
            f"{row[f'gap_z_{features[0]}']:+.4f}"
            if not np.isnan(row[f"gap_z_{features[0]}"])
            else "NaN"
        )
        g_eda = (
            f"{row[f'gap_z_{features[1]}']:+.4f}"
            if not np.isnan(row[f"gap_z_{features[1]}"])
            else "NaN"
        )
        g_temp = (
            f"{row[f'gap_z_{features[2]}']:+.4f}"
            if not np.isnan(row[f"gap_z_{features[2]}"])
            else "NaN"
        )
        d_f1 = f"{row['delta_f1']:+.4f}"
        d_rec = f"{row['delta_stress_recall']:+.4f}"

        line = f"{s:<9} {g_hr:<13} {g_eda:<13} {g_temp:<13} {d_f1:<11} {d_rec:<16}"
        lines.append(line)

    lines.extend(["\n", "Spearman Rank Correlations vs delta_f1 (Exploratory / Descriptive):"])
    corr_header = f"{'Feature':<16} {'Variant':<20} {'Spearman r':<12} {'p-value':<10} {'n':<4}"
    lines.extend([corr_header, "-" * len(corr_header)])

    for c in correlations:
        feat = str(c["feature"])
        var = str(c["variant"])
        r_str = f"{c['spearman_r']:+.4f}" if not np.isnan(c["spearman_r"]) else "NaN"
        p_str = f"{c['p_value']:.4f}" if not np.isnan(c["p_value"]) else "NaN"
        n_str = str(c["n"])
        lines.append(f"{feat:<16} {var:<20} {r_str:<12} {p_str:<10} {n_str:<4}")

    lines.extend(["", banner, DISCLAIMER_NOTE, banner])
    return "\n".join(lines)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Exploratory post hoc calibration quality analysis (Phase 4)."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features_phase2.csv"),
        help="Path to windowed features CSV.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("data/processed/personalization_per_subject.csv"),
        help="Path to per-subject personalization results CSV.",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parsed = parse_args(args)
    if not parsed.features.exists():
        print(f"Error: Features file not found: {parsed.features}")
        return 1
    if not parsed.results.exists():
        print(f"Error: Results file not found: {parsed.results}")
        return 1

    features_df = pd.read_csv(parsed.features)
    results_df = pd.read_csv(parsed.results)

    diag_df, correlations = analyze_calibration_quality(
        features_df=features_df,
        results_df=results_df,
    )
    report = format_calibration_report(diag_df, correlations)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
