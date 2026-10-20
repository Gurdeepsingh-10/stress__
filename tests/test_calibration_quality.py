"""Unit tests for exploratory post hoc calibration quality diagnostic.

All tests use synthetic data only with fixed seeds. No real data is loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.calibration_quality import (
    TARGET_FEATURES,
    analyze_calibration_quality,
    compute_feature_gap_z,
    compute_subject_gaps,
    extract_subject_baseline_windows,
    format_calibration_report,
    parse_args,
)


def make_synthetic_subject_windows(
    n_baseline: int = 35,
    n_stress: int = 25,
    cal_elevation: float = 0.0,
    later_std: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic single-subject window DataFrame."""
    rng = np.random.RandomState(seed)
    n_total = n_baseline + n_stress

    rows = []
    for w in range(n_total):
        is_stress = w >= n_baseline
        label_3class = 2 if is_stress else 1
        label_binary = 1 if is_stress else 0

        # Calibration windows are the first 10 baseline windows (w < 10)
        is_cal = w < 10
        elevation = cal_elevation if is_cal else 0.0
        noise = rng.normal(0, later_std) if later_std > 0 else 0.0

        row = {
            "subject": 1,
            "window_start_s": float(w * 30),
            "label_3class": label_3class,
            "label_binary": label_binary,
            "hrv_mean_hr": 70.0 + elevation + noise,
            "eda_mean": 2.0 + elevation + noise,
            "temp_mean": 32.0 + elevation + noise,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def test_gap_z_is_zero_when_cal_and_later_have_same_mean() -> None:
    """1. gap_z is zero when cal and later windows have the exact same mean."""
    cal_df = pd.DataFrame({"hrv_mean_hr": [75.0, 75.0, 75.0]})
    later_df = pd.DataFrame({"hrv_mean_hr": [70.0, 75.0, 80.0]})

    gap_z = compute_feature_gap_z(cal_df, later_df, "hrv_mean_hr")
    assert gap_z == pytest.approx(0.0, abs=1e-12)


def test_gap_z_sign_positive_when_cal_windows_elevated() -> None:
    """2. gap_z sign is positive when cal windows are elevated relative to later.

    cal = [5.0, 5.5, 6.0] (mean 5.5, std 0.5)
    later = [2.0, 3.0, 4.0] (mean 3.0, std 1.0)
    gap_z = (5.5 - 3.0) / 1.0 = 2.5.
    Dividing by cal std (0.5) would give 5.0 and must fail.
    """
    cal_df = pd.DataFrame({"eda_mean": [5.0, 5.5, 6.0]})
    later_df = pd.DataFrame({"eda_mean": [2.0, 3.0, 4.0]})

    gap_z = compute_feature_gap_z(cal_df, later_df, "eda_mean")
    assert gap_z > 0.0
    assert gap_z == pytest.approx(2.5, abs=1e-12)

    # Inverted: cal depressed relative to later -> negative gap_z
    gap_z_inv = compute_feature_gap_z(later_df, cal_df, "eda_mean")
    assert gap_z_inv < 0.0
    assert gap_z_inv == pytest.approx(-5.0, abs=1e-12)


def test_std_zero_gives_nan() -> None:
    """3. std = 0 in later baseline windows guards division by returning NaN."""
    cal_df = pd.DataFrame({"temp_mean": [34.0, 35.0]})
    # Perfectly constant later windows -> std == 0
    later_df = pd.DataFrame({"temp_mean": [32.0, 32.0, 32.0]})

    gap_z = compute_feature_gap_z(cal_df, later_df, "temp_mean")
    assert np.isnan(gap_z)

    # Empty or single-row later window -> NaN
    single_later_df = pd.DataFrame({"temp_mean": [32.0]})
    assert np.isnan(compute_feature_gap_z(cal_df, single_later_df, "temp_mean"))


def test_later_windows_exclude_every_calibration_window_and_buffer() -> None:
    """4. Later windows strictly exclude every calibration window and overlapping buffer.

    With 60s windows, 30s step:
    cal windows: windows 0..9 (starts: 0s, 30s, ..., 270s; end of cal window 9: 270+60 = 330s).
    Buffer windows: windows 10..10+k_max-1.
    T_eval starts at window k_max = 10 (start: 300s, but window 9 spans 270..330s,
    so split_calibration_and_test drops buffer windows so zero overlap exists).
    """
    df = make_synthetic_subject_windows(n_baseline=35, n_stress=25, seed=123)
    cal_df, later_df = extract_subject_baseline_windows(df, k=10, k_max=10)

    assert len(cal_df) == 10
    assert (cal_df["label_3class"] == 1).all()

    cal_indices = set(cal_df.index)
    later_indices = set(later_df.index)

    # Disjoint index sets: later baseline contains zero calibration rows
    assert cal_indices.isdisjoint(later_indices)

    # Check temporal ordering and zero overlap in time
    last_cal_end_s = float(cal_df["window_start_s"].iloc[-1] + 60.0)
    first_later_start_s = float(later_df["window_start_s"].iloc[0])
    assert first_later_start_s >= last_cal_end_s, (
        f"Temporal overlap detected: cal ends at {last_cal_end_s}s "
        f"but later baseline starts at {first_later_start_s}s"
    )

    # All later windows are strictly true baseline
    assert (later_df["label_3class"] == 1).all()


def test_compute_subject_gaps_returns_all_target_features() -> None:
    """compute_subject_gaps returns valid dictionary with all target features."""
    df = make_synthetic_subject_windows(n_baseline=30, n_stress=20, cal_elevation=5.0, seed=7)
    gaps = compute_subject_gaps(df, features=TARGET_FEATURES, k=10, k_max=10)

    for feat in TARGET_FEATURES:
        assert feat in gaps
        # Since cal was elevated by +5.0, gap_z must be positive
        assert gaps[feat] > 0.0


def test_analyze_and_format_calibration_quality_end_to_end() -> None:
    """End-to-end integration test on synthetic multi-subject data."""
    # Build synthetic features for 4 subjects
    feature_rows = []
    for s in range(1, 5):
        s_df = make_synthetic_subject_windows(
            n_baseline=30,
            n_stress=20,
            cal_elevation=float(s * 2.0),
            seed=s * 10,
        )
        s_df["subject"] = s
        feature_rows.append(s_df)
    features_df = pd.concat(feature_rows, ignore_index=True)

    # Build synthetic results DataFrame with Reference (k=0) and Method A (k=10)
    results_rows = []
    for s in range(1, 5):
        results_rows.append(
            {
                "subject": s,
                "method": "reference",
                "model": "logistic",
                "group": "all",
                "k": 0,
                "macro_f1": 0.80,
                "recall_stress": 0.85,
                "recall_not_stress": 0.85,
                "roc_auc": 0.90,
                "n_eval": 40,
                "n_eval_stress": 20,
                "theta": 0.5,
            }
        )
        # Synthetic Method A: subject 1 got worse (-0.10), subject 4 improved (+0.05)
        results_rows.append(
            {
                "subject": s,
                "method": "method_a",
                "model": "logistic",
                "group": "all",
                "k": 10,
                "macro_f1": 0.80 + (-0.10 if s == 1 else 0.05),
                "recall_stress": 0.85 + (-0.10 if s == 1 else 0.05),
                "recall_not_stress": 0.85,
                "roc_auc": 0.90,
                "n_eval": 40,
                "n_eval_stress": 20,
                "theta": 0.5,
            }
        )
    results_df = pd.DataFrame(results_rows)

    diag_df, correlations = analyze_calibration_quality(
        features_df,
        results_df,
        features=TARGET_FEATURES,
        model="logistic",
        group="all",
        k=10,
    )

    assert len(diag_df) == 4
    assert "delta_f1" in diag_df.columns
    assert "delta_stress_recall" in diag_df.columns
    for feat in TARGET_FEATURES:
        assert f"gap_z_{feat}" in diag_df.columns

    # 3 features x 2 variants (signed, abs) = 6 correlation records
    assert len(correlations) == 6
    for c in correlations:
        assert "feature" in c
        assert "spearman_r" in c
        assert "p_value" in c
        assert c["n"] == 4

    # Format report string contains disclaimer banner
    report = format_calibration_report(diag_df, correlations)
    assert "EXPLORATORY POST HOC ANALYSIS" in report
    assert "descriptive only" in report.lower()
    assert "gap_z(hr)" in report


def test_spearman_perfect_negative_correlation_when_delta_f1_decreases_with_elevation() -> None:
    """Signed Spearman r is exactly -1.0 when delta_f1 strictly decreases with elevation."""
    n_subjects = 5
    feature_rows = []
    # Each subject s has strictly increasing calibration elevation: s * 10.0
    for s in range(1, n_subjects + 1):
        s_df = make_synthetic_subject_windows(
            n_baseline=30,
            n_stress=20,
            cal_elevation=float(s * 10.0),
            later_std=1.0,
            seed=s * 10,
        )
        s_df["subject"] = s
        feature_rows.append(s_df)
    features_df = pd.concat(feature_rows, ignore_index=True)

    # delta_f1 is a strictly decreasing function of s (and therefore of calibration elevation)
    results_rows = []
    for s in range(1, n_subjects + 1):
        results_rows.append(
            {
                "subject": s,
                "method": "reference",
                "model": "logistic",
                "group": "all",
                "k": 0,
                "macro_f1": 0.80,
                "recall_stress": 0.80,
                "recall_not_stress": 0.80,
                "roc_auc": 0.90,
                "n_eval": 40,
                "n_eval_stress": 20,
                "theta": 0.5,
            }
        )
        results_rows.append(
            {
                "subject": s,
                "method": "method_a",
                "model": "logistic",
                "group": "all",
                "k": 10,
                "macro_f1": 0.80 - float(s) * 0.05,  # strictly decreasing with s
                "recall_stress": 0.80 - float(s) * 0.05,
                "recall_not_stress": 0.80,
                "roc_auc": 0.90,
                "n_eval": 40,
                "n_eval_stress": 20,
                "theta": 0.5,
            }
        )
    results_df = pd.DataFrame(results_rows)

    diag_df, correlations = analyze_calibration_quality(
        features_df,
        results_df,
        features=TARGET_FEATURES,
        model="logistic",
        group="all",
        k=10,
    )

    signed_corrs = [c for c in correlations if c["variant"] == "gap_z (signed)"]
    assert len(signed_corrs) == len(TARGET_FEATURES)
    for c in signed_corrs:
        assert c["spearman_r"] == pytest.approx(-1.0, abs=1e-12)
        assert c["n"] == n_subjects


def test_cli_parse_args() -> None:
    """CLI argument parser handles defaults and custom paths."""
    args = parse_args([])
    assert args.features.name == "features_phase2.csv"
    assert args.results.name == "personalization_per_subject.csv"

    custom = parse_args(["--features", "f.csv", "--results", "r.csv"])
    assert custom.features.name == "f.csv"
    assert custom.results.name == "r.csv"
