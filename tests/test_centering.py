"""Unit tests for subject baseline centering (Method A).

Synthetic data only: no real WESAD data is loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stress.personalization.centering import (
    center_features,
    compute_calibration_mean,
)
from stress.personalization.splitting import (
    get_calibration_windows,
    split_calibration_and_test,
)

FEATURE_COLS = ("hrv_mean_hr", "eda_mean", "acc_mag")


def create_synthetic_subject(
    subject_id: int = 2,
    n_baseline: int = 20,
    n_stress: int = 15,
) -> pd.DataFrame:
    """Create a synthetic subject dataframe with given class counts."""
    n_total = n_baseline + n_stress
    starts = [float(i * 30) for i in range(n_total)]
    labels_3class = [1] * n_baseline + [2] * n_stress
    labels_binary = [0] * n_baseline + [1] * n_stress

    # Specific distinctive feature values
    hr = [72.0 + (i % 5) for i in range(n_baseline)] + [88.0 + (i % 5) for i in range(n_stress)]
    eda = [1.5 + 0.1 * (i % 4) for i in range(n_baseline)] + [
        4.0 + 0.2 * (i % 4) for i in range(n_stress)
    ]
    acc = [0.95 + 0.05 * (i % 3) for i in range(n_total)]

    return pd.DataFrame(
        {
            "subject": subject_id,
            "window_start_s": starts,
            "label_3class": labels_3class,
            "label_binary": labels_binary,
            "hrv_mean_hr": hr,
            "eda_mean": eda,
            "acc_mag": acc,
        }
    )


def test_centered_calibration_windows_have_zero_mean() -> None:
    """(a) The mean of the centered calibration windows is 0 for every feature."""
    df = create_synthetic_subject()
    split = split_calibration_and_test(df, k=5, k_max=10)

    cal_mean = compute_calibration_mean(split.cal_df, FEATURE_COLS)
    centered_cal = center_features(split.cal_df, cal_mean, FEATURE_COLS)

    centered_cal_mean = centered_cal[list(FEATURE_COLS)].mean(axis=0)
    for col in FEATURE_COLS:
        assert abs(centered_cal_mean[col]) < 1e-9


def test_changing_non_calibration_windows_does_not_change_cal_mean() -> None:
    """(b) Changing values of non-calibration windows does NOT change cal_mean (leakage test)."""
    df1 = create_synthetic_subject()
    split1 = split_calibration_and_test(df1, k=5, k_max=10)
    cal_mean1 = compute_calibration_mean(split1.cal_df, FEATURE_COLS)

    # Modify non-calibration windows drastically
    df2 = df1.copy()
    non_cal_mask = ~df2.index.isin(split1.cal_df.index)
    df2.loc[non_cal_mask, "hrv_mean_hr"] = 999.0
    df2.loc[non_cal_mask, "eda_mean"] = -500.0

    split2 = split_calibration_and_test(df2, k=5, k_max=10)
    cal_mean2 = compute_calibration_mean(split2.cal_df, FEATURE_COLS)

    # cal_mean must remain identical
    pd.testing.assert_series_equal(cal_mean1, cal_mean2)


def test_k_zero_returns_identical_values() -> None:
    """(c) k=0 returns identical values."""
    df = create_synthetic_subject()
    split = split_calibration_and_test(df, k=0, k_max=10)

    cal_mean = compute_calibration_mean(split.cal_df, FEATURE_COLS)
    assert len(cal_mean) == 0

    centered_test = center_features(split.test_df, cal_mean, FEATURE_COLS)
    pd.testing.assert_frame_equal(split.test_df, centered_test)


def test_non_feature_columns_row_order_and_index_unchanged() -> None:
    """(d) Non-feature columns, row order and index are unchanged."""
    df = create_synthetic_subject()
    # Scramble index and order to ensure robustness
    df.index = [f"row_{i}" for i in range(len(df))]
    df = df.sample(frac=1.0, random_state=42)

    split = split_calibration_and_test(df, k=5, k_max=10)
    cal_mean = compute_calibration_mean(split.cal_df, FEATURE_COLS)

    centered_df = center_features(df, cal_mean, FEATURE_COLS)

    # Index and order match exactly
    assert list(centered_df.index) == list(df.index)
    # Non-feature columns untouched
    for non_feat in ("subject", "window_start_s", "label_3class", "label_binary"):
        pd.testing.assert_series_equal(centered_df[non_feat], df[non_feat])


def test_training_subject_centering_uses_true_baseline_windows() -> None:
    """(e) Training subject whose first windows are amusement is centered by true baseline."""
    # Subject starts with 4 amusement windows, then 10 baseline windows, then 10 stress windows
    n_amuse = 4
    n_base = 10
    n_stress = 10
    total = n_amuse + n_base + n_stress

    starts = [float(i * 30) for i in range(total)]
    labels_3class = [3] * n_amuse + [1] * n_base + [2] * n_stress
    labels_binary = [0] * n_amuse + [0] * n_base + [1] * n_stress

    # Amusement HR = 120, Baseline HR = 70, Stress HR = 90
    hr = [120.0] * n_amuse + [70.0] * n_base + [90.0] * n_stress

    train_subject_df = pd.DataFrame(
        {
            "subject": 10,
            "window_start_s": starts,
            "label_3class": labels_3class,
            "label_binary": labels_binary,
            "hrv_mean_hr": hr,
        }
    )

    # In Method A, training subject is centered with its own first k baseline windows
    cal_windows = get_calibration_windows(
        train_subject_df, k=5, target_col="label_3class", baseline_val=1
    )
    # Must have chosen true baseline (HR = 70), not amusement (HR = 120)
    assert len(cal_windows) == 5
    assert (cal_windows["label_3class"] == 1).all()

    cal_mean = compute_calibration_mean(cal_windows, ["hrv_mean_hr"])
    assert cal_mean["hrv_mean_hr"] == 70.0

    centered_train = center_features(train_subject_df, cal_mean, ["hrv_mean_hr"])
    # Centered baseline should be 70 - 70 = 0.0
    baseline_centered_hr = centered_train.loc[
        centered_train["label_3class"] == 1, "hrv_mean_hr"
    ].to_numpy()
    np.testing.assert_allclose(baseline_centered_hr, 0.0)


def test_nan_calibration_mean_sets_feature_to_nan_for_imputation() -> None:
    """If a feature has NaN calibration mean, set column to NaN for median imputation."""
    df = create_synthetic_subject()
    # Set one feature to all NaN in calibration windows
    split = split_calibration_and_test(df, k=5, k_max=10)
    cal_df_with_nan = split.cal_df.copy()
    cal_df_with_nan["eda_mean"] = np.nan

    cal_mean = compute_calibration_mean(cal_df_with_nan, FEATURE_COLS)
    assert np.isnan(cal_mean["eda_mean"])

    centered_test = center_features(split.test_df, cal_mean, FEATURE_COLS)
    # eda_mean in centered_test must be explicitly NaN so SimpleImputer handles it
    assert centered_test["eda_mean"].isna().all()
    # Other features centered normally
    assert not centered_test["hrv_mean_hr"].isna().any()


def test_centering_preserves_within_subject_differences_and_variance() -> None:
    """Centering must preserve within-subject differences and standard deviations.

    Centering shifts feature values by a constant (the calibration baseline mean);
    it never rescales them. Therefore:
    - Pairwise differences between any two windows are unchanged.
    - Standard deviation of each feature across windows is unchanged.
    """
    df = create_synthetic_subject()
    split = split_calibration_and_test(df, k=5, k_max=10)
    cal_mean = compute_calibration_mean(split.cal_df, FEATURE_COLS)
    centered_df = center_features(df, cal_mean, FEATURE_COLS)

    for col in FEATURE_COLS:
        # Standard deviation must be bitwise/numerically unchanged
        orig_std = float(df[col].std())
        cent_std = float(centered_df[col].std())
        assert cent_std > 0.0, f"Expected non-zero std for {col}"
        np.testing.assert_allclose(cent_std, orig_std, rtol=1e-12)

        # Pairwise differences between consecutive windows must be exact
        orig_diffs = np.diff(df[col].to_numpy())
        cent_diffs = np.diff(centered_df[col].to_numpy())
        np.testing.assert_allclose(cent_diffs, orig_diffs, atol=1e-12)

        # Arbitrary pairwise differences across windows (e.g. window 0 vs window 15)
        orig_delta = float(df[col].iloc[15] - df[col].iloc[0])
        cent_delta = float(centered_df[col].iloc[15] - centered_df[col].iloc[0])
        np.testing.assert_allclose(cent_delta, orig_delta, atol=1e-12)
