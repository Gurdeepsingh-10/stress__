"""Leakage and invariant tests for personalization splitting.

Synthetic data only: no real WESAD data is loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stress.personalization.splitting import (
    get_calibration_windows,
    split_calibration_and_test,
)


@pytest.fixture
def synthetic_subject_windows() -> pd.DataFrame:
    """Create a synthetic subject window DataFrame with 50% overlap.

    Layout:
    - 60-second windows with 30-second step (50% overlap).
    - Window starts: 0, 30, 60, ..., 2970 s (100 windows).
    - First 40 windows are baseline (label_binary = 0, label_3class = 1).
    - Next 30 windows are stress (label_binary = 1, label_3class = 2).
    - Final 30 windows are baseline (label_binary = 0, label_3class = 1).
    """
    n_windows = 100
    step_s = 30.0
    starts = np.arange(n_windows) * step_s

    # Labels: 40 baseline, 30 stress, 30 baseline
    labels_3class = np.array([1] * 40 + [2] * 30 + [1] * 30)
    labels_binary = (labels_3class == 2).astype(int)

    # Synthetic features
    np.random.seed(42)
    hrv_mean_hr = np.where(labels_binary == 1, 85.0, 70.0) + np.random.randn(n_windows) * 2.0
    eda_mean = np.where(labels_binary == 1, 4.0, 1.0) + np.random.randn(n_windows) * 0.2

    df = pd.DataFrame(
        {
            "subject": 2,
            "window_start_s": starts,
            "label_3class": labels_3class,
            "label_binary": labels_binary,
            "hrv_mean_hr": hrv_mean_hr,
            "eda_mean": eda_mean,
        }
    )
    return df


def test_calibration_labels_all_baseline(synthetic_subject_windows: pd.DataFrame) -> None:
    """Invariant: all calibration windows must have baseline label."""
    for k in (1, 3, 5, 10):
        split = split_calibration_and_test(synthetic_subject_windows, k=k, k_max=10)
        assert len(split.cal_df) == k
        assert (split.cal_df["label_binary"] == 0).all()
        assert (split.cal_df["label_3class"] == 1).all()


def test_calibration_temporal_ordering(synthetic_subject_windows: pd.DataFrame) -> None:
    """Invariant: calibration windows are strictly the earliest baseline windows in time."""
    # Shuffled dataframe to verify sorting by time works regardless of row order
    shuffled_df = synthetic_subject_windows.sample(frac=1.0, random_state=123).reset_index(
        drop=True
    )
    cal_df = get_calibration_windows(shuffled_df, k=10)

    assert len(cal_df) == 10
    # Strictly monotonically increasing
    assert cal_df["window_start_s"].is_monotonic_increasing
    # Must match the first 10 baseline starts in synthetic dataset (0, 30, ..., 270)
    expected_starts = [i * 30.0 for i in range(10)]
    np.testing.assert_allclose(cal_df["window_start_s"].to_numpy(), expected_starts)


def test_zero_overlap_invariant(synthetic_subject_windows: pd.DataFrame) -> None:
    """Invariant: no test window overlaps any window in the master calibration set (k_max).

    With window_seconds=60.0, overlap is nonzero iff |t_test - t_cal| < 60.0.
    Thus, for every pair (c, w), |t_w - t_c| >= 60.0 must hold.
    """
    k_max = 10
    window_seconds = 60.0

    for k in (0, 3, 5, 10):
        split = split_calibration_and_test(
            synthetic_subject_windows, k=k, k_max=k_max, window_seconds=window_seconds
        )
        test_starts = split.test_df["window_start_s"].to_numpy()

        # Calibration master windows
        master_cal = get_calibration_windows(synthetic_subject_windows, k=k_max)
        cal_starts = master_cal["window_start_s"].to_numpy()

        # Check every test window against every master calibration window
        for c in cal_starts:
            diffs = np.abs(test_starts - c)
            # Must be strictly at or greater than window_seconds (no overlapping time)
            assert np.all(diffs >= window_seconds - 1e-6), (
                f"Found test window overlapping calibration window at t={c}: "
                f"violating diffs = {diffs[diffs < window_seconds]}"
            )

        # Check indices intersection is strictly empty
        assert set(master_cal.index).isdisjoint(set(split.test_df.index))
        if k > 0:
            assert set(split.cal_df.index).isdisjoint(set(split.test_df.index))


def test_identical_eval_set_across_k(synthetic_subject_windows: pd.DataFrame) -> None:
    """Invariant: test evaluation set is bitwise identical for all k in {0, 3, 5, 10}."""
    k_max = 10
    splits = {
        k: split_calibration_and_test(synthetic_subject_windows, k=k, k_max=k_max)
        for k in (0, 3, 5, 10)
    }

    ref_test_df = splits[0].test_df
    for k in (3, 5, 10):
        test_df_k = splits[k].test_df
        pd.testing.assert_frame_equal(ref_test_df, test_df_k)


def test_dropped_buffer_count_matches_theory(synthetic_subject_windows: pd.DataFrame) -> None:
    """Verify exactly 11 windows are excluded for k_max=10 consecutive windows at start.

    10 calibration windows (t=0..270) + 1 overlapping buffer window (t=300) = 11 dropped windows.
    100 total windows -> 89 test windows.
    """
    split = split_calibration_and_test(synthetic_subject_windows, k=10, k_max=10)
    assert len(synthetic_subject_windows) == 100
    assert len(split.cal_df) == 10
    # 10 calibration windows + 1 buffer window = 11 dropped windows
    assert len(split.test_df) == 89

    # Earliest test window must start at 330 s
    assert split.test_df["window_start_s"].min() == 330.0


def test_k_bounds_validation(synthetic_subject_windows: pd.DataFrame) -> None:
    """Validate argument bounds on k and k_max."""
    with pytest.raises(ValueError, match="k must be non-negative"):
        split_calibration_and_test(synthetic_subject_windows, k=-1, k_max=10)

    with pytest.raises(ValueError, match=r"k \(11\) cannot exceed k_max \(10\)"):
        split_calibration_and_test(synthetic_subject_windows, k=11, k_max=10)


def test_k_zero_returns_empty_cal_and_full_test_subject(
    synthetic_subject_windows: pd.DataFrame,
) -> None:
    """For k=0 with k_max=10, cal_df is empty, but test_df still respects k_max exclusion."""
    split = split_calibration_and_test(synthetic_subject_windows, k=0, k_max=10)
    assert len(split.cal_df) == 0
    assert list(split.cal_df.columns) == list(synthetic_subject_windows.columns)
    assert len(split.test_df) == 89


def test_calibration_never_selects_amusement_even_if_earlier() -> None:
    """True baseline calibration must use label_3class == 1, NEVER amusement (3).

    Even if amusement windows (label_3class = 3, label_binary = 0) occur earlier
    in time than any baseline window, calibration must strictly pick true baseline.
    """
    # 5 amusement windows at t=0..120, then 10 baseline windows at t=150..420
    starts = [float(i * 30) for i in range(15)]
    labels_3class = [3] * 5 + [1] * 10
    labels_binary = [0] * 15  # Both baseline and amusement map to binary 0!

    df = pd.DataFrame(
        {
            "subject": 5,
            "window_start_s": starts,
            "label_3class": labels_3class,
            "label_binary": labels_binary,
            "hrv_mean_hr": [70.0] * 15,
        }
    )

    split = split_calibration_and_test(df, k=5, k_max=10)
    # Cal windows must all be true baseline (label_3class == 1), not amusement (label_3class == 3)
    assert len(split.cal_df) == 5
    assert (split.cal_df["label_3class"] == 1).all()
    # First true baseline starts at 150.0 s
    assert split.cal_df["window_start_s"].min() == 150.0
    # No amusement window in calibration
    assert 3 not in split.cal_df["label_3class"].values
