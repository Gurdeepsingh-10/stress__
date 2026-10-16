"""Personalization splitting: extract calibration windows and leak-free test sets."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd


class CalibrationSplit(NamedTuple):
    """Result of splitting a subject's windows into calibration and test sets."""

    cal_df: pd.DataFrame
    test_df: pd.DataFrame
    k: int
    k_max: int


def get_calibration_windows(
    df: pd.DataFrame,
    k: int,
    *,
    target_col: str = "label_3class",
    baseline_val: int = 1,
    time_col: str = "window_start_s",
) -> pd.DataFrame:
    """Extract the first k baseline calibration windows for a subject, ordered by time.

    Args:
        df: Subject DataFrame containing window features, timestamps, and labels.
        k: Number of calibration windows to extract. If k <= 0, returns an empty DataFrame.
        target_col: Column indicating class label.
        baseline_val: Value corresponding to baseline (resting/calm) state.
        time_col: Column indicating window start time in seconds.

    Returns:
        DataFrame containing the first k baseline windows, sorted by time_col.
    """
    if k <= 0:
        return df.iloc[0:0].copy()

    baseline_mask = df[target_col] == baseline_val
    baseline_df = df[baseline_mask].sort_values(by=time_col)
    return baseline_df.iloc[:k].copy()


def split_calibration_and_test(
    df: pd.DataFrame,
    k: int,
    k_max: int = 10,
    *,
    window_seconds: float = 60.0,
    target_col: str = "label_3class",
    baseline_val: int = 1,
    time_col: str = "window_start_s",
) -> CalibrationSplit:
    """Split a subject's windows into calibration set and an overlap-free test set.

    Protocol rules:
    1. Calibration windows are the first k baseline windows, ordered chronologically.
    2. The master calibration set consists of the first k_max baseline windows.
    3. To prevent leakage from 50% window overlap, any window in df whose time interval
       strictly overlaps with any window in the master calibration set is excluded from the
       test set: i.e., excluded if |t_w - t_c| < window_seconds for any c in master calibration.
    4. Test set is identical for all k in [0, k_max] because exclusion is determined by k_max.
    5. Calibration labels are guaranteed to be baseline_val only.

    Args:
        df: Subject DataFrame containing window features, timestamps, and labels.
        k: Calibration window count for the current condition (0 <= k <= k_max).
        k_max: Maximum calibration window count defining the fixed excluded buffer (default: 10).
        window_seconds: Length of a window in seconds (default: 60.0).
        target_col: Label column name (default: 'label_binary').
        baseline_val: Value corresponding to baseline class (default: 0).
        time_col: Timestamp column name for window start (default: 'window_start_s').

    Returns:
        CalibrationSplit namedtuple with fields (cal_df, test_df, k, k_max).
    """
    if k < 0:
        raise ValueError(f"k must be non-negative, got {k}")
    if k > k_max:
        raise ValueError(f"k ({k}) cannot exceed k_max ({k_max})")

    # 1. Master calibration set (up to k_max)
    master_cal = get_calibration_windows(
        df,
        k=k_max,
        target_col=target_col,
        baseline_val=baseline_val,
        time_col=time_col,
    )

    # 2. Current calibration set (first k windows of master_cal)
    cal_df = master_cal.iloc[:k].copy() if k > 0 else df.iloc[0:0].copy()

    # 3. Buffer exclusion against master_cal
    if len(master_cal) == 0:
        test_df = df.copy()
    else:
        cal_starts = master_cal[time_col].to_numpy()
        all_starts = df[time_col].to_numpy()

        # A test window at t_w overlaps with a cal window at t_c iff |t_w - t_c| < window_seconds
        # Using a small numerical tolerance for floating point comparisons
        eps = 1e-6
        diffs = np.abs(all_starts[:, np.newaxis] - cal_starts[np.newaxis, :])
        overlaps_any_cal = np.any(diffs < (window_seconds - eps), axis=1)

        test_df = df[~overlaps_any_cal].copy()

    return CalibrationSplit(cal_df=cal_df, test_df=test_df, k=k, k_max=k_max)
