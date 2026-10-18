"""Subject baseline centering (Method A)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def compute_calibration_mean(
    cal_df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.Series:
    """Compute per-feature NaN-aware mean over calibration windows only.

    The function takes only the calibration DataFrame, ensuring no information from
    test or non-calibration windows can leak into the calibration mean.

    Args:
        cal_df: DataFrame containing only calibration windows for one subject.
        feature_cols: List or tuple of feature column names to average.

    Returns:
        pd.Series indexed by feature column names with the NaN-aware mean per feature.
        If cal_df is empty, returns an empty Series.
    """
    if len(cal_df) == 0:
        return pd.Series(dtype=float, index=pd.Index([], dtype=str))

    present_cols = [col for col in feature_cols if col in cal_df.columns]
    return cal_df[present_cols].mean(axis=0)


def center_features(
    df: pd.DataFrame,
    cal_mean: pd.Series,
    feature_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Subtract calibration mean from feature columns, leaving non-features untouched.

    Centering only: no division by calibration standard deviation.

    Behavior:
    - For k=0 (empty cal_mean), returns an unchanged copy of df.
    - If a feature's calibration mean is NaN (e.g., all calibration windows had NaN for that
      feature), the feature column in the returned DataFrame is explicitly set to NaN for that
      subject so downstream SimpleImputer(strategy='median') handles it during fit/transform.
    - Non-feature columns, index, and row order are strictly preserved.

    Args:
        df: Subject DataFrame containing features and metadata.
        cal_mean: Series containing per-feature calibration baseline means.
        feature_cols: Optional sequence of feature column names. If None, uses cal_mean.index.

    Returns:
        DataFrame with centered features and unchanged non-feature metadata.
    """
    out_df = df.copy()

    if len(cal_mean) == 0:
        return out_df

    cols_to_center = (
        [col for col in feature_cols if col in out_df.columns]
        if feature_cols is not None
        else [col for col in cal_mean.index if col in out_df.columns]
    )

    for col in cols_to_center:
        if col not in cal_mean.index:
            continue
        c_mean = cal_mean[col]
        if pd.isna(c_mean):
            # Documented: if calibration mean is NaN, set column to NaN for median imputation
            out_df[col] = np.nan
        else:
            out_df[col] = out_df[col] - float(c_mean)

    return out_df
