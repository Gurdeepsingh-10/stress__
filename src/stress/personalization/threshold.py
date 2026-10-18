"""Calibrated probability thresholding (Method B2)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def calibrated_threshold(
    p_cal: Sequence[float] | np.ndarray | pd.Series,
) -> float:
    """Compute personalized decision threshold theta = max(0.5, max(p_cal)).

    Fixed rule with no tunable parameters:
    - If p_cal is empty (k=0), returns default threshold 0.5.
    - If all calibration scores are <= 0.5, returns default threshold 0.5.
    - If max(p_cal) > 0.5, returns max(p_cal), guaranteeing that every calibration
      baseline window itself is classified as non-stress under strict inequality.
    - Threshold is strictly bounded below by 0.5.

    Args:
        p_cal: Predicted probabilities of stress P(stress) on calibration baseline windows.

    Returns:
        Calibrated threshold float.
    """
    if len(p_cal) == 0:
        return 0.5

    p_arr = np.asarray(p_cal, dtype=float)
    if p_arr.size == 0 or np.all(np.isnan(p_arr)):
        return 0.5

    max_p = float(np.nanmax(p_arr))
    return max(0.5, max_p)


def predict_with_threshold(
    p: Sequence[float] | np.ndarray | pd.Series,
    theta: float,
) -> np.ndarray:
    """Predict binary stress class using calibrated threshold.

    Decision rule:
        Predict stress (1) iff p > theta (strictly greater).
        Otherwise predict not-stress (0).

    Args:
        p: Predicted probabilities of stress P(stress) on evaluation windows.
        theta: Decision threshold.

    Returns:
        1D integer numpy array of binary predictions (0 or 1).
    """
    p_arr = np.asarray(p, dtype=float)
    return (p_arr > theta).astype(int)
