"""Unit tests for calibrated probability thresholding (Method B2).

Synthetic data only: no real WESAD data is loaded.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from stress.personalization.threshold import (
    calibrated_threshold,
    predict_with_threshold,
)


def test_empty_returns_half() -> None:
    """Empty calibration scores array/list returns 0.5."""
    assert calibrated_threshold([]) == 0.5
    assert calibrated_threshold(np.array([])) == 0.5


def test_all_calibration_scores_below_half_returns_half() -> None:
    """All calibration scores below 0.5 -> 0.5."""
    p_cal = [0.1, 0.25, 0.49, 0.05]
    assert calibrated_threshold(p_cal) == 0.5


def test_max_above_half_returns_that_max() -> None:
    """Max calibration score above 0.5 -> returns that exact max."""
    p_cal = [0.1, 0.45, 0.73, 0.62]
    assert calibrated_threshold(p_cal) == 0.73


def test_threshold_is_never_below_half() -> None:
    """Threshold is strictly bounded below by 0.5 across diverse inputs."""
    assert calibrated_threshold([0.0, 0.001]) >= 0.5
    assert calibrated_threshold([-1.0]) >= 0.5
    assert calibrated_threshold([0.5]) >= 0.5


def test_theta_half_identical_to_plain_logistic_regression_predict() -> None:
    """With theta=0.5, predictions are identical to plain model's predict method.

    Tests with a fitted sklearn LogisticRegression on synthetic data.
    """
    np.random.seed(42)
    # 50 samples, 3 features
    x_train = np.random.randn(50, 3)
    y_train = (x_train[:, 0] + x_train[:, 1] > 0).astype(int)

    model = LogisticRegression(random_state=42)
    model.fit(x_train, y_train)

    x_test = np.random.randn(30, 3)
    p_test = model.predict_proba(x_test)[:, 1]

    expected_preds = model.predict(x_test)
    threshold_preds = predict_with_threshold(p_test, theta=0.5)

    np.testing.assert_array_equal(threshold_preds, expected_preds)


def test_every_calibration_window_predicted_not_stress() -> None:
    """Every calibration window is predicted not-stress (0) when theta = max(p_cal).

    Because the decision rule is strictly p > theta:
    for any p in p_cal, p <= max(p_cal) = theta, so p > theta is FALSE -> prediction is 0.
    """
    p_cal = np.array([0.15, 0.42, 0.68, 0.81, 0.33])
    theta = calibrated_threshold(p_cal)
    assert theta == pytest.approx(0.81)

    cal_preds = predict_with_threshold(p_cal, theta)
    # Every calibration baseline window must be predicted 0 (not-stress)
    np.testing.assert_array_equal(cal_preds, np.zeros_like(cal_preds))


def test_strict_inequality_at_boundary() -> None:
    """A score just above theta is predicted stress; a score equal to theta is not."""
    theta = 0.65
    scores = np.array([theta - 1e-9, theta, theta + 1e-9])
    preds = predict_with_threshold(scores, theta=theta)
    np.testing.assert_array_equal(preds, np.array([0, 0, 1]))
