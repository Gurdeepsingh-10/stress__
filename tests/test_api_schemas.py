"""Unit tests for FastAPI Pydantic schemas (Stage 1).

Tests schema validation, ranges, rejection of nan/inf, rejection of unknown fields,
missing required fields, and acceptance of null HRV features.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from stress.api.schemas import (
    RESEARCH_DISCLAIMER,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)


def valid_payload_dict() -> dict[str, float | int | None]:
    """Return a valid dictionary representing typical physiological feature values."""
    return {
        "hrv_mean_hr": 75.0,
        "hrv_hr_std": 5.2,
        "hrv_sdnn": 45.0,
        "hrv_rmssd": 38.0,
        "hrv_pnn50": 0.15,
        "hrv_num_beats": 74,
        "eda_mean": 2.5,
        "eda_std": 0.4,
        "eda_min": 1.8,
        "eda_max": 3.4,
        "eda_slope": 0.002,
        "eda_num_peaks": 3,
        "temp_mean": 32.5,
        "temp_std": 0.15,
        "temp_slope": -0.001,
        "acc_magnitude_mean": 0.98,
        "acc_magnitude_std": 0.04,
    }


def test_valid_payload_passes() -> None:
    """Valid payload parses without errors."""
    data = valid_payload_dict()
    req = PredictRequest(**data)
    assert req.hrv_mean_hr == 75.0
    assert req.eda_mean == 2.5
    assert req.acc_magnitude_mean == 0.98


def test_null_hrv_features_allowed() -> None:
    """None / null is allowed for HRV interval features when peak detection is poor."""
    data = valid_payload_dict()
    for hrv_feat in ("hrv_mean_hr", "hrv_hr_std", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"):
        data[hrv_feat] = None

    req = PredictRequest(**data)
    assert req.hrv_mean_hr is None
    assert req.hrv_sdnn is None
    # Required non-interval beat count is preserved
    assert req.hrv_num_beats == 74


@pytest.mark.parametrize(
    "hrv_field",
    ["hrv_mean_hr", "hrv_hr_std", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"],
)
def test_omitting_hrv_interval_field_rejected(hrv_field: str) -> None:
    """Omitting a required HRV interval field raises ValidationError with 'Field required'."""
    data = valid_payload_dict()
    del data[hrv_field]
    with pytest.raises(ValidationError, match="Field required"):
        PredictRequest(**data)


@pytest.mark.parametrize(
    ("field", "bad_value", "match_str"),
    [
        ("hrv_mean_hr", 20.0, "greater_than_equal"),
        ("hrv_mean_hr", 250.0, "less_than_equal"),
        ("hrv_pnn50", 1.5, "less_than_equal"),
        ("hrv_pnn50", -0.1, "greater_than_equal"),
        ("eda_mean", -0.5, "greater_than_equal"),
        ("temp_mean", 10.0, "greater_than_equal"),
        ("temp_mean", 50.0, "less_than_equal"),
        ("acc_magnitude_mean", -1.0, "greater_than_equal"),
    ],
)
def test_physiological_range_validation(field: str, bad_value: float, match_str: str) -> None:
    """Physiological features outside valid bounds are rejected with ValidationError."""
    data = valid_payload_dict()
    data[field] = bad_value
    with pytest.raises(ValidationError, match=match_str):
        PredictRequest(**data)


def test_hrv_sdnn_rmssd_extended_ranges() -> None:
    """Observed high values from peak artifacts pass; values above 10000.0 are rejected."""
    # 6800.7 passes for hrv_rmssd
    data = valid_payload_dict()
    data["hrv_rmssd"] = 6800.7
    req = PredictRequest(**data)
    assert req.hrv_rmssd == 6800.7

    # 4771.3 passes for hrv_sdnn
    data = valid_payload_dict()
    data["hrv_sdnn"] = 4771.3
    req = PredictRequest(**data)
    assert req.hrv_sdnn == 4771.3

    # 10000.1 is rejected for hrv_rmssd
    data = valid_payload_dict()
    data["hrv_rmssd"] = 10000.1
    with pytest.raises(ValidationError, match="less_than_equal"):
        PredictRequest(**data)

    # 10000.1 is rejected for hrv_sdnn
    data = valid_payload_dict()
    data["hrv_sdnn"] = 10000.1
    with pytest.raises(ValidationError, match="less_than_equal"):
        PredictRequest(**data)


@pytest.mark.parametrize(
    "bad_float",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nan_and_inf_rejected(bad_float: float) -> None:
    """Explicit NaN and infinite float values are rejected across float fields."""
    data = valid_payload_dict()
    data["eda_mean"] = bad_float
    with pytest.raises(ValidationError):
        PredictRequest(**data)

    data = valid_payload_dict()
    data["temp_mean"] = bad_float
    with pytest.raises(ValidationError):
        PredictRequest(**data)


@pytest.mark.parametrize(
    "hrv_field",
    ["hrv_mean_hr", "hrv_hr_std", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"],
)
@pytest.mark.parametrize(
    "bad_float",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nan_and_inf_rejected_on_hrv_interval_fields(hrv_field: str, bad_float: float) -> None:
    """Explicit NaN and infinite values are rejected for all HRV interval fields.

    None is still accepted for the same fields (only explicit bad floats are rejected).
    """
    # bad float must be rejected
    data = valid_payload_dict()
    data[hrv_field] = bad_float
    with pytest.raises(ValidationError):
        PredictRequest(**data)

    # None must still be accepted
    data = valid_payload_dict()
    data[hrv_field] = None
    req = PredictRequest(**data)
    assert getattr(req, hrv_field) is None


@pytest.mark.parametrize(
    "bad_float",
    [float("nan"), float("inf"), float("-inf")],
)
def test_nan_and_inf_rejected_unbounded_slopes(bad_float: float) -> None:
    """NaN and infinite values are rejected for the unbounded slope fields."""
    data = valid_payload_dict()
    data["eda_slope"] = bad_float
    with pytest.raises(ValidationError):
        PredictRequest(**data)

    data = valid_payload_dict()
    data["temp_slope"] = bad_float
    with pytest.raises(ValidationError):
        PredictRequest(**data)


def test_extra_unknown_field_rejected() -> None:
    """Extra attributes not in the 17 features are strictly forbidden."""
    data = valid_payload_dict()
    data["unknown_extra_sensor"] = 12.34
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PredictRequest(**data)


def test_missing_required_field_rejected() -> None:
    """Missing a required feature raises a ValidationError."""
    data = valid_payload_dict()
    del data["temp_mean"]
    with pytest.raises(ValidationError, match="Field required"):
        PredictRequest(**data)

    data = valid_payload_dict()
    del data["eda_mean"]
    with pytest.raises(ValidationError, match="Field required"):
        PredictRequest(**data)


def test_predicted_label_literal() -> None:
    """predicted_label only accepts 0 or 1; any other value is rejected."""
    good_0 = PredictResponse(
        stress_probability=0.2,
        predicted_label=0,
        model_version="1.0.0",
    )
    assert good_0.predicted_label == 0

    good_1 = PredictResponse(
        stress_probability=0.8,
        predicted_label=1,
        model_version="1.0.0",
    )
    assert good_1.predicted_label == 1

    with pytest.raises(ValidationError):
        PredictResponse(
            stress_probability=0.5,
            predicted_label=2,  # type: ignore[arg-type]
            model_version="1.0.0",
        )


def test_response_models_valid() -> None:
    """Response models instantiate correctly with default disclaimers."""
    pred_res = PredictResponse(
        stress_probability=0.75,
        predicted_label=1,
        model_version="1.0.0",
    )
    assert pred_res.threshold == 0.5
    assert pred_res.disclaimer == RESEARCH_DISCLAIMER

    info_res = ModelInfoResponse(
        model_name="logistic_all",
        model_version="1.0.0",
        target="label_binary",
        trained_at_utc="2026-10-19T00:00:00Z",
        git_commit="abcdef",
        git_dirty=False,
        features=["hrv_mean_hr", "eda_mean"],
        n_features=2,
        training_samples=1015,
        environment={"python": "3.11", "scikit-learn": "1.3.0", "numpy": "1.26.0"},
    )
    assert info_res.disclaimer == RESEARCH_DISCLAIMER

    health_res = HealthResponse(status="healthy", model_loaded=True)
    assert health_res.status == "healthy"
