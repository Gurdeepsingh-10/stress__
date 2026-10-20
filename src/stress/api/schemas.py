"""Pydantic schemas for the stress inference API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

RESEARCH_DISCLAIMER: str = (
    "Research demonstration only. Not a medical device. Not intended for clinical "
    "diagnosis, decision-making, or treatment."
)


class PredictRequest(BaseModel):
    """Input payload containing the 17 physiological features for one 60-second window.

    All float features reject NaN and infinite values via allow_inf_nan=False.
    Unrecognized extra attributes are rejected via extra='forbid'.
    HRV interval features are required but accept None / null when peak detection
    quality is low (fewer than 10 peaks detected in the window).
    """

    model_config = ConfigDict(extra="forbid")

    # HRV features (BVP derived) — required, nullable when peak detection is poor
    hrv_mean_hr: float | None = Field(
        ...,
        ge=30.0,
        le=240.0,
        allow_inf_nan=False,
        description="Mean heart rate in bpm. None if fewer than 10 peaks detected.",
    )
    hrv_hr_std: float | None = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Standard deviation of heart rate. None if fewer than 10 peaks detected.",
    )
    hrv_sdnn: float | None = Field(
        ...,
        ge=0.0,
        le=10000.0,
        allow_inf_nan=False,
        description=(
            "Standard deviation of inter-beat intervals in ms. None if <10 peaks. "
            "Values above ~300 ms are likely peak-detection artifacts."
        ),
    )
    hrv_rmssd: float | None = Field(
        ...,
        ge=0.0,
        le=10000.0,
        allow_inf_nan=False,
        description=(
            "Root mean square of successive IBI differences in ms. None if <10 peaks. "
            "Values above ~300 ms are likely peak-detection artifacts."
        ),
    )
    hrv_pnn50: float | None = Field(
        ...,
        ge=0.0,
        le=1.0,
        allow_inf_nan=False,
        description="Fraction of successive IBI differences > 50 ms. None if <10 peaks.",
    )
    hrv_num_beats: int = Field(
        ...,
        ge=0,
        le=300,
        description="Total detected peaks in window.",
    )

    # EDA features
    eda_mean: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Mean electrodermal activity in uS.",
    )
    eda_std: float = Field(
        ...,
        ge=0.0,
        le=50.0,
        allow_inf_nan=False,
        description="Standard deviation of electrodermal activity.",
    )
    eda_min: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Minimum electrodermal activity in window.",
    )
    eda_max: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        allow_inf_nan=False,
        description="Maximum electrodermal activity in window.",
    )
    eda_slope: float = Field(
        ...,
        allow_inf_nan=False,
        description="Linear trend slope of electrodermal activity.",
    )
    eda_num_peaks: int = Field(
        ...,
        ge=0,
        le=100,
        description="Detected skin conductance response peak count.",
    )

    # Temperature features
    temp_mean: float = Field(
        ...,
        ge=15.0,
        le=45.0,
        allow_inf_nan=False,
        description="Mean skin temperature in degrees Celsius.",
    )
    temp_std: float = Field(
        ...,
        ge=0.0,
        le=10.0,
        allow_inf_nan=False,
        description="Standard deviation of skin temperature.",
    )
    temp_slope: float = Field(
        ...,
        allow_inf_nan=False,
        description="Linear trend slope of skin temperature.",
    )

    # Accelerometer features
    acc_magnitude_mean: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        allow_inf_nan=False,
        description="Mean acceleration vector magnitude in g.",
    )
    acc_magnitude_std: float = Field(
        ...,
        ge=0.0,
        le=20.0,
        allow_inf_nan=False,
        description="Standard deviation of acceleration vector magnitude in g.",
    )


class PredictResponse(BaseModel):
    """Output prediction response from the stress classifier."""

    stress_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Predicted posterior probability of acute stress class (label 1).",
    )
    predicted_label: Literal[0, 1] = Field(
        ...,
        description="Binary classification decision (0 = non-stress, 1 = stress).",
    )
    threshold: float = Field(
        default=0.5,
        description="Decision threshold applied to stress probability.",
    )
    model_version: str = Field(
        ...,
        description="Model semantic version.",
    )
    disclaimer: str = Field(
        default=RESEARCH_DISCLAIMER,
        description="Mandatory medical and research disclaimer.",
    )


class ModelInfoResponse(BaseModel):
    """Metadata response describing the active model artifact and its provenance."""

    model_name: str
    model_version: str
    target: str
    trained_at_utc: str
    git_commit: str
    git_dirty: bool
    features: list[str]
    n_features: int
    training_samples: int
    environment: dict[str, str]
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER)


class HealthResponse(BaseModel):
    """Health status response."""

    status: str
    model_loaded: bool
