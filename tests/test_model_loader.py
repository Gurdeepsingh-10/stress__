"""Tests for model_loader.py using synthetic models and data."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn

from stress.api.model_loader import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    LoadedModel,
    ModelLoadError,
    load_model,
    predict_stress_probability,
)
from stress.api.schemas import PredictRequest
from stress.models.loso import build_pipeline

SCHEMA_FEATURES = list(PredictRequest.model_fields.keys())


def _make_synthetic_df(n_samples: int = 40, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic DataFrame with the 17 schema features and label_binary."""
    rng = np.random.RandomState(seed)
    data: dict[str, Any] = {}
    for i, feat in enumerate(SCHEMA_FEATURES):
        # Give different scales so weights become distinct
        data[feat] = rng.normal(loc=(i + 1) * 2.0, scale=1.0, size=n_samples)
    # Ensure both classes are present
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    rng.shuffle(labels)
    data["label_binary"] = labels
    return pd.DataFrame(data)


def _save_valid_artifact_bundle(
    target_dir: Path,
    features: list[str] | None = None,
    classes: list[int] | None = None,
    sklearn_version: str | None = None,
    numpy_version: str | None = None,
    pipeline: Any | None = None,
) -> tuple[Path, Path]:
    """Helper to write a model bundle for loader testing."""
    target_dir.mkdir(parents=True, exist_ok=True)
    feats = features if features is not None else list(SCHEMA_FEATURES)

    if pipeline is None:
        df = _make_synthetic_df()
        pipe = build_pipeline("logistic", seed=42)
        pipe.fit(df[feats], df["label_binary"])
    else:
        pipe = pipeline

    if classes is not None:
        pipe.named_steps["clf"].classes_ = np.array(classes)

    meta = {
        "model_name": "logistic_all",
        "model_version": "1.0.0",
        "target": "label_binary",
        "features": feats,
        "n_features": len(feats),
        "training_samples": 40,
        "class_counts": {0: 20, 1: 20},
        "classes_": [int(c) for c in getattr(pipe, "classes_", [0, 1])],
        "trained_at_utc": "2026-10-21T00:00:00Z",
        "git_commit": "abcdef0",
        "git_dirty": False,
        "environment": {
            "python": "3.11.9",
            "scikit-learn": sklearn_version if sklearn_version is not None else sklearn.__version__,
            "numpy": numpy_version if numpy_version is not None else np.__version__,
            "joblib": joblib.__version__,
        },
    }

    joblib_path = target_dir / MODEL_FILENAME
    json_path = target_dir / METADATA_FILENAME

    joblib.dump(pipe, joblib_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return joblib_path, json_path


def _valid_request_dict() -> dict[str, Any]:
    """Construct a complete, valid PredictRequest dictionary."""
    return {
        "hrv_mean_hr": 72.0,
        "hrv_hr_std": 4.5,
        "hrv_sdnn": 45.0,
        "hrv_rmssd": 38.0,
        "hrv_pnn50": 0.12,
        "hrv_num_beats": 70,
        "eda_mean": 2.1,
        "eda_std": 0.35,
        "eda_min": 1.5,
        "eda_max": 2.9,
        "eda_slope": 0.001,
        "eda_num_peaks": 3,
        "temp_mean": 32.4,
        "temp_std": 0.12,
        "temp_slope": -0.0005,
        "acc_magnitude_mean": 0.99,
        "acc_magnitude_std": 0.03,
    }


def test_golden_loader_probability_matches_pipeline(tmp_path: Path) -> None:
    """Loader probability equals direct pipeline.predict_proba on the same row."""
    _save_valid_artifact_bundle(tmp_path)
    loaded = load_model(tmp_path)
    assert isinstance(loaded, LoadedModel)

    req_data = _valid_request_dict()
    req = PredictRequest(**req_data)

    # Directly evaluate pipeline on DataFrame row
    df_row = pd.DataFrame([req_data], columns=loaded.metadata["features"])
    direct_proba = float(loaded.pipeline.predict_proba(df_row)[0, 1])

    loader_proba = predict_stress_probability(loaded, req)

    # Must match to 1e-12 precision
    assert math.isclose(loader_proba, direct_proba, abs_tol=1e-12)
    # Ensure class 0 and class 1 probabilities are non-degenerate
    direct_class_0_proba = float(loaded.pipeline.predict_proba(df_row)[0, 0])
    assert not math.isclose(loader_proba, direct_class_0_proba, abs_tol=1e-4)


def test_feature_order_shuffled_metadata_followed(tmp_path: Path) -> None:
    """Loader follows metadata feature ordering, not request/dict order.

    Trained on shuffled features with distinct coefficient weights.
    """
    # Reverse the feature order
    shuffled_features = list(reversed(SCHEMA_FEATURES))
    assert shuffled_features != SCHEMA_FEATURES

    req_data = _valid_request_dict()
    rng = np.random.RandomState(123)
    df = pd.DataFrame(
        [{f: req_data[f] + rng.uniform(-1.0, 1.0) for f in shuffled_features} for _ in range(50)]
    )
    df["label_binary"] = [0, 1] * 25

    pipe = build_pipeline("logistic", seed=42)
    # Fit pipeline using the shuffled order
    pipe.fit(df[shuffled_features], df["label_binary"])
    # Set distinct non-saturating coefficients per feature
    n_feats = len(shuffled_features)
    pipe.named_steps["clf"].coef_ = np.array([[0.01 * (i + 1) for i in range(n_feats)]])
    pipe.named_steps["clf"].intercept_ = np.array([0.0])

    _save_valid_artifact_bundle(
        tmp_path,
        features=shuffled_features,
        pipeline=pipe,
    )

    loaded = load_model(tmp_path)
    assert loaded.metadata["features"] == shuffled_features

    req = PredictRequest(**req_data)

    # Direct prediction evaluated using the metadata (shuffled) order
    expected_row = pd.DataFrame(
        [{f: req_data[f] for f in shuffled_features}],
        columns=shuffled_features,
    )
    expected_proba = float(pipe.predict_proba(expected_row)[0, 1])

    loader_proba = predict_stress_probability(loaded, req)
    assert math.isclose(loader_proba, expected_proba, abs_tol=1e-12)

    # Contrast with wrong ordering (schema / request dict order)
    wrong_order_row = pd.DataFrame(
        [[req_data[f] for f in SCHEMA_FEATURES]],
        columns=shuffled_features,
    )
    wrong_order_proba = float(pipe.predict_proba(wrong_order_row)[0, 1])
    assert not math.isclose(loader_proba, wrong_order_proba, abs_tol=1e-4)


def test_missing_joblib_raises_model_load_error(tmp_path: Path) -> None:
    """Missing joblib file raises ModelLoadError."""
    _save_valid_artifact_bundle(tmp_path)
    (tmp_path / MODEL_FILENAME).unlink()
    with pytest.raises(ModelLoadError, match="Model artifact not found"):
        load_model(tmp_path)


def test_missing_json_raises_model_load_error(tmp_path: Path) -> None:
    """Missing metadata JSON raises ModelLoadError."""
    _save_valid_artifact_bundle(tmp_path)
    (tmp_path / METADATA_FILENAME).unlink()
    with pytest.raises(ModelLoadError, match="Model metadata not found"):
        load_model(tmp_path)


def test_sklearn_version_mismatch_raises_model_load_error(tmp_path: Path) -> None:
    """Major/minor scikit-learn version mismatch in metadata raises ModelLoadError."""
    _save_valid_artifact_bundle(tmp_path, sklearn_version="0.24.2")
    with pytest.raises(ModelLoadError, match="scikit-learn version mismatch"):
        load_model(tmp_path)


def test_numpy_version_mismatch_raises_model_load_error(tmp_path: Path) -> None:
    """Major/minor numpy version mismatch in metadata raises ModelLoadError."""
    _save_valid_artifact_bundle(tmp_path, numpy_version="1.19.5")
    with pytest.raises(ModelLoadError, match="numpy version mismatch"):
        load_model(tmp_path)


def test_feature_list_mismatch_raises_model_load_error(tmp_path: Path) -> None:
    """Metadata with missing feature raises ModelLoadError."""
    bad_features = [f for f in SCHEMA_FEATURES if f != "temp_mean"]
    _save_valid_artifact_bundle(tmp_path, features=bad_features)
    with pytest.raises(ModelLoadError, match="features do not match"):
        load_model(tmp_path)


def test_feature_list_duplicates_raises_model_load_error(tmp_path: Path) -> None:
    """Metadata whose feature list has duplicates (length != 17) raises ModelLoadError."""
    df = _make_synthetic_df()
    pipe = build_pipeline("logistic", seed=42)
    pipe.fit(df[SCHEMA_FEATURES], df["label_binary"])
    duplicate_features = list(SCHEMA_FEATURES) + [SCHEMA_FEATURES[0]]
    _save_valid_artifact_bundle(tmp_path, features=duplicate_features, pipeline=pipe)
    with pytest.raises(ModelLoadError, match="features do not match"):
        load_model(tmp_path)


def test_wrong_classes_raises_model_load_error(tmp_path: Path) -> None:
    """Pipeline with classes_ != [0, 1] raises ModelLoadError."""
    _save_valid_artifact_bundle(tmp_path, classes=[0, 2])
    with pytest.raises(ModelLoadError, match="Expected pipeline classes_"):
        load_model(tmp_path)


def test_hrv_none_imputed_successfully(tmp_path: Path) -> None:
    """None in nullable HRV fields converts to NaN and is imputed without error."""
    _save_valid_artifact_bundle(tmp_path)
    loaded = load_model(tmp_path)

    req_data = _valid_request_dict()
    req_data["hrv_mean_hr"] = None
    req_data["hrv_sdnn"] = None
    req_data["hrv_rmssd"] = None
    req = PredictRequest(**req_data)

    proba = predict_stress_probability(loaded, req)
    assert isinstance(proba, float)
    assert 0.0 <= proba <= 1.0
