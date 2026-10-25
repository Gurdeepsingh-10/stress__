"""Model loading and prediction service for the stress detection API.

Note: joblib artifacts execute arbitrary code on load; only load self-built,
trusted artifacts.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stress.api.schemas import PredictRequest

MODEL_FILENAME = "final_logistic_all.joblib"
METADATA_FILENAME = "final_logistic_all.json"


class ModelLoadError(Exception):
    """Raised when model artifact or metadata is missing, invalid, or incompatible."""


@dataclass
class LoadedModel:
    """Container for the loaded scikit-learn pipeline and its training metadata."""

    pipeline: Any
    metadata: dict[str, Any]


def _parse_major_minor(version_str: str) -> tuple[int, int]:
    """Extract (major, minor) integers from a semver-like version string."""
    parts = version_str.split(".")
    return int(parts[0]), int(parts[1])


def get_default_model_dir() -> Path:
    """Return the default model directory from STRESS_MODEL_DIR or 'models'."""
    return Path(os.environ.get("STRESS_MODEL_DIR", "models"))


def validate_features(metadata: dict[str, Any]) -> None:
    """Validate that metadata feature list matches PredictRequest schema fields.

    Args:
        metadata: Metadata dictionary, must contain a 'features' list.

    Raises:
        ModelLoadError: If features are missing, wrong length, or don't match the schema.
    """
    schema_fields = set(PredictRequest.model_fields.keys())
    meta_features = metadata.get("features")
    if (
        not isinstance(meta_features, list)
        or len(meta_features) != len(PredictRequest.model_fields)
        or set(meta_features) != schema_fields
    ):
        raise ModelLoadError(
            f"Model metadata features do not match PredictRequest schema fields: {meta_features}"
        )


def load_model(model_dir: Path | str | None = None) -> LoadedModel:
    """Load model artifact and metadata from directory, validating compatibility.

    Args:
        model_dir: Directory containing final_logistic_all.joblib and .json.
            Defaults to STRESS_MODEL_DIR environment variable or 'models'.

    Returns:
        LoadedModel containing the deserialized pipeline and metadata dictionary.

    Raises:
        ModelLoadError: If files are missing, feature set mismatches schema,
            library versions mismatch (scikit-learn or numpy major.minor),
            or pipeline classes_ != [0, 1].
    """
    import joblib
    import sklearn

    base_dir = Path(model_dir) if model_dir is not None else get_default_model_dir()
    joblib_path = base_dir / MODEL_FILENAME
    json_path = base_dir / METADATA_FILENAME

    if not joblib_path.is_file():
        raise ModelLoadError(f"Model artifact not found: {joblib_path}")
    if not json_path.is_file():
        raise ModelLoadError(f"Model metadata not found: {json_path}")

    try:
        with json_path.open("r", encoding="utf-8") as f:
            metadata: dict[str, Any] = json.load(f)
    except Exception as exc:
        raise ModelLoadError(f"Failed to parse metadata JSON {json_path}: {exc}") from exc

    # Feature list validation (sets must match schema PredictRequest fields, length must match 17)
    validate_features(metadata)

    # Library version compatibility check (major.minor must match)
    env_info = metadata.get("environment", {})
    if not isinstance(env_info, dict):
        raise ModelLoadError("Metadata missing 'environment' section")

    meta_sklearn = env_info.get("scikit-learn")
    if not meta_sklearn:
        raise ModelLoadError("Metadata environment missing 'scikit-learn' version")
    try:
        if _parse_major_minor(str(meta_sklearn)) != _parse_major_minor(sklearn.__version__):
            raise ModelLoadError(
                f"scikit-learn version mismatch: model trained with {meta_sklearn}, "
                f"current runtime is {sklearn.__version__}"
            )
    except ValueError as exc:
        raise ModelLoadError(f"Invalid scikit-learn version format: {meta_sklearn}") from exc

    meta_numpy = env_info.get("numpy")
    if not meta_numpy:
        raise ModelLoadError("Metadata environment missing 'numpy' version")
    try:
        if _parse_major_minor(str(meta_numpy)) != _parse_major_minor(np.__version__):
            raise ModelLoadError(
                f"numpy version mismatch: model trained with {meta_numpy}, "
                f"current runtime is {np.__version__}"
            )
    except ValueError as exc:
        raise ModelLoadError(f"Invalid numpy version format: {meta_numpy}") from exc

    # Load serialized pipeline
    try:
        pipeline = joblib.load(joblib_path)
    except Exception as exc:
        raise ModelLoadError(f"Failed to load joblib pipeline from {joblib_path}: {exc}") from exc

    # Validate classes_
    classes = getattr(pipeline, "classes_", None)
    if classes is None:
        raise ModelLoadError("Loaded pipeline has no classes_ attribute")
    if list(classes) != [0, 1]:
        raise ModelLoadError(f"Expected pipeline classes_ [0, 1], got {list(classes)}")

    return LoadedModel(pipeline=pipeline, metadata=metadata)


def predict_stress_probability(loaded: LoadedModel, request: PredictRequest) -> float:
    """Predict the probability of stress (class 1) for a given PredictRequest.

    The single input row is assembled following the exact feature ordering from
    loaded.metadata['features'] (not the request dictionary order).
    Nullable HRV fields with None are converted to np.nan for pipeline imputer.

    Args:
        loaded: LoadedModel with pipeline and metadata.
        request: PredictRequest with validated feature values.

    Returns:
        Predicted probability of stress (class 1) as a float in [0.0, 1.0].
    """
    import pandas as pd

    features: list[str] = loaded.metadata["features"]
    row: dict[str, Any] = {}
    for feat in features:
        val = getattr(request, feat)
        row[feat] = np.nan if val is None else float(val)

    df_row = pd.DataFrame([row], columns=features)
    proba = loaded.pipeline.predict_proba(df_row)

    classes = list(loaded.pipeline.classes_)
    class_1_idx = classes.index(1)
    return float(proba[0, class_1_idx])
