"""Unit tests for ONNX pipeline export, parity checking, and inference backend.

All tests use synthetic data generated via numpy with fixed seeds (no real data).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

from scripts.bench_onnx import assert_functions_agree, sklearn_numpy_fn
from scripts.check_onnx_parity import compare_probabilities
from scripts.export_onnx import build_sidecar, convert_pipeline
from stress.api.model_loader import LoadedModel, predict_stress_probability
from stress.api.onnx_backend import load_onnx_session, onnx_stress_probability
from stress.api.schemas import PredictRequest
from stress.models.loso import build_pipeline

FEATURE_NAMES: list[str] = list(PredictRequest.model_fields.keys())
HRV_NULLABLE: list[str] = ["hrv_mean_hr", "hrv_hr_std", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"]


def _generate_synthetic_dict(
    rng: np.random.Generator,
    inject_nans: bool = False,
) -> dict[str, Any]:
    """Generate realistic feature values within PredictRequest schema bounds."""
    d: dict[str, Any] = {
        "hrv_mean_hr": float(rng.uniform(60.0, 100.0)),
        "hrv_hr_std": float(rng.uniform(2.0, 20.0)),
        "hrv_sdnn": float(rng.uniform(20.0, 150.0)),
        "hrv_rmssd": float(rng.uniform(15.0, 100.0)),
        "hrv_pnn50": float(rng.uniform(0.01, 0.5)),
        "hrv_num_beats": int(rng.integers(40, 120)),
        "eda_mean": float(rng.uniform(1.0, 20.0)),
        "eda_std": float(rng.uniform(0.1, 5.0)),
        "eda_min": float(rng.uniform(0.5, 10.0)),
        "eda_max": float(rng.uniform(10.0, 30.0)),
        "eda_slope": float(rng.uniform(-1.0, 1.0)),
        "eda_num_peaks": int(rng.integers(0, 15)),
        "temp_mean": float(rng.uniform(28.0, 36.0)),
        "temp_std": float(rng.uniform(0.01, 2.0)),
        "temp_slope": float(rng.uniform(-0.5, 0.5)),
        "acc_magnitude_mean": float(rng.uniform(0.8, 2.5)),
        "acc_magnitude_std": float(rng.uniform(0.01, 1.5)),
    }
    if inject_nans:
        for feat in HRV_NULLABLE:
            if rng.random() < 0.4:
                d[feat] = None
    return d


def _build_synthetic_dataset(
    n_samples: int = 150,
    seed: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic training DataFrame with NaN values in HRV columns."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _ in range(n_samples):
        row = _generate_synthetic_dict(rng, inject_nans=True)
        row_clean = {k: np.nan if v is None else v for k, v in row.items()}
        rows.append(row_clean)

    df = pd.DataFrame(rows, columns=FEATURE_NAMES)
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    rng.shuffle(y)
    return df, y


@pytest.fixture
def fitted_synthetic_pipeline() -> Any:
    """Build and fit a logistic pipeline on synthetic data."""
    df, y = _build_synthetic_dataset(n_samples=150, seed=42)
    pipe = build_pipeline("logistic", seed=42)
    pipe.fit(df, y)
    return pipe


def test_onnx_vs_sklearn_probabilities_match(
    fitted_synthetic_pipeline: Any,
    tmp_path: Path,
) -> None:
    """(a) ONNX vs sklearn probabilities match within 1e-5 on 200 synthetic rows."""
    pipe = fitted_synthetic_pipeline
    onnx_bytes = convert_pipeline(pipe, n_features=len(FEATURE_NAMES))
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(onnx_bytes)

    session = load_onnx_session(onnx_file)
    metadata = {"features": FEATURE_NAMES, "classes_": [0, 1]}
    loaded = LoadedModel(pipeline=pipe, metadata=metadata)

    rng = np.random.default_rng(12345)
    p_sk_list: list[float] = []
    p_onnx_list: list[float] = []

    for i in range(200):
        d = _generate_synthetic_dict(rng, inject_nans=(i % 2 == 0))
        req = PredictRequest.model_validate(d)
        p_sk = predict_stress_probability(loaded, req)
        p_onnx = onnx_stress_probability(session, req, FEATURE_NAMES)

        p_sk_list.append(p_sk)
        p_onnx_list.append(p_onnx)

        assert abs(p_sk - p_onnx) <= 1e-5
        assert (p_sk > 0.5) == (p_onnx > 0.5)

    parity = compare_probabilities(p_sk_list, p_onnx_list, tol=1e-5)
    assert parity.ok is True
    assert parity.max_abs_diff <= 1e-5
    assert parity.n_label_disagreements == 0


def test_build_sidecar_sha256_matches_joblib_bytes(
    fitted_synthetic_pipeline: Any,
    tmp_path: Path,
) -> None:
    """(b) build_sidecar sha256 equals hashlib.sha256 of the joblib file bytes."""
    pipe = fitted_synthetic_pipeline
    joblib_path = tmp_path / "synthetic_model.joblib"
    joblib.dump(pipe, joblib_path)
    joblib_bytes = joblib_path.read_bytes()

    onnx_bytes = convert_pipeline(pipe, n_features=len(FEATURE_NAMES))
    metadata = {"features": FEATURE_NAMES, "n_features": len(FEATURE_NAMES)}

    sidecar = build_sidecar(
        joblib_bytes=joblib_bytes,
        onnx_bytes=onnx_bytes,
        metadata=metadata,
    )

    expected_joblib_sha256 = hashlib.sha256(joblib_bytes).hexdigest()
    expected_onnx_sha256 = hashlib.sha256(onnx_bytes).hexdigest()

    assert sidecar["source_joblib_sha256"] == expected_joblib_sha256
    assert sidecar["onnx_sha256"] == expected_onnx_sha256
    assert sidecar["source_joblib_sha256"] != sidecar["onnx_sha256"]
    assert sidecar["n_features"] == len(FEATURE_NAMES)
    assert sidecar["features"] == FEATURE_NAMES
    assert sidecar["opset"] == 17
    assert "library_versions" in sidecar
    # Explicitly verify removed keys are absent
    assert "joblib_sha256" not in sidecar
    assert "target_opset" not in sidecar
    assert "environment" not in sidecar


def test_onnx_stress_probability_returns_class_1_probability(
    fitted_synthetic_pipeline: Any,
    tmp_path: Path,
) -> None:
    """(c) onnx_stress_probability returns class-1 probability matching predict_proba[:,1]."""
    pipe = fitted_synthetic_pipeline
    onnx_bytes = convert_pipeline(pipe, n_features=len(FEATURE_NAMES))
    onnx_file = tmp_path / "model.onnx"
    onnx_file.write_bytes(onnx_bytes)

    session = load_onnx_session(onnx_file)

    rng = np.random.default_rng(999)
    d = _generate_synthetic_dict(rng, inject_nans=False)
    req = PredictRequest.model_validate(d)

    df_row = pd.DataFrame([[getattr(req, f) for f in FEATURE_NAMES]], columns=FEATURE_NAMES)
    sk_probs = pipe.predict_proba(df_row)[0]
    p_class0 = float(sk_probs[0])
    p_class1 = float(sk_probs[1])

    assert abs(p_class1 - 0.5) > 0.05

    p_onnx = onnx_stress_probability(session, req, FEATURE_NAMES)

    assert abs(p_onnx - p_class1) < 1e-5
    assert abs(p_onnx - p_class0) > 0.05
    assert abs(p_onnx - (1.0 - p_class1)) > 0.05


def test_convert_pipeline_raises_on_invalid_classes(
    fitted_synthetic_pipeline: Any,
) -> None:
    """(d) convert_pipeline raises on a pipeline whose classes_ != [0, 1]."""
    pipe = fitted_synthetic_pipeline

    pipe.named_steps["clf"].classes_ = np.array([1, 2])
    with pytest.raises(ValueError, match=r"classes_"):
        convert_pipeline(pipe, n_features=len(FEATURE_NAMES))

    pipe.named_steps["clf"].classes_ = np.array([0])
    with pytest.raises(ValueError, match=r"classes_"):
        convert_pipeline(pipe, n_features=len(FEATURE_NAMES))

    pipe.named_steps["clf"].classes_ = None
    with pytest.raises(ValueError, match=r"classes_"):
        convert_pipeline(pipe, n_features=len(FEATURE_NAMES))


def test_compare_probabilities_flags_disagreements_and_tolerance() -> None:
    """(e) compare_probabilities flags a disagreement and flags max diff above tol."""
    p_sk = [0.49, 0.40]
    p_onnx = [0.51, 0.40]
    res_disagree = compare_probabilities(p_sk, p_onnx, tol=1e-5)
    assert res_disagree.ok is False
    assert res_disagree.n_label_disagreements == 1
    assert res_disagree.max_abs_diff > 1e-5

    p_sk2 = [0.70]
    p_onnx2 = [0.7001]
    res_diff = compare_probabilities(p_sk2, p_onnx2, tol=1e-5)
    assert res_diff.ok is False
    assert res_diff.n_label_disagreements == 0
    assert res_diff.max_abs_diff > 1e-5

    res_threshold = compare_probabilities([0.5], [0.5], tol=1e-5)
    assert res_threshold.n_label_disagreements == 0
    assert res_threshold.max_abs_diff <= 1e-5
    assert res_threshold.ok is True


def test_sklearn_numpy_fn_matches_predict_stress_probability(
    fitted_synthetic_pipeline: Any,
) -> None:
    """sklearn_numpy_fn agrees with predict_stress_probability on synthetic rows with NaNs."""
    pipe = fitted_synthetic_pipeline
    metadata = {"features": FEATURE_NAMES, "classes_": [0, 1]}
    loaded = LoadedModel(pipeline=pipe, metadata=metadata)

    rng = np.random.default_rng(54321)
    for i in range(100):
        d = _generate_synthetic_dict(rng, inject_nans=(i % 2 == 0))
        req = PredictRequest.model_validate(d)

        p_df = predict_stress_probability(loaded, req)
        p_np = sklearn_numpy_fn(pipe, req, FEATURE_NAMES)

        assert abs(p_df - p_np) <= 1e-5
        assert (p_df > 0.5) == (p_np > 0.5)


def test_assert_functions_agree_validates_tolerance() -> None:
    """assert_functions_agree accepts matching arrays and raises on disagreements."""
    assert_functions_agree([0.5, 0.8], [0.5, 0.8], [0.5, 0.8], tol=1e-5)

    with pytest.raises(ValueError, match=r"Functions disagree"):
        assert_functions_agree([0.5, 0.8], [0.5, 0.8], [0.5, 0.801], tol=1e-5)
