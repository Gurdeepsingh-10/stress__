"""Tests for FastAPI endpoints in stress.api.main."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stress.api.main import THRESHOLD, create_app, decide_label
from stress.api.model_loader import ModelLoadError, predict_stress_probability
from stress.api.schemas import PredictRequest
from tests.test_model_loader import _save_valid_artifact_bundle, _valid_request_dict


def test_decide_label_boundary_values() -> None:
    """decide_label implements strict inequality (p > threshold)."""
    assert decide_label(0.5) == 0
    assert decide_label(0.5000001) == 1
    assert decide_label(0.4999) == 0
    assert decide_label(0.7, threshold=0.7) == 0
    assert decide_label(0.7001, threshold=0.7) == 1


def test_predict_probability_matches_loader_and_pipeline(tmp_path: Path) -> None:
    """POST /predict probability equals predict_stress_probability and direct pipeline (1e-12)."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        req_data = _valid_request_dict()
        res = client.post("/predict", json=req_data)
        assert res.status_code == 200

        data = res.json()
        assert "stress_probability" in data
        assert "predicted_label" in data
        assert data["threshold"] == THRESHOLD
        assert data["model_version"] == "1.0.0"

        endpoint_prob = float(data["stress_probability"])

        # Compare with predict_stress_probability
        loaded = client.app.state.model
        req_obj = PredictRequest(**req_data)
        loader_prob = predict_stress_probability(loaded, req_obj)
        assert math.isclose(endpoint_prob, loader_prob, abs_tol=1e-12)

        # Compare with direct pipeline prediction on DataFrame row
        df_row = pd.DataFrame([req_data], columns=loaded.metadata["features"])
        direct_pipe_prob = float(loaded.pipeline.predict_proba(df_row)[0, 1])
        assert math.isclose(endpoint_prob, direct_pipe_prob, abs_tol=1e-12)

        expected_label = decide_label(endpoint_prob, threshold=THRESHOLD)
        assert data["predicted_label"] == expected_label


@pytest.mark.parametrize(
    "case_desc",
    ["nan_eda_slope", "unknown_extra_field", "missing_field", "out_of_range"],
)
def test_predict_validation_errors_return_422(tmp_path: Path, case_desc: str) -> None:
    """POST /predict returns 422 for invalid payloads."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        if case_desc == "nan_eda_slope":
            # Send raw JSON text with NaN (invalid JSON literal rejected by Pydantic)
            raw_text = json.dumps(_valid_request_dict()).replace(
                '"eda_slope": 0.001',
                '"eda_slope": NaN',
            )
            res = client.post(
                "/predict",
                content=raw_text,
                headers={"Content-Type": "application/json"},
            )
            assert res.status_code == 422

        elif case_desc == "unknown_extra_field":
            bad_data = dict(_valid_request_dict())
            bad_data["spurious_extra_sensor"] = 12.34
            res = client.post("/predict", json=bad_data)
            assert res.status_code == 422

        elif case_desc == "missing_field":
            bad_data = dict(_valid_request_dict())
            del bad_data["temp_mean"]
            res = client.post("/predict", json=bad_data)
            assert res.status_code == 422

        elif case_desc == "out_of_range":
            bad_data = dict(_valid_request_dict())
            bad_data["temp_mean"] = 999.0
            res = client.post("/predict", json=bad_data)
            assert res.status_code == 422


def test_model_info_matches_metadata_json(tmp_path: Path) -> None:
    """GET /model-info returns metadata matching the serialized JSON."""
    _, json_path = _save_valid_artifact_bundle(tmp_path)
    with json_path.open("r", encoding="utf-8") as f:
        expected_meta: dict[str, Any] = json.load(f)

    app = create_app(tmp_path)
    with TestClient(app) as client:
        res = client.get("/model-info")
        assert res.status_code == 200
        info = res.json()

        assert info["model_name"] == expected_meta["model_name"]
        assert info["model_version"] == expected_meta["model_version"]
        assert info["target"] == expected_meta["target"]
        assert info["trained_at_utc"] == expected_meta["trained_at_utc"]
        assert info["git_commit"] == expected_meta["git_commit"]
        assert info["git_dirty"] == expected_meta["git_dirty"]
        assert info["features"] == expected_meta["features"]
        assert info["n_features"] == expected_meta["n_features"]
        assert info["training_samples"] == expected_meta["training_samples"]
        assert info["environment"] == expected_meta["environment"]


def test_missing_artifact_raises_model_load_error(tmp_path: Path) -> None:
    """create_app on empty directory raises ModelLoadError on startup in TestClient."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    app = create_app(empty_dir)

    with pytest.raises(ModelLoadError):
        with TestClient(app):
            pass


def test_disclaimers_present_in_responses(tmp_path: Path) -> None:
    """Mandatory research and non-medical disclaimers are present in responses."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        pred_res = client.post("/predict", json=_valid_request_dict())
        assert pred_res.status_code == 200
        pred_data = pred_res.json()
        assert "disclaimer" in pred_data
        assert "medical" in pred_data["disclaimer"].lower()

        info_res = client.get("/model-info")
        assert info_res.status_code == 200
        info_data = info_res.json()
        assert "disclaimer" in info_data
        assert "medical" in info_data["disclaimer"].lower()


def test_hrv_null_accepted_by_predict(tmp_path: Path) -> None:
    """POST /predict accepts explicit null values for nullable HRV interval fields."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        payload = dict(_valid_request_dict())
        payload["hrv_mean_hr"] = None
        payload["hrv_sdnn"] = None
        payload["hrv_rmssd"] = None
        res = client.post("/predict", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert 0.0 <= data["stress_probability"] <= 1.0


def test_health_endpoint(tmp_path: Path) -> None:
    """GET /health returns 200 with status ok and model_loaded True."""
    _save_valid_artifact_bundle(tmp_path)
    app = create_app(tmp_path)

    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json() == {"status": "ok", "model_loaded": True}
