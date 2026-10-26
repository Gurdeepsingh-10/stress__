"""Tests for deployment artifacts, schema example, root redirect, and ONNX serving."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stress.api.main import create_app
from stress.api.schemas import PredictRequest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY_MODELS = ROOT / "deploy" / "models"


def test_deploy_artifact_files_and_integrity() -> None:
    """The 3 deployment model files exist, SHA-256 matches sidecar, features match schema."""
    onnx_file = DEPLOY_MODELS / "final_logistic_all.onnx"
    sidecar_file = DEPLOY_MODELS / "final_logistic_all.onnx.json"
    meta_file = DEPLOY_MODELS / "final_logistic_all.json"

    assert onnx_file.is_file(), f"Missing {onnx_file}"
    assert sidecar_file.is_file(), f"Missing {sidecar_file}"
    assert meta_file.is_file(), f"Missing {meta_file}"

    sidecar = json.loads(sidecar_file.read_text(encoding="utf-8"))
    meta = json.loads(meta_file.read_text(encoding="utf-8"))

    actual_sha = hashlib.sha256(onnx_file.read_bytes()).hexdigest()
    assert actual_sha == sidecar["onnx_sha256"], "ONNX SHA-256 does not match sidecar"

    schema_fields = list(PredictRequest.model_fields.keys())
    assert sidecar["features"] == meta["features"] == schema_fields, (
        "Sidecar features, metadata features, and PredictRequest fields must match exactly"
    )


def test_schema_example_validates_and_appears_in_json_schema() -> None:
    """PredictRequest has a synthetic example in json_schema_extra that validates."""
    schema = PredictRequest.model_json_schema()
    examples = schema.get("examples")
    assert examples and isinstance(examples, list) and len(examples) == 1, (
        f"Expected exactly 1 example in json_schema, got: {examples}"
    )

    example = examples[0]
    validated = PredictRequest.model_validate(example)
    assert isinstance(validated, PredictRequest)


def test_root_redirect_to_docs() -> None:
    """GET / returns 307 redirecting to /docs without entering application lifespan."""
    app = create_app()
    client = TestClient(app)
    response = client.get("/", follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


def test_deploy_model_onnx_inference() -> None:
    """load_onnx_model on deploy/models succeeds and scores the schema example."""
    pytest.importorskip("onnxruntime")
    from stress.api.onnx_backend import onnx_stress_probability
    from stress.api.onnx_loader import load_onnx_model

    loaded = load_onnx_model(DEPLOY_MODELS)
    example = PredictRequest.model_json_schema()["examples"][0]
    req = PredictRequest.model_validate(example)
    p = onnx_stress_probability(loaded.session, req, loaded.metadata["features"])

    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0
