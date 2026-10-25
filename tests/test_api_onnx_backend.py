"""Tests for the ONNX serving backend and API integration.

Uses pytest.importorskip for skl2onnx and onnxruntime so the test suite
skips gracefully in environments without the [ml] extra.
All data is synthetic; no real WESAD feature rows are used.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

skl2onnx = pytest.importorskip("skl2onnx")
ort = pytest.importorskip("onnxruntime")

import joblib  # noqa: E402 – after importorskip
from fastapi.testclient import TestClient  # noqa: E402

from scripts.export_onnx import build_sidecar, convert_pipeline  # noqa: E402
from stress.api.main import create_app  # noqa: E402
from stress.api.model_loader import ModelLoadError  # noqa: E402
from stress.api.onnx_loader import load_onnx_model  # noqa: E402
from stress.api.schemas import PredictRequest  # noqa: E402
from stress.models.loso import build_pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# Schema feature list (same ordering as PredictRequest)
# ---------------------------------------------------------------------------
SCHEMA_FEATURES: list[str] = list(PredictRequest.model_fields.keys())
N_FEATURES: int = len(SCHEMA_FEATURES)

# HRV nullable fields
_HRV_NULLABLE = {"hrv_mean_hr", "hrv_hr_std", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50"}


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_synthetic_df(n_samples: int = 60, seed: int = 42) -> Any:
    """Return a pandas DataFrame with schema features, NaNs in some HRV columns."""
    import pandas as pd

    rng = np.random.RandomState(seed)
    data: dict[str, Any] = {}
    for i, feat in enumerate(SCHEMA_FEATURES):
        col = rng.normal(loc=(i + 1) * 2.0, scale=1.0, size=n_samples).astype(float)
        if feat in _HRV_NULLABLE:
            # Introduce ~20 % NaN
            mask = rng.rand(n_samples) < 0.2
            col[mask] = np.nan
        data[feat] = col
    # Balanced binary labels
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
    rng.shuffle(labels)
    data["label_binary"] = labels
    return pd.DataFrame(data)


def _make_synthetic_payload(seed: int = 0, with_nulls: bool = False) -> dict[str, Any]:
    """Build a complete valid PredictRequest dict, optionally with null HRV fields."""
    rng = np.random.RandomState(seed)
    payload: dict[str, Any] = {
        "hrv_mean_hr": float(rng.uniform(45, 150)),
        "hrv_hr_std": float(rng.uniform(0, 20)),
        "hrv_sdnn": float(rng.uniform(0, 200)),
        "hrv_rmssd": float(rng.uniform(0, 200)),
        "hrv_pnn50": float(rng.uniform(0, 1)),
        "hrv_num_beats": int(rng.randint(10, 150)),
        "eda_mean": float(rng.uniform(0, 20)),
        "eda_std": float(rng.uniform(0, 5)),
        "eda_min": float(rng.uniform(0, 5)),
        "eda_max": float(rng.uniform(5, 30)),
        "eda_slope": float(rng.uniform(-0.01, 0.01)),
        "eda_num_peaks": int(rng.randint(0, 10)),
        "temp_mean": float(rng.uniform(28, 38)),
        "temp_std": float(rng.uniform(0, 2)),
        "temp_slope": float(rng.uniform(-0.005, 0.005)),
        "acc_magnitude_mean": float(rng.uniform(0.8, 1.2)),
        "acc_magnitude_std": float(rng.uniform(0, 0.5)),
    }
    if with_nulls:
        for f in _HRV_NULLABLE:
            payload[f] = None
    return payload


def _write_metadata(path: Path, features: list[str], n_samples: int = 60) -> None:
    """Write a minimal valid metadata JSON to path."""
    import sklearn

    meta = {
        "model_name": "logistic_all",
        "model_version": "1.0.0",
        "target": "label_binary",
        "features": features,
        "n_features": len(features),
        "training_samples": n_samples,
        "class_counts": {"0": n_samples // 2, "1": n_samples - n_samples // 2},
        "classes_": [0, 1],
        "trained_at_utc": "2026-10-25T00:00:00Z",
        "git_commit": "synthetic",
        "git_dirty": False,
        "environment": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Fixture: full ONNX model directory
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def onnx_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a complete ONNX model directory from a synthetic sklearn pipeline."""
    d = tmp_path_factory.mktemp("onnx_model")

    # Train sklearn pipeline on synthetic data
    df = _make_synthetic_df(n_samples=60, seed=42)
    pipe = build_pipeline("logistic", seed=42)
    pipe.fit(df[SCHEMA_FEATURES], df["label_binary"])

    # Write metadata JSON (final_logistic_all.json)
    meta_path = d / "final_logistic_all.json"
    _write_metadata(meta_path, SCHEMA_FEATURES, n_samples=60)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    # Convert to ONNX
    onnx_bytes = convert_pipeline(pipe, n_features=N_FEATURES)
    onnx_path = d / "final_logistic_all.onnx"
    onnx_path.write_bytes(onnx_bytes)

    # Write joblib (needed for build_sidecar source_joblib_sha256)
    joblib_path = d / "_tmp.joblib"
    joblib.dump(pipe, joblib_path)
    joblib_bytes = joblib_path.read_bytes()
    joblib_path.unlink()

    # Write sidecar
    sidecar = build_sidecar(joblib_bytes=joblib_bytes, onnx_bytes=onnx_bytes, metadata=metadata)
    sidecar_path = d / "final_logistic_all.onnx.json"
    sidecar_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    return d


@pytest.fixture(scope="module")
def sklearn_model_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build a sklearn-only model directory (no ONNX files)."""
    from tests.test_model_loader import _save_valid_artifact_bundle

    d = tmp_path_factory.mktemp("sklearn_model")
    _save_valid_artifact_bundle(d)
    return d


# ---------------------------------------------------------------------------
# (a) /predict with backend="onnx" matches sklearn within 1e-5
# ---------------------------------------------------------------------------


class TestOnnxPredictParity:
    """ONNX /predict probability matches in-process sklearn pipeline."""

    def test_predict_parity_50_payloads(self, onnx_model_dir: Path) -> None:
        """ONNX and sklearn probabilities agree within 1e-5 on 50 synthetic payloads."""
        import pandas as pd

        from stress.api.onnx_backend import load_onnx_session, onnx_stress_probability

        # Load sklearn reference
        metadata = json.loads((onnx_model_dir / "final_logistic_all.json").read_text())
        features = metadata["features"]
        # Re-train the same pipeline for direct comparison
        df = _make_synthetic_df(n_samples=60, seed=42)
        pipe = build_pipeline("logistic", seed=42)
        pipe.fit(df[SCHEMA_FEATURES], df["label_binary"])
        session = load_onnx_session(onnx_model_dir / "final_logistic_all.onnx")

        app = create_app(onnx_model_dir, backend="onnx")
        with TestClient(app) as client:
            for seed in range(50):
                with_nulls = seed % 5 == 0  # every 5th has null HRV
                payload = _make_synthetic_payload(seed=seed, with_nulls=with_nulls)
                res = client.post("/predict", json=payload)
                assert res.status_code == 200, f"seed={seed}: {res.text}"
                data = res.json()
                onnx_p = float(data["stress_probability"])

                req_obj = PredictRequest(**payload)
                onnx_ref_p = onnx_stress_probability(session, req_obj, features)
                assert math.isclose(onnx_p, onnx_ref_p, abs_tol=1e-12), (
                    f"seed={seed}: endpoint {onnx_p} != onnx_stress_probability {onnx_ref_p}"
                )

                # Direct comparison with fitted sklearn pipeline
                row_dict = {
                    feat: [np.nan if payload[feat] is None else payload[feat]]
                    for feat in SCHEMA_FEATURES
                }
                row_df = pd.DataFrame(row_dict)
                sklearn_p = float(pipe.predict_proba(row_df)[:, 1][0])
                assert math.isclose(onnx_p, sklearn_p, abs_tol=1e-5), (
                    f"seed={seed}: endpoint {onnx_p} != sklearn {sklearn_p}"
                )

                # label follows strict p > 0.5 rule
                expected_label = 1 if onnx_p > 0.5 else 0
                assert data["predicted_label"] == expected_label, f"seed={seed}"


# ---------------------------------------------------------------------------
# (b) /model-info backend field
# ---------------------------------------------------------------------------


class TestModelInfoBackend:
    """backend field in /model-info response."""

    def test_model_info_onnx_backend(self, onnx_model_dir: Path) -> None:
        """/model-info reports backend='onnx' when app is created with backend='onnx'."""
        app = create_app(onnx_model_dir, backend="onnx")
        with TestClient(app) as client:
            res = client.get("/model-info")
            assert res.status_code == 200
            assert res.json()["backend"] == "onnx"

    def test_model_info_sklearn_backend(self, sklearn_model_dir: Path) -> None:
        """/model-info reports backend='sklearn' for the sklearn backend."""
        app = create_app(sklearn_model_dir, backend="sklearn")
        with TestClient(app) as client:
            res = client.get("/model-info")
            assert res.status_code == 200
            assert res.json()["backend"] == "sklearn"


# ---------------------------------------------------------------------------
# (c) load_onnx_model error cases
# ---------------------------------------------------------------------------


class TestLoadOnnxModelErrors:
    """load_onnx_model raises ModelLoadError for integrity failures."""

    def test_missing_onnx_file(self, onnx_model_dir: Path, tmp_path: Path) -> None:
        """ModelLoadError when .onnx file is absent."""
        import shutil

        shutil.copy(
            onnx_model_dir / "final_logistic_all.json", tmp_path / "final_logistic_all.json"
        )
        shutil.copy(
            onnx_model_dir / "final_logistic_all.onnx.json",
            tmp_path / "final_logistic_all.onnx.json",
        )
        # .onnx not copied
        with pytest.raises(ModelLoadError, match="ONNX model not found"):
            load_onnx_model(tmp_path)

    def test_sha256_mismatch(self, onnx_model_dir: Path, tmp_path: Path) -> None:
        """ModelLoadError when one byte of the .onnx file is altered."""
        import shutil

        shutil.copy(
            onnx_model_dir / "final_logistic_all.json", tmp_path / "final_logistic_all.json"
        )
        shutil.copy(
            onnx_model_dir / "final_logistic_all.onnx.json",
            tmp_path / "final_logistic_all.onnx.json",
        )
        original = (onnx_model_dir / "final_logistic_all.onnx").read_bytes()
        # Flip last byte
        corrupted = original[:-1] + bytes([original[-1] ^ 0xFF])
        (tmp_path / "final_logistic_all.onnx").write_bytes(corrupted)
        with pytest.raises(ModelLoadError, match="SHA-256 mismatch"):
            load_onnx_model(tmp_path)

    def test_sidecar_features_differ_from_metadata(
        self, onnx_model_dir: Path, tmp_path: Path
    ) -> None:
        """ModelLoadError when sidecar features list differs from metadata features."""
        import shutil

        shutil.copy(
            onnx_model_dir / "final_logistic_all.onnx", tmp_path / "final_logistic_all.onnx"
        )
        shutil.copy(
            onnx_model_dir / "final_logistic_all.json", tmp_path / "final_logistic_all.json"
        )
        # Write sidecar with a modified feature list
        sidecar = json.loads((onnx_model_dir / "final_logistic_all.onnx.json").read_text())
        bad_sidecar = dict(sidecar)
        bad_sidecar["features"] = list(reversed(sidecar["features"]))
        (tmp_path / "final_logistic_all.onnx.json").write_text(
            json.dumps(bad_sidecar), encoding="utf-8"
        )
        with pytest.raises(ModelLoadError, match="Sidecar features do not match"):
            load_onnx_model(tmp_path)

    def test_metadata_features_differ_from_schema(
        self, onnx_model_dir: Path, tmp_path: Path
    ) -> None:
        """ModelLoadError when metadata features don't match PredictRequest schema."""
        import shutil

        shutil.copy(
            onnx_model_dir / "final_logistic_all.onnx", tmp_path / "final_logistic_all.onnx"
        )
        shutil.copy(
            onnx_model_dir / "final_logistic_all.onnx.json",
            tmp_path / "final_logistic_all.onnx.json",
        )
        # Write metadata with a wrong feature list
        meta = json.loads((onnx_model_dir / "final_logistic_all.json").read_text())
        bad_meta = dict(meta)
        bad_features = [f for f in meta["features"] if f != "temp_mean"]
        bad_meta["features"] = bad_features
        bad_meta["n_features"] = len(bad_features)
        (tmp_path / "final_logistic_all.json").write_text(json.dumps(bad_meta), encoding="utf-8")
        with pytest.raises(ModelLoadError, match="features do not match"):
            load_onnx_model(tmp_path)


# ---------------------------------------------------------------------------
# (d) create_app backend selection
# ---------------------------------------------------------------------------


class TestCreateAppBackendSelection:
    """create_app backend resolution: arg > env > default."""

    def test_invalid_backend_raises_value_error(self) -> None:
        """create_app(backend='bogus') raises ValueError at call time."""
        with pytest.raises(ValueError, match="bogus"):
            create_app(backend="bogus")

    def test_env_onnx_selects_onnx_backend(
        self, monkeypatch: pytest.MonkeyPatch, onnx_model_dir: Path
    ) -> None:
        """STRESS_BACKEND=onnx env variable selects onnx backend."""
        monkeypatch.setenv("STRESS_BACKEND", "onnx")
        app = create_app(onnx_model_dir)
        with TestClient(app) as client:
            res = client.get("/model-info")
            assert res.json()["backend"] == "onnx"

    def test_unset_env_gives_sklearn(
        self, monkeypatch: pytest.MonkeyPatch, sklearn_model_dir: Path
    ) -> None:
        """Without STRESS_BACKEND env var the default backend is 'sklearn'."""
        monkeypatch.delenv("STRESS_BACKEND", raising=False)
        app = create_app(sklearn_model_dir)
        with TestClient(app) as client:
            res = client.get("/model-info")
            assert res.json()["backend"] == "sklearn"


# ---------------------------------------------------------------------------
# (e) Slim-import test: no sklearn/joblib/pandas/scipy at top-level
# ---------------------------------------------------------------------------

_SLIM_SCRIPT = """
import sys
# Block heavy ML libs at the module level
for name in list(sys.modules):
    for blocked in ("sklearn", "joblib", "pandas", "scipy"):
        if name == blocked or name.startswith(blocked + "."):
            sys.modules[name] = None  # type: ignore[assignment]

for blocked in ("sklearn", "joblib", "pandas", "scipy"):
    sys.modules[blocked] = None  # type: ignore[assignment]

import json, pathlib
model_dir = pathlib.Path(sys.argv[1])

from stress.api.main import create_app
from fastapi.testclient import TestClient

app = create_app(model_dir=model_dir, backend="onnx")
payload = json.loads(sys.argv[2])
with TestClient(app) as client:
    res = client.post("/predict", json=payload)
    print(res.status_code)
    print(res.json()["stress_probability"])
"""


class TestSlimImport:
    """ONNX backend works without sklearn/joblib/pandas/scipy at import time."""

    def test_slim_import_onnx_predict(self, onnx_model_dir: Path) -> None:
        """Subprocess blocks sklearn/joblib/pandas/scipy, then runs a /predict call."""
        payload = _make_synthetic_payload(seed=7, with_nulls=False)
        payload_json = json.dumps(payload)
        result = subprocess.run(
            [sys.executable, "-c", _SLIM_SCRIPT, str(onnx_model_dir), payload_json],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Slim-import subprocess failed:\nstdout={result.stdout}\nstderr={result.stderr}"
        )
        assert "200" in result.stdout, (
            f"Expected '200' in stdout, got:\n{result.stdout}\nstderr={result.stderr}"
        )
