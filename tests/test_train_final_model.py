"""Tests for scripts/train_final_model.py using synthetic DataFrames only."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from scripts.train_final_model import (
    METADATA_FILENAME,
    MODEL_FILENAME,
    build_metadata,
    get_git_info,
    load_training_frame,
    train_and_save,
)
from stress.api.schemas import PredictRequest
from stress.models.loso import build_pipeline

SCHEMA_FEATURES = list(PredictRequest.model_fields.keys())


def _make_synthetic_df(n_samples: int = 30, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic training DataFrame."""
    rng = np.random.RandomState(seed)
    data: dict[str, Any] = {}
    for feat in SCHEMA_FEATURES:
        data[feat] = rng.uniform(1.0, 50.0, size=n_samples)
    data["label_binary"] = rng.choice([0, 1], size=n_samples)
    return pd.DataFrame(data)


def test_train_and_save_creates_valid_artifacts(tmp_path: Path) -> None:
    """train_and_save persists joblib and JSON with complete metadata and class counts."""
    df = _make_synthetic_df(n_samples=50, seed=7)
    counts = df["label_binary"].value_counts().to_dict()

    injected_git = ("test_commit_hash_123", False)
    joblib_path, json_path = train_and_save(
        df=df,
        out_dir=tmp_path,
        git_info=injected_git,
    )

    assert joblib_path.is_file()
    assert json_path.is_file()
    assert joblib_path.name == MODEL_FILENAME
    assert json_path.name == METADATA_FILENAME

    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    # Check required metadata fields
    expected_keys = {
        "model_name",
        "model_version",
        "target",
        "features",
        "n_features",
        "training_samples",
        "class_counts",
        "classes_",
        "trained_at_utc",
        "git_commit",
        "git_dirty",
        "environment",
    }
    assert expected_keys.issubset(meta.keys())
    assert meta["model_name"] == "logistic_all"
    assert meta["model_version"] == "1.0.0"
    assert meta["target"] == "label_binary"
    assert meta["features"] == SCHEMA_FEATURES
    assert meta["n_features"] == 17
    assert meta["training_samples"] == 50
    assert meta["classes_"] == [0, 1]
    assert meta["git_commit"] == "test_commit_hash_123"
    assert meta["git_dirty"] is False
    assert meta["class_counts"] == {str(k): v for k, v in counts.items()}

    env = meta["environment"]
    for pkg in ("python", "scikit-learn", "numpy", "joblib"):
        assert pkg in env and isinstance(env[pkg], str)

    pipeline = joblib.load(joblib_path)
    assert list(pipeline.feature_names_in_) == meta["features"]


def test_train_and_save_pipeline_feature_names_matches_meta(tmp_path: Path) -> None:
    """After train_and_save, pipeline.feature_names_in_ matches meta['features']."""
    df = _make_synthetic_df(n_samples=20)
    joblib_path, json_path = train_and_save(
        df=df,
        out_dir=tmp_path,
        git_info=("commit_abc", False),
    )
    with json_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    pipeline = joblib.load(joblib_path)
    assert list(pipeline.feature_names_in_) == meta["features"]


def test_train_and_save_missing_feature_column_raises(tmp_path: Path) -> None:
    """train_and_save raises ValueError when a schema feature column is missing."""
    df = _make_synthetic_df()
    df_missing = df.drop(columns=["temp_mean"])
    with pytest.raises(ValueError, match="Missing required feature columns"):
        train_and_save(df=df_missing, out_dir=tmp_path, git_info=("test", False))


def test_train_and_save_missing_target_column_raises(tmp_path: Path) -> None:
    """train_and_save raises ValueError when 'label_binary' column is missing."""
    df = _make_synthetic_df()
    df_no_target = df.drop(columns=["label_binary"])
    with pytest.raises(ValueError, match="Missing target column"):
        train_and_save(df=df_no_target, out_dir=tmp_path, git_info=("test", False))


def test_get_git_info_returns_unknown_on_subprocess_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_git_info returns ('unknown', None) without crashing if subprocess fails."""

    def mock_run(*args: Any, **kwargs: Any) -> Any:
        raise subprocess.SubprocessError("git not installed")

    monkeypatch.setattr(subprocess, "run", mock_run)
    commit, is_dirty = get_git_info()
    assert commit == "unknown"
    assert is_dirty is None


def test_load_training_frame_valid(tmp_path: Path) -> None:
    """load_training_frame loads CSV when all schema features and target are present."""
    csv_file = tmp_path / "valid.csv"
    df = _make_synthetic_df(n_samples=20)
    df.to_csv(csv_file, index=False)

    loaded_df = load_training_frame(csv_file)
    assert len(loaded_df) == 20
    assert set(SCHEMA_FEATURES).issubset(loaded_df.columns)
    assert "label_binary" in loaded_df.columns


def test_load_training_frame_missing_file(tmp_path: Path) -> None:
    """load_training_frame raises FileNotFoundError when CSV does not exist."""
    missing_csv = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError, match="Feature CSV not found"):
        load_training_frame(missing_csv)


def test_load_training_frame_missing_columns(tmp_path: Path) -> None:
    """load_training_frame raises ValueError when schema features are missing."""
    csv_file = tmp_path / "incomplete.csv"
    df = _make_synthetic_df()
    df.drop(columns=["eda_mean"]).to_csv(csv_file, index=False)

    with pytest.raises(ValueError, match="Missing 1 required feature columns"):
        load_training_frame(csv_file)


def test_build_metadata_structure() -> None:
    """build_metadata returns expected structure and captures provided provenance."""
    df = _make_synthetic_df(n_samples=10)
    pipe = build_pipeline("logistic", seed=42)
    pipe.fit(df[SCHEMA_FEATURES], df["label_binary"])
    meta = build_metadata(
        pipeline=pipe,
        df=df,
        git_info=("custom_commit", True),
        trained_at_utc="2026-10-21T12:00:00Z",
    )
    assert meta["git_commit"] == "custom_commit"
    assert meta["git_dirty"] is True
    assert meta["trained_at_utc"] == "2026-10-21T12:00:00Z"
    assert meta["training_samples"] == 10
