"""Unit tests for preprocessing module."""

from pathlib import Path

import numpy as np
import pytest

from stress.data.preprocessing import load_npz_subject


def test_load_npz_subject_round_trip(tmp_path: Path) -> None:
    """Test that load_npz_subject correctly reads a synthesized .npz file."""
    # Create a tiny synthetic npz
    subject_id = 99
    npz_file = tmp_path / "S99.npz"

    rng = np.random.default_rng(0)
    bvp = rng.standard_normal(3840).astype(np.float32)  # 60s at 64 Hz
    eda = rng.standard_normal(240).astype(np.float32)  # 60s at 4 Hz
    temp = rng.standard_normal(240).astype(np.float32)  # 60s at 4 Hz
    acc = rng.standard_normal((1920, 3)).astype(np.float32)  # 60s at 32 Hz
    label = np.ones(42000, dtype=np.uint8)  # 60s at 700 Hz

    np.savez_compressed(npz_file, BVP=bvp, EDA=eda, TEMP=temp, ACC=acc, label=label)

    # Load and verify
    result = load_npz_subject(subject_id, tmp_path)

    assert "wrist" in result
    assert "label" in result
    assert set(result["wrist"].keys()) == {"BVP", "EDA", "TEMP", "ACC"}
    assert np.array_equal(result["wrist"]["BVP"], bvp)
    assert np.array_equal(result["wrist"]["EDA"], eda)
    assert np.array_equal(result["wrist"]["TEMP"], temp)
    assert np.array_equal(result["wrist"]["ACC"], acc)
    assert np.array_equal(result["label"], label)


def test_load_npz_subject_missing_file(tmp_path: Path) -> None:
    """Test that load_npz_subject raises FileNotFoundError for missing file."""
    subject_id = 88
    with pytest.raises(FileNotFoundError, match="NPZ file not found"):
        load_npz_subject(subject_id, tmp_path)
