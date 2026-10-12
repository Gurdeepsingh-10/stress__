"""Unit tests for WESAD loader with synthetic data."""

from pathlib import Path

import numpy as np
import pytest

from stress.data.wesad_loader import discover_subjects, load_subject


def create_mock_pickle(tmp_path: Path, subject_id: int) -> Path:
    """Create a mock S{subject_id}.pkl file for testing.

    Structure matches real WESAD pickles: data['signal']['wrist'] and data['label'].
    """
    import pickle

    pkl_dir = tmp_path / f"S{subject_id}"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = pkl_dir / f"S{subject_id}.pkl"

    # Synthetic data: 6079 second recording (matches S2 from inspection)
    duration_sec = 6079.0

    data = {
        "subject": subject_id,
        "signal": {
            "wrist": {
                "BVP": np.random.randn(int(duration_sec * 64), 1).astype(np.float64),
                "EDA": np.random.randn(int(duration_sec * 4), 1).astype(np.float64),
                "TEMP": np.random.randn(int(duration_sec * 4), 1).astype(np.float64),
                "ACC": np.random.randn(int(duration_sec * 32), 3).astype(np.float64),
            },
            "chest": {
                "ECG": np.random.randn(int(duration_sec * 700)).astype(np.float64),
            },
        },
        "label": np.random.randint(0, 5, int(duration_sec * 700), dtype=np.uint8),
    }

    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)

    return pkl_file


def test_loader_flattens_signals(tmp_path: Path) -> None:
    """Test that (N, 1) signals are flattened to 1D."""
    create_mock_pickle(tmp_path, 2)
    result = load_subject(2, tmp_path)

    assert result.wrist["BVP"].ndim == 1, "BVP should be 1D after flattening"
    assert result.wrist["EDA"].ndim == 1, "EDA should be 1D after flattening"
    assert result.wrist["TEMP"].ndim == 1, "TEMP should be 1D after flattening"
    assert result.wrist["ACC"].ndim == 2, "ACC should remain 2D"
    assert result.wrist["ACC"].shape[1] == 3, "ACC should have 3 columns"


def test_loader_validates_duration(tmp_path: Path) -> None:
    """Test that duration mismatches are detected and reported."""
    # Create pickle with intentional duration mismatch
    import pickle

    pkl_dir = tmp_path / "S2"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = pkl_dir / "S2.pkl"

    duration_sec = 6079.0
    data = {
        "subject": 2,
        "signal": {
            "wrist": {
                "BVP": np.random.randn(int(duration_sec * 64), 1).astype(np.float64),
                "EDA": np.random.randn(int(duration_sec * 4) + 10, 1).astype(
                    np.float64
                ),  # Mismatch!
                "TEMP": np.random.randn(int(duration_sec * 4), 1).astype(np.float64),
                "ACC": np.random.randn(int(duration_sec * 32), 3).astype(np.float64),
            },
            "chest": {},
        },
        "label": np.random.randint(0, 5, int(duration_sec * 700), dtype=np.uint8),
    }

    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)

    result = load_subject(2, tmp_path, tolerance_sec=0.1)

    assert result.duration_anomaly, "Should flag duration anomaly"
    assert "EDA" in result.anomaly_message, "EDA mismatch should be reported"
    print(result.anomaly_message)


def test_loader_validates_label_alignment_64hz(tmp_path: Path) -> None:
    """Test timestamp-based label alignment for 64 Hz signal."""
    import pickle

    pkl_dir = tmp_path / "S2"
    pkl_dir.mkdir(parents=True, exist_ok=True)
    pkl_file = pkl_dir / "S2.pkl"

    # 100 second recording for easy calculation
    duration_sec = 100.0

    # Create synthetic labels: change at t=50s
    labels = np.zeros(int(duration_sec * 700), dtype=np.uint8)
    labels[: int(50 * 700)] = 1
    labels[int(50 * 700) :] = 2

    data = {
        "subject": 2,
        "signal": {
            "wrist": {
                "BVP": np.random.randn(int(duration_sec * 64), 1).astype(np.float64),
                "EDA": np.random.randn(int(duration_sec * 4), 1).astype(np.float64),
                "TEMP": np.random.randn(int(duration_sec * 4), 1).astype(np.float64),
                "ACC": np.random.randn(int(duration_sec * 32), 3).astype(np.float64),
            },
            "chest": {},
        },
        "label": labels,
    }

    with open(pkl_file, "wb") as f:
        pickle.dump(data, f)

    result = load_subject(2, tmp_path)
    assert not result.duration_anomaly

    # Verify label array is correct
    assert result.label[0] == 1, "First label should be 1"
    assert result.label[-1] == 2, "Last label should be 2"


def test_discover_subjects(tmp_path: Path) -> None:
    """Test subject discovery by scanning for S*.pkl files."""
    # Create mock pickles for S2, S3, S5 (skip S4)
    for subject_id in [2, 3, 5]:
        create_mock_pickle(tmp_path, subject_id)

    subjects = discover_subjects(tmp_path)

    assert subjects == [2, 3, 5], f"Expected [2, 3, 5], got {subjects}"


def test_loader_file_not_found(tmp_path: Path) -> None:
    """Test that FileNotFoundError is raised for missing pickle."""
    with pytest.raises(FileNotFoundError):
        load_subject(99, tmp_path)
