"""Unit tests for feature extraction."""

from pathlib import Path

import numpy as np
import pytest

from stress.features.extract import (
    compute_hrv_from_ibi,
    extract_acc_features,
    extract_eda_features,
    extract_features_all_subjects,
    extract_hrv_features,
    extract_temp_features,
)


def test_extract_hrv_features_sine_wave() -> None:
    """Test HRV extraction on a 1.2 Hz sine wave (should give ~72 bpm HR)."""
    fs = 64.0
    duration_sec = 60.0
    t = np.arange(int(duration_sec * fs)) / fs

    # 1.2 Hz sine = 1.2 * 60 = 72 bpm (expected HR)
    bvp = np.sin(2 * np.pi * 1.2 * t)

    features = extract_hrv_features(bvp, fs=fs)

    # Check that we got meaningful output
    assert features["num_beats"] >= 60, "Should detect ~72 beats in 60s"
    assert 65 < features["mean_hr"] < 85, f"HR should be ~72 bpm, got {features['mean_hr']:.1f}"
    assert features["hr_std"] < 10, "HR std should be low for sine wave"


def test_extract_hrv_features_flat_signal() -> None:
    """Test HRV extraction on a flat signal (no beats)."""
    bvp = np.ones(3840)  # 60s at 64 Hz
    features = extract_hrv_features(bvp, fs=64.0)

    # Flat signal has no peaks
    assert features["num_beats"] < 5, "Flat signal should have few beats"
    assert np.isnan(features["mean_hr"]), "Should be NaN when < 10 beats"
    assert np.isnan(features["hr_std"]), "Should be NaN when < 10 beats"
    assert np.isnan(features["sdnn"]), "Should be NaN when < 10 beats"


def test_extract_eda_features_flat() -> None:
    """Test EDA extraction on flat signal."""
    eda = np.ones(240)  # 60s at 4 Hz
    features = extract_eda_features(eda, fs=4.0)

    assert features["eda_mean"] == 1.0
    assert features["eda_std"] == 0.0
    assert features["eda_slope"] == pytest.approx(0.0, abs=1e-9)
    assert features["eda_num_peaks"] == 0


def test_extract_eda_features_ramp() -> None:
    """Test EDA extraction on a linear ramp (positive slope)."""
    eda = np.linspace(0, 10, 240)  # 60s at 4 Hz, linear from 0 to 10
    features = extract_eda_features(eda, fs=4.0)

    assert features["eda_mean"] == pytest.approx(5.0, abs=0.1)
    assert features["eda_slope"] > 0, "Positive ramp should have positive slope"


def test_extract_temp_features_flat() -> None:
    """Test TEMP extraction on flat signal."""
    temp = np.ones(240) * 36.5  # 60s at 4 Hz, constant 36.5°C
    features = extract_temp_features(temp, fs=4.0)

    assert features["temp_mean"] == pytest.approx(36.5, abs=0.01)
    assert features["temp_std"] == 0.0
    assert features["temp_slope"] == pytest.approx(0.0, abs=1e-9)


def test_extract_acc_features_stationary() -> None:
    """Test ACC extraction on stationary acceleration (no motion).

    Nx3 accelerometer in E4 raw counts: 1 g = 64 counts, so gravity is 64 counts.
    """
    acc = np.zeros((1920, 3))  # 60s at 32 Hz
    acc[:, 2] = 64.0  # Z-axis = 1 g in E4 counts

    features = extract_acc_features(acc, fs=32.0)

    # Vector magnitude should be ~1.0 g
    assert features["acc_magnitude_mean"] == pytest.approx(1.0, abs=0.01)
    assert features["acc_magnitude_std"] == pytest.approx(0.0, abs=0.01)


def test_extract_acc_features_moving() -> None:
    """Test ACC extraction on moving acceleration (input counts, output g)."""
    # Sinusoidal acceleration on X-axis: 0.5 g oscillation at 0.5 Hz, in E4 counts
    t = np.arange(1920) / 32.0
    acc = np.zeros((1920, 3))
    acc[:, 0] = 0.5 * np.sin(2 * np.pi * 0.5 * t) * 64.0
    acc[:, 2] = 64.0  # 1 g gravity on Z

    features = extract_acc_features(acc, fs=32.0)

    # Magnitude should vary and include gravity
    assert features["acc_magnitude_std"] > 0.01, "Moving signal should have variance"
    assert features["acc_magnitude_mean"] > 0.9, "Mean magnitude should include gravity"


def test_extract_hrv_features_insufficient_beats() -> None:
    """Test that HRV features are NaN when fewer than 10 beats detected."""
    # Very low amplitude, noisy signal → few peaks detected
    bvp = np.random.randn(3840) * 0.01  # Small noise, no clear peaks
    features = extract_hrv_features(bvp, fs=64.0)

    if features["num_beats"] < 10:
        assert np.isnan(features["mean_hr"])
        assert np.isnan(features["sdnn"])
        assert np.isnan(features["rmssd"])
        assert np.isnan(features["pnn50"])


def test_compute_hrv_from_ibi_known_values() -> None:
    """Test HRV computation on hand-built IBI series with known RMSSD, SDNN, pNN50.

    IBI series: [0.8]*10 seconds (constant), giving 11 beats total
    - Mean IBI: 0.8 s -> Mean HR: 60/0.8 = 75 bpm
    - IBI std (SDNN): 0 ms (all intervals constant)
    - IBI diffs: all 0 ms
    - RMSSD: 0 ms
    - pNN50: 0 (no intervals differ by > 50 ms)
    """
    ibi = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8])  # 10 intervals = 11 beats
    features = compute_hrv_from_ibi(ibi)

    # Mean HR: 60 / 0.8 = 75 bpm
    assert features["mean_hr"] == pytest.approx(75.0, abs=0.1)

    # SDNN: std of IBI in ms (should be 0 for constant IBI)
    assert features["sdnn"] == pytest.approx(0.0, abs=1e-9)

    # RMSSD: sqrt(mean(diff(IBI_ms)^2)) (should be 0 for constant IBI)
    assert features["rmssd"] == pytest.approx(0.0, abs=1e-9)

    # pNN50: 0 (no differences)
    assert features["pnn50"] == pytest.approx(0.0, abs=1e-9)


def test_compute_hrv_from_ibi_insufficient() -> None:
    """Test that HRV returns NaN when < 10 intervals (< 11 beats)."""
    ibi = np.array([0.8, 0.8, 0.8, 0.8, 0.8])  # 5 intervals only
    features = compute_hrv_from_ibi(ibi)

    assert np.isnan(features["mean_hr"])
    assert np.isnan(features["sdnn"])
    assert np.isnan(features["rmssd"])
    assert np.isnan(features["pnn50"])


def test_extract_features_all_subjects_writes_true_window_starts(tmp_path: Path) -> None:
    """window_start_s must be the true grid time, not the index of kept windows.

    Structure: 200s label 1, 200s label 0 (gap), 200s label 1 -> 10 kept windows.
    Their start times are 0..120s and 420..540s, not 0..270s.
    """
    duration_sec = 600.0
    labels = np.zeros(int(duration_sec * 700), dtype=np.uint8)
    labels[: int(200 * 700)] = 1
    labels[int(200 * 700) : int(400 * 700)] = 0
    labels[int(400 * 700) :] = 1

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    np.savez_compressed(
        processed_dir / "S1.npz",
        BVP=np.ones(int(duration_sec * 64), dtype=np.float32),
        EDA=np.ones(int(duration_sec * 4), dtype=np.float32),
        TEMP=np.ones(int(duration_sec * 4), dtype=np.float32),
        ACC=np.ones((int(duration_sec * 32), 3), dtype=np.float32),
        label=labels,
    )

    # discover_subjects only needs the file to exist; signals come from the .npz
    raw_dir = tmp_path / "raw" / "WESAD" / "S1"
    raw_dir.mkdir(parents=True)
    (raw_dir / "S1.pkl").touch()

    df = extract_features_all_subjects(str(processed_dir), str(tmp_path / "features.csv"))

    assert df["subject"].tolist() == [1] * 10
    assert df["window_start_s"].tolist() == [
        0.0,
        30.0,
        60.0,
        90.0,
        120.0,
        420.0,
        450.0,
        480.0,
        510.0,
        540.0,
    ]
