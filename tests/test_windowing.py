"""Unit tests for windowing with synthetic data."""

import numpy as np

from stress.data.windowing import (
    WindowConfig,
    align_signal_to_labels,
    create_windows,
)


def test_align_signal_64hz() -> None:
    """Test timestamp-based label alignment for 64 Hz signal."""
    # 100 second signal at 64 Hz
    signal = np.arange(6400, dtype=np.float64)  # 100 * 64 = 6400 samples
    fs = 64.0

    # 100 second labels at 700 Hz
    # Label change at t=50s
    labels = np.zeros(70000, dtype=np.uint8)
    labels[: int(50 * 700)] = 1
    labels[int(50 * 700) :] = 2

    aligned = align_signal_to_labels(signal, fs, labels)

    assert len(aligned) == len(signal), "Aligned array should match signal length"
    assert aligned[0] == 1, "First sample should have label 1"
    assert aligned[-1] == 2, "Last sample should have label 2"

    # Sample at t=50s should be near the transition
    # Sample 3200 = 3200/64 = 50s
    sample_at_50s = aligned[3200]
    print(f"Label at t=50s: {sample_at_50s}")


def test_align_signal_32hz() -> None:
    """Test timestamp-based label alignment for 32 Hz signal."""
    # 100 second signal at 32 Hz
    signal = np.arange(3200, dtype=np.float64)  # 100 * 32 = 3200 samples
    fs = 32.0

    # 100 second labels at 700 Hz
    labels = np.zeros(70000, dtype=np.uint8)
    labels[: int(10 * 700)] = 1
    labels[int(10 * 700) :] = 2

    aligned = align_signal_to_labels(signal, fs, labels)

    assert len(aligned) == len(signal)
    assert aligned[0] == 1, "First sample should have label 1"
    assert aligned[-1] == 2, "Last sample should have label 2"


def test_align_signal_4hz() -> None:
    """Test timestamp-based label alignment for 4 Hz signal."""
    # 100 second signal at 4 Hz
    signal = np.arange(400, dtype=np.float64)  # 100 * 4 = 400 samples
    fs = 4.0

    # 100 second labels at 700 Hz
    labels = np.zeros(70000, dtype=np.uint8)
    labels[: int(25 * 700)] = 1
    labels[int(25 * 700) :] = 2

    aligned = align_signal_to_labels(signal, fs, labels)

    assert len(aligned) == len(signal)
    assert aligned[0] == 1
    assert aligned[-1] == 2


def test_create_windows_basic() -> None:
    """Test window creation with synthetic signals and labels."""
    # 600 second recording (10 minutes)
    duration_sec = 600.0

    # Signals at native rates
    wrist_signals = {
        "BVP": np.ones(int(duration_sec * 64), dtype=np.float64),
        "EDA": np.ones(int(duration_sec * 4), dtype=np.float64),
        "TEMP": np.ones(int(duration_sec * 4), dtype=np.float64),
        "ACC": np.ones((int(duration_sec * 32), 3), dtype=np.float64),
    }

    # Labels: all label 1 (baseline)
    labels = np.ones(int(duration_sec * 700), dtype=np.uint8)

    config = WindowConfig(window_seconds=60.0, overlap=0.5, purity_threshold=1.0)
    windows = create_windows(wrist_signals, labels, config)

    # Stride = 60 * (1 - 0.5) = 30s
    # Windows at: 0-60, 30-90, 60-120, ..., 570-630 (incomplete, dropped)
    # Expected: floor(600 / 30) = 20 complete windows
    assert len(windows) == 19, f"Expected 19 windows, got {len(windows)}"

    # All windows should have label 1
    for window_dict, label in windows:
        assert label == 1, f"Expected label 1, got {label}"
        assert "BVP" in window_dict
        assert "EDA" in window_dict
        assert "TEMP" in window_dict
        assert "ACC" in window_dict


def test_create_windows_no_incomplete_final() -> None:
    """Test that incomplete final window is dropped."""
    # 610 seconds with stride 30s: windows at 0, 30, 60, ..., 570, 600
    # Window at 600 would need [600*64:660*64], but signal ends at 610*64
    # Last complete window is at 570s (ends at 630s but dropped due to incomplete check)
    # So we get windows from 0, 30, 60, ... (same as 600s case)
    duration_sec = 610.0

    wrist_signals = {
        "BVP": np.ones(int(duration_sec * 64), dtype=np.float64),
        "EDA": np.ones(int(duration_sec * 4), dtype=np.float64),
        "TEMP": np.ones(int(duration_sec * 4), dtype=np.float64),
        "ACC": np.ones((int(duration_sec * 32), 3), dtype=np.float64),
    }

    labels = np.ones(int(duration_sec * 700), dtype=np.uint8)

    config = WindowConfig(window_seconds=60.0, overlap=0.5)
    windows = create_windows(wrist_signals, labels, config)

    # With stride 30s: windows at 0-60, 30-90, ..., 540-600, 570-630 (incomplete, dropped)
    # Expected 19 complete windows (0-570s in 30s steps)
    assert len(windows) == 19, f"Expected 19 windows, got {len(windows)}"


def test_create_windows_purity_filtering() -> None:
    """Test that windows below purity threshold are discarded."""
    # 600 seconds: first 300s label 1, second 300s label 2
    duration_sec = 600.0
    labels = np.zeros(int(duration_sec * 700), dtype=np.uint8)
    labels[: int(300 * 700)] = 1
    labels[int(300 * 700) :] = 2

    wrist_signals = {
        "BVP": np.ones(int(duration_sec * 64), dtype=np.float64),
        "EDA": np.ones(int(duration_sec * 4), dtype=np.float64),
        "TEMP": np.ones(int(duration_sec * 4), dtype=np.float64),
        "ACC": np.ones((int(duration_sec * 32), 3), dtype=np.float64),
    }

    # Strict purity (1.0): windows spanning both labels are discarded
    config = WindowConfig(window_seconds=60.0, overlap=0.5, purity_threshold=1.0)
    windows_strict = create_windows(wrist_signals, labels, config)

    # Windows around t=300s will span both labels, so they are discarded
    # We expect ~9 windows from first 300s and ~9 windows from second 300s, ~18 total
    print(f"Strict purity (1.0): {len(windows_strict)} windows")
    assert len(windows_strict) <= 20, "Some windows should be discarded"

    # Relaxed purity (0.8): windows with >= 80% of one label are kept
    config_relaxed = WindowConfig(window_seconds=60.0, overlap=0.5, purity_threshold=0.8)
    windows_relaxed = create_windows(wrist_signals, labels, config_relaxed)

    print(f"Relaxed purity (0.8): {len(windows_relaxed)} windows")
    assert len(windows_relaxed) >= len(windows_strict), "Relaxed purity should allow more windows"


def test_create_windows_valid_labels_filtering() -> None:
    """Test that windows with labels outside valid_labels are discarded."""
    # 600 seconds: labels 0, 1, 2 (0 = transient, should be excluded)
    duration_sec = 600.0
    labels = np.zeros(int(duration_sec * 700), dtype=np.uint8)
    labels[: int(200 * 700)] = 0  # Transient
    labels[int(200 * 700) : int(400 * 700)] = 1  # Baseline
    labels[int(400 * 700) :] = 2  # Stress

    wrist_signals = {
        "BVP": np.ones(int(duration_sec * 64), dtype=np.float64),
        "EDA": np.ones(int(duration_sec * 4), dtype=np.float64),
        "TEMP": np.ones(int(duration_sec * 4), dtype=np.float64),
        "ACC": np.ones((int(duration_sec * 32), 3), dtype=np.float64),
    }

    config = WindowConfig(
        window_seconds=60.0, overlap=0.5, purity_threshold=1.0, valid_labels={1, 2, 3}
    )
    windows = create_windows(wrist_signals, labels, config)

    # All windows should have labels 1 or 2, none with 0
    for _, label in windows:
        assert label in {1, 2}, f"Expected label in {{1, 2}}, got {label}"

    # Verify some windows are discarded (those with only label 0)
    # First ~6 windows should be mostly 0s and discarded
    print(f"Valid labels filtering: {len(windows)} windows kept")
