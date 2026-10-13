"""Unit tests for report_windows script."""

from pathlib import Path

import numpy as np

from scripts.report_windows import report_windows


def test_report_windows_synthetic_baseline_only(tmp_path: Path, capsys) -> None:
    """Test report with 600s of label 1 (baseline) gives exactly 19 windows."""
    # Create synthetic npz file: 600s of label 1
    npz_file = tmp_path / "S1.npz"

    duration_sec = 600.0
    bvp = np.ones(int(duration_sec * 64), dtype=np.float32)
    eda = np.ones(int(duration_sec * 4), dtype=np.float32)
    temp = np.ones(int(duration_sec * 4), dtype=np.float32)
    acc = np.ones((int(duration_sec * 32), 3), dtype=np.float32)
    label = np.ones(int(duration_sec * 700), dtype=np.uint8)  # All label 1

    np.savez_compressed(npz_file, BVP=bvp, EDA=eda, TEMP=temp, ACC=acc, label=label)

    # Run report with defaults (60s windows, 50% overlap, purity 1.0)
    exit_code = report_windows(tmp_path)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "S1" in captured.out
    assert "19" in captured.out  # 19 baseline windows
    assert "0" in captured.out  # 0 stress, 0 amusement


def test_report_windows_empty_directory(tmp_path: Path, capsys) -> None:
    """Test report returns 1 when no .npz files found."""
    exit_code = report_windows(tmp_path)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "ERROR: no .npz files found" in captured.err


def test_report_windows_purity_threshold(tmp_path: Path, capsys) -> None:
    """Test that purity_threshold parameter is respected.

    Create a 600s signal with label 1, except for a 6s gap of label 0 at [300, 306)s.
    With 60s windows at 50% overlap (30s stride):
    - Windows starting at 270s and 300s have purity 0.9 (54s label 1, 6s label 0)
    - At purity_threshold=1.0 (strict): both windows are dropped → 17 total windows
    - At purity_threshold=0.8 (relaxed): both windows are kept → 19 total windows
    """
    npz_file = tmp_path / "S2.npz"

    duration_sec = 600.0
    labels = np.ones(int(duration_sec * 700), dtype=np.uint8)
    # Gap from 300-306s (6 seconds)
    labels[int(300 * 700) : int(306 * 700)] = 0

    bvp = np.ones(int(duration_sec * 64), dtype=np.float32)
    eda = np.ones(int(duration_sec * 4), dtype=np.float32)
    temp = np.ones(int(duration_sec * 4), dtype=np.float32)
    acc = np.ones((int(duration_sec * 32), 3), dtype=np.float32)

    np.savez_compressed(npz_file, BVP=bvp, EDA=eda, TEMP=temp, ACC=acc, label=labels)

    # Test strict purity (1.0)
    exit_code_strict = report_windows(tmp_path, purity_threshold=1.0)
    assert exit_code_strict == 0
    captured_strict = capsys.readouterr()
    # Extract the total windows count from the TOTAL row
    total_lines_strict = [
        line for line in captured_strict.out.split("\n") if line.startswith("TOTAL")
    ]
    assert len(total_lines_strict) > 0
    parts_strict = total_lines_strict[0].split()
    total_strict = int(parts_strict[-1])

    # Test relaxed purity (0.8)
    exit_code_relaxed = report_windows(tmp_path, purity_threshold=0.8)
    assert exit_code_relaxed == 0
    captured_relaxed = capsys.readouterr()
    total_lines_relaxed = [
        line for line in captured_relaxed.out.split("\n") if line.startswith("TOTAL")
    ]
    assert len(total_lines_relaxed) > 0
    parts_relaxed = total_lines_relaxed[0].split()
    total_relaxed = int(parts_relaxed[-1])

    # Strict purity should drop windows with purity 0.9
    assert total_strict == 17, f"Expected 17 windows at purity=1.0, got {total_strict}"
    # Relaxed purity should keep those windows
    assert total_relaxed == 19, f"Expected 19 windows at purity=0.8, got {total_relaxed}"
