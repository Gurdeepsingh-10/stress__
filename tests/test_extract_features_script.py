"""Unit tests for the extract_features script."""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.extract_features import class_means, main, sanity_checks


def _table(stress_higher: bool) -> pd.DataFrame:
    """Small feature table where stress either beats baseline or does not."""
    if stress_higher:
        hr = [70.0, 72.0, 84.0, 86.0]
        eda = [1.0, 1.0, 3.0, 3.0]
    else:
        hr = [70.0, 72.0, 60.0, 62.0]
        eda = [3.0, 3.0, 1.0, 1.0]
    return pd.DataFrame(
        {
            "label_3class": [1, 1, 2, 2],
            "hrv_mean_hr": hr,
            "eda_mean": eda,
        }
    )


def test_class_means_per_label() -> None:
    """Per-class means are computed per label code."""
    means = class_means(_table(stress_higher=True))

    assert means.loc[1, "hrv_mean_hr"] == 71.0
    assert means.loc[2, "hrv_mean_hr"] == 85.0
    assert means.loc[1, "eda_mean"] == 1.0
    assert means.loc[2, "eda_mean"] == 3.0


def test_sanity_checks_pass_when_stress_higher() -> None:
    """Both expectations pass when stress HR and EDA exceed baseline."""
    results = sanity_checks(_table(stress_higher=True))

    assert [passed for _, passed in results] == [True, True]
    assert "-> PASS" in results[0][0]


def test_sanity_checks_fail_when_stress_lower() -> None:
    """Both expectations fail when stress HR and EDA are below baseline."""
    results = sanity_checks(_table(stress_higher=False))

    assert [passed for _, passed in results] == [False, False]
    assert "-> FAIL" in results[0][0]


def test_sanity_checks_missing_class() -> None:
    """A missing class makes the check fail rather than silently pass."""
    df = pd.DataFrame({"label_3class": [1, 1], "hrv_mean_hr": [70.0, 72.0], "eda_mean": [1.0, 1.0]})

    results = sanity_checks(df)

    assert all(passed is False for _, passed in results)


def test_main_end_to_end_on_synthetic_subject(tmp_path: Path, capsys) -> None:
    """main() writes the table and passes when stress shows higher HR and EDA.

    Signals: baseline 300s at 1.0 Hz (60 bpm), stress 300s at 1.4 Hz (84 bpm);
    EDA constant 1.0 uS then 3.0 uS. With 60s/50% windows and purity 1.0 that is
    9 baseline + 9 stress windows.
    """
    duration_sec = 600.0
    split_sec = 300.0
    t = np.arange(int(duration_sec * 64)) / 64.0

    bvp = np.where(t < split_sec, np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 1.4 * t))
    eda = np.where(t[::16] < split_sec, 1.0, 3.0).astype(np.float32)

    labels = np.ones(int(duration_sec * 700), dtype=np.uint8)
    labels[int(split_sec * 700) :] = 2

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    np.savez_compressed(
        processed_dir / "S1.npz",
        BVP=bvp.astype(np.float32),
        EDA=eda,
        TEMP=np.ones(int(duration_sec * 4), dtype=np.float32),
        ACC=np.ones((int(duration_sec * 32), 3), dtype=np.float32),
        label=labels,
    )

    # discover_subjects only needs the file to exist; signals come from the .npz
    raw_dir = tmp_path / "raw" / "WESAD" / "S1"
    raw_dir.mkdir(parents=True)
    (raw_dir / "S1.pkl").touch()

    out_csv = tmp_path / "features.csv"
    exit_code = main(["--processed-dir", str(processed_dir), "--out", str(out_csv)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert out_csv.exists()
    assert "SANITY CHECK PASSED" in captured.out

    df = pd.read_csv(out_csv)
    assert len(df) == 18
    assert df["label_3class"].value_counts().to_dict() == {1: 9, 2: 9}
