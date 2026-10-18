"""Unit tests for the run_loso script (scripts/run_loso.py)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from scripts.run_loso import load_features, log_loso_result, main
from stress.models.loso import LOSOResult

# ---------------------------------------------------------------------------
# Canonical feature columns (must match FEATURE_GROUPS)
# ---------------------------------------------------------------------------
_FEATURE_COLS: list[str] = [
    "hrv_mean_hr",
    "hrv_hr_std",
    "hrv_sdnn",
    "hrv_rmssd",
    "hrv_pnn50",
    "hrv_num_beats",
    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",
    "eda_slope",
    "eda_num_peaks",
    "temp_mean",
    "temp_std",
    "temp_slope",
    "acc_magnitude_mean",
    "acc_magnitude_std",
]


def _make_feature_csv(path: Path, n_subjects: int = 3, windows_per_subject: int = 8) -> Path:
    """Write a minimal synthetic feature CSV for script integration tests.

    Args:
        path: File path to write (must end in .csv).
        n_subjects: Number of subjects.
        windows_per_subject: Windows per subject.

    Returns:
        The same path after writing.
    """
    rng = np.random.default_rng(42)
    rows = []
    for sid in range(1, n_subjects + 1):
        for w in range(windows_per_subject):
            row: dict = {
                "subject": sid,
                "window_start_s": float(w * 30),
                "label_3class": 1 if w % 2 == 0 else 2,
                "label_binary": w % 2,
            }
            for col in _FEATURE_COLS:
                row[col] = float(rng.standard_normal())
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _make_feature_parquet(path: Path, n_subjects: int = 3, windows_per_subject: int = 8) -> Path:
    """Write a minimal synthetic feature Parquet for format tests."""
    rng = np.random.default_rng(1)
    rows = []
    for sid in range(1, n_subjects + 1):
        for w in range(windows_per_subject):
            row: dict = {
                "subject": sid,
                "window_start_s": float(w * 30),
                "label_3class": 1 if w % 2 == 0 else 2,
                "label_binary": w % 2,
            }
            for col in _FEATURE_COLS:
                row[col] = float(rng.standard_normal())
            rows.append(row)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


# ===========================================================================
# load_features
# ===========================================================================


def test_load_features_csv(tmp_path: Path) -> None:
    """load_features reads a CSV and returns a DataFrame with the expected columns."""
    csv_path = _make_feature_csv(tmp_path / "features.csv")

    df = load_features(csv_path)

    assert isinstance(df, pd.DataFrame)
    assert "subject" in df.columns
    assert "label_binary" in df.columns


def test_load_features_parquet(tmp_path: Path) -> None:
    """load_features reads a Parquet file and returns a DataFrame."""
    parquet_path = _make_feature_parquet(tmp_path / "features.parquet")

    df = load_features(parquet_path)

    assert isinstance(df, pd.DataFrame)
    assert "subject" in df.columns


def test_load_features_unsupported_extension_raises(tmp_path: Path) -> None:
    """An unsupported extension (.txt) must raise ValueError, not silently fail."""
    txt_path = tmp_path / "features.txt"
    txt_path.write_text("subject,label_binary\n1,0\n")

    with pytest.raises(ValueError, match="Unsupported feature file format"):
        load_features(txt_path)


# ===========================================================================
# log_loso_result
# ===========================================================================


def test_log_loso_result_logs_all_params_and_metrics(tmp_path: Path) -> None:
    """log_loso_result must record model, feature_group, macro_f1, and per-class recalls."""
    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow_test.db'}"
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("test_log_result")

    # Build a minimal synthetic LOSOResult
    result = LOSOResult(
        model_name="logistic",
        feature_group="hrv",
        macro_f1=0.72,
        per_class_recall={0: 0.80, 1: 0.64},
        n_folds=3,
    )

    with mlflow.start_run() as run:
        log_loso_result(result, target_col="label_binary")
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
    run_data = client.get_run(run_id)

    params = run_data.data.params
    metrics = run_data.data.metrics

    assert params["model"] == "logistic"
    assert params["feature_group"] == "hrv"
    assert params["target"] == "label_binary"
    assert float(metrics["macro_f1"]) == pytest.approx(0.72, abs=1e-6)
    assert float(metrics["recall_class_0"]) == pytest.approx(0.80, abs=1e-6)
    assert float(metrics["recall_class_1"]) == pytest.approx(0.64, abs=1e-6)


# ===========================================================================
# main() integration
# ===========================================================================


def test_main_returns_1_when_features_file_missing(tmp_path: Path) -> None:
    """main() must return exit code 1 when the feature table does not exist."""
    missing = tmp_path / "no_such_file.csv"

    exit_code = main(["--features", str(missing)])

    assert exit_code == 1


def test_main_returns_1_on_bad_extension(tmp_path: Path) -> None:
    """main() must return exit code 1 for an unsupported file format."""
    bad = tmp_path / "features.txt"
    bad.write_text("subject,label_binary\n1,0\n")

    exit_code = main(["--features", str(bad)])

    assert exit_code == 1


def test_main_end_to_end_returns_0(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """main() must return 0 and print a results table on a valid feature CSV.

    Uses a small synthetic dataset (3 subjects, 8 windows each) so the full
    (model x group) grid finishes quickly.  MLflow is redirected to a temp DB.
    """
    csv_path = _make_feature_csv(tmp_path / "features.csv", n_subjects=3, windows_per_subject=8)
    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    exit_code = main(
        [
            "--features",
            str(csv_path),
            "--target",
            "label_binary",
            "--experiment",
            "test_main",
            "--mlflow-uri",
            mlflow_uri,
        ]
    )

    assert exit_code == 0


def test_main_prints_results_table(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """main() must print the LOSO results table header to stdout."""
    csv_path = _make_feature_csv(tmp_path / "features.csv", n_subjects=3, windows_per_subject=6)
    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"

    main(
        [
            "--features",
            str(csv_path),
            "--target",
            "label_binary",
            "--experiment",
            "test_print",
            "--mlflow-uri",
            mlflow_uri,
        ]
    )

    captured = capsys.readouterr()
    assert "LOSO Baseline Results" in captured.out
    assert "macro_f1" in captured.out


def test_main_logs_runs_to_mlflow(tmp_path: Path) -> None:
    """main() must create one MLflow run per (model x feature-group) combination.

    3 models x 5 groups (hrv, eda, temp, acc, all) = 15 runs.
    """
    csv_path = _make_feature_csv(tmp_path / "features.csv", n_subjects=3, windows_per_subject=6)
    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "test_runs_count"

    main(
        [
            "--features",
            str(csv_path),
            "--target",
            "label_binary",
            "--experiment",
            experiment_name,
            "--mlflow-uri",
            mlflow_uri,
        ]
    )

    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    # 3 models x 5 groups = 15 runs
    assert len(runs) == 15


def test_main_all_runs_have_macro_f1_metric(tmp_path: Path) -> None:
    """Every MLflow run created by main() must record a macro_f1 metric."""
    csv_path = _make_feature_csv(tmp_path / "features.csv", n_subjects=3, windows_per_subject=6)
    mlflow_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    experiment_name = "test_metric_presence"

    main(
        [
            "--features",
            str(csv_path),
            "--target",
            "label_binary",
            "--experiment",
            experiment_name,
            "--mlflow-uri",
            mlflow_uri,
        ]
    )

    mlflow.set_tracking_uri(mlflow_uri)
    client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    for run in runs:
        assert "macro_f1" in run.data.metrics, (
            f"Run {run.info.run_name!r} is missing macro_f1 metric"
        )
