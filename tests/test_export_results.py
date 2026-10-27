"""Unit tests for scripts/export_results.py.

All tests use synthetic data in tmp_path only. Real datasets and mlflow.db are never touched.
"""

from __future__ import annotations

import pathlib

import mlflow
import pandas as pd
import pytest

from scripts.export_results import (
    export_loso_baselines,
    export_personalization_summary,
    main,
)
from stress.personalization.experiment import format_summary_table


def test_export_loso_baselines_synthetic(tmp_path: pathlib.Path) -> None:
    """Export LOSO baselines extracts only params./metrics. columns rounded to 4 decimals."""
    db_path = tmp_path / "mlflow_test.db"
    tracking_uri = f"sqlite:///{db_path}"
    exp_name = "test_loso_experiment"

    mlflow.set_tracking_uri(tracking_uri)
    exp_id = mlflow.create_experiment(exp_name)

    # Log two runs with float metrics/params having >4 decimals, plus tags
    with mlflow.start_run(experiment_id=exp_id, run_name="run_1"):
        mlflow.log_param("model_name", "logistic")
        mlflow.log_param("learning_rate", 0.0123456)
        mlflow.log_metric("macro_f1", 0.8548765)
        mlflow.log_metric("recall_class_0", 0.7654321)
        mlflow.set_tag("run_tag", "tag_value_1")

    with mlflow.start_run(experiment_id=exp_id, run_name="run_2"):
        mlflow.log_param("model_name", "random_forest")
        mlflow.log_param("learning_rate", 0.0543219)
        mlflow.log_metric("macro_f1", 0.7891234)
        mlflow.log_metric("recall_class_0", 0.6543219)
        mlflow.set_tag("run_tag", "tag_value_2")

    out_csv = tmp_path / "exported_loso.csv"
    ret = main(
        [
            "loso-baselines",
            "--tracking-uri",
            tracking_uri,
            "--experiment",
            exp_name,
            "--out",
            str(out_csv),
        ]
    )
    assert ret == 0
    assert out_csv.exists()

    df_exported = pd.read_csv(out_csv)

    # 1. Assert only params. and metrics. columns are present
    assert len(df_exported.columns) > 0
    assert all(c.startswith(("params.", "metrics.")) for c in df_exported.columns)
    assert not any(c.startswith("tags.") for c in df_exported.columns)
    assert "Unnamed: 0" not in df_exported.columns

    # 2. Assert values are rounded to four decimals (and not two decimals)
    for col in df_exported.columns:
        if col.startswith("metrics."):
            for val in df_exported[col]:
                assert round(val, 4) == pytest.approx(val, abs=1e-6)
                # Confirm precision is four decimals rather than truncated to two decimals
                assert round(val, 2) != pytest.approx(val, abs=1e-4)


def test_export_personalization_summary_synthetic(tmp_path: pathlib.Path) -> None:
    """Export personalization summary aggregates per-subject CSV and outputs summary CSV and TXT."""
    per_sub_df = pd.DataFrame(
        [
            {
                "subject": 1,
                "method": "method_a",
                "model": "logistic",
                "group": "all",
                "k": 0,
                "macro_f1": 0.50,
                "recall_stress": 0.50,
                "recall_not_stress": 0.50,
                "roc_auc": 0.50,
                "n_eval": 10,
                "n_eval_stress": 5,
                "theta": 0.50,
            },
            {
                "subject": 2,
                "method": "method_a",
                "model": "logistic",
                "group": "all",
                "k": 0,
                "macro_f1": 0.50,
                "recall_stress": 0.50,
                "recall_not_stress": 0.50,
                "roc_auc": 0.50,
                "n_eval": 10,
                "n_eval_stress": 5,
                "theta": 0.50,
            },
            {
                "subject": 1,
                "method": "method_a",
                "model": "logistic",
                "group": "all",
                "k": 10,
                "macro_f1": 0.80,
                "recall_stress": 0.80,
                "recall_not_stress": 0.80,
                "roc_auc": 0.80,
                "n_eval": 10,
                "n_eval_stress": 5,
                "theta": 0.50,
            },
            {
                "subject": 2,
                "method": "method_a",
                "model": "logistic",
                "group": "all",
                "k": 10,
                "macro_f1": 0.90,
                "recall_stress": 0.90,
                "recall_not_stress": 0.90,
                "roc_auc": 0.90,
                "n_eval": 10,
                "n_eval_stress": 5,
                "theta": 0.50,
            },
        ]
    )

    per_sub_path = tmp_path / "per_sub.csv"
    per_sub_df.to_csv(per_sub_path, index=False)

    summary_csv = tmp_path / "summary.csv"
    summary_txt = tmp_path / "summary.txt"

    ret = main(
        [
            "personalization-summary",
            "--per-subject",
            str(per_sub_path),
            "--summary-csv",
            str(summary_csv),
            "--summary-txt",
            str(summary_txt),
        ]
    )
    assert ret == 0
    assert summary_csv.exists()
    assert summary_txt.exists()

    # 1. Assert summary CSV exists without an index column
    df_summary = pd.read_csv(summary_csv)
    assert "Unnamed: 0" not in df_summary.columns
    assert list(df_summary.columns[:4]) == ["method", "model", "group", "k"]

    # 2. Assert TXT ends with a single newline and equals format_summary_table + newline
    content = summary_txt.read_text(encoding="utf-8")
    assert content.endswith("\n")
    assert not content.endswith("\n\n")

    expected_content = format_summary_table(df_summary) + "\n"
    assert content == expected_content


def test_missing_input_returns_1(tmp_path: pathlib.Path) -> None:
    """Missing input files or non-existent experiments return returncode 1."""
    missing_csv = tmp_path / "does_not_exist.csv"
    missing_db = tmp_path / "does_not_exist.db"

    # Direct function calls
    assert export_personalization_summary(per_subject=missing_csv) == 1
    assert export_loso_baselines(tracking_uri=f"sqlite:///{missing_db}") == 1

    # CLI main calls
    ret_pers = main(["personalization-summary", "--per-subject", str(missing_csv)])
    assert ret_pers == 1

    ret_loso = main(["loso-baselines", "--tracking-uri", f"sqlite:///{missing_db}"])
    assert ret_loso == 1
