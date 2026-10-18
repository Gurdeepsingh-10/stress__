"""Tests for scripts/make_results_tables.py using synthetic CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.make_results_tables import (
    build_binary_baselines_table,
    build_method_b2_theta_table,
    build_personalization_summary_table,
    build_primary_per_subject_table,
    build_t_eval_table,
    build_three_class_baselines_table,
    compute_primary_comparison_stats,
    main,
)


@pytest.fixture
def synthetic_baselines_csv(tmp_path: Path) -> Path:
    """Create a minimal synthetic loso_baselines.csv."""
    data = [
        {
            "params.target": "label_binary",
            "params.model": "toy_model",
            "params.feature_group": "all",
            "metrics.macro_f1": 0.87654,
            "metrics.recall_class_0": 0.91234,
            "metrics.recall_class_1": 0.84074,
            "metrics.recall_class_2": None,
            "metrics.recall_class_3": None,
        },
        {
            "params.target": "label_3class",
            "params.model": "toy_model",
            "params.feature_group": "all",
            "metrics.macro_f1": 0.61234,
            "metrics.recall_class_0": None,
            "metrics.recall_class_1": 0.85004,
            "metrics.recall_class_2": 0.80004,
            "metrics.recall_class_3": 0.15004,
        },
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / "loso_baselines.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_summary_csv(tmp_path: Path) -> Path:
    """Create a minimal synthetic personalization_summary.csv."""
    data = [
        {
            "method": "method_a",
            "model": "logistic",
            "group": "all",
            "k": 10,
            "macro_f1_mean": 0.899456,
            "macro_f1_median": 0.944568,
            "median_delta_f1": 0.040935,
            "n_improved": 9,
            "n_worse": 4,
            "n_equal": 2,
            "wilcoxon_p": 0.278707,
            "comparison": "primary",
        },
        {
            "method": "method_b2",
            "model": "logistic",
            "group": "all",
            "k": 10,
            "macro_f1_mean": 0.842756,
            "macro_f1_median": 0.843886,
            "median_delta_f1": 0.0,
            "n_improved": 2,
            "n_worse": 2,
            "n_equal": 11,
            "wilcoxon_p": 0.715000,
            "comparison": "exploratory",
        },
    ]
    df = pd.DataFrame(data)
    csv_path = tmp_path / "personalization_summary.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def synthetic_per_subject_csv(tmp_path: Path) -> Path:
    """Create a minimal synthetic personalization_per_subject.csv.

    Subjects 1 to 5 with deliberate deltas:
      Subject 1: Ref 0.80, A10 0.90 -> delta +0.10
      Subject 2: Ref 0.70, A10 0.50 -> delta -0.20
      Subject 3: Ref 0.60, A10 0.90 -> delta +0.30
      Subject 4: Ref 0.50, A10 0.50 -> delta 0.00
      Subject 5: Ref 0.90, A10 0.90 -> delta 0.00
    Non-zero deltas: 3 (+0.1, -0.2, +0.3).
    Hand-checked Wilcoxon W on these deltas with zero_method='wilcox': W = 2.0.
    """
    rows = []
    # Reference condition
    ref_f1s = [0.80, 0.70, 0.60, 0.50, 0.90]
    n_evals = [54, 55, 56, 57, 59]
    for i, s in enumerate([1, 2, 3, 4, 5]):
        rows.append(
            {
                "subject": s,
                "method": "reference",
                "model": "logistic",
                "group": "all",
                "k": 0,
                "macro_f1": ref_f1s[i],
                "recall_stress": 0.75,
                "recall_not_stress": 0.95,
                "roc_auc": 0.98,
                "n_eval": n_evals[i],
                "n_eval_stress": 20,
                "theta": 0.5,
            }
        )

    # Method A k=10 condition
    a10_f1s = [0.90, 0.50, 0.90, 0.50, 0.90]
    for i, s in enumerate([1, 2, 3, 4, 5]):
        rows.append(
            {
                "subject": s,
                "method": "method_a",
                "model": "logistic",
                "group": "all",
                "k": 10,
                "macro_f1": a10_f1s[i],
                "recall_stress": 0.85,
                "recall_not_stress": 0.96,
                "roc_auc": 0.99,
                "n_eval": n_evals[i],
                "n_eval_stress": 20,
                "theta": 0.5,
            }
        )

    # Method B2 condition with varying theta
    # 3 subjects: one with theta=0.55 (>0.5), one with 0.50, one with 0.48
    thetas = [0.55, 0.50, 0.48]
    for s, th in zip([1, 2, 3], thetas, strict=True):
        rows.append(
            {
                "subject": s,
                "method": "method_b2",
                "model": "logistic",
                "group": "all",
                "k": 3,
                "macro_f1": 0.82,
                "recall_stress": 0.70,
                "recall_not_stress": 0.92,
                "roc_auc": 0.95,
                "n_eval": 55,
                "n_eval_stress": 20,
                "theta": th,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "personalization_per_subject.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_baseline_tables_exact_row_values(synthetic_baselines_csv: Path) -> None:
    """Test that generated baseline tables contain rows with exact formatted values."""
    df = pd.read_csv(synthetic_baselines_csv)

    # Binary table
    binary_md = build_binary_baselines_table(df)
    expected_binary_row = "| toy_model | all | 0.8765 | 0.9123 | 0.8407 |"
    assert expected_binary_row in binary_md

    # 3-class table
    multi_md = build_three_class_baselines_table(df)
    expected_multi_row = "| toy_model | all | 0.6123 | 0.8500 | 0.8000 | 0.1500 |"
    assert expected_multi_row in multi_md


def test_personalization_summary_table_exact_row_values(synthetic_summary_csv: Path) -> None:
    """Test that personalization summary table contains exact formatted values."""
    df = pd.read_csv(synthetic_summary_csv)
    summary_md = build_personalization_summary_table(df)

    expected_row = (
        "| method_a | logistic | all | 10 | 0.8995 | 0.9446 | +0.0409 | 9/4/2 | 0.2787 | primary |"
    )
    assert expected_row in summary_md


def test_primary_comparison_stats_wilcoxon_and_nonzero_counts(
    synthetic_per_subject_csv: Path,
) -> None:
    """Test W statistic and non-zero count on hand-checked delta vector with zeros.

    Deltas: [+0.10, -0.20, +0.30, 0.00, 0.00].
    Non-zero count must be 3.
    Wilcoxon W must be 2.0 with zero_method='wilcox'.
    """
    df = pd.read_csv(synthetic_per_subject_csv)
    stats = compute_primary_comparison_stats(df)

    assert stats["non_zero_diffs"] == 3
    assert stats["total_subjects"] == 5
    assert stats["wilcoxon_w"] == pytest.approx(2.0, abs=1e-6)
    assert stats["wilcoxon_p"] == pytest.approx(0.75, abs=1e-6)
    assert stats["mean_delta_f1"] == pytest.approx(0.04, abs=1e-6)
    assert stats["median_delta_f1"] == pytest.approx(0.0, abs=1e-6)


def test_method_b2_theta_counts(synthetic_per_subject_csv: Path) -> None:
    """Test that subjects with raised theta (> 0.5) are counted correctly."""
    df = pd.read_csv(synthetic_per_subject_csv)
    theta_md = build_method_b2_theta_table(df)

    # 1 of 3 subjects has theta > 0.5 (theta=0.55)
    expected_row = "| logistic | all | 3 | 1 | 3 |"
    assert expected_row in theta_md


def test_t_eval_table_min_max_mean(synthetic_per_subject_csv: Path) -> None:
    """Test calculation of min, max, and mean n_eval with corresponding subjects."""
    df = pd.read_csv(synthetic_per_subject_csv)
    t_eval_md = build_t_eval_table(df)

    # Min is 54 (subject 1), Max is 59 (subject 5), Mean is 56.20
    assert "Minimum Evaluation Set Size | 54.0 | Subject 1" in t_eval_md
    assert "Maximum Evaluation Set Size | 59.0 | Subject 5" in t_eval_md
    assert "Mean Evaluation Set Size | 56.20 | All 5 subjects" in t_eval_md


def test_primary_per_subject_table(synthetic_per_subject_csv: Path) -> None:
    """Test per-subject breakdown table generation."""
    df = pd.read_csv(synthetic_per_subject_csv)
    sub_md = build_primary_per_subject_table(df)

    expected_sub1_row = "| 1 | 0.8000 | 0.9000 | +0.1000 | 0.7500 | 0.8500 | 0.9500 | 0.9600 |"
    assert expected_sub1_row in sub_md


def test_main_missing_input_file_exits_nonzero(tmp_path: Path) -> None:
    """Test that CLI exits with non-zero status code when an input file is missing."""
    missing_file = tmp_path / "nonexistent.csv"
    existing_file = tmp_path / "dummy.csv"
    existing_file.write_text("a,b\n1,2", encoding="utf-8")

    exit_code = main(
        [
            "--baselines",
            str(missing_file),
            "--summary",
            str(existing_file),
            "--per-subject",
            str(existing_file),
            "--out",
            str(tmp_path / "out.md"),
        ]
    )
    assert exit_code != 0


def test_main_successful_execution(
    synthetic_baselines_csv: Path,
    synthetic_summary_csv: Path,
    synthetic_per_subject_csv: Path,
    tmp_path: Path,
) -> None:
    """Test end-to-end CLI generation of markdown file with synthetic inputs."""
    out_file = tmp_path / "results_tables.md"

    exit_code = main(
        [
            "--baselines",
            str(synthetic_baselines_csv),
            "--summary",
            str(synthetic_summary_csv),
            "--per-subject",
            str(synthetic_per_subject_csv),
            "--out",
            str(out_file),
        ]
    )
    assert exit_code == 0
    assert out_file.exists()

    content = out_file.read_text(encoding="utf-8")
    assert "## 1. Binary Classification Baselines" in content
    assert "## 2. 3-Class Classification Baselines" in content
    assert "## 3. Personalization Summary (Method A and Method B2)" in content
    assert "## 4. Pre-Registered Primary Comparison" in content
    assert "## 5. Per-Subject Breakdown for Primary Comparison" in content
    assert "## 6. Evaluation Window Count Distribution (T_eval)" in content
    assert "## 7. Method B2 Threshold Shift Counts (theta > 0.5)" in content
