"""Unit tests for Phase 4, Stage 3 personalization experiment runner.

All tests use synthetic data only with fixed seeds. No real data is loaded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stress.models.feature_groups import FEATURE_GROUPS
from stress.models.loso import build_pipeline
from stress.personalization.experiment import (
    PRIMARY_CONDITION,
    run_personalization_experiment,
)
from stress.personalization.splitting import split_calibration_and_test
from stress.personalization.threshold import calibrated_threshold, predict_with_threshold


def make_synthetic_dataset(
    n_subjects: int = 12,
    n_windows: int = 60,
    n_base: int = 35,
    offset_scale: float = 0.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic subjects with physiological feature columns."""
    rng = np.random.RandomState(seed)
    feature_cols = [col for group in FEATURE_GROUPS.values() for col in group]

    rows = []
    for s in range(1, n_subjects + 1):
        s_offset = rng.uniform(-offset_scale, offset_scale) if offset_scale > 0 else 0.0
        for w in range(n_windows):
            is_stress = w >= n_base
            label_3class = 2 if is_stress else 1
            label_binary = 1 if is_stress else 0
            stress_effect = 8.0 if is_stress else 0.0

            row = {
                "subject": s,
                "window_start_s": float(w * 30),
                "label_3class": label_3class,
                "label_binary": label_binary,
            }
            for col in feature_cols:
                row[col] = s_offset + stress_effect + rng.normal(0, 1.0)
            rows.append(row)

    return pd.DataFrame(rows)


def test_reference_and_k0_identical() -> None:
    """1. Method A and B2 at k=0 give predictions identical to plain population model on T_eval."""
    df = make_synthetic_dataset(n_subjects=3, n_windows=40, seed=123)
    per_sub, _ = run_personalization_experiment(
        df,
        models=["logistic"],
        groups=["all"],
        k_values=[0],
        seed=42,
    )

    for s in df["subject"].unique():
        s_sub = per_sub[per_sub["subject"] == s]
        ref_row = s_sub[s_sub["method"] == "reference"].iloc[0]
        a_row = s_sub[s_sub["method"] == "method_a"].iloc[0]
        b2_row = s_sub[s_sub["method"] == "method_b2"].iloc[0]

        assert a_row["macro_f1"] == pytest.approx(ref_row["macro_f1"], abs=1e-12)
        assert b2_row["macro_f1"] == pytest.approx(ref_row["macro_f1"], abs=1e-12)
        assert a_row["recall_stress"] == pytest.approx(ref_row["recall_stress"], abs=1e-12)
        assert b2_row["recall_stress"] == pytest.approx(ref_row["recall_stress"], abs=1e-12)


def test_b2_invariants() -> None:
    """2. B2 invariant: stress predictions at any k are a subset of k=0; ROC-AUC equals k=0."""
    df = make_synthetic_dataset(n_subjects=3, n_windows=40, offset_scale=15.0, seed=999)
    per_sub, _ = run_personalization_experiment(
        df,
        models=["logistic"],
        groups=["all"],
        k_values=[0, 3, 5, 10],
        seed=42,
    )

    all_cols = [c for c in df.columns if c.startswith(("hrv_", "eda_", "temp_", "acc_"))]

    for s in df["subject"].unique():
        s_mask = df["subject"] == s
        s_df = df[s_mask]
        train_df = df[~s_mask]

        pipe = build_pipeline("logistic", seed=42)
        pipe.fit(train_df[all_cols], train_df["label_binary"])

        split_0 = split_calibration_and_test(
            s_df, k=0, k_max=10, target_col="label_3class", baseline_val=1
        )
        t_eval = split_0.test_df
        p_eval = pipe.predict_proba(t_eval[all_cols])[:, 1]
        preds_k0 = predict_with_threshold(p_eval, theta=0.5)

        s_sub = per_sub[per_sub["subject"] == s]
        ref_auc = s_sub[s_sub["method"] == "reference"]["roc_auc"].iloc[0]

        for k in (3, 5, 10):
            b2_row = s_sub[(s_sub["method"] == "method_b2") & (s_sub["k"] == k)].iloc[0]
            # ROC-AUC invariant: probabilities unchanged, so ranking is identical
            assert b2_row["roc_auc"] == pytest.approx(ref_auc, abs=1e-12)

            split_k = split_calibration_and_test(
                s_df, k=k, k_max=10, target_col="label_3class", baseline_val=1
            )
            p_cal = pipe.predict_proba(split_k.cal_df[all_cols])[:, 1]
            theta_k = b2_row["theta"]
            assert theta_k == pytest.approx(calibrated_threshold(p_cal), abs=1e-12)
            assert theta_k >= float(np.max(p_cal))
            preds_k = predict_with_threshold(p_eval, theta=theta_k)

            # Subset invariant: where preds_k == 1, preds_k0 must also be 1
            stress_indices_k = set(np.where(preds_k == 1)[0])
            stress_indices_k0 = set(np.where(preds_k0 == 1)[0])
            assert stress_indices_k.issubset(stress_indices_k0)


def test_positive_control_method_a() -> None:
    """3. Positive control: 12 subjects x 60 windows, large offsets; Method A k=10 delta > 0.05."""
    df_pos = make_synthetic_dataset(n_subjects=12, n_windows=60, offset_scale=30.0, seed=42)
    _, summary_pos = run_personalization_experiment(
        df_pos,
        models=["logistic"],
        groups=["all"],
        k_values=[0, 10],
        seed=42,
    )

    row_a_k10 = summary_pos[(summary_pos["method"] == "method_a") & (summary_pos["k"] == 10)].iloc[
        0
    ]
    assert row_a_k10["median_delta_f1"] > 0.05


def test_negative_control_method_a() -> None:
    """4. Negative control: 12 subjects x 60 windows, NO offsets; Method A delta within +/-0.05."""
    df_neg = make_synthetic_dataset(n_subjects=12, n_windows=60, offset_scale=0.0, seed=42)
    _, summary_neg = run_personalization_experiment(
        df_neg,
        models=["logistic"],
        groups=["all"],
        k_values=[0, 10],
        seed=42,
    )

    row_a_k10 = summary_neg[(summary_neg["method"] == "method_a") & (summary_neg["k"] == 10)].iloc[
        0
    ]
    assert -0.05 <= row_a_k10["median_delta_f1"] <= 0.05


def test_t_eval_row_indices_identical() -> None:
    """5. T_eval row indices are identical across all conditions for each subject."""
    df = make_synthetic_dataset(n_subjects=4, n_windows=50, seed=777)
    per_sub, _ = run_personalization_experiment(
        df,
        models=["logistic"],
        groups=["all"],
        k_values=[0, 3, 5, 10],
        seed=42,
    )

    for s in df["subject"].unique():
        s_rows = per_sub[per_sub["subject"] == s]
        n_eval_unique = s_rows["n_eval"].unique()
        assert len(n_eval_unique) == 1, (
            f"T_eval size differs across conditions for subject {s}: {n_eval_unique}"
        )
        s_df = df[df["subject"] == s]
        split_ref = split_calibration_and_test(
            s_df, k=0, k_max=10, target_col="label_3class", baseline_val=1
        )
        assert n_eval_unique[0] == len(split_ref.test_df)


def test_primary_comparison_tagged_once() -> None:
    """6. The primary comparison is tagged exactly once in the experiment output."""
    df = make_synthetic_dataset(n_subjects=3, n_windows=40, seed=555)
    _, summary_df = run_personalization_experiment(
        df,
        models=["logistic", "random_forest"],
        groups=["all", "hrv"],
        k_values=[0, 3, 5, 10],
        seed=42,
    )

    primary_rows = summary_df[summary_df["is_primary"]]
    assert len(primary_rows) == 1
    p_row = primary_rows.iloc[0]
    assert p_row["method"] == PRIMARY_CONDITION["method"]
    assert p_row["model"] == PRIMARY_CONDITION["model"]
    assert p_row["group"] == PRIMARY_CONDITION["group"]
    assert p_row["k"] == PRIMARY_CONDITION["k"]
    assert p_row["comparison"] == "primary"

    exploratory_rows = summary_df[~summary_df["is_primary"]]
    assert (exploratory_rows["comparison"] == "exploratory").all()


def test_fold_never_fits_on_held_out_subject(monkeypatch: pytest.MonkeyPatch) -> None:
    """7. Assert that the training frame contains no rows from the held-out subject."""
    df = make_synthetic_dataset(n_subjects=4, n_windows=40, seed=321)

    import stress.personalization.experiment as exp_mod

    original_build = exp_mod.build_pipeline

    fitted_train_indices: list[pd.Index] = []

    class _PipelineSpy:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def fit(self, x: pd.DataFrame, y: pd.Series) -> _PipelineSpy:
            fitted_train_indices.append(x.index)
            self._inner.fit(x, y)
            return self

        def predict(self, x: pd.DataFrame) -> np.ndarray:
            return self._inner.predict(x)

        def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
            return self._inner.predict_proba(x)

    monkeypatch.setattr(
        exp_mod, "build_pipeline", lambda m, seed=42: _PipelineSpy(original_build(m, seed=seed))
    )

    run_personalization_experiment(
        df,
        models=["logistic"],
        groups=["all"],
        k_values=[0, 3],
        seed=42,
    )

    all_subjects = set(df["subject"].unique())
    assert len(fitted_train_indices) > 0
    for idx in fitted_train_indices:
        subjects_in_train = set(df.loc[df.index.isin(idx), "subject"].unique())
        assert subjects_in_train != all_subjects, (
            f"Training frame contained all {len(all_subjects)} subjects — held-out subject leaked!"
        )
        assert len(all_subjects - subjects_in_train) == 1


def test_method_a_held_out_centering_uses_only_first_k_baseline_windows() -> None:
    """Method A centers held-out subject using only first k baseline windows, not all."""
    df_clean = make_synthetic_dataset(n_subjects=4, n_windows=50, seed=42)

    # Subject 1 baseline windows: windows 0..34 are baseline (label_3class == 1)
    # k=3 uses windows 0, 1, 2. Windows 3..9 are discarded buffer windows.
    # T_eval starts at window 10.
    per_sub_clean, _ = run_personalization_experiment(
        df_clean,
        models=["logistic"],
        groups=["all"],
        k_values=[3],
        seed=42,
    )
    clean_f1 = per_sub_clean[
        (per_sub_clean["subject"] == 1) & (per_sub_clean["method"] == "method_a")
    ]["macro_f1"].iloc[0]

    # Perturb ONLY the discarded buffer baseline windows (windows 3..9) of subject 1
    df_perturbed = df_clean.copy()
    feature_cols = [
        c for c in df_perturbed.columns if c.startswith(("hrv_", "eda_", "temp_", "acc_"))
    ]
    s1_mask = df_perturbed["subject"] == 1
    s1_indices = df_perturbed[s1_mask].index
    buffer_indices = s1_indices[3:10]  # windows 3 through 9 (discarded buffer)
    df_perturbed.loc[buffer_indices, feature_cols] += 10000.0

    per_sub_perturbed, _ = run_personalization_experiment(
        df_perturbed,
        models=["logistic"],
        groups=["all"],
        k_values=[3],
        seed=42,
    )
    perturbed_f1 = per_sub_perturbed[
        (per_sub_perturbed["subject"] == 1) & (per_sub_perturbed["method"] == "method_a")
    ]["macro_f1"].iloc[0]

    # Since buffer windows are discarded and NOT part of the first k=3 calibration windows,
    # clean and perturbed F1 must be identical. If all baseline windows were used, this fails.
    assert clean_f1 == pytest.approx(perturbed_f1, abs=1e-12)


def test_cli_parse_args() -> None:
    """CLI argument parser parses defaults and custom arguments properly."""
    from scripts.run_personalization import parse_args

    args = parse_args([])
    assert args.features.name == "features_phase2.csv"
    assert args.experiment == "personalization_phase4"

    custom_args = parse_args(
        ["--features", "custom.csv", "--experiment", "custom_exp", "--mlflow-uri", "custom_uri"]
    )
    assert custom_args.features.name == "custom.csv"
    assert custom_args.experiment == "custom_exp"
    assert custom_args.mlflow_uri == "custom_uri"


def test_cli_missing_file_returns_1(tmp_path: pytest.TempPathFactory) -> None:
    """CLI returns 1 when features file does not exist."""
    from scripts.run_personalization import main

    missing_path = str(tmp_path / "nonexistent.csv")
    exit_code = main(["--features", missing_path])
    assert exit_code == 1
