"""Personalization experiment runner (Phase 4, Stage 3).

Pure, testable functions for running Method A (baseline centering) and
Method B2 (calibrated decision threshold) across held-out subjects and
evaluating them strictly on the shared evaluation set T_eval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline

from stress.models.feature_groups import NON_FEATURE_COLUMNS, ablation_sets
from stress.models.loso import build_pipeline
from stress.personalization.centering import center_features, compute_calibration_mean
from stress.personalization.splitting import get_calibration_windows, split_calibration_and_test
from stress.personalization.threshold import calibrated_threshold, predict_with_threshold

SUPPORTED_MODELS: tuple[str, ...] = ("logistic", "random_forest")
SUPPORTED_GROUPS: tuple[str, ...] = ("all", "hrv")
SUPPORTED_K_VALUES: tuple[int, ...] = (0, 3, 5, 10)
SUPPORTED_METHODS: tuple[str, ...] = ("reference", "method_a", "method_b2")

PRIMARY_CONDITION: dict[str, Any] = {
    "method": "method_a",
    "model": "logistic",
    "group": "all",
    "k": 10,
}


@dataclass(frozen=True)
class SubjectConditionResult:
    """Evaluation result for one held-out subject under one experimental condition."""

    subject: int | str
    method: str
    model: str
    group: str
    k: int
    macro_f1: float
    recall_stress: float
    recall_not_stress: float
    roc_auc: float
    n_eval: int
    n_eval_stress: int
    theta: float


def compute_eval_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute standard classification metrics on evaluation set T_eval.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Predicted probabilities of class 1 (stress), optional.

    Returns:
        Dict with keys: macro_f1, recall_stress, recall_not_stress,
        roc_auc, n_eval, n_eval_stress.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    n_eval = int(len(y_true_arr))
    n_eval_stress = int((y_true_arr == 1).sum())

    macro_f1 = float(f1_score(y_true_arr, y_pred_arr, average="macro", labels=[0, 1]))
    recall_stress = float(
        recall_score(y_true_arr, y_pred_arr, pos_label=1, labels=[0, 1], zero_division=0.0)
    )
    recall_not_stress = float(
        recall_score(y_true_arr, y_pred_arr, pos_label=0, labels=[0, 1], zero_division=0.0)
    )

    roc_auc = float("nan")
    if y_prob is not None:
        y_prob_arr = np.asarray(y_prob, dtype=float)
        # roc_auc is only defined when both classes are present in y_true
        if len(np.unique(y_true_arr)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr))
            except ValueError:
                roc_auc = float("nan")

    return {
        "macro_f1": macro_f1,
        "recall_stress": recall_stress,
        "recall_not_stress": recall_not_stress,
        "roc_auc": roc_auc,
        "n_eval": n_eval,
        "n_eval_stress": n_eval_stress,
    }


def prepare_centered_training(
    train_df: pd.DataFrame,
    k: int,
    feature_cols: Sequence[str],
    *,
    subject_col: str = "subject",
    target_col: str = "label_3class",
    baseline_val: int = 1,
) -> pd.DataFrame:
    """Center every training subject by the mean of its own first k true-baseline windows.

    Args:
        train_df: Training DataFrame containing all training subjects.
        k: Number of baseline calibration windows to center with.
        feature_cols: Feature columns to center.
        subject_col: Column identifying subjects.
        target_col: Target column identifying true baseline condition.
        baseline_val: Value identifying true baseline (default: 1 for label_3class).

    Returns:
        DataFrame with each training subject centered by its own baseline mean.

    Raises:
        ValueError: If any training subject has fewer than k baseline windows.
    """
    if k == 0:
        return train_df.copy()

    centered_parts: list[pd.DataFrame] = []
    for s_id, s_group in train_df.groupby(subject_col, sort=False):
        cal_windows = get_calibration_windows(
            s_group,
            k=k,
            target_col=target_col,
            baseline_val=baseline_val,
        )
        if len(cal_windows) < k:
            raise ValueError(
                f"Training subject {s_id} has fewer than {k} baseline windows "
                f"(found {len(cal_windows)})"
            )
        cal_mean = compute_calibration_mean(cal_windows, feature_cols)
        s_centered = center_features(s_group, cal_mean, feature_cols)
        centered_parts.append(s_centered)

    return pd.concat(centered_parts, axis=0)


def aggregate_conditions(
    per_subject_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute aggregate statistics across subjects and paired deltas vs k=0.

    Args:
        per_subject_df: DataFrame of per-subject results with columns:
            [subject, method, model, group, k, macro_f1, recall_stress,
             recall_not_stress, roc_auc, n_eval, n_eval_stress, theta].

    Returns:
        Summary DataFrame with one row per condition and paired delta metrics.
    """
    # Build lookup for k=0 reference macro_f1 per (subject, model, group)
    k0_mask = per_subject_df["k"] == 0
    ref_f1_lookup: dict[tuple[Any, str, str], float] = {}
    for _, row in per_subject_df[k0_mask].iterrows():
        key = (row["subject"], row["model"], row["group"])
        ref_f1_lookup[key] = float(row["macro_f1"])

    summary_rows: list[dict[str, Any]] = []
    grouped = per_subject_df.groupby(["method", "model", "group", "k"], sort=False)

    for (method, model, group, k), cond_df in grouped:
        f1_vals = cond_df["macro_f1"].to_numpy()
        rec_stress_vals = cond_df["recall_stress"].to_numpy()
        rec_not_stress_vals = cond_df["recall_not_stress"].to_numpy()
        roc_auc_vals = cond_df["roc_auc"].dropna().to_numpy()

        # Compute paired deltas vs k=0
        deltas: list[float] = []
        for _, row in cond_df.iterrows():
            ref_key = (row["subject"], row["model"], row["group"])
            ref_f1 = ref_f1_lookup.get(ref_key, float("nan"))
            deltas.append(float(row["macro_f1"] - ref_f1))

        deltas_arr = np.array(deltas, dtype=float)

        n_improved = int((deltas_arr > 1e-9).sum())
        n_worse = int((deltas_arr < -1e-9).sum())
        n_equal = int((np.abs(deltas_arr) <= 1e-9).sum())

        # Wilcoxon signed-rank test handling zero differences without crashing
        if len(deltas_arr) < 1 or np.all(np.isclose(deltas_arr, 0.0)):
            wilcoxon_p = 1.0
        else:
            try:
                # zero_method='wilcox' discards zero deltas
                res = wilcoxon(deltas_arr, zero_method="wilcox")
                wilcoxon_p = float(res.pvalue)
            except ValueError:
                wilcoxon_p = 1.0

        is_primary = bool(
            method == PRIMARY_CONDITION["method"]
            and model == PRIMARY_CONDITION["model"]
            and group == PRIMARY_CONDITION["group"]
            and k == PRIMARY_CONDITION["k"]
        )

        summary_rows.append(
            {
                "method": method,
                "model": model,
                "group": group,
                "k": k,
                "macro_f1_mean": float(np.mean(f1_vals)),
                "macro_f1_median": float(np.median(f1_vals)),
                "macro_f1_std": float(np.std(f1_vals, ddof=1)) if len(f1_vals) > 1 else 0.0,
                "recall_stress_mean": float(np.mean(rec_stress_vals)),
                "recall_not_stress_mean": float(np.mean(rec_not_stress_vals)),
                "roc_auc_mean": (
                    float(np.mean(roc_auc_vals)) if len(roc_auc_vals) > 0 else float("nan")
                ),
                "mean_delta_f1": float(np.mean(deltas_arr)),
                "median_delta_f1": float(np.median(deltas_arr)),
                "std_delta_f1": float(np.std(deltas_arr, ddof=1)) if len(deltas_arr) > 1 else 0.0,
                "n_improved": n_improved,
                "n_worse": n_worse,
                "n_equal": n_equal,
                "wilcoxon_p": wilcoxon_p,
                "is_primary": is_primary,
                "comparison": "primary" if is_primary else "exploratory",
            }
        )

    return pd.DataFrame(summary_rows)


def run_personalization_experiment(
    df: pd.DataFrame,
    *,
    models: Sequence[str] = SUPPORTED_MODELS,
    groups: Sequence[str] = SUPPORTED_GROUPS,
    k_values: Sequence[int] = SUPPORTED_K_VALUES,
    seed: int = 42,
    target_col: str = "label_binary",
    subject_col: str = "subject",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full personalization experiments over all held-out subjects.

    Protocol:
    - Target: binary target label_binary only.
    - Models: logistic, random_forest.
    - Feature groups: all, hrv.
    - k in {0, 3, 5, 10} only.
    - T_eval comes from split_calibration_and_test(k_max=10) and is identical for all k.
    - Reference (k=0): fit pipeline on other subjects' raw features; predict T_eval.
    - Method A (k in {3, 5, 10}): center training subjects and held-out subject with first k
      true-baseline windows (label_3class == 1); fit fresh pipeline on centered training subjects;
      predict T_eval.
    - Method B2 (k in {3, 5, 10}): reuse k=0 model; p_cal on calibration windows;
      theta = max(0.5, max(p_cal)); predict T_eval with p > theta.

    Args:
        df: Feature DataFrame with subject, labels, and features.
        models: Sequence of model names.
        groups: Sequence of feature group names.
        k_values: Sequence of calibration window counts.
        seed: Random seed for model building.
        target_col: Target column name (must be label_binary).
        subject_col: Column identifying subjects.

    Returns:
        Tuple of (per_subject_df, summary_df).
    """
    subjects = sorted(df[subject_col].unique())
    all_feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    ab_sets = ablation_sets(all_feature_cols)

    subject_results: list[SubjectConditionResult] = []

    for model_name in models:
        for group_name in groups:
            feature_cols = ab_sets[group_name]

            # Cache the fitted Reference (k=0) models per held-out subject
            fitted_ref_models: dict[Any, Pipeline] = {}
            ref_subject_results: dict[Any, SubjectConditionResult] = {}

            # Phase 1: Reference (k=0) for all subjects
            for s in subjects:
                s_mask = df[subject_col] == s
                s_df = df[s_mask].copy()
                train_df = df[~s_mask].copy()

                # T_eval is identical across all k via k_max=10
                split_0 = split_calibration_and_test(
                    s_df,
                    k=0,
                    k_max=10,
                    target_col="label_3class",
                    baseline_val=1,
                )
                t_eval = split_0.test_df
                y_eval = t_eval[target_col].to_numpy()

                pipe_k0 = build_pipeline(model_name, seed=seed)
                pipe_k0.fit(train_df[feature_cols], train_df[target_col])
                fitted_ref_models[s] = pipe_k0

                p_eval = pipe_k0.predict_proba(t_eval[feature_cols])[:, 1]
                y_pred = pipe_k0.predict(t_eval[feature_cols])
                m = compute_eval_metrics(y_eval, y_pred, p_eval)

                res_ref = SubjectConditionResult(
                    subject=s,
                    method="reference",
                    model=model_name,
                    group=group_name,
                    k=0,
                    macro_f1=m["macro_f1"],
                    recall_stress=m["recall_stress"],
                    recall_not_stress=m["recall_not_stress"],
                    roc_auc=m["roc_auc"],
                    n_eval=m["n_eval"],
                    n_eval_stress=m["n_eval_stress"],
                    theta=0.5,
                )
                ref_subject_results[s] = res_ref
                subject_results.append(res_ref)

            # Phase 2: Method A for k in k_values
            for k in k_values:
                for s in subjects:
                    s_mask = df[subject_col] == s
                    s_df = df[s_mask].copy()
                    train_df = df[~s_mask].copy()

                    split_k = split_calibration_and_test(
                        s_df,
                        k=k,
                        k_max=10,
                        target_col="label_3class",
                        baseline_val=1,
                    )
                    t_eval = split_k.test_df
                    y_eval = t_eval[target_col].to_numpy()

                    if k == 0:
                        # Method A at k=0 is equivalent to Reference k=0
                        ref_res = ref_subject_results[s]
                        subject_results.append(
                            SubjectConditionResult(
                                subject=s,
                                method="method_a",
                                model=model_name,
                                group=group_name,
                                k=0,
                                macro_f1=ref_res.macro_f1,
                                recall_stress=ref_res.recall_stress,
                                recall_not_stress=ref_res.recall_not_stress,
                                roc_auc=ref_res.roc_auc,
                                n_eval=ref_res.n_eval,
                                n_eval_stress=ref_res.n_eval_stress,
                                theta=0.5,
                            )
                        )
                    else:
                        if len(split_k.cal_df) < k:
                            raise ValueError(
                                f"Subject {s} has fewer than {k} baseline windows "
                                f"(found {len(split_k.cal_df)})"
                            )

                        # Center training subjects
                        train_centered = prepare_centered_training(
                            train_df,
                            k=k,
                            feature_cols=feature_cols,
                            target_col="label_3class",
                            baseline_val=1,
                        )

                        # Center held-out subject's T_eval using its first k calibration windows
                        cal_mean_s = compute_calibration_mean(split_k.cal_df, feature_cols)
                        t_eval_centered = center_features(t_eval, cal_mean_s, feature_cols)

                        pipe_a = build_pipeline(model_name, seed=seed)
                        pipe_a.fit(train_centered[feature_cols], train_centered[target_col])

                        p_eval = pipe_a.predict_proba(t_eval_centered[feature_cols])[:, 1]
                        y_pred = pipe_a.predict(t_eval_centered[feature_cols])
                        m = compute_eval_metrics(y_eval, y_pred, p_eval)

                        subject_results.append(
                            SubjectConditionResult(
                                subject=s,
                                method="method_a",
                                model=model_name,
                                group=group_name,
                                k=k,
                                macro_f1=m["macro_f1"],
                                recall_stress=m["recall_stress"],
                                recall_not_stress=m["recall_not_stress"],
                                roc_auc=m["roc_auc"],
                                n_eval=m["n_eval"],
                                n_eval_stress=m["n_eval_stress"],
                                theta=0.5,
                            )
                        )

            # Phase 3: Method B2 for k in k_values
            for k in k_values:
                for s in subjects:
                    s_mask = df[subject_col] == s
                    s_df = df[s_mask].copy()

                    split_k = split_calibration_and_test(
                        s_df,
                        k=k,
                        k_max=10,
                        target_col="label_3class",
                        baseline_val=1,
                    )
                    t_eval = split_k.test_df
                    y_eval = t_eval[target_col].to_numpy()

                    # Reuse k=0 fitted model
                    pipe_ref = fitted_ref_models[s]
                    p_eval = pipe_ref.predict_proba(t_eval[feature_cols])[:, 1]

                    if k == 0:
                        theta = 0.5
                    else:
                        p_cal = (
                            pipe_ref.predict_proba(split_k.cal_df[feature_cols])[:, 1]
                            if len(split_k.cal_df) > 0
                            else []
                        )
                        theta = calibrated_threshold(p_cal)

                    y_pred = predict_with_threshold(p_eval, theta=theta)
                    m = compute_eval_metrics(y_eval, y_pred, p_eval)

                    subject_results.append(
                        SubjectConditionResult(
                            subject=s,
                            method="method_b2",
                            model=model_name,
                            group=group_name,
                            k=k,
                            macro_f1=m["macro_f1"],
                            recall_stress=m["recall_stress"],
                            recall_not_stress=m["recall_not_stress"],
                            roc_auc=m["roc_auc"],
                            n_eval=m["n_eval"],
                            n_eval_stress=m["n_eval_stress"],
                            theta=theta,
                        )
                    )

    per_subject_df = pd.DataFrame([asdict(r) for r in subject_results])
    summary_df = aggregate_conditions(per_subject_df)
    return per_subject_df, summary_df


def format_summary_table(summary_df: pd.DataFrame) -> str:
    """Format aggregate conditions into a plain-ASCII summary table.

    Args:
        summary_df: Aggregated conditions DataFrame from aggregate_conditions.

    Returns:
        Formatted ASCII table string.
    """
    header = (
        f"{'Method':<11} {'Model':<14} {'Group':<6} {'k':<3} "
        f"{'Mean F1':<8} {'Med F1':<8} {'Med Δ':<8} {'+ / - / =':<11} "
        f"{'Wilcox p':<9} {'Comparison'}"
    )
    divider = "-" * len(header)
    lines = [header, divider]

    for _, row in summary_df.iterrows():
        method = str(row["method"])
        model = str(row["model"])
        group = str(row["group"])
        k = int(row["k"])
        mean_f1 = f"{row['macro_f1_mean']:.4f}"
        med_f1 = f"{row['macro_f1_median']:.4f}"
        med_delta = f"{row['median_delta_f1']:+.4f}"
        n_imp, n_wor, n_eq = int(row["n_improved"]), int(row["n_worse"]), int(row["n_equal"])
        counts = f"{n_imp:>2} / {n_wor:>2} / {n_eq:>2}"
        wilcox_p = f"{row['wilcoxon_p']:.4f}"
        comp_tag = "* PRIMARY *" if row["is_primary"] else "exploratory"

        line = (
            f"{method:<11} {model:<14} {group:<6} {k:<3} "
            f"{mean_f1:<8} {med_f1:<8} {med_delta:<8} {counts:<11} "
            f"{wilcox_p:<9} {comp_tag}"
        )
        lines.append(line)

    return "\n".join(lines)
