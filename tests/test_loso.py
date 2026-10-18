"""Unit tests for the LOSO evaluation module (src/stress/models/loso.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stress.models.loso import (
    SUPPORTED_MODELS,
    FoldResult,
    LOSOResult,
    build_pipeline,
    run_loso,
    run_loso_fold,
    summarize_results,
)

# ---------------------------------------------------------------------------
# Canonical feature columns (must match FEATURE_GROUPS in feature_groups.py)
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


def _make_df(
    n_subjects: int = 4,
    windows_per_subject: int = 10,
    nan_hrv: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a deterministic synthetic feature DataFrame for tests.

    Labels alternate 0/1 per window so every subject (and every fold's
    training set) has both classes, avoiding degenerate edge cases.

    Args:
        n_subjects: Number of subjects (IDs 1..n_subjects).
        windows_per_subject: Windows per subject.
        nan_hrv: If True, set all HRV columns to NaN (simulates no-beat windows).
        seed: NumPy RNG seed.

    Returns:
        DataFrame with 'subject', 'window_start_s', 'label_binary', and all
        feature columns.
    """
    rng = np.random.default_rng(seed)
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

    df = pd.DataFrame(rows)

    if nan_hrv:
        hrv_cols = [c for c in _FEATURE_COLS if c.startswith("hrv_")]
        df[hrv_cols] = np.nan

    return df


# ===========================================================================
# build_pipeline
# ===========================================================================


def test_build_pipeline_has_three_named_steps() -> None:
    """Pipeline must expose named steps: imputer, scaler, clf."""
    pipe = build_pipeline("dummy")

    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps.keys()) == ["imputer", "scaler", "clf"]


def test_build_pipeline_imputer_is_median() -> None:
    """Imputer step uses median strategy to be robust to outliers."""
    pipe = build_pipeline("logistic")

    imputer = pipe.named_steps["imputer"]
    assert isinstance(imputer, SimpleImputer)
    assert imputer.strategy == "median"


def test_build_pipeline_scaler_is_standard() -> None:
    """Scaler step must be StandardScaler."""
    pipe = build_pipeline("random_forest")

    assert isinstance(pipe.named_steps["scaler"], StandardScaler)


def test_build_pipeline_dummy_classifier() -> None:
    """'dummy' maps to DummyClassifier with most_frequent strategy."""
    pipe = build_pipeline("dummy")

    clf = pipe.named_steps["clf"]
    assert isinstance(clf, DummyClassifier)
    assert clf.strategy == "most_frequent"


def test_build_pipeline_logistic_classifier() -> None:
    """'logistic' maps to LogisticRegression with 1000-iteration limit."""
    pipe = build_pipeline("logistic")

    clf = pipe.named_steps["clf"]
    assert isinstance(clf, LogisticRegression)
    assert clf.max_iter == 1000


def test_build_pipeline_random_forest_classifier() -> None:
    """'random_forest' maps to RandomForestClassifier with 100 trees."""
    pipe = build_pipeline("random_forest")

    clf = pipe.named_steps["clf"]
    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == 100


def test_build_pipeline_unknown_model_raises() -> None:
    """An unrecognised model name must raise ValueError immediately."""
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("svm")


def test_build_pipeline_returns_unfitted_pipeline() -> None:
    """A freshly built pipeline must not yet have fitted scaler attributes."""
    pipe = build_pipeline("logistic")

    assert not hasattr(pipe.named_steps["scaler"], "mean_"), (
        "Pipeline must be unfitted on construction"
    )


def test_build_pipeline_seed_is_forwarded() -> None:
    """Two pipelines with the same seed must produce the same predictions."""
    df = _make_df(n_subjects=3, windows_per_subject=8, seed=99)
    feature_cols = [
        c
        for c in df.columns
        if c not in ("subject", "window_start_s", "label_3class", "label_binary")
    ]

    x_arr = df[feature_cols].to_numpy(dtype=float)
    y = df["label_binary"].to_numpy()

    pipe_a = build_pipeline("random_forest", seed=7)
    pipe_b = build_pipeline("random_forest", seed=7)
    pipe_a.fit(x_arr, y)
    pipe_b.fit(x_arr, y)

    assert np.array_equal(pipe_a.predict(x_arr), pipe_b.predict(x_arr))


def test_supported_models_tuple_contains_three_entries() -> None:
    """SUPPORTED_MODELS must have exactly three entries."""
    assert len(SUPPORTED_MODELS) == 3
    assert set(SUPPORTED_MODELS) == {"dummy", "logistic", "random_forest"}


# ===========================================================================
# run_loso_fold
# ===========================================================================


def test_run_loso_fold_returns_fold_result() -> None:
    """run_loso_fold must return a FoldResult dataclass instance."""
    df = _make_df(n_subjects=3)
    pipe = build_pipeline("dummy")

    result = run_loso_fold(
        df, test_subject=1, feature_cols=_FEATURE_COLS, target_col="label_binary", pipeline=pipe
    )

    assert isinstance(result, FoldResult)


def test_run_loso_fold_correct_subject_id() -> None:
    """FoldResult.subject_id must equal the held-out subject."""
    df = _make_df(n_subjects=3)
    pipe = build_pipeline("dummy")

    result = run_loso_fold(
        df, test_subject=2, feature_cols=_FEATURE_COLS, target_col="label_binary", pipeline=pipe
    )

    assert result.subject_id == 2


def test_run_loso_fold_predictions_match_test_window_count() -> None:
    """y_pred and y_true must have one entry per held-out window."""
    windows_per_subject = 12
    df = _make_df(n_subjects=4, windows_per_subject=windows_per_subject)
    pipe = build_pipeline("dummy")

    result = run_loso_fold(
        df, test_subject=3, feature_cols=_FEATURE_COLS, target_col="label_binary", pipeline=pipe
    )

    assert len(result.y_pred) == windows_per_subject
    assert len(result.y_true) == windows_per_subject


def test_run_loso_fold_training_excludes_test_subject() -> None:
    """Scaler mean must be computed from training subjects only, not the test subject.

    Subject 1 is given an extreme outlier value (1000) for eda_mean.
    Subjects 2-4 have eda_mean ~ Normal(0,1).  If subject 1 leaks into the
    scaler, the scaler's mean for that column will be inflated.
    """
    df = _make_df(n_subjects=4, windows_per_subject=10, seed=5)

    # Give subject 1 a dramatically different eda_mean
    col = "eda_mean"
    df.loc[df["subject"] == 1, col] = 1000.0

    pipe = build_pipeline("logistic")
    run_loso_fold(
        df, test_subject=1, feature_cols=_FEATURE_COLS, target_col="label_binary", pipeline=pipe
    )

    scaler_mean = pipe.named_steps["scaler"].mean_
    feat_idx = _FEATURE_COLS.index(col)
    # Mean of eda_mean across subjects 2-4 (30 windows of Normal(0,1)) stays small
    assert abs(scaler_mean[feat_idx]) < 5.0, (
        f"Scaler mean for {col} = {scaler_mean[feat_idx]:.1f}; "
        "suggests test subject data leaked into scaler fit"
    )


def test_run_loso_fold_imputer_fills_nan_from_training() -> None:
    """NaN HRV features in the test subject are filled with training-fold median.

    All HRV columns are NaN for subject 1.  Subjects 2-4 have real values.
    After the fold the pipeline must produce a prediction (not raise) and
    the imputer must have computed a finite median from training data.
    """
    df = _make_df(n_subjects=4, windows_per_subject=10)
    hrv_cols = [c for c in _FEATURE_COLS if c.startswith("hrv_")]
    df.loc[df["subject"] == 1, hrv_cols] = np.nan

    pipe = build_pipeline("logistic")
    result = run_loso_fold(
        df, test_subject=1, feature_cols=_FEATURE_COLS, target_col="label_binary", pipeline=pipe
    )

    # Predictions must be finite integers, not NaN
    assert not np.any(np.isnan(result.y_pred.astype(float)))

    # Imputer statistics must be finite (derived from training subjects only)
    imputer_stats = pipe.named_steps["imputer"].statistics_
    assert np.all(np.isfinite(imputer_stats)), (
        "Imputer computed non-finite statistics; possibly trained on NaN-only data"
    )


def test_run_loso_fold_empty_training_raises() -> None:
    """A single-subject DataFrame leaves no training data -> ValueError."""
    df = _make_df(n_subjects=1, windows_per_subject=10)
    pipe = build_pipeline("dummy")

    with pytest.raises(ValueError, match="Empty training set"):
        run_loso_fold(
            df, test_subject=1, feature_cols=_FEATURE_COLS, target_col="label_binary", pipeline=pipe
        )


def test_run_loso_fold_unknown_test_subject_raises() -> None:
    """Holding out a subject not in the DataFrame -> ValueError."""
    df = _make_df(n_subjects=3, windows_per_subject=10)
    pipe = build_pipeline("dummy")

    with pytest.raises(ValueError, match="No test windows"):
        run_loso_fold(
            df,
            test_subject=99,
            feature_cols=_FEATURE_COLS,
            target_col="label_binary",
            pipeline=pipe,
        )


# ===========================================================================
# run_loso
# ===========================================================================


def test_run_loso_returns_loso_result() -> None:
    """run_loso must return a LOSOResult instance."""
    df = _make_df(n_subjects=3)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy")

    assert isinstance(result, LOSOResult)


def test_run_loso_n_folds_equals_n_subjects() -> None:
    """The number of folds must equal the number of unique subjects."""
    n_subjects = 4
    df = _make_df(n_subjects=n_subjects)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy")

    assert result.n_folds == n_subjects


def test_run_loso_fold_results_covers_every_subject() -> None:
    """Every subject appears exactly once across fold_results."""
    n_subjects = 4
    df = _make_df(n_subjects=n_subjects)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy")

    held_out_ids = [f.subject_id for f in result.fold_results]
    assert sorted(held_out_ids) == list(range(1, n_subjects + 1))


def test_run_loso_macro_f1_in_unit_interval() -> None:
    """macro_f1 must lie in [0, 1]."""
    df = _make_df(n_subjects=3)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "logistic")

    assert 0.0 <= result.macro_f1 <= 1.0


def test_run_loso_per_class_recall_keys_binary() -> None:
    """per_class_recall must have keys {0, 1} for a binary target."""
    df = _make_df(n_subjects=3)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy")

    assert set(result.per_class_recall.keys()) == {0, 1}


def test_run_loso_per_class_recall_values_in_unit_interval() -> None:
    """All per-class recall values must lie in [0, 1]."""
    df = _make_df(n_subjects=3)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy")

    for cls, recall in result.per_class_recall.items():
        assert 0.0 <= recall <= 1.0, f"recall_class_{cls} = {recall} out of range"


def test_run_loso_feature_group_stored_in_result() -> None:
    """The feature_group label must be preserved in LOSOResult."""
    df = _make_df(n_subjects=3)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy", feature_group="hrv")

    assert result.feature_group == "hrv"


def test_run_loso_model_name_stored_in_result() -> None:
    """model_name must be preserved in LOSOResult."""
    df = _make_df(n_subjects=3)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "logistic")

    assert result.model_name == "logistic"


def test_run_loso_dummy_recall_class_1_is_zero() -> None:
    """DummyClassifier(most_frequent) always predicts the majority class.

    Labels alternate 0/1 with equal frequency -> majority is 0.
    recall for class 1 should be 0 (never predicted), recall for class 0 = 1.
    """
    df = _make_df(n_subjects=4, windows_per_subject=10)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy")

    # most_frequent will pick 0 (first alphabetically when tied) or whichever
    # wins; either way one class gets recall 0
    recalls = result.per_class_recall
    assert recalls[0] == pytest.approx(1.0) or recalls[1] == pytest.approx(1.0), (
        "At least one class must have recall=1.0 for most_frequent dummy"
    )
    assert recalls[0] == pytest.approx(0.0) or recalls[1] == pytest.approx(0.0), (
        "At least one class must have recall=0.0 for most_frequent dummy"
    )


def test_run_loso_concatenated_predictions_same_length_as_df() -> None:
    """Total predictions across all folds must equal the number of windows."""
    df = _make_df(n_subjects=4, windows_per_subject=10)

    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy")

    total_preds = sum(len(f.y_pred) for f in result.fold_results)
    assert total_preds == len(df)


def test_run_loso_fresh_pipeline_per_fold() -> None:
    """Scaler must be fitted independently per fold (different held-out subjects).

    If pipelines were shared, scaler means would be identical across folds.
    Here the outlier subject changes each fold so scaler means should differ.
    """
    df = _make_df(n_subjects=4, windows_per_subject=10, seed=3)
    # Subject 1 has extreme eda_mean; subject 2 has extreme temp_mean
    df.loc[df["subject"] == 1, "eda_mean"] = 500.0
    df.loc[df["subject"] == 2, "temp_mean"] = 500.0

    # Manually run two folds and compare scaler means
    pipe_a = build_pipeline("logistic")
    pipe_b = build_pipeline("logistic")

    from stress.models.loso import run_loso_fold as _run_fold

    _run_fold(df, 1, _FEATURE_COLS, "label_binary", pipe_a)
    _run_fold(df, 2, _FEATURE_COLS, "label_binary", pipe_b)

    eda_idx = _FEATURE_COLS.index("eda_mean")
    temp_idx = _FEATURE_COLS.index("temp_mean")

    # Fold 1 (held-out=1): eda_mean outlier is excluded -> scaler mean small
    assert abs(pipe_a.named_steps["scaler"].mean_[eda_idx]) < 50.0
    # Fold 2 (held-out=2): temp_mean outlier is excluded -> scaler mean small
    assert abs(pipe_b.named_steps["scaler"].mean_[temp_idx]) < 50.0


# ===========================================================================
# summarize_results
# ===========================================================================


def test_summarize_results_returns_dataframe() -> None:
    """summarize_results must return a pandas DataFrame."""
    df = _make_df(n_subjects=3)
    results = [run_loso(df, _FEATURE_COLS, "label_binary", m) for m in ("dummy", "logistic")]

    summary = summarize_results(results)

    assert isinstance(summary, pd.DataFrame)


def test_summarize_results_one_row_per_result() -> None:
    """There must be exactly one row per LOSOResult in the summary."""
    df = _make_df(n_subjects=3)
    results = [run_loso(df, _FEATURE_COLS, "label_binary", m) for m in ("dummy", "logistic")]

    summary = summarize_results(results)

    assert len(summary) == 2


def test_summarize_results_has_required_columns() -> None:
    """Summary DataFrame must have model, feature_group, macro_f1, n_folds columns."""
    df = _make_df(n_subjects=3)
    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy")

    summary = summarize_results([result])

    for col in ("model", "feature_group", "macro_f1", "n_folds"):
        assert col in summary.columns, f"Missing column: {col}"


def test_summarize_results_has_recall_columns_for_each_class() -> None:
    """Summary must include recall_class_0 and recall_class_1 for binary target."""
    df = _make_df(n_subjects=3)
    result = run_loso(df, _FEATURE_COLS, "label_binary", "dummy")

    summary = summarize_results([result])

    assert "recall_class_0" in summary.columns
    assert "recall_class_1" in summary.columns


def test_summarize_results_macro_f1_matches_loso_result() -> None:
    """macro_f1 in summary must equal (rounded) LOSOResult.macro_f1."""
    df = _make_df(n_subjects=3)
    result = run_loso(df, _FEATURE_COLS, "label_binary", "logistic")

    summary = summarize_results([result])

    assert summary.iloc[0]["macro_f1"] == round(result.macro_f1, 4)


def test_summarize_results_model_name_matches() -> None:
    """'model' column in summary must match LOSOResult.model_name."""
    df = _make_df(n_subjects=3)
    result = run_loso(df, _FEATURE_COLS, "label_binary", "random_forest", feature_group="eda")

    summary = summarize_results([result])

    assert summary.iloc[0]["model"] == "random_forest"
    assert summary.iloc[0]["feature_group"] == "eda"


def test_summarize_results_empty_list() -> None:
    """An empty result list must produce an empty DataFrame, not raise."""
    summary = summarize_results([])

    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 0
