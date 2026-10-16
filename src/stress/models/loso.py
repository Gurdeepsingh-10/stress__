"""LOSO (leave-one-subject-out) evaluation for stress classification baselines.

Three classifiers are supported: DummyClassifier (most-frequent), logistic
regression, and random forest.  Every pipeline uses the same two preprocessing
steps fitted exclusively on the training fold:

  SimpleImputer(strategy="median")
      Fills NaN HRV features that arise when fewer than 10 beats are detected
      in a 60-second window.
  StandardScaler
      Zero-mean, unit-variance normalisation.

No statistics from the held-out subject ever touch the imputer or scaler,
preventing leakage between folds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SUPPORTED_MODELS: tuple[str, ...] = ("dummy", "logistic", "random_forest")


@dataclass
class FoldResult:
    """Predictions and ground truth for a single held-out subject."""

    subject_id: int
    y_true: np.ndarray
    y_pred: np.ndarray


@dataclass
class LOSOResult:
    """Aggregated results of a complete LOSO cross-validation run."""

    model_name: str
    feature_group: str
    macro_f1: float
    per_class_recall: dict[int, float]
    n_folds: int
    fold_results: list[FoldResult] = field(default_factory=list)


def build_pipeline(model_name: str, seed: int = 42) -> Pipeline:
    """Build an impute -> scale -> classify Pipeline.

    All preprocessing steps (imputer, scaler) are fitted on training data only;
    this function returns an *unfitted* pipeline so the caller decides when to fit.

    Args:
        model_name: One of 'dummy', 'logistic', 'random_forest'.
        seed: Random seed for classifiers that support one.

    Returns:
        An unfitted sklearn Pipeline with named steps 'imputer', 'scaler', 'clf'.

    Raises:
        ValueError: If model_name is not in SUPPORTED_MODELS.
    """
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unknown model {model_name!r}; expected one of {sorted(SUPPORTED_MODELS)}"
        )

    if model_name == "dummy":
        classifier: Any = DummyClassifier(strategy="most_frequent", random_state=seed)
    elif model_name == "logistic":
        classifier = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    else:  # random_forest
        classifier = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1)

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def run_loso_fold(
    df: pd.DataFrame,
    test_subject: int,
    feature_cols: list[str],
    target_col: str,
    pipeline: Pipeline,
) -> FoldResult:
    """Train on all subjects except test_subject; predict on test_subject.

    The pipeline is fitted only on the training rows.  Imputer statistics and
    scaler parameters are never contaminated by the held-out subject's data.

    Args:
        df: Feature table with columns 'subject', the feature columns, and
            target_col.
        test_subject: Subject ID to hold out as the test set.
        feature_cols: Names of feature columns to use as predictors.
        target_col: Name of the target column.
        pipeline: An *unfitted* sklearn Pipeline; it is mutated (fitted) in-place.

    Returns:
        FoldResult with subject_id, y_true, and y_pred arrays.

    Raises:
        ValueError: If the training set or test set contains zero rows.
    """
    train_mask = df["subject"] != test_subject
    test_mask = df["subject"] == test_subject

    x_train = df.loc[train_mask, feature_cols].to_numpy(dtype=float)
    y_train = df.loc[train_mask, target_col].to_numpy()
    x_test = df.loc[test_mask, feature_cols].to_numpy(dtype=float)
    y_test = df.loc[test_mask, target_col].to_numpy()

    if len(x_train) == 0:
        raise ValueError(f"Empty training set when holding out subject {test_subject}")
    if len(x_test) == 0:
        raise ValueError(f"No test windows found for subject {test_subject}")

    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    return FoldResult(subject_id=test_subject, y_true=y_test, y_pred=y_pred)


def run_loso(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_name: str,
    feature_group: str = "all",
    seed: int = 42,
) -> LOSOResult:
    """Run full leave-one-subject-out cross-validation.

    For each unique subject in df['subject']: build a fresh pipeline, train on
    the remaining subjects, predict on the held-out subject.  All held-out
    predictions are concatenated and evaluated together to compute macro-F1 and
    per-class recall (reported alongside accuracy per project rules).

    Args:
        df: Feature table with 'subject', feature columns, and target_col.
        feature_cols: Feature columns to use as predictors.
        target_col: Name of the classification target column.
        model_name: Model identifier; must be in SUPPORTED_MODELS.
        feature_group: Label for this feature subset (stored in the result only).
        seed: Fixed random seed; use 42 by project convention.

    Returns:
        LOSOResult with macro_f1, per_class_recall, n_folds, and fold_results.
    """
    subjects = sorted(df["subject"].unique())
    fold_results: list[FoldResult] = []

    for subject_id in subjects:
        pipeline = build_pipeline(model_name, seed=seed)
        fold = run_loso_fold(df, int(subject_id), feature_cols, target_col, pipeline)
        fold_results.append(fold)

    y_true_all = np.concatenate([f.y_true for f in fold_results])
    y_pred_all = np.concatenate([f.y_pred for f in fold_results])

    classes = sorted(set(y_true_all.tolist()))
    macro_f1 = float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0))
    recalls_arr = recall_score(
        y_true_all, y_pred_all, labels=classes, average=None, zero_division=0
    )
    per_class_recall = {
        int(cls): float(rec) for cls, rec in zip(classes, recalls_arr, strict=False)
    }

    return LOSOResult(
        model_name=model_name,
        feature_group=feature_group,
        macro_f1=macro_f1,
        per_class_recall=per_class_recall,
        n_folds=len(fold_results),
        fold_results=fold_results,
    )


def summarize_results(results: list[LOSOResult]) -> pd.DataFrame:
    """Format a list of LOSOResult objects into a readable DataFrame.

    Args:
        results: One or more LOSOResult objects returned by run_loso().

    Returns:
        DataFrame with columns model, feature_group, macro_f1, n_folds, and
        recall_class_<N> for each class present across all results.
    """
    rows: list[dict[str, Any]] = []
    for r in results:
        row: dict[str, Any] = {
            "model": r.model_name,
            "feature_group": r.feature_group,
            "macro_f1": round(r.macro_f1, 4),
            "n_folds": r.n_folds,
        }
        for cls, recall in sorted(r.per_class_recall.items()):
            row[f"recall_class_{cls}"] = round(recall, 4)
        rows.append(row)
    return pd.DataFrame(rows)
