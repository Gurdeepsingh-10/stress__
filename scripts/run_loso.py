"""Run LOSO baseline experiments and log every run to MLflow.

Usage:
    .venv/Scripts/python.exe scripts/run_loso.py
    .venv/Scripts/python.exe scripts/run_loso.py --features data/processed/features_phase2.csv
    .venv/Scripts/python.exe scripts/run_loso.py --target label_binary

Models: dummy, logistic, random_forest
Feature groups: hrv, eda, temp, acc, all  (one ablation per group + all combined)
Each (model, group) pair is a separate MLflow run logged to sqlite:///mlflow.db.

Exit code:
    0  all runs completed successfully
    1  feature table not found or unsupported format
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd

from stress.models.feature_groups import NON_FEATURE_COLUMNS, ablation_sets
from stress.models.loso import SUPPORTED_MODELS, LOSOResult, run_loso, summarize_results

SEED: int = 42
DEFAULT_MLFLOW_URI: str = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT: str = "loso_baselines"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv when None).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run LOSO baseline experiments and log results to MLflow."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/processed/features_phase2.csv"),
        help="Feature table CSV or Parquet (default: data/processed/features_phase2.csv)",
    )
    parser.add_argument(
        "--target",
        default="label_binary",
        choices=["label_binary", "label_3class"],
        help="Target column (default: label_binary)",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT})",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=DEFAULT_MLFLOW_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_MLFLOW_URI})",
    )
    return parser.parse_args(argv)


def load_features(path: Path) -> pd.DataFrame:
    """Load the feature table from a CSV or Parquet file.

    Args:
        path: File path ending in .csv or .parquet.

    Returns:
        Loaded DataFrame.

    Raises:
        ValueError: If the file extension is not .csv or .parquet.
    """
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature file format: {path.suffix!r}; use .csv or .parquet")


def log_loso_result(result: LOSOResult, target_col: str) -> None:
    """Log params and metrics of one LOSOResult into the active MLflow run.

    Must be called inside an active mlflow.start_run() context.

    Args:
        result: Completed LOSO result to log.
        target_col: Target column name used in this experiment.
    """
    mlflow.log_param("model", result.model_name)
    mlflow.log_param("feature_group", result.feature_group)
    mlflow.log_param("target", target_col)
    mlflow.log_param("seed", SEED)
    mlflow.log_param("n_folds", result.n_folds)

    mlflow.log_metric("macro_f1", result.macro_f1)
    for cls, recall in result.per_class_recall.items():
        mlflow.log_metric(f"recall_class_{cls}", recall)


def main(argv: list[str] | None = None) -> int:
    """Run all (model x feature-group) LOSO experiments; log to MLflow; print summary.

    Args:
        argv: Command-line arguments (defaults to sys.argv when None).

    Returns:
        0 on success, 1 if the feature table is missing or unreadable.
    """
    args = parse_args(argv)

    if not args.features.exists():
        print(f"ERROR: feature table not found: {args.features}", file=sys.stderr)
        return 1

    try:
        df = load_features(args.features)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    n_subjects = df["subject"].nunique()
    print(f"Loaded {len(df)} windows from {n_subjects} subjects")
    print(f"Target: {args.target}  |  Unique labels: {sorted(df[args.target].unique())}")

    feature_cols_all = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    ab_sets = ablation_sets(feature_cols_all)

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    all_results: list[LOSOResult] = []

    for model_name in SUPPORTED_MODELS:
        for group_name, feature_cols in ab_sets.items():
            print(
                f"  Running: model={model_name:<14}  group={group_name:<5}"
                f"  n_features={len(feature_cols)}"
            )
            result = run_loso(
                df=df,
                feature_cols=feature_cols,
                target_col=args.target,
                model_name=model_name,
                feature_group=group_name,
                seed=SEED,
            )
            with mlflow.start_run(run_name=f"{model_name}__{group_name}"):
                log_loso_result(result, args.target)

            all_results.append(result)
            print(
                f"    macro_F1={result.macro_f1:.4f}"
                f"  recalls={result.per_class_recall}"
            )

    print("\n=== LOSO Baseline Results ===")
    summary = summarize_results(all_results)
    print(summary.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
