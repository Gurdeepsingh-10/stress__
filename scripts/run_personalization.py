"""CLI runner for Phase 4 personalization experiments.

Runs Method A (baseline centering) and Method B2 (calibrated threshold)
across held-out subjects on T_eval, logs all conditions to MLflow,
saves per-subject metrics to CSV, and prints an ASCII summary table.

Usage:
    .venv/Scripts/python.exe scripts/run_personalization.py
    .venv/Scripts/python.exe scripts/run_personalization.py \\
        --features data/processed/features_phase2.csv \\
        --mlflow-uri sqlite:///mlflow.db \\
        --out data/processed/personalization_per_subject.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd

from stress.personalization.experiment import (
    format_summary_table,
    run_personalization_experiment,
)

DEFAULT_FEATURES_PATH: Path = Path("data/processed/features_phase2.csv")
DEFAULT_MLFLOW_URI: str = "sqlite:///mlflow.db"
DEFAULT_EXPERIMENT: str = "personalization_phase4"
DEFAULT_OUT_PATH: Path = Path("data/processed/personalization_per_subject.csv")
SEED: int = 42


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line argument list (default: sys.argv[1:]).

    Returns:
        Parsed Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Run Phase 4 personalization experiments and log to MLflow."
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=DEFAULT_FEATURES_PATH,
        help=f"Path to feature table CSV/Parquet (default: {DEFAULT_FEATURES_PATH})",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=DEFAULT_MLFLOW_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_MLFLOW_URI})",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_EXPERIMENT})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_PATH,
        help=f"Path for per-subject CSV output (default: {DEFAULT_OUT_PATH})",
    )
    return parser.parse_args(argv)


def load_features(path: Path) -> pd.DataFrame:
    """Load features table from CSV or Parquet.

    Args:
        path: Path to features file.

    Returns:
        DataFrame.
    """
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported feature file format: {path.suffix!r}; use .csv or .parquet")


def log_condition_to_mlflow(
    condition_row: pd.Series,
    per_subject_csv_path: Path,
    seed: int = SEED,
) -> None:
    """Log one experimental condition to MLflow with parameters and metrics.

    Args:
        condition_row: Row from summary_df.
        per_subject_csv_path: Path to saved per-subject CSV artifact.
        seed: Random seed.
    """
    mlflow.log_param("method", condition_row["method"])
    mlflow.log_param("model", condition_row["model"])
    mlflow.log_param("group", condition_row["group"])
    mlflow.log_param("k", int(condition_row["k"]))
    mlflow.log_param("target", "label_binary")
    mlflow.log_param("seed", seed)

    comparison_tag = "primary" if condition_row["is_primary"] else "exploratory"
    mlflow.set_tag("comparison", comparison_tag)

    mlflow.log_metric("macro_f1_mean", condition_row["macro_f1_mean"])
    mlflow.log_metric("macro_f1_median", condition_row["macro_f1_median"])
    mlflow.log_metric("macro_f1_std", condition_row["macro_f1_std"])
    mlflow.log_metric("recall_stress_mean", condition_row["recall_stress_mean"])
    mlflow.log_metric("recall_not_stress_mean", condition_row["recall_not_stress_mean"])
    if not pd.isna(condition_row["roc_auc_mean"]):
        mlflow.log_metric("roc_auc_mean", condition_row["roc_auc_mean"])

    mlflow.log_metric("mean_delta_f1", condition_row["mean_delta_f1"])
    mlflow.log_metric("median_delta_f1", condition_row["median_delta_f1"])
    mlflow.log_metric("std_delta_f1", condition_row["std_delta_f1"])
    mlflow.log_metric("n_improved", condition_row["n_improved"])
    mlflow.log_metric("n_worse", condition_row["n_worse"])
    mlflow.log_metric("n_equal", condition_row["n_equal"])
    mlflow.log_metric("wilcoxon_p", condition_row["wilcoxon_p"])

    if per_subject_csv_path.exists():
        mlflow.log_artifact(str(per_subject_csv_path))


def main(argv: list[str] | None = None) -> int:
    """Main CLI entrypoint.

    Args:
        argv: Command-line arguments.

    Returns:
        0 on success, 1 on error.
    """
    args = parse_args(argv)

    if not args.features.exists():
        print(f"ERROR: feature table not found: {args.features}", file=sys.stderr)
        return 1

    try:
        df = load_features(args.features)
    except Exception as exc:
        print(f"ERROR: failed loading feature table: {exc}", file=sys.stderr)
        return 1

    n_sub = df["subject"].nunique()
    print(f"Running personalization experiments on {len(df)} windows from {n_sub} subjects...")
    per_subject_df, summary_df = run_personalization_experiment(df, seed=SEED)

    # Save per-subject CSV
    args.out.parent.mkdir(parents=True, exist_ok=True)
    per_subject_df.to_csv(args.out, index=False)
    print(f"Saved per-subject results to {args.out}")

    # Log to MLflow
    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)

    for _, row in summary_df.iterrows():
        run_name = f"{row['method']}__{row['model']}__{row['group']}__k{row['k']}"
        with mlflow.start_run(run_name=run_name):
            log_condition_to_mlflow(row, args.out, seed=SEED)

    # Print summary table
    print("\nSummary Table:")
    print(format_summary_table(summary_df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
