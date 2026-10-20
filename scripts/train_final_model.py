"""Train the final stress detection model on all subjects and save artifacts.

Trains the logistic regression pipeline (SimpleImputer, StandardScaler, LogisticRegression)
on all available training data using the 17 features from PredictRequest schema.
Saves serialized model and companion metadata JSON for the FastAPI service.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from stress.api.schemas import PredictRequest
from stress.models.loso import build_pipeline

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

MODEL_FILENAME = "final_logistic_all.joblib"
METADATA_FILENAME = "final_logistic_all.json"
DEFAULT_CSV = Path("data/processed/features_phase2.csv")


def load_training_frame(csv_path: Path | str) -> pd.DataFrame:
    """Load processed features CSV and verify all required schema features exist.

    Args:
        csv_path: Path to the feature CSV.

    Returns:
        Loaded pandas DataFrame.

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If any required feature or target column is missing.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {path}")

    df = pd.read_csv(path)
    required_features = list(PredictRequest.model_fields.keys())
    missing = [col for col in required_features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing {len(missing)} required feature columns in CSV: {missing}")

    if "label_binary" not in df.columns:
        raise ValueError("Target column 'label_binary' not found in CSV")

    return df


def get_git_info() -> tuple[str, bool | None]:
    """Retrieve current git commit hash and dirty status.

    Never crashes; returns ('unknown', None) if git is not available or fails.

    Returns:
        Tuple of (commit_hash, is_dirty).
    """
    try:
        rev_proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit = rev_proc.stdout.strip()

        status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        is_dirty = bool(status_proc.stdout.strip())
        return commit, is_dirty
    except Exception:
        return "unknown", None


def build_metadata(
    pipeline: Pipeline,
    df: pd.DataFrame,
    git_info: tuple[str, bool | None] | None = None,
    trained_at_utc: str | None = None,
) -> dict[str, Any]:
    """Construct metadata dictionary describing model provenance and environment.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        df: Training DataFrame.
        git_info: Optional pre-computed (commit, is_dirty) tuple.
        trained_at_utc: Optional ISO UTC timestamp string.

    Returns:
        Structured metadata dictionary.
    """
    if git_info is None:
        git_commit, git_dirty = get_git_info()
    else:
        git_commit, git_dirty = git_info

    if trained_at_utc is None:
        trained_at_utc = datetime.now(UTC).isoformat()

    features = list(PredictRequest.model_fields.keys())
    counts = df["label_binary"].value_counts().sort_index().to_dict()
    class_counts = {int(k): int(v) for k, v in counts.items()}
    classes_ = [int(c) for c in getattr(pipeline, "classes_", [0, 1])]

    return {
        "model_name": "logistic_all",
        "model_version": "1.0.0",
        "target": "label_binary",
        "features": features,
        "n_features": len(features),
        "training_samples": len(df),
        "class_counts": class_counts,
        "classes_": classes_,
        "trained_at_utc": trained_at_utc,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "environment": {
            "python": sys.version.split()[0],
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }


def train_and_save(
    df: pd.DataFrame,
    out_dir: Path | str,
    git_info: tuple[str, bool | None] | None = None,
) -> tuple[Path, Path]:
    """Train logistic pipeline on all samples and persist model and metadata.

    Args:
        df: DataFrame containing the 17 schema features and 'label_binary'.
        out_dir: Target directory where artifacts will be saved.
        git_info: Optional git info tuple to avoid subprocess calls in tests.

    Returns:
        Tuple of (joblib_path, json_path).
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    features = list(PredictRequest.model_fields.keys())
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    if "label_binary" not in df.columns:
        raise ValueError("Missing target column 'label_binary'")

    x_train = df[features]
    y_train = df["label_binary"].astype(int)

    pipeline = build_pipeline("logistic", seed=42)
    pipeline.fit(x_train, y_train)

    meta = build_metadata(pipeline, df, git_info=git_info)

    joblib_path = out_path / MODEL_FILENAME
    json_path = out_path / METADATA_FILENAME

    joblib.dump(pipeline, joblib_path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return joblib_path, json_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final logistic regression stress detection model on all subjects."
    )
    default_out = os.environ.get("STRESS_MODEL_DIR", "models")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(default_out),
        help=f"Directory to save trained model artifacts (default: {default_out})",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to feature CSV (default: {DEFAULT_CSV})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = _parse_args(argv)
    print(f"Loading training data from {args.csv}...")
    df = load_training_frame(args.csv)
    print(f"Training final logistic model on {len(df)} samples...")
    joblib_path, json_path = train_and_save(df, args.out_dir)
    print(f"Saved pipeline to: {joblib_path}")
    print(f"Saved metadata to: {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
