"""Verify parity between scikit-learn and ONNX probability predictions.

Computes predictions across all rows of the features dataset using both
predict_stress_probability (scikit-learn) and onnx_stress_probability (ONNX).
Validates maximum absolute probability difference and label agreement.

Usage:
    python -m scripts.check_onnx_parity
        [--csv data/processed/features_phase2.csv]
        [--model-dir models]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from stress.api.model_loader import load_model, predict_stress_probability
from stress.api.onnx_backend import load_onnx_session, onnx_stress_probability
from stress.api.payload_builder import row_to_dict
from stress.api.schemas import PredictRequest

DEFAULT_CSV: Path = Path("data/processed/features_phase2.csv")
DEFAULT_MODEL_DIR: Path = Path("models")
ONNX_MODEL_FILENAME: str = "final_logistic_all.onnx"


class ParityResult(NamedTuple):
    max_abs_diff: float
    n_label_disagreements: int
    n_rows_within_1e_4_of_threshold: int
    ok: bool


def compare_probabilities(
    p_sk: list[float] | np.ndarray,
    p_onnx: list[float] | np.ndarray,
    tol: float = 1e-5,
) -> ParityResult:
    """Compare prediction probabilities from scikit-learn and ONNX backends.

    Args:
        p_sk: Sequence or array of scikit-learn class-1 probabilities.
        p_onnx: Sequence or array of ONNX class-1 probabilities.
        tol: Maximum allowable absolute difference.

    Returns:
        ParityResult with max_abs_diff, n_label_disagreements,
        n_rows_within_1e_4_of_threshold, and ok status.
    """
    arr_sk = np.asarray(p_sk, dtype=np.float64)
    arr_onnx = np.asarray(p_onnx, dtype=np.float64)

    if arr_sk.shape != arr_onnx.shape:
        raise ValueError(f"Array shapes do not match: {arr_sk.shape} vs {arr_onnx.shape}")

    diffs = np.abs(arr_sk - arr_onnx)
    max_abs_diff = float(np.max(diffs)) if diffs.size > 0 else 0.0

    # Strict p > 0.5 for binary classification label comparison
    labels_sk = arr_sk > 0.5
    labels_onnx = arr_onnx > 0.5
    n_label_disagreements = int(np.sum(labels_sk != labels_onnx))

    # Number of rows within 1e-4 of the 0.5 decision threshold
    near_threshold = (np.abs(arr_sk - 0.5) <= 1e-4) | (np.abs(arr_onnx - 0.5) <= 1e-4)
    n_rows_within_1e_4 = int(np.sum(near_threshold))

    ok = bool(max_abs_diff <= tol and n_label_disagreements == 0)

    return ParityResult(
        max_abs_diff=max_abs_diff,
        n_label_disagreements=n_label_disagreements,
        n_rows_within_1e_4_of_threshold=n_rows_within_1e_4,
        ok=ok,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check numerical and decision parity between scikit-learn and ONNX."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to feature CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help=f"Path to model directory (default: {DEFAULT_MODEL_DIR})",
    )
    args = parser.parse_args(argv)

    if not args.csv.is_file():
        print(f"Error: Feature CSV not found: {args.csv}", file=sys.stderr)
        return 1

    onnx_path = args.model_dir / ONNX_MODEL_FILENAME
    if not onnx_path.is_file():
        print(f"Error: ONNX model not found: {onnx_path}", file=sys.stderr)
        return 1

    try:
        loaded = load_model(args.model_dir)
        session = load_onnx_session(onnx_path)
    except Exception as exc:
        print(f"Error loading models: {exc}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.csv)
    features: list[str] = loaded.metadata["features"]

    hrv_nullable_fields: tuple[str, ...] = (
        "hrv_mean_hr",
        "hrv_hr_std",
        "hrv_sdnn",
        "hrv_rmssd",
        "hrv_pnn50",
    )
    p_sk_list: list[float] = []
    p_onnx_list: list[float] = []
    n_null_hrv_rows: int = 0

    for idx, row in df.iterrows():
        try:
            row_dict = row_to_dict(row, row_index=int(idx))
            request = PredictRequest.model_validate(row_dict)
            if any(getattr(request, f) is None for f in hrv_nullable_fields):
                n_null_hrv_rows += 1
            p_sk = predict_stress_probability(loaded, request)
            p_onnx = onnx_stress_probability(session, request, features)
            p_sk_list.append(p_sk)
            p_onnx_list.append(p_onnx)
        except Exception as exc:
            print(f"Error processing row {idx}: {exc}", file=sys.stderr)
            return 1

    result = compare_probabilities(p_sk_list, p_onnx_list, tol=1e-5)

    print(f"Evaluated rows: {len(p_sk_list)}")
    print(f"Rows with at least one null HRV field: {n_null_hrv_rows}")
    print(f"Max absolute diff: {result.max_abs_diff:.8e}")
    print(f"Label disagreements (p > 0.5): {result.n_label_disagreements}")
    print(f"Rows within 1e-4 of threshold: {result.n_rows_within_1e_4_of_threshold}")
    print(f"Parity OK: {result.ok}")

    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
