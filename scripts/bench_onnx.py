"""Benchmark function-level inference latency: scikit-learn vs ONNX Runtime.

Measures end-to-end latency from PredictRequest payload to float probability
for three inference implementations:
1. predict_stress_probability (scikit-learn with DataFrame input)
2. sklearn_numpy_fn (scikit-learn with numpy array input)
3. onnx_stress_probability (ONNX Runtime session with numpy array input)

Usage:
    python -m scripts.bench_onnx [--csv data/processed/features_phase2.csv] [--model-dir models]
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import skl2onnx
import sklearn

from stress.api.model_loader import load_model, predict_stress_probability
from stress.api.onnx_backend import load_onnx_session, onnx_stress_probability
from stress.api.payload_builder import row_to_dict
from stress.api.schemas import PredictRequest

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

DEFAULT_CSV: Path = Path("data/processed/features_phase2.csv")
DEFAULT_MODEL_DIR: Path = Path("models")
ONNX_MODEL_FILENAME: str = "final_logistic_all.onnx"
WARMUP_CALLS: int = 200
TIMED_CALLS: int = 2000
NUM_REPEATS: int = 5


def sklearn_numpy_fn(
    pipeline: Pipeline,
    request: PredictRequest,
    features: list[str],
) -> float:
    """Predict class-1 probability using pipeline with a numpy array input.

    Builds a float64 numpy array of shape (1, 17) in metadata feature order,
    converting None to np.nan (matching pipeline imputer rules).
    Suppresses only the UserWarning regarding missing feature names.
    """
    row_values: list[float] = []
    for feat in features:
        val = getattr(request, feat)
        row_values.append(np.nan if val is None else float(val))

    arr = np.array([row_values], dtype=np.float64)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=r".*X does not have valid feature names.*",
        )
        proba = pipeline.predict_proba(arr)

    classes = list(pipeline.classes_)
    class_1_idx = classes.index(1)
    return float(proba[0, class_1_idx])


def assert_functions_agree(
    p_df: list[float] | np.ndarray,
    p_np: list[float] | np.ndarray,
    p_onnx: list[float] | np.ndarray,
    tol: float = 1e-5,
) -> None:
    """Assert that the three prediction paths agree within tolerance on all rows.

    Args:
        p_df: Probabilities from scikit-learn DataFrame path.
        p_np: Probabilities from scikit-learn numpy array path.
        p_onnx: Probabilities from ONNX Runtime path.
        tol: Maximum acceptable difference between any pair.

    Raises:
        ValueError: If any pair of predictions exceeds tolerance.
    """
    arr_df = np.asarray(p_df, dtype=np.float64)
    arr_np = np.asarray(p_np, dtype=np.float64)
    arr_on = np.asarray(p_onnx, dtype=np.float64)

    diff_df_np = np.abs(arr_df - arr_np)
    diff_df_on = np.abs(arr_df - arr_on)
    diff_np_on = np.abs(arr_np - arr_on)

    diffs = np.maximum(np.maximum(diff_df_np, diff_df_on), diff_np_on)
    max_diff = float(np.max(diffs)) if len(diffs) > 0 else 0.0

    if not (max_diff <= tol):
        bad_indices = [i for i, d in enumerate(diffs) if d > tol]
        details = [
            f"Row {i}: df={arr_df[i]:.8f}, np={arr_np[i]:.8f}, "
            f"onnx={arr_on[i]:.8f}, max_diff={diffs[i]:.8e}"
            for i in bad_indices[:10]
        ]
        raise ValueError(
            f"Functions disagree on {len(bad_indices)} rows (tol={tol}):\n" + "\n".join(details)
        )


def run_benchmark_calls(
    fn: Callable[[PredictRequest], float],
    requests: list[PredictRequest],
    n_calls: int,
) -> list[float]:
    """Run n_calls timing each individually with time.perf_counter in microseconds."""
    n_req = len(requests)
    latencies_us: list[float] = []
    for i in range(n_calls):
        req = requests[i % n_req]
        t0 = time.perf_counter()
        _ = fn(req)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)
    return latencies_us


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark scikit-learn vs ONNX Runtime inference latency."
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

    loaded = load_model(args.model_dir)
    session = load_onnx_session(onnx_path)
    features: list[str] = loaded.metadata["features"]

    df = pd.read_csv(args.csv)
    requests: list[PredictRequest] = []
    for idx, row in df.iterrows():
        row_dict = row_to_dict(row, row_index=int(idx))
        requests.append(PredictRequest.model_validate(row_dict))

    print("# Inference Benchmark: scikit-learn vs ONNX Runtime")
    print(f"- **Platform**: {platform.platform()} ({platform.processor() or platform.machine()})")
    print(f"- **Python**: {sys.version.split()[0]}")
    print(f"- **scikit-learn**: {sklearn.__version__}")
    print(f"- **onnxruntime**: {ort.__version__}")
    print(f"- **skl2onnx**: {skl2onnx.__version__}")
    print(f"- **onnx**: {onnx.__version__}")
    print(f"- **numpy**: {np.__version__}")
    print(f"- **CSV rows used**: {len(requests)}")
    print(
        f"- **Warmup calls**: {WARMUP_CALLS} | "
        f"**Timed calls**: {TIMED_CALLS} | "
        f"**Repeats**: {NUM_REPEATS}"
    )
    print()

    def sk_df_fn(req: PredictRequest) -> float:
        return predict_stress_probability(loaded, req)

    def sk_np_fn(req: PredictRequest) -> float:
        return sklearn_numpy_fn(loaded.pipeline, req, features)

    def onnx_fn(req: PredictRequest) -> float:
        return onnx_stress_probability(session, req, features)

    # Pre-benchmark parity verification on ALL rows
    p_df_list = [sk_df_fn(req) for req in requests]
    p_np_list = [sk_np_fn(req) for req in requests]
    p_on_list = [onnx_fn(req) for req in requests]

    try:
        assert_functions_agree(p_df_list, p_np_list, p_on_list, tol=1e-5)
    except ValueError as exc:
        print(f"Parity check failed before benchmark: {exc}", file=sys.stderr)
        return 1

    # Warmup
    for i in range(WARMUP_CALLS):
        req = requests[i % len(requests)]
        _ = sk_df_fn(req)
        _ = sk_np_fn(req)
        _ = onnx_fn(req)

    named_fns: list[tuple[str, Callable[[PredictRequest], float]]] = [
        ("sk_df", sk_df_fn),
        ("sk_np", sk_np_fn),
        ("onnx", onnx_fn),
    ]

    sk_df_medians: list[float] = []
    sk_df_p95s: list[float] = []
    sk_df_p99s: list[float] = []

    sk_np_medians: list[float] = []
    sk_np_p95s: list[float] = []
    sk_np_p99s: list[float] = []

    onnx_medians: list[float] = []
    onnx_p95s: list[float] = []
    onnx_p99s: list[float] = []

    results_table: list[dict[str, Any]] = []

    for r in range(NUM_REPEATS):
        shift = r % 3
        current_order = named_fns[shift:] + named_fns[:shift]
        order_names = [name for name, _ in current_order]

        run_results: dict[str, list[float]] = {}
        for name, fn in current_order:
            run_results[name] = run_benchmark_calls(fn, requests, TIMED_CALLS)

        df_med = float(np.median(run_results["sk_df"]))
        df_p95 = float(np.percentile(run_results["sk_df"], 95))
        df_p99 = float(np.percentile(run_results["sk_df"], 99))

        np_med = float(np.median(run_results["sk_np"]))
        np_p95 = float(np.percentile(run_results["sk_np"], 95))
        np_p99 = float(np.percentile(run_results["sk_np"], 99))

        on_med = float(np.median(run_results["onnx"]))
        on_p95 = float(np.percentile(run_results["onnx"], 95))
        on_p99 = float(np.percentile(run_results["onnx"], 99))

        ratio_df = df_med / on_med if on_med > 0 else 0.0
        ratio_np = np_med / on_med if on_med > 0 else 0.0

        sk_df_medians.append(df_med)
        sk_df_p95s.append(df_p95)
        sk_df_p99s.append(df_p99)

        sk_np_medians.append(np_med)
        sk_np_p95s.append(np_p95)
        sk_np_p99s.append(np_p99)

        onnx_medians.append(on_med)
        onnx_p95s.append(on_p95)
        onnx_p99s.append(on_p99)

        results_table.append(
            {
                "repeat": r + 1,
                "first": order_names[0],
                "df_med": df_med,
                "df_p95": df_p95,
                "df_p99": df_p99,
                "np_med": np_med,
                "np_p95": np_p95,
                "np_p99": np_p99,
                "on_med": on_med,
                "on_p95": on_p95,
                "on_p99": on_p99,
                "ratio_df": ratio_df,
                "ratio_np": ratio_np,
            }
        )

    # Markdown Table
    hdr = (
        "| Repeat | First | Sklearn DF Med (us) | Sklearn DF p95 (us) | Sklearn DF p99 (us) | "
        "Sklearn NP Med (us) | Sklearn NP p95 (us) | Sklearn NP p99 (us) | "
        "ONNX Med (us) | ONNX p95 (us) | ONNX p99 (us) | Sklearn DF/ONNX | Sklearn NP/ONNX |"
    )
    print(hdr)
    sep = "| :--- | :--- |" + " ---: |" * 11
    print(sep)
    for row in results_table:
        print(
            f"| {row['repeat']} | {row['first']} | "
            f"{row['df_med']:.1f} | {row['df_p95']:.1f} | {row['df_p99']:.1f} | "
            f"{row['np_med']:.1f} | {row['np_p95']:.1f} | {row['np_p99']:.1f} | "
            f"{row['on_med']:.1f} | {row['on_p95']:.1f} | {row['on_p99']:.1f} | "
            f"{row['ratio_df']:.2f}x | {row['ratio_np']:.2f}x |"
        )

    summary_df_med = float(np.median(sk_df_medians))
    summary_df_p95 = float(np.median(sk_df_p95s))
    summary_df_p99 = float(np.median(sk_df_p99s))

    summary_np_med = float(np.median(sk_np_medians))
    summary_np_p95 = float(np.median(sk_np_p95s))
    summary_np_p99 = float(np.median(sk_np_p99s))

    summary_on_med = float(np.median(onnx_medians))
    summary_on_p95 = float(np.median(onnx_p95s))
    summary_on_p99 = float(np.median(onnx_p99s))

    overall_ratio_df = summary_df_med / summary_on_med if summary_on_med > 0 else 0.0
    overall_ratio_np = summary_np_med / summary_on_med if summary_on_med > 0 else 0.0

    print(
        f"| **Median** | - | "
        f"**{summary_df_med:.1f}** | **{summary_df_p95:.1f}** | **{summary_df_p99:.1f}** | "
        f"**{summary_np_med:.1f}** | **{summary_np_p95:.1f}** | **{summary_np_p99:.1f}** | "
        f"**{summary_on_med:.1f}** | **{summary_on_p95:.1f}** | **{summary_on_p99:.1f}** | "
        f"**{overall_ratio_df:.2f}x** | **{overall_ratio_np:.2f}x** |"
    )
    print()
    print(f"**Speed ratio (Sklearn DF median / ONNX median)**: {overall_ratio_df:.2f}x")
    print(f"**Speed ratio (Sklearn NP median / ONNX median)**: {overall_ratio_np:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
