"""ONNX Runtime backend for low-latency stress probability inference.

onnxruntime is imported lazily inside functions so environments without
the [ml] optional dependency can import this module without error.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from stress.api.schemas import PredictRequest

if TYPE_CHECKING:
    import onnxruntime as ort


def load_onnx_session(path: str | Path) -> ort.InferenceSession:
    """Load an ONNX model into an InferenceSession with single-threaded CPU execution.

    Args:
        path: Path to the .onnx model file.

    Returns:
        Configured onnxruntime.InferenceSession.
    """
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )


def onnx_stress_probability(
    session: ort.InferenceSession,
    request: PredictRequest,
    features: list[str],
) -> float:
    """Predict class-1 (stress) probability using an ONNX inference session.

    Builds a float32 numpy array of shape (1, 17) in the exact metadata feature
    order. Nullable HRV values (None) are converted to np.nan following the same
    rule as predict_stress_probability.

    Class-1 selection:
        The ONNX model outputs probabilities as a 2D float tensor of shape (N, 2),
        where column 0 is class 0 and column 1 is class 1. Because pipeline classes_
        are verified to be strictly [0, 1] at export time, index 1 corresponds
        explicitly to class 1 (stress).

    Args:
        session: Active onnxruntime.InferenceSession.
        request: Validated PredictRequest payload.
        features: Ordered list of feature names from model metadata.

    Returns:
        Predicted probability of stress (class 1) as float.
    """
    row_values: list[float] = []
    for feat in features:
        val = getattr(request, feat)
        row_values.append(np.nan if val is None else float(val))

    arr = np.array([row_values], dtype=np.float32)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})
    # outputs[1] is the float tensor of shape (1, 2) containing [p0, p1]
    probabilities = outputs[1]
    return float(probabilities[0, 1])
