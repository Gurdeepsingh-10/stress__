"""Export trained scikit-learn logistic regression pipeline to ONNX format.

Converts final_logistic_all.joblib to ONNX format with float32 input tensor and
probabilities output tensor (zipmap=False). Generates sidecar JSON metadata.

Usage:
    python -m scripts.export_onnx [--model-dir models] [--out-dir models]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import skl2onnx
import sklearn
from skl2onnx.common.data_types import FloatTensorType

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline

TARGET_OPSET: int = 17
SOURCE_MODEL_NAME: str = "final_logistic_all.joblib"
SOURCE_METADATA_NAME: str = "final_logistic_all.json"
OUTPUT_MODEL_NAME: str = "final_logistic_all.onnx"
OUTPUT_METADATA_NAME: str = "final_logistic_all.onnx.json"


def convert_pipeline(
    pipeline: Pipeline,
    n_features: int,
    target_opset: int = TARGET_OPSET,
) -> bytes:
    """Convert a scikit-learn classification pipeline to ONNX bytes.

    Verifies pipeline classes_ == [0, 1] before conversion to ensure binary
    classification output column indexing is deterministic.

    Args:
        pipeline: Fitted scikit-learn pipeline or estimator.
        n_features: Number of input features.
        target_opset: ONNX opset version for conversion.

    Returns:
        Serialized ONNX model as bytes.

    Raises:
        ValueError: If pipeline classes_ is not [0, 1].
    """
    classes = getattr(pipeline, "classes_", None)
    if classes is None or list(classes) != [0, 1]:
        raise ValueError(f"Expected pipeline classes_ [0, 1], got {classes}")

    initial_types = [("float_input", FloatTensorType([None, n_features]))]
    onx = skl2onnx.convert_sklearn(
        pipeline,
        initial_types=initial_types,
        options={"zipmap": False},
        target_opset=target_opset,
    )
    return onx.SerializeToString()


def build_sidecar(
    joblib_bytes: bytes,
    onnx_bytes: bytes,
    metadata: dict[str, Any],
    opset: int = TARGET_OPSET,
) -> dict[str, Any]:
    """Build metadata sidecar describing ONNX artifact provenance.

    Args:
        joblib_bytes: Raw bytes of the source joblib file.
        onnx_bytes: Raw bytes of the generated ONNX model.
        metadata: Metadata dictionary loaded from the source model JSON.
        opset: Target ONNX opset version.

    Returns:
        Dictionary containing file hashes, feature metadata, and library versions.
    """
    source_joblib_sha256 = hashlib.sha256(joblib_bytes).hexdigest()
    onnx_sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    features = list(metadata.get("features", []))
    n_features = int(metadata.get("n_features", len(features)))

    versions = {
        "skl2onnx": skl2onnx.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
    }

    return {
        "source_joblib_sha256": source_joblib_sha256,
        "onnx_sha256": onnx_sha256,
        "opset": opset,
        "n_features": n_features,
        "features": features,
        "library_versions": versions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export scikit-learn pipeline to ONNX with metadata sidecar."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing source model artifacts (default: models)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("models"),
        help="Directory to save exported ONNX artifacts (default: models)",
    )
    args = parser.parse_args(argv)

    joblib_path = args.model_dir / SOURCE_MODEL_NAME
    json_path = args.model_dir / SOURCE_METADATA_NAME

    if not joblib_path.is_file():
        print(f"Error: Source model file not found: {joblib_path}", file=sys.stderr)
        return 1
    if not json_path.is_file():
        print(f"Error: Source metadata file not found: {json_path}", file=sys.stderr)
        return 1

    joblib_bytes = joblib_path.read_bytes()
    pipeline = joblib.load(joblib_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))

    n_features = int(metadata.get("n_features", len(metadata.get("features", []))))
    try:
        onnx_bytes = convert_pipeline(pipeline, n_features=n_features, target_opset=TARGET_OPSET)
    except Exception as exc:
        print(f"Error converting pipeline to ONNX: {exc}", file=sys.stderr)
        return 1

    sidecar = build_sidecar(
        joblib_bytes=joblib_bytes,
        onnx_bytes=onnx_bytes,
        metadata=metadata,
        opset=TARGET_OPSET,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_onnx = args.out_dir / OUTPUT_MODEL_NAME
    out_json = args.out_dir / OUTPUT_METADATA_NAME

    out_onnx.write_bytes(onnx_bytes)
    out_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    print(f"Exported ONNX model: {out_onnx}")
    print(f"Exported ONNX sidecar: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
