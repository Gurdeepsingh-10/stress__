"""ONNX model loader for the stress detection API.

Loads the exported logistic pipeline from ONNX format and validates
artifact integrity via SHA-256 hash, feature schema, and session shape checks.

Security note: ONNX files are data, not pickled code, and the hash check
guards against accidental mismatch only. The onnx_sha256 in the sidecar is
compared against the file on disk; it does not constitute a cryptographic
trust chain.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from stress.api.model_loader import ModelLoadError, get_default_model_dir, validate_features

if TYPE_CHECKING:
    import onnxruntime as ort

ONNX_MODEL_NAME = "final_logistic_all.onnx"
ONNX_SIDECAR_NAME = "final_logistic_all.onnx.json"
METADATA_NAME = "final_logistic_all.json"


@dataclass
class LoadedOnnxModel:
    """Container for a loaded ONNX inference session and its metadata."""

    session: ort.InferenceSession
    metadata: dict[str, Any]
    sidecar: dict[str, Any]


def load_onnx_model(model_dir: Path | str | None = None) -> LoadedOnnxModel:
    """Load ONNX model artifacts from directory with full integrity validation.

    Loads final_logistic_all.onnx, final_logistic_all.onnx.json (sidecar), and
    final_logistic_all.json (metadata). Validates:
    - All files present.
    - Metadata features match PredictRequest schema (via validate_features).
    - Sidecar features match metadata features.
    - SHA-256 of ONNX file bytes matches sidecar['onnx_sha256'].
    - ONNX session has exactly 1 input, input shape[1] == len(features), and 2 outputs.

    Args:
        model_dir: Directory containing ONNX artifacts. Defaults to the value of
            STRESS_MODEL_DIR env variable or 'models'.

    Returns:
        LoadedOnnxModel with session, metadata, and sidecar.

    Raises:
        ModelLoadError: For any missing file, integrity failure, or shape mismatch.
    """
    from stress.api.onnx_backend import load_onnx_session

    base_dir = Path(model_dir) if model_dir is not None else get_default_model_dir()
    onnx_path = base_dir / ONNX_MODEL_NAME
    sidecar_path = base_dir / ONNX_SIDECAR_NAME
    meta_path = base_dir / METADATA_NAME

    # --- File existence checks ---
    if not onnx_path.is_file():
        raise ModelLoadError(f"ONNX model not found: {onnx_path}")
    if not sidecar_path.is_file():
        raise ModelLoadError(f"ONNX sidecar not found: {sidecar_path}")
    if not meta_path.is_file():
        raise ModelLoadError(f"Model metadata not found: {meta_path}")

    # --- Load JSON files ---
    try:
        sidecar: dict[str, Any] = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelLoadError(f"Failed to parse ONNX sidecar {sidecar_path}: {exc}") from exc

    try:
        metadata: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelLoadError(f"Failed to parse metadata JSON {meta_path}: {exc}") from exc

    # --- Feature validation: metadata vs schema ---
    validate_features(metadata)

    # --- Feature validation: sidecar vs metadata ---
    meta_features: list[str] = metadata["features"]
    sidecar_features = sidecar.get("features")
    if sidecar_features != meta_features:
        raise ModelLoadError(
            f"Sidecar features do not match metadata features: "
            f"sidecar={sidecar_features!r} vs metadata={meta_features!r}"
        )

    # --- SHA-256 integrity check ---
    onnx_bytes = onnx_path.read_bytes()
    actual_sha256 = hashlib.sha256(onnx_bytes).hexdigest()
    expected_sha256 = sidecar.get("onnx_sha256", "")
    if actual_sha256 != expected_sha256:
        raise ModelLoadError(
            f"ONNX file SHA-256 mismatch: expected {expected_sha256!r}, got {actual_sha256!r}"
        )

    # --- Load ONNX session ---
    session = load_onnx_session(onnx_path)

    # --- Session shape validation ---
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    n_features = len(meta_features)

    if len(inputs) != 1:
        raise ModelLoadError(f"ONNX session must have exactly 1 input, got {len(inputs)}")
    input_shape = inputs[0].shape
    if len(input_shape) < 2 or input_shape[1] != n_features:
        raise ModelLoadError(
            f"ONNX input shape[1] must equal {n_features} features, got shape {input_shape}"
        )
    if len(outputs) != 2:
        raise ModelLoadError(f"ONNX session must have exactly 2 outputs, got {len(outputs)}")

    return LoadedOnnxModel(session=session, metadata=metadata, sidecar=sidecar)
