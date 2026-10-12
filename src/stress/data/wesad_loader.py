"""WESAD dataset loader with per-subject validation."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LoaderResult:
    """Result of loading a subject, with anomaly flags."""

    subject_id: int
    wrist: dict[str, np.ndarray]
    label: np.ndarray
    duration_anomaly: bool = False
    anomaly_message: str = ""


def discover_subjects(data_dir: Path) -> list[int]:
    """Discover subjects by scanning for S*.pkl files.

    Args:
        data_dir: Path to data/raw/WESAD/ directory.

    Returns:
        Sorted list of subject IDs (e.g., [2, 3, 4, 5, ...]).
    """
    data_dir = Path(data_dir)
    subjects = []
    for pkl_file in sorted(data_dir.glob("S*/S*.pkl")):
        try:
            subject_id = int(pkl_file.parent.name[1:])  # S2 -> 2
            subjects.append(subject_id)
        except (ValueError, IndexError):
            continue
    return sorted(set(subjects))


def load_subject(subject_id: int, data_dir: Path, tolerance_sec: float = 0.1) -> LoaderResult:
    """Load one subject from pickle, validate, flatten signals.

    Per-subject validation:
    - Check that len(signal) / fs ≈ len(labels) / 700 within tolerance
    - Flatten (N, 1) signals to 1D; keep ACC as (N, 3)
    - Report duration anomalies loudly, never silently ignore

    Args:
        subject_id: Subject ID (e.g., 2 for S2).
        data_dir: Path to data/raw/WESAD/ directory.
        tolerance_sec: Max allowed duration mismatch in seconds.

    Returns:
        LoaderResult with wrist dict, label array, and anomaly flags.

    Raises:
        FileNotFoundError: If pickle file does not exist.
    """
    data_dir = Path(data_dir)
    pkl_file = data_dir / f"S{subject_id}" / f"S{subject_id}.pkl"

    if not pkl_file.exists():
        raise FileNotFoundError(f"Pickle not found: {pkl_file}")

    with open(pkl_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # Extract wrist signals and labels
    wrist_raw = data["signal"]["wrist"]
    labels = data["label"].astype(np.uint8)

    # Expected reference duration from labels at 700 Hz
    label_duration = len(labels) / 700.0

    # Flatten (N, 1) signals; keep ACC as (N, 3)
    wrist = {}
    anomalies = []

    # BVP at 64 Hz
    bvp = wrist_raw["BVP"]
    if bvp.ndim == 2:
        bvp = bvp.flatten()
    wrist["BVP"] = bvp
    bvp_duration = len(bvp) / 64.0
    bvp_diff = abs(bvp_duration - label_duration)
    if bvp_diff > tolerance_sec:
        anomalies.append(
            f"BVP duration mismatch: {bvp_duration:.2f} s (labels: {label_duration:.2f} s), "
            f"diff={bvp_diff:.3f} s > tolerance={tolerance_sec:.3f} s"
        )

    # EDA at 4 Hz
    eda = wrist_raw["EDA"]
    if eda.ndim == 2:
        eda = eda.flatten()
    wrist["EDA"] = eda
    eda_duration = len(eda) / 4.0
    eda_diff = abs(eda_duration - label_duration)
    if eda_diff > tolerance_sec:
        anomalies.append(
            f"EDA duration mismatch: {eda_duration:.2f} s (labels: {label_duration:.2f} s), "
            f"diff={eda_diff:.3f} s > tolerance={tolerance_sec:.3f} s"
        )

    # TEMP at 4 Hz
    temp = wrist_raw["TEMP"]
    if temp.ndim == 2:
        temp = temp.flatten()
    wrist["TEMP"] = temp
    temp_duration = len(temp) / 4.0
    temp_diff = abs(temp_duration - label_duration)
    if temp_diff > tolerance_sec:
        anomalies.append(
            f"TEMP duration mismatch: {temp_duration:.2f} s (labels: {label_duration:.2f} s), "
            f"diff={temp_diff:.3f} s > tolerance={tolerance_sec:.3f} s"
        )

    # ACC at 32 Hz (keep as Nx3)
    acc = wrist_raw["ACC"]
    if acc.ndim == 2 and acc.shape[1] == 3:
        wrist["ACC"] = acc
    else:
        # Handle unexpected shape
        wrist["ACC"] = acc.reshape(-1, 3) if acc.size % 3 == 0 else acc
    acc_duration = wrist["ACC"].shape[0] / 32.0
    acc_diff = abs(acc_duration - label_duration)
    if acc_diff > tolerance_sec:
        anomalies.append(
            f"ACC duration mismatch: {acc_duration:.2f} s (labels: {label_duration:.2f} s), "
            f"diff={acc_diff:.3f} s > tolerance={tolerance_sec:.3f} s"
        )

    anomaly_message = ""
    duration_anomaly = bool(anomalies)
    if anomalies:
        anomaly_message = f"S{subject_id} anomalies:\n" + "\n".join(anomalies)

    return LoaderResult(
        subject_id=subject_id,
        wrist=wrist,
        label=labels,
        duration_anomaly=duration_anomaly,
        anomaly_message=anomaly_message,
    )
