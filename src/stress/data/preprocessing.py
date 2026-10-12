"""Preprocessing: extract wrist-only .npz files from pickles."""

from pathlib import Path

import numpy as np

from stress.data.wesad_loader import LoaderResult, load_subject


def preprocess_subject(subject_id: int, data_dir: Path, output_dir: Path) -> tuple[Path, bool]:
    """Load subject, validate, save wrist+labels to .npz, free memory.

    Args:
        subject_id: Subject ID (e.g., 2).
        data_dir: Path to data/raw/WESAD/.
        output_dir: Path to data/processed/ (will be created if needed).

    Returns:
        (Path to saved .npz file, has_anomaly flag).

    Raises:
        FileNotFoundError: If pickle not found.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load subject with validation
    result: LoaderResult = load_subject(subject_id, data_dir)

    # Report anomalies loudly
    if result.duration_anomaly:
        print(result.anomaly_message)

    # Save to .npz
    npz_file = output_dir / f"S{subject_id}.npz"
    np.savez_compressed(
        npz_file,
        BVP=result.wrist["BVP"],
        EDA=result.wrist["EDA"],
        TEMP=result.wrist["TEMP"],
        ACC=result.wrist["ACC"],
        label=result.label,
    )

    return npz_file, result.duration_anomaly


def preprocess_all_subjects(data_dir: Path, output_dir: Path) -> dict[int, tuple[Path, bool]]:
    """Preprocess all discovered subjects.

    Args:
        data_dir: Path to data/raw/WESAD/.
        output_dir: Path to data/processed/.

    Returns:
        Dict mapping subject_id -> (npz_path, has_anomaly).
    """
    from stress.data.wesad_loader import discover_subjects

    subjects = discover_subjects(data_dir)
    results = {}

    for subject_id in subjects:
        try:
            npz_path, has_anomaly = preprocess_subject(subject_id, data_dir, output_dir)
            results[subject_id] = (npz_path, has_anomaly)
            status = "⚠️  ANOMALY" if has_anomaly else "✓"
            print(f"{status} S{subject_id}: {npz_path}")
        except Exception as e:
            print(f"✗ S{subject_id}: {e}")
            results[subject_id] = (None, True)

    return results
