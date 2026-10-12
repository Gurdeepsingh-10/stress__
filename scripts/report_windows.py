"""Report script: print window counts per subject per class."""

from pathlib import Path

import numpy as np

from stress.data.windowing import WindowConfig, create_windows


def load_npz_subject(subject_id: int, processed_dir: Path) -> dict:
    """Load wrist signals and labels from .npz file.

    Args:
        subject_id: Subject ID (e.g., 2).
        processed_dir: Path to data/processed/.

    Returns:
        Dict with keys 'wrist' (dict of signals), 'label' (1D array).
    """
    npz_file = processed_dir / f"S{subject_id}.npz"
    data = np.load(npz_file, allow_pickle=False)

    wrist_signals = {
        "BVP": data["BVP"],
        "EDA": data["EDA"],
        "TEMP": data["TEMP"],
        "ACC": data["ACC"],
    }

    return {
        "wrist": wrist_signals,
        "label": data["label"],
    }


def report_windows(
    data_dir: Path, processed_dir: Path, window_seconds: float = 60.0, overlap: float = 0.5
) -> None:
    """Generate report of window counts per subject per class.

    Args:
        data_dir: Path to data/raw/WESAD/.
        processed_dir: Path to data/processed/.
        window_seconds: Window length in seconds.
        overlap: Overlap fraction (0-1).
    """
    from stress.data.wesad_loader import discover_subjects

    subjects = discover_subjects(data_dir)

    config = WindowConfig(window_seconds=window_seconds, overlap=overlap, purity_threshold=1.0)

    print(f"\n{'Subject':<10} {'Baseline':<12} {'Stress':<12} {'Amusement':<12} {'Total':<8}")
    print("-" * 54)

    all_totals = {"1": 0, "2": 0, "3": 0}

    for subject_id in subjects:
        npz_file = processed_dir / f"S{subject_id}.npz"
        if not npz_file.exists():
            print(f"S{subject_id:<9} (not preprocessed)")
            continue

        subject_data = load_npz_subject(subject_id, processed_dir)
        windows = create_windows(subject_data["wrist"], subject_data["label"], config)

        counts = {"1": 0, "2": 0, "3": 0}
        for _, label in windows:
            counts[str(label)] += 1

        total = sum(counts.values())
        print(f"S{subject_id:<9} {counts['1']:<12} {counts['2']:<12} {counts['3']:<12} {total:<8}")

        all_totals["1"] += counts["1"]
        all_totals["2"] += counts["2"]
        all_totals["3"] += counts["3"]

    grand_total = sum(all_totals.values())
    print("-" * 54)
    print(
        f"{'TOTAL':<10} {all_totals['1']:<12} {all_totals['2']:<12} "
        f"{all_totals['3']:<12} {grand_total:<8}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python scripts/report_windows.py <data_dir>")
        print("  Example: python scripts/report_windows.py data/raw/WESAD")
        sys.exit(1)

    data_dir = Path(sys.argv[1])
    processed_dir = data_dir.parent / "processed"

    report_windows(data_dir, processed_dir)
