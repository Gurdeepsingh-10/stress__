"""Inspect WESAD dataset structure and format.

Loads S2 from WESAD pickle files and prints:
- Top-level keys in the pickle file
- Wrist signal names, shapes, dtypes, NaN counts
- CSV header data (sampling rates, timestamps) vs. inferred rates
- Label segments showing condition order and duration
- NaN counts per label segment
"""

import csv
import pickle
import sys
from pathlib import Path

import numpy as np

# Module-level label name mapping
LABEL_NAMES = {
    0: "transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
}

DOCUMENTED_FS = {
    "BVP": 64.0,
    "EDA": 4.0,
    "TEMP": 4.0,
    "ACC": 32.0,
}


def read_csv_header(csv_path: Path) -> tuple[float, float] | None:
    """Read sampling rate and start timestamp from CSV header.

    Returns (timestamp, sampling_rate) or None if file doesn't exist.
    """
    if not csv_path.exists():
        return None

    try:
        with open(csv_path) as f:
            reader = csv.reader(f)
            timestamp_str = next(reader)[0]
            fs_str = next(reader)[0]
            return (float(timestamp_str), float(fs_str))
    except (StopIteration, ValueError, IndexError):
        return None


def main(data_dir: str) -> None:
    """Load and inspect S2 subject from WESAD dataset."""
    data_path = Path(data_dir)
    pkl_file = data_path / "S2" / "S2.pkl"

    if not pkl_file.exists():
        print(f"ERROR: {pkl_file} not found")
        sys.exit(1)

    print(f"Loading: {pkl_file}")
    with open(pkl_file, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    print("\n=== Top-level keys ===")
    print(sorted(data.keys()))

    print("\n=== Subject ===")
    print(f"Subject ID: {data['subject']}")

    # Use label array as the independent clock
    labels = data["label"]
    duration_sec = len(labels) / 700.0
    print("\n=== Duration (from labels at 700 Hz) ===")
    print(f"{duration_sec:.1f} seconds ({len(labels)} samples)")

    print("\n=== Wrist signals ===")
    wrist = data["signal"]["wrist"]
    for signal_name in sorted(wrist.keys()):
        signal = wrist[signal_name]
        if signal.ndim == 2:  # ACC is (N, 3)
            n_samples = signal.shape[0]
            shape_str = f"{signal.shape[0]}x{signal.shape[1]}"
        else:
            n_samples = len(signal)
            shape_str = str(n_samples)

        inferred_fs = n_samples / duration_sec
        documented_fs = DOCUMENTED_FS.get(signal_name, "?")
        diff = inferred_fs - documented_fs if documented_fs != "?" else 0

        is_float = signal.dtype in (float, np.float32, np.float64)
        nan_count = np.sum(np.isnan(signal)) if is_float else 0

        print(
            f"{signal_name:6s} shape=({shape_str:10s}) dtype={str(signal.dtype):10s} "
            f"inferred_fs={inferred_fs:7.2f} Hz  documented={documented_fs:5.1f} Hz  "
            f"diff={diff:+7.2f}  NaNs={nan_count:6d}"
        )

    print("\n=== CSV header data (wrist E4 device) ===")
    e4_data_dir = data_path / "S2" / "S2_E4_Data"
    if not e4_data_dir.exists():
        print(f"Directory not found: {e4_data_dir}")
        print(f"Contents of {data_path / 'S2'}:")
        for item in sorted((data_path / "S2").iterdir()):
            print(f"  {item.name}")
    else:
        for signal_name in ["BVP", "EDA", "TEMP", "ACC"]:
            csv_path = e4_data_dir / f"{signal_name}.csv"
            header_data = read_csv_header(csv_path)
            if header_data is not None:
                timestamp, csv_fs = header_data
                print(f"{signal_name:6s} CSV fs={csv_fs:6.1f} Hz  timestamp={timestamp:.0f}")
            else:
                status = "not found" if not csv_path.exists() else "read error"
                print(f"{signal_name:6s} {status}")

    print("\n=== Chest signals (keys only) ===")
    chest = data["signal"]["chest"]
    print(sorted(chest.keys()))

    print("\n=== Label segments ===")
    print("(condition runs in order)")
    current_label = labels[0]
    start_idx = 0
    segment_count = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            end_idx = i
            start_sec = start_idx / 700.0
            end_sec = end_idx / 700.0
            duration_seg = (end_idx - start_idx) / 700.0
            label_name = LABEL_NAMES.get(current_label, f"unknown({current_label})")
            print(
                f"  Segment {segment_count}: label={int(current_label)} ({label_name:12s}) "
                f"t=[{start_sec:7.1f}, {end_sec:7.1f}] s  duration={duration_seg:7.1f} s"
            )
            segment_count += 1
            current_label = labels[i]
            start_idx = i

    # Final segment
    end_idx = len(labels)
    start_sec = start_idx / 700.0
    end_sec = end_idx / 700.0
    duration_seg = (end_idx - start_idx) / 700.0
    label_name = LABEL_NAMES.get(current_label, f"unknown({current_label})")
    print(
        f"  Segment {segment_count}: label={int(current_label)} ({label_name:12s}) "
        f"t=[{start_sec:7.1f}, {end_sec:7.1f}] s  duration={duration_seg:7.1f} s"
    )

    print("\n=== Label distribution ===")
    unique, counts = np.unique(labels, return_counts=True)
    for val, count in sorted(zip(unique, counts, strict=False)):
        label_name = LABEL_NAMES.get(val, "unknown")
        sec = count / 700.0
        pct = 100 * count / len(labels)
        print(f"  {int(val)}: {label_name:12s} {count:7d} samples ({sec:7.1f} s, {pct:5.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/inspect_wesad.py <data_dir>")
        print("  Example: python scripts/inspect_wesad.py data/raw/WESAD")
        sys.exit(1)
    main(sys.argv[1])
