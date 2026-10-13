"""Report script: print window counts per subject per class."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stress.data.preprocessing import load_npz_subject
from stress.data.windowing import WindowConfig, create_windows


def report_windows(
    processed_dir: Path,
    window_seconds: float = 60.0,
    overlap: float = 0.5,
    purity_threshold: float = 1.0,
) -> int:
    """Generate report of window counts per subject per class.

    Args:
        processed_dir: Path to data/processed/.
        window_seconds: Window length in seconds.
        overlap: Overlap fraction (0-1).
        purity_threshold: Purity threshold for windowing.

    Returns:
        0 on success, 1 if any error occurred.
    """
    processed_dir = Path(processed_dir)

    # Find all .npz files in processed_dir
    npz_files = sorted(processed_dir.glob("S*.npz"))

    if not npz_files:
        print(
            f"ERROR: no .npz files found in {processed_dir}",
            file=sys.stderr,
        )
        return 1

    config = WindowConfig(
        window_seconds=window_seconds, overlap=overlap, purity_threshold=purity_threshold
    )

    print(f"\n{'Subject':<10} {'Baseline':<12} {'Stress':<12} {'Amusement':<12} {'Total':<8}")
    print("-" * 54)

    all_totals = {"1": 0, "2": 0, "3": 0}
    errors: list[int] = []

    for npz_file in npz_files:
        # Extract subject ID from filename (e.g., "S2.npz" -> 2)
        subject_id = int(npz_file.stem[1:])

        try:
            subject_data = load_npz_subject(subject_id, processed_dir)
            windows = create_windows(subject_data["wrist"], subject_data["label"], config)

            counts = {"1": 0, "2": 0, "3": 0}
            for _, label in windows:
                counts[str(label)] += 1

            total = sum(counts.values())
            print(
                f"S{subject_id:<9} {counts['1']:<12} {counts['2']:<12} {counts['3']:<12} {total:<8}"
            )

            all_totals["1"] += counts["1"]
            all_totals["2"] += counts["2"]
            all_totals["3"] += counts["3"]
        except Exception as e:  # noqa: BLE001
            print(f"ERROR S{subject_id}: {e}", file=sys.stderr)
            errors.append(subject_id)

    if errors:
        print(
            f"\nERROR: failed to process {len(errors)} subject(s): {errors}",
            file=sys.stderr,
        )
        return 1

    grand_total = sum(all_totals.values())
    print("-" * 54)
    print(
        f"{'TOTAL':<10} {all_totals['1']:<12} {all_totals['2']:<12}"
        f" {all_totals['3']:<12} {grand_total:<8}"
    )

    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Report window counts per subject per class.")
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to processed data directory (default: data/processed)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=60.0,
        help="Window length in seconds (default: 60)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap fraction [0, 1) (default: 0.5)",
    )
    parser.add_argument(
        "--purity",
        type=float,
        default=1.0,
        help="Purity threshold for windowing (default: 1.0)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = report_windows(
        processed_dir=args.processed_dir,
        window_seconds=args.window_seconds,
        overlap=args.overlap,
        purity_threshold=args.purity,
    )
    sys.exit(exit_code)
