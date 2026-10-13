"""Preprocess WESAD subjects: extract wrist-only .npz files from pickles.

Usage:
    .venv/Scripts/python.exe scripts/preprocess_wesad.py
    .venv/Scripts/python.exe scripts/preprocess_wesad.py --subject S2
    .venv/Scripts/python.exe scripts/preprocess_wesad.py --data-dir data/raw/WESAD \\
        --out-dir data/processed --force
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from stress.data.preprocessing import preprocess_subject
from stress.data.wesad_loader import discover_subjects


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess WESAD subjects to wrist-only .npz files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/WESAD"),
        help="Path to WESAD raw directory (default: data/raw/WESAD)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed"),
        help="Path to output directory for .npz files (default: data/processed)",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Process only this subject, e.g. S2 (default: all discovered subjects)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .npz files (default: skip if already present)",
    )
    return parser.parse_args()


def main() -> int:
    """Run preprocessing and return exit code (0 = success, 1 = any failure)."""
    args = parse_args()

    # Resolve subject list
    if args.subject is not None:
        # Accept "S2" or bare "2"
        s = args.subject.lstrip("Ss")
        try:
            subject_ids = [int(s)]
        except ValueError:
            print(
                f"ERROR: cannot parse --subject {args.subject!r}; expect e.g. S2 or 2",
                file=sys.stderr,
            )
            return 1
    else:
        subject_ids = discover_subjects(args.data_dir)
        if not subject_ids:
            print(f"ERROR: no subjects found in {args.data_dir}", file=sys.stderr)
            return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    n_skipped = 0
    failed: list[int] = []
    anomaly_subjects: list[int] = []

    for subject_id in subject_ids:
        npz_path = args.out_dir / f"S{subject_id}.npz"

        # Skip if already exists and --force not given
        if npz_path.exists() and not args.force:
            size_mb = npz_path.stat().st_size / 1_048_576
            print(
                f"S{subject_id}: skip (already exists, {size_mb:.1f} MB)"
                " -- use --force to overwrite"
            )
            n_skipped += 1
            continue

        t0 = time.perf_counter()
        try:
            saved_path, has_anomaly = preprocess_subject(subject_id, args.data_dir, args.out_dir)
            elapsed = time.perf_counter() - t0
            size_mb = saved_path.stat().st_size / 1_048_576
            if has_anomaly:
                # Anomaly detail already printed by preprocess_subject
                print(
                    f"WARN S{subject_id}: loaded in {elapsed:.1f} s,"
                    f" wrote {saved_path} ({size_mb:.1f} MB) [ANOMALY]"
                )
                anomaly_subjects.append(subject_id)
            else:
                print(
                    f"OK   S{subject_id}: loaded in {elapsed:.1f} s,"
                    f" wrote {saved_path} ({size_mb:.1f} MB)"
                )
            n_processed += 1
        except FileNotFoundError as e:
            print(f"ERROR S{subject_id}: {e}", file=sys.stderr)
            failed.append(subject_id)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR S{subject_id}: unexpected error: {e}", file=sys.stderr)
            failed.append(subject_id)

    # Summary
    print(f"\nSummary: {n_processed} processed, {n_skipped} skipped, {len(failed)} failed")
    if anomaly_subjects:
        print(f"Duration anomalies in: {[f'S{s}' for s in anomaly_subjects]}")
    if failed:
        print(f"Failed subjects: {[f'S{s}' for s in failed]}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
