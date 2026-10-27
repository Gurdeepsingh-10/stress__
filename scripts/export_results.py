"""Export LOSO baselines and personalization summary results.

Provides subcommands to export:
1. loso-baselines: query MLflow runs, filter params and metrics, round to 4 decimals, and save CSV.
2. personalization-summary: aggregate per-subject personalization results,
   save CSV and text summary.

Usage:
    .venv/Scripts/python.exe -m scripts.export_results loso-baselines
    .venv/Scripts/python.exe -m scripts.export_results personalization-summary
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mlflow
import pandas as pd

from stress.personalization.experiment import aggregate_conditions, format_summary_table

DEFAULT_TRACKING_URI: str = "sqlite:///mlflow.db"
DEFAULT_LOSO_EXPERIMENT: str = "loso_baselines"
DEFAULT_LOSO_OUT: Path = Path("docs/results/loso_baselines.csv")

DEFAULT_PER_SUBJECT: Path = Path("data/processed/personalization_per_subject.csv")
DEFAULT_SUMMARY_CSV: Path = Path("docs/results/personalization_summary.csv")
DEFAULT_SUMMARY_TXT: Path = Path("docs/results/personalization_summary.txt")


def export_loso_baselines(
    tracking_uri: str = DEFAULT_TRACKING_URI,
    experiment_name: str = DEFAULT_LOSO_EXPERIMENT,
    out_path: Path | str = DEFAULT_LOSO_OUT,
) -> int:
    """Export MLflow runs for the LOSO baselines experiment to a CSV file.

    Filters columns to only those starting with 'params.' or 'metrics.',
    rounds values to 4 decimal places, and writes to CSV without an index.

    Args:
        tracking_uri: MLflow tracking URI.
        experiment_name: Name of the MLflow experiment.
        out_path: Destination path for the output CSV.

    Returns:
        0 on success, 1 on error.
    """
    out_path = Path(out_path)

    if tracking_uri.startswith("sqlite:///"):
        sqlite_file = Path(tracking_uri.replace("sqlite:///", ""))
        if not sqlite_file.exists():
            print(f"ERROR: SQLite tracking database not found: {sqlite_file}", file=sys.stderr)
            return 1

    mlflow.set_tracking_uri(tracking_uri)

    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        print(f"ERROR: MLflow experiment not found: {experiment_name}", file=sys.stderr)
        return 1

    df = mlflow.search_runs(experiment_names=[experiment_name])
    keep = [c for c in df.columns if c.startswith(("params.", "metrics."))]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df[keep].round(4).to_csv(out_path, index=False)
    return 0


def export_personalization_summary(
    per_subject: Path | str = DEFAULT_PER_SUBJECT,
    summary_csv: Path | str = DEFAULT_SUMMARY_CSV,
    summary_txt: Path | str = DEFAULT_SUMMARY_TXT,
) -> int:
    """Aggregate per-subject personalization results into summary CSV and TXT files.

    Args:
        per_subject: Path to the per-subject CSV results.
        summary_csv: Destination path for the summary CSV.
        summary_txt: Destination path for the formatted text summary.

    Returns:
        0 on success, 1 on error.
    """
    per_subject = Path(per_subject)
    summary_csv = Path(summary_csv)
    summary_txt = Path(summary_txt)

    if not per_subject.exists():
        print(f"ERROR: Per-subject file not found: {per_subject}", file=sys.stderr)
        return 1

    d = pd.read_csv(per_subject)
    s = aggregate_conditions(d)

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(summary_csv, index=False)

    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(format_summary_table(s) + "\n")

    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Export LOSO baselines and personalization summary results."
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: loso-baselines
    loso_parser = subparsers.add_parser(
        "loso-baselines", help="Export LOSO baselines from MLflow to CSV."
    )
    loso_parser.add_argument(
        "--tracking-uri",
        "--mlflow-uri",
        default=DEFAULT_TRACKING_URI,
        help=f"MLflow tracking URI (default: {DEFAULT_TRACKING_URI})",
    )
    loso_parser.add_argument(
        "--experiment",
        "--experiment-name",
        default=DEFAULT_LOSO_EXPERIMENT,
        help=f"MLflow experiment name (default: {DEFAULT_LOSO_EXPERIMENT})",
    )
    loso_parser.add_argument(
        "--out",
        "--out-path",
        type=Path,
        default=DEFAULT_LOSO_OUT,
        help=f"Output CSV path (default: {DEFAULT_LOSO_OUT})",
    )

    # Subcommand: personalization-summary
    pers_parser = subparsers.add_parser(
        "personalization-summary",
        help="Aggregate per-subject results into summary CSV and text files.",
    )
    pers_parser.add_argument(
        "--per-subject",
        type=Path,
        default=DEFAULT_PER_SUBJECT,
        help=f"Path to per-subject CSV (default: {DEFAULT_PER_SUBJECT})",
    )
    pers_parser.add_argument(
        "--summary-csv",
        type=Path,
        default=DEFAULT_SUMMARY_CSV,
        help=f"Output summary CSV path (default: {DEFAULT_SUMMARY_CSV})",
    )
    pers_parser.add_argument(
        "--summary-txt",
        type=Path,
        default=DEFAULT_SUMMARY_TXT,
        help=f"Output summary text path (default: {DEFAULT_SUMMARY_TXT})",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        0 on success, non-zero on failure.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1

    if args.subcommand == "loso-baselines":
        return export_loso_baselines(
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
            out_path=args.out,
        )
    elif args.subcommand == "personalization-summary":
        return export_personalization_summary(
            per_subject=args.per_subject,
            summary_csv=args.summary_csv,
            summary_txt=args.summary_txt,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
