"""Update the README load test section from an aggregate load test summary.

Extracts:
  (a) the "### Per-Configuration Results" table, and
  (b) the paragraph under "### Notes"
from the summary markdown file, and replaces everything between the markers
``<!-- LOADTEST:START -->`` and ``<!-- LOADTEST:END -->`` in the README.

Usage:
    python -m scripts.update_readme_loadtest
        [--readme README.md]
        [--summary docs/results/loadtest/summary.md]
        [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

START_MARKER: str = "<!-- LOADTEST:START -->"
END_MARKER: str = "<!-- LOADTEST:END -->"
SECTION_CONFIG: str = "### Per-Configuration Results"
SECTION_NOTES: str = "### Notes"


def extract_table(summary_text: str) -> str:
    """Extract markdown table lines under '### Per-Configuration Results'."""
    lines = summary_text.splitlines()
    in_section = False
    table_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == SECTION_CONFIG:
            in_section = True
            continue
        if in_section:
            if line.startswith("|"):
                table_lines.append(line)
            elif table_lines and not line.startswith("|"):
                break
    if not table_lines:
        raise ValueError(f"No table rows found under '{SECTION_CONFIG}'")
    return "\n".join(table_lines)


def extract_notes(summary_text: str) -> str:
    """Extract the notes paragraph under '### Notes'."""
    lines = summary_text.splitlines()
    in_section = False
    para_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped == SECTION_NOTES:
            in_section = True
            continue
        if in_section:
            if stripped.startswith("#"):
                break
            if stripped:
                para_lines.append(stripped)
            elif para_lines:
                break
    if not para_lines:
        raise ValueError(f"No paragraph found under '{SECTION_NOTES}'")
    return " ".join(para_lines)


def build_replacement_block(
    summary_text: str,
    summary_path: Path,
) -> str:
    """Extract table and notes and build the inner content between markers."""
    table = extract_table(summary_text)
    notes = extract_notes(summary_text)
    summary_link = f"[{summary_path.as_posix()}]({summary_path.as_posix()})"
    return f"\n{table}\n\n{notes}\n\n{summary_link}\n"


def update_readme_content(
    readme_text: str,
    summary_text: str,
    summary_path: Path,
) -> str:
    """Validate markers and return updated README text."""
    start_count = readme_text.count(START_MARKER)
    if start_count == 0:
        raise ValueError(f"Start marker '{START_MARKER}' missing")
    if start_count > 1:
        raise ValueError(f"Start marker '{START_MARKER}' duplicated ({start_count} occurrences)")

    end_count = readme_text.count(END_MARKER)
    if end_count == 0:
        raise ValueError(f"End marker '{END_MARKER}' missing")
    if end_count > 1:
        raise ValueError(f"End marker '{END_MARKER}' duplicated ({end_count} occurrences)")

    start_idx = readme_text.index(START_MARKER)
    end_idx = readme_text.index(END_MARKER)
    if end_idx < start_idx:
        raise ValueError(f"End marker '{END_MARKER}' appears before start marker '{START_MARKER}'")

    inner = build_replacement_block(summary_text, summary_path)
    prefix = readme_text[: start_idx + len(START_MARKER)]
    suffix = readme_text[end_idx:]
    return f"{prefix}{inner}{suffix}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Update README load test section from aggregate summary."
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="Path to README.md",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/results/loadtest/summary.md"),
        help="Path to aggregate summary markdown file",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check whether README is up-to-date without writing",
    )

    args = parser.parse_args(argv)

    readme_path: Path = args.readme
    if not readme_path.is_file():
        print(f"Error: README file not found: {readme_path}", file=sys.stderr)
        return 1

    summary_path: Path = args.summary
    if not summary_path.is_file():
        print(f"Error: Summary file not found: {summary_path}", file=sys.stderr)
        return 1

    readme_text = readme_path.read_text(encoding="utf-8")
    summary_text = summary_path.read_text(encoding="utf-8")

    if SECTION_CONFIG not in summary_text:
        print(
            f"Error: Section '{SECTION_CONFIG}' missing in summary file: {summary_path}",
            file=sys.stderr,
        )
        return 1

    if SECTION_NOTES not in summary_text:
        print(
            f"Error: Section '{SECTION_NOTES}' missing in summary file: {summary_path}",
            file=sys.stderr,
        )
        return 1

    try:
        updated_text = update_readme_content(readme_text, summary_text, summary_path)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.check:
        if readme_text != updated_text:
            print(
                f"Error: README block at {readme_path} differs from generated content",
                file=sys.stderr,
            )
            return 1
        return 0

    if readme_text != updated_text:
        readme_path.write_text(updated_text, encoding="utf-8")
        print(f"Updated {readme_path} with load test results from {summary_path}")
    else:
        print(f"{readme_path} is already up to date")

    return 0


if __name__ == "__main__":
    sys.exit(main())
