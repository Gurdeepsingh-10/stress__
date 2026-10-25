"""Unit tests for scripts/update_readme_loadtest.py.

All tests use synthetic markdown files created in tmp_path (no real data).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.update_readme_loadtest import (
    END_MARKER,
    SECTION_CONFIG,
    SECTION_NOTES,
    START_MARKER,
    main,
    update_readme_content,
)


def _make_sample_summary(
    config_section: str = SECTION_CONFIG,
    notes_section: str = SECTION_NOTES,
    table: str = "| Users | Repeats | Latency |\n| :--- | ---: | ---: |\n| A | B | C |",
    notes: str = "This is a synthetic test notes paragraph for testing.",
) -> str:
    return (
        f"# Load Test Summary\n\n"
        f"{config_section}\n\n"
        f"{table}\n\n"
        f"### Per-Run Results\n\n"
        f"| Run | Detail |\n| :--- | ---: |\n| r1 | 10 |\n\n"
        f"{notes_section}\n\n"
        f"{notes}\n"
    )


def _make_sample_readme(
    inner_content: str = "\nOld placeholder content\n",
    start_marker: str = START_MARKER,
    end_marker: str = END_MARKER,
    prefix: str = "# Project\n\nBefore load test\n\n",
    suffix: str = "\n\nAfter load test\n",
) -> str:
    return f"{prefix}{start_marker}{inner_content}{end_marker}{suffix}"


# ---------------------------------------------------------------------------
# Happy path: replace works and is idempotent
# ---------------------------------------------------------------------------


def test_replace_works(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"

    prefix = "# Project\n\nBefore load test\n\n"
    suffix = "\n\nAfter load test\n"
    table = "| Users | Repeats | Latency |\n| :--- | ---: | ---: |\n| A | B | C |"
    notes = "This is a synthetic test notes paragraph for testing."

    readme.write_text(
        _make_sample_readme(
            inner_content="\nOld placeholder content\n",
            prefix=prefix,
            suffix=suffix,
        ),
        encoding="utf-8",
    )
    summary.write_text(_make_sample_summary(table=table, notes=notes), encoding="utf-8")

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 0

    content = readme.read_text(encoding="utf-8")
    link_line = f"[{summary.as_posix()}]({summary.as_posix()})"
    expected = f"{prefix}{START_MARKER}\n{table}\n\n{notes}\n\n{link_line}\n{END_MARKER}{suffix}"
    assert content == expected
    assert content.startswith(prefix)
    assert content.endswith(suffix)
    assert "Old placeholder content" not in content
    assert content.count(START_MARKER) == 1
    assert content.count(END_MARKER) == 1


def test_per_run_table_rows_not_in_readme(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"

    readme.write_text(_make_sample_readme(), encoding="utf-8")
    summary.write_text(_make_sample_summary(), encoding="utf-8")

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 0

    content = readme.read_text(encoding="utf-8")
    assert "| r1 | 10 |" not in content
    assert "| Users | Repeats | Latency |" in content


def test_multiline_notes_paragraph_joined_with_single_space(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"

    multiline_notes = "Line one of notes.\nLine two of notes."
    readme.write_text(_make_sample_readme(), encoding="utf-8")
    summary.write_text(_make_sample_summary(notes=multiline_notes), encoding="utf-8")

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 0

    content = readme.read_text(encoding="utf-8")
    assert "Line one of notes. Line two of notes." in content
    assert "Line one of notes.\nLine two of notes." not in content


def test_notes_stops_at_blank_line_ignoring_second_paragraph(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"

    first_paragraph = "First synthetic notes paragraph."
    second_paragraph = "Second paragraph must not be copied."
    notes = f"{first_paragraph}\n\n{second_paragraph}"

    readme_text = _make_sample_readme()
    summary_text = _make_sample_summary(notes=notes)
    readme.write_text(readme_text, encoding="utf-8")
    summary.write_text(summary_text, encoding="utf-8")

    result = update_readme_content(readme_text, summary_text, summary)

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 0
    content = readme.read_text(encoding="utf-8")
    assert content == result

    # a. the first paragraph text is in the result
    assert first_paragraph in result
    # b. "Second paragraph must not be copied." is NOT in the result
    assert second_paragraph not in result
    # c. the result's notes block equals the first paragraph only
    inner = result.split(START_MARKER)[1].split(END_MARKER)[0].strip()
    _, notes_block, _ = inner.split("\n\n")
    assert notes_block == first_paragraph


def test_idempotent(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"

    readme.write_text(_make_sample_readme(), encoding="utf-8")
    summary.write_text(_make_sample_summary(), encoding="utf-8")

    rc1 = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc1 == 0
    first_run_bytes = readme.read_bytes()

    rc2 = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc2 == 0
    second_run_bytes = readme.read_bytes()

    assert first_run_bytes == second_run_bytes


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_error_readme_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    summary = tmp_path / "summary.md"
    summary.write_text(_make_sample_summary(), encoding="utf-8")
    nonexistent = tmp_path / "NONEXISTENT_README.md"

    rc = main(["--readme", str(nonexistent), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "README file not found" in err
    assert "NONEXISTENT_README.md" in err


def test_error_summary_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(_make_sample_readme(), encoding="utf-8")
    nonexistent = tmp_path / "NONEXISTENT_SUMMARY.md"

    rc = main(["--readme", str(readme), "--summary", str(nonexistent)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "Summary file not found" in err
    assert "NONEXISTENT_SUMMARY.md" in err


def test_error_start_marker_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    # End marker present, start marker missing
    readme.write_text(f"# Project\n\n{END_MARKER}\n", encoding="utf-8")
    summary.write_text(_make_sample_summary(), encoding="utf-8")

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert START_MARKER in err
    assert "missing" in err


def test_error_end_marker_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    # Start marker present, end marker missing
    readme.write_text(f"# Project\n\n{START_MARKER}\n", encoding="utf-8")
    summary.write_text(_make_sample_summary(), encoding="utf-8")

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert END_MARKER in err
    assert "missing" in err


def test_error_start_marker_duplicated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    readme.write_text(
        f"{START_MARKER}\nBlock 1\n{START_MARKER}\n{END_MARKER}\n",
        encoding="utf-8",
    )
    summary.write_text(_make_sample_summary(), encoding="utf-8")

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert START_MARKER in err
    assert "duplicated" in err


def test_error_end_marker_duplicated(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    readme.write_text(
        f"{START_MARKER}\nBlock\n{END_MARKER}\nExtra\n{END_MARKER}\n",
        encoding="utf-8",
    )
    summary.write_text(_make_sample_summary(), encoding="utf-8")

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert END_MARKER in err
    assert "duplicated" in err


def test_error_end_before_start(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    readme.write_text(
        f"{END_MARKER}\nInverted\n{START_MARKER}\n",
        encoding="utf-8",
    )
    summary.write_text(_make_sample_summary(), encoding="utf-8")

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "appears before start marker" in err


def test_error_summary_missing_config_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    readme.write_text(_make_sample_readme(), encoding="utf-8")
    # Summary without ### Per-Configuration Results
    summary.write_text(
        f"# Summary\n\n{SECTION_NOTES}\n\nSome notes paragraph\n",
        encoding="utf-8",
    )

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert SECTION_CONFIG in err


def test_error_summary_missing_notes_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    readme.write_text(_make_sample_readme(), encoding="utf-8")
    # Summary without ### Notes
    summary.write_text(
        f"# Summary\n\n{SECTION_CONFIG}\n\n| A | B |\n|---|---|\n| 1 | 2 |\n",
        encoding="utf-8",
    )

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert SECTION_NOTES in err


def test_error_summary_missing_table_rows(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    readme.write_text(_make_sample_readme(), encoding="utf-8")
    # Section present, but no markdown table rows
    summary.write_text(
        f"# Summary\n\n{SECTION_CONFIG}\n\nNo rows here\n\n{SECTION_NOTES}\n\nNotes\n",
        encoding="utf-8",
    )

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "No table rows found" in err


def test_error_summary_missing_notes_paragraph(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    readme.write_text(_make_sample_readme(), encoding="utf-8")
    # Section present, but empty paragraph
    summary.write_text(
        f"# Summary\n\n{SECTION_CONFIG}\n\n| A |\n|---|\n| 1 |\n\n{SECTION_NOTES}\n\n",
        encoding="utf-8",
    )

    rc = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "No paragraph found" in err


# ---------------------------------------------------------------------------
# --check behavior
# ---------------------------------------------------------------------------


def test_check_returns_1_when_differs_and_does_not_modify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    initial_readme = _make_sample_readme()
    readme.write_text(initial_readme, encoding="utf-8")
    summary.write_text(_make_sample_summary(), encoding="utf-8")

    rc = main(["--readme", str(readme), "--summary", str(summary), "--check"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "differs from generated content" in err

    # Ensure file was not modified
    assert readme.read_text(encoding="utf-8") == initial_readme


def test_check_returns_0_when_matching_and_does_not_modify(tmp_path: Path) -> None:
    readme = tmp_path / "README.md"
    summary = tmp_path / "summary.md"
    readme.write_text(_make_sample_readme(), encoding="utf-8")
    summary.write_text(_make_sample_summary(), encoding="utf-8")

    # First update the file so it matches generated content
    rc_update = main(["--readme", str(readme), "--summary", str(summary)])
    assert rc_update == 0
    updated_content = readme.read_text(encoding="utf-8")

    # Now run with --check
    rc_check = main(["--readme", str(readme), "--summary", str(summary), "--check"])
    assert rc_check == 0
    assert readme.read_text(encoding="utf-8") == updated_content


# ---------------------------------------------------------------------------
# Module invocation via -m
# ---------------------------------------------------------------------------


def test_cli_module_execution_help() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    res = subprocess.run(
        [sys.executable, "-m", "scripts.update_readme_loadtest", "--help"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--readme" in res.stdout
    assert "--summary" in res.stdout
    assert "--check" in res.stdout
