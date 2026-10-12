"""Smoke tests for stress package."""

import stress


def test_version() -> None:
    """Verify package version is correct."""
    assert stress.__version__ == "0.1.0"
