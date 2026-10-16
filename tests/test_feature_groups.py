"""Unit tests for the feature-group ablation plumbing."""

import numpy as np
import pytest

from stress.features.extract import extract_features_from_window
from stress.models.feature_groups import (
    FEATURE_GROUPS,
    NON_FEATURE_COLUMNS,
    ablation_sets,
    feature_columns,
)


def _window() -> dict[str, np.ndarray]:
    """One synthetic 60 s window: 1.2 Hz beats, flat EDA/TEMP, gravity on Z."""
    t = np.arange(3840) / 64.0
    return {
        "BVP": np.sin(2 * np.pi * 1.2 * t),
        "EDA": np.ones(240),
        "TEMP": np.ones(240) * 36.5,
        "ACC": np.tile(np.array([0.0, 0.0, 64.0]), (1920, 1)),
    }


def test_groups_cover_every_extracted_feature() -> None:
    """Groups plus non-feature columns must cover exactly what the extractor emits."""
    produced = set(
        extract_features_from_window(_window(), subject_id=1, window_start_s=0.0, label_3class=1)
    )

    declared = set(NON_FEATURE_COLUMNS)
    for columns in FEATURE_GROUPS.values():
        declared.update(columns)

    assert produced == declared


def test_groups_are_disjoint_and_non_empty() -> None:
    """No feature belongs to two groups and no group is empty."""
    seen: set[str] = set()
    for group, columns in FEATURE_GROUPS.items():
        assert columns, f"group {group} is empty"
        assert not seen.intersection(columns), f"group {group} overlaps another group"
        seen.update(columns)


def test_ablation_sets_have_one_set_per_group_plus_all() -> None:
    """ablation_sets returns each group plus 'all' as the concatenation of groups."""
    available = [column for columns in FEATURE_GROUPS.values() for column in columns]

    sets = ablation_sets(available)

    assert list(sets) == [*FEATURE_GROUPS, "all"]
    for group, columns in FEATURE_GROUPS.items():
        assert sets[group] == list(columns)
    assert sets["all"] == available


def test_feature_columns_rejects_unknown_group() -> None:
    """An unknown group name is an error, not an empty selection."""
    with pytest.raises(KeyError, match="unknown feature group"):
        feature_columns(["nope"], [])


def test_feature_columns_rejects_missing_column() -> None:
    """A group whose column is absent from the table is an error."""
    with pytest.raises(ValueError, match="missing from the feature table"):
        feature_columns(["eda"], ["eda_mean"])


def test_feature_columns_deduplicates() -> None:
    """Repeating a group does not repeat its columns."""
    available = list(FEATURE_GROUPS["hrv"])

    assert feature_columns(["hrv", "hrv"], available) == list(FEATURE_GROUPS["hrv"])
