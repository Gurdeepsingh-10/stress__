"""Feature-group definitions for ablation studies.

Groups are declared here and validated against the columns actually produced by
`stress.features.extract.extract_features_from_window`, so the ablation sets cannot
silently drift from the feature table.
"""

from __future__ import annotations

from collections.abc import Iterable

# Columns that identify a window rather than describe the physiology
NON_FEATURE_COLUMNS: tuple[str, ...] = (
    "subject",
    "window_start_s",
    "label_3class",
    "label_binary",
)

# "hrv" holds the heart rate and heart rate variability features derived from BVP
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "hrv": (
        "hrv_mean_hr",
        "hrv_hr_std",
        "hrv_sdnn",
        "hrv_rmssd",
        "hrv_pnn50",
        "hrv_num_beats",
    ),
    "eda": (
        "eda_mean",
        "eda_std",
        "eda_min",
        "eda_max",
        "eda_slope",
        "eda_num_peaks",
    ),
    "temp": (
        "temp_mean",
        "temp_std",
        "temp_slope",
    ),
    "acc": (
        "acc_magnitude_mean",
        "acc_magnitude_std",
    ),
}

# Ablation names in run order: one per group, then everything together
ABLATION_NAMES: tuple[str, ...] = (*FEATURE_GROUPS, "all")


def feature_columns(groups: Iterable[str], available: Iterable[str]) -> list[str]:
    """Resolve group names to feature column names.

    Args:
        groups: Group names, e.g. ("hrv", "eda").
        available: Column names present in the feature table.

    Returns:
        Column names in group declaration order, de-duplicated.

    Raises:
        KeyError: If a group name is unknown.
        ValueError: If a declared feature column is missing from available.
    """
    available_set = set(available)
    columns: list[str] = []

    for group in groups:
        if group not in FEATURE_GROUPS:
            raise KeyError(
                f"unknown feature group {group!r}; expected one of {sorted(FEATURE_GROUPS)}"
            )
        for column in FEATURE_GROUPS[group]:
            if column not in available_set:
                raise ValueError(
                    f"column {column!r} of group {group!r} missing from the feature table"
                )
            if column not in columns:
                columns.append(column)

    return columns


def ablation_sets(available: Iterable[str]) -> dict[str, list[str]]:
    """Build the ablation column sets: one per feature group, plus 'all'.

    Args:
        available: Column names present in the feature table.

    Returns:
        Mapping of ablation name -> feature columns, in ABLATION_NAMES order.

    Raises:
        KeyError: If a group name is unknown.
        ValueError: If a declared feature column is missing from available.
    """
    available = list(available)
    sets = {group: feature_columns([group], available) for group in FEATURE_GROUPS}
    sets["all"] = feature_columns(FEATURE_GROUPS, available)
    return sets
