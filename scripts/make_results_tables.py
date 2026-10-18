"""Generate markdown results tables directly from experiment CSV outputs.

Usage:
    .venv/Scripts/python.exe scripts/make_results_tables.py
    .venv/Scripts/python.exe scripts/make_results_tables.py \\
        --baselines docs/results/loso_baselines.csv \\
        --summary docs/results/personalization_summary.csv \\
        --per-subject docs/results/personalization_per_subject.csv \\
        --out docs/results/results_tables.md
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def build_binary_baselines_table(df: pd.DataFrame) -> str:
    """Build markdown table for binary classification baselines.

    Args:
        df: DataFrame loaded from loso_baselines.csv.

    Returns:
        Markdown table string.
    """
    binary_df = df[df["params.target"] == "label_binary"].copy()

    header = (
        "| Model | Feature Group | Macro F1 | "
        "Recall Class 0 (Non-Stress) | Recall Class 1 (Stress) |"
    )
    lines = [
        header,
        "|:---|:---|:---:|:---:|:---:|",
    ]

    for _, row in binary_df.iterrows():
        model = str(row["params.model"])
        group = str(row["params.feature_group"])
        f1 = f"{float(row['metrics.macro_f1']):.4f}"
        rec0 = f"{float(row['metrics.recall_class_0']):.4f}"
        rec1 = f"{float(row['metrics.recall_class_1']):.4f}"
        lines.append(f"| {model} | {group} | {f1} | {rec0} | {rec1} |")

    return "\n".join(lines)


def build_three_class_baselines_table(df: pd.DataFrame) -> str:
    """Build markdown table for 3-class classification baselines.

    Args:
        df: DataFrame loaded from loso_baselines.csv.

    Returns:
        Markdown table string.
    """
    multi_df = df[df["params.target"] == "label_3class"].copy()

    header = (
        "| Model | Feature Group | Macro F1 | "
        "Recall Class 1 (Baseline) | Recall Class 2 (Stress) | "
        "Recall Class 3 (Amusement) |"
    )
    lines = [
        header,
        "|:---|:---|:---:|:---:|:---:|:---:|",
    ]

    for _, row in multi_df.iterrows():
        model = str(row["params.model"])
        group = str(row["params.feature_group"])
        f1 = f"{float(row['metrics.macro_f1']):.4f}"
        rec1 = f"{float(row['metrics.recall_class_1']):.4f}"
        rec2 = f"{float(row['metrics.recall_class_2']):.4f}"
        rec3 = f"{float(row['metrics.recall_class_3']):.4f}"
        lines.append(f"| {model} | {group} | {f1} | {rec1} | {rec2} | {rec3} |")

    return "\n".join(lines)


def build_personalization_summary_table(df: pd.DataFrame) -> str:
    """Build markdown table for personalization summary (Method A and Method B2).

    Args:
        df: DataFrame loaded from personalization_summary.csv.

    Returns:
        Markdown table string.
    """
    methods_df = df[df["method"].isin(["method_a", "method_b2"])].copy()

    header = (
        "| Method | Model | Group | k | Mean F1 | Median F1 | "
        "Median Delta F1 | Improved/Worse/Equal | Wilcoxon p | Comparison |"
    )
    lines = [
        header,
        "|:---|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|",
    ]

    for _, row in methods_df.iterrows():
        method = str(row["method"])
        model = str(row["model"])
        group = str(row["group"])
        k = int(row["k"])
        mean_f1 = f"{float(row['macro_f1_mean']):.4f}"
        med_f1 = f"{float(row['macro_f1_median']):.4f}"
        med_delta = f"{float(row['median_delta_f1']):+.4f}"
        imp_worse_eq = f"{int(row['n_improved'])}/{int(row['n_worse'])}/{int(row['n_equal'])}"
        p_val = f"{float(row['wilcoxon_p']):.4f}"
        comp = str(row["comparison"])
        lines.append(
            f"| {method} | {model} | {group} | {k} | {mean_f1} | "
            f"{med_f1} | {med_delta} | {imp_worse_eq} | {p_val} | {comp} |"
        )

    return "\n".join(lines)


def compute_primary_comparison_stats(per_subject_df: pd.DataFrame) -> dict[str, Any]:
    """Compute exact primary comparison statistics from per-subject data.

    Primary condition: Logistic regression, feature group 'all', Method A (k=10) vs Reference (k=0).

    Args:
        per_subject_df: DataFrame loaded from personalization_per_subject.csv.

    Returns:
        Dict with primary comparison summary metrics.
    """
    ref_mask = (
        (per_subject_df["method"] == "reference")
        & (per_subject_df["model"] == "logistic")
        & (per_subject_df["group"] == "all")
        & (per_subject_df["k"] == 0)
    )
    ref_df = per_subject_df[ref_mask].set_index("subject")

    a10_mask = (
        (per_subject_df["method"] == "method_a")
        & (per_subject_df["model"] == "logistic")
        & (per_subject_df["group"] == "all")
        & (per_subject_df["k"] == 10)
    )
    a10_df = per_subject_df[a10_mask].set_index("subject")

    common_subjects = sorted(set(ref_df.index).intersection(set(a10_df.index)))

    ref_f1 = ref_df.loc[common_subjects, "macro_f1"].to_numpy(dtype=float)
    a10_f1 = a10_df.loc[common_subjects, "macro_f1"].to_numpy(dtype=float)
    deltas = a10_f1 - ref_f1

    non_zero = int(np.sum(deltas != 0))

    if non_zero > 0:
        res = wilcoxon(deltas, zero_method="wilcox")
        w_stat = float(res.statistic)
        p_val = float(res.pvalue)
    else:
        w_stat = float("nan")
        p_val = float("nan")

    return {
        "mean_ref_f1": float(np.mean(ref_f1)),
        "median_ref_f1": float(np.median(ref_f1)),
        "mean_a10_f1": float(np.mean(a10_f1)),
        "median_a10_f1": float(np.median(a10_f1)),
        "mean_delta_f1": float(np.mean(deltas)),
        "median_delta_f1": float(np.median(deltas)),
        "wilcoxon_w": w_stat,
        "wilcoxon_p": p_val,
        "non_zero_diffs": non_zero,
        "total_subjects": len(common_subjects),
    }


def build_primary_comparison_table(stats: dict[str, Any]) -> str:
    """Build markdown table for pre-registered primary comparison.

    Args:
        stats: Computed primary comparison statistics.

    Returns:
        Markdown table string.
    """
    mean_ref = f"{stats['mean_ref_f1']:.4f}"
    mean_a10 = f"{stats['mean_a10_f1']:.4f}"
    mean_d = f"{stats['mean_delta_f1']:+.4f}"

    med_ref = f"{stats['median_ref_f1']:.4f}"
    med_a10 = f"{stats['median_a10_f1']:.4f}"
    med_d = f"{stats['median_delta_f1']:+.4f}"

    w_val = f"{stats['wilcoxon_w']:.1f}"
    p_val = f"{stats['wilcoxon_p']:.4f}"
    nz_str = f"{stats['non_zero_diffs']} / {stats['total_subjects']}"

    lines = [
        "| Metric | Reference (k=0) | Method A (k=10) | Delta (A - Ref) |",
        "|:---|:---:|:---:|:---:|",
        f"| Mean Macro F1 | {mean_ref} | {mean_a10} | {mean_d} |",
        f"| Median Macro F1 | {med_ref} | {med_a10} | {med_d} |",
        f"| Wilcoxon W Statistic | - | - | {w_val} |",
        f"| Wilcoxon p-value | - | - | {p_val} |",
        f"| Non-Zero Differences | - | - | {nz_str} |",
    ]
    return "\n".join(lines)


def build_primary_per_subject_table(per_subject_df: pd.DataFrame) -> str:
    """Build markdown table for per-subject primary comparison results.

    Args:
        per_subject_df: DataFrame loaded from personalization_per_subject.csv.

    Returns:
        Markdown table string.
    """
    ref_mask = (
        (per_subject_df["method"] == "reference")
        & (per_subject_df["model"] == "logistic")
        & (per_subject_df["group"] == "all")
        & (per_subject_df["k"] == 0)
    )
    ref_df = per_subject_df[ref_mask].set_index("subject")

    a10_mask = (
        (per_subject_df["method"] == "method_a")
        & (per_subject_df["model"] == "logistic")
        & (per_subject_df["group"] == "all")
        & (per_subject_df["k"] == 10)
    )
    a10_df = per_subject_df[a10_mask].set_index("subject")

    common_subjects = sorted(set(ref_df.index).intersection(set(a10_df.index)))

    header = (
        "| Subject | F1 Reference | F1 Method A (k=10) | Delta F1 | "
        "Stress Recall (Ref) | Stress Recall (A10) | "
        "Not-Stress Recall (Ref) | Not-Stress Recall (A10) |"
    )
    lines = [
        header,
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for s in common_subjects:
        r_f1 = float(ref_df.loc[s, "macro_f1"])
        a_f1 = float(a10_df.loc[s, "macro_f1"])
        d_f1 = a_f1 - r_f1
        r_stress = float(ref_df.loc[s, "recall_stress"])
        a_stress = float(a10_df.loc[s, "recall_stress"])
        r_not_stress = float(ref_df.loc[s, "recall_not_stress"])
        a_not_stress = float(a10_df.loc[s, "recall_not_stress"])

        lines.append(
            f"| {s} | {r_f1:.4f} | {a_f1:.4f} | {d_f1:+.4f} | "
            f"{r_stress:.4f} | {a_stress:.4f} | "
            f"{r_not_stress:.4f} | {a_not_stress:.4f} |"
        )

    return "\n".join(lines)


def build_t_eval_table(per_subject_df: pd.DataFrame) -> str:
    """Build markdown table for frozen evaluation set sizes across subjects.

    Args:
        per_subject_df: DataFrame loaded from personalization_per_subject.csv.

    Returns:
        Markdown table string.
    """
    ref_mask = (
        (per_subject_df["method"] == "reference")
        & (per_subject_df["model"] == "logistic")
        & (per_subject_df["group"] == "all")
        & (per_subject_df["k"] == 0)
    )
    subset = per_subject_df[ref_mask].copy()

    min_n = float(subset["n_eval"].min())
    max_n = float(subset["n_eval"].max())
    mean_n = float(subset["n_eval"].mean())

    min_subjects = sorted(subset.loc[subset["n_eval"] == min_n, "subject"].tolist())
    max_subjects = sorted(subset.loc[subset["n_eval"] == max_n, "subject"].tolist())

    min_sub_str = ", ".join(str(s) for s in min_subjects)
    max_sub_str = ", ".join(str(s) for s in max_subjects)

    lines = [
        "| Statistic | Value (Windows) | Subject ID(s) |",
        "|:---|:---:|:---|",
        f"| Minimum Evaluation Set Size | {min_n:.1f} | Subject {min_sub_str} |",
        f"| Maximum Evaluation Set Size | {max_n:.1f} | Subject {max_sub_str} |",
        f"| Mean Evaluation Set Size | {mean_n:.2f} | All {len(subset)} subjects |",
    ]
    return "\n".join(lines)


def build_method_b2_theta_table(per_subject_df: pd.DataFrame) -> str:
    """Build markdown table for Method B2 threshold elevations (theta > 0.5).

    Args:
        per_subject_df: DataFrame loaded from personalization_per_subject.csv.

    Returns:
        Markdown table string.
    """
    b2_mask = (per_subject_df["method"] == "method_b2") & (per_subject_df["k"] > 0)
    b2_df = per_subject_df[b2_mask].copy()

    lines = [
        "| Model | Feature Group | k | Subjects with Raised Theta (theta > 0.5) | Total Subjects |",
        "|:---|:---|:---:|:---:|:---:|",
    ]

    grouped = b2_df.groupby(["model", "group", "k"], as_index=False)
    for (model, group, k), sub_df in grouped:
        raised = int((sub_df["theta"] > 0.5).sum())
        total = len(sub_df)
        lines.append(f"| {model} | {group} | {k} | {raised} | {total} |")

    return "\n".join(lines)


def generate_results_tables_markdown(
    baselines_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    per_subject_df: pd.DataFrame,
) -> str:
    """Generate the full results_tables.md markdown text.

    Args:
        baselines_df: Loaded loso_baselines.csv.
        summary_df: Loaded personalization_summary.csv.
        per_subject_df: Loaded personalization_per_subject.csv.

    Returns:
        Complete markdown formatted document.
    """
    primary_stats = compute_primary_comparison_stats(per_subject_df)

    intro = (
        "This document contains markdown tables generated directly from "
        "reproducible experimental results in `docs/results/`."
    )
    source_p1 = (
        "*Source: `docs/results/personalization_per_subject.csv` and "
        "`docs/results/personalization_summary.csv`*"
    )

    sections = [
        "# Benchmark Results Tables",
        "",
        intro,
        "",
        "## 1. Binary Classification Baselines",
        "",
        build_binary_baselines_table(baselines_df),
        "",
        "*Source: `docs/results/loso_baselines.csv`*",
        "",
        "## 2. 3-Class Classification Baselines",
        "",
        build_three_class_baselines_table(baselines_df),
        "",
        "*Source: `docs/results/loso_baselines.csv`*",
        "",
        "## 3. Personalization Summary (Method A and Method B2)",
        "",
        build_personalization_summary_table(summary_df),
        "",
        "*Source: `docs/results/personalization_summary.csv`*",
        "",
        "## 4. Pre-Registered Primary Comparison",
        "",
        "Condition: Logistic Regression, Feature Group `all`, Method A (k=10) vs Reference (k=0).",
        "",
        build_primary_comparison_table(primary_stats),
        "",
        source_p1,
        "",
        "## 5. Per-Subject Breakdown for Primary Comparison",
        "",
        build_primary_per_subject_table(per_subject_df),
        "",
        "*Source: `docs/results/personalization_per_subject.csv`*",
        "",
        "## 6. Evaluation Window Count Distribution (T_eval)",
        "",
        build_t_eval_table(per_subject_df),
        "",
        "*Source: `docs/results/personalization_per_subject.csv`*",
        "",
        "## 7. Method B2 Threshold Shift Counts (theta > 0.5)",
        "",
        build_method_b2_theta_table(per_subject_df),
        "",
        "*Source: `docs/results/personalization_per_subject.csv`*",
        "",
    ]
    return "\n".join(sections)


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate markdown results tables from experimental CSV files."
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("docs/results/loso_baselines.csv"),
        help="Path to loso_baselines.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/results/personalization_summary.csv"),
        help="Path to personalization_summary.csv",
    )
    parser.add_argument(
        "--per-subject",
        type=Path,
        default=Path("docs/results/personalization_per_subject.csv"),
        help="Path to personalization_per_subject.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/results/results_tables.md"),
        help="Output markdown file path",
    )
    return parser.parse_args(args)


def main(args: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parsed = parse_args(args)

    for path, label in [
        (parsed.baselines, "Baselines CSV"),
        (parsed.summary, "Summary CSV"),
        (parsed.per_subject, "Per-subject CSV"),
    ]:
        if not path.exists():
            print(f"Error: {label} file not found: {path}")
            return 1

    baselines_df = pd.read_csv(parsed.baselines)
    summary_df = pd.read_csv(parsed.summary)
    per_subject_df = pd.read_csv(parsed.per_subject)

    md_content = generate_results_tables_markdown(
        baselines_df=baselines_df,
        summary_df=summary_df,
        per_subject_df=per_subject_df,
    )

    parsed.out.parent.mkdir(parents=True, exist_ok=True)
    parsed.out.write_text(md_content, encoding="utf-8")
    print(f"Results tables successfully written to {parsed.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
