from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Friedman + Nemenyi analysis and generate a critical difference diagram "
            "from a sequence x method metric table."
        )
    )
    parser.add_argument("input", type=Path, nargs="?", help="Input CSV/TSV file.")
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Method columns to compare, in display order before ranking.",
    )
    parser.add_argument(
        "--sequence-column",
        default="sequence",
        help="Sequence name column. Default: sequence",
    )
    parser.add_argument(
        "--metric-name",
        default="RMSE",
        help="Metric label used in reports. Default: RMSE",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for Friedman/Nemenyi. Default: 0.05",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        default=True,
        help="Rank smaller values as better. Enabled by default.",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        help="Rank larger values as better.",
    )
    parser.add_argument(
        "--exclude-sequences",
        nargs="*",
        default=[],
        help="Exact sequence names to exclude after trimming whitespace.",
    )
    parser.add_argument(
        "--exclude-substrings",
        nargs="*",
        default=[],
        help="Exclude rows whose sequence contains any of these substrings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("friedman_nemenyi_output"),
        help="Output directory. Default: friedman_nemenyi_output",
    )
    parser.add_argument(
        "--pairwise-pattern",
        default="wilcoxon_*.csv",
        help="Glob pattern for pairwise Wilcoxon CSV files when no input table is provided.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Datasets to include from pairwise files, for example: TUM bonn",
    )
    parser.add_argument(
        "--metric-column",
        default=None,
        help="Metric suffix in pairwise CSV columns, for example RMSE or MEAN. Defaults to --metric-name.",
    )
    return parser.parse_args()


def detect_separator(file_path: Path) -> str:
    for sep in (",", ";", "\t"):
        try:
            rows = pd.read_csv(file_path, sep=sep)
        except Exception:
            continue
        if len(rows.columns) >= 2:
            return sep
    raise ValueError(f"Could not detect a valid separator for {file_path}")


def find_column_case_insensitive(rows: pd.DataFrame, expected_name: str) -> str | None:
    expected_lower = expected_name.lower()
    for column in rows.columns:
        if column.lower() == expected_lower:
            return column
    return None


def build_wide_table_from_pairwise(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    pairwise_files = sorted(Path(".").glob(args.pairwise_pattern))
    if not pairwise_files:
        raise FileNotFoundError(f"No pairwise files matched: {args.pairwise_pattern}")

    datasets_filter = {item.lower() for item in args.datasets} if args.datasets else set()
    metric_suffix = (args.metric_column or args.metric_name).upper()
    anchor_method = args.methods[0]
    required_methods = list(args.methods[1:])

    dataset_to_files: dict[str, list[Path]] = {}
    for file_path in pairwise_files:
        parts = file_path.stem.split("_")
        if len(parts) < 5 or parts[0].lower() != "wilcoxon":
            continue
        dataset_name = parts[1]
        if datasets_filter and dataset_name.lower() not in datasets_filter:
            continue
        dataset_to_files.setdefault(dataset_name, []).append(file_path)

    dataset_frames: list[pd.DataFrame] = []
    for dataset_name, dataset_files in sorted(dataset_to_files.items()):
        merged: pd.DataFrame | None = None
        available_methods: set[str] = set()

        for current_file in sorted(dataset_files):
            current_parts = current_file.stem.split("_")
            rival_name = current_parts[4]
            rival_method = rival_name.upper() if rival_name.lower() != "panoptic" else "Panoptic"
            rows_pair = pd.read_csv(current_file)
            sequence_col_pair = find_column_case_insensitive(rows_pair, "sequence")
            if sequence_col_pair is None:
                raise KeyError(f"Missing sequence column in {current_file}")
            anchor_col = find_column_case_insensitive(rows_pair, f"{anchor_method}-{metric_suffix}")
            rival_col = find_column_case_insensitive(rows_pair, f"{rival_method}-{metric_suffix}")
            if anchor_col is None or rival_col is None:
                available = ", ".join(rows_pair.columns)
                raise KeyError(
                    f"Missing metric columns in {current_file}. "
                    f"Expected {anchor_method}-{metric_suffix} and {rival_method}-{metric_suffix}. "
                    f"Available columns: {available}"
                )

            piece = rows_pair[[sequence_col_pair, anchor_col, rival_col]].copy()
            piece[sequence_col_pair] = piece[sequence_col_pair].astype(str).str.strip()
            piece = piece.rename(
                columns={
                    sequence_col_pair: "sequence",
                    anchor_col: anchor_method,
                    rival_col: rival_method,
                }
            )
            available_methods.add(rival_method)
            if merged is None:
                merged = piece
            else:
                merged = merged.merge(piece[["sequence", rival_method]], on="sequence", how="outer")

        if merged is None:
            continue

        missing_methods = [method for method in required_methods if method not in available_methods]
        if missing_methods:
            raise ValueError(
                f"Dataset {dataset_name} is missing pairwise files for methods: {', '.join(missing_methods)}"
            )

        merged = merged[["sequence"] + args.methods].copy()
        merged["dataset"] = dataset_name
        dataset_frames.append(merged)

    if not dataset_frames:
        raise ValueError(
            "No dataset rows were built from pairwise files. "
            "Check --pairwise-pattern and --datasets."
        )

    combined = pd.concat(dataset_frames, ignore_index=True)
    combined = combined[["dataset", "sequence"] + args.methods].copy()
    combined["sequence"] = combined["sequence"].astype(str).str.strip()
    combined["sequence"] = combined["dataset"].astype(str) + "::" + combined["sequence"]
    return combined.drop(columns=["dataset"]), "pairwise-auto"


def resolve_columns(rows: pd.DataFrame, sequence_column: str, methods: Iterable[str]) -> tuple[str, list[str]]:
    sequence_col = find_column_case_insensitive(rows, sequence_column)
    if sequence_col is None:
        raise KeyError(f"Missing sequence column: {sequence_column}")

    resolved_methods = []
    for method in methods:
        col = find_column_case_insensitive(rows, method)
        if col is None:
            available = ", ".join(rows.columns)
            raise KeyError(f"Missing method column: {method}. Available columns: {available}")
        resolved_methods.append(col)
    return sequence_col, resolved_methods


def load_and_clean_data(args: argparse.Namespace) -> tuple[pd.DataFrame, str]:
    if args.input is not None:
        sep = detect_separator(args.input)
        rows = pd.read_csv(args.input, sep=sep)
        if rows.empty:
            raise ValueError(f"Input file is empty: {args.input}")

        sequence_col, method_cols = resolve_columns(rows, args.sequence_column, args.methods)
        rows = rows[[sequence_col] + method_cols].copy()
        rows = rows.rename(columns={sequence_col: "sequence", **dict(zip(method_cols, args.methods))})
        rows["sequence"] = rows["sequence"].astype(str).str.strip()
    else:
        rows, sep = build_wide_table_from_pairwise(args)

    excluded_exact = {name.strip() for name in args.exclude_sequences}
    if excluded_exact:
        rows = rows[~rows["sequence"].isin(excluded_exact)].copy()

    if args.exclude_substrings:
        mask = pd.Series(False, index=rows.index)
        lowered = rows["sequence"].str.lower()
        for token in args.exclude_substrings:
            mask = mask | lowered.str.contains(token.lower(), regex=False)
        rows = rows[~mask].copy()

    for method in args.methods:
        rows[method] = pd.to_numeric(rows[method], errors="coerce")

    rows = rows.dropna(subset=args.methods).copy()
    rows = rows.reset_index(drop=True)
    if rows.empty:
        raise ValueError("No aligned sequences remain after cleaning and dropping missing values.")

    return rows, sep


def compute_ranks(rows: pd.DataFrame, methods: list[str], lower_is_better: bool) -> pd.DataFrame:
    values = rows[methods].to_numpy(dtype=float)
    rank_input = values if lower_is_better else -values
    ranks = np.apply_along_axis(rankdata, 1, rank_input, method="average")
    rank_columns = [f"{method}_rank" for method in methods]
    rank_rows = rows[["sequence"] + methods].copy()
    rank_rows[rank_columns] = ranks
    return rank_rows


def run_friedman(rank_rows: pd.DataFrame, methods: list[str]) -> tuple[float, float, pd.Series]:
    rank_columns = [f"{method}_rank" for method in methods]
    samples = [rank_rows[column].to_numpy(dtype=float) for column in rank_columns]
    result = friedmanchisquare(*samples)
    average_ranks = rank_rows[rank_columns].mean()
    average_ranks.index = methods
    return float(result.statistic), float(result.pvalue), average_ranks


def nemenyi_p_value(rank_diff: float, k: int, n: int) -> float:
    scale = math.sqrt(k * (k + 1) / (6.0 * n))
    q_stat = rank_diff / scale
    return float(studentized_range.sf(q_stat * math.sqrt(2.0), k, np.inf))


def run_nemenyi(average_ranks: pd.Series, n: int, alpha: float) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    methods = list(average_ranks.index)
    k = len(methods)
    cd = studentized_range.isf(alpha, k, np.inf) * math.sqrt(k * (k + 1) / (6.0 * n)) / math.sqrt(2.0)

    pvals = pd.DataFrame(np.ones((k, k)), index=methods, columns=methods)
    significant = pd.DataFrame(False, index=methods, columns=methods)

    for i, method_i in enumerate(methods):
        for j, method_j in enumerate(methods):
            if i == j:
                continue
            diff = abs(float(average_ranks[method_i]) - float(average_ranks[method_j]))
            p_value = nemenyi_p_value(diff, k, n)
            pvals.loc[method_i, method_j] = p_value
            significant.loc[method_i, method_j] = p_value < alpha

    return pvals, significant, float(cd)


def find_nonsignificant_groups(ordered_methods: list[str], significant: pd.DataFrame) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    total = len(ordered_methods)
    for start in range(total):
        for end in range(start + 1, total):
            methods_in_span = ordered_methods[start : end + 1]
            all_nonsignificant = True
            for i, method_i in enumerate(methods_in_span):
                for method_j in methods_in_span[i + 1 :]:
                    if bool(significant.loc[method_i, method_j]):
                        all_nonsignificant = False
                        break
                if not all_nonsignificant:
                    break
            if all_nonsignificant:
                spans.append((start, end))

    maximal_spans: list[tuple[int, int]] = []
    for span in spans:
        if any(other != span and other[0] <= span[0] and other[1] >= span[1] for other in spans):
            continue
        maximal_spans.append(span)
    return maximal_spans


def draw_cd_diagram(
    average_ranks: pd.Series,
    significant: pd.DataFrame,
    cd: float,
    title: str,
    output_path: Path,
) -> None:
    ordered = average_ranks.sort_values()
    methods = list(ordered.index)
    values = ordered.to_numpy(dtype=float)
    groups = find_nonsignificant_groups(methods, significant)

    k = len(methods)
    fig_width = max(10.0, 2.4 * k)
    fig_height = max(4.8, 2.8 + 0.45 * len(groups) + 0.35 * k)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    x_min = 1.0
    x_max = float(k)
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(-1.5, max(3.5, 1.8 + len(groups)))
    ax.set_yticks([])
    ax.set_xlabel("Average rank (smaller is better)")
    ax.set_title(title)

    baseline_y = 0.0
    ax.hlines(baseline_y, x_min, x_max, color="black", linewidth=1.2)
    for tick in range(1, k + 1):
        ax.vlines(tick, baseline_y - 0.08, baseline_y + 0.08, color="black", linewidth=1.0)
        ax.text(tick, baseline_y - 0.28, str(tick), ha="center", va="top", fontsize=10)

    label_y_start = 0.75
    for idx, (method, value) in enumerate(zip(methods, values)):
        y = label_y_start + idx * 0.45
        ax.vlines(value, baseline_y, y - 0.05, color="tab:blue", linewidth=1.4)
        ax.scatter([value], [baseline_y], color="tab:blue", s=28, zorder=3)
        ax.text(value, y, f"{method} ({value:.3f})", ha="center", va="bottom", fontsize=10)

    cd_y = -0.8
    cd_start = x_min
    cd_end = min(x_max, x_min + cd)
    ax.hlines(cd_y, cd_start, cd_end, color="tab:red", linewidth=2.0)
    ax.vlines([cd_start, cd_end], cd_y - 0.08, cd_y + 0.08, color="tab:red", linewidth=2.0)
    ax.text((cd_start + cd_end) / 2.0, cd_y - 0.18, f"CD = {cd:.3f}", color="tab:red", ha="center", va="top")

    for level, (start, end) in enumerate(groups, start=1):
        y = 1.4 + 0.35 * level
        ax.hlines(
            y,
            values[start],
            values[end],
            color="tab:green",
            linewidth=3.0,
            alpha=0.9,
        )

    for side in ("left", "right", "top", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xticks([])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_summary(
    input_path: Path | None,
    sep: str,
    metric_name: str,
    methods: list[str],
    sequence_count: int,
    friedman_stat: float,
    friedman_p: float,
    average_ranks: pd.Series,
    cd: float,
    alpha: float,
) -> str:
    lines = [
        "Friedman + Nemenyi analysis",
        f"Input source: {input_path if input_path is not None else 'auto-built from pairwise Wilcoxon CSVs'}",
        f"Separator: {repr(sep)}",
        f"Metric: {metric_name}",
        f"Aligned sequences used: {sequence_count}",
        f"Methods: {', '.join(methods)}",
        "",
        f"Friedman chi-square: {friedman_stat:.6f}",
        f"Friedman p-value: {friedman_p:.6g}",
        f"Significant at alpha={alpha:.3f}: {'yes' if friedman_p < alpha else 'no'}",
        "",
        "Average ranks:",
    ]
    for method, value in average_ranks.sort_values().items():
        lines.append(f"  {method}: {value:.6f}")
    lines.extend(
        [
            "",
            f"Nemenyi critical difference (alpha={alpha:.3f}): {cd:.6f}",
            "Pairwise Nemenyi results are saved in CSV files.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    lower_is_better = not args.higher_is_better
    methods = list(args.methods)

    rows, sep = load_and_clean_data(args)
    rank_rows = compute_ranks(rows, methods, lower_is_better=lower_is_better)
    friedman_stat, friedman_p, average_ranks = run_friedman(rank_rows, methods)
    nemenyi_pvals, significance, cd = run_nemenyi(average_ranks, n=len(rank_rows), alpha=args.alpha)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = args.output_dir / "cleaned_input.csv"
    ranks_path = args.output_dir / "sequence_ranks.csv"
    average_ranks_path = args.output_dir / "average_ranks.csv"
    pvals_path = args.output_dir / "nemenyi_pvalues.csv"
    sig_path = args.output_dir / "nemenyi_significant_matrix.csv"
    summary_path = args.output_dir / "summary.txt"
    figure_path = args.output_dir / "critical_difference.png"

    rows.to_csv(cleaned_path, index=False)
    rank_rows.to_csv(ranks_path, index=False)
    (
        average_ranks.sort_values()
        .rename("average_rank")
        .rename_axis("method")
        .reset_index()
        .to_csv(average_ranks_path, index=False, float_format="%.10f")
    )
    nemenyi_pvals.to_csv(pvals_path, float_format="%.10f")
    significance.astype(int).to_csv(sig_path)
    summary_path.write_text(
        format_summary(
            input_path=args.input,
            sep=sep,
            metric_name=args.metric_name,
            methods=methods,
            sequence_count=len(rank_rows),
            friedman_stat=friedman_stat,
            friedman_p=friedman_p,
            average_ranks=average_ranks,
            cd=cd,
            alpha=args.alpha,
        ),
        encoding="utf-8",
    )

    draw_cd_diagram(
        average_ranks=average_ranks,
        significant=significance,
        cd=cd,
        title=f"Critical Difference Diagram ({args.metric_name})",
        output_path=figure_path,
    )

    print(summary_path.read_text(encoding="utf-8"))
    print("Saved files:")
    print(f"  {cleaned_path}")
    print(f"  {ranks_path}")
    print(f"  {average_ranks_path}")
    print(f"  {pvals_path}")
    print(f"  {sig_path}")
    print(f"  {figure_path}")


if __name__ == "__main__":
    main()
