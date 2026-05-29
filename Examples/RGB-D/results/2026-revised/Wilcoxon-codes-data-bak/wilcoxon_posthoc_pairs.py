from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import rankdata, wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a wide sequence x method table from wilcoxon_*.csv pairwise files "
            "and run pairwise Wilcoxon signed-rank tests for selected methods."
        )
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        required=True,
        help="Methods to compare pairwise, for example: SG Panoptic NGD",
    )
    parser.add_argument(
        "--anchor-method",
        default="ROVK",
        help="Anchor method used in wilcoxon_<dataset>_<anchor>_vs_<method>.csv files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=[],
        help="Datasets to include, for example: TUM bonn",
    )
    parser.add_argument(
        "--metric-column",
        default="RMSE",
        help="Metric suffix in pairwise CSV columns, for example RMSE, MEAN, STD.",
    )
    parser.add_argument(
        "--pairwise-pattern",
        default="wilcoxon_*.csv",
        help="Glob pattern for pairwise CSV files. Default: wilcoxon_*.csv",
    )
    parser.add_argument(
        "--exclude-sequences",
        nargs="*",
        default=[],
        help="Exact sequence names to exclude after trimming whitespace.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save pairwise Wilcoxon outputs.",
    )
    return parser.parse_args()


def find_column_case_insensitive(rows: pd.DataFrame, expected_name: str) -> str | None:
    expected_lower = expected_name.lower()
    for column in rows.columns:
        if column.lower() == expected_lower:
            return column
    return None


def build_wide_table(
    methods: list[str],
    anchor_method: str,
    datasets: list[str],
    metric_suffix: str,
    pairwise_pattern: str,
) -> pd.DataFrame:
    pairwise_files = sorted(Path(".").glob(pairwise_pattern))
    if not pairwise_files:
        raise FileNotFoundError(f"No pairwise files matched: {pairwise_pattern}")

    datasets_filter = {item.lower() for item in datasets} if datasets else set()
    required_methods = list(methods)

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
            if rival_method not in required_methods:
                continue

            rows_pair = pd.read_csv(current_file)
            sequence_col = find_column_case_insensitive(rows_pair, "sequence")
            if sequence_col is None:
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

            piece = rows_pair[[sequence_col, rival_col]].copy()
            piece[sequence_col] = piece[sequence_col].astype(str).str.strip()
            piece = piece.rename(columns={sequence_col: "sequence", rival_col: rival_method})
            available_methods.add(rival_method)
            if merged is None:
                merged = piece
            else:
                merged = merged.merge(piece, on="sequence", how="outer")

        missing_methods = [method for method in required_methods if method not in available_methods]
        if missing_methods:
            raise ValueError(
                f"Dataset {dataset_name} is missing pairwise files for methods: {', '.join(missing_methods)}"
            )

        merged["dataset"] = dataset_name
        dataset_frames.append(merged[["dataset", "sequence"] + required_methods].copy())

    if not dataset_frames:
        raise ValueError("No dataset rows were built from pairwise files.")

    combined = pd.concat(dataset_frames, ignore_index=True)
    combined["sequence"] = combined["sequence"].astype(str).str.strip()
    combined["sequence"] = combined["dataset"].astype(str) + "::" + combined["sequence"]
    return combined[["sequence"] + required_methods].copy()


def rank_biserial_correlation(x_values: pd.Series, y_values: pd.Series) -> float:
    diffs = x_values - y_values
    non_zero_diffs = diffs[diffs != 0]
    if non_zero_diffs.empty:
        raise ValueError("All paired differences are zero.")

    abs_ranks = rankdata(non_zero_diffs.abs(), method="average")
    positive_rank_sum = abs_ranks[non_zero_diffs > 0].sum()
    negative_rank_sum = abs_ranks[non_zero_diffs < 0].sum()
    total_rank_sum = positive_rank_sum + negative_rank_sum
    if total_rank_sum == 0:
        raise ValueError("Rank sum is zero; effect size is undefined.")
    return float((negative_rank_sum - positive_rank_sum) / total_rank_sum)


def run_pairwise_tests(rows: pd.DataFrame, methods: list[str]) -> tuple[pd.DataFrame, list[str]]:
    summary_rows = []
    detail_lines = []
    for method_a, method_b in combinations(methods, 2):
        pair_rows = rows[["sequence", method_a, method_b]].dropna().copy()
        pair_rows[method_a] = pd.to_numeric(pair_rows[method_a], errors="coerce")
        pair_rows[method_b] = pd.to_numeric(pair_rows[method_b], errors="coerce")
        pair_rows = pair_rows.dropna().copy()

        x_values = pair_rows[method_a]
        y_values = pair_rows[method_b]
        diffs = x_values - y_values
        wins_a = int((diffs < 0).sum())
        wins_b = int((diffs > 0).sum())
        ties = int((diffs == 0).sum())

        if (diffs == 0).all():
            stat = 0.0
            p_value = 1.0
            effect = 0.0
        else:
            result = wilcoxon(
                x_values,
                y_values,
                zero_method="wilcox",
                alternative="two-sided",
            )
            stat = float(result.statistic)
            p_value = float(result.pvalue)
            effect = rank_biserial_correlation(x_values, y_values)

        summary_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "n_sequences": len(pair_rows),
                "mean_a": float(x_values.mean()),
                "mean_b": float(y_values.mean()),
                "wins_a": wins_a,
                "wins_b": wins_b,
                "ties": ties,
                "wilcoxon_statistic": stat,
                "p_value": p_value,
                "significant_at_0_05": int(p_value < 0.05),
                "rank_biserial_correlation": effect,
            }
        )

        detail_lines.append("=" * 60)
        detail_lines.append(f"{method_a} vs {method_b}")
        detail_lines.append("-" * 60)
        detail_lines.append(f"Aligned sequences: {len(pair_rows)}")
        detail_lines.append(f"{method_a} mean: {x_values.mean():.6f}")
        detail_lines.append(f"{method_b} mean: {y_values.mean():.6f}")
        detail_lines.append(f"{method_a} wins: {wins_a}")
        detail_lines.append(f"{method_b} wins: {wins_b}")
        detail_lines.append(f"Ties: {ties}")
        detail_lines.append(f"Wilcoxon statistic: {stat:.6f}")
        detail_lines.append(f"p-value: {p_value:.10g}")
        detail_lines.append(f"Rank-biserial correlation: {effect:.6f}")
        detail_lines.append("")
        detail_lines.append(
            f"{'sequence':<52} {method_a:>12} {method_b:>12} {'diff':>12}"
        )
        for sequence, value_a, value_b, diff in zip(
            pair_rows["sequence"], x_values, y_values, diffs
        ):
            detail_lines.append(
                f"{sequence:<52} {value_a:>12.6f} {value_b:>12.6f} {diff:>12.6f}"
            )
        detail_lines.append("")

    summary = pd.DataFrame(summary_rows)
    return summary, detail_lines


def main() -> None:
    args = parse_args()
    rows = build_wide_table(
        methods=list(args.methods),
        anchor_method=args.anchor_method,
        datasets=args.datasets,
        metric_suffix=args.metric_column.upper(),
        pairwise_pattern=args.pairwise_pattern,
    )

    excluded_exact = {name.strip() for name in args.exclude_sequences}
    if excluded_exact:
        rows = rows[~rows["sequence"].isin(excluded_exact)].copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = args.output_dir / "aligned_input.csv"
    summary_path = args.output_dir / "wilcoxon_pairwise_summary.csv"
    details_path = args.output_dir / "wilcoxon_pairwise_details.txt"

    rows.to_csv(cleaned_path, index=False)
    summary, details = run_pairwise_tests(rows, list(args.methods))
    summary.to_csv(summary_path, index=False, float_format="%.10f")
    details_path.write_text("\n".join(details) + "\n", encoding="utf-8")

    print(summary.to_string(index=False))
    print()
    print(f"Saved: {cleaned_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {details_path}")


if __name__ == "__main__":
    main()
