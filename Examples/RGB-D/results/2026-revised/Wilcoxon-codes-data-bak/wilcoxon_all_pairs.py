from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon
from scipy.stats import rankdata


METRICS = ["RMSE", "MEAN", "STD"]


def detect_separator(file_path):
    for sep in (",", ";", "\t"):
        try:
            rows = pd.read_csv(file_path, sep=sep)
        except Exception:
            continue
        if "sequence" in rows.columns and len(rows.columns) >= 3:
            return sep
    raise ValueError(f"Could not detect a valid separator for {file_path}")


def parse_file_metadata(file_path):
    parts = file_path.stem.split("_")
    if len(parts) >= 5 and parts[0].lower() == "wilcoxon":
        dataset = parts[1]
        algo_a = parts[2].upper()
        if parts[3].lower() == "vs":
            algo_b = parts[4]
        else:
            algo_b = "unknown"
    else:
        dataset = "unknown"
        algo_a = "ROVK"
        algo_b = "unknown"
    return dataset, algo_a, algo_b


def load_rows(file_path):
    sep = detect_separator(file_path)
    rows = pd.read_csv(file_path, sep=sep)
    if rows.empty:
        raise ValueError(f"Input file is empty: {file_path}")
    rows["sequence"] = rows["sequence"].astype(str).str.strip()
    return rows, sep


def find_column_case_insensitive(rows, expected_name):
    expected_lower = expected_name.lower()
    for column in rows.columns:
        if column.lower() == expected_lower:
            return column
    return None


def find_metric_columns(rows, algo_a, algo_b, metric):
    algo_a_expected = f"{algo_a}-{metric}"
    algo_b_expected = f"{algo_b}-{metric}"
    algo_a_col = find_column_case_insensitive(rows, algo_a_expected)
    algo_b_col = find_column_case_insensitive(rows, algo_b_expected)
    if algo_a_col and algo_b_col:
        return algo_a_col, algo_b_col

    available = ", ".join(rows.columns)
    raise KeyError(
        f"Missing columns for metric {metric}. Expected {algo_a_expected} and {algo_b_expected}. "
        f"Available columns: {available}"
    )


def wilcoxon_two_sided(x_values, y_values):
    diffs = x_values - y_values
    if (diffs == 0).all():
        raise ValueError("All paired differences are zero.")
    result = wilcoxon(
        x_values,
        y_values,
        zero_method="wilcox",
        alternative="two-sided",
    )
    return result.statistic, result.pvalue


def rank_biserial_correlation(x_values, y_values):
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

    return (negative_rank_sum - positive_rank_sum) / total_rank_sum


def analyse_metric(rows, algo_a, algo_b, metric):
    algo_a_col, algo_b_col = find_metric_columns(rows, algo_a, algo_b, metric)
    metric_rows = rows[["sequence", algo_a_col, algo_b_col]].copy()
    metric_rows[algo_a_col] = pd.to_numeric(metric_rows[algo_a_col], errors="coerce")
    metric_rows[algo_b_col] = pd.to_numeric(metric_rows[algo_b_col], errors="coerce")

    valid_mask = metric_rows[algo_a_col].notna() & metric_rows[algo_b_col].notna()
    valid_rows = metric_rows[valid_mask].copy()
    skipped_rows = metric_rows[~valid_mask].copy()

    print("=" * 60)
    print(f"Metric: {metric}")
    print("-" * 60)
    print(f"Total sequences:   {len(metric_rows)}")
    print(f"Valid pairs used:  {len(valid_rows)}")
    print(f"Skipped sequences: {len(skipped_rows)}")

    if valid_rows.empty:
        print("No valid paired data for this metric.")
        print()
        return skipped_rows["sequence"].tolist()

    algo_a_values = valid_rows[algo_a_col]
    algo_b_values = valid_rows[algo_b_col]
    diff = algo_a_values - algo_b_values
    win_algo_a = int((diff < 0).sum())
    win_algo_b = int((diff > 0).sum())
    tie = int((diff == 0).sum())

    print(f"{algo_a} mean: {algo_a_values.mean():.6f}")
    print(f"{algo_b} mean: {algo_b_values.mean():.6f}")
    print(f"{algo_a} wins: {win_algo_a}")
    print(f"{algo_b} wins: {win_algo_b}")
    print(f"Ties:        {tie}")

    try:
        stat, p = wilcoxon_two_sided(algo_a_values, algo_b_values)
        effect_size = rank_biserial_correlation(algo_a_values, algo_b_values)
        print(f"Wilcoxon statistic: {stat:.6f}")
        print(f"p-value ({algo_a} != {algo_b}): {p:.6g}")
        print(f"Rank-biserial correlation: {effect_size:.6f}")
    except ValueError as exc:
        print("Wilcoxon failed:", exc)

    print(f"\nPer-sequence differences ({algo_a} - {algo_b}):")
    print(f"{'sequence':<40} {algo_a_col:>16} {algo_b_col:>16} {'diff':>16}")
    for sequence, algo_a_value, algo_b_value, diff_value in zip(
        valid_rows["sequence"], algo_a_values, algo_b_values, diff
    ):
        print(
            f"{sequence:<40} {algo_a_value:>16.6f} {algo_b_value:>16.6f} {diff_value:>16.6f}"
        )
    print()

    return skipped_rows["sequence"].tolist()


def analyse_file(file_path):
    dataset, algo_a, algo_b = parse_file_metadata(file_path)
    rows, sep = load_rows(file_path)

    print("#" * 60)
    print(f"File: {file_path.name}")
    print(f"Dataset: {dataset}")
    print(f"Comparison: {algo_a} vs {algo_b}")
    print(f"Separator: {repr(sep)}")
    print(f"Loaded sequences: {len(rows)}")
    print()

    skipped_by_metric = {}
    for metric in METRICS:
        skipped_by_metric[metric] = analyse_metric(rows, algo_a, algo_b, metric)

    print("=" * 60)
    print("Skipped sequences reminder")
    print("-" * 60)
    for metric in METRICS:
        skipped_sequences = skipped_by_metric[metric]
        if skipped_sequences:
            print(f"{metric}: {', '.join(skipped_sequences)}")
        else:
            print(f"{metric}: None")
    print()


def main():
    csv_files = sorted(Path(".").glob("wilcoxon_*.csv"))
    if not csv_files:
        raise FileNotFoundError("No wilcoxon_*.csv files found in the current directory.")

    for file_path in csv_files:
        analyse_file(file_path)


if __name__ == "__main__":
    main()
