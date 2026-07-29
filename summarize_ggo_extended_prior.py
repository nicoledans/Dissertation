"""Summarize GGO-extended possible-nodule prior by HU-derived nodule type."""

import argparse
import csv
import os
from collections import defaultdict

import numpy as np


def _read_csv(path):
    with open(path, newline="") as file:
        return list(csv.DictReader(file))


def _as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes"}


def _type_from_mean_hu(mean_hu):
    value = float(mean_hu)
    if value < -500.0:
        return "ground_glass"
    if value < -300.0:
        return "intermediate"
    return "solid"


def _summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped["overall"].append(row)
        grouped[row["nodule_type"]].append(row)

    output = {}
    for group, group_rows in grouped.items():
        hits = [_as_bool(row["hit"]) for row in group_rows]
        coverages = [float(row["coverage"]) for row in group_rows]
        areas = [float(row["candidate_area_pct"]) for row in group_rows]
        full = sum(coverage >= 0.999999 for coverage in coverages)
        output[group] = {
            "n": len(group_rows),
            "hit_pct": sum(hits) / max(len(hits), 1) * 100.0,
            "full_pct": full / max(len(group_rows), 1) * 100.0,
            "coverage_pct": float(np.mean(coverages)) * 100.0 if coverages else float("nan"),
            "median_coverage_pct": float(np.median(coverages)) * 100.0 if coverages else float("nan"),
            "area_pct": float(np.mean(areas)) if areas else float("nan"),
        }
    return output


def _format_table(summary):
    lines = [
        "Group | N | Hit % | 100% covered % | Mean coverage % | Median coverage % | Mean area %",
        "--- | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    for group in ["overall", "solid", "intermediate", "ground_glass"]:
        stats = summary.get(group, {"n": 0})
        lines.append(
            f"{group} | {stats.get('n', 0)} | "
            f"{stats.get('hit_pct', float('nan')):.2f} | "
            f"{stats.get('full_pct', float('nan')):.2f} | "
            f"{stats.get('coverage_pct', float('nan')):.2f} | "
            f"{stats.get('median_coverage_pct', float('nan')):.2f} | "
            f"{stats.get('area_pct', float('nan')):.2f}"
        )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument(
        "--phenotype-csv",
        default=r"results\csv findings\nodule_case_summary.csv",
    )
    parser.add_argument(
        "--original-eval-csv",
        default=r"results\possible_nodule_mask_eval_best_dil1\possible_nodule_mask_eval.csv",
    )
    parser.add_argument("--out-dir", default=r"results\prior_ggo_extended")
    args = parser.parse_args()

    eval_rows = _read_csv(args.eval_csv)
    phenotype_rows = {row["cache_index"]: row for row in _read_csv(args.phenotype_csv)}
    original_rows = _read_csv(args.original_eval_csv) if os.path.exists(args.original_eval_csv) else []
    original_rows_by_index = {row["cache_index"]: row for row in original_rows}

    joined = []
    for row in eval_rows:
        phenotype = phenotype_rows.get(row["cache_index"])
        if phenotype is None:
            continue
        merged = dict(row)
        merged["mean_hu"] = phenotype["mean_hu"]
        merged["hu_pattern"] = phenotype.get("hu_pattern", "")
        merged["nodule_type"] = _type_from_mean_hu(phenotype["mean_hu"])
        joined.append(merged)

    original_joined = []
    for index, original in original_rows_by_index.items():
        phenotype = phenotype_rows.get(index)
        if phenotype is None:
            continue
        merged = dict(original)
        merged["mean_hu"] = phenotype["mean_hu"]
        merged["nodule_type"] = _type_from_mean_hu(phenotype["mean_hu"])
        original_joined.append(merged)

    summary = _summarize(joined)
    original_summary = _summarize(original_joined) if original_joined else {}

    os.makedirs(args.out_dir, exist_ok=True)
    joined_path = os.path.join(args.out_dir, "ggo_extended_prior_eval_joined.csv")
    with open(joined_path, "w", newline="") as file:
        fieldnames = list(joined[0].keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(joined)

    lines = [
        "=== GGO-EXTENDED POSSIBLE-NODULE PRIOR ===",
        f"New eval CSV: {args.eval_csv}",
        f"Original eval CSV: {args.original_eval_csv}",
        "Type split: solid mean_hu >= -300; intermediate -500 <= mean_hu < -300; ground_glass mean_hu < -500.",
        "",
        "## New GGO-Extended Prior",
        *_format_table(summary),
        "",
        "## Original Best Prior, Same Type Split",
        *_format_table(original_summary),
        "",
        f"Original known GGO hit reference from phenotype report: 77.09%",
        f"New GGO hit: {summary.get('ground_glass', {}).get('hit_pct', float('nan')):.2f}%",
        f"Target GGO hit > 85%: {'YES' if summary.get('ground_glass', {}).get('hit_pct', 0.0) > 85.0 else 'NO'}",
        "",
        f"Joined CSV: {joined_path}",
    ]
    summary_path = os.path.join(args.out_dir, "ggo_extended_prior_summary.md")
    with open(summary_path, "w") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
