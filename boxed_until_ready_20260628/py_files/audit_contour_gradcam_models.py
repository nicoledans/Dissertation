"""Compare multiple checkpoints by Grad-CAM mass inside LIDC nodule contours.

Radiologist contours are used only for post-hoc audit metrics. They are not
used for training or model selection.
"""

import argparse
import csv
import os

import numpy as np
import torch

from audit_soft_blob_candidates import (
    _gradcam_metrics,
    _load_cache,
    _load_model,
    _map_metrics,
    _match_samples,
)


def _model_specs(values):
    specs = []
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                "Model specs must use name=checkpoint_path format."
            )
        name, path = value.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name:
            raise argparse.ArgumentTypeError("Model spec name cannot be empty.")
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(f"Checkpoint not found: {path}")
        specs.append((name, path))
    return specs


def _safe_mean(values):
    values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


def _safe_median(values):
    values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.median(values)) if values else float("nan")


def _prediction_summary(rows):
    correct = [int(row["prediction"]) == int(row["label"]) for row in rows]
    return sum(correct), len(correct)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="cache/cache_hu_soft_blobs.pkl")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--mask-source", choices=["auto", "hu", "ts"], default="auto")
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--top-percentile", type=float, default=90.0)
    parser.add_argument("--center-radius-px", type=float, default=4.0)
    parser.add_argument("--out-dir", default="results/contour_gradcam_model_audit")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Model checkpoint as name=path. Repeat for multiple models.",
    )
    args = parser.parse_args()

    model_specs = _model_specs(args.model)
    samples = _load_cache(args.cache_path)
    matches = _match_samples(samples, args.split, args.max_matches, args.mask_source)
    if not matches:
        raise RuntimeError("No soft-blob cache samples could be matched to LIDC contours.")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Matched samples: {len(matches)}")
    print(f"Device: {device}")

    base_rows = [
        _map_metrics(match, args.top_percentile, args.center_radius_px)
        for match in matches
    ]

    summary_rows = []
    detail_rows = []
    for model_name, checkpoint in model_specs:
        print(f"\nAuditing model: {model_name}")
        model = _load_model(checkpoint, device)
        model_rows = []
        for index, match in enumerate(matches, start=1):
            if index % 25 == 0 or index == len(matches):
                print(f"  {index}/{len(matches)}")
            metrics = _gradcam_metrics(model, match, device)
            row = dict(base_rows[index - 1])
            row["model"] = model_name
            row.update(metrics)
            detail_rows.append(row)
            model_rows.append(row)
        model.remove_hooks()

        correct, total = _prediction_summary(model_rows)
        summary_rows.append(
            {
                "model": model_name,
                "matched_samples": len(model_rows),
                "accuracy_on_matched": correct / total if total else float("nan"),
                "mean_cam_inside_true_contour_pct": _safe_mean(
                    row["cam_mass_inside_true_contour_pct"] for row in model_rows
                ),
                "median_cam_inside_true_contour_pct": _safe_median(
                    row["cam_mass_inside_true_contour_pct"] for row in model_rows
                ),
                "mean_cam_inside_lung_pct": _safe_mean(
                    row["cam_mass_inside_lung_pct"] for row in model_rows
                ),
                "mean_cam_soft_blob_overlap": _safe_mean(
                    row["cam_mass_inside_candidate_weighted"] for row in model_rows
                ),
                "mean_cam_inside_high_blob_pct": _safe_mean(
                    row["cam_mass_inside_high_candidate_pct"] for row in model_rows
                ),
                "peak_proxy_samples_gt_1pct_contour": sum(
                    row["cam_mass_inside_true_contour_pct"] >= 1.0
                    for row in model_rows
                ),
                "peak_proxy_samples_gt_5pct_contour": sum(
                    row["cam_mass_inside_true_contour_pct"] >= 5.0
                    for row in model_rows
                ),
            }
        )

    detail_path = os.path.join(args.out_dir, "contour_gradcam_model_audit.csv")
    with open(detail_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail_rows[0].keys()))
        writer.writeheader()
        writer.writerows(detail_rows)

    summary_path = os.path.join(args.out_dir, "contour_gradcam_model_audit_summary.csv")
    with open(summary_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    txt_path = os.path.join(args.out_dir, "contour_gradcam_model_audit_summary.txt")
    lines = [
        "=== GRAD-CAM VS RADIOLOGIST NODULE CONTOUR AUDIT ===",
        "Contours are audit-only and were not used for training.",
        f"Cache: {args.cache_path}",
        f"Split: {args.split}",
        f"Matched samples: {len(matches)}",
        "",
        (
            "model | acc_matched | mean contour CAM | median contour CAM | "
            "mean lung CAM | mean blob overlap | >=1% contour | >=5% contour"
        ),
    ]
    for row in summary_rows:
        lines.append(
            f"{row['model']} | {row['accuracy_on_matched']:.3f} | "
            f"{row['mean_cam_inside_true_contour_pct']:.2f}% | "
            f"{row['median_cam_inside_true_contour_pct']:.2f}% | "
            f"{row['mean_cam_inside_lung_pct']:.2f}% | "
            f"{row['mean_cam_soft_blob_overlap']:.4f} | "
            f"{row['peak_proxy_samples_gt_1pct_contour']}/{row['matched_samples']} | "
            f"{row['peak_proxy_samples_gt_5pct_contour']}/{row['matched_samples']}"
        )
    with open(txt_path, "w") as file:
        file.write("\n".join(lines) + "\n")

    print("\n" + "\n".join(lines))
    print(f"\nSaved detail CSV: {detail_path}")
    print(f"Saved summary CSV: {summary_path}")
    print(f"Saved summary TXT: {txt_path}")


if __name__ == "__main__":
    main()
