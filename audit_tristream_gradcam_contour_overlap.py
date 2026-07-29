"""Audit TriStream Grad-CAM overlap with actual LIDC nodule contours.

Contours are audit-only. They are not used for training, model selection, or
candidate/prior generation. Grad-CAM is computed from stream 1, the raw CT
backbone, so all TriStream ablations are compared on the same visual stream.
"""

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F

from audit_staged_gradcam_contour_overlap import (
    _load_cache,
    _match_samples,
    _mean,
    _row_metrics,
    _write_summary,
)
from config import IMG_SIZE
from train_tristream import TriStreamDataset, TriStreamNet


def _gradcam_for_match(model, match, device, ablation, candidate_dilation, target):
    sample = match["cache_sample"]
    dataset = TriStreamDataset(
        [sample],
        augment=False,
        ablation=ablation,
        candidate_dilation=candidate_dilation,
    )
    x1, x2, x3, _lung, label = dataset[0]
    x1 = x1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)
    x3 = x3.unsqueeze(0).to(device)
    label = label.view(1).to(device)

    model.zero_grad(set_to_none=True)
    logits = model(x1, x2, x3)
    probability = torch.sigmoid(logits)[0].item()
    prediction = int(probability >= 0.5)

    if target == "correct":
        score = logits * (label * 2.0 - 1.0)
    elif target == "malignant":
        score = logits
    else:
        signs = torch.where(logits.detach() >= 0, 1.0, -1.0)
        score = logits * signs
    score.sum().backward()

    cam = model.stream1.gradcam()
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu().numpy()
    return cam, probability, prediction


def _load_model(checkpoint, device):
    model = TriStreamNet().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-path",
        default="cache/cache_ts_possible_nodule_prior_v4_original_ts_restricted.pkl",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ablation", choices=["full", "nocandidate", "nomask"], default="full")
    parser.add_argument("--candidate-dilation", type=int, default=1)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--cam-target", choices=["correct", "predicted", "malignant"], default="correct")
    parser.add_argument("--include-benign", action="store_true")
    parser.add_argument("--max-matches", type=int, default=0)
    args = parser.parse_args()
    args.malignant_only = not args.include_benign

    thresholds = [0.0, 0.001, 0.01, 0.05, 0.10, 0.20]
    samples = _load_cache(args.cache_path)
    matches = _match_samples(samples, args.split, args.malignant_only, args.max_matches)
    if not matches:
        raise RuntimeError("No cache samples could be matched to LIDC contours.")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Matched samples: {len(matches)}")
    print(f"Device: {device}")
    print(f"TriStream ablation: {args.ablation}; Grad-CAM stream: raw CT stream 1")
    model = _load_model(args.checkpoint, device)

    rows = []
    for index, match in enumerate(matches, start=1):
        if index % 20 == 0 or index == len(matches):
            print(f"Auditing {index}/{len(matches)}")
        cam, probability, prediction = _gradcam_for_match(
            model,
            match,
            device,
            args.ablation,
            args.candidate_dilation,
            args.cam_target,
        )
        rows.append(_row_metrics(match, cam, probability, prediction, thresholds))
    model.remove_hooks()

    csv_path = os.path.join(args.out_dir, "tristream_gradcam_contour_overlap.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_path = os.path.join(args.out_dir, "tristream_gradcam_contour_overlap_summary.txt")
    _write_summary(rows, thresholds, summary_path, args)
    with open(summary_path, "a") as file:
        file.write("\nTriStream-specific notes:\n")
        file.write(f"Ablation: {args.ablation}\n")
        file.write(f"Candidate dilation: {args.candidate_dilation}\n")
        file.write("Grad-CAM stream: stream 1 raw CT backbone\n")
        file.write(f"Mean CAM mass inside majority contour: {_mean(rows, 'cam_mass_inside_majority_contour_pct'):.2f}%\n")
    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
