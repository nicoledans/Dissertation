"""Save Grad-CAM examples with audit-only LIDC nodule contours overlaid."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from audit_staged_gradcam_contour_overlap import (
    _gradcam_for_match,
    _load_cache,
    _load_model,
    _match_samples,
)


def _center_image(sample):
    image = np.asarray(sample["image"], dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image[1]
    return np.clip(image, 0.0, 1.0)


def _overlay(image, cam, contour, lung=None):
    base = np.stack([image, image, image], axis=-1)
    heat = plt.get_cmap("magma")(cam)[..., :3]
    out = np.clip(base * 0.55 + heat * 0.45, 0.0, 1.0)
    if lung is not None and np.any(lung):
        # Keep a faint green lung tint so the TS attention region is visible.
        out[lung.astype(bool), 1] = np.maximum(out[lung.astype(bool), 1], 0.45)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="cache/cache_ts_possible_nodule_prior_best.pkl")
    parser.add_argument("--checkpoint", default="results/zhang_ts_progressive_prior_best_aug/best_model.pt")
    parser.add_argument("--out-dir", default="results/zhang_ts_progressive_prior_best_aug")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--cam-target", choices=["correct", "predicted", "malignant"], default="correct")
    parser.add_argument("--include-benign", action="store_true")
    parser.add_argument("--max-matches", type=int, default=12)
    parser.add_argument("--out-name", default="gradcam_examples_with_nodule_contours.png")
    args = parser.parse_args()
    args.malignant_only = not args.include_benign

    samples = _load_cache(args.cache_path)
    matches = _match_samples(samples, args.split, args.malignant_only, args.max_matches)
    if not matches:
        raise RuntimeError("No cache samples could be matched to LIDC contours.")

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, device)

    records = []
    for match in matches:
        cam, probability, prediction = _gradcam_for_match(
            model,
            match,
            device,
            args.cam_target,
        )
        sample = match["cache_sample"]
        contour = np.asarray(match["majority"]).astype(bool)
        union = np.asarray(match["union"]).astype(bool)
        lung = np.asarray(match["lung_mask"]).astype(bool)
        cam_mass = cam / (float(cam.sum()) + 1e-8)
        records.append(
            {
                "image": _center_image(sample),
                "cam": cam,
                "contour": contour,
                "union": union,
                "lung": lung,
                "patient_id": match["patient_id"],
                "label": int(match["label"]),
                "probability": probability,
                "prediction": prediction,
                "contour_mass": float((cam_mass * contour).sum() * 100.0),
                "lung_mass": float((cam_mass * lung).sum() * 100.0),
            }
        )
    model.remove_hooks()

    rows = len(records)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 2.7))
    if rows == 1:
        axes = axes[None, :]
    for row_idx, record in enumerate(records):
        image = record["image"]
        panels = [
            ("CT + nodule contour", image, "gray"),
            ("Grad-CAM", record["cam"], "magma"),
            ("Overlay + contour", _overlay(image, record["cam"], record["contour"]), None),
            ("Overlay + TS + contour", _overlay(image, record["cam"], record["contour"], record["lung"]), None),
        ]
        for col_idx, (title, panel, cmap) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel, cmap=cmap, vmin=0.0 if cmap else None, vmax=1.0 if cmap else None)
            if col_idx in (0, 2, 3):
                ax.contour(record["union"].astype(float), levels=[0.5], colors=["deepskyblue"], linewidths=0.8)
                ax.contour(record["contour"].astype(float), levels=[0.5], colors=["yellow"], linewidths=1.1)
            if col_idx == 3:
                ax.contour(record["lung"].astype(float), levels=[0.5], colors=["lime"], linewidths=0.7)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
        axes[row_idx, 0].text(
            0.0,
            -0.08,
            (
                f"{record['patient_id']} label={record['label']} "
                f"p={record['probability']:.2f} pred={record['prediction']} "
                f"CAM contour={record['contour_mass']:.2f}% lung={record['lung_mass']:.1f}%"
            ),
            transform=axes[row_idx, 0].transAxes,
            fontsize=7,
            va="top",
        )

    fig.suptitle(
        "Grad-CAM with audit-only LIDC nodule contours. Yellow=majority contour, blue=any-reader union, green=TS lung.",
        fontsize=11,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    out_path = os.path.join(args.out_dir, args.out_name)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
