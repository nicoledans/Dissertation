"""Preview Grad-CAM, possible-nodule prior, TS mask, and LIDC contour together.

LIDC contours are used only for audit/visualisation. They are not used for
training, model selection, or prior generation.
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

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


def _normalise_map(values):
    values = np.asarray(values, dtype=np.float32)
    values = values - float(values.min())
    peak = float(values.max())
    if peak > 0:
        values = values / (peak + 1e-8)
    return np.clip(values, 0.0, 1.0)


def _resize_like(values, shape, binary=False):
    values = np.asarray(values, dtype=np.float32)
    if tuple(values.shape) == tuple(shape):
        return values.astype(bool) if binary else values
    tensor = torch.from_numpy(values).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(
        tensor,
        size=shape,
        mode="nearest" if binary else "bilinear",
        align_corners=False if not binary else None,
    ).squeeze().numpy()
    return resized > 0.5 if binary else resized


def _overlay_heat(image, heat, alpha=0.45, cmap_name="magma"):
    base = np.stack([image, image, image], axis=-1)
    heat_rgb = plt.get_cmap(cmap_name)(_normalise_map(heat))[..., :3]
    return np.clip(base * (1.0 - alpha) + heat_rgb * alpha, 0.0, 1.0)


def _draw_common_contours(ax, contour, union=None, lung=None, prior_binary=None):
    if lung is not None and np.any(lung):
        ax.contour(lung.astype(float), levels=[0.5], colors=["lime"], linewidths=0.7)
    if prior_binary is not None and np.any(prior_binary):
        ax.contour(prior_binary.astype(float), levels=[0.5], colors=["cyan"], linewidths=0.8)
    if union is not None and np.any(union):
        ax.contour(union.astype(float), levels=[0.5], colors=["deepskyblue"], linewidths=0.8)
    if np.any(contour):
        ax.contour(contour.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="cache/cache_ts_possible_nodule_prior_best.pkl")
    parser.add_argument("--checkpoint", default="results/zhang_ts_aug/selected_model.pt")
    parser.add_argument("--out-dir", default="results/gradcam_prior_mask_nodule_examples")
    parser.add_argument("--out-name", default="zhang_ts_gradcam_prior_mask_nodule_examples.png")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--cam-target", choices=["correct", "predicted", "malignant"], default="correct")
    parser.add_argument("--include-benign", action="store_true")
    parser.add_argument("--max-matches", type=int, default=8)
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
        image = _center_image(sample)
        image_shape = image.shape
        contour = _resize_like(match["majority"], image_shape, binary=True)
        union = _resize_like(match["union"], image_shape, binary=True)
        lung = _resize_like(sample.get("ts_mask", match["lung_mask"]), image_shape, binary=True)
        prior = _resize_like(sample["possible_nodule_prior"], image_shape, binary=False)
        prior_binary = _resize_like(sample["possible_nodule_mask"], image_shape, binary=True)
        cam_mass = cam / (float(cam.sum()) + 1e-8)
        records.append(
            {
                "image": image,
                "cam": cam,
                "prior": prior,
                "prior_binary": prior_binary,
                "lung": lung,
                "contour": contour,
                "union": union,
                "patient_id": match["patient_id"],
                "label": int(match["label"]),
                "probability": float(probability),
                "prediction": int(prediction),
                "cam_nodule_pct": float((cam_mass * contour).sum() * 100.0),
                "cam_lung_pct": float((cam_mass * lung).sum() * 100.0),
                "prior_contour_coverage_pct": (
                    float((prior_binary & contour).sum() / contour.sum() * 100.0)
                    if np.any(contour)
                    else 0.0
                ),
            }
        )
    model.remove_hooks()

    cols = 5
    rows = len(records)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.25))
    if rows == 1:
        axes = axes[None, :]

    for row_idx, record in enumerate(records):
        image = record["image"]
        panels = [
            ("CT + nodule", image, "gray", {}),
            ("TS lung mask", image, "gray", {"lung": record["lung"]}),
            ("Possible prior", record["prior"], "magma", {"prior_binary": record["prior_binary"]}),
            ("Grad-CAM", record["cam"], "jet", {}),
            ("All overlay", _overlay_heat(image, record["cam"], alpha=0.40, cmap_name="jet"), None, {
                "lung": record["lung"],
                "prior_binary": record["prior_binary"],
            }),
        ]
        for col_idx, (title, panel, cmap, extras) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(panel, cmap=cmap, vmin=0.0 if cmap else None, vmax=1.0 if cmap else None)
            _draw_common_contours(
                ax,
                record["contour"],
                union=record["union"] if col_idx in (0, 4) else None,
                lung=extras.get("lung"),
                prior_binary=extras.get("prior_binary"),
            )
            ax.axis("off")
            ax.set_xlim(-0.5, image.shape[1] - 0.5)
            ax.set_ylim(image.shape[0] - 0.5, -0.5)
            if row_idx == 0:
                ax.set_title(title, fontsize=9)
        axes[row_idx, 0].text(
            0.02,
            0.98,
            (
                f"{record['patient_id']}\n"
                f"y={record['label']} p={record['probability']:.2f} pred={record['prediction']}\n"
                f"CAM nodule {record['cam_nodule_pct']:.2f}%\n"
                f"prior covers {record['prior_contour_coverage_pct']:.1f}%"
            ),
            transform=axes[row_idx, 0].transAxes,
            fontsize=6.5,
            va="top",
            ha="left",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
        )

    fig.suptitle(
        (
            "Zhang TS Grad-CAM, possible-nodule prior, TS lung mask, and audit-only LIDC contour\n"
            "Yellow=majority nodule contour, blue=any-reader union, lime=TS lung, cyan=binary prior"
        ),
        fontsize=11,
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.94, wspace=0.03, hspace=0.18)
    out_path = os.path.join(args.out_dir, args.out_name)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
