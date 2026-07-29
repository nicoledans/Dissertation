"""Make a LUNGx hits/misses visual grid for attention-transfer ablation."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import IMG_SIZE, SEED
from evaluate_attention_transfer_ablation import _gradcam, _guided_image, _load_model
from evaluate_lungx import _load_hu_slice, _load_lungx_rows, _prepare_image


def _read_predictions(path):
    with open(path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {(row["scan_id"], int(row["nodule_number"])): row for row in rows}


def _record_lookup(xlsx_path):
    records = _load_lungx_rows(xlsx_path)
    return {(row["scan_id"], int(row["nodule_number"])): row for row in records}


def _centre_224(record, raw_slice):
    scale_x = IMG_SIZE / float(raw_slice.shape[1])
    scale_y = IMG_SIZE / float(raw_slice.shape[0])
    cx = (float(record["center_x_1based"]) - 1.0) * scale_x
    cy = (float(record["center_y_1based"]) - 1.0) * scale_y
    return cy, cx


def _overlay(image, guide, alpha=0.45):
    image = np.asarray(image, dtype=np.float32)
    guide = np.asarray(guide, dtype=np.float32)
    cmap = plt.get_cmap("jet")(guide)[..., :3]
    base = np.stack([image, image, image], axis=-1)
    return np.clip(base * (1.0 - alpha) + cmap * alpha, 0.0, 1.0)


def _circle(ax, cy, cx, radius=15, color="yellow"):
    ax.add_patch(plt.Circle((cx, cy), radius, fill=False, color=color, linewidth=1.2))
    ax.plot([cx], [cy], marker="+", color=color, markersize=7, markeredgewidth=1.4)


def _prob(row):
    return float(row["zhang_guided_probability_malignant"])


@torch.no_grad()
def _zhang_alone_prob(model, image):
    return float(torch.sigmoid(model(image).squeeze(1))[0].detach().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", default=r"results\attention_transfer_ablation")
    parser.add_argument("--xlsx", default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    parser.add_argument("--zhang-checkpoint", default=r"results\zhang_ts_aug_full\selected_model.pt")
    parser.add_argument("--progressive-checkpoint", default=r"results\zhang_ts_progressive_prior_best_aug\best_model.pt")
    parser.add_argument("--base-checkpoint", default=r"results\base_aug\baseline_model.pt")
    parser.add_argument("--out-path", default=r"results\attention_transfer_ablation\lungx_hits_misses.png")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--count-per-group", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(args.seed)
    ablation_dir = Path(args.ablation_dir)
    prog = _read_predictions(ablation_dir / "progressive_prior_guide_lungx_predictions.csv")
    rand = _read_predictions(ablation_dir / "random_noise_guide_lungx_predictions.csv")
    base_pred = _read_predictions(ablation_dir / "base_aug_guide_lungx_predictions.csv")
    records = _record_lookup(args.xlsx)

    hits = [key for key, row in prog.items() if int(row["correct"]) == 1]
    misses = [key for key, row in prog.items() if int(row["correct"]) == 0]
    chosen = []
    hit_indices = rng.choice(len(hits), size=min(args.count_per_group, len(hits)), replace=False)
    miss_indices = rng.choice(len(misses), size=min(args.count_per_group, len(misses)), replace=False)
    chosen.extend(("HIT", hits[int(index)]) for index in hit_indices)
    chosen.extend(("MISS", misses[int(index)]) for index in miss_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    zhang_model = _load_model(args.zhang_checkpoint, device)
    prog_model = _load_model(args.progressive_checkpoint, device)
    base_model = _load_model(args.base_checkpoint, device)

    fig, axes = plt.subplots(len(chosen), 6, figsize=(18, 3.1 * len(chosen)))
    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)
    headers = [
        "CT + centre",
        "Progressive guide",
        "Random guide",
        "Base guide",
        "Zhang alone",
        "Zhang + prior guide",
    ]

    for row_idx, (status, key) in enumerate(chosen):
        scan_id, nodule_number = key
        pred_row = prog[key]
        record = records[key]
        raw_slice = _load_hu_slice(pred_row["dicom_path"])
        image_tensor = _prepare_image(raw_slice).unsqueeze(0).to(device)
        image_np = image_tensor[0, 0].detach().cpu().numpy()
        cy, cx = _centre_224(record, raw_slice)

        prog_cam = _gradcam(prog_model, image_tensor, device)[0].detach().cpu().numpy()
        base_cam = _gradcam(base_model, image_tensor, device)[0].detach().cpu().numpy()
        noise = torch.randn_like(torch.from_numpy(prog_cam)).abs()
        noise = (noise / (noise.max() + 1e-8)).numpy()
        guided_prior = _guided_image(image_tensor, torch.from_numpy(prog_cam).unsqueeze(0).to(device), 0.5)
        zhang_prob = _zhang_alone_prob(zhang_model, image_tensor)

        panels = [
            (image_np, "gray", None, ""),
            (_overlay(image_np, prog_cam), None, None, f"p={_prob(prog[key]):.3f}"),
            (_overlay(image_np, noise), None, None, f"p={_prob(rand[key]):.3f}"),
            (_overlay(image_np, base_cam), None, None, f"p={_prob(base_pred[key]):.3f}"),
            (image_np, "gray", None, f"p={zhang_prob:.3f}"),
            (guided_prior[0, 0].detach().cpu().numpy(), "gray", None, f"p={_prob(prog[key]):.3f}"),
        ]
        for col_idx, (data, cmap, _unused, subtitle) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            _circle(ax, cy, cx)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(headers[col_idx], fontsize=10)
            if subtitle:
                ax.text(
                    0.02,
                    0.98,
                    subtitle,
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    fontsize=8,
                    color="white",
                    bbox=dict(facecolor="black", alpha=0.6, edgecolor="none", pad=2),
                )
            if col_idx == 0:
                true_label = int(pred_row["label"])
                ax.set_ylabel(f"{status}\n{scan_id} n{nodule_number}\ntrue={true_label}", fontsize=8)

    fig.suptitle(
        "Attention Transfer Ablation - LUNGx Hits and Misses\n"
        "Yellow circle = approximate nodule centre (15px radius)",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    with open("PROGRESS.md", "a", encoding="utf-8") as file:
        file.write(
            "\n\n## Attention Transfer Ablation Visual Summary\n"
            f"- Output: `{out_path}`\n"
            "- 5 progressive-guide hits and 5 misses from LUNGx visualised.\n"
        )
    zhang_model.remove_hooks()
    prog_model.remove_hooks()
    base_model.remove_hooks()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
