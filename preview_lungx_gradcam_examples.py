"""Preview LUNGx Grad-CAM examples using nodule centre annotations.

LUNGx provides nodule centre points, not contours. The yellow marker/circle in
these figures is therefore an approximate visual reference only, not a contour.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from config import IMG_SIZE
from evaluate_lungx import (
    _case_dirs,
    _dicom_index,
    _load_hu_slice,
    _load_lungx_rows,
    _prepare_image,
)
from model import NoduleClassifier
from train_tristream import TriStreamNet, _lungx_streams


DEFAULT_MODELS = [
    (
        "base_aug",
        "single",
        r"results\base_aug\baseline_model.pt",
    ),
    (
        "zhang_ts_aug",
        "single",
        r"results\zhang_ts_aug\selected_model.pt",
    ),
    (
        "progressive_prior_best",
        "single",
        r"results\zhang_ts_progressive_prior_best_aug\best_model.pt",
    ),
    (
        "tristream_dil1",
        "tristream",
        r"results\tristream_dil1_aug\best_model.pt",
    ),
    (
        "tristream_nocandidate",
        "tristream",
        r"results\tristream_nocandidate_aug\best_model.pt",
    ),
]


def _load_single(path, device):
    model = NoduleClassifier(input_channels=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _load_tristream(path, device):
    model = TriStreamNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _single_gradcam(model, raw_slice, device):
    image = _prepare_image(raw_slice).unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    model.clear_hooks()
    logits = model(image).squeeze(1)
    probability = torch.sigmoid(logits)[0].item()
    score = logits if probability >= 0.5 else -logits
    score.sum().backward()
    cam = model.get_gradcam(normalise=True)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu().numpy()
    model.clear_hooks()
    return cam, probability


def _tristream_gradcam(model, raw_slice, device, ablation):
    x1, x2, x3 = _lungx_streams(raw_slice, ablation)
    x1 = x1.unsqueeze(0).to(device)
    x2 = x2.unsqueeze(0).to(device)
    x3 = x3.unsqueeze(0).to(device)
    model.zero_grad(set_to_none=True)
    logits = model(x1, x2, x3)
    probability = torch.sigmoid(logits)[0].item()
    score = logits if probability >= 0.5 else -logits
    score.sum().backward()
    cam = model.stream1.gradcam()
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu().numpy()
    return cam, probability


def _normalised_image(raw_slice):
    tensor = _prepare_image(raw_slice)
    return tensor[0].detach().cpu().numpy()


def _center_224(record, raw_shape):
    # LUNGx sheet coordinates are x,y and appear 1-based.
    x = (float(record["center_x_1based"]) - 1.0) * IMG_SIZE / float(raw_shape[1])
    y = (float(record["center_y_1based"]) - 1.0) * IMG_SIZE / float(raw_shape[0])
    return x, y


def _overlay(image, cam):
    heat = plt.get_cmap("magma")(cam)[..., :3]
    base = np.stack([image, image, image], axis=-1)
    return np.clip(base * 0.55 + heat * 0.45, 0.0, 1.0)


def _draw_marker(ax, center, radius=8):
    x, y = center
    ax.scatter([x], [y], c="yellow", s=24, marker="+", linewidths=1.5)
    circle = plt.Circle((x, y), radius, color="yellow", fill=False, linewidth=1.0)
    ax.add_patch(circle)


def _pick_records(records, count, seed, labelled_only=True):
    usable = [row for row in records if (row["label"] is not None or not labelled_only)]
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(usable), size=min(count, len(usable)), replace=False)
    return [usable[int(index)] for index in indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", default=r"C:\repo\manifest-cgqtDj7Y2699835271585651107")
    parser.add_argument("--xlsx", default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    parser.add_argument("--out-dir", default=r"results\lungx_gradcam_visual_compare_seed42")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    image_root = os.path.join(args.manifest_root, "SPIE-AAPM Lung CT Challenge")
    cases = _case_dirs(image_root)
    records = _pick_records(_load_lungx_rows(args.xlsx), args.count, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded = []
    for name, kind, path in DEFAULT_MODELS:
        if not os.path.exists(path):
            print(f"Skipping missing model: {name} ({path})")
            continue
        model = _load_single(path, device) if kind == "single" else _load_tristream(path, device)
        loaded.append((name, kind, model))

    dicom_cache = {}
    case_rows = []
    for record in records:
        if record["scan_id"] not in dicom_cache:
            dicom_cache[record["scan_id"]] = _dicom_index(cases[record["scan_id"]])
        entries, by_instance = dicom_cache[record["scan_id"]]
        dicom_path = by_instance.get(record["center_image"])
        if dicom_path is None:
            dicom_path = entries[max(0, min(record["center_image"] - 1, len(entries) - 1))][2]
        raw_slice = _load_hu_slice(dicom_path)
        image = _normalised_image(raw_slice)
        center = _center_224(record, raw_slice.shape)
        case_rows.append((record, raw_slice, image, center, dicom_path))

    fig, axes = plt.subplots(
        len(case_rows),
        len(loaded) + 1,
        figsize=((len(loaded) + 1) * 3.2, len(case_rows) * 3.0),
    )
    if len(case_rows) == 1:
        axes = axes[None, :]

    summary = [
        "LUNGx Grad-CAM visual comparison",
        "Yellow marker/circle = approximate nodule centre only; LUNGx has no contours.",
        f"Seed: {args.seed}",
        "",
    ]

    for row_idx, (record, raw_slice, image, center, dicom_path) in enumerate(case_rows):
        ax = axes[row_idx, 0]
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        _draw_marker(ax, center)
        ax.axis("off")
        ax.set_title("CT + centre" if row_idx == 0 else "", fontsize=9)
        ax.text(
            0.0,
            -0.08,
            (
                f"{record['scan_id']} n{record['nodule_number']} "
                f"label={record['label']} img={record['center_image']}"
            ),
            transform=ax.transAxes,
            fontsize=7,
            va="top",
        )

        case_name = f"{record['scan_id']}_n{record['nodule_number']}_img{record['center_image']}"
        summary.append(f"{case_name}: label={record['label']} diagnosis={record['diagnosis']} path={dicom_path}")

        for col_idx, (name, kind, model) in enumerate(loaded, start=1):
            if kind == "single":
                cam, probability = _single_gradcam(model, raw_slice, device)
            else:
                ablation = "nocandidate" if name == "tristream_nocandidate" else "full"
                cam, probability = _tristream_gradcam(model, raw_slice, device, ablation)
            ax = axes[row_idx, col_idx]
            ax.imshow(_overlay(image, cam))
            _draw_marker(ax, center)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(name, fontsize=9)
            ax.text(
                0.0,
                -0.08,
                f"p={probability:.2f}",
                transform=ax.transAxes,
                fontsize=7,
                va="top",
            )

        indiv_path = os.path.join(args.out_dir, f"{case_name}.png")
        indiv_fig, indiv_axes = plt.subplots(1, len(loaded) + 1, figsize=((len(loaded) + 1) * 3.2, 3.0))
        indiv_axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
        _draw_marker(indiv_axes[0], center)
        indiv_axes[0].set_title("CT + centre", fontsize=9)
        indiv_axes[0].axis("off")
        for col_idx, (name, kind, model) in enumerate(loaded, start=1):
            if kind == "single":
                cam, probability = _single_gradcam(model, raw_slice, device)
            else:
                ablation = "nocandidate" if name == "tristream_nocandidate" else "full"
                cam, probability = _tristream_gradcam(model, raw_slice, device, ablation)
            indiv_axes[col_idx].imshow(_overlay(image, cam))
            _draw_marker(indiv_axes[col_idx], center)
            indiv_axes[col_idx].set_title(f"{name}\np={probability:.2f}", fontsize=8)
            indiv_axes[col_idx].axis("off")
        indiv_fig.tight_layout()
        indiv_fig.savefig(indiv_path, dpi=180)
        plt.close(indiv_fig)

    for _name, kind, model in loaded:
        if kind == "single":
            model.remove_hooks()
        else:
            model.remove_hooks()

    fig.suptitle(
        "LUNGx Grad-CAM comparison. Yellow circle marks approximate nodule centre, not contour.",
        fontsize=12,
        y=0.997,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.992))
    grid_path = os.path.join(args.out_dir, "lungx_gradcam_same5_grid.png")
    fig.savefig(grid_path, dpi=180)
    plt.close(fig)

    with open(os.path.join(args.out_dir, "lungx_gradcam_same5_summary.txt"), "w") as file:
        file.write("\n".join(summary) + "\n")

    print(f"Saved grid: {grid_path}")
    print(f"Saved individual PNGs and summary to: {args.out_dir}")


if __name__ == "__main__":
    main()
