"""Preview original TotalSegmentator masks vs filled/dilated TS masks.

This script is visualization-only. It draws mask borders on the original cached
CT image so the lung area is not hidden by a solid overlay.
"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_erosion, binary_fill_holes


def _load_cache(path):
    with open(path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{path} is not a completed cache list.")
    return samples


def _sample_key(sample):
    return (
        sample.get("patient_id"),
        sample.get("scan_id"),
        sample.get("nodule_id"),
        sample.get("slice_idx"),
    )


def _mask_border(mask):
    mask = np.asarray(mask).astype(bool)
    if not np.any(mask):
        return mask
    return mask & ~binary_erosion(mask)


def _disk(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _hu_gated_ts_mask(image, ts_mask_raw):
    """Keep only lung-like HU pixels inside TS, then fill/dilate for preview."""
    hu_reconstructed = np.asarray(image, dtype=np.float32) * 1400.0 - 1000.0
    lung_only = hu_reconstructed < -300.0
    ts_mask = np.asarray(ts_mask_raw).astype(bool) & lung_only.astype(bool)
    ts_mask = binary_fill_holes(ts_mask)
    ts_mask = binary_closing(ts_mask, structure=_disk(1))
    ts_mask = binary_fill_holes(ts_mask)
    ts_mask = binary_dilation(ts_mask, structure=_disk(1))
    return ts_mask.astype(bool)


def _draw_base(axis, image, title):
    axis.imshow(np.asarray(image), cmap="gray", vmin=0.0, vmax=1.0)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _draw_outline(axis, mask, color, linewidth=0.9, label=None):
    mask = np.asarray(mask).astype(bool)
    if np.any(mask):
        axis.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)
    if label:
        axis.text(
            0.02,
            0.98,
            label,
            transform=axis.transAxes,
            va="top",
            ha="left",
            color=color,
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.55, "pad": 2, "edgecolor": "none"},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-cache", default="cache/cache_ts.pkl")
    parser.add_argument("--filled-cache", default="cache/cache_ts_filled_dil1.pkl")
    parser.add_argument("--out-dir", default="results/ts_original_vs_filled_preview")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    original = _load_cache(args.original_cache)
    filled = _load_cache(args.filled_cache)
    filled_by_key = {_sample_key(sample): sample for sample in filled}

    paired = []
    for index, sample in enumerate(original):
        match = filled_by_key.get(_sample_key(sample))
        if match is not None and sample.get("ts_mask") is not None and match.get("ts_mask") is not None:
            paired.append((index, sample, match))
    if not paired:
        raise RuntimeError("No matching samples with TS masks were found.")

    rng = np.random.default_rng(args.seed)
    choice = rng.choice(len(paired), size=min(args.count, len(paired)), replace=False)
    records = [paired[int(i)] for i in choice]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = len(records)
    fig, axes = plt.subplots(rows, 5, figsize=(17.5, max(2.4 * rows, 4.0)))
    if rows == 1:
        axes = axes[None, :]

    summary_lines = [
        "Original TS vs filled/dilated TS preview",
        f"Original cache: {args.original_cache}",
        f"Filled cache: {args.filled_cache}",
        "Cyan = original cache_ts.pkl TS outline",
        "Magenta = cache_ts_filled_dil1.pkl TS outline",
        "Green = pixels added by filled/dil1 post-process",
        "",
    ]

    for row, (cache_index, original_sample, filled_sample) in enumerate(records):
        image = np.asarray(original_sample["image"])
        original_mask = np.asarray(original_sample["ts_mask"]).astype(bool)
        filled_mask = np.asarray(filled_sample["ts_mask"]).astype(bool)
        hu_gated_mask = _hu_gated_ts_mask(image, original_mask)
        added = filled_mask & ~original_mask

        title_prefix = (
            f"{cache_index} | {original_sample.get('patient_id')} | "
            f"label={original_sample.get('label')}"
        )
        summary_lines.append(
            f"{title_prefix} | original area={original_mask.mean() * 100:.2f}% | "
            f"filled area={filled_mask.mean() * 100:.2f}% | "
            f"HU-gated+cleanup area={hu_gated_mask.mean() * 100:.2f}% | "
            f"added={added.mean() * 100:.2f}%"
        )

        _draw_base(axes[row, 0], image, f"{title_prefix}\nCT only")

        _draw_base(axes[row, 1], image, "CT + original TS outline")
        _draw_outline(axes[row, 1], original_mask, "cyan", 0.9, "original")

        _draw_base(axes[row, 2], image, "CT + filled/dil1 TS outline")
        _draw_outline(axes[row, 2], filled_mask, "magenta", 0.9, "filled/dil1")

        _draw_base(axes[row, 3], image, "Both outlines + added border")
        _draw_outline(axes[row, 3], original_mask, "cyan", 0.8, "original")
        _draw_outline(axes[row, 3], filled_mask, "magenta", 0.8, "filled")
        _draw_outline(axes[row, 3], added, "lime", 1.0, "added")

        _draw_base(axes[row, 4], image, "HU-gated TS + cleanup")
        _draw_outline(axes[row, 4], hu_gated_mask, "yellow", 0.9, "HU<-300")

    fig.suptitle(
        "TS lung mask comparison: original vs close+dil1 cache",
        fontsize=13,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    out_png = os.path.join(args.out_dir, "ts_original_vs_filled_10_random.png")
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_txt = os.path.join(args.out_dir, "ts_original_vs_filled_10_random.txt")
    with open(out_txt, "w") as file:
        file.write("\n".join(summary_lines) + "\n")

    print(f"Saved PNG: {out_png}")
    print(f"Saved summary: {out_txt}")


if __name__ == "__main__":
    main()
