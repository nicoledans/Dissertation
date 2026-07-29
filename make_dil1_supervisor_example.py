"""Create one compact before/after TS dil1 mask example for presentation."""

import argparse
import csv
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_erosion

from build_soft_blob_cache import _hashable_center_image
from lidc_matching import _eligible_candidates, _image_hash


def _load_cache(path):
    with open(path, "rb") as file:
        return pickle.load(file)


def _center_image(sample):
    image = np.asarray(sample["image"], dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image[1]
    return np.clip(_hashable_center_image(image), 0.0, 1.0)


def _mask(sample):
    value = np.asarray(sample["ts_mask"])
    if value.ndim == 3 and value.shape[0] == 3:
        value = value[1]
    return value.astype(bool)


def _contour_for_sample(sample):
    import pylidc as pl

    candidates = []
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == sample["patient_id"]).all()
    for scan in scans:
        candidates.extend(_eligible_candidates(scan))
    by_key = defaultdict(list)
    for candidate in candidates:
        by_key[(candidate["label"], candidate["hash"])].append(candidate)

    key = (int(sample["label"]), _image_hash(_hashable_center_image(sample["image"])))
    matches = by_key.get(key, [])
    if not matches:
        raise RuntimeError(f"No contour match found for {sample['patient_id']}")
    return np.asarray(matches[0]["majority"], dtype=bool)


def _outline(mask):
    return mask & ~binary_erosion(mask)


def _colored_overlay(image, mask, nodule):
    rgb = np.stack([image, image, image], axis=-1)
    covered = nodule & mask
    missed = nodule & ~mask
    rgb[covered] = [0.0, 0.95, 0.25]
    rgb[missed] = [1.0, 0.05, 0.05]
    return rgb


def _set_common(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-cache", default=r"cache\cache_ts.pkl")
    parser.add_argument("--new-cache", default=r"cache\cache_ts_filled_dil1.pkl")
    parser.add_argument("--fixed-csv", default=r"results\mask_nodule_overlap_ts_filled_dil1\fixed_from_original_failures.csv")
    parser.add_argument("--cache-index", type=int, default=903)
    parser.add_argument("--top-fixed", type=int, default=0, help="Generate this many top fixed examples by old missed pixels.")
    parser.add_argument("--out-dir", default=r"results\dil1_supervisor_example")
    args = parser.parse_args()

    old_samples = _load_cache(args.old_cache)
    new_samples = _load_cache(args.new_cache)
    fixed_rows = []
    with open(args.fixed_csv, newline="", encoding="utf-8") as file:
        fixed_rows = list(csv.DictReader(file))
    if args.top_fixed > 0:
        selected_rows = sorted(
            fixed_rows,
            key=lambda row: int(float(row["old_missed_pixels"])),
            reverse=True,
        )[: args.top_fixed]
    else:
        selected_rows = [
            row for row in fixed_rows if int(row["cache_index"]) == args.cache_index
        ]
        if not selected_rows:
            selected_rows = [{"cache_index": str(args.cache_index)}]

    os.makedirs(args.out_dir, exist_ok=True)
    saved = []
    for selected in selected_rows:
        args.cache_index = int(selected["cache_index"])
        old = old_samples[args.cache_index]
        new = new_samples[args.cache_index]
        out_path = _make_one(args, old, new, selected)
        saved.append(out_path)
    print("Saved:")
    for path in saved:
        print(f"  {path}")


def _make_one(args, old, new, fixed_row):
    image = _center_image(old)
    old_mask = _mask(old)
    new_mask = _mask(new)
    nodule = _contour_for_sample(old)
    added = new_mask & ~old_mask

    if not fixed_row or "old_coverage" not in fixed_row:
        fixed_row = {
            "old_coverage": float((old_mask & nodule).sum() / max(nodule.sum(), 1)),
            "new_coverage": float((new_mask & nodule).sum() / max(nodule.sum(), 1)),
            "old_missed_pixels": int((nodule & ~old_mask).sum()),
            "new_missed_pixels": int((nodule & ~new_mask).sum()),
            "nodule_pixels": int(nodule.sum()),
        }

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].contour(_outline(old_mask).astype(float), levels=[0.5], colors=["cyan"], linewidths=1.0)
    axes[0].contour(nodule.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.2)
    axes[0].set_title(
        f"Before dil1\ncoverage {float(fixed_row['old_coverage']) * 100:.1f}%"
    )

    axes[1].imshow(_colored_overlay(image, old_mask, nodule), vmin=0, vmax=1)
    axes[1].contour(_outline(old_mask).astype(float), levels=[0.5], colors=["cyan"], linewidths=1.0)
    axes[1].set_title(
        f"Before: nodule pixels\nred missed={int(float(fixed_row['old_missed_pixels']))}"
    )

    axes[2].imshow(_colored_overlay(image, new_mask, nodule), vmin=0, vmax=1)
    axes[2].contour(_outline(new_mask).astype(float), levels=[0.5], colors=["deepskyblue"], linewidths=1.0)
    axes[2].set_title(
        f"After dil1\ncoverage {float(fixed_row['new_coverage']) * 100:.1f}%"
    )

    diff = np.stack([image, image, image], axis=-1)
    diff[added] = [0.2, 0.5, 1.0]
    diff[nodule & new_mask] = [0.0, 0.95, 0.25]
    axes[3].imshow(diff, vmin=0, vmax=1)
    axes[3].contour(nodule.astype(float), levels=[0.5], colors=["yellow"], linewidths=1.2)
    axes[3].set_title("Added mask area\nblue=added, green=nodule")

    for ax in axes:
        _set_common(ax)

    fig.suptitle(
        (
            f"TS mask correction example: {old['patient_id']} | cache index {args.cache_index}\n"
            "Yellow=nodule contour, cyan/blue=TS mask outline, red=missed nodule, green=covered nodule"
        ),
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path = os.path.join(
        args.out_dir,
        f"dil1_before_after_idx_{args.cache_index}_{old['patient_id']}.png",
    )
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    main()
