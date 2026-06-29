"""Generate before/after examples for the HU lung-mask filling change."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl

from build_cache import (
    HU_MAX,
    HU_MIN,
    MALIGNANCY_THRESHOLD,
    MIN_RADIOLOGISTS,
    SEED,
    _get_lung_mask,
    _get_middle_slice_idx,
    _normalise_hu_slice,
    _resize_image_224,
    _resize_mask_224,
)
from cache_build_step_preview import _legacy_hu_mask_for_before_after
from lidc_matching import _annotation_masks_full


def _overlay_added(image, old_mask, new_mask):
    base = np.stack([image, image, image], axis=-1)
    added = new_mask.astype(bool) & ~old_mask.astype(bool)
    removed = old_mask.astype(bool) & ~new_mask.astype(bool)
    overlay = base.copy()
    overlay[added, 0] *= 0.25
    overlay[added, 1] = 1.0
    overlay[added, 2] *= 0.25
    overlay[removed, 0] = 1.0
    overlay[removed, 1] *= 0.25
    overlay[removed, 2] *= 0.25
    return overlay


def _overlay_lung_and_nodule(image, lung_mask, nodule_mask):
    base = np.stack([image, image, image], axis=-1)
    overlay = base.copy()
    lung = lung_mask.astype(bool)
    nodule = nodule_mask.astype(bool)
    overlay[lung, 0] = 1.0
    overlay[lung, 1] *= 0.35
    overlay[lung, 2] *= 0.35
    overlay[nodule, 0] = 1.0
    overlay[nodule, 1] = 0.9
    overlay[nodule, 2] = 0.0
    return overlay


def _overlay_nodule_only(image, nodule_mask):
    base = np.stack([image, image, image], axis=-1)
    overlay = base.copy()
    nodule = nodule_mask.astype(bool)
    overlay[nodule, 0] = 1.0
    overlay[nodule, 1] = 0.9
    overlay[nodule, 2] = 0.0
    return overlay


def _accepted_examples(count, seed):
    scans = pl.query(pl.Scan).all()
    rng = np.random.default_rng(seed)
    scans = [scans[i] for i in rng.permutation(len(scans))]

    examples = []
    for scan_order, scan in enumerate(scans):
        if len(examples) >= count:
            break
        try:
            volume = scan.to_volume()
            groups = scan.cluster_annotations()
        except Exception as error:
            print(f"WARNING: skipped scan {getattr(scan, 'patient_id', '?')}: {error}")
            continue

        group_order = rng.permutation(len(groups)) if groups else []
        for group_index in group_order:
            if len(examples) >= count:
                break
            group = groups[int(group_index)]
            if len(group) < MIN_RADIOLOGISTS:
                continue
            avg_malignancy = float(np.mean([ann.malignancy for ann in group]))
            if avg_malignancy == MALIGNANCY_THRESHOLD:
                continue
            slice_idx = int(
                np.clip(_get_middle_slice_idx(group), 0, volume.shape[2] - 1)
            )
            raw_slice = volume[:, :, slice_idx].astype(np.float32)
            try:
                _union_raw, _majority_raw, _union_224, majority_224, _readers = (
                    _annotation_masks_full(group, slice_idx, raw_slice.shape)
                )
            except Exception as error:
                print(
                    f"WARNING: skipped nodule contour for {scan.patient_id} "
                    f"group {group_index}: {error}"
                )
                continue
            old_raw = _legacy_hu_mask_for_before_after(raw_slice)
            new_raw = _get_lung_mask(raw_slice)
            image_224 = _resize_image_224(_normalise_hu_slice(raw_slice))
            old_mask = _resize_mask_224(old_raw)
            new_mask = _resize_mask_224(new_raw)
            nodule = majority_224.astype(bool)
            old_overlap = int((old_mask.astype(bool) & nodule).sum())
            new_overlap = int((new_mask.astype(bool) & nodule).sum())
            nodule_pixels = int(nodule.sum())
            examples.append(
                {
                    "patient_id": scan.patient_id,
                    "scan_id": str(getattr(scan, "id", scan.patient_id)),
                    "scan_order": scan_order,
                    "group_index": int(group_index),
                    "slice_idx": slice_idx,
                    "avg_malignancy": avg_malignancy,
                    "label": int(avg_malignancy > MALIGNANCY_THRESHOLD),
                    "image": image_224,
                    "old_mask": old_mask,
                    "new_mask": new_mask,
                    "nodule_mask": majority_224.astype(np.uint8),
                    "old_pixels": int(old_mask.sum()),
                    "new_pixels": int(new_mask.sum()),
                    "added_pixels": int(
                        (new_mask.astype(bool) & ~old_mask.astype(bool)).sum()
                    ),
                    "nodule_pixels": nodule_pixels,
                    "old_nodule_coverage": old_overlap / max(nodule_pixels, 1),
                    "new_nodule_coverage": new_overlap / max(nodule_pixels, 1),
                }
            )
    if len(examples) < count:
        raise RuntimeError(f"Only found {len(examples)} accepted examples.")
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED + 100)
    parser.add_argument("--out-dir", default="results/hu_mask_before_after")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")

    os.makedirs(args.out_dir, exist_ok=True)
    examples = _accepted_examples(args.count, args.seed)

    fig, axes = plt.subplots(args.count, 6, figsize=(19, args.count * 3.0))
    if args.count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, example in enumerate(examples):
        title = (
            f"{row + 1}. {example['patient_id']} slice {example['slice_idx']} "
            f"label {example['label']} avg {example['avg_malignancy']:.2f}"
        )
        panels = [
            ("CT", example["image"], "gray"),
            ("Cancer contour", _overlay_nodule_only(example["image"], example["nodule_mask"]), None),
            ("Old HU mask", example["old_mask"], "gray"),
            ("New filled HU mask", example["new_mask"], "gray"),
            (
                f"Added +{example['added_pixels']} px",
                _overlay_added(example["image"], example["old_mask"], example["new_mask"]),
                None,
            ),
            (
                f"New mask + cancer\ncovers {example['new_nodule_coverage']:.0%}",
                _overlay_lung_and_nodule(
                    example["image"],
                    example["new_mask"],
                    example["nodule_mask"],
                ),
                None,
            ),
        ]
        for col, (panel_title, image, cmap) in enumerate(panels):
            ax = axes[row, col]
            ax.imshow(image, cmap=cmap)
            ax.axis("off")
            ax.set_title(panel_title if row == 0 else "", fontsize=10)
            if col == 0:
                ax.text(
                    0.0,
                    -0.08,
                    title,
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                )

    fig.suptitle("HU Mask Before vs After Fill/Closing", fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    png_path = os.path.join(args.out_dir, "hu_mask_before_after_10.png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    report_path = os.path.join(args.out_dir, "hu_mask_before_after_10.txt")
    with open(report_path, "w") as file:
        file.write("HU mask before/after examples\n")
        file.write("Mask generation uses CT intensities only; nodule annotations are not used.\n\n")
        for index, example in enumerate(examples, start=1):
            file.write(
                f"{index}. patient={example['patient_id']} scan={example['scan_id']} "
                f"group={example['group_index']} slice={example['slice_idx']} "
                f"label={example['label']} avg_malignancy={example['avg_malignancy']:.3f} "
                f"old_pixels={example['old_pixels']} new_pixels={example['new_pixels']} "
                f"added_pixels={example['added_pixels']} "
                f"nodule_pixels={example['nodule_pixels']} "
                f"old_nodule_coverage={example['old_nodule_coverage']:.3f} "
                f"new_nodule_coverage={example['new_nodule_coverage']:.3f}\n"
            )

    print(f"Saved PNG -> {png_path}")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
