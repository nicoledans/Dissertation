"""Create a visual, real-scan walkthrough of the cache preprocessing pipeline."""

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
    SLICE_OFFSET_MM,
    SLICE_OFFSET_TOLERANCE_MM,
    _get_consensus_nodule_center,
    _get_lung_mask,
    _get_middle_slice_idx,
    _get_ts_vol_mask,
    _normalise_hu_slice,
    _resize_image_224,
    _resize_mask_224,
    _select_25d_slice_indices,
)
from lidc_matching import _annotation_masks_full
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.morphology import disk as morpho_disk


def _legacy_hu_mask_for_before_after(slice_2d):
    """Previous HU mask behavior before filling internal lung holes."""
    from build_cache import LUNG_HU_THRESHOLD, MASK_DILATION
    from scipy.ndimage import center_of_mass, label

    air = slice_2d < LUNG_HU_THRESHOLD
    body = slice_2d >= LUNG_HU_THRESHOLD
    body_labeled, n_body = label(body)
    if n_body == 0:
        return np.zeros(slice_2d.shape, dtype=np.uint8)

    body_sizes = np.bincount(body_labeled.ravel())[1:]
    body_label = int(np.argmax(body_sizes) + 1)
    body_filled = binary_fill_holes(body_labeled == body_label)

    internal_air = air & body_filled
    labeled, _ = label(internal_air)

    border_labels = set()
    border_labels.update(labeled[0, :])
    border_labels.update(labeled[-1, :])
    border_labels.update(labeled[:, 0])
    border_labels.update(labeled[:, -1])
    border_labels.discard(0)
    if border_labels:
        labeled = labeled.copy()
        for lbl in border_labels:
            labeled[labeled == lbl] = 0

    labeled2, num2 = label(labeled > 0)
    if num2 == 0:
        return np.zeros(slice_2d.shape, dtype=np.uint8)

    region_sizes = np.bincount(labeled2.ravel())[1:]
    keep_labels = np.argsort(region_sizes)[::-1][:min(2, len(region_sizes))] + 1
    lung_regions = [labeled2 == lbl for lbl in keep_labels]
    if len(lung_regions) == 2:
        cx = [center_of_mass(m)[1] for m in lung_regions]
        lungs = [
            lung_regions[int(np.argmin(cx))],
            lung_regions[int(np.argmax(cx))],
        ]
    else:
        lungs = lung_regions

    combined = lungs[0].copy()
    for lung in lungs[1:]:
        combined |= lung
    return binary_dilation(combined, structure=morpho_disk(MASK_DILATION)).astype(np.uint8)

try:
    import torch
except ImportError:
    torch = None


def _display_hu(image):
    return np.clip(image, HU_MIN, HU_MAX)


def _overlay(image, mask):
    base = np.stack([image, image, image], axis=-1)
    mask_bool = mask.astype(bool)
    overlay = base.copy()
    overlay[mask_bool, 0] = 1.0
    overlay[mask_bool, 1] *= 0.35
    overlay[mask_bool, 2] *= 0.35
    return overlay


def _overlay_green(image, mask):
    base = np.stack([image, image, image], axis=-1)
    mask_bool = mask.astype(bool)
    overlay = base.copy()
    overlay[mask_bool, 0] *= 0.25
    overlay[mask_bool, 1] = 1.0
    overlay[mask_bool, 2] *= 0.25
    return overlay


def _overlay_nodule(image, nodule_mask, center=None):
    base = np.stack([image, image, image], axis=-1)
    mask_bool = nodule_mask.astype(bool)
    overlay = base.copy()
    overlay[mask_bool, 0] = 1.0
    overlay[mask_bool, 1] = 0.85
    overlay[mask_bool, 2] *= 0.15
    if center is not None:
        row, col = [int(round(value)) for value in center]
        rr0, rr1 = max(row - 4, 0), min(row + 5, overlay.shape[0])
        cc0, cc1 = max(col - 4, 0), min(col + 5, overlay.shape[1])
        overlay[rr0:rr1, col:col + 1, :] = [0.0, 1.0, 1.0]
        overlay[row:row + 1, cc0:cc1, :] = [0.0, 1.0, 1.0]
    return overlay


def _find_first_accepted():
    scans = pl.query(pl.Scan).all()
    rng = np.random.default_rng(SEED)
    shuffled = [scans[i] for i in rng.permutation(len(scans))]

    for scan_order, scan in enumerate(shuffled):
        volume = scan.to_volume()
        groups = scan.cluster_annotations()
        for nodule_index, group in enumerate(groups):
            if len(group) < MIN_RADIOLOGISTS:
                continue
            avg_malignancy = float(np.mean([ann.malignancy for ann in group]))
            if avg_malignancy == MALIGNANCY_THRESHOLD:
                continue
            slice_idx = int(np.clip(_get_middle_slice_idx(group), 0, volume.shape[2] - 1))
            return scan_order, scan, volume, nodule_index, group, avg_malignancy, slice_idx
    raise RuntimeError("No accepted LIDC nodule found.")


def _write_report(path, details, mismatches):
    lines = [
        "Cache preprocessing preview",
        "",
        f"Patient ID: {details['patient_id']}",
        f"Scan ID: {details['scan_id']}",
        f"Shuffled scan index: {details['scan_order']}",
        f"Nodule group index: {details['nodule_index']}",
        f"Radiologist annotations: {details['reader_count']}",
        f"Average malignancy: {details['avg_malignancy']:.3f}",
        f"Binary label: {details['label']} ({'malignant' if details['label'] else 'benign'})",
        f"Selected middle slice index: {details['slice_idx']}",
        f"Consensus nodule centre (row, col): {details['center']}",
        f"Raw volume shape: {details['volume_shape']}",
        f"Raw slice HU range: {details['raw_range']}",
        f"Normalised image range: {details['norm_range']}",
        f"Old HU mask pixels after resize: {details['legacy_hu_mask_pixels']}",
        f"New filled HU mask pixels after resize: {details['hu_mask_pixels']}",
        f"Newly included HU mask pixels after fill: {details['hu_mask_added_pixels']}",
        f"Radiologist-majority nodule pixels after resize: {details['nodule_majority_pixels_224']}",
        f"TS mask available in preview: {details['ts_available']}",
        f"2.5D preview available: {details['stack_available']}",
        "",
        "Flags / caveats compared with your checklist:",
    ]
    lines.extend([f"- {item}" for item in mismatches])
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="results/cache_build_step_preview")
    parser.add_argument(
        "--with-ts",
        action="store_true",
        help="Also run TotalSegmentator for the selected scan. This can be slow.",
    )
    parser.add_argument(
        "--show-25d",
        action="store_true",
        help="Also show the current default 2.5D stack selection.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    (
        scan_order,
        scan,
        volume,
        nodule_index,
        group,
        avg_malignancy,
        slice_idx,
    ) = _find_first_accepted()

    raw_slice = volume[:, :, slice_idx].astype(np.float32)
    _union_raw, majority_raw, _union_224, majority_224, reader_count = (
        _annotation_masks_full(group, slice_idx, raw_slice.shape)
    )
    legacy_hu_mask_raw = _legacy_hu_mask_for_before_after(raw_slice)
    hu_mask_raw = _get_lung_mask(raw_slice)
    normalised = _normalise_hu_slice(raw_slice)
    image_224 = _resize_image_224(normalised)
    legacy_mask_224 = _resize_mask_224(legacy_hu_mask_raw)
    mask_224 = _resize_mask_224(hu_mask_raw)
    center = _get_consensus_nodule_center(group)
    center_224 = (
        center[0] * 224.0 / raw_slice.shape[0],
        center[1] * 224.0 / raw_slice.shape[1],
    )
    label = int(avg_malignancy > MALIGNANCY_THRESHOLD)

    ts_mask_224 = None
    if args.with_ts:
        if torch is None:
            raise RuntimeError("PyTorch is required for TotalSegmentator device selection.")
        device = "gpu" if torch.cuda.is_available() else "cpu"
        ts_volume = _get_ts_vol_mask(volume, scan, device)
        if ts_volume is not None:
            ts_mask_224 = _resize_mask_224(ts_volume[:, :, slice_idx].astype(np.float32))

    stack_channels = None
    stack_indices = None
    if args.show_25d:
        meta = _select_25d_slice_indices(
            scan,
            slice_idx,
            SLICE_OFFSET_MM,
            SLICE_OFFSET_TOLERANCE_MM,
        )
        stack_indices = meta["indices"]
        stack_channels = [
            _resize_image_224(_normalise_hu_slice(volume[:, :, idx].astype(np.float32)))
            for idx in stack_indices
        ]

    added_mask = np.logical_and(mask_224.astype(bool), ~legacy_mask_224.astype(bool))

    panels = [
        ("1. raw middle slice\nHU windowed for display", _display_hu(raw_slice), "gray"),
        ("2. cancer nodule\nradiologist majority + centre", _overlay_nodule(normalised, majority_raw, center), None),
        ("3. HU air threshold\nslice < -500 HU", raw_slice < -500, "gray"),
        ("4. old HU mask\nbefore hole fill", legacy_hu_mask_raw, "gray"),
        ("5. new HU mask\nfilled + dilated", hu_mask_raw, "gray"),
        (
            "6. filled pixels added\ngreen = newly included",
            _overlay_green(image_224, added_mask),
            None,
        ),
        ("5. clipped + normalised\n[-1000,400] -> [0,1]", normalised, "gray"),
        ("6. resized image\n224 x 224", image_224, "gray"),
        ("7. resized cancer nodule\nnot saved to training cache", _overlay_nodule(image_224, majority_224, center_224), None),
        ("8. old resized HU mask", legacy_mask_224, "gray"),
        ("9. new resized HU mask", mask_224, "gray"),
        ("10. new HU mask overlay\ntraining attention mask", _overlay(image_224, mask_224), None),
    ]
    if ts_mask_224 is not None:
        panels.append(("8. TS mask resized\nnearest + rebinarise", ts_mask_224, "gray"))
        panels.append(("9. TS overlay\noptional mask source", _overlay(image_224, ts_mask_224), None))
    if stack_channels is not None:
        for idx, channel in zip(stack_indices, stack_channels):
            panels.append((f"2.5D channel\nslice {idx}", channel, "gray"))

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 4.0))
    axes = np.asarray(axes).reshape(-1)
    for ax, (title, image, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        f"{scan.patient_id} | nodule group {nodule_index} | "
        f"avg malignancy {avg_malignancy:.2f} -> label {label}",
        fontsize=12,
    )
    fig.tight_layout()
    png_path = os.path.join(args.out_dir, "cache_build_steps.png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)

    mismatches = [
        "The active default cache path uses 2.5D full-slice inputs when --slice-offset-mm is supplied; then the saved image is a 3-channel stack, not a single 224 x 224 array.",
        "Invalid-sample rejection is distributed: build_cache.py rejects empty HU masks, exact duplicate model inputs, invalid 2.5D neighbours, and caught exceptions; full schema validation happens later in dataset.py.",
        "Per-nodule checkpoints are saved before appending the sample, deliberately preferring a possible missed sample over a duplicate if interrupted at exactly the wrong moment.",
        "Training loaders require completed plain-list cache files; checkpoint dictionaries are for build resume only.",
    ]
    if not args.with_ts:
        mismatches.append("TotalSegmentator was not run for this preview PNG; rerun with --with-ts to visualise that optional branch.")
    if not args.show_25d:
        mismatches.append("2.5D channels were not drawn in this preview; rerun with --show-25d to visualise the current stacked-input branch.")

    details = {
        "patient_id": scan.patient_id,
        "scan_id": str(getattr(scan, "id", scan.patient_id)),
        "scan_order": scan_order,
        "nodule_index": nodule_index,
        "reader_count": reader_count,
        "avg_malignancy": avg_malignancy,
        "label": label,
        "slice_idx": slice_idx,
        "center": tuple(round(v, 2) for v in center),
        "volume_shape": tuple(int(v) for v in volume.shape),
        "raw_range": (float(raw_slice.min()), float(raw_slice.max())),
        "norm_range": (float(normalised.min()), float(normalised.max())),
        "hu_mask_pixels": int(mask_224.sum()),
        "legacy_hu_mask_pixels": int(legacy_mask_224.sum()),
        "hu_mask_added_pixels": int(added_mask.sum()),
        "nodule_majority_pixels_224": int(majority_224.sum()),
        "ts_available": ts_mask_224 is not None,
        "stack_available": stack_channels is not None,
    }
    report_path = os.path.join(args.out_dir, "cache_build_steps_report.txt")
    _write_report(report_path, details, mismatches)

    print(f"Saved PNG -> {png_path}")
    print(f"Saved report -> {report_path}")


if __name__ == "__main__":
    main()
