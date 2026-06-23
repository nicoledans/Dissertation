"""Preview soft blob maps against LIDC contours for visual smoke testing.

This script does not build a cache and does not use annotations to generate the
soft maps. It reconstructs raw LIDC slices only to create the image-derived map,
then overlays majority/union contours for audit-only visual inspection.
"""

import argparse
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
from scipy.ndimage import zoom

from config import TRAIN_CACHE_PATH
from soft_blob_generator import soft_blob_maps
from visualize_nodule_gradcam import _eligible_candidates, _image_hash


def _load_cache(path):
    with open(path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{path} is not a completed cache list.")
    return samples


def _hashable_center_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        return image[1]
    return image


def _resize_mask_to_raw(mask_224, raw_shape):
    factors = (raw_shape[0] / mask_224.shape[0], raw_shape[1] / mask_224.shape[1])
    return zoom(mask_224.astype(np.uint8), factors, order=0).astype(bool)


def _resize_map_to_224(map_raw):
    factors = (224 / map_raw.shape[0], 224 / map_raw.shape[1])
    resized = zoom(map_raw.astype(np.float32), factors, order=1)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _candidate_lookup_for_patient(patient_id):
    candidates = []
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    for scan in scans:
        try:
            candidates.extend(_eligible_candidates(scan))
        except Exception as error:
            print(f"  WARNING: could not reconstruct scan {scan.id}: {error}")
    by_key = defaultdict(list)
    for candidate in candidates:
        by_key[(candidate["label"], candidate["hash"])].append(candidate)
    return by_key


def _match_samples(samples, max_samples, seed):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(samples))
    rng.shuffle(indices)
    by_patient = defaultdict(list)
    for index in indices:
        sample = samples[int(index)]
        by_patient[sample["patient_id"]].append((int(index), sample))

    records = []
    for patient_number, (patient_id, patient_samples) in enumerate(by_patient.items(), start=1):
        if len(records) >= max_samples:
            break
        print(
            f"Matching patient {patient_number}/{len(by_patient)}: "
            f"{patient_id} ({len(records)}/{max_samples})"
        )
        lookup = _candidate_lookup_for_patient(patient_id)
        for cache_index, sample in patient_samples:
            if sample.get("mask") is None:
                continue
            key = (int(sample["label"]), _image_hash(_hashable_center_image(sample["image"])))
            matches = lookup.get(key, [])
            if not matches:
                continue
            match = matches.pop(0)
            lung_raw = _resize_mask_to_raw(
                np.asarray(sample["mask"]).astype(bool),
                np.asarray(match["raw_slice"]).shape,
            )
            maps = soft_blob_maps(
                np.asarray(match["raw_slice"], dtype=np.float32),
                lung_raw,
                float(match["pixel_spacing_mm"]),
                adjacent_slices=match.get("adjacent_raw_slices", []),
            )
            records.append(
                {
                    "cache_index": cache_index,
                    "sample": sample,
                    "match": match,
                    "soft_blob_map": _resize_map_to_224(maps["soft_blob_map"]),
                    "solid_like_map": _resize_map_to_224(maps["solid_like_map"]),
                    "subsolid_like_map": _resize_map_to_224(maps["subsolid_like_map"]),
                    "search_mask": _resize_map_to_224(maps["search_mask"]) > 0.5,
                }
            )
            if len(records) >= max_samples:
                break
    return records


def _draw_contours(axis, majority, union=None):
    if union is not None and np.any(union):
        axis.contour(union.astype(float), levels=[0.5], colors=["deepskyblue"], linewidths=0.9)
    if np.any(majority):
        axis.contour(majority.astype(float), levels=[0.5], colors=["lime"], linewidths=1.3)


def _mean_inside(values, mask):
    return float(values[mask].mean()) if np.any(mask) else float("nan")


def _save_preview(records, output_path):
    columns = [
        "CT + audit contour",
        "Solid-like map",
        "Subsolid/GGO-like map",
        "Combined soft blob map",
        "High response + contour",
    ]
    fig, axes = plt.subplots(
        len(records),
        len(columns),
        figsize=(4.1 * len(columns), 3.7 * len(records)),
        squeeze=False,
    )
    for column, title in enumerate(columns):
        axes[0, column].set_title(title, fontsize=11, fontweight="bold")

    summary_lines = [
        "Soft blob preview with LIDC contours.",
        "Green contour = majority-vote nodule. Blue contour = any-reader union.",
        "Contours are audit-only and are not used to generate soft maps.",
        "",
    ]
    for row, record in enumerate(records):
        sample = record["sample"]
        match = record["match"]
        image = np.asarray(match["image"], dtype=np.float32)
        majority = np.asarray(match["majority"]).astype(bool)
        union = np.asarray(match["union"]).astype(bool)
        solid = record["solid_like_map"]
        subsolid = record["subsolid_like_map"]
        blob = record["soft_blob_map"]
        lung = np.asarray(sample["mask"]).astype(bool)
        lung_values = blob[lung] if np.any(lung) else blob.ravel()
        high_threshold = float(np.percentile(lung_values, 90.0)) if lung_values.size else 1.0
        high = blob >= high_threshold
        touched = float((high & majority).sum() / max(majority.sum(), 1) * 100.0)
        inside = _mean_inside(blob, majority)
        outside = _mean_inside(blob, lung & ~majority)

        axes[row, 0].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].contour(lung.astype(float), levels=[0.5], colors=["red"], linewidths=0.6)
        _draw_contours(axes[row, 0], majority, union)

        axes[row, 1].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].imshow(solid, cmap="magma", alpha=0.62, vmin=0, vmax=1)
        _draw_contours(axes[row, 1], majority)

        axes[row, 2].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(subsolid, cmap="viridis", alpha=0.62, vmin=0, vmax=1)
        _draw_contours(axes[row, 2], majority)

        axes[row, 3].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 3].imshow(blob, cmap="inferno", alpha=0.62, vmin=0, vmax=1)
        _draw_contours(axes[row, 3], majority)

        axes[row, 4].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, 4].imshow(np.ma.masked_where(~high, high), cmap="autumn", alpha=0.55)
        _draw_contours(axes[row, 4], majority, union)
        axes[row, 4].text(
            0.02,
            0.02,
            f"top10 hit {touched:.1f}%\ninside {inside:.3f} outside {outside:.3f}",
            transform=axes[row, 4].transAxes,
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )

        label = "malignant" if int(match["label"]) else "benign"
        axes[row, 0].set_ylabel(
            f"{match['patient_id']}\nidx={record['cache_index']} {label}\n"
            f"rating={match['mean_rating']:.2f}",
            rotation=0,
            labelpad=48,
            va="center",
            fontsize=8,
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

        summary_lines.append(
            f"index={record['cache_index']} patient={match['patient_id']} "
            f"label={match['label']} rating={match['mean_rating']:.2f} "
            f"top10_contour_hit={touched:.2f}% "
            f"mean_blob_inside={inside:.4f} mean_blob_outside={outside:.4f}"
        )

    fig.suptitle(
        "Soft blob map smoke preview vs actual LIDC annotation (audit-only)",
        fontsize=13,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    info_path = os.path.splitext(output_path)[0] + ".txt"
    with open(info_path, "w") as file:
        file.write("\n".join(summary_lines) + "\n")
    return info_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--out", default="results/soft_blob_annotation_preview.png")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be positive")
    samples = _load_cache(args.cache_path)
    records = _match_samples(samples, args.samples, args.seed)
    if not records:
        raise RuntimeError("No cached samples could be matched to LIDC annotations.")
    info_path = _save_preview(records, args.out)
    print(f"Saved preview PNG -> {args.out}")
    print(f"Saved preview summary -> {info_path}")


if __name__ == "__main__":
    main()
