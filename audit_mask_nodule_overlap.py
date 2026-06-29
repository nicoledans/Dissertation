"""Audit cache lung masks against LIDC radiologist nodule contours."""

import argparse
import csv
import os
import pickle
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl

from build_soft_blob_cache import _hashable_center_image
from lidc_matching import _eligible_candidates, _image_hash


def _load_cache(path):
    with open(path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{path} is not a completed cache list.")
    return samples


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


def _mask_for_sample(sample, mask_key):
    mask = sample.get(mask_key)
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[0] == 3:
        mask = mask[1]
    return mask.astype(bool)


def _image_for_display(sample):
    image = np.asarray(sample["image"], dtype=np.float32)
    image = _hashable_center_image(image)
    return np.clip(image, 0.0, 1.0)


def _overlay(image, mask, contour):
    base = np.stack([image, image, image], axis=-1)
    overlay = base.copy()
    mask = mask.astype(bool)
    contour = contour.astype(bool)
    overlay[mask, 0] = 1.0
    overlay[mask, 1] *= 0.35
    overlay[mask, 2] *= 0.35
    overlay[contour, 0] = 1.0
    overlay[contour, 1] = 0.9
    overlay[contour, 2] = 0.0
    return overlay


def _audit(args):
    samples = _load_cache(args.cache_path)
    indexed = list(enumerate(samples))
    if args.limit_samples:
        indexed = indexed[: args.limit_samples]

    by_patient = defaultdict(list)
    for index, sample in indexed:
        by_patient[sample["patient_id"]].append((index, sample))

    rows = []
    missing = []
    for patient_number, (patient_id, patient_samples) in enumerate(
        sorted(by_patient.items()), start=1
    ):
        print(
            f"Patient {patient_number}/{len(by_patient)}: {patient_id} "
            f"({len(patient_samples)} cached samples)"
        )
        lookup = _candidate_lookup_for_patient(patient_id)
        for cache_index, sample in patient_samples:
            mask = _mask_for_sample(sample, args.mask_key)
            if mask is None:
                missing.append((cache_index, patient_id, f"missing_{args.mask_key}"))
                continue
            image_hash = _image_hash(_hashable_center_image(sample["image"]))
            key = (int(sample["label"]), image_hash)
            matches = lookup.get(key, [])
            if not matches:
                missing.append((cache_index, patient_id, "no_raw_match"))
                continue
            match = matches.pop(0)
            contour = np.asarray(match["majority"], dtype=bool)
            if contour.shape != mask.shape:
                missing.append((cache_index, patient_id, "shape_mismatch"))
                continue
            contour_pixels = int(contour.sum())
            overlap_pixels = int((mask & contour).sum())
            coverage = overlap_pixels / max(contour_pixels, 1)
            rows.append(
                {
                    "cache_index": cache_index,
                    "patient_id": patient_id,
                    "scan_id": sample.get("scan_id", match.get("scan_id", "")),
                    "nodule_id": sample.get("nodule_id", ""),
                    "group_index": match.get("group_index", ""),
                    "slice_idx": sample.get("slice_idx", match.get("slice_idx", "")),
                    "label": int(sample["label"]),
                    "mean_rating": match.get("mean_rating", ""),
                    "reader_count": match.get("reader_count", ""),
                    "mask_key": args.mask_key,
                    "mask_pixels": int(mask.sum()),
                    "nodule_pixels": contour_pixels,
                    "overlap_pixels": overlap_pixels,
                    "coverage": coverage,
                    "is_100pct": coverage >= args.full_threshold,
                    "image": _image_for_display(sample),
                    "mask": mask,
                    "contour": contour,
                }
            )
    return rows, missing, len(samples)


def _write_outputs(rows, missing, total_samples, args):
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "mask_nodule_overlap.csv")
    summary_path = os.path.join(args.out_dir, "mask_nodule_overlap_summary.txt")
    failures_path = os.path.join(args.out_dir, "mask_nodule_overlap_failures.csv")

    public_fields = [
        "cache_index",
        "patient_id",
        "scan_id",
        "nodule_id",
        "group_index",
        "slice_idx",
        "label",
        "mean_rating",
        "reader_count",
        "mask_key",
        "mask_pixels",
        "nodule_pixels",
        "overlap_pixels",
        "coverage",
        "is_100pct",
    ]
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=public_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in public_fields})

    failures = [row for row in rows if not row["is_100pct"]]
    with open(failures_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=public_fields)
        writer.writeheader()
        for row in failures:
            writer.writerow({field: row[field] for field in public_fields})

    coverages = np.asarray([row["coverage"] for row in rows], dtype=np.float32)
    full_count = int(sum(row["is_100pct"] for row in rows))
    lines = [
        "Mask vs LIDC radiologist-majority nodule contour audit",
        f"Cache: {args.cache_path}",
        f"Mask key: {args.mask_key}",
        f"Total cache samples: {total_samples}",
        f"Matched/audited samples: {len(rows)}",
        f"Missing/unmatched samples: {len(missing)}",
        f"100% covered: {full_count}",
        f"Not 100% covered: {len(failures)}",
    ]
    if len(rows):
        lines.extend(
            [
                f"Coverage mean: {float(coverages.mean()):.4f}",
                f"Coverage min: {float(coverages.min()):.4f}",
                f"Coverage p05: {float(np.percentile(coverages, 5)):.4f}",
                f"Coverage p50: {float(np.percentile(coverages, 50)):.4f}",
            ]
        )
    lines.extend(
        [
            "",
            "Failure CSV contains the exact cache indices/patient IDs where coverage is not 100%.",
            f"Full results CSV: {csv_path}",
            f"Failures CSV: {failures_path}",
        ]
    )
    if missing:
        lines.append("")
        lines.append("Missing/unmatched:")
        for cache_index, patient_id, reason in missing[:100]:
            lines.append(f"cache_index={cache_index} patient={patient_id} reason={reason}")
        if len(missing) > 100:
            lines.append(f"... {len(missing) - 100} more omitted")

    with open(summary_path, "w") as file:
        file.write("\n".join(lines) + "\n")

    if args.max_failure_images and failures:
        _write_failure_grid(failures[: args.max_failure_images], args.out_dir)

    print("\n".join(lines))


def _write_failure_grid(failures, out_dir):
    cols = 3
    rows = len(failures)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_index, row in enumerate(failures):
        panels = [
            ("CT", row["image"], "gray"),
            ("Mask + nodule", _overlay(row["image"], row["mask"], row["contour"]), None),
            ("Nodule contour", row["contour"], "gray"),
        ]
        for col_index, (title, image, cmap) in enumerate(panels):
            ax = axes[row_index, col_index]
            ax.imshow(image, cmap=cmap)
            ax.axis("off")
            if row_index == 0:
                ax.set_title(title)
            if col_index == 0:
                ax.text(
                    0.0,
                    -0.08,
                    (
                        f"idx {row['cache_index']} {row['patient_id']} "
                        f"cov {row['coverage']:.3f}"
                    ),
                    transform=ax.transAxes,
                    fontsize=8,
                    va="top",
                )
    fig.tight_layout()
    path = os.path.join(out_dir, "mask_nodule_overlap_failures.png")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--mask-key", choices=["mask", "ts_mask"], default="mask")
    parser.add_argument("--out-dir", default="results/mask_nodule_overlap_audit")
    parser.add_argument(
        "--full-threshold",
        type=float,
        default=0.999999,
        help="Coverage threshold counted as 100 percent.",
    )
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--max-failure-images", type=int, default=30)
    args = parser.parse_args()

    if args.limit_samples < 0:
        parser.error("--limit-samples must be non-negative")
    if args.max_failure_images < 0:
        parser.error("--max-failure-images must be non-negative")
    if not 0 < args.full_threshold <= 1:
        parser.error("--full-threshold must be in (0, 1]")

    rows, missing, total_samples = _audit(args)
    _write_outputs(rows, missing, total_samples, args)


if __name__ == "__main__":
    main()
