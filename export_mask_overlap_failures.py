"""Export paged PNG sheets for mask-vs-nodule coverage failures."""

import argparse
import csv
import math
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


def _load_failure_indices(path):
    with open(path, newline="") as file:
        rows = list(csv.DictReader(file))
    return [int(row["cache_index"]) for row in rows]


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
        raise ValueError(f"Sample has no {mask_key}")
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
    missed = contour & ~mask

    overlay[mask, 0] = 1.0
    overlay[mask, 1] *= 0.35
    overlay[mask, 2] *= 0.35

    overlay[contour, 0] = 1.0
    overlay[contour, 1] = 0.9
    overlay[contour, 2] = 0.0

    overlay[missed, 0] = 0.0
    overlay[missed, 1] = 1.0
    overlay[missed, 2] = 1.0
    return overlay


def _match_failures(samples, failure_indices, mask_key):
    wanted = [(index, samples[index]) for index in failure_indices]
    by_patient = defaultdict(list)
    for index, sample in wanted:
        by_patient[sample["patient_id"]].append((index, sample))

    records = []
    for patient_number, (patient_id, patient_samples) in enumerate(
        sorted(by_patient.items()), start=1
    ):
        print(
            f"Patient {patient_number}/{len(by_patient)}: {patient_id} "
            f"({len(patient_samples)} failures)"
        )
        lookup = _candidate_lookup_for_patient(patient_id)
        for index, sample in patient_samples:
            image_hash = _image_hash(_hashable_center_image(sample["image"]))
            key = (int(sample["label"]), image_hash)
            matches = lookup.get(key, [])
            if not matches:
                print(f"  WARNING: no raw match for cache_index={index}")
                continue
            match = matches.pop(0)
            mask = _mask_for_sample(sample, mask_key)
            contour = np.asarray(match["majority"], dtype=bool)
            overlap = int((mask & contour).sum())
            total = int(contour.sum())
            coverage = overlap / max(total, 1)
            records.append(
                {
                    "cache_index": index,
                    "patient_id": patient_id,
                    "scan_id": sample.get("scan_id", match.get("scan_id", "")),
                    "nodule_id": sample.get("nodule_id", ""),
                    "slice_idx": sample.get("slice_idx", match.get("slice_idx", "")),
                    "label": int(sample["label"]),
                    "mean_rating": float(match.get("mean_rating", float("nan"))),
                    "coverage": coverage,
                    "mask": mask,
                    "contour": contour,
                    "image": _image_for_display(sample),
                }
            )
    records.sort(key=lambda row: row["cache_index"])
    return records


def _write_pages(records, out_dir, per_page):
    os.makedirs(out_dir, exist_ok=True)
    pages = int(math.ceil(len(records) / per_page))
    for page in range(pages):
        chunk = records[page * per_page : (page + 1) * per_page]
        rows = len(chunk)
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.0))
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_index, record in enumerate(chunk):
            panels = [
                ("CT", record["image"], "gray"),
                ("Mask", record["mask"], "gray"),
                ("Cancer contour", record["contour"], "gray"),
                ("Overlay", _overlay(record["image"], record["mask"], record["contour"]), None),
            ]
            for col_index, (title, image, cmap) in enumerate(panels):
                ax = axes[row_index, col_index]
                ax.imshow(image, cmap=cmap)
                ax.axis("off")
                if row_index == 0:
                    ax.set_title(title, fontsize=10)
                if col_index == 0:
                    ax.text(
                        0.0,
                        -0.08,
                        (
                            f"idx {record['cache_index']} {record['patient_id']} "
                            f"slice {record['slice_idx']} cov {record['coverage']:.3f}"
                        ),
                        transform=ax.transAxes,
                        fontsize=8,
                        va="top",
                    )

        fig.suptitle(
            "HU Mask Failures: red=mask, yellow=cancer, cyan=missed cancer",
            fontsize=12,
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        filename = f"hu_mask_failures_page_{page + 1:02d}_of_{pages:02d}.png"
        fig.savefig(os.path.join(out_dir, filename), dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="cache/cache_hu_2d_filled.pkl")
    parser.add_argument(
        "--failures-csv",
        default=(
            "results/mask_nodule_overlap_hu_2d_filled/"
            "mask_nodule_overlap_failures.csv"
        ),
    )
    parser.add_argument("--mask-key", choices=["mask", "ts_mask"], default="mask")
    parser.add_argument("--out-dir", default="results/hu_mask_before_after/failures")
    parser.add_argument("--per-page", type=int, default=10)
    args = parser.parse_args()

    if args.per_page < 1:
        parser.error("--per-page must be at least 1")

    samples = _load_cache(args.cache_path)
    failure_indices = _load_failure_indices(args.failures_csv)
    records = _match_failures(samples, failure_indices, args.mask_key)
    _write_pages(records, args.out_dir, args.per_page)
    print(f"Exported {len(records)} failures to {args.out_dir}")


if __name__ == "__main__":
    main()
