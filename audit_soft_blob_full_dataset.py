"""Streaming full-dataset audit of soft blob maps against LIDC contours.

This avoids writing a full soft-blob cache.  For each cached sample it matches
the reconstructed raw LIDC slice, creates the image-derived soft blob map, and
compares it with the majority-vote contour for audit-only metrics.
"""

import argparse
import csv
import os
import pickle
from collections import defaultdict

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
    return np.clip(zoom(map_raw.astype(np.float32), factors, order=1), 0.0, 1.0)


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


def _center_disk(mask, radius_px):
    coords = np.argwhere(mask)
    center = coords.mean(axis=0)
    rr, cc = np.ogrid[:mask.shape[0], :mask.shape[1]]
    return (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= float(radius_px) ** 2


def _safe_mean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if values.size else float("nan")


def _safe_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else float("nan")


def _audit_row(sample_index, sample, match, args):
    raw_slice = np.asarray(match["raw_slice"], dtype=np.float32)
    lung_raw = _resize_mask_to_raw(np.asarray(sample[args.mask_key]).astype(bool), raw_slice.shape)
    maps = soft_blob_maps(
        raw_slice,
        lung_raw,
        float(match["pixel_spacing_mm"]),
        adjacent_slices=match.get("adjacent_raw_slices", []),
        lung_dilation_px=args.lung_dilation_px,
        min_diameter_mm=args.min_diameter_mm,
        max_diameter_mm=args.max_diameter_mm,
        vessel_suppression=args.vessel_suppression,
        blur_sigma_px=args.blur_sigma_px,
        persistence_weight=args.persistence_weight,
    )
    blob = _resize_map_to_224(maps["soft_blob_map"])
    majority = np.asarray(match["majority"]).astype(bool)
    lung = np.asarray(sample[args.mask_key]).astype(bool)
    outside = lung & ~majority
    lung_values = blob[lung] if np.any(lung) else blob.ravel()
    threshold = float(np.percentile(lung_values, args.top_percentile)) if lung_values.size else 1.0
    nonzero = blob > args.nonzero_threshold
    high = blob >= threshold
    center = _center_disk(majority, args.center_radius_px)
    contour_area = int(majority.sum())
    return {
        "index": sample_index,
        "patient_id": match["patient_id"],
        "scan_id": match["scan_id"],
        "group_index": match["group_index"],
        "slice_idx": match["slice_idx"],
        "label": match["label"],
        "mean_rating": match["mean_rating"],
        "reader_count": match["reader_count"],
        "contour_area_px": contour_area,
        "blob_nonzero_area_pct": float(nonzero.mean() * 100.0),
        "blob_high_area_pct": float(high.mean() * 100.0),
        "contour_touched_nonzero_pct": float((nonzero & majority).sum() / max(contour_area, 1) * 100.0),
        "contour_touched_top_pct": float((high & majority).sum() / max(contour_area, 1) * 100.0),
        "center_inside_high_response": bool(np.any(center & high)),
        "mean_blob_inside_contour": float(blob[majority].mean()) if contour_area else float("nan"),
        "mean_blob_outside_contour": float(blob[outside].mean()) if np.any(outside) else float("nan"),
        "inside_minus_outside": (
            float(blob[majority].mean() - blob[outside].mean())
            if contour_area and np.any(outside)
            else float("nan")
        ),
        "high_response_threshold": threshold,
    }


def run(args):
    samples = _load_cache(args.cache_path)
    if args.limit_samples:
        samples = samples[: args.limit_samples]
    by_patient = defaultdict(list)
    for index, sample in enumerate(samples):
        if sample.get(args.mask_key) is not None:
            by_patient[sample["patient_id"]].append((index, sample))

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "soft_blob_full_dataset_audit.csv")
    rows = []
    missing = []
    fields = None
    with open(csv_path, "w", newline="") as file:
        writer = None
        for patient_number, (patient_id, patient_samples) in enumerate(sorted(by_patient.items()), start=1):
            print(
                f"Patient {patient_number}/{len(by_patient)}: {patient_id} "
                f"matched={len(rows)} missing={len(missing)}"
            )
            lookup = _candidate_lookup_for_patient(patient_id)
            for index, sample in patient_samples:
                key = (int(sample["label"]), _image_hash(_hashable_center_image(sample["image"])))
                matches = lookup.get(key, [])
                if not matches:
                    missing.append((index, patient_id, "no_raw_match"))
                    continue
                match = matches.pop(0)
                row = _audit_row(index, sample, match, args)
                rows.append(row)
                if writer is None:
                    fields = list(row.keys())
                    writer = csv.DictWriter(file, fieldnames=fields)
                    writer.writeheader()
                writer.writerow(row)

    summary_path = os.path.join(args.out_dir, "soft_blob_full_dataset_audit_summary.txt")
    unique_scans = {row["scan_id"] for row in rows}
    unique_patients = {row["patient_id"] for row in rows}
    centre_hits = sum(row["center_inside_high_response"] for row in rows)
    lines = [
        "=== FULL DATASET SOFT BLOB VS LIDC CONTOUR AUDIT ===",
        "Contours are audit-only and were not used to generate maps.",
        f"Cache: {args.cache_path}",
        f"Mask key: {args.mask_key}",
        f"Matched nodule-slice samples: {len(rows)} / {len(samples)}",
        f"Missing samples: {len(missing)}",
        f"Unique CT scans matched: {len(unique_scans)}",
        f"Unique patients matched: {len(unique_patients)}",
        "",
        f"Mean contour area touched by non-zero blob response: {_safe_mean([r['contour_touched_nonzero_pct'] for r in rows]):.2f}%",
        f"Median contour area touched by non-zero blob response: {_safe_median([r['contour_touched_nonzero_pct'] for r in rows]):.2f}%",
        f"Mean contour area touched by top-{100 - args.top_percentile:.0f}% blob response: {_safe_mean([r['contour_touched_top_pct'] for r in rows]):.2f}%",
        f"Median contour area touched by top-{100 - args.top_percentile:.0f}% blob response: {_safe_median([r['contour_touched_top_pct'] for r in rows]):.2f}%",
        f"Centre inclusion in high response: {centre_hits}/{len(rows)} ({centre_hits / max(len(rows), 1) * 100:.2f}%)",
        f"Mean blob value inside contour: {_safe_mean([r['mean_blob_inside_contour'] for r in rows]):.4f}",
        f"Mean blob value outside contour: {_safe_mean([r['mean_blob_outside_contour'] for r in rows]):.4f}",
        f"Mean inside-minus-outside blob value: {_safe_mean([r['inside_minus_outside'] for r in rows]):.4f}",
        "",
        "CSV fields include per-sample overlap, centre inclusion, inside/outside blob values, patient_id and scan_id.",
    ]
    if missing:
        missing_path = os.path.join(args.out_dir, "soft_blob_full_dataset_missing.txt")
        with open(missing_path, "w") as file:
            for index, patient_id, reason in missing:
                file.write(f"index={index} patient={patient_id} reason={reason}\n")
        lines.append(f"Missing sample list: {missing_path}")

    with open(summary_path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\nSaved CSV -> {csv_path}")
    print(f"Saved summary -> {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--out-dir", default="results/soft_blob_full_dataset_audit")
    parser.add_argument("--mask-key", choices=["mask", "ts_mask"], default="mask")
    parser.add_argument("--top-percentile", type=float, default=90.0)
    parser.add_argument("--nonzero-threshold", type=float, default=1e-4)
    parser.add_argument("--center-radius-px", type=float, default=4.0)
    parser.add_argument("--lung-dilation-px", type=int, default=3)
    parser.add_argument("--min-diameter-mm", type=float, default=3.0)
    parser.add_argument("--max-diameter-mm", type=float, default=30.0)
    parser.add_argument("--vessel-suppression", type=float, default=0.30)
    parser.add_argument("--blur-sigma-px", type=float, default=2.0)
    parser.add_argument("--persistence-weight", type=float, default=0.25)
    parser.add_argument("--limit-samples", type=int, default=0)
    args = parser.parse_args()
    if not (0.0 < args.top_percentile < 100.0):
        parser.error("--top-percentile must be between 0 and 100")
    if args.center_radius_px <= 0:
        parser.error("--center-radius-px must be positive")
    run(args)


if __name__ == "__main__":
    main()
