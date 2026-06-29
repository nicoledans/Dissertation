"""Attach continuous soft blob maps to an existing LIDC cache.

The maps are generated from raw HU slices plus a lung/search mask.  LIDC
contours are only used for optional audit matching metadata in downstream
scripts, never to create the maps.
"""

import argparse
import os
import pickle
from collections import defaultdict

import numpy as np

if not hasattr(np, "int"):
    np.int = int

import pylidc as pl
from scipy.ndimage import zoom

from config import TRAIN_CACHE_PATH
from lidc_matching import _eligible_candidates, _image_hash
from soft_blob_generator import soft_blob_maps


def _resize_mask_to_raw(mask_224, raw_shape):
    factors = (raw_shape[0] / mask_224.shape[0], raw_shape[1] / mask_224.shape[1])
    return zoom(mask_224.astype(np.uint8), factors, order=0).astype(bool)


def _resize_map_to_224(map_raw):
    factors = (224 / map_raw.shape[0], 224 / map_raw.shape[1])
    resized = zoom(map_raw.astype(np.float32), factors, order=1)
    return np.clip(resized, 0.0, 1.0).astype(np.float32)


def _resize_binary_to_224(mask_raw):
    factors = (224 / mask_raw.shape[0], 224 / mask_raw.shape[1])
    resized = zoom(mask_raw.astype(np.uint8), factors, order=0)
    return (resized > 0).astype(np.uint8)


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


def _stats(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(values.mean()),
        "p95": float(np.percentile(values, 95.0)),
        "nonzero_pct": float((values > 1e-4).mean() * 100.0),
        "high_pct": float((values >= 0.50).mean() * 100.0),
    }


def build(args):
    samples = _load_cache(args.cache_path)
    by_patient = defaultdict(list)
    for index, sample in enumerate(samples):
        by_patient[sample["patient_id"]].append((index, sample))

    output_samples = [None] * len(samples)
    rows = []
    missing = []
    for patient_number, (patient_id, patient_samples) in enumerate(sorted(by_patient.items()), start=1):
        if args.max_patients and patient_number > args.max_patients:
            for index, _sample in patient_samples:
                missing.append((index, patient_id, "max_patients_stop"))
            continue
        print(
            f"Patient {patient_number}/{len(by_patient)}: {patient_id} "
            f"({len(patient_samples)} cached samples)"
        )
        lookup = _candidate_lookup_for_patient(patient_id)
        for index, sample in patient_samples:
            mask = sample.get(args.mask_key)
            if mask is None:
                missing.append((index, patient_id, f"missing_{args.mask_key}"))
                continue
            image_hash = _image_hash(_hashable_center_image(sample["image"]))
            key = (int(sample["label"]), image_hash)
            matches = lookup.get(key, [])
            if not matches:
                missing.append((index, patient_id, "no_raw_match"))
                continue
            match = matches.pop(0)
            raw_slice = np.asarray(match["raw_slice"], dtype=np.float32)
            lung_raw = _resize_mask_to_raw(np.asarray(mask).astype(bool), raw_slice.shape)
            result = soft_blob_maps(
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

            new_sample = dict(sample)
            for key_name in (
                "soft_blob_map",
                "solid_like_map",
                "subsolid_like_map",
                "vesselness_map",
            ):
                new_sample[key_name] = _resize_map_to_224(result[key_name])
            new_sample["search_mask"] = _resize_binary_to_224(result["search_mask"])
            new_sample["soft_blob_method"] = f"soft_blob_{args.mask_key}"
            new_sample["soft_blob_params"] = result["params"]
            output_samples[index] = new_sample

            blob_stats = _stats(new_sample["soft_blob_map"])
            rows.append(
                {
                    "index": index,
                    "patient_id": patient_id,
                    "label": int(sample["label"]),
                    "mask_key": args.mask_key,
                    "slice_idx": int(match["slice_idx"]),
                    "mean_rating": float(match["mean_rating"]),
                    "pixel_spacing_mm": float(match["pixel_spacing_mm"]),
                    "blob_mean": blob_stats["mean"],
                    "blob_p95": blob_stats["p95"],
                    "blob_nonzero_pct": blob_stats["nonzero_pct"],
                    "blob_high_pct": blob_stats["high_pct"],
                }
            )

    if missing and not args.allow_missing:
        raise RuntimeError(
            f"Could not build soft maps for {len(missing)} samples. "
            "Use --allow-missing to keep those samples with zero maps."
        )
    for index, sample in enumerate(samples):
        if output_samples[index] is not None:
            continue
        new_sample = dict(sample)
        reference_mask = sample.get(args.mask_key, sample.get("mask"))
        zeros = np.zeros_like(reference_mask, dtype=np.float32)
        for key_name in (
            "soft_blob_map",
            "solid_like_map",
            "subsolid_like_map",
            "vesselness_map",
            "search_mask",
        ):
            new_sample[key_name] = zeros.copy()
        new_sample["soft_blob_method"] = "missing_raw_match_zero_map"
        new_sample["soft_blob_params"] = {}
        output_samples[index] = new_sample

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "wb") as file:
        pickle.dump(output_samples, file)

    info_path = os.path.splitext(args.out)[0] + "_info.txt"
    with open(info_path, "w") as file:
        file.write("Soft blob map cache\n")
        file.write(f"Source cache: {args.cache_path}\n")
        file.write(f"Output cache: {args.out}\n")
        file.write(f"Mask key used for lung restriction: {args.mask_key}\n")
        file.write("Contours are not used for map generation.\n")
        file.write(f"Generated maps: {len(rows)} / {len(samples)}\n")
        file.write(f"Missing maps: {len(missing)}\n")
        file.write(f"Lung dilation: {args.lung_dilation_px} px at native resolution\n")
        file.write(f"Diameter scale: {args.min_diameter_mm:g}-{args.max_diameter_mm:g} mm\n")
        file.write(f"Vessel suppression cap: {args.vessel_suppression:g}\n")
        file.write(f"Gaussian blur sigma: {args.blur_sigma_px:g} px\n")
        file.write(f"2.5D persistence weight: {args.persistence_weight:g}\n")
        if rows:
            file.write(
                f"Mean soft map value: {np.mean([row['blob_mean'] for row in rows]):.4f}\n"
            )
            file.write(
                f"Mean non-zero soft map area: {np.mean([row['blob_nonzero_pct'] for row in rows]):.2f}%\n"
            )
            file.write(
                f"Mean high-response area >=0.50: {np.mean([row['blob_high_pct'] for row in rows]):.2f}%\n"
            )
        file.write("\nMissing samples:\n")
        for index, patient_id, reason in missing:
            file.write(f"index={index} patient={patient_id} reason={reason}\n")

    csv_path = os.path.splitext(args.out)[0] + "_summary.csv"
    with open(csv_path, "w") as file:
        fields = [
            "index",
            "patient_id",
            "label",
            "mask_key",
            "slice_idx",
            "mean_rating",
            "pixel_spacing_mm",
            "blob_mean",
            "blob_p95",
            "blob_nonzero_pct",
            "blob_high_pct",
        ]
        file.write(",".join(fields) + "\n")
        for row in rows:
            file.write(",".join(str(row[field]) for field in fields) + "\n")

    print(f"Saved soft blob cache -> {args.out}")
    print(f"Saved info -> {info_path}")
    print(f"Saved summary -> {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--out", default="cache/cache_hu_soft_blobs.pkl")
    parser.add_argument("--mask-key", choices=["mask", "ts_mask"], default="mask")
    parser.add_argument("--lung-dilation-px", type=int, default=3)
    parser.add_argument("--min-diameter-mm", type=float, default=3.0)
    parser.add_argument("--max-diameter-mm", type=float, default=30.0)
    parser.add_argument("--vessel-suppression", type=float, default=0.30)
    parser.add_argument("--blur-sigma-px", type=float, default=2.0)
    parser.add_argument("--persistence-weight", type=float, default=0.25)
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    if args.lung_dilation_px < 0:
        parser.error("--lung-dilation-px must be non-negative")
    for name in (
        "min_diameter_mm",
        "max_diameter_mm",
        "vessel_suppression",
        "blur_sigma_px",
        "persistence_weight",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.max_diameter_mm < args.min_diameter_mm:
        parser.error("--max-diameter-mm must be >= --min-diameter-mm")
    build(args)


if __name__ == "__main__":
    main()
