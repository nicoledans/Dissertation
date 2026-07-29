"""Evaluate the possible-nodule-region mask over a full cache.

The candidate mask uses only cached CT image, HU mask, and TS mask. LIDC
contours are used only for audit metrics.
"""

import argparse
import csv
import os
import pickle
from collections import defaultdict

import numpy as np
import pylidc as pl
from scipy.ndimage import binary_dilation

from lidc_matching import _eligible_candidates, _image_hash
from build_soft_blob_cache import _hashable_center_image
from preview_possible_nodule_mask import (
    _candidate_source_sample,
    _load_candidate_ts_lookup,
    _make_prior,
    _mask,
    _possible_nodule_mask,
)


def _candidate_lookup_for_patient(patient_id):
    candidates = []
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    for scan in scans:
        candidates.extend(_eligible_candidates(scan))
    by_key = defaultdict(list)
    for candidate in candidates:
        by_key[(candidate["label"], candidate["hash"])].append(candidate)
    return by_key


def _contour_for_sample(sample, lookup):
    image_hash = _image_hash(_hashable_center_image(sample["image"]))
    key = (int(sample["label"]), image_hash)
    matches = lookup.get(key, [])
    if not matches:
        return None, None
    match = matches.pop(0)
    return np.asarray(match["majority"], dtype=bool), match


def _percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="cache/cache_ts_filled_dil1.pkl")
    parser.add_argument("--candidate-ts-cache", default=None)
    parser.add_argument("--out-dir", default="results/possible_nodule_mask_eval")
    parser.add_argument("--prior-mode", choices=["standard", "component-weighted"], default="standard")
    parser.add_argument("--hu-threshold", type=float, default=-600.0)
    parser.add_argument("--dual-hu-dense", action="store_true")
    parser.add_argument("--ground-glass-min-hu", type=float, default=-750.0)
    parser.add_argument("--ground-glass-max-hu", type=float, default=-300.0)
    parser.add_argument("--solid-min-hu", type=float, default=-300.0)
    parser.add_argument("--solid-max-hu", type=float, default=100.0)
    parser.add_argument("--ggo-extended", action="store_true")
    parser.add_argument("--ggo-min-hu", type=float, default=-800.0)
    parser.add_argument("--ggo-max-hu", type=float, default=-600.0)
    parser.add_argument("--ggo-min-area", type=int, default=5)
    parser.add_argument("--ggo-max-area", type=int, default=100)
    parser.add_argument("--ggo-max-elongation", type=float, default=1.5)
    parser.add_argument("--boundary-radius", type=int, default=3)
    parser.add_argument("--no-boundary-band", action="store_true")
    parser.add_argument("--inner-erosion-radius", type=int, default=1)
    parser.add_argument("--neighbourhood-radius", type=int, default=4)
    parser.add_argument("--max-ts-hole-area", type=int, default=0)
    parser.add_argument("--restrict-to-ts-or-holes", action="store_true")
    parser.add_argument("--exclude-ts-edge-radius", type=int, default=0)
    parser.add_argument("--candidate-lung-erosion-radius", type=int, default=0)
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=3)
    parser.add_argument("--max-area", type=int, default=350)
    parser.add_argument("--max-elongation", type=float, default=4.0)
    parser.add_argument("--max-internal-area", type=int, default=160)
    parser.add_argument("--max-internal-elongation", type=float, default=3.0)
    parser.add_argument("--blur-sigma", type=float, default=4.0)
    parser.add_argument("--prior-dilation-iterations", type=int, default=0)
    parser.add_argument(
        "--candidate-dilation-iterations",
        type=int,
        default=0,
        help="Dilate the binary candidate mask before hit/coverage metrics.",
    )
    parser.add_argument("--added-ts-border-downweight", type=float, default=1.0)
    parser.add_argument("--component-area-scale", type=float, default=50.0)
    parser.add_argument("--solid-bonus-scale", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.cache_path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{args.cache_path} is not a completed cache list.")
    candidate_ts_lookup = _load_candidate_ts_lookup(args.candidate_ts_cache)

    by_patient = defaultdict(list)
    for index, sample in enumerate(samples):
        by_patient[sample["patient_id"]].append((index, sample))

    rows = []
    missing = []
    for patient_number, (patient_id, patient_samples) in enumerate(
        sorted(by_patient.items()), start=1
    ):
        print(
            f"Patient {patient_number}/{len(by_patient)}: {patient_id} "
            f"({len(patient_samples)} samples)"
        )
        lookup = _candidate_lookup_for_patient(patient_id)
        for index, sample in patient_samples:
            contour, match = _contour_for_sample(sample, lookup)
            if contour is None:
                missing.append((index, patient_id, "no_contour_match"))
                continue
            candidate_sample = _candidate_source_sample(sample, candidate_ts_lookup)
            candidate, _parts = _possible_nodule_mask(candidate_sample, args)
            if args.candidate_dilation_iterations > 0:
                candidate = binary_dilation(
                    candidate,
                    iterations=int(args.candidate_dilation_iterations),
                )
            soft_prior = _make_prior(
                candidate,
                _parts,
                args,
                _mask(sample, "ts_mask"),
                _mask(candidate_sample, "ts_mask"),
            )
            overlap = int((candidate & contour).sum())
            nodule_pixels = int(contour.sum())
            candidate_pixels = int(candidate.sum())
            image_pixels = int(candidate.size)
            rows.append(
                {
                    "cache_index": index,
                    "patient_id": patient_id,
                    "slice_idx": sample.get("slice_idx", ""),
                    "label": int(sample["label"]),
                    "mean_rating": match.get("mean_rating", ""),
                    "reader_count": match.get("reader_count", ""),
                    "hit": overlap > 0,
                    "coverage": overlap / max(nodule_pixels, 1),
                    "overlap_pixels": overlap,
                    "missed_pixels": nodule_pixels - overlap,
                    "nodule_pixels": nodule_pixels,
                    "candidate_pixels": candidate_pixels,
                    "candidate_area_pct": candidate_pixels / max(image_pixels, 1) * 100.0,
                    "soft_prior_mean_in_contour": float(soft_prior[contour].mean()) if nodule_pixels else 0.0,
                    "soft_prior_max_in_contour": float(soft_prior[contour].max()) if nodule_pixels else 0.0,
                    "soft_prior_mean_image": float(soft_prior.mean()),
                }
            )

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "possible_nodule_mask_eval.csv")
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    hits = [row["hit"] for row in rows]
    coverages = [float(row["coverage"]) for row in rows]
    areas = [float(row["candidate_area_pct"]) for row in rows]
    missed = [int(row["missed_pixels"]) for row in rows]
    soft_mean_contour = [float(row["soft_prior_mean_in_contour"]) for row in rows]
    soft_max_contour = [float(row["soft_prior_max_in_contour"]) for row in rows]
    full = sum(c >= 0.999999 for c in coverages)

    lines = [
        "Possible nodule mask evaluation",
        f"Cache: {args.cache_path}",
        f"Candidate TS cache: {args.candidate_ts_cache}",
        f"Samples evaluated: {len(rows)}",
        f"Missing/unmatched contours: {len(missing)}",
        "",
        "Parameters:",
        f"  prior mode: {args.prior_mode}",
        f"  HU threshold: {args.hu_threshold}",
        f"  dual HU dense: {args.dual_hu_dense}",
        f"  ground-glass HU band: {args.ground_glass_min_hu} to {args.ground_glass_max_hu}",
        f"  solid HU band: {args.solid_min_hu} to {args.solid_max_hu}",
        f"  GGO extended: {args.ggo_extended}",
        f"  GGO HU band: {args.ggo_min_hu} to {args.ggo_max_hu}",
        f"  GGO component filter: min_area={args.ggo_min_area}, max_area={args.ggo_max_area}, max_elongation={args.ggo_max_elongation}",
        f"  boundary radius: {args.boundary_radius}",
        f"  boundary band used: {not args.no_boundary_band}",
        f"  inner erosion radius: {args.inner_erosion_radius}",
        f"  neighbourhood radius: {args.neighbourhood_radius}",
        f"  max TS hole area: {args.max_ts_hole_area}",
        f"  restrict to TS or holes: {args.restrict_to_ts_or_holes}",
        f"  exclude TS edge radius: {args.exclude_ts_edge_radius}",
        f"  candidate lung erosion radius: {args.candidate_lung_erosion_radius}",
        f"  closing radius: {args.closing_radius}",
        f"  min area: {args.min_area}",
        f"  max area: {args.max_area}",
        f"  max elongation: {args.max_elongation}",
        f"  max internal area: {args.max_internal_area}",
        f"  max internal elongation: {args.max_internal_elongation}",
        f"  blur sigma: {args.blur_sigma}",
        f"  prior dilation iterations before blur: {args.prior_dilation_iterations}",
        f"  candidate dilation iterations for binary metrics: {args.candidate_dilation_iterations}",
        f"  added TS border downweight: {args.added_ts_border_downweight}",
        f"  component area scale: {args.component_area_scale}",
        f"  solid bonus scale: {args.solid_bonus_scale}",
        "",
        "Metrics:",
        f"  Hit rate: {sum(hits)}/{len(hits)} ({sum(hits) / max(len(hits), 1) * 100:.2f}%)",
        f"  100% contour covered: {full}/{len(rows)} ({full / max(len(rows), 1) * 100:.2f}%)",
        f"  Mean contour coverage: {float(np.mean(coverages)):.4f}",
        f"  Median contour coverage: {_percentile(coverages, 50):.4f}",
        f"  p05 contour coverage: {_percentile(coverages, 5):.4f}",
        f"  Mean missed pixels: {float(np.mean(missed)):.2f}",
        f"  Mean candidate area: {float(np.mean(areas)):.2f}%",
        f"  Median candidate area: {_percentile(areas, 50):.2f}%",
        f"  p90 candidate area: {_percentile(areas, 90):.2f}%",
        f"  Mean soft-prior value inside contour: {float(np.mean(soft_mean_contour)):.4f}",
        f"  Median soft-prior value inside contour: {_percentile(soft_mean_contour, 50):.4f}",
        f"  Mean max soft-prior value inside contour: {float(np.mean(soft_max_contour)):.4f}",
        "",
        f"Detailed CSV: {csv_path}",
    ]
    summary_path = os.path.join(args.out_dir, "possible_nodule_mask_eval_summary.txt")
    with open(summary_path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
