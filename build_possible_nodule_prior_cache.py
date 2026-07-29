"""Add a possible-nodule binary mask and blurred soft prior to a cache.

The prior is generated from CT HU values, HU masks, and TS masks only. It does
not use LIDC nodule contours.
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np

from preview_possible_nodule_mask import _make_prior, _mask, _possible_nodule_mask


def _sample_key(sample):
    return (
        sample.get("patient_id"),
        sample.get("nodule_id"),
        sample.get("slice_idx"),
    )


def _load_candidate_ts_lookup(path):
    if not path:
        return None
    with open(path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{path} is not a completed cache list.")
    lookup = {}
    for sample in samples:
        key = _sample_key(sample)
        if key[0] is not None and key[1] is not None:
            lookup[key] = sample.get("ts_mask")
    return lookup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", default="cache/cache_ts_filled_dil1.pkl")
    parser.add_argument("--output-cache", default="cache/cache_ts_possible_nodule_prior.pkl")
    parser.add_argument(
        "--candidate-ts-cache",
        default=None,
        help=(
            "Optional cache whose ts_mask is used only for possible-nodule "
            "candidate generation. Output samples keep masks from --input-cache."
        ),
    )
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
    parser.add_argument(
        "--no-boundary-band",
        action="store_true",
        help="Do not include the TS lung boundary/outline band in candidate generation.",
    )
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
    parser.add_argument("--pleural-dense-blob", action="store_true")
    parser.add_argument("--pleural-edge-radius", type=int, default=5)
    parser.add_argument("--pleural-inner-erosion-radius", type=int, default=1)
    parser.add_argument("--pleural-min-area", type=int, default=5)
    parser.add_argument("--pleural-max-area", type=int, default=220)
    parser.add_argument("--pleural-max-elongation", type=float, default=2.5)
    parser.add_argument("--pleural-min-mean-hu", type=float, default=-250.0)
    parser.add_argument("--pleural-min-contrast-hu", type=float, default=80.0)
    parser.add_argument("--pleural-contrast-ring-radius", type=int, default=5)
    parser.add_argument("--pleural-core-hu-threshold", type=float, default=0.0)
    parser.add_argument("--pleural-core-dilation", type=int, default=1)
    parser.add_argument("--pleural-keep-large-rim", action="store_true")
    parser.add_argument("--pleural-preserve-after-filter", action="store_true")
    parser.add_argument("--blur-sigma", type=float, default=4.0)
    parser.add_argument(
        "--prior-dilation-iterations",
        type=int,
        default=0,
        help="Dilate the binary candidate mask this many iterations before Gaussian blur.",
    )
    parser.add_argument("--added-ts-border-downweight", type=float, default=1.0)
    parser.add_argument("--component-area-scale", type=float, default=50.0)
    parser.add_argument("--solid-bonus-scale", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.input_cache, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{args.input_cache} is not a completed cache list.")
    candidate_ts_lookup = _load_candidate_ts_lookup(args.candidate_ts_cache)

    areas = []
    soft_means = []
    candidate_ts_used = 0
    for sample in samples:
        candidate_sample = sample
        if candidate_ts_lookup is not None:
            candidate_ts = candidate_ts_lookup.get(_sample_key(sample))
            if candidate_ts is not None:
                candidate_sample = dict(sample)
                candidate_sample["ts_mask"] = candidate_ts
                candidate_ts_used += 1
        candidate, _parts = _possible_nodule_mask(candidate_sample, args)
        soft_prior = _make_prior(
            candidate,
            _parts,
            args,
            _mask(sample, "ts_mask"),
            _mask(candidate_sample, "ts_mask"),
        )
        sample["possible_nodule_mask"] = candidate.astype(np.uint8)
        sample["possible_nodule_prior"] = soft_prior.astype(np.float32)
        sample["possible_nodule_prior_method"] = (
            "hu_ts_holes_component_weighted_prior"
            if args.prior_mode == "component-weighted"
            else (
                "hu_ts_holes_dense_blur_no_boundary"
                if args.no_boundary_band
                else "hu_ts_holes_boundary_dense_blur"
            )
        )
        sample["possible_nodule_prior_params"] = {
            "prior_mode": args.prior_mode,
            "hu_threshold": args.hu_threshold,
            "dual_hu_dense": args.dual_hu_dense,
            "ground_glass_min_hu": args.ground_glass_min_hu,
            "ground_glass_max_hu": args.ground_glass_max_hu,
            "solid_min_hu": args.solid_min_hu,
            "solid_max_hu": args.solid_max_hu,
            "ggo_extended": args.ggo_extended,
            "ggo_min_hu": args.ggo_min_hu,
            "ggo_max_hu": args.ggo_max_hu,
            "ggo_min_area": args.ggo_min_area,
            "ggo_max_area": args.ggo_max_area,
            "ggo_max_elongation": args.ggo_max_elongation,
            "boundary_radius": args.boundary_radius,
            "boundary_band_used": not args.no_boundary_band,
            "inner_erosion_radius": args.inner_erosion_radius,
            "neighbourhood_radius": args.neighbourhood_radius,
            "max_ts_hole_area": args.max_ts_hole_area,
            "restrict_to_ts_or_holes": args.restrict_to_ts_or_holes,
            "exclude_ts_edge_radius": args.exclude_ts_edge_radius,
            "candidate_lung_erosion_radius": args.candidate_lung_erosion_radius,
            "closing_radius": args.closing_radius,
            "min_area": args.min_area,
            "max_area": args.max_area,
            "max_elongation": args.max_elongation,
            "max_internal_area": args.max_internal_area,
            "max_internal_elongation": args.max_internal_elongation,
            "pleural_dense_blob": args.pleural_dense_blob,
            "pleural_edge_radius": args.pleural_edge_radius,
            "pleural_inner_erosion_radius": args.pleural_inner_erosion_radius,
            "pleural_min_area": args.pleural_min_area,
            "pleural_max_area": args.pleural_max_area,
            "pleural_max_elongation": args.pleural_max_elongation,
            "pleural_min_mean_hu": args.pleural_min_mean_hu,
            "pleural_min_contrast_hu": args.pleural_min_contrast_hu,
            "pleural_contrast_ring_radius": args.pleural_contrast_ring_radius,
            "pleural_core_hu_threshold": args.pleural_core_hu_threshold,
            "pleural_core_dilation": args.pleural_core_dilation,
            "pleural_keep_large_rim": args.pleural_keep_large_rim,
            "pleural_preserve_after_filter": args.pleural_preserve_after_filter,
            "blur_sigma": args.blur_sigma,
            "prior_dilation_iterations": args.prior_dilation_iterations,
            "added_ts_border_downweight": args.added_ts_border_downweight,
            "component_area_scale": args.component_area_scale,
            "solid_bonus_scale": args.solid_bonus_scale,
            "candidate_ts_cache": args.candidate_ts_cache,
        }
        areas.append(float(candidate.mean() * 100.0))
        soft_means.append(float(soft_prior.mean()))

    os.makedirs(os.path.dirname(args.output_cache) or ".", exist_ok=True)
    with open(args.output_cache, "wb") as file:
        pickle.dump(samples, file, protocol=pickle.HIGHEST_PROTOCOL)

    info_path = os.path.splitext(args.output_cache)[0] + "_info.txt"
    lines = [
        "Possible nodule prior cache",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Input cache: {args.input_cache}",
        f"Output cache: {args.output_cache}",
        f"Candidate TS cache: {args.candidate_ts_cache}",
        f"Candidate TS masks replaced: {candidate_ts_used}/{len(samples)}",
        "Contours used: no",
        f"Samples: {len(samples)}",
        f"Mean binary candidate area: {float(np.mean(areas)):.2f}%",
        f"Mean soft prior value over image: {float(np.mean(soft_means)):.4f}",
        f"Prior mode: {args.prior_mode}",
        f"Dual HU dense: {args.dual_hu_dense}",
        f"GGO extended: {args.ggo_extended}",
        f"GGO HU band: {args.ggo_min_hu} to {args.ggo_max_hu}",
        f"GGO component filter: min_area={args.ggo_min_area}, max_area={args.ggo_max_area}, max_elongation={args.ggo_max_elongation}",
        f"Solid HU band for GGO mode: {args.hu_threshold} to {args.solid_max_hu}",
        f"Blur sigma: {args.blur_sigma}",
        f"Prior dilation iterations before blur: {args.prior_dilation_iterations}",
        f"Added TS border downweight: {args.added_ts_border_downweight}",
        f"Component area scale: {args.component_area_scale}",
        f"Solid bonus scale: {args.solid_bonus_scale}",
        f"Pleural dense blob enabled: {args.pleural_dense_blob}",
        f"Pleural dense blob max area: {args.pleural_max_area}",
        f"Pleural dense blob max elongation: {args.pleural_max_elongation}",
        f"Pleural dense blob min contrast HU: {args.pleural_min_contrast_hu}",
        f"Pleural dense blob core HU threshold: {args.pleural_core_hu_threshold}",
        f"Pleural dense blob core dilation: {args.pleural_core_dilation}",
        f"Pleural dense blob keep large rim: {args.pleural_keep_large_rim}",
        f"Pleural dense blob preserved after filter: {args.pleural_preserve_after_filter}",
        f"Boundary band used: {not args.no_boundary_band}",
        f"Max TS hole area: {args.max_ts_hole_area}",
        f"Restrict to TS or holes: {args.restrict_to_ts_or_holes}",
        f"Exclude TS edge radius: {args.exclude_ts_edge_radius}",
        f"Candidate lung erosion radius: {args.candidate_lung_erosion_radius}",
    ]
    with open(info_path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Info: {info_path}")


if __name__ == "__main__":
    main()
