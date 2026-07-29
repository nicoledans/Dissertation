"""Prototype a possible-nodule-region mask and audit it on a small sample.

The candidate mask uses only cached CT image, HU mask, and TS mask.  LIDC
contours are used only for audit hit/coverage statistics and visualization.
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
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    label,
)

from build_soft_blob_cache import _hashable_center_image
from lidc_matching import _eligible_candidates, _image_hash


def _disk(radius):
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _center_image(sample):
    image = np.asarray(sample["image"], dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image[1]
    return np.clip(_hashable_center_image(image), 0.0, 1.0)


def _image_to_hu(image):
    return np.asarray(image, dtype=np.float32) * 1400.0 - 1000.0


def _mask(sample, key):
    value = np.asarray(sample[key])
    if value.ndim == 3 and value.shape[0] == 3:
        value = value[1]
    return value.astype(bool)


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
        return None
    return np.asarray(matches.pop(0)["majority"], dtype=bool)


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


def _candidate_source_sample(sample, candidate_ts_lookup):
    if candidate_ts_lookup is None:
        return sample
    candidate_ts = candidate_ts_lookup.get(_sample_key(sample))
    if candidate_ts is None:
        return sample
    candidate_sample = dict(sample)
    candidate_sample["ts_mask"] = candidate_ts
    return candidate_sample


def _downweight_added_ts_border(prior, training_ts_mask, candidate_ts_mask, factor):
    factor = float(factor)
    if factor >= 1.0:
        return prior
    added_border = training_ts_mask.astype(bool) & ~candidate_ts_mask.astype(bool)
    output = np.asarray(prior, dtype=np.float32).copy()
    output[added_border] *= max(factor, 0.0)
    max_value = float(output.max())
    if max_value > 0:
        output /= max_value + 1e-8
    return output.astype(np.float32)


def _component_filter(mask, min_area, max_area, max_elongation):
    labeled, n_labels = label(mask)
    output = np.zeros_like(mask, dtype=bool)
    for idx in range(1, n_labels + 1):
        comp = labeled == idx
        area = int(comp.sum())
        if area < min_area:
            continue
        rows, cols = np.nonzero(comp)
        height = max(int(rows.max() - rows.min() + 1), 1)
        width = max(int(cols.max() - cols.min() + 1), 1)
        elongation = max(height / width, width / height)
        if area > max_area and elongation > max_elongation:
            continue
        output |= comp
    return output


def _component_filter_strict(mask, min_area, max_area, max_elongation):
    labeled, n_labels = label(mask)
    output = np.zeros_like(mask, dtype=bool)
    for idx in range(1, n_labels + 1):
        comp = labeled == idx
        area = int(comp.sum())
        if area < min_area or area > max_area:
            continue
        rows, cols = np.nonzero(comp)
        height = max(int(rows.max() - rows.min() + 1), 1)
        width = max(int(cols.max() - cols.min() + 1), 1)
        elongation = max(height / width, width / height)
        if elongation > max_elongation:
            continue
        output |= comp
    return output


def _max_component_area_filter(mask, max_area):
    if max_area is None or max_area <= 0:
        return mask
    labeled, n_labels = label(mask)
    output = np.zeros_like(mask, dtype=bool)
    for idx in range(1, n_labels + 1):
        comp = labeled == idx
        if int(comp.sum()) <= int(max_area):
            output |= comp
    return output


def _pleural_dense_blob_mask(hu, dense, candidate_ts_mask, lung_neighbourhood, args):
    if not getattr(args, "pleural_dense_blob", False):
        return np.zeros_like(candidate_ts_mask, dtype=bool)

    outer_radius = int(getattr(args, "pleural_edge_radius", 5))
    inner_radius = int(getattr(args, "pleural_inner_erosion_radius", 1))
    edge_zone = binary_dilation(candidate_ts_mask, structure=_disk(outer_radius))
    if inner_radius > 0:
        edge_zone &= ~binary_erosion(candidate_ts_mask, structure=_disk(inner_radius))

    seed = dense & edge_zone & lung_neighbourhood
    labeled, n_labels = label(seed)
    output = np.zeros_like(seed, dtype=bool)
    min_area = int(getattr(args, "pleural_min_area", 5))
    max_area = int(getattr(args, "pleural_max_area", 220))
    max_elongation = float(getattr(args, "pleural_max_elongation", 2.5))
    min_mean_hu = float(getattr(args, "pleural_min_mean_hu", -250.0))
    min_contrast_hu = float(getattr(args, "pleural_min_contrast_hu", 80.0))
    ring_radius = int(getattr(args, "pleural_contrast_ring_radius", 5))
    keep_large_rim = bool(getattr(args, "pleural_keep_large_rim", False))
    core_hu_threshold = float(getattr(args, "pleural_core_hu_threshold", 0.0))
    core_dilation = int(getattr(args, "pleural_core_dilation", 1))

    for idx in range(1, n_labels + 1):
        comp = labeled == idx
        area = int(comp.sum())
        if area < min_area:
            continue
        is_large = area > max_area
        if is_large and not keep_large_rim:
            continue
        rows, cols = np.nonzero(comp)
        height = max(int(rows.max() - rows.min() + 1), 1)
        width = max(int(cols.max() - cols.min() + 1), 1)
        elongation = max(height / width, width / height)
        if elongation > max_elongation:
            continue

        comp_mean = float(hu[comp].mean())
        if comp_mean < min_mean_hu:
            continue

        outer = binary_dilation(comp, structure=_disk(ring_radius))
        inner = binary_dilation(comp, structure=_disk(1))
        ring = outer & ~inner & lung_neighbourhood
        if np.any(ring):
            ring_mean = float(hu[ring].mean())
            if comp_mean - ring_mean < min_contrast_hu:
                continue

        if is_large and keep_large_rim:
            core_seed = comp & (hu > core_hu_threshold)
            core_labeled, n_core_labels = label(core_seed)
            for core_idx in range(1, n_core_labels + 1):
                core = core_labeled == core_idx
                core_area = int(core.sum())
                if core_area < min_area or core_area > max_area:
                    continue
                core_rows, core_cols = np.nonzero(core)
                core_height = max(int(core_rows.max() - core_rows.min() + 1), 1)
                core_width = max(int(core_cols.max() - core_cols.min() + 1), 1)
                core_elongation = max(core_height / core_width, core_width / core_height)
                if core_elongation > max_elongation:
                    continue
                if core_dilation > 0:
                    core = binary_dilation(core, structure=_disk(core_dilation)) & comp
                output |= core
        else:
            output |= comp

    return output


def _component_weighted_prior(binary_mask, solid_mask, args):
    labeled, n_labels = label(binary_mask)
    prior = np.zeros_like(binary_mask, dtype=np.float32)
    for idx in range(1, n_labels + 1):
        comp = labeled == idx
        area = int(comp.sum())
        if area < args.min_area:
            continue
        rows, cols = np.nonzero(comp)
        height = max(int(rows.max() - rows.min() + 1), 1)
        width = max(int(cols.max() - cols.min() + 1), 1)
        elongation = max(height / width, width / height)
        area_score = 1.0 / (area / float(args.component_area_scale) + 1.0)
        shape_score = 1.0 / (elongation + 1e-8)
        compactness = float(np.clip(area_score * shape_score * 4.0, 0.0, 1.0))
        solid_overlap = float((comp & solid_mask).sum() / max(area, 1))
        solid_bonus = 1.0 + solid_overlap * float(args.solid_bonus_scale)
        component = binary_dilation(
            comp,
            iterations=int(args.prior_dilation_iterations),
        )
        component = gaussian_filter(
            component.astype(np.float32),
            sigma=float(args.blur_sigma),
        )
        peak = float(component.max())
        if peak > 0:
            component /= peak + 1e-8
        prior = np.maximum(prior, component * compactness * solid_bonus)
    max_value = float(prior.max())
    if max_value > 0:
        prior /= max_value + 1e-8
    return prior.astype(np.float32)


def _possible_nodule_mask(sample, args):
    image = _center_image(sample)
    hu = _image_to_hu(image)
    hu_mask = _mask(sample, "mask")
    ts_mask = _mask(sample, "ts_mask")
    candidate_ts_mask = ts_mask
    candidate_hu_mask = hu_mask
    candidate_lung_erosion_radius = int(getattr(args, "candidate_lung_erosion_radius", 0))
    if candidate_lung_erosion_radius > 0:
        erosion_structure = _disk(candidate_lung_erosion_radius)
        candidate_ts_mask = binary_erosion(ts_mask, structure=erosion_structure)
        candidate_hu_mask = binary_erosion(hu_mask, structure=erosion_structure)

    ts_holes = binary_fill_holes(candidate_ts_mask) & ~candidate_ts_mask
    ts_holes = _max_component_area_filter(
        ts_holes,
        getattr(args, "max_ts_hole_area", 0),
    )
    hu_holes = binary_fill_holes(candidate_hu_mask) & ~candidate_hu_mask
    boundary_band = (
        binary_dilation(candidate_ts_mask, structure=_disk(args.boundary_radius))
        & ~binary_erosion(candidate_ts_mask, structure=_disk(args.inner_erosion_radius))
    )
    lung_neighbourhood = binary_dilation(
        candidate_ts_mask | candidate_hu_mask,
        structure=_disk(args.neighbourhood_radius),
    )

    if getattr(args, "ggo_extended", False):
        solid = (hu > args.hu_threshold) & (hu < args.solid_max_hu)
        ggo_raw = (hu > args.ggo_min_hu) & (hu < args.ggo_max_hu)
        ground_glass = _component_filter_strict(
            ggo_raw & candidate_ts_mask,
            args.ggo_min_area,
            args.ggo_max_area,
            args.ggo_max_elongation,
        )
        dense = solid | ground_glass
    elif getattr(args, "dual_hu_dense", False):
        ground_glass = (hu > args.ground_glass_min_hu) & (hu < args.ground_glass_max_hu)
        solid = (hu > args.solid_min_hu) & (hu < args.solid_max_hu)
        dense = ground_glass | solid
    else:
        ground_glass = np.zeros_like(hu, dtype=bool)
        solid = hu > args.hu_threshold
        dense = hu > args.hu_threshold
    compact_dense_inside_lung = _component_filter(
        dense & candidate_ts_mask,
        args.min_area,
        args.max_internal_area,
        args.max_internal_elongation,
    )
    pleural_dense_blob = _pleural_dense_blob_mask(
        hu,
        dense,
        candidate_ts_mask,
        lung_neighbourhood,
        args,
    )
    use_boundary_band = not getattr(args, "no_boundary_band", False)
    defect_region = ts_holes | hu_holes
    if use_boundary_band:
        defect_region |= boundary_band
    raw = dense & lung_neighbourhood & (
        defect_region | compact_dense_inside_lung | pleural_dense_blob
    )
    if getattr(args, "restrict-to-ts-or-holes", None):
        raise AttributeError("Use args.restrict_to_ts_or_holes, not args.restrict-to-ts-or-holes.")
    if getattr(args, "restrict_to_ts_or_holes", False):
        raw &= candidate_ts_mask | ts_holes | hu_holes
    exclude_edge_radius = int(getattr(args, "exclude_ts_edge_radius", 0))
    if exclude_edge_radius > 0:
        edge_band = candidate_ts_mask & ~binary_erosion(
            candidate_ts_mask,
            structure=_disk(exclude_edge_radius),
        )
        raw &= ~edge_band
    else:
        edge_band = np.zeros_like(candidate_ts_mask, dtype=bool)
    raw = binary_closing(raw, structure=_disk(args.closing_radius))
    raw = _component_filter(raw, args.min_area, args.max_area, args.max_elongation)
    if getattr(args, "pleural_preserve_after_filter", False):
        raw |= pleural_dense_blob
    return raw.astype(bool), {
        "dense": dense,
        "ground_glass": ground_glass,
        "solid": solid,
        "ts_holes": ts_holes,
        "hu_holes": hu_holes,
        "boundary_band": boundary_band,
        "lung_neighbourhood": lung_neighbourhood,
        "compact_dense_inside_lung": compact_dense_inside_lung,
        "pleural_dense_blob": pleural_dense_blob,
        "candidate_ts_mask": candidate_ts_mask,
        "candidate_hu_mask": candidate_hu_mask,
        "excluded_edge_band": edge_band,
    }


def _soft_prior(candidate, sigma, dilation_iterations=0):
    mask = candidate.astype(bool)
    if dilation_iterations > 0:
        mask = binary_dilation(mask, iterations=int(dilation_iterations))
    prior = gaussian_filter(mask.astype(np.float32), sigma=float(sigma))
    max_value = float(prior.max())
    if max_value > 0:
        prior /= max_value + 1e-8
    return prior.astype(np.float32)


def _prior_from_candidate(candidate, parts, args):
    if getattr(args, "prior-mode", None):
        raise AttributeError("Use args.prior_mode, not args.prior-mode.")
    if getattr(args, "prior_mode", "standard") == "component-weighted":
        return _component_weighted_prior(candidate, parts["solid"], args)
    return _soft_prior(
        candidate,
        args.blur_sigma,
        args.prior_dilation_iterations,
    )


def _make_prior(candidate, parts, args, training_ts_mask=None, candidate_ts_mask=None):
    prior = _prior_from_candidate(candidate, parts, args)
    factor = float(getattr(args, "added_ts_border_downweight", 1.0))
    if (
        factor < 1.0
        and training_ts_mask is not None
        and candidate_ts_mask is not None
    ):
        prior = _downweight_added_ts_border(
            prior,
            training_ts_mask,
            candidate_ts_mask,
            factor,
        )
    return prior


def _overlay(image, candidate, contour):
    base = np.stack([image, image, image], axis=-1)
    overlay = base.copy()
    hit = candidate & contour
    missed = contour & ~candidate
    overlay[candidate, 1] = 0.85
    overlay[candidate, 0] *= 0.35
    overlay[candidate, 2] *= 0.35
    overlay[contour, 0] = 1.0
    overlay[contour, 1] = 0.9
    overlay[contour, 2] = 0.0
    overlay[hit, 0] = 0.0
    overlay[hit, 1] = 1.0
    overlay[hit, 2] = 0.0
    overlay[missed, 0] = 1.0
    overlay[missed, 1] = 0.0
    overlay[missed, 2] = 1.0
    return overlay


def _pick_indices(samples, args):
    rng = np.random.default_rng(args.seed)
    if args.indices:
        return [int(item) for item in args.indices.split(",")]
    return sorted(rng.choice(len(samples), size=min(args.count, len(samples)), replace=False).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="cache/cache_ts_filled_dil1.pkl")
    parser.add_argument(
        "--candidate-ts-cache",
        default=None,
        help="Optional cache whose original ts_mask is used only for candidate generation.",
    )
    parser.add_argument("--out-dir", default="results/possible_nodule_mask_preview")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--indices", default=None)
    parser.add_argument("--seed", type=int, default=7)
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
    parser.add_argument("--boundary-radius", type=int, default=4)
    parser.add_argument(
        "--no-boundary-band",
        action="store_true",
        help="Do not include the TS lung boundary/outline band in candidate generation.",
    )
    parser.add_argument("--inner-erosion-radius", type=int, default=1)
    parser.add_argument("--neighbourhood-radius", type=int, default=4)
    parser.add_argument("--max-ts-hole-area", type=int, default=0)
    parser.add_argument(
        "--restrict-to-ts-or-holes",
        action="store_true",
        help="After candidate generation, keep only pixels inside TS mask or TS/HU holes.",
    )
    parser.add_argument("--exclude-ts-edge-radius", type=int, default=0)
    parser.add_argument(
        "--candidate-lung-erosion-radius",
        type=int,
        default=0,
        help="Erode lung masks for candidate generation only, leaving training masks unchanged.",
    )
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--min-area", type=int, default=3)
    parser.add_argument("--max-area", type=int, default=600)
    parser.add_argument("--max-elongation", type=float, default=6.0)
    parser.add_argument("--max-internal-area", type=int, default=180)
    parser.add_argument("--max-internal-elongation", type=float, default=3.0)
    parser.add_argument(
        "--pleural-dense-blob",
        action="store_true",
        help="Add compact high-HU blobs near the TS lung boundary as candidates.",
    )
    parser.add_argument("--pleural-edge-radius", type=int, default=5)
    parser.add_argument("--pleural-inner-erosion-radius", type=int, default=1)
    parser.add_argument("--pleural-min-area", type=int, default=5)
    parser.add_argument("--pleural-max-area", type=int, default=220)
    parser.add_argument("--pleural-max-elongation", type=float, default=2.5)
    parser.add_argument("--pleural-min-mean-hu", type=float, default=-250.0)
    parser.add_argument("--pleural-min-contrast-hu", type=float, default=80.0)
    parser.add_argument("--pleural-contrast-ring-radius", type=int, default=5)
    parser.add_argument(
        "--pleural-core-hu-threshold",
        type=float,
        default=0.0,
        help="For large dense edge components, keep only local cores above this HU.",
    )
    parser.add_argument("--pleural-core-dilation", type=int, default=1)
    parser.add_argument(
        "--pleural-keep-large-rim",
        action="store_true",
        help="Allow large dense boundary components, but split them into high-HU cores.",
    )
    parser.add_argument(
        "--pleural-preserve-after-filter",
        action="store_true",
        help="Do not let the final global component filter delete the pleural dense term.",
    )
    parser.add_argument("--blur-sigma", type=float, default=4.0)
    parser.add_argument("--prior-dilation-iterations", type=int, default=0)
    parser.add_argument(
        "--added-ts-border-downweight",
        type=float,
        default=1.0,
        help="After prior generation, multiply the added TS dilation rim by this factor.",
    )
    parser.add_argument("--component-area-scale", type=float, default=50.0)
    parser.add_argument("--solid-bonus-scale", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.cache_path, "rb") as file:
        samples = pickle.load(file)
    candidate_ts_lookup = _load_candidate_ts_lookup(args.candidate_ts_cache)
    indices = _pick_indices(samples, args)
    by_patient = defaultdict(list)
    for index in indices:
        by_patient[samples[index]["patient_id"]].append(index)

    records = []
    for patient_id, patient_indices in sorted(by_patient.items()):
        lookup = _candidate_lookup_for_patient(patient_id)
        for index in patient_indices:
            sample = samples[index]
            contour = _contour_for_sample(sample, lookup)
            if contour is None:
                continue
            candidate_sample = _candidate_source_sample(sample, candidate_ts_lookup)
            candidate, parts = _possible_nodule_mask(candidate_sample, args)
            training_ts_mask = _mask(sample, "ts_mask")
            candidate_ts_mask = _mask(candidate_sample, "ts_mask")
            soft_prior = _make_prior(
                candidate,
                parts,
                args,
                training_ts_mask,
                candidate_ts_mask,
            )
            overlap = int((candidate & contour).sum())
            total = int(contour.sum())
            records.append(
                {
                    "index": index,
                    "sample": sample,
                    "image": _center_image(sample),
                    "candidate_ts_mask": candidate_ts_mask,
                    "training_ts_mask": training_ts_mask,
                    "candidate": candidate,
                    "soft_prior": soft_prior,
                    "parts": parts,
                    "contour": contour,
                    "hit": overlap > 0,
                    "coverage": overlap / max(total, 1),
                    "overlap": overlap,
                    "total": total,
                    "area_pct": float(candidate.mean() * 100.0),
                }
            )

    os.makedirs(args.out_dir, exist_ok=True)
    cols = 8
    rows = len(records)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.8))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, record in enumerate(records):
        parts_rgb = np.zeros((*record["candidate"].shape, 3), dtype=np.float32)
        parts_rgb[record["parts"]["boundary_band"], 2] = 0.65
        parts_rgb[record["parts"].get("pleural_dense_blob", np.zeros_like(record["candidate"], dtype=bool)), 0] = 1.0
        parts_rgb[record["parts"].get("pleural_dense_blob", np.zeros_like(record["candidate"], dtype=bool)), 1] = 0.5
        parts_rgb[record["parts"].get("excluded_edge_band", np.zeros_like(record["candidate"], dtype=bool)), 0] = 0.65
        parts_rgb[record["parts"].get("excluded_edge_band", np.zeros_like(record["candidate"], dtype=bool)), 2] = 0.65
        parts_rgb[record["parts"]["ts_holes"] | record["parts"]["hu_holes"], 0] = 0.9
        if np.any(record["parts"].get("ground_glass", False)):
            parts_rgb[record["parts"]["ground_glass"], 1] = np.maximum(
                parts_rgb[record["parts"]["ground_glass"], 1],
                0.45,
            )
            parts_rgb[record["parts"]["solid"], 1] = 0.95
            parts_rgb[record["parts"]["solid"], 0] = np.maximum(
                parts_rgb[record["parts"]["solid"], 0],
                0.45,
            )
        else:
            parts_rgb[record["parts"]["dense"], 1] = np.maximum(parts_rgb[record["parts"]["dense"], 1], 0.35)
        panels = [
            ("CT", record["image"], "gray"),
            ("CT + training TS outline", record["image"], "gray", record["training_ts_mask"], "red"),
            ("CT + original TS outline", record["image"], "gray", record["candidate_ts_mask"], "cyan"),
            ("cancer nodule", record["contour"], "gray"),
            ("candidate", record["candidate"], "gray"),
            ("soft prior", record["soft_prior"], "magma"),
            ("parts: blue=boundary magenta=excluded red=holes green=dense", parts_rgb, None),
            ("audit: green=hit magenta=miss", _overlay(record["image"], record["candidate"], record["contour"]), None),
        ]
        for col_idx, panel in enumerate(panels):
            if len(panel) == 5:
                title, image, cmap, contour_mask, contour_color = panel
            else:
                title, image, cmap = panel
                contour_mask = None
                contour_color = None
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap=cmap)
            if contour_mask is not None and np.any(contour_mask):
                ax.contour(
                    contour_mask.astype(float),
                    levels=[0.5],
                    colors=contour_color,
                    linewidths=0.8,
                )
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                ax.text(
                    0.0,
                    -0.08,
                    (
                        f"idx {record['index']} hit={record['hit']} "
                        f"cov={record['coverage']:.3f}\n"
                        f"overlap {record['overlap']}/{record['total']} px "
                        f"area={record['area_pct']:.1f}%"
                    ),
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                )
    hits = sum(record["hit"] for record in records)
    mean_cov = float(np.mean([record["coverage"] for record in records])) if records else 0.0
    mean_area = float(np.mean([record["area_pct"] for record in records])) if records else 0.0
    fig.suptitle(
        (
            f"Possible nodule mask preview: hits {hits}/{len(records)}, "
            f"mean coverage {mean_cov:.3f}, mean area {mean_area:.1f}%"
        ),
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    png_path = os.path.join(args.out_dir, "possible_nodule_mask_10_examples.png")
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Hits: {hits}/{len(records)}")
    print(f"Mean coverage: {mean_cov:.4f}")
    print(f"Mean candidate area: {mean_area:.2f}%")


if __name__ == "__main__":
    main()
