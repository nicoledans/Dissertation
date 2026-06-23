"""Soft nodule-like blob maps from CT physics cues.

These maps are weak image-derived priors, not lesion annotations.  They are
intended for gentle Grad-CAM regularisation and audit, so the output is
continuous and permissive rather than a binary, ranked candidate mask.
"""

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    gaussian_filter,
    gaussian_laplace,
    label as ndi_label,
)
from skimage.measure import perimeter_crofton
from skimage.morphology import disk


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _normalise(values, mask=None):
    values = np.asarray(values, dtype=np.float32)
    if mask is None:
        active = values[np.isfinite(values)]
    else:
        active = values[np.asarray(mask).astype(bool) & np.isfinite(values)]
    if active.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo = float(np.percentile(active, 1.0))
    hi = float(np.percentile(active, 99.0))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _hu_gate(raw_slice, lower, upper, softness):
    lower_gate = _sigmoid((raw_slice - lower) / max(float(softness), 1e-6))
    upper_gate = _sigmoid((upper - raw_slice) / max(float(softness), 1e-6))
    return (lower_gate * upper_gate).astype(np.float32)


def _log_response(image, search_mask, sigmas):
    response = np.zeros_like(image, dtype=np.float32)
    for sigma in sigmas:
        # Bright blobs give a negative LoG centre; flip and scale-normalise.
        scale_response = -float(sigma) ** 2 * gaussian_laplace(image, sigma=float(sigma))
        response = np.maximum(response, scale_response.astype(np.float32))
    response[~search_mask] = 0.0
    return _normalise(response, search_mask)


def _hessian_blobness_and_vesselness(image, search_mask, sigmas):
    blobness = np.zeros_like(image, dtype=np.float32)
    vesselness = np.zeros_like(image, dtype=np.float32)
    for sigma in sigmas:
        sigma = float(sigma)
        dxx = gaussian_filter(image, sigma=sigma, order=(2, 0)) * sigma * sigma
        dxy = gaussian_filter(image, sigma=sigma, order=(1, 1)) * sigma * sigma
        dyy = gaussian_filter(image, sigma=sigma, order=(0, 2)) * sigma * sigma
        trace = dxx + dyy
        delta = np.sqrt(np.maximum((dxx - dyy) ** 2 + 4.0 * dxy ** 2, 0.0))
        eig1 = 0.5 * (trace - delta)
        eig2 = 0.5 * (trace + delta)
        abs1 = np.abs(eig1)
        abs2 = np.abs(eig2)
        small = np.minimum(abs1, abs2)
        large = np.maximum(abs1, abs2)
        bright_structure = (eig1 < 0.0) & (eig2 < 0.0)
        compact = np.sqrt(np.maximum(abs1 * abs2, 0.0))
        line_like = (1.0 - np.clip(small / (large + 1e-6), 0.0, 1.0)) * large
        blobness = np.maximum(blobness, np.where(bright_structure, compact, 0.0))
        vesselness = np.maximum(vesselness, np.where(large > 0.0, line_like, 0.0))
    blobness[~search_mask] = 0.0
    vesselness[~search_mask] = 0.0
    return _normalise(blobness, search_mask), _normalise(vesselness, search_mask)


def _local_cues(image, search_mask):
    background = gaussian_filter(image, sigma=5.0)
    local_contrast = _normalise(np.maximum(image - background, 0.0), search_mask)
    mean = gaussian_filter(image, sigma=2.0)
    mean_sq = gaussian_filter(image * image, sigma=2.0)
    texture = _normalise(np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)), search_mask)
    return local_contrast, texture


def _compactness_weight(response, search_mask, threshold=0.20):
    active = (response >= threshold) & search_mask
    labels, n_labels = ndi_label(active)
    weights = np.ones_like(response, dtype=np.float32)
    for component_id in range(1, n_labels + 1):
        component = labels == component_id
        area = float(component.sum())
        if area <= 0.0:
            continue
        perimeter = float(perimeter_crofton(component, directions=4))
        compactness = 0.0 if perimeter <= 0.0 else min(1.0, 4.0 * np.pi * area / (perimeter * perimeter))
        weights[component] = 0.75 + 0.25 * compactness
    return weights


def _single_slice_maps(
    raw_slice,
    lung_mask,
    pixel_spacing_mm,
    lung_dilation_px,
    min_diameter_mm,
    max_diameter_mm,
    vessel_suppression,
    blur_sigma_px,
):
    raw = np.asarray(raw_slice, dtype=np.float32)
    lung = binary_fill_holes(np.asarray(lung_mask).astype(bool))
    if lung_dilation_px > 0:
        lung = binary_dilation(lung, structure=disk(int(lung_dilation_px)))
    if not np.any(lung):
        empty = np.zeros_like(raw, dtype=np.float32)
        return empty, empty, empty, empty

    clipped = np.clip(raw, -1000.0, 400.0)
    image = ((clipped + 1000.0) / 1400.0).astype(np.float32)
    min_sigma = min_diameter_mm / (2.0 * np.sqrt(2.0) * pixel_spacing_mm)
    max_sigma = max_diameter_mm / (2.0 * np.sqrt(2.0) * pixel_spacing_mm)
    sigmas = np.geomspace(max(0.6, min_sigma), max(0.7, max_sigma), num=8)

    log_map = _log_response(image, lung, sigmas)
    hessian_map, vesselness = _hessian_blobness_and_vesselness(image, lung, sigmas)
    local_contrast, texture = _local_cues(image, lung)

    solid_hu = _hu_gate(raw, lower=-550.0, upper=350.0, softness=90.0)
    subsolid_hu = _hu_gate(raw, lower=-900.0, upper=-120.0, softness=130.0)
    suppression = 1.0 - min(max(float(vessel_suppression), 0.0), 0.8) * vesselness

    solid = (
        (0.55 * log_map + 0.30 * hessian_map + 0.15 * local_contrast)
        * solid_hu
        * suppression
    )
    subsolid = (
        (0.45 * log_map + 0.20 * hessian_map + 0.20 * local_contrast + 0.15 * texture)
        * (0.35 + 0.65 * subsolid_hu)
        * suppression
    )
    solid *= _compactness_weight(solid, lung)
    subsolid *= _compactness_weight(subsolid, lung)
    solid[~lung] = 0.0
    subsolid[~lung] = 0.0

    if blur_sigma_px > 0:
        solid = gaussian_filter(solid, sigma=float(blur_sigma_px)) * lung
        subsolid = gaussian_filter(subsolid, sigma=float(blur_sigma_px)) * lung
    return _normalise(solid, lung), _normalise(subsolid, lung), _normalise(vesselness, lung), lung


def soft_blob_maps(
    raw_slice,
    lung_mask,
    pixel_spacing_mm,
    adjacent_slices=None,
    lung_dilation_px=3,
    min_diameter_mm=3.0,
    max_diameter_mm=30.0,
    vessel_suppression=0.30,
    blur_sigma_px=2.0,
    persistence_weight=0.25,
):
    """Return continuous solid, subsolid, and combined soft blob maps."""
    solid, subsolid, vesselness, search_mask = _single_slice_maps(
        raw_slice,
        lung_mask,
        pixel_spacing_mm,
        lung_dilation_px,
        min_diameter_mm,
        max_diameter_mm,
        vessel_suppression,
        blur_sigma_px,
    )
    combined = np.maximum(solid, subsolid)

    if adjacent_slices and persistence_weight > 0 and np.any(search_mask):
        adjacent_combined = np.zeros_like(combined, dtype=np.float32)
        for adjacent in adjacent_slices:
            adj_solid, adj_subsolid, _adj_vessel, _adj_search = _single_slice_maps(
                adjacent,
                lung_mask,
                pixel_spacing_mm,
                lung_dilation_px,
                min_diameter_mm,
                max_diameter_mm,
                vessel_suppression,
                blur_sigma_px,
            )
            adjacent_combined = np.maximum(adjacent_combined, np.maximum(adj_solid, adj_subsolid))
        persistence = _normalise(adjacent_combined, search_mask)
        combined = combined * (1.0 + float(persistence_weight) * persistence)
        solid = solid * (1.0 + float(persistence_weight) * persistence)
        subsolid = subsolid * (1.0 + float(persistence_weight) * persistence)

    combined = _normalise(combined, search_mask)
    return {
        "soft_blob_map": combined.astype(np.float32),
        "solid_like_map": _normalise(solid, search_mask).astype(np.float32),
        "subsolid_like_map": _normalise(subsolid, search_mask).astype(np.float32),
        "vesselness_map": vesselness.astype(np.float32),
        "search_mask": search_mask.astype(np.uint8),
        "params": {
            "lung_dilation_px": int(lung_dilation_px),
            "min_diameter_mm": float(min_diameter_mm),
            "max_diameter_mm": float(max_diameter_mm),
            "vessel_suppression": float(vessel_suppression),
            "blur_sigma_px": float(blur_sigma_px),
            "persistence_weight": float(persistence_weight),
        },
    }
