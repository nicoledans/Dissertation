"""Helpers for matching cached LIDC samples back to raw pylidc annotations."""

import hashlib
import math

import numpy as np
import pylidc as pl
from scipy.ndimage import zoom
from skimage.draw import polygon

from config import (
    HU_MAX,
    HU_MIN,
    IMG_SIZE,
    MALIGNANCY_THRESHOLD,
    MIN_RADIOLOGISTS,
)


def _image_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _middle_slice_idx(annotation_group):
    indices = sorted(
        {
            contour.image_k_position
            for annotation in annotation_group
            for contour in annotation.contours
        }
    )
    if not indices:
        raise ValueError("Annotation group has no contour slices.")
    return int(indices[len(indices) // 2])


def _mean_annotation_attr(annotation_group, name):
    values = [
        float(getattr(annotation, name))
        for annotation in annotation_group
        if getattr(annotation, name, None) is not None
    ]
    return float(np.mean(values)) if values else float("nan")


def _prepare_image(raw_slice):
    image = np.clip(raw_slice.astype(np.float32), HU_MIN, HU_MAX)
    image = (image - HU_MIN) / (HU_MAX - HU_MIN)
    factors = (IMG_SIZE / image.shape[0], IMG_SIZE / image.shape[1])
    return zoom(image, factors, order=1).astype(np.float32)


def _reader_mask(annotation, slice_idx, shape):
    mask = np.zeros(shape, dtype=bool)
    for contour in annotation.contours:
        if int(contour.image_k_position) != int(slice_idx):
            continue
        coordinates = contour.to_matrix(include_k=False)
        rows, cols = polygon(coordinates[:, 0], coordinates[:, 1], shape=shape)
        inclusion = str(getattr(contour, "inclusion", "TRUE")).upper()
        if inclusion == "FALSE":
            mask[rows, cols] = False
        else:
            mask[rows, cols] = True
    return mask


def _annotation_masks_full(annotation_group, slice_idx, shape):
    reader_masks = [
        _reader_mask(annotation, slice_idx, shape)
        for annotation in annotation_group
    ]
    reader_masks = [mask for mask in reader_masks if np.any(mask)]
    if not reader_masks:
        raise ValueError("No radiologist contour exists on selected slice.")

    votes = np.stack(reader_masks).sum(axis=0)
    union = votes >= 1
    majority = votes >= math.ceil(len(reader_masks) / 2)
    if not np.any(majority):
        majority = union

    factors = (IMG_SIZE / shape[0], IMG_SIZE / shape[1])
    union_224 = zoom(union.astype(np.uint8), factors, order=0).astype(bool)
    majority_224 = zoom(majority.astype(np.uint8), factors, order=0).astype(bool)
    return union, majority, union_224, majority_224, len(reader_masks)


def _eligible_candidates(scan):
    volume = scan.to_volume()
    zvals = np.sort(scan.slice_zvals)
    slice_spacing_mm = (
        float(np.median(np.diff(zvals)))
        if len(zvals) > 1
        else float(scan.slice_thickness)
    )
    candidates = []
    for group_index, annotation_group in enumerate(scan.cluster_annotations()):
        if len(annotation_group) < MIN_RADIOLOGISTS:
            continue
        mean_rating = float(np.mean([a.malignancy for a in annotation_group]))
        if mean_rating == MALIGNANCY_THRESHOLD:
            continue
        label = int(mean_rating > MALIGNANCY_THRESHOLD)
        slice_idx = _middle_slice_idx(annotation_group)
        slice_idx = int(np.clip(slice_idx, 0, volume.shape[2] - 1))
        raw_slice = volume[:, :, slice_idx]
        adjacent_raw_slices = []
        for adjacent_idx in (slice_idx - 1, slice_idx + 1):
            if 0 <= adjacent_idx < volume.shape[2]:
                adjacent_raw_slices.append(volume[:, :, adjacent_idx].astype(np.float32))
        image = _prepare_image(raw_slice)
        pixel_spacing = float(scan.pixel_spacing)
        effective_spacing_row_mm = pixel_spacing * raw_slice.shape[0] / IMG_SIZE
        effective_spacing_col_mm = pixel_spacing * raw_slice.shape[1] / IMG_SIZE
        union_raw, majority_raw, union, majority, readers = _annotation_masks_full(
            annotation_group, slice_idx, raw_slice.shape
        )
        candidates.append(
            {
                "raw_slice": raw_slice.astype(np.float32),
                "image": image,
                "hash": _image_hash(image),
                "label": label,
                "patient_id": scan.patient_id,
                "scan_id": str(scan.id),
                "group_index": group_index,
                "slice_idx": slice_idx,
                "mean_rating": mean_rating,
                "mean_subtlety": _mean_annotation_attr(annotation_group, "subtlety"),
                "mean_margin": _mean_annotation_attr(annotation_group, "margin"),
                "mean_spiculation": _mean_annotation_attr(annotation_group, "spiculation"),
                "mean_texture": _mean_annotation_attr(annotation_group, "texture"),
                "reader_count": readers,
                "source_shape": raw_slice.shape,
                "pixel_spacing_mm": pixel_spacing,
                "slice_spacing_mm": slice_spacing_mm,
                "effective_spacing_row_mm": effective_spacing_row_mm,
                "effective_spacing_col_mm": effective_spacing_col_mm,
                "adjacent_raw_slices": adjacent_raw_slices,
                "union_raw": union_raw,
                "majority_raw": majority_raw,
                "union": union,
                "majority": majority,
            }
        )
    return candidates
