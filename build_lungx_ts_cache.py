"""Build a LUNGx cache with TotalSegmentator lung masks.

The cache has one sample per annotated LUNGx nodule centre. It mirrors the LIDC
cache shape used by the training/evaluation scripts:
  image: 224x224 CT slice in [0, 1]
  mask: 224x224 HU fallback lung mask
  ts_mask_original: 224x224 TotalSegmentator mask from the full CT volume
  ts_mask: 224x224 postprocessed TS mask, filled + closing radius 1 + dil1
  label, scan_id, nodule_number, centre metadata
"""

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pydicom

from build_cache import (
    _TS_INSTALLED,
    _get_lung_mask,
    _get_ts_vol_mask,
    _resize_mask_224,
)
from config import HU_MAX, HU_MIN, IMG_SIZE
from evaluate_lungx import _case_dirs, _dicom_index, _load_hu_slice, _load_lungx_rows
from postprocess_ts_masks import _postprocess_mask
from train_tristream import _resize_224


def _normalise_hu(raw_slice):
    image = np.clip(raw_slice.astype(np.float32), HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def _scan_meta(scan_id, entries):
    first = pydicom.dcmread(entries[0][2], stop_before_pixels=True)
    spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    pixel_spacing = float(spacing[0])
    slice_thickness = float(getattr(first, "SliceThickness", 1.0))
    zvals = []
    for _instance, z_pos, path in entries:
        if z_pos is None:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            if hasattr(ds, "ImagePositionPatient"):
                z_pos = float(ds.ImagePositionPatient[2])
        zvals.append(float(z_pos) if z_pos is not None else float(len(zvals)))
    return SimpleNamespace(
        patient_id=scan_id,
        pixel_spacing=pixel_spacing,
        slice_thickness=slice_thickness,
        slice_zvals=np.asarray(zvals, dtype=np.float32),
    )


def _load_volume(entries):
    slices = []
    for _instance, _z_pos, path in entries:
        slices.append(_load_hu_slice(path).astype(np.float32))
    return np.stack(slices, axis=2)


def _centre_to_224(record, raw_slice):
    scale_x = IMG_SIZE / float(raw_slice.shape[1])
    scale_y = IMG_SIZE / float(raw_slice.shape[0])
    return (
        (float(record["center_x_1based"]) - 1.0) * scale_x,
        (float(record["center_y_1based"]) - 1.0) * scale_y,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", default=r"C:\repo\manifest-cgqtDj7Y2699835271585651107")
    parser.add_argument("--xlsx", default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    parser.add_argument("--output-cache", default=r"cache\cache_lungx_ts_filled_dil1.pkl")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    parser.add_argument("--limit-scans", type=int, default=0, help="Debug only: process first N scans.")
    args = parser.parse_args()

    if not _TS_INSTALLED:
        raise RuntimeError("TotalSegmentator/nibabel/torch imports are not available in this environment.")

    import torch

    device = args.device
    if device == "auto":
        device = "gpu" if torch.cuda.is_available() else "cpu"

    image_root = os.path.join(args.manifest_root, "SPIE-AAPM Lung CT Challenge")
    cases = _case_dirs(image_root)
    records = _load_lungx_rows(args.xlsx)
    by_scan = {}
    for record in records:
        by_scan.setdefault(record["scan_id"], []).append(record)

    samples = []
    failed_scans = []
    processed_scans = 0
    for scan_id in sorted(by_scan):
        if args.limit_scans and processed_scans >= args.limit_scans:
            break
        case_dir = cases.get(scan_id)
        if case_dir is None:
            failed_scans.append((scan_id, "missing DICOM directory"))
            continue
        print(f"Processing {scan_id} ({len(by_scan[scan_id])} nodules)")
        entries, by_instance = _dicom_index(case_dir)
        index_by_instance = {int(instance): idx for idx, (instance, _z, _path) in enumerate(entries)}
        vol = _load_volume(entries)
        meta = _scan_meta(scan_id, entries)
        ts_vol = _get_ts_vol_mask(vol, meta, device=device)
        if ts_vol is None:
            failed_scans.append((scan_id, "TotalSegmentator returned no mask"))
            continue

        for record in by_scan[scan_id]:
            dicom_path = by_instance.get(record["center_image"])
            if dicom_path is None:
                slice_idx = max(0, min(record["center_image"] - 1, len(entries) - 1))
                dicom_path = entries[slice_idx][2]
            else:
                slice_idx = index_by_instance.get(record["center_image"], record["center_image"] - 1)
            raw_slice = _load_hu_slice(dicom_path)
            image_224 = _resize_224(_normalise_hu(raw_slice), order=1).astype(np.float32)
            hu_mask_224 = _resize_mask_224(_get_lung_mask(raw_slice))
            ts_original = _resize_mask_224(ts_vol[:, :, slice_idx].astype(np.uint8))
            ts_filled, _before = _postprocess_mask(
                ts_original,
                closing_radius=1,
                dilation_radius=1,
            )
            center_x_224, center_y_224 = _centre_to_224(record, raw_slice)
            samples.append(
                {
                    "image": image_224,
                    "mask": hu_mask_224.astype(np.uint8),
                    "ts_mask_original": ts_original.astype(np.uint8),
                    "ts_mask": ts_filled.astype(np.uint8),
                    "label": record["label"],
                    "patient_id": scan_id,
                    "scan_id": scan_id,
                    "nodule_number": record["nodule_number"],
                    "center_x_1based": record["center_x_1based"],
                    "center_y_1based": record["center_y_1based"],
                    "center_x_224": center_x_224,
                    "center_y_224": center_y_224,
                    "center_image": record["center_image"],
                    "diagnosis": record["diagnosis"],
                    "dicom_path": dicom_path,
                    "ts_postprocess": "fill_holes + closing_radius_1 + dilation_radius_1",
                }
            )
        processed_scans += 1
        print(f"  cached samples so far: {len(samples)}")

    output_cache = Path(args.output_cache)
    output_cache.parent.mkdir(parents=True, exist_ok=True)
    with output_cache.open("wb") as file:
        pickle.dump(samples, file, protocol=pickle.HIGHEST_PROTOCOL)

    info_path = output_cache.with_name(output_cache.stem + "_info.txt")
    lines = [
        "LUNGx TotalSegmentator cache",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Manifest root: {args.manifest_root}",
        f"Annotation sheet: {args.xlsx}",
        f"Output cache: {args.output_cache}",
        f"Device: {device}",
        f"Scans requested: {len(by_scan)}",
        f"Scans processed: {processed_scans}",
        f"Samples cached: {len(samples)}",
        f"Samples with TS mask: {sum(1 for s in samples if s.get('ts_mask') is not None)}/{len(samples)}",
        "ts_mask_original: TS full-volume lung mask, resized to 224",
        "ts_mask: ts_mask_original after fill_holes + closing radius 1 + dilation radius 1",
        f"Failed scans: {len(failed_scans)}",
    ]
    for scan_id, reason in failed_scans:
        lines.append(f"  {scan_id}: {reason}")
    info_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"Saved cache: {output_cache}")
    print(f"Saved info: {info_path}")


if __name__ == "__main__":
    main()
