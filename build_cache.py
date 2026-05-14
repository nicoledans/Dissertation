import os
import pickle
import tempfile
import shutil
import numpy as np
import pylidc as pl
from scipy.ndimage import (
    label, binary_fill_holes, binary_dilation, binary_opening, zoom
) #Used for image processing functions

# TotalSegmentator (used in oracle-ct also)
TS_AVAILABLE = False
try:
    import nibabel as nib
    import torch as _torch
    from totalsegmentator.python_api import totalsegmentator as _run_ts
    TS_AVAILABLE = True
except ImportError:
    pass

# Hardcoded settings 
# HU values below threshold are "air-like" (lung interior / outside air)
LUNG_HU_THRESHOLD = -500 
# Grow mask by 3 pixels
MASK_DILATION = 3
# radiologists rate 1–5, threshold at 3
    # if avg_mal 3 - skipped not added to cache
    # if avg_mal > 3 - malignant; <3 benign
MALIGNANCY_THRESHOLD = 3
# At least 2 radiologist must have annotated the module
MIN_RADIOLOGISTS = 2
# How many to cache
SAMPLE_SIZE = 2000
CACHE_PATH = "results/cache.pkl"

# Physics-based lung foreground mask
    # Uses HU thresholding as a Grad-CAM
    # Attention training signal

# This follows logic in HU_bg_gf_separation.ipynb
def _get_lung_mask(slice_2d):

    binary = slice_2d < LUNG_HU_THRESHOLD

    labeled, num_features = label(binary)

    # Background air - those that touch image border
    border_labels = set()
    border_labels.update(labeled[0, :])    # top row
    border_labels.update(labeled[-1, :])   # bottom row
    border_labels.update(labeled[:, 0])    # left col
    border_labels.update(labeled[:, -1])   # right col
    border_labels.discard(0)

    # Keep only internal regions
    labeled_no_border = labeled.copy()
    for lbl in border_labels:
        labeled_no_border[labeled_no_border == lbl] = 0
    binary_internal = labeled_no_border > 0

    # Re-label remaining internal regions
    labeled2, num2 = label(binary_internal)

    if num2 == 0:
        return np.zeros(slice_2d.shape, dtype=np.uint8)

    # Keep 2 largest internal regions (left, right lung)
    region_sizes = np.bincount(labeled2.ravel())[1:]  # index 0 is background
    n_keep = min(2, len(region_sizes))
    sorted_idx = np.argsort(region_sizes)[::-1]
    keep_labels = sorted_idx[:n_keep] + 1

    largest_regions_mask = np.zeros_like(binary_internal, dtype=bool)
    for lbl in keep_labels:
        largest_regions_mask |= (labeled2 == lbl)

    # Fill holes
    filled_mask = binary_fill_holes(largest_regions_mask)

    # Dilate to capture others at boundary
    dilated_mask = binary_dilation(filled_mask, iterations=MASK_DILATION)

    # Binary opening to remove noise
    clean_mask = binary_opening(dilated_mask, structure=np.ones((3, 3)))

    # [ADDED] mediastinum exclusion via central column zeroing
    h, w = clean_mask.shape
    col_start = int(w * 0.40)
    col_end = int(w * 0.60)
    clean_mask = clean_mask.copy()
    clean_mask[:, col_start:col_end] = 0

    return clean_mask.astype(np.uint8)


_TS_LUNG_LABELS = [
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]

# Total Segmentartor Mask
def _get_ts_vol_mask(vol, patient_id):
    if not TS_AVAILABLE:
        return None
    
    # Makes temp folder as needs NIfTI format
    tmpdir = tempfile.mkdtemp(prefix=f"ts_{patient_id}_")
    try:
        nifti_in = os.path.join(tmpdir, "ct.nii.gz")
        nifti_out = os.path.join(tmpdir, "seg")
        os.makedirs(nifti_out, exist_ok=True)

        nib_img = nib.Nifti1Image(vol.astype(np.float32), affine=np.eye(4))
        nib.save(nib_img, nifti_in)

        # Run TotalSegmentator on the saved NIfTI. Writes one mask file per organ
        device = "gpu" if _torch.cuda.is_available() else "cpu"
        _run_ts(nifti_in, nifti_out, task="total", device=device, quiet=True)

        combined = None
        for lname in _TS_LUNG_LABELS:
            seg_path = os.path.join(nifti_out, f"{lname}.nii.gz")
            if os.path.exists(seg_path):
                seg = nib.load(seg_path).get_fdata()
                combined = seg if combined is None else combined + seg

        # Combine into one whole lung mask
        if combined is None:
            return None

        ts_3d = (combined > 0).astype(bool)
        ts_3d = binary_dilation(ts_3d, iterations=MASK_DILATION)
        ts_3d = binary_opening(ts_3d, structure=np.ones((3, 3, 3)))

        w = ts_3d.shape[1]
        col_start = int(w * 0.40)
        col_end = int(w * 0.60)
        ts_3d[:, col_start:col_end, :] = False

        return ts_3d.astype(np.uint8)

    except Exception as e:
        print(f"  TS WARNING for {patient_id}: {e} — using HU fallback")
        return None
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _get_middle_slice_idx(ann):
    # Use image_k_position
    k_indices = sorted(set([c.image_k_position for c in ann.contours]))
    return k_indices[len(k_indices) // 2]


def main():
    os.makedirs("results", exist_ok=True)
    scans = pl.query(pl.Scan).all()
    samples = [] #P ull every scan in the LIDC db as a list of pylidc Scan objects.

    for scan_idx, scan in enumerate(scans):
        if len(samples) >= SAMPLE_SIZE:
            break

        print(f"Scan {scan_idx:04d} | patient {scan.patient_id} | cached so far: {len(samples)}")

        try:
            vol = scan.to_volume()
            nod_groups = scan.cluster_annotations()

            # TotalSegmentator mask- computed once per scan
            ts_vol_mask = _get_ts_vol_mask(vol, scan.patient_id)

            for ann_group in nod_groups:
                if len(samples) >= SAMPLE_SIZE:
                    break

                if len(ann_group) < MIN_RADIOLOGISTS:
                    continue

                avg_mal = float(np.mean([a.malignancy for a in ann_group]))

                if avg_mal == MALIGNANCY_THRESHOLD:
                    continue

                label_val = 1 if avg_mal > MALIGNANCY_THRESHOLD else 0

                ann = ann_group[0]

                try:
                    slice_idx = _get_middle_slice_idx(ann)
                    slice_idx = int(np.clip(slice_idx, 0, vol.shape[2] - 1))

                    raw_slice = vol[:, :, slice_idx].astype(np.float32)

                    lung_mask = _get_lung_mask(raw_slice)

                    # Clip and normalise HU to [0, 1]
                    raw_slice = np.clip(raw_slice, -1000.0, 400.0)
                    raw_slice = (raw_slice - (-1000.0)) / (400.0 - (-1000.0))

                    # Resize to 224x224
                    zh = 224.0 / raw_slice.shape[0]
                    zw = 224.0 / raw_slice.shape[1]
                    image_224 = zoom(raw_slice, (zh, zw), order=1).astype(np.float32)
                    mask_224 = zoom(lung_mask.astype(np.float32), (zh, zw), order=0)
                    mask_224 = (mask_224 > 0.5).astype(np.uint8)

                    # TS mask: extract 2D slice from precomputed 3D mas +  resize
                    if ts_vol_mask is not None:
                        ts_slice = ts_vol_mask[:, :, slice_idx].astype(np.float32)
                        ts_mask_224 = zoom(ts_slice, (zh, zw), order=0)
                        ts_mask_224 = (ts_mask_224 > 0.5).astype(np.uint8)
                    else:
                        ts_mask_224 = None  # TS unavailable — no fallback

                    samples.append({
                        "image": image_224,
                        "mask": mask_224,
                        "ts_mask": ts_mask_224,
                        "label": label_val,
                        "patient_id": scan.patient_id,
                    })

                except Exception as e:
                    print(f"  WARNING: skipped nodule in {scan.patient_id}: {e}")
                    continue

        except Exception as e:
            print(f"  WARNING: skipped scan {scan.patient_id}: {e}")
            continue

    with open(CACHE_PATH, "wb") as f:
        pickle.dump(samples, f)

    n_mal = sum(s["label"] for s in samples)
    n_ben = len(samples) - n_mal
    print(f"\nDone. Cached {len(samples)} nodule slices to {CACHE_PATH}")
    print(f"  {n_ben} benign, {n_mal} malignant")
    print("Now run train.py and baseline.py — loading will be instant")


if __name__ == "__main__":
    main()
