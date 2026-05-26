import os
import pickle
import tempfile
import shutil
import argparse
import numpy as np
import pylidc as pl
from scipy.ndimage import (
    label, binary_fill_holes, binary_dilation, binary_opening, zoom
) #Used for image processing functions
from config import (
    SAMPLE_SIZE as _CFG_SAMPLE_SIZE,
    LUNG_HU_THRESHOLD as _CFG_LUNG_HU_THRESHOLD,
    MASK_DILATION as _CFG_MASK_DILATION,
    CACHE_PATH as _CFG_CACHE_PATH,
    MIN_RADIOLOGISTS as _CFG_MIN_RADIOLOGISTS,
    MALIGNANCY_THRESHOLD as _CFG_MALIGNANCY_THRESHOLD,
)

# TotalSegmentator (used in oracle-ct also)
TS_AVAILABLE = False
try:
    import nibabel as nib
    import torch as _torch
    from totalsegmentator.python_api import totalsegmentator as _run_ts
    TS_AVAILABLE = True
except ImportError:
    pass

# All constants from config.py — single source of truth
LUNG_HU_THRESHOLD = _CFG_LUNG_HU_THRESHOLD
MASK_DILATION = _CFG_MASK_DILATION
MALIGNANCY_THRESHOLD = _CFG_MALIGNANCY_THRESHOLD
MIN_RADIOLOGISTS = _CFG_MIN_RADIOLOGISTS
CACHE_PATH = _CFG_CACHE_PATH

CHECKPOINT_PATH = "results/cache_checkpoint.pkl"

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
def _get_ts_vol_mask(vol, patient_id, device):
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
        _run_ts(
            nifti_in, nifti_out, task="total", device=device, fast=True, quiet=True,
            roi_subset=[
                "lung_upper_lobe_left", "lung_lower_lobe_left",
                "lung_upper_lobe_right", "lung_middle_lobe_right",
                "lung_lower_lobe_right",
            ],
        )

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


def _save_checkpoint(samples, next_scan_idx, path):
    with open(path, "wb") as f:
        pickle.dump({"samples": samples, "next_scan_idx": next_scan_idx}, f)


def _load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size", type=int, default=_CFG_SAMPLE_SIZE,
        help="Override SAMPLE_SIZE from config (e.g. 200 for a quick demo run)"
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing checkpoint and start from scratch"
    )
    parser.add_argument(
        "--no-ts", action="store_true",
        help="Skip TotalSegmentator — cache HU masks only (faster, for Exp 1)"
    )
    parser.add_argument(
        "--cache-path", type=str, default=None,
        help="Override cache output path (default: results/cache.pkl)"
    )
    args = parser.parse_args()
    SAMPLE_SIZE = args.sample_size

    cache_out = args.cache_path if args.cache_path else CACHE_PATH
    checkpoint_out = cache_out.replace(".pkl", "_checkpoint.pkl")

    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(cache_out)), exist_ok=True)

    # GPU status
    if args.no_ts:
        device = "cpu"
        print("[NO-TS] Skipping TotalSegmentator — HU masks only (ts_mask=None for all samples).")
    elif TS_AVAILABLE and _torch.cuda.is_available():
        device = "gpu"
        gpu_name = _torch.cuda.get_device_name(0)
        print(f"[GPU] TotalSegmentator will use CUDA: {gpu_name}")
    else:
        device = "cpu"
        if TS_AVAILABLE:
            print("[CPU] No CUDA detected — TotalSegmentator running on CPU (slow). "
                  "Install CUDA drivers + torch-cuda if you have a GPU.")
        else:
            print("[CPU] TotalSegmentator not available — TS masks will be None.")

    print(f"[CACHE] Output → {cache_out}")

    # Checkpoint resume
    checkpoint = None if args.fresh else _load_checkpoint(checkpoint_out)
    if checkpoint:
        samples = checkpoint["samples"]
        start_scan_idx = checkpoint["next_scan_idx"]
        print(f"[RESUME] Loaded checkpoint: {len(samples)} samples already collected, "
              f"resuming from scan index {start_scan_idx}")
    else:
        samples = []
        start_scan_idx = 0
        if args.fresh and os.path.exists(checkpoint_out):
            os.remove(checkpoint_out)
            print("[FRESH] Deleted existing checkpoint, starting from scratch.")

    scans = pl.query(pl.Scan).all()

    for scan_idx, scan in enumerate(scans):
        if scan_idx < start_scan_idx:
            continue

        if len(samples) >= SAMPLE_SIZE:
            break

        print(f"Scan {scan_idx:04d} | patient {scan.patient_id} | cached so far: {len(samples)}/{SAMPLE_SIZE}")

        try:
            vol = scan.to_volume()
            nod_groups = scan.cluster_annotations()

            # TotalSegmentator mask — computed once per scan (None if --no-ts)
            if args.no_ts:
                ts_vol_mask = None
            else:
                ts_vol_mask = _get_ts_vol_mask(vol, scan.patient_id, device)

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

        # Save checkpoint after every scan so a crash loses at most one scan's worth
        _save_checkpoint(samples, scan_idx + 1, checkpoint_out)

    with open(cache_out, "wb") as f:
        pickle.dump(samples, f)

    # Keep checkpoint so a future run with a larger SAMPLE_SIZE can resume from here
    # (use --fresh to discard it and start over)

    n_mal = sum(s["label"] for s in samples)
    n_ben = len(samples) - n_mal
    ts_count = sum(1 for s in samples if s.get("ts_mask") is not None)
    print(f"\nDone. Cached {len(samples)} nodule slices to {cache_out}")
    print(f"  {n_ben} benign, {n_mal} malignant | TS masks: {ts_count}/{len(samples)}")
    print(f"Now run baseline.py / train.py — pass --cache-path {cache_out} if not using default")


if __name__ == "__main__":
    main()
