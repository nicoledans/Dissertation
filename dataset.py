import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import label, binary_fill_holes, binary_dilation, binary_opening
import torchvision.transforms.functional as TF
import pylidc as pl
from config import (
    SAMPLE_SIZE, IMG_SIZE, HU_MIN, HU_MAX,
    LUNG_HU_THRESHOLD, MASK_DILATION, SEED,
    RESULTS_DIR,
)


def _get_lung_mask(volume):
    # HU Physics-based lung foreground mask
    binary = volume > LUNG_HU_THRESHOLD

    # Label all connected regions
    labeled, n_regions = label(binary)

    # Remove border-connected regions (background air)
    border_labels = set()
    for ax in range(3):
        for sl in [0, labeled.shape[ax] - 1]:
            idx = [slice(None)] * 3
            idx[ax] = sl
            border_labels.update(np.unique(labeled[tuple(idx)]))
    border_labels.discard(0)

    internal = np.zeros_like(labeled, dtype=bool)
    for lbl in range(1, n_regions + 1):
        if lbl not in border_labels:
            internal |= labeled == lbl

    # Re-label remaining internal regions
    internal_labeled, n_internal = label(internal)

    # Keep 2 largest internal regions (left + right lung)
    sizes = [(internal_labeled == i).sum() for i in range(1, n_internal + 1)]
    top2 = sorted(range(1, n_internal + 1), key=lambda i: sizes[i - 1], reverse=True)[:2]
    lung_mask = np.zeros_like(internal_labeled, dtype=bool)
    for lbl in top2:
        lung_mask |= internal_labeled == lbl

    lung_mask = binary_fill_holes(lung_mask)

    # dilate to capture boundary findings
    lung_mask = binary_dilation(lung_mask, iterations=MASK_DILATION)

    lung_mask = binary_opening(lung_mask)

    # [ADDED] ediastinum exclusion via central column zeroing
    w = lung_mask.shape[1]
    col_start = int(w * 0.40)
    col_end = int(w * 0.60)
    lung_mask = lung_mask.copy()
    lung_mask[:, col_start:col_end, :] = False

    return lung_mask


def _normalise_hu(volume):
    volume = np.clip(volume, HU_MIN, HU_MAX)
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
    return volume.astype(np.float32)


def _patch_to_tensor(patch_2d, size=IMG_SIZE):
    img = torch.from_numpy(patch_2d).unsqueeze(0)  # (1, H, W)
    img = TF.resize(img, [size, size], antialias=True)
    img = img.repeat(3, 1, 1)  # (3, H, W) — replicate to 3-channel for ResNet
    return img


def _mask_to_tensor(mask_2d, size=IMG_SIZE):
    m = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0)  # (1, H, W)
    m = TF.resize(m, [size, size], antialias=True)
    return (m > 0.5).float()


def load_nodules(sample_size=SAMPLE_SIZE, seed=SEED):
    """Load up to sample_size nodules from LIDC-IDRI via pylidc, or from cache."""
    cache_path = os.path.join(RESULTS_DIR, "cache.pkl")
    if os.path.exists(cache_path):
        print("Loading from cache...")
        with open(cache_path, "rb") as f:
            raw = pickle.load(f)
        nodules = []
        for s in raw:
            nodules.append({
                "patch": s["image"],
                "mask": s["mask"],
                "ts_mask": s["ts_mask"],
                "label": s["label"],
                "nodule_mask_full": np.zeros((512, 512), dtype=np.float32),
                "patient_id": s["patient_id"],
            })
        print(f"Loaded {len(nodules)} samples from cache")
        return nodules

    # No cache - go back to pylidc loading
    rng = np.random.default_rng(seed)
    scans = pl.query(pl.Scan).all()
    rng.shuffle(scans)

    nodules = []
    for scan in scans:
        if len(nodules) >= sample_size:
            break
        try:
            vol = scan.to_volume()
        except Exception:
            continue

        lung_mask_3d = _get_lung_mask(vol)

        nods = scan.cluster_annotations()
        for ann_group in nods:
            if len(nodules) >= sample_size:
                break
            if len(ann_group) < 2:
                continue

            mean_score = np.mean([a.malignancy for a in ann_group])
            if mean_score == 3.0:
                continue  # skip ambiguous
            label_val = 1 if mean_score > 3.0 else 0

            ann = ann_group[0]
            bbox = ann.bbox()
            cx = (bbox[0].start + bbox[0].stop) // 2
            cy = (bbox[1].start + bbox[1].stop) // 2
            cz = (bbox[2].start + bbox[2].stop) // 2

            half = 32  # 64x64 patch around nodule centre
            z_idx = cz
            x0, x1 = max(0, cx - half), min(vol.shape[0], cx + half)
            y0, y1 = max(0, cy - half), min(vol.shape[1], cy + half)

            patch = vol[x0:x1, y0:y1, z_idx]
            mask_patch = lung_mask_3d[x0:x1, y0:y1, z_idx]

            if patch.shape[0] < 4 or patch.shape[1] < 4:
                continue

            # Build nodule annotation mask (full 512x512 slice) for evaluation
            nodule_mask_full = np.zeros((512, 512), dtype=np.float32)
            try:
                # mark nodule region on full slice
                nodule_mask_full[x0:x1, y0:y1] = 1.0
            except Exception:
                pass

            nodules.append({
                "patch": _normalise_hu(patch),
                "mask": mask_patch,
                "label": label_val,
                "nodule_mask_full": nodule_mask_full,
                "patient_id": scan.patient_id,
            })

    return nodules


def patient_split(nodules, seed=SEED):
    """Patient-level 70/15/15 train/val/test split."""
    rng = np.random.default_rng(seed)
    patients = list({n["patient_id"] for n in nodules})
    rng.shuffle(patients)

    n = len(patients)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train + n_val])
    test_p = set(patients[n_train + n_val:])

    train = [n for n in nodules if n["patient_id"] in train_p]
    val = [n for n in nodules if n["patient_id"] in val_p]
    test = [n for n in nodules if n["patient_id"] in test_p]
    return train, val, test


class LIDCDataset(Dataset):
    def __init__(self, nodule_list, use_ts_mask=False):
        if use_ts_mask:
            nodule_list = [n for n in nodule_list if n.get("ts_mask") is not None]
        self.nodules = nodule_list
        self.labels = [n["label"] for n in nodule_list]
        self.use_ts_mask = use_ts_mask

    def __len__(self):
        return len(self.nodules)

    def __getitem__(self, idx):
        item = self.nodules[idx]
        image = _patch_to_tensor(item["patch"])           # (3, 224, 224)
        mask_key = "ts_mask" if self.use_ts_mask else "mask"
        mask = _mask_to_tensor(item[mask_key])            # (1, 224, 224)
        label_t = torch.tensor(item["label"], dtype=torch.float32)

        # Full-resolution nodule mask for evaluation (1, 512, 512)
        nodule_mask = torch.from_numpy(item["nodule_mask_full"]).unsqueeze(0)

        return image, mask, label_t, nodule_mask

    def class_weights(self):
        """Compute pos_weight for BCEWithLogitsLoss."""
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(1.0)
        return torch.tensor(n_neg / n_pos, dtype=torch.float32)
