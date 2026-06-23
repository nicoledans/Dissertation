import os
import pickle

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset

from config import IMG_SIZE, SEED, TRAIN_CACHE_PATH


def _patch_to_tensor(patch, size=IMG_SIZE):
    """Return a 3-channel image tensor for ResNet.

    Supported cache formats:
    - 2D image:   (H, W), repeated into three identical channels.
    - 2.5D image: (3, H, W), already stacked as below/middle/above slices.
    """
    patch = np.asarray(patch, dtype=np.float32)
    if patch.ndim == 2:
        img = torch.from_numpy(patch).unsqueeze(0)  # (1, H, W)
        img = TF.resize(img, [size, size], antialias=True)
        return img.repeat(3, 1, 1)  # (3, H, W)
    if patch.ndim == 3 and patch.shape[0] == 3:
        img = torch.from_numpy(patch)  # (3, H, W)
        return TF.resize(img, [size, size], antialias=True)
    raise ValueError(
        f"Unsupported image shape {patch.shape}; expected (H, W) or (3, H, W)."
    )


def _mask_to_tensor(mask_2d, size=IMG_SIZE):
    m = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0)  # (1, H, W)
    m = TF.resize(m, [size, size], antialias=False)
    return (m > 0.5).float()


def _map_to_tensor(map_2d, size=IMG_SIZE):
    m = torch.from_numpy(map_2d.astype(np.float32)).unsqueeze(0)
    m = TF.resize(m, [size, size], antialias=True)
    return m.clamp(0.0, 1.0).float()


def _augment_image_and_mask(image, mask):
    """Apply conservative CT-safe training augmentation.

    The geometry is shared between image and mask so attention experiments can
    still use the returned mask correctly. Intensity jitter is deliberately
    small and grayscale-only because CT values carry physical meaning.
    """
    if torch.rand(()) < 0.25:
        return image, mask

    if torch.rand(()) < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    angle = float(torch.empty(()).uniform_(-7.0, 7.0).item())
    translate = [
        int(torch.empty(()).uniform_(-0.04, 0.04).item() * IMG_SIZE),
        int(torch.empty(()).uniform_(-0.04, 0.04).item() * IMG_SIZE),
    ]
    scale = float(torch.empty(()).uniform_(0.96, 1.04).item())
    image = TF.affine(
        image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    mask = TF.affine(
        mask,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    )

    contrast = float(torch.empty(()).uniform_(0.92, 1.08).item())
    brightness = float(torch.empty(()).uniform_(-0.03, 0.03).item())
    image = (image - 0.5) * contrast + 0.5 + brightness

    if torch.rand(()) < 0.25:
        image = image + torch.randn_like(image) * 0.01

    return image.clamp(0.0, 1.0), (mask > 0.5).float()


def _load_cache(cache_path):
    cache_path = cache_path or TRAIN_CACHE_PATH
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache not found: {cache_path}. Run build_cache.py first."
        )
    print("Loading from cache...")
    with open(cache_path, "rb") as f:
        raw = pickle.load(f)

    if not isinstance(raw, list):
        raise ValueError(
            f"{cache_path} is a checkpoint, not a completed cache. "
            "Finish build_cache.py before training."
        )
    for index, sample in enumerate(raw):
        if not isinstance(sample, dict):
            raise ValueError(
                f"Cache sample {index} has type {type(sample).__name__}; expected a dictionary."
            )

    print(f"Loaded {len(raw)} samples from cache")
    return raw


def _validate_sample(sample, index, mask_key):
    required = {"image", mask_key, "label", "patient_id"}
    missing = required.difference(sample)
    if missing:
        raise ValueError(f"Cache sample {index} is missing keys: {sorted(missing)}")

    image = np.asarray(sample["image"])
    mask = np.asarray(sample[mask_key])

    valid_image_shapes = {(IMG_SIZE, IMG_SIZE), (3, IMG_SIZE, IMG_SIZE)}
    if image.shape not in valid_image_shapes:
        raise ValueError(
            f"Cache sample {index} image has shape {image.shape}; "
            f"expected {(IMG_SIZE, IMG_SIZE)} or {(3, IMG_SIZE, IMG_SIZE)}."
        )
    if mask.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(
            f"Cache sample {index} {mask_key} has shape {mask.shape}; "
            f"expected {(IMG_SIZE, IMG_SIZE)}."
        )
    if not np.isfinite(image).all():
        raise ValueError(f"Cache sample {index} image contains NaN or infinite values.")
    if not np.isfinite(mask).all():
        raise ValueError(
            f"Cache sample {index} {mask_key} contains NaN or infinite values."
        )
    if sample["label"] not in (0, 1):
        raise ValueError(
            f"Cache sample {index} has invalid label {sample['label']!r}; expected 0 or 1."
        )
    if not sample["patient_id"]:
        raise ValueError(f"Cache sample {index} has an empty patient_id.")
    if not np.any(mask):
        raise ValueError(f"Cache sample {index} has an empty {mask_key}.")


def load_nodules_hu(cache_path=None):
    """Load nodules from cache using HU lung masks."""
    raw = _load_cache(cache_path)
    for index, sample in enumerate(raw):
        _validate_sample(sample, index, "mask")

    nodules = []
    for s in raw:
        item = {
            "patch": s["image"],
            "mask": s["mask"],
            "label": s["label"],
            "patient_id": s["patient_id"],
        }
        if "candidate_mask" in s:
            item["candidate_mask"] = s["candidate_mask"]
            item["candidate_method"] = s.get("candidate_method", "precomputed")
            item["candidate_count"] = s.get("candidate_count")
            item["candidate_area_pct"] = s.get("candidate_area_pct")
        for key in (
            "soft_blob_map",
            "solid_like_map",
            "subsolid_like_map",
            "vesselness_map",
            "search_mask",
        ):
            if key in s:
                item[key] = s[key]
        if "soft_blob_method" in s:
            item["soft_blob_method"] = s["soft_blob_method"]
        if "soft_blob_params" in s:
            item["soft_blob_params"] = s["soft_blob_params"]
        nodules.append(item)
    return nodules


def _copy_optional_maps(source, target):
    for key in (
        "candidate_mask",
        "candidate_method",
        "candidate_count",
        "candidate_area_pct",
        "soft_blob_map",
        "solid_like_map",
        "subsolid_like_map",
        "vesselness_map",
        "search_mask",
        "soft_blob_method",
        "soft_blob_params",
    ):
        if key in source:
            target[key] = source[key]


def load_nodules_ts(cache_path=None):
    """Load nodules from cache using TotalSegmentator masks.

    Raises ValueError if the cache contains no TS masks (built with --no-ts).
    """
    raw = _load_cache(cache_path)
    available = [
        (index, sample)
        for index, sample in enumerate(raw)
        if sample.get("ts_mask") is not None
    ]
    dropped = len(raw) - len(available)

    for index, sample in available:
        _validate_sample(sample, index, "ts_mask")

    nodules = []
    for _, s in available:
        item = {
            "patch": s["image"],
            "mask": s["ts_mask"],
            "label": s["label"],
            "patient_id": s["patient_id"],
        }
        _copy_optional_maps(s, item)
        nodules.append(item)
    if not nodules:
        raise ValueError(
            "No TotalSegmentator masks found in cache. "
            "Run build_cache.py without --no-ts."
        )
    if dropped:
        print(f"Dropped {dropped}/{len(raw)} samples without TotalSegmentator masks.")
    return nodules


def patient_split(nodules, seed=SEED):
    """Patient-level 70/15/15 train/val/test split."""
    rng = np.random.default_rng(seed)
    patients = sorted({n["patient_id"] for n in nodules})
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
    def __init__(self, nodule_list, augment=False):
        self.nodules = nodule_list
        self.labels = [n["label"] for n in nodule_list]
        self.augment = augment

    def __len__(self):
        return len(self.nodules)

    def __getitem__(self, idx):
        item = self.nodules[idx]
        image = _patch_to_tensor(item["patch"])           # (3, 224, 224)
        mask = _mask_to_tensor(item["mask"])              # (1, 224, 224)
        if self.augment:
            image, mask = _augment_image_and_mask(image, mask)
        label_t = torch.tensor(item["label"], dtype=torch.float32)
        return image, mask, label_t

    def class_weights(self):
        """Compute pos_weight for BCEWithLogitsLoss."""
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(1.0)
        return torch.tensor(n_neg / n_pos, dtype=torch.float32)
