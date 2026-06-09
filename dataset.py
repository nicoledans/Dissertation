import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from config import IMG_SIZE, SEED, TRAIN_CACHE_PATH


def _patch_to_tensor(patch_2d, size=IMG_SIZE):
    img = torch.from_numpy(patch_2d).unsqueeze(0)  # (1, H, W)
    img = TF.resize(img, [size, size], antialias=True)
    img = img.repeat(3, 1, 1)  # (3, H, W) — replicate to 3-channel for ResNet
    return img


def _mask_to_tensor(mask_2d, size=IMG_SIZE):
    m = torch.from_numpy(mask_2d.astype(np.float32)).unsqueeze(0)  # (1, H, W)
    m = TF.resize(m, [size, size], antialias=False)
    return (m > 0.5).float()


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

    if image.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(
            f"Cache sample {index} image has shape {image.shape}; "
            f"expected {(IMG_SIZE, IMG_SIZE)}."
        )
    if mask.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(
            f"Cache sample {index} {mask_key} has shape {mask.shape}; "
            f"expected {(IMG_SIZE, IMG_SIZE)}."
        )
    if not np.isfinite(image).all():
        raise ValueError(f"Cache sample {index} image contains NaN or infinite values.")
    if not np.isfinite(mask).all():
        raise ValueError(f"Cache sample {index} {mask_key} contains NaN or infinite values.")
    if sample["label"] not in (0, 1):
        raise ValueError(f"Cache sample {index} has invalid label {sample['label']!r}; expected 0 or 1.")
    if not sample["patient_id"]:
        raise ValueError(f"Cache sample {index} has an empty patient_id.")
    if not np.any(mask):
        raise ValueError(f"Cache sample {index} has an empty {mask_key}.")


def load_nodules_hu(cache_path=None):
    """Load nodules from cache using HU lung masks."""
    raw = _load_cache(cache_path)
    for index, sample in enumerate(raw):
        _validate_sample(sample, index, "mask")

    return [
        {
            "patch": s["image"],
            "mask": s["mask"],
            "label": s["label"],
            "patient_id": s["patient_id"],
        }
        for s in raw
    ]


def load_nodules_ts(cache_path=None):
    """Load nodules from cache using TotalSegmentator masks.

    Raises ValueError if the cache contains no TS masks (built with --no-ts).
    """
    raw = _load_cache(cache_path)
    available = [(index, sample) for index, sample in enumerate(raw) if sample.get("ts_mask") is not None]
    dropped = len(raw) - len(available)

    for index, sample in available:
        _validate_sample(sample, index, "ts_mask")

    nodules = [
        {
            "patch": s["image"],
            "mask": s["ts_mask"],
            "label": s["label"],
            "patient_id": s["patient_id"],
        }
        for _, s in available
    ]
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
    # sorted() before shuffle so the pre-shuffle order is deterministic across runs
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
    def __init__(self, nodule_list):
        self.nodules = nodule_list
        self.labels = [n["label"] for n in nodule_list]

    def __len__(self):
        return len(self.nodules)

    def __getitem__(self, idx):
        item = self.nodules[idx]
        image = _patch_to_tensor(item["patch"])           # (3, 224, 224)
        mask = _mask_to_tensor(item["mask"])              # (1, 224, 224)
        label_t = torch.tensor(item["label"], dtype=torch.float32)
        return image, mask, label_t

    def class_weights(self):
        """Compute pos_weight for BCEWithLogitsLoss."""
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(1.0)
        return torch.tensor(n_neg / n_pos, dtype=torch.float32)
