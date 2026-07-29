import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
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


def _dilate_binary_tensor(mask, iterations):
    output = mask
    for _ in range(max(int(iterations), 0)):
        output = F.max_pool2d(output.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    return (output > 0.5).float()


def _blur_tensor_map(mask, sigma):
    sigma = float(sigma)
    if sigma <= 0:
        return mask.clamp(0.0, 1.0)
    kernel_size = int(max(3, round(sigma * 6) // 2 * 2 + 1))
    blurred = TF.gaussian_blur(mask, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma])
    peak = blurred.max()
    if float(peak) > 0:
        blurred = blurred / (peak + 1e-8)
    return blurred.clamp(0.0, 1.0)


def _augment_image_and_mask(
    image,
    mask,
    prior=None,
    prior_is_binary=False,
    strength="standard",
):
    """Apply conservative CT-safe training augmentation.

    The geometry is shared between image and mask so attention experiments can
    still use the returned mask correctly. Intensity jitter is deliberately
    small and grayscale-only because CT values carry physical meaning.
    """
    if strength == "standard":
        unchanged_p = 0.25
        rotation = 7.0
        translate_frac = 0.04
        scale_min, scale_max = 0.96, 1.04
        contrast_min, contrast_max = 0.92, 1.08
        brightness_abs = 0.03
        noise_sigma = 0.01
        noise_p = 0.25
    elif strength == "medium":
        unchanged_p = 0.20
        rotation = 10.0
        translate_frac = 0.05
        scale_min, scale_max = 0.94, 1.06
        contrast_min, contrast_max = 0.90, 1.10
        brightness_abs = 0.035
        noise_sigma = 0.012
        noise_p = 0.30
    elif strength == "strong":
        unchanged_p = 0.15
        rotation = 12.0
        translate_frac = 0.06
        scale_min, scale_max = 0.92, 1.08
        contrast_min, contrast_max = 0.88, 1.12
        brightness_abs = 0.04
        noise_sigma = 0.015
        noise_p = 0.35
    else:
        raise ValueError(f"Unknown augmentation strength: {strength}")

    if torch.rand(()) < unchanged_p:
        return image, mask, prior

    if torch.rand(()) < 0.5:
        image = TF.hflip(image)
        mask = TF.hflip(mask)
        if prior is not None:
            prior = TF.hflip(prior)

    angle = float(torch.empty(()).uniform_(-rotation, rotation).item())
    translate = [
        int(torch.empty(()).uniform_(-translate_frac, translate_frac).item() * IMG_SIZE),
        int(torch.empty(()).uniform_(-translate_frac, translate_frac).item() * IMG_SIZE),
    ]
    scale = float(torch.empty(()).uniform_(scale_min, scale_max).item())
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
    if prior is not None:
        prior = TF.affine(
            prior,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=(
                InterpolationMode.NEAREST
                if prior_is_binary
                else InterpolationMode.BILINEAR
            ),
            fill=0.0,
        )

    contrast = float(torch.empty(()).uniform_(contrast_min, contrast_max).item())
    brightness = float(torch.empty(()).uniform_(-brightness_abs, brightness_abs).item())
    image = (image - 0.5) * contrast + 0.5 + brightness

    if torch.rand(()) < noise_p:
        image = image + torch.randn_like(image) * noise_sigma

    image = image.clamp(0.0, 1.0)
    mask = (mask > 0.5).float()
    if prior is not None:
        prior = (prior > 0.5).float() if prior_is_binary else prior.clamp(0.0, 1.0)
    return image, mask, prior


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
            "possible_nodule_mask",
            "possible_nodule_prior",
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
        "possible_nodule_mask",
        "possible_nodule_prior",
        "possible_nodule_prior_method",
        "possible_nodule_prior_params",
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
    def __init__(
        self,
        nodule_list,
        augment=False,
        prior_mode="none",
        prior_as_channel=False,
        return_prior=False,
        prior_runtime_dilation=0,
        prior_runtime_blur_sigma=0.0,
        augment_strength="standard",
    ):
        self.nodules = nodule_list
        self.labels = [n["label"] for n in nodule_list]
        self.augment = augment
        self.prior_mode = prior_mode
        self.prior_as_channel = prior_as_channel
        self.return_prior = return_prior
        self.prior_runtime_dilation = int(prior_runtime_dilation)
        self.prior_runtime_blur_sigma = float(prior_runtime_blur_sigma)
        self.augment_strength = augment_strength
        if prior_mode not in ("none", "binary", "blurred"):
            raise ValueError(f"Unknown prior_mode: {prior_mode}")
        if prior_as_channel and prior_mode == "none":
            raise ValueError("prior_as_channel requires prior_mode binary or blurred")
        if return_prior and prior_mode == "none":
            raise ValueError("return_prior requires prior_mode binary or blurred")

    def __len__(self):
        return len(self.nodules)

    def __getitem__(self, idx):
        item = self.nodules[idx]
        image = _patch_to_tensor(item["patch"])           # (3, 224, 224)
        mask = _mask_to_tensor(item["mask"])              # (1, 224, 224)
        prior = None
        if self.prior_mode == "binary":
            prior = _mask_to_tensor(item["possible_nodule_mask"])
        elif self.prior_mode == "blurred":
            if self.prior_runtime_dilation > 0:
                prior = _mask_to_tensor(item["possible_nodule_mask"])
                prior = _dilate_binary_tensor(prior, self.prior_runtime_dilation)
                prior = _blur_tensor_map(prior, self.prior_runtime_blur_sigma)
            else:
                prior = _map_to_tensor(item["possible_nodule_prior"])
        if self.augment:
            image, mask, prior = _augment_image_and_mask(
                image,
                mask,
                prior,
                prior_is_binary=self.prior_mode == "binary",
                strength=self.augment_strength,
            )
        if self.prior_as_channel:
            image = torch.cat([image, prior], dim=0)
        label_t = torch.tensor(item["label"], dtype=torch.float32)
        if self.return_prior:
            return image, mask, prior, label_t
        return image, mask, label_t

    def class_weights(self):
        """Compute pos_weight for BCEWithLogitsLoss."""
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(1.0)
        return torch.tensor(n_neg / n_pos, dtype=torch.float32)
