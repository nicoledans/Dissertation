"""Train anatomy-supervised classification with background-appearance consistency.

Model 5 (explanation consistency): --alpha > 0 --beta > 0 --gamma > 0
Model 4 (combined):    --alpha > 0 --beta > 0 --gamma 0
Model 3 (later ablation): --alpha 0 --beta > 0

The intervention preserves lung pixels exactly while mildly perturbing the
appearance outside the lungs. No anatomy is replaced or removed.
"""

import argparse
import csv
import os
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from config import (
    ALPHA,
    BATCH_SIZE,
    BETA,
    EPOCHS,
    LR,
    RESULTS_DIR,
    SEED,
    TRAIN_CACHE_PATH,
)
from dataset import LIDCDataset, load_nodules_hu, patient_split
from model import NoduleClassifier


BLANK_CAM_EPS = 1e-8
FEATHER_KERNEL = 9
INTENSITY_SCALE_RANGE = (0.90, 1.10)
INTENSITY_OFFSET_RANGE = (-0.03, 0.03)
NOISE_SIGMA_RANGE = (0.005, 0.02)
BLUR_SIGMA_RANGE = (0.4, 0.8)
PERTURBATION_NAMES = ("intensity", "noise", "blur")


def _make_run_id():
    return datetime.now().strftime("intervention_%Y%m%d_%H%M%S")


def _seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _classification_metrics(labels, probabilities):
    predictions = [int(probability >= 0.5) for probability in probabilities]
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(labels))),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }


def _blend_mask(masks, dtype):
    """Preserve every lung-mask pixel and feather only outward."""
    binary = masks.to(dtype=dtype)
    blurred = TF.gaussian_blur(binary, kernel_size=[FEATHER_KERNEL, FEATHER_KERNEL])
    return torch.maximum(binary, blurred).clamp(0.0, 1.0)


def _uniform_per_sample(images, low, high):
    shape = (images.shape[0], 1, 1, 1)
    return torch.empty(shape, device=images.device, dtype=images.dtype).uniform_(low, high)


def _appearance_variant(images, mode):
    """Create a mild anatomy-preserving appearance variant of each image."""
    if mode == "intensity":
        scale = _uniform_per_sample(images, *INTENSITY_SCALE_RANGE)
        offset = _uniform_per_sample(images, *INTENSITY_OFFSET_RANGE)
        return (images * scale + offset).clamp(0.0, 1.0)
    if mode == "noise":
        sigma = _uniform_per_sample(images, *NOISE_SIGMA_RANGE)
        return (images + torch.randn_like(images) * sigma).clamp(0.0, 1.0)
    if mode == "blur":
        sigma = float(torch.empty((), device=images.device).uniform_(*BLUR_SIGMA_RANGE).item())
        return TF.gaussian_blur(images, kernel_size=[5, 5], sigma=[sigma, sigma])
    raise ValueError(f"Unknown appearance perturbation: {mode}")


def _intervene(images, masks, mode="mixed"):
    """Perturb only outside-lung appearance; preserve binary lung pixels exactly."""
    if mode == "mixed":
        variants = torch.stack(
            [_appearance_variant(images, name) for name in PERTURBATION_NAMES],
            dim=0,
        )
        choices = torch.randint(len(PERTURBATION_NAMES), (images.shape[0],), device=images.device)
        batch_indices = torch.arange(images.shape[0], device=images.device)
        replacement = variants[choices, batch_indices]
    else:
        replacement = _appearance_variant(images, mode)

    blend_mask = _blend_mask(masks, images.dtype)
    intervened = images * blend_mask + replacement * (1.0 - blend_mask)
    return intervened, blend_mask


def _adaptive_margin_loss(model, logits, masks, delta):
    """Zhang-inspired confidence-adaptive predicted-class Grad-CAM margin."""
    scores = model.class_scores(logits)
    raw_cam = model.differentiable_gradcam(scores, normalise=False)
    blank = raw_cam.detach().flatten(start_dim=1).amax(dim=1) <= BLANK_CAM_EPS
    cam = model.normalise_gradcam(raw_cam)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=masks.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    mask_2d = masks.squeeze(1)
    inside_mean = (cam * mask_2d).sum(dim=(1, 2)) / (
        mask_2d.sum(dim=(1, 2)) + 1e-8
    )
    outside_mask = 1.0 - mask_2d
    outside_mean = (cam * outside_mask).sum(dim=(1, 2)) / (
        outside_mask.sum(dim=(1, 2)) + 1e-8
    )
    margin = F.relu(outside_mean - inside_mean + delta)
    confidence = 2.0 * (torch.sigmoid(logits).detach() - 0.5).abs()
    return confidence * margin, margin, confidence, blank, cam


def _same_class_gradcam(model, logits, original_logits, masks):
    """Return Grad-CAM for the original view's predicted class."""
    signs = torch.where(original_logits.detach() >= 0, 1.0, -1.0)
    raw_cam = model.differentiable_gradcam(logits * signs, normalise=False)
    cam = model.normalise_gradcam(raw_cam)
    return F.interpolate(
        cam.unsqueeze(1),
        size=masks.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)


def _inside_explanation_consistency(original_cam, perturbed_cam, masks):
    """Mean absolute Grad-CAM difference within each sample's lung mask."""
    mask_2d = masks.squeeze(1)
    per_sample = (
        (original_cam - perturbed_cam).abs() * mask_2d
    ).sum(dim=(1, 2)) / (mask_2d.sum(dim=(1, 2)) + 1e-8)
    return per_sample.mean()


def _scheduled_weight(target, epoch, start_epoch, ramp_epochs):
    """Linearly introduce a loss term, reaching its target after ramp_epochs."""
    if target == 0 or epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 1:
        return target
    progress = min(1.0, (epoch - start_epoch + 1) / ramp_epochs)
    return target * progress


def _configure_freezing(model, freeze_through):
    stages = [
        ("stem", [model.backbone.conv1, model.backbone.bn1]),
        ("layer1", [model.backbone.layer1]),
        ("layer2", [model.backbone.layer2]),
        ("layer3", [model.backbone.layer3]),
    ]
    frozen_modules = []
    if freeze_through == "none":
        return frozen_modules
    for name, modules in stages:
        for module in modules:
            module.requires_grad_(False)
            module.eval()
            frozen_modules.append(module)
        if name == freeze_through:
            break
    return frozen_modules


def _evaluate_pair(model, loader, device, mode="intensity"):
    model.eval()
    original_probs, perturbed_probs, labels = [], [], []
    original_logits, perturbed_logits = [], []

    with torch.no_grad():
        for images, masks, batch_labels in loader:
            images = images.to(device)
            masks = masks.to(device)
            perturbed, _ = _intervene(images, masks, mode=mode)

            logits_original = model(images).squeeze(1)
            model.clear_hooks()
            logits_perturbed = model(perturbed).squeeze(1)
            model.clear_hooks()

            original_logits.extend(logits_original.cpu().tolist())
            perturbed_logits.extend(logits_perturbed.cpu().tolist())
            original_probs.extend(torch.sigmoid(logits_original).cpu().tolist())
            perturbed_probs.extend(torch.sigmoid(logits_perturbed).cpu().tolist())
            labels.extend(batch_labels.tolist())

    original_metrics = _classification_metrics(labels, original_probs)
    perturbed_metrics = _classification_metrics(labels, perturbed_probs)
    original_probs = np.asarray(original_probs)
    perturbed_probs = np.asarray(perturbed_probs)
    original_logits = np.asarray(original_logits)
    perturbed_logits = np.asarray(perturbed_logits)
    original_binary = original_probs >= 0.5
    perturbed_binary = perturbed_probs >= 0.5

    return {
        **{f"original_{key}": value for key, value in original_metrics.items()},
        **{f"perturbed_{key}": value for key, value in perturbed_metrics.items()},
        "mean_abs_probability_change": float(np.mean(np.abs(original_probs - perturbed_probs))),
        "mean_logit_mse": float(np.mean((original_logits - perturbed_logits) ** 2)),
        "prediction_flip_rate": float(np.mean(original_binary != perturbed_binary)),
    }


def _evaluate_pair_seeded(model, loader, device, mode, seed):
    """Evaluate a reproducible perturbation without changing training RNG state."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    metrics = _evaluate_pair(model, loader, device, mode)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return metrics


def _gradcam_pair_stats(model, loader, device):
    model.eval()
    original_outside, perturbed_outside = [], []
    inside_explanation_difference = []
    original_blank = perturbed_blank = 0

    with torch.enable_grad():
        for images, masks, _labels in loader:
            images = images.to(device)
            masks = masks.to(device)
            perturbed, _ = _intervene(images, masks, mode="intensity")

            paired_cams = []
            original_signs = None
            for view, values, name in (
                (images, original_outside, "original"),
                (perturbed, perturbed_outside, "perturbed"),
            ):
                model.zero_grad(set_to_none=True)
                model.clear_hooks()
                logits = model(view).squeeze(1)
                if original_signs is None:
                    original_signs = torch.where(logits.detach() >= 0, 1.0, -1.0)
                (logits * original_signs).sum().backward()
                raw_cam = model.get_gradcam(normalise=False)
                blanks = int(
                    (raw_cam.flatten(start_dim=1).amax(dim=1) <= BLANK_CAM_EPS).sum().item()
                )
                if name == "original":
                    original_blank += blanks
                else:
                    perturbed_blank += blanks

                cam = model.normalise_gradcam(raw_cam)
                cam = F.interpolate(
                    cam.unsqueeze(1),
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                paired_cams.append(cam.detach())
                mask_2d = masks.squeeze(1)
                outside = (cam * (1.0 - mask_2d)).sum(dim=(1, 2)) / (
                    cam.sum(dim=(1, 2)) + 1e-8
                )
                values.extend(outside.detach().cpu().tolist())
                model.clear_hooks()

            mask_2d = masks.squeeze(1)
            difference = (
                (paired_cams[0] - paired_cams[1]).abs() * mask_2d
            ).sum(dim=(1, 2)) / (mask_2d.sum(dim=(1, 2)) + 1e-8)
            inside_explanation_difference.extend(difference.cpu().tolist())

    return {
        "original_gradcam_outside": float(np.mean(original_outside)),
        "original_gradcam_inside": 1.0 - float(np.mean(original_outside)),
        "perturbed_gradcam_outside": float(np.mean(perturbed_outside)),
        "perturbed_gradcam_inside": 1.0 - float(np.mean(perturbed_outside)),
        "inside_gradcam_mean_abs_difference": float(
            np.mean(inside_explanation_difference)
        ),
        "original_blank_rate": original_blank / max(len(original_outside), 1),
        "perturbed_blank_rate": perturbed_blank / max(len(perturbed_outside), 1),
    }


def _gradcam_pair_stats_seeded(model, loader, device, seed):
    """Measure reproducible paired Grad-CAM statistics without changing training RNG."""
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    metrics = _gradcam_pair_stats(model, loader, device)
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state_all(cuda_state)
    return metrics


def _write_run_info(run_dir, args, cache_path, train_nods, val_nods, test_nods):
    def summary(samples):
        patients = {sample["patient_id"] for sample in samples}
        labels = Counter(sample["label"] for sample in samples)
        return len(samples), len(patients), labels[0], labels[1], patients

    train = summary(train_nods)
    val = summary(val_nods)
    test = summary(test_nods)
    overlap = train[4] & val[4] or train[4] & test[4] or val[4] & test[4]
    if overlap:
        raise RuntimeError(f"Patient leakage detected across splits: {sorted(overlap)}")

    if args.gamma > 0:
        mode = "prediction-and-explanation-consistency"
    elif args.alpha > 0:
        mode = "combined"
    else:
        mode = "consistency-only"
    lines = [
        "=== OUTSIDE-LUNG APPEARANCE-CONSISTENCY TRAINING ===",
        f"Mode: {mode}",
        f"Cache: {os.path.abspath(cache_path)}",
        "Intervention: mild random appearance perturbation outside the lung mask",
        "Training perturbations: per-sample random intensity, Gaussian noise, or Gaussian blur",
        f"Intensity scale range: {INTENSITY_SCALE_RANGE}",
        f"Intensity offset range: {INTENSITY_OFFSET_RANGE}",
        f"Noise sigma range: {NOISE_SIGMA_RANGE}",
        f"Blur sigma range: {BLUR_SIGMA_RANGE}",
        "Boundary handling: binary lung pixels exact; feathering extends only outward",
        "Grad-CAM supervision mask: original binary lung mask",
        "Classification: weighted BCE averaged across original and perturbed views",
        "Attention: Zhang-inspired confidence-adaptive predicted-class Grad-CAM margin",
        f"Attention margin delta: {args.delta}",
        "Consistency: MSE between original and perturbed logits",
        "Explanation consistency: mean absolute same-class Grad-CAM difference inside lung mask",
        "Paired forwards: identical dropout randomness to isolate the background intervention",
        "Checkpoint selection: highest original-image validation AUC",
        "Test evaluated during training: no",
        f"Alpha: {args.alpha}",
        f"Beta: {args.beta}",
        f"Gamma: {args.gamma}",
        f"Attention/prediction-consistency start epoch: {args.guidance_start_epoch}",
        f"Explanation-consistency start epoch: {args.explanation_start_epoch}",
        f"Loss-weight ramp epochs: {args.ramp_epochs}",
        "Before guidance starts: original-view BCE only",
        "After guidance starts: BCE averaged across original and intervened views",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"Optimizer: AdamW, learning rate {args.lr}, weight decay {args.weight_decay}",
        f"Initial checkpoint: {os.path.abspath(args.init_checkpoint) if args.init_checkpoint else 'ImageNet initialization'}",
        f"Freeze through: {args.freeze_through}",
        "Frozen modules remain in eval mode so BatchNorm statistics do not update",
        f"Patient split seed: {SEED}",
        f"Training seed: {args.training_seed}",
        "",
        f"Train: {train[0]} samples, {train[1]} patients, {train[2]} benign, {train[3]} malignant",
        f"Validation: {val[0]} samples, {val[1]} patients, {val[2]} benign, {val[3]} malignant",
        f"Test held out: {test[0]} samples, {test[1]} patients, {test[2]} benign, {test[3]} malignant",
        "Patient overlap across splits: none",
    ]
    with open(os.path.join(run_dir, "intervention_info.txt"), "w") as file:
        file.write("\n".join(lines) + "\n")


def _write_metrics(path, title, metrics):
    lines = [title]
    for key, value in metrics.items():
        lines.append(f"{key}: {value:.6f}")
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def _save_intervention_examples(run_dir, nodules):
    images, masks, _labels = next(
        iter(DataLoader(LIDCDataset(nodules), batch_size=min(4, len(nodules)), shuffle=False))
    )
    cpu_state = torch.get_rng_state()
    torch.manual_seed(SEED)
    intensity, _ = _intervene(images, masks, mode="intensity")
    noise, _ = _intervene(images, masks, mode="noise")
    blur, _ = _intervene(images, masks, mode="blur")
    torch.set_rng_state(cpu_state)
    rows = images.shape[0]
    fig, axes = plt.subplots(rows, 5, figsize=(17.5, 3.5 * rows))
    if rows == 1:
        axes = axes[np.newaxis, :]
    for index in range(rows):
        panels = (
            (images[index, 0], "Original"),
            (masks[index, 0], "Binary lung mask"),
            (intensity[index, 0], "Outside-lung intensity"),
            (noise[index, 0], "Outside-lung noise"),
            (blur[index, 0], "Outside-lung blur"),
        )
        for axis, (panel, title) in zip(axes[index], panels):
            axis.imshow(panel.numpy(), cmap="gray", vmin=0, vmax=1)
            axis.set_title(title)
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "intervention_examples.png"), dpi=150)
    plt.close(fig)


def _save_training_summary(run_dir, rows, initial_validation):
    """Save a compact visual summary of AUC, robustness, and training losses."""
    epochs = [0] + [int(row["epoch"]) for row in rows]
    original_auc = [initial_validation["original_auc"]] + [
        row["original_auc"] for row in rows
    ]
    perturbed_auc = [initial_validation["perturbed_auc"]] + [
        row["perturbed_auc"] for row in rows
    ]
    probability_change = [initial_validation["mean_abs_probability_change"]] + [
        row["mean_abs_probability_change"] for row in rows
    ]
    flip_rate = [initial_validation["prediction_flip_rate"] * 100] + [
        row["prediction_flip_rate"] * 100 for row in rows
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(epochs, original_auc, marker="o", label="Original")
    axes[0].plot(epochs, perturbed_auc, marker="o", label="Intervened")
    axes[0].set(title="Validation AUC", xlabel="Fine-tuning epoch", ylabel="AUC")
    axes[0].legend()

    axes[1].plot(epochs, probability_change, marker="o", label="Mean probability change")
    axes[1].set(
        title="Prediction Sensitivity",
        xlabel="Fine-tuning epoch",
        ylabel="Absolute probability change",
    )
    secondary = axes[1].twinx()
    secondary.plot(epochs, flip_rate, marker="s", color="tab:red", label="Flip rate")
    secondary.set_ylabel("Prediction flip rate (%)")

    trained_epochs = [int(row["epoch"]) for row in rows]
    axes[2].plot(
        trained_epochs,
        [row["train_classification"] for row in rows],
        marker="o",
        label="Classification",
    )
    axes[2].plot(
        trained_epochs,
        [row["train_attention"] for row in rows],
        marker="o",
        label="Adaptive margin",
    )
    axes[2].plot(
        trained_epochs,
        [row["train_consistency"] for row in rows],
        marker="o",
        label="Consistency",
    )
    if any(row.get("train_explanation_consistency", 0.0) > 0 for row in rows):
        axes[2].plot(
            trained_epochs,
            [row["train_explanation_consistency"] for row in rows],
            marker="o",
            label="Explanation consistency",
        )
    axes[2].set(title="Training Loss Terms", xlabel="Fine-tuning epoch", ylabel="Loss")
    axes[2].legend()

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "intervention_training_summary.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Train outside-lung appearance consistency with optional Grad-CAM supervision."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--beta", type=float, default=BETA)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="Weight for inside-lung Grad-CAM consistency between paired views",
    )
    parser.add_argument(
        "--guidance-start-epoch",
        type=int,
        default=1,
        help="First epoch for paired-view BCE, attention, and prediction consistency",
    )
    parser.add_argument(
        "--explanation-start-epoch",
        type=int,
        default=1,
        help="First epoch for inside-lung Grad-CAM consistency",
    )
    parser.add_argument(
        "--ramp-epochs",
        type=int,
        default=1,
        help="Epochs used to linearly ramp scheduled loss weights to their targets",
    )
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument(
        "--freeze-through",
        choices=["none", "stem", "layer1", "layer2", "layer3"],
        default="none",
    )
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--training-seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="Optional model checkpoint to fine-tune instead of starting from ImageNet",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if (
        args.guidance_start_epoch < 1
        or args.explanation_start_epoch < 1
        or args.ramp_epochs < 1
    ):
        parser.error(
            "--guidance-start-epoch, --explanation-start-epoch, and --ramp-epochs "
            "must be at least 1"
        )
    if (
        args.alpha < 0
        or args.beta < 0
        or args.gamma < 0
        or args.delta < 0
        or args.weight_decay < 0
        or args.lr <= 0
    ):
        parser.error(
            "--alpha, --beta, --gamma, --delta, and --weight-decay must be non-negative; "
            "--lr must be positive"
        )
    if args.beta == 0:
        parser.error("--beta must be greater than zero for an intervention experiment")

    run_id = args.run_id or _make_run_id()
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=False)

    _seed_everything(args.training_seed)
    nodules = load_nodules_hu(args.cache_path)
    train_nods, val_nods, test_nods = patient_split(nodules)
    _write_run_info(run_dir, args, args.cache_path, train_nods, val_nods, test_nods)
    _save_intervention_examples(run_dir, train_nods)

    train_ds = LIDCDataset(train_nods)
    drop_singleton_train_batch = len(train_ds) % args.batch_size == 1
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=drop_singleton_train_batch,
        generator=torch.Generator().manual_seed(args.training_seed),
    )
    val_batch_size = args.batch_size
    if len(val_nods) % val_batch_size == 1:
        val_batch_size = max(2, val_batch_size - 1)
    val_loader = DataLoader(
        LIDCDataset(val_nods),
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleClassifier().to(device)
    if args.init_checkpoint:
        if not os.path.exists(args.init_checkpoint):
            parser.error(f"Initial checkpoint not found: {args.init_checkpoint}")
        model.load_state_dict(torch.load(args.init_checkpoint, map_location=device))
        print(f"Loaded initial checkpoint: {args.init_checkpoint}")
    frozen_modules = _configure_freezing(model, args.freeze_through)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_ds.class_weights().to(device))
    optimizer = torch.optim.AdamW(
        trainable_parameters, lr=args.lr, weight_decay=args.weight_decay
    )

    checkpoint = os.path.join(run_dir, "best_intervention_model.pt")
    epoch_csv = os.path.join(run_dir, "intervention_epochs.csv")
    initial_validation = _evaluate_pair_seeded(
        model, val_loader, device, mode="intensity", seed=SEED + 100
    )
    initial_validation.update(
        _gradcam_pair_stats_seeded(model, val_loader, device, seed=SEED + 100)
    )
    _write_metrics(
        os.path.join(run_dir, "initial_validation.txt"),
        "=== INITIAL CHECKPOINT VALIDATION RESULTS ===",
        initial_validation,
    )
    best_auc = initial_validation["original_auc"]
    best_epoch = 0
    if not np.isfinite(best_auc):
        raise RuntimeError("Initial checkpoint produced no finite validation AUC.")
    torch.save(model.state_dict(), checkpoint)
    print(
        f"Preserved initial checkpoint as current best "
        f"(original validation AUC {best_auc:.4f})"
    )
    rows = []

    print(f"[INTERVENTION] Output: {run_dir}")
    print(
        f"Device: {device} | alpha={args.alpha} | beta={args.beta} | gamma={args.gamma}"
    )
    print(
        f"freeze_through={args.freeze_through} | "
        f"trainable={trainable_count:,}/{total_parameters:,} parameters"
    )
    print("Held-out test set will not be evaluated.")

    for epoch in range(1, args.epochs + 1):
        effective_alpha = _scheduled_weight(
            args.alpha, epoch, args.guidance_start_epoch, args.ramp_epochs
        )
        effective_beta = _scheduled_weight(
            args.beta, epoch, args.guidance_start_epoch, args.ramp_epochs
        )
        effective_gamma = _scheduled_weight(
            args.gamma, epoch, args.explanation_start_epoch, args.ramp_epochs
        )
        paired_training = epoch >= args.guidance_start_epoch
        print(
            f"Epoch {epoch:02d} weights | alpha={effective_alpha:.4f} | "
            f"beta={effective_beta:.4f} | gamma={effective_gamma:.4f}"
        )
        model.train()
        for module in frozen_modules:
            module.eval()
        sums = Counter()
        samples = 0

        for batch_index, (images, masks, labels) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            perturbed = None
            if paired_training:
                perturbed, _ = _intervene(images, masks, mode="mixed")

            optimizer.zero_grad()
            cpu_rng_before = torch.get_rng_state()
            cuda_rng_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            logits_original = model(images).squeeze(1)
            cpu_rng_after = torch.get_rng_state()
            cuda_rng_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            bce_original = criterion(logits_original, labels)

            if effective_alpha > 0 or effective_gamma > 0:
                adaptive, margin, confidence, blank, original_cam = _adaptive_margin_loss(
                    model, logits_original, masks, args.delta
                )
                attention_loss = (
                    adaptive.mean()
                    if effective_alpha > 0
                    else torch.zeros((), device=device)
                )
            else:
                margin = torch.zeros_like(labels)
                confidence = torch.zeros_like(labels)
                blank = torch.zeros_like(labels, dtype=torch.bool)
                attention_loss = torch.zeros((), device=device)
                original_cam = None

            if paired_training:
                # Reuse the original forward's dropout mask for the paired view so
                # consistency measures the background intervention, not dropout noise.
                torch.set_rng_state(cpu_rng_before)
                if cuda_rng_before is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_before)
                logits_perturbed = model(perturbed).squeeze(1)
                if effective_gamma > 0:
                    perturbed_cam = _same_class_gradcam(
                        model, logits_perturbed, logits_original, masks
                    )
                    explanation_consistency = _inside_explanation_consistency(
                        original_cam, perturbed_cam, masks
                    )
                else:
                    explanation_consistency = torch.zeros((), device=device)
                torch.set_rng_state(cpu_rng_after)
                if cuda_rng_after is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_after)
                bce_perturbed = criterion(logits_perturbed, labels)
                classification_loss = 0.5 * (bce_original + bce_perturbed)
                consistency_loss = F.mse_loss(logits_original, logits_perturbed)
            else:
                explanation_consistency = torch.zeros((), device=device)
                classification_loss = bce_original
                consistency_loss = torch.zeros((), device=device)
            total_loss = (
                classification_loss
                + effective_alpha * attention_loss
                + effective_beta * consistency_loss
                + effective_gamma * explanation_consistency
            )
            total_loss.backward()
            optimizer.step()
            model.clear_hooks()

            n = labels.numel()
            samples += n
            sums["classification"] += classification_loss.item() * n
            sums["attention"] += attention_loss.item() * n
            sums["consistency"] += consistency_loss.item() * n
            sums["explanation_consistency"] += explanation_consistency.item() * n
            sums["total"] += total_loss.item() * n
            sums["margin"] += margin.detach().sum().item()
            sums["confidence"] += confidence.detach().sum().item()
            sums["blank"] += int(blank.sum().item())

            print(
                f"Epoch {epoch:02d} | Batch {batch_index:04d} | "
                f"Class {classification_loss.item():.4f} | "
                f"AdaptiveMargin {attention_loss.item():.4f} | "
                f"Consistency {consistency_loss.item():.4f} | "
                f"ExplainConsistency {explanation_consistency.item():.4f} | "
                f"Total {total_loss.item():.4f}"
            )

        validation = _evaluate_pair_seeded(
            model, val_loader, device, mode="intensity", seed=SEED + 100
        )
        row = {
            "epoch": epoch,
            "effective_alpha": effective_alpha,
            "effective_beta": effective_beta,
            "effective_gamma": effective_gamma,
            **{f"train_{key}": value / max(samples, 1) for key, value in sums.items()},
            **validation,
        }
        rows.append(row)
        with open(epoch_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"Epoch {epoch:02d} | Original Val AUC {validation['original_auc']:.4f} | "
            f"Perturbed Val AUC {validation['perturbed_auc']:.4f} | "
            f"ProbChange {validation['mean_abs_probability_change']:.4f} | "
            f"FlipRate {validation['prediction_flip_rate'] * 100:.2f}%"
        )

        if np.isfinite(validation["original_auc"]) and validation["original_auc"] > best_auc:
            best_auc = validation["original_auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)
            print(f"  -> Saved best model (original validation AUC {best_auc:.4f})")

    if not os.path.exists(checkpoint):
        raise RuntimeError("No finite validation AUC was produced; no checkpoint was saved.")
    _save_training_summary(run_dir, rows, initial_validation)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    final_metrics = {}
    for index, perturbation in enumerate(PERTURBATION_NAMES):
        evaluation_seed = SEED + 100 if perturbation == "intensity" else SEED + index + 1
        final_metrics.update(
            {
                f"{perturbation}_{key}": value
                for key, value in _evaluate_pair_seeded(
                    model,
                    val_loader,
                    device,
                    perturbation,
                    seed=evaluation_seed,
                ).items()
            }
        )
    final_metrics.update(
        _gradcam_pair_stats_seeded(model, val_loader, device, seed=SEED + 100)
    )
    final_metrics.update(
        {
            "selected_epoch": best_epoch,
            "initial_original_auc": initial_validation["original_auc"],
            "selected_original_auc_change": (
                final_metrics["intensity_original_auc"]
                - initial_validation["original_auc"]
            ),
            "initial_intensity_probability_change": initial_validation[
                "mean_abs_probability_change"
            ],
            "selected_intensity_probability_change_change": (
                final_metrics["intensity_mean_abs_probability_change"]
                - initial_validation["mean_abs_probability_change"]
            ),
            "initial_intensity_flip_rate": initial_validation["prediction_flip_rate"],
            "selected_intensity_flip_rate_change": (
                final_metrics["intensity_prediction_flip_rate"]
                - initial_validation["prediction_flip_rate"]
            ),
            "initial_inside_gradcam_mean_abs_difference": initial_validation[
                "inside_gradcam_mean_abs_difference"
            ],
            "selected_inside_gradcam_mean_abs_difference_change": (
                final_metrics["inside_gradcam_mean_abs_difference"]
                - initial_validation["inside_gradcam_mean_abs_difference"]
            ),
        }
    )
    _write_metrics(
        os.path.join(run_dir, "intervention_validation.txt"),
        "=== BEST-CHECKPOINT VALIDATION RESULTS ===",
        final_metrics,
    )
    model.remove_hooks()
    print(f"\nBest model saved -> {checkpoint}")
    print("The held-out test set remains untouched.")


if __name__ == "__main__":
    main()
