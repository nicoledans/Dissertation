"""Train Grad-CAM attention losses with weak soft-blob guidance."""

import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from config import BATCH_SIZE, EPOCHS, LR, RESULTS_DIR, SEED, TRAIN_CACHE_PATH
from dataset import (
    _map_to_tensor,
    _mask_to_tensor,
    _patch_to_tensor,
    load_nodules_hu,
    load_nodules_ts,
    patient_split,
)
from lidc_matching import _eligible_candidates, _image_hash
from model import NoduleClassifier


BLANK_CAM_EPS = 1e-8


def _parse_values(value, name, require_zero=False):
    try:
        values = sorted({float(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from exc
    if not values or any(not np.isfinite(item) or item < 0 for item in values):
        raise argparse.ArgumentTypeError(f"{name} must contain finite non-negative values")
    if require_zero and 0.0 not in values:
        raise argparse.ArgumentTypeError(f"{name} must include 0.0 as the reference")
    return values


def _tag(value):
    return f"{value:.10g}".replace(".", "p")


class SoftBlobDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.labels = [sample["label"] for sample in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = _patch_to_tensor(sample["patch"])
        lung = _mask_to_tensor(sample["mask"])
        blob = _map_to_tensor(sample["soft_blob_map"])
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return image, lung, blob, label

    def class_weights(self):
        pos = max(sum(self.labels), 1)
        neg = max(len(self.labels) - sum(self.labels), 1)
        return torch.tensor([neg / pos], dtype=torch.float32)


def _validate_soft_blob_samples(samples):
    missing = [idx for idx, sample in enumerate(samples) if "soft_blob_map" not in sample]
    if missing:
        raise ValueError(
            f"{len(missing)} samples have no soft_blob_map. "
            "Run build_soft_blob_cache.py first."
        )


def _metrics(labels, probabilities):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    predictions = probabilities >= 0.5
    try:
        auc = float(roc_auc_score(labels, probabilities))
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "accuracy": float(np.mean(predictions == labels)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }


def _evaluate_classification(model, samples, criterion, device, batch_size):
    loader = DataLoader(SoftBlobDataset(samples), batch_size=batch_size, shuffle=False)
    probabilities, labels = [], []
    loss_sum = 0.0
    model.eval()
    with torch.no_grad():
        for images, _lung, _blob, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(images).squeeze(1)
            loss_sum += criterion(logits, batch_labels).item() * batch_labels.numel()
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
            model.clear_hooks()
    return {
        "loss": loss_sum / max(len(labels), 1),
        **_metrics(labels, probabilities),
        "labels": labels,
        "probabilities": probabilities,
    }


def _loss_parts(
    model,
    logits,
    labels,
    lung_masks,
    blob_maps,
    delta,
    min_attention_weight,
    target_overlap,
    attention_loss,
    nonblob_cost,
    blob_overlap_power,
    blob_hinge,
    blob_correctness_weight,
):
    raw_cam = model.differentiable_gradcam(
        model.class_scores(logits), normalise=False
    )
    blank = raw_cam.detach().flatten(start_dim=1).amax(dim=1) <= BLANK_CAM_EPS
    cam = model.normalise_gradcam(raw_cam)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=lung_masks.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    lung = lung_masks.squeeze(1)
    blob = blob_maps.squeeze(1)
    outside = 1.0 - lung
    inside_mean = (cam * lung).sum(dim=(1, 2)) / (lung.sum(dim=(1, 2)) + 1e-8)
    outside_mean = (cam * outside).sum(dim=(1, 2)) / (outside.sum(dim=(1, 2)) + 1e-8)
    margin = F.relu(outside_mean - inside_mean + delta)

    probability = torch.sigmoid(logits).detach()
    labels_detached = labels.detach().to(dtype=probability.dtype)
    correct_class_probability = torch.where(
        labels_detached > 0.5,
        probability,
        1.0 - probability,
    )
    attention_weight = (
        min_attention_weight
        + (1.0 - min_attention_weight) * correct_class_probability
    ).detach()
    adaptive = attention_weight * margin

    cam_powered = cam.clamp_min(0.0).pow(float(blob_overlap_power))
    cam_mass = cam / (cam.sum(dim=(1, 2), keepdim=True) + 1e-8)
    hot_cam_mass = cam_powered / (cam_powered.sum(dim=(1, 2), keepdim=True) + 1e-8)
    blob_overlap = (hot_cam_mass * blob).sum(dim=(1, 2))
    standard_blob_overlap = (cam_mass * blob).sum(dim=(1, 2))
    blob_gap = F.relu(float(target_overlap) - blob_overlap)
    blob_hinge_loss = blob_gap.pow(2.0) if blob_hinge == "squared" else blob_gap
    blob_weight = attention_weight if blob_correctness_weight else torch.ones_like(blob_hinge_loss)
    weighted_blob_hinge = blob_weight * blob_hinge_loss
    lung_attention = (cam_mass * lung).sum(dim=(1, 2))
    outside_mass = (cam_mass * outside).sum(dim=(1, 2))
    inside_nonblob = lung * (1.0 - blob)
    inside_nonblob_mass = (cam_mass * inside_nonblob).sum(dim=(1, 2))
    high_blob_attention = (cam_mass * (blob >= 0.50).to(cam.dtype)).sum(dim=(1, 2))

    if attention_loss == "zhang-margin":
        attention = adaptive
        blob_loss = weighted_blob_hinge
    elif attention_loss == "outside-tiered":
        attention = outside_mass + float(nonblob_cost) * inside_nonblob_mass
        blob_loss = torch.zeros_like(blob_overlap)
    elif attention_loss == "outside-blob-hinge":
        attention = outside_mass
        blob_loss = weighted_blob_hinge
    elif attention_loss == "outside-blob-reward":
        attention = outside_mass
        blob_loss = -blob_overlap
    else:
        raise ValueError(f"Unknown attention loss mode: {attention_loss}")

    return {
        "attention": attention,
        "adaptive": adaptive,
        "margin": margin,
        "blob": blob_loss,
        "blob_hinge": blob_hinge_loss,
        "blob_overlap": blob_overlap,
        "standard_blob_overlap": standard_blob_overlap,
        "blob_gap": blob_gap,
        "blob_weight": blob_weight,
        "lung_attention": lung_attention,
        "outside_mass": outside_mass,
        "inside_nonblob_mass": inside_nonblob_mass,
        "high_blob_attention": high_blob_attention,
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "correct_class_probability": correct_class_probability.detach(),
        "attention_weight": attention_weight,
        "blank": blank,
    }


def _effective_epoch_weights(epoch, alpha, beta, args):
    """Return the alpha/beta actually used this epoch.

    The outside-then-blob curriculum is intentionally simple:
    1. Start with outside-lung Grad-CAM cleanup only.
    2. Add a small blob hinge.
    3. Finish with the requested final alpha/beta values.
    """
    if args.curriculum != "outside-then-blob":
        return alpha, beta, "fixed"

    phase1_end = args.curriculum_phase1_epochs
    phase2_end = phase1_end + args.curriculum_phase2_epochs
    if epoch <= phase1_end:
        return args.curriculum_phase1_alpha, 0.0, "phase1_outside_only"
    if epoch <= phase2_end:
        return args.curriculum_phase2_alpha, args.curriculum_phase2_beta, "phase2_gentle_blob"
    return alpha, beta, "phase3_final"


def _evaluate_alignment(
    model,
    samples,
    device,
    target_overlap,
    delta,
    min_attention_weight,
    blob_overlap_power=1.0,
    blob_hinge="linear",
    blob_correctness_weight=False,
):
    """Evaluate Grad-CAM alignment without building second-order training graphs."""
    loader = DataLoader(SoftBlobDataset(samples), batch_size=1, shuffle=False)
    values = Counter()
    n = 0
    model.eval()
    with torch.enable_grad():
        for images, lung, blob, labels in loader:
            images = images.to(device)
            lung = lung.to(device)
            blob = blob.to(device)
            labels = labels.to(device)
            model.zero_grad(set_to_none=True)
            model.clear_hooks()
            logits = model(images).squeeze(1)
            model.class_scores(logits).sum().backward()
            raw_cam = model.get_gradcam(normalise=False)
            blank = raw_cam.detach().flatten(start_dim=1).amax(dim=1) <= BLANK_CAM_EPS
            cam = model.normalise_gradcam(raw_cam)
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=lung.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            lung_2d = lung.squeeze(1)
            blob_2d = blob.squeeze(1)
            outside = 1.0 - lung_2d
            inside_mean = (cam * lung_2d).sum(dim=(1, 2)) / (lung_2d.sum(dim=(1, 2)) + 1e-8)
            outside_mean = (cam * outside).sum(dim=(1, 2)) / (outside.sum(dim=(1, 2)) + 1e-8)
            margin = F.relu(outside_mean - inside_mean + delta)
            probability = torch.sigmoid(logits).detach()
            labels_detached = labels.detach().to(dtype=probability.dtype)
            correct_class_probability = torch.where(
                labels_detached > 0.5,
                probability,
                1.0 - probability,
            )
            attention_weight = (
                min_attention_weight
                + (1.0 - min_attention_weight) * correct_class_probability
            ).detach()
            cam_mass = cam / (cam.sum(dim=(1, 2), keepdim=True) + 1e-8)
            cam_powered = cam.clamp_min(0.0).pow(float(blob_overlap_power))
            hot_cam_mass = cam_powered / (cam_powered.sum(dim=(1, 2), keepdim=True) + 1e-8)
            blob_overlap = (hot_cam_mass * blob_2d).sum(dim=(1, 2))
            standard_blob_overlap = (cam_mass * blob_2d).sum(dim=(1, 2))
            blob_gap = F.relu(float(target_overlap) - blob_overlap)
            blob_loss = blob_gap.pow(2.0) if blob_hinge == "squared" else blob_gap
            blob_weight = attention_weight if blob_correctness_weight else torch.ones_like(blob_loss)
            weighted_blob_loss = blob_weight * blob_loss
            lung_attention = (cam_mass * lung_2d).sum(dim=(1, 2))
            outside_mass = (cam_mass * outside).sum(dim=(1, 2))
            inside_nonblob_mass = (cam_mass * lung_2d * (1.0 - blob_2d)).sum(dim=(1, 2))
            high_blob_attention = (cam_mass * (blob_2d >= 0.50).to(cam.dtype)).sum(dim=(1, 2))

            values["blob_overlap"] += float(blob_overlap.detach().mean().cpu())
            values["standard_blob_overlap"] += float(standard_blob_overlap.detach().mean().cpu())
            values["blob"] += float(weighted_blob_loss.detach().mean().cpu())
            values["blob_gap"] += float(blob_gap.detach().mean().cpu())
            values["blob_weight"] += float(blob_weight.detach().mean().cpu())
            values["lung_attention"] += float(lung_attention.detach().mean().cpu())
            values["outside_mass"] += float(outside_mass.detach().mean().cpu())
            values["inside_nonblob_mass"] += float(inside_nonblob_mass.detach().mean().cpu())
            values["high_blob_attention"] += float(high_blob_attention.detach().mean().cpu())
            values["margin"] += float(margin.detach().mean().cpu())
            values["blank"] += int(blank.sum().item())
            n += 1
            model.clear_hooks()
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {key: value / max(n, 1) for key, value in values.items()} | {"n": n}


def _hashable_center_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        return image[1]
    return image


def _annotation_lookup_for_patient(patient_id):
    if not hasattr(_annotation_lookup_for_patient, "_cache"):
        _annotation_lookup_for_patient._cache = {}
    cache = _annotation_lookup_for_patient._cache
    if patient_id in cache:
        return cache[patient_id]

    candidates = []
    try:
        import pylidc as pl

        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        for scan in scans:
            candidates.extend(_eligible_candidates(scan))
    except Exception as error:
        print(f"  WARNING: could not reconstruct LIDC contours for {patient_id}: {error}")

    lookup = defaultdict(list)
    for candidate in candidates:
        lookup[(int(candidate["label"]), candidate["hash"])].append(candidate)
    cache[patient_id] = lookup
    return lookup


def _annotation_match_for_sample(sample):
    try:
        image_hash = _image_hash(_hashable_center_image(sample["patch"]))
        key = (int(sample["label"]), image_hash)
        matches = _annotation_lookup_for_patient(sample["patient_id"]).get(key, [])
        return matches[0] if matches else None
    except Exception as error:
        print(f"  WARNING: contour match failed for {sample.get('patient_id', 'unknown')}: {error}")
        return None


def _draw_contours(axis, majority=None, union=None):
    if union is not None and np.any(union):
        axis.contour(union.astype(float), levels=[0.5], colors=["deepskyblue"], linewidths=0.8)
    if majority is not None and np.any(majority):
        axis.contour(majority.astype(float), levels=[0.5], colors=["lime"], linewidths=1.0)


def _save_gradcam_examples(model, samples, device, run_dir, filename="gradcam_examples_soft_blob.png"):
    """Save CT, masks, audit-only contours, soft-blob map, Grad-CAM, and overlay."""
    if not samples:
        print("No samples available - skipping Grad-CAM visualisation")
        return

    dataset = SoftBlobDataset(samples)
    n_rows = min(4, len(samples))
    fig, axes = plt.subplots(n_rows, 6, figsize=(21, 3.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    model.eval()
    for collected in range(n_rows):
        sample = samples[collected]
        image, lung, blob, label_tensor = dataset[collected]
        images = image.unsqueeze(0).to(device)
        lung = lung.unsqueeze(0).to(device)
        blob = blob.unsqueeze(0).to(device)

        model.zero_grad(set_to_none=True)
        model.clear_hooks()
        logits = model(images).squeeze(1)
        probability = torch.sigmoid(logits).detach().item()
        prediction = int(probability >= 0.5)
        model.class_scores(logits).sum().backward()
        cam = model.get_gradcam()
        cam_up = F.interpolate(
            cam.unsqueeze(1),
            size=lung.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze().detach().cpu().numpy()

        image_np = images[0, 0].detach().cpu().numpy()
        lung_np = lung[0, 0].detach().cpu().numpy()
        blob_np = blob[0, 0].detach().cpu().numpy()
        label = int(label_tensor.item())
        cam_mass = cam_up / (cam_up.sum() + 1e-8)
        lung_attention = float((cam_mass * (lung_np > 0.5)).sum() * 100.0)
        blob_overlap = float((cam_mass * blob_np).sum())
        contour_match = _annotation_match_for_sample(sample)
        majority = np.asarray(contour_match["majority"]).astype(bool) if contour_match else None
        union = np.asarray(contour_match["union"]).astype(bool) if contour_match else None

        ax = axes[collected]
        ax[0].imshow(image_np, cmap="gray")
        ax[0].set_title(f"CT label={label}")
        ax[0].axis("off")

        ax[1].imshow(lung_np, cmap="gray")
        ax[1].set_title("HU lung mask")
        ax[1].axis("off")

        ax[2].imshow(image_np, cmap="gray")
        ax[2].imshow(blob_np, cmap="magma", alpha=0.65, vmin=0, vmax=1)
        ax[2].set_title("Soft blob map")
        ax[2].axis("off")

        ax[3].imshow(image_np, cmap="gray")
        if contour_match:
            ax[3].imshow(union, cmap="Blues", alpha=0.25)
            ax[3].imshow(majority, cmap="Greens", alpha=0.42)
            _draw_contours(ax[3], majority, union)
            ax[3].set_title("Radiologist contour\n(audit only)")
        else:
            ax[3].text(0.5, 0.5, "No contour match", ha="center", va="center")
            ax[3].set_title("Radiologist contour\n(audit only)")
        ax[3].axis("off")

        ax[4].imshow(cam_up, cmap="jet", vmin=0, vmax=1)
        ax[4].set_title("Grad-CAM")
        ax[4].axis("off")

        ax[5].imshow(image_np, cmap="gray")
        ax[5].imshow(cam_up, cmap="jet", alpha=0.48, vmin=0, vmax=1)
        ax[5].contour(lung_np, levels=[0.5], colors="lime", linewidths=0.8)
        if np.any(blob_np >= 0.5):
            ax[5].contour(blob_np, levels=[0.5], colors="yellow", linewidths=0.8)
        if contour_match:
            _draw_contours(ax[5], majority, union)
        ax[5].set_title(
            f"p={probability:.2f} pred={prediction}\n"
            f"lung={lung_attention:.0f}% blob={blob_overlap:.3f}"
        )
        ax[5].axis("off")

        model.clear_hooks()

    plt.tight_layout()
    out_path = os.path.join(run_dir, filename)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"Saved Grad-CAM examples -> {out_path}")


def _gradient_norms(model):
    groups = {
        "stem": [model.backbone.conv1, model.backbone.bn1],
        "layer1": [model.backbone.layer1],
        "layer2": [model.backbone.layer2],
        "layer3": [model.backbone.layer3],
        "layer4": [model.backbone.layer4],
        "head": [model.backbone.fc],
    }
    values = {}
    for name, modules in groups.items():
        total = 0.0
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is not None:
                    total += parameter.grad.detach().float().pow(2).sum().item()
        values[name] = total ** 0.5
    return values


def _train_trial(alpha, beta, target_overlap, train, val, device, args, trial_dir):
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.training_seed)

    train_ds = SoftBlobDataset(train)
    generator = torch.Generator().manual_seed(args.training_seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_ds.class_weights().to(device))
    model = NoduleClassifier().to(device)
    if args.init_checkpoint:
        state_dict = torch.load(args.init_checkpoint, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded initial checkpoint -> {args.init_checkpoint}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint = os.path.join(trial_dir, "best_model.pt")
    epoch_csv = os.path.join(trial_dir, "epochs.csv")
    best_auc = float("-inf")
    best_epoch = 0
    rows = []

    for epoch in range(1, args.epochs + 1):
        epoch_alpha, epoch_beta, curriculum_phase = _effective_epoch_weights(
            epoch, alpha, beta, args
        )
        model.train()
        sums = Counter()
        grad_sums = Counter()
        sample_count = 0
        batch_count = 0
        for images, lung, blob, labels in train_loader:
            images = images.to(device)
            lung = lung.to(device)
            blob = blob.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            bce = criterion(logits, labels)
            needs_attention_loss = epoch_alpha > 0.0 or epoch_beta > 0.0
            if needs_attention_loss:
                parts = _loss_parts(
                    model,
                    logits,
                    labels,
                    lung,
                    blob,
                    args.delta,
                    args.min_attention_weight,
                    target_overlap,
                    args.attention_loss,
                    args.nonblob_cost,
                    args.blob_overlap_power,
                    args.blob_hinge,
                    args.blob_correctness_weight,
                )
                attention_loss = parts["attention"].mean()
                blob_loss = parts["blob"].mean()
            else:
                parts = None
                attention_loss = logits.new_tensor(0.0)
                blob_loss = logits.new_tensor(0.0)
            total = bce + epoch_alpha * attention_loss + epoch_beta * blob_loss
            total.backward()
            for name, value in _gradient_norms(model).items():
                grad_sums[name] += value
            optimizer.step()
            model.clear_hooks()

            n = labels.numel()
            sample_count += n
            batch_count += 1
            sums["bce"] += bce.item() * n
            sums["attention"] += attention_loss.item() * n
            sums["weighted_attention"] += epoch_alpha * attention_loss.item() * n
            sums["blob"] += blob_loss.item() * n
            sums["weighted_blob"] += epoch_beta * blob_loss.item() * n
            sums["total"] += total.item() * n
            if parts is not None:
                for key in (
                    "adaptive",
                    "blob_overlap",
                    "standard_blob_overlap",
                    "blob_hinge",
                    "blob_gap",
                    "blob_weight",
                    "lung_attention",
                    "outside_mass",
                    "inside_nonblob_mass",
                    "high_blob_attention",
                    "margin",
                    "inside_mean",
                    "outside_mean",
                    "attention_weight",
                ):
                    sums[key] += parts[key].detach().sum().item()
                sums["blank"] += parts["blank"].sum().item()

        train_metrics = _evaluate_classification(model, train, criterion, device, args.batch_size)
        val_metrics = _evaluate_classification(model, val, criterion, device, args.batch_size)
        for key in (
            "adaptive",
            "blob_overlap",
            "standard_blob_overlap",
            "blob_hinge",
            "blob_gap",
            "blob_weight",
            "lung_attention",
            "outside_mass",
            "inside_nonblob_mass",
            "high_blob_attention",
            "margin",
            "inside_mean",
            "outside_mean",
            "attention_weight",
            "blank",
        ):
            sums.setdefault(key, 0.0)
        row = {
            "epoch": epoch,
            "alpha": alpha,
            "beta": beta,
            "effective_alpha": epoch_alpha,
            "effective_beta": epoch_beta,
            "curriculum_phase": curriculum_phase,
            "target_overlap": target_overlap,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            **{f"train_{key}": value / max(sample_count, 1) for key, value in sums.items()},
            **{f"grad_{key}": value / max(batch_count, 1) for key, value in grad_sums.items()},
        }
        rows.append(row)
        with open(epoch_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(
            f"a={epoch_alpha:g} b={epoch_beta:g} t={target_overlap:g} "
            f"{curriculum_phase} epoch={epoch:02d} "
            f"val_auc={row['val_auc']:.4f} blob_overlap={row['train_blob_overlap']:.4f} "
            f"w_attn={row['train_weighted_attention']:.5f} w_blob={row['train_weighted_blob']:.5f}"
        )
        if np.isfinite(row["val_auc"]) and row["val_auc"] > best_auc:
            best_auc = row["val_auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    best_metrics = _evaluate_classification(model, val, criterion, device, args.batch_size)
    alignment = _evaluate_alignment(
        model,
        val,
        device,
        target_overlap,
        args.delta,
        args.min_attention_weight,
        args.blob_overlap_power,
        args.blob_hinge,
        args.blob_correctness_weight,
    )
    model.remove_hooks()
    return {
        "alpha": alpha,
        "beta": beta,
        "target_overlap": target_overlap,
        "best_epoch": best_epoch,
        "val_auc": best_metrics["auc"],
        "val_accuracy": best_metrics["accuracy"],
        "val_f1": best_metrics["f1"],
        **{f"val_{key}": value for key, value in alignment.items()},
        "checkpoint": checkpoint,
    }


def _write_info(run_dir, args, train, val, test, device):
    def patients(samples):
        return {sample["patient_id"] for sample in samples}
    train_p, val_p, test_p = patients(train), patients(val), patients(test)
    if train_p & val_p or train_p & test_p or val_p & test_p:
        raise RuntimeError("Patient leakage detected.")
    blob_means = [float(np.asarray(sample["soft_blob_map"]).mean()) for sample in train + val + test]
    if args.attention_loss == "zhang-margin":
        blob_overlap_name = (
            "hot_blob_overlap"
            if args.blob_overlap_power != 1.0
            else "blob_overlap"
        )
        hinge_expr = (
            f"relu(target_overlap - {blob_overlap_name})^2"
            if args.blob_hinge == "squared"
            else f"relu(target_overlap - {blob_overlap_name})"
        )
        if args.blob_correctness_weight:
            hinge_expr = f"correctness_weight * {hinge_expr}"
        loss_formula = (
            f"weighted BCE + alpha * Zhang margin + beta * {hinge_expr}"
        )
        alpha_meaning = "alpha scales the confidence-weighted Zhang inside/outside lung margin."
        beta_meaning = "beta scales the soft blob overlap hinge."
    elif args.attention_loss == "outside-tiered":
        loss_formula = (
            "weighted BCE + alpha * (outside_mass + nonblob_cost * inside_nonblob_mass)"
        )
        alpha_meaning = "alpha scales the direct outside-lung plus mild inside-nonblob CAM mass penalty."
        beta_meaning = "beta is ignored in outside-tiered mode; use beta=0 for clean runs."
    elif args.attention_loss == "outside-blob-hinge":
        loss_formula = (
            "weighted BCE + alpha * outside_mass + beta * relu(target_overlap - blob_overlap)"
        )
        alpha_meaning = "alpha scales direct outside-lung CAM mass."
        beta_meaning = "beta scales the soft-blob overlap hinge."
    else:
        loss_formula = "weighted BCE + alpha * outside_mass - beta * blob_overlap"
        alpha_meaning = "alpha scales direct outside-lung CAM mass."
        beta_meaning = "beta scales an explicit soft-blob CAM reward."

    lines = [
        "=== GRAD-CAM ATTENTION LOSS + SOFT BLOB EXPERIMENT ===",
        f"Cache: {os.path.abspath(args.cache_path)}",
        f"Device: {device}",
        f"Lung mask source: {args.mask_source}",
        f"Attention loss mode: {args.attention_loss}",
        f"Alphas: {args.alphas}",
        f"Betas: {args.betas}",
        f"Target overlaps: {args.target_overlaps}",
        f"Delta: {args.delta}",
        f"Minimum attention weight: {args.min_attention_weight}",
        f"Inside nonblob cost: {args.nonblob_cost}",
        f"Blob overlap power: {args.blob_overlap_power}",
        f"Blob hinge: {args.blob_hinge}",
        f"Blob correctness weighting: {args.blob_correctness_weight}",
        f"Curriculum: {args.curriculum}",
        f"Curriculum phase 1 epochs: {args.curriculum_phase1_epochs}",
        f"Curriculum phase 1 alpha: {args.curriculum_phase1_alpha}",
        f"Curriculum phase 2 epochs: {args.curriculum_phase2_epochs}",
        f"Curriculum phase 2 alpha: {args.curriculum_phase2_alpha}",
        f"Curriculum phase 2 beta: {args.curriculum_phase2_beta}",
        f"Selection priority: {args.selection_priority}",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"Initial checkpoint: {args.init_checkpoint or 'none'}",
        f"Optimizer: AdamW, learning rate {args.lr}, weight decay {args.weight_decay}",
        f"Loss: {loss_formula}",
        alpha_meaning,
        beta_meaning,
        "Standard blob overlap: sum((Grad-CAM / CAM sum) * soft_blob_map)",
        "Loss blob overlap: sum((Grad-CAM^power / sum(Grad-CAM^power)) * soft_blob_map)",
        "Correctness weight, if enabled: min_weight + (1 - min_weight) * p(correct class), detached",
        "Contours are not used for training; they are audit-only.",
        f"Mean soft blob map value across samples: {np.mean(blob_means):.4f}",
        "",
        f"Train: {len(train)} samples, {len(train_p)} patients",
        f"Validation: {len(val)} samples, {len(val_p)} patients",
        f"Test held out: {len(test)} samples, {len(test_p)} patients",
        "Patient overlap across splits: none",
    ]
    with open(os.path.join(run_dir, "zhang_soft_blob_info.txt"), "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _summarize(results, run_dir, tolerance, selection_priority):
    reference = next(
        (row for row in results if row["alpha"] == 0.0 and row["beta"] == 0.0),
        results[0],
    )
    threshold = reference["val_auc"] - tolerance
    eligible = [row for row in results if row["val_auc"] >= threshold]
    if selection_priority == "auc":
        selected = max(
            eligible,
            key=lambda row: (row["val_auc"], row["val_blob_overlap"], row["val_lung_attention"]),
        )
    else:
        selected = max(
            eligible,
            key=lambda row: (row["val_blob_overlap"], row["val_lung_attention"], row["val_auc"]),
        )
    fields = [key for key in results[0] if key != "checkpoint"]
    with open(os.path.join(run_dir, "zhang_soft_blob_sensitivity.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in results)
    lines = [
        "=== SOFT BLOB SENSITIVITY ===",
        f"Eligibility: validation AUC >= {threshold:.4f}",
        f"Selection priority: {selection_priority}",
        "alpha | beta | target | best_epoch | val_auc | blob_overlap | lung_cam | eligible",
    ]
    for row in results:
        lines.append(
            f"{row['alpha']:>5.3g} | {row['beta']:>5.3g} | {row['target_overlap']:>6.3g} | "
            f"{row['best_epoch']:>10} | {row['val_auc']:.4f} | "
            f"{row['val_blob_overlap']:.4f} | {row['val_lung_attention']:.4f} | "
            f"{row['val_auc'] >= threshold}"
        )
    lines.extend(
        [
            "",
            f"SELECTED: alpha={selected['alpha']}, beta={selected['beta']}, "
            f"target_overlap={selected['target_overlap']}",
            "Held-out test untouched unless --evaluate-test is supplied.",
        ]
    )
    text = "\n".join(lines)
    with open(os.path.join(run_dir, "zhang_soft_blob_sensitivity.txt"), "w") as file:
        file.write(text + "\n")
    print("\n" + text)
    shutil.copy2(selected["checkpoint"], os.path.join(run_dir, "selected_model.pt"))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--mask-source", choices=["hu", "ts"], default="hu")
    parser.add_argument("--alphas", type=lambda v: _parse_values(v, "alphas"), default=[0.0, 0.15])
    parser.add_argument("--betas", type=lambda v: _parse_values(v, "betas"), default=[0.0, 0.01])
    parser.add_argument("--target-overlaps", type=lambda v: _parse_values(v, "target-overlaps"), default=[0.10])
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--min-attention-weight", type=float, default=0.25)
    parser.add_argument(
        "--attention-loss",
        choices=[
            "zhang-margin",
            "outside-tiered",
            "outside-blob-hinge",
            "outside-blob-reward",
        ],
        default="zhang-margin",
        help=(
            "Grad-CAM attention objective. zhang-margin preserves the original "
            "adaptive Zhang-style margin; outside-tiered directly penalizes outside-lung "
            "and mildly penalizes inside-lung nonblob mass; outside-blob-hinge directly "
            "penalizes outside-lung mass and adds a soft-blob overlap hinge; "
            "outside-blob-reward directly penalizes outside-lung mass and rewards soft-blob overlap."
        ),
    )
    parser.add_argument(
        "--nonblob-cost",
        type=float,
        default=0.10,
        help="Cost for CAM mass inside lung but away from soft blob regions in outside-tiered mode.",
    )
    parser.add_argument(
        "--blob-overlap-power",
        type=float,
        default=1.0,
        help=(
            "Power applied to Grad-CAM before soft-blob overlap. "
            "1.0 preserves the original mass overlap; 2.0 emphasizes hot CAM regions."
        ),
    )
    parser.add_argument(
        "--blob-hinge",
        choices=["linear", "squared"],
        default="linear",
        help="Use linear relu(target-overlap - overlap) or squared hinge for blob loss.",
    )
    parser.add_argument(
        "--blob-correctness-weight",
        action="store_true",
        help=(
            "Scale blob hinge by detached correct-class probability weight. "
            "This lets BCE dominate while the model is wrong."
        ),
    )
    parser.add_argument(
        "--curriculum",
        choices=["none", "outside-then-blob"],
        default="none",
        help=(
            "Optional epoch schedule for attention weights. outside-then-blob "
            "first applies outside-lung cleanup only, then introduces the blob hinge."
        ),
    )
    parser.add_argument("--curriculum-phase1-epochs", type=int, default=5)
    parser.add_argument("--curriculum-phase1-alpha", type=float, default=0.03)
    parser.add_argument("--curriculum-phase2-epochs", type=int, default=5)
    parser.add_argument("--curriculum-phase2-alpha", type=float, default=0.02)
    parser.add_argument("--curriculum-phase2-beta", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--auc-tolerance", type=float, default=0.02)
    parser.add_argument(
        "--selection-priority",
        choices=["attention", "auc"],
        default="attention",
        help=(
            "How to choose the selected model from AUC-eligible trials. "
            "'attention' preserves the original behavior; 'auc' is better for AUC-first runs."
        ),
    )
    parser.add_argument("--training-seed", type=int, default=SEED)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional NoduleClassifier checkpoint to warm-start/fine-tune from.",
    )
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    for name in (
        "delta",
        "min_attention_weight",
        "auc_tolerance",
        "lr",
        "weight_decay",
        "nonblob_cost",
        "blob_overlap_power",
        "curriculum_phase1_alpha",
        "curriculum_phase2_alpha",
        "curriculum_phase2_beta",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.curriculum_phase1_epochs < 0 or args.curriculum_phase2_epochs < 0:
        parser.error("--curriculum-phase*-epochs must be non-negative")
    if args.blob_overlap_power <= 0:
        parser.error("--blob-overlap-power must be greater than 0")
    if args.min_attention_weight > 1:
        parser.error("--min-attention-weight must be between 0 and 1")
    if args.nonblob_cost > 1:
        parser.error("--nonblob-cost must be between 0 and 1")
    if args.init_checkpoint is not None and not os.path.exists(args.init_checkpoint):
        parser.error(f"--init-checkpoint not found: {args.init_checkpoint}")
    if args.limit_samples is not None and args.limit_samples < 30:
        parser.error("--limit-samples must be at least 30")
    if args.curriculum == "outside-then-blob" and args.attention_loss != "outside-blob-hinge":
        parser.error("--curriculum outside-then-blob is intended for --attention-loss outside-blob-hinge")

    run_id = args.run_id or datetime.now().strftime("zhang_soft_blob_%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=False)

    samples = (
        load_nodules_ts(args.cache_path)
        if args.mask_source == "ts"
        else load_nodules_hu(args.cache_path)
    )
    _validate_soft_blob_samples(samples)
    if args.limit_samples is not None:
        rng = np.random.default_rng(args.training_seed)
        keep = rng.choice(len(samples), size=min(args.limit_samples, len(samples)), replace=False)
        samples = [samples[index] for index in sorted(keep)]
    train, val, test = patient_split(samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _write_info(run_dir, args, train, val, test, device)

    results = []
    for alpha in args.alphas:
        for beta in args.betas:
            for target_overlap in args.target_overlaps:
                if beta == 0.0 and target_overlap != args.target_overlaps[0]:
                    continue
                trial_dir = os.path.join(
                    run_dir,
                    f"alpha_{_tag(alpha)}_beta_{_tag(beta)}_target_{_tag(target_overlap)}",
                )
                os.makedirs(trial_dir)
                results.append(
                    _train_trial(alpha, beta, target_overlap, train, val, device, args, trial_dir)
                )
    selected = _summarize(results, run_dir, args.auc_tolerance, args.selection_priority)

    if args.evaluate_test:
        model = NoduleClassifier().to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, "selected_model.pt"), map_location=device))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=SoftBlobDataset(train).class_weights().to(device))
        test_metrics = _evaluate_classification(model, test, criterion, device, args.batch_size)
        test_alignment = _evaluate_alignment(
            model,
            test,
            device,
            selected["target_overlap"],
            args.delta,
            args.min_attention_weight,
            args.blob_overlap_power,
            args.blob_hinge,
            args.blob_correctness_weight,
        )
        lines = [
            f"=== HELD-OUT TEST: {args.attention_loss.upper()} + SOFT BLOB ===",
            f"Selected alpha={selected['alpha']}, beta={selected['beta']}, target={selected['target_overlap']}",
            f"AUC: {test_metrics['auc']:.4f}",
            f"Accuracy: {test_metrics['accuracy']:.4f}",
            f"F1: {test_metrics['f1']:.4f}",
            f"Sensitivity: {test_metrics['sensitivity']:.4f}",
            f"Specificity: {test_metrics['specificity']:.4f}",
            f"CAM mass inside lung: {test_alignment['lung_attention'] * 100:.2f}%",
            f"CAM mass outside lung: {test_alignment['outside_mass'] * 100:.2f}%",
            f"Loss soft blob overlap: {test_alignment['blob_overlap']:.4f}",
            f"Standard soft blob overlap: {test_alignment['standard_blob_overlap']:.4f}",
            f"Blob loss value: {test_alignment['blob']:.6f}",
            f"CAM mass inside high blob response: {test_alignment['high_blob_attention'] * 100:.2f}%",
        ]
        result_name = f"test_results_{args.attention_loss.replace('-', '_')}_soft_blob.txt"
        with open(os.path.join(run_dir, result_name), "w") as file:
            file.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        _save_gradcam_examples(
            model,
            test,
            device,
            run_dir,
            filename="gradcam_examples_soft_blob_test.png",
        )
        model.remove_hooks()
    else:
        model = NoduleClassifier().to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, "selected_model.pt"), map_location=device))
        _save_gradcam_examples(
            model,
            val,
            device,
            run_dir,
            filename="gradcam_examples_soft_blob_val.png",
        )
        model.remove_hooks()


if __name__ == "__main__":
    main()
