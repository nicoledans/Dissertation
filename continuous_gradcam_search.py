"""Validation-only search for continuous adaptive outside-Grad-CAM supervision.

The penalty continuously measures the fraction of predicted-class Grad-CAM
energy outside a fixed HU lung mask and scales it by detached confidence.
"""

import argparse
import csv
import os
import shutil
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from config import BATCH_SIZE, EPOCHS, LR, RESULTS_DIR, SEED, TRAIN_CACHE_PATH
from dataset import LIDCDataset, load_nodules_hu, patient_split
from model import NoduleClassifier


DEFAULT_ALPHAS = [0.0, 0.05, 0.15]
BLANK_CAM_EPS = 1e-8


def _parse_alphas(value):
    try:
        alphas = sorted({float(item.strip()) for item in value.split(",")})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from exc
    if not alphas or any(not np.isfinite(alpha) or alpha < 0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alphas must contain finite non-negative values")
    if 0.0 not in alphas:
        raise argparse.ArgumentTypeError("alphas must include 0.0 as the reference")
    return alphas


def _tag(value):
    return f"{value:.10g}".replace(".", "p")


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


def _evaluate_classification(model, loader, criterion, device):
    model.eval()
    probabilities, labels = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for images, _masks, batch_labels in loader:
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
    }


def _continuous_outside_penalty(model, logits, masks):
    """Return confidence-scaled outside Grad-CAM energy fraction."""
    raw_cam = model.differentiable_gradcam(
        model.class_scores(logits), normalise=False
    )
    blank = raw_cam.detach().flatten(start_dim=1).amax(dim=1) <= BLANK_CAM_EPS
    cam = model.normalise_gradcam(raw_cam)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=masks.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    mask = masks.squeeze(1)
    inside_mean = (cam * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) + 1e-8)
    outside_mask = 1.0 - mask
    outside_mean = (cam * outside_mask).sum(dim=(1, 2)) / (
        outside_mask.sum(dim=(1, 2)) + 1e-8
    )
    outside_fraction = (cam * outside_mask).sum(dim=(1, 2)) / (
        cam.sum(dim=(1, 2)) + 1e-8
    )
    confidence = 2.0 * (torch.sigmoid(logits).detach() - 0.5).abs()
    return (
        confidence * outside_fraction,
        outside_fraction,
        confidence,
        inside_mean,
        outside_mean,
        blank,
    )


def _gradient_groups(model):
    return {
        "stem": [model.backbone.conv1, model.backbone.bn1],
        "layer1": [model.backbone.layer1],
        "layer2": [model.backbone.layer2],
        "layer3": [model.backbone.layer3],
        "layer4": [model.backbone.layer4],
        "head": [model.backbone.fc],
    }


def _gradient_norms(model):
    values = {}
    for name, modules in _gradient_groups(model).items():
        squared = 0.0
        for module in modules:
            for parameter in module.parameters():
                if parameter.grad is not None:
                    squared += parameter.grad.detach().float().pow(2).sum().item()
        values[name] = squared ** 0.5
    return values


def _evaluate_alignment(model, nodules, device):
    model.eval()
    inside_fractions, mask_areas = [], []
    blank_count = 0
    loader = DataLoader(LIDCDataset(nodules), batch_size=1, shuffle=False)
    with torch.enable_grad():
        for images, masks, _labels in loader:
            images, masks = images.to(device), masks.to(device)
            model.zero_grad(set_to_none=True)
            model.clear_hooks()
            logits = model(images).squeeze(1)
            model.class_scores(logits).sum().backward()
            raw_cam = model.get_gradcam(normalise=False)
            blank_count += int(raw_cam.amax().item() <= BLANK_CAM_EPS)
            cam = model.normalise_gradcam(raw_cam)
            cam = F.interpolate(
                cam.unsqueeze(1), size=masks.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            mask = masks.squeeze(1)
            inside_fractions.append(
                (cam * mask).sum().item() / (cam.sum().item() + 1e-8)
            )
            mask_areas.append(mask.mean().item())
            model.clear_hooks()
    alignment = float(np.mean(inside_fractions))
    mask_area = float(np.mean(mask_areas))
    return {
        "alignment": alignment,
        "mask_area": mask_area,
        "normalized_alignment": alignment / mask_area,
        "blank_rate": blank_count / max(len(inside_fractions), 1),
    }


def _plot_freezing_diagnostics(epoch_rows, trial_dir, alpha):
    """Visualize overfitting and stage-level gradient activity."""
    epochs = [row["epoch"] for row in epoch_rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].plot(epochs, [row["train_auc"] for row in epoch_rows], label="Train")
    axes[0, 0].plot(epochs, [row["val_auc"] for row in epoch_rows], label="Validation")
    axes[0, 0].set(title="Classification AUC", xlabel="Epoch", ylabel="AUC")
    axes[0, 0].legend()

    axes[0, 1].plot(epochs, [row["train_loss"] for row in epoch_rows], label="Train")
    axes[0, 1].plot(epochs, [row["val_loss"] for row in epoch_rows], label="Validation")
    axes[0, 1].set(title="Classification Loss", xlabel="Epoch", ylabel="Weighted BCE")
    axes[0, 1].legend()

    axes[1, 0].plot(epochs, [row["auc_gap"] for row in epoch_rows], label="Train AUC - Val AUC")
    axes[1, 0].plot(epochs, [row["loss_gap"] for row in epoch_rows], label="Val loss - Train loss")
    axes[1, 0].axhline(0, color="black", linewidth=1)
    axes[1, 0].set(title="Generalization Gaps", xlabel="Epoch", ylabel="Gap")
    axes[1, 0].legend()

    for stage in ("stem", "layer1", "layer2", "layer3", "layer4", "head"):
        axes[1, 1].plot(
            epochs,
            [row[f"grad_{stage}"] for row in epoch_rows],
            label=stage,
        )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set(
        title="Mean Gradient Norm by Trainable Stage",
        xlabel="Epoch",
        ylabel="Gradient norm (log scale)",
    )
    axes[1, 1].legend(ncol=2)

    for axis in axes.flat:
        axis.grid(True, alpha=0.3)
    fig.suptitle(f"Freezing Diagnostics: alpha={alpha:g}")
    fig.tight_layout()
    fig.savefig(os.path.join(trial_dir, "freezing_diagnostics.png"), dpi=150)
    plt.close(fig)


def _train_trial(alpha, train_nods, val_nods, device, args, trial_dir):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    train_ds = LIDCDataset(train_nods)
    val_ds = LIDCDataset(val_nods)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = NoduleClassifier().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_ds.class_weights().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    checkpoint = os.path.join(trial_dir, "best_model.pt")
    epoch_csv = os.path.join(trial_dir, "epochs.csv")
    best_auc, best_epoch = float("-inf"), 0
    epoch_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        sums = Counter()
        grad_sums = Counter()
        sample_count = batch_count = 0

        for images, masks, labels in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images).squeeze(1)
            bce = criterion(logits, labels)
            adaptive, outside_fraction, confidence, inside, outside, blank = (
                _continuous_outside_penalty(model, logits, masks)
            )
            adaptive_loss = adaptive.mean()
            total = bce + alpha * adaptive_loss
            total.backward()
            for name, value in _gradient_norms(model).items():
                grad_sums[name] += value
            optimizer.step()
            model.clear_hooks()

            n = labels.numel()
            sample_count += n
            batch_count += 1
            sums["bce"] += bce.item() * n
            sums["adaptive"] += adaptive_loss.item() * n
            sums["total"] += total.item() * n
            sums["outside_fraction"] += outside_fraction.detach().sum().item()
            sums["confidence"] += confidence.detach().sum().item()
            sums["inside_mean"] += inside.detach().sum().item()
            sums["outside_mean"] += outside.detach().sum().item()
            sums["blank"] += blank.sum().item()

        train_metrics = _evaluate_classification(model, train_eval_loader, criterion, device)
        val_metrics = _evaluate_classification(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_auc": train_metrics["auc"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_auc": val_metrics["auc"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1": val_metrics["f1"],
            "auc_gap": train_metrics["auc"] - val_metrics["auc"],
            "loss_gap": val_metrics["loss"] - train_metrics["loss"],
            "train_weighted_adaptive": alpha * sums["adaptive"] / max(sample_count, 1),
            **{f"train_{key}": value / max(sample_count, 1) for key, value in sums.items()},
            **{f"grad_{key}": value / max(batch_count, 1) for key, value in grad_sums.items()},
        }
        epoch_rows.append(row)
        with open(epoch_csv, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=epoch_rows[0].keys())
            writer.writeheader()
            writer.writerows(epoch_rows)
        print(
            f"alpha={alpha:.2f} epoch={epoch:02d} "
            f"train_auc={row['train_auc']:.4f} val_auc={row['val_auc']:.4f} "
            f"gap={row['auc_gap']:.4f} "
            f"outside={row['train_outside_fraction'] * 100:.1f}%"
        )
        if np.isfinite(row["val_auc"]) and row["val_auc"] > best_auc:
            best_auc, best_epoch = row["val_auc"], epoch
            torch.save(model.state_dict(), checkpoint)

    _plot_freezing_diagnostics(epoch_rows, trial_dir, alpha)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    best_metrics = _evaluate_classification(model, val_loader, criterion, device)
    alignment = _evaluate_alignment(model, val_nods, device)
    model.remove_hooks()
    return {
        "alpha": alpha,
        "best_epoch": best_epoch,
        "val_auc": best_metrics["auc"],
        "val_accuracy": best_metrics["accuracy"],
        "val_f1": best_metrics["f1"],
        **{f"val_{key}": value for key, value in alignment.items()},
        "checkpoint": checkpoint,
    }


def _write_info(run_dir, args, device, train, val, test):
    def patients(samples):
        return {sample["patient_id"] for sample in samples}
    train_p, val_p, test_p = patients(train), patients(val), patients(test)
    if train_p & val_p or train_p & test_p or val_p & test_p:
        raise RuntimeError("Patient leakage detected.")
    lines = [
        "=== CONTINUOUS ADAPTIVE OUTSIDE-GRADCAM ALPHA SEARCH ===",
        f"Cache: {os.path.abspath(args.cache_path)}",
        f"Device: {device}",
        f"Seed: {SEED}",
        f"Alphas: {args.alphas}",
        f"Epochs per alpha: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"AUC tolerance: {args.auc_tolerance}",
        "Explanation: predicted-class differentiable Grad-CAM",
        "Supervision: fixed binary HU lung mask",
        "Loss: BCE + alpha * detached-confidence * outside-Grad-CAM fraction",
        "Outside fraction: sum(Grad-CAM outside mask) / sum(all Grad-CAM)",
        "Penalty activity: continuous while outside Grad-CAM energy is nonzero",
        "Confidence: 2 * abs(sigmoid(logit) - 0.5), detached",
        "Checkpoint selection: validation AUC",
        "Held-out test evaluated: no",
        "Backbone: fully trainable",
        "Freezing diagnostics: train/val loss, AUC, accuracy, gaps, and stage gradient norms",
        "",
        f"Train: {len(train)} samples, {len(train_p)} patients",
        f"Validation: {len(val)} samples, {len(val_p)} patients",
        f"Test held out: {len(test)} samples, {len(test_p)} patients",
        "Patient overlap across splits: none",
    ]
    with open(os.path.join(run_dir, "continuous_search_info.txt"), "w") as file:
        file.write("\n".join(lines) + "\n")


def _summarize(results, run_dir, tolerance):
    reference = next(result for result in results if result["alpha"] == 0.0)
    threshold = reference["val_auc"] - tolerance
    eligible = [r for r in results if r["val_auc"] >= threshold]
    selected = max(
        eligible,
        key=lambda r: (r["val_normalized_alignment"], r["val_auc"], -r["alpha"]),
    )
    fields = [key for key in results[0] if key != "checkpoint"]
    with open(os.path.join(run_dir, "continuous_sensitivity.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in results)
    lines = [
        "=== CONTINUOUS ADAPTIVE OUTSIDE-GRADCAM SENSITIVITY ===",
        f"Eligibility: validation AUC >= {threshold:.4f} "
        f"(alpha=0 AUC {reference['val_auc']:.4f} - tolerance {tolerance:.4f})",
        "",
        "alpha | best_epoch | val_auc | alignment | norm_align | eligible",
    ]
    for row in results:
        lines.append(
            f"{row['alpha']:>5.2f} | {row['best_epoch']:>10} | {row['val_auc']:.4f} | "
            f"{row['val_alignment'] * 100:>8.2f}% | "
            f"{row['val_normalized_alignment']:>10.3f} | {row['val_auc'] >= threshold}"
        )
    lines.extend(["", f"SELECTED ALPHA: {selected['alpha']:.4f}", "Held-out test untouched."])
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(run_dir, "continuous_sensitivity.txt"), "w") as file:
        file.write(text + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    alphas = [row["alpha"] for row in results]
    axes[0].plot(alphas, [row["val_auc"] for row in results], marker="o")
    axes[0].set(xlabel="Alpha", ylabel="Validation AUC")
    axes[1].plot(alphas, [row["val_alignment"] * 100 for row in results], marker="o")
    axes[1].set(xlabel="Alpha", ylabel="Validation Grad-CAM inside lung (%)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "continuous_sensitivity.png"), dpi=150)
    plt.close(fig)
    shutil.copy2(selected["checkpoint"], os.path.join(run_dir, "selected_model.pt"))
    return selected


def main():
    parser = argparse.ArgumentParser(
        description="Continuous adaptive outside-Grad-CAM alpha search."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--alphas", type=_parse_alphas, default=DEFAULT_ALPHAS)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--auc-tolerance", type=float, default=0.02)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    if args.auc_tolerance < 0:
        parser.error("--auc-tolerance must be non-negative")

    run_id = args.run_id or datetime.now().strftime("continuous_grid_%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=False)
    nodules = load_nodules_hu(args.cache_path)
    train, val, test = patient_split(nodules)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _write_info(run_dir, args, device, train, val, test)

    results = []
    for alpha in args.alphas:
        trial_dir = os.path.join(run_dir, f"alpha_{_tag(alpha)}")
        os.makedirs(trial_dir)
        results.append(_train_trial(alpha, train, val, device, args, trial_dir))
    selected = _summarize(results, run_dir, args.auc_tolerance)
    print(f"Selected alpha: {selected['alpha']}. Test set remains untouched.")


if __name__ == "__main__":
    main()
