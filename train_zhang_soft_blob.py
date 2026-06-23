"""Train Zhang-style lung Grad-CAM loss plus weak soft-blob overlap loss."""

import argparse
import csv
import os
import shutil
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
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


def _loss_parts(model, logits, labels, lung_masks, blob_maps, delta, min_attention_weight, target_overlap):
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

    cam_mass = cam / (cam.sum(dim=(1, 2), keepdim=True) + 1e-8)
    blob_overlap = (cam_mass * blob).sum(dim=(1, 2))
    blob_loss = F.relu(float(target_overlap) - blob_overlap)
    lung_attention = (cam_mass * lung).sum(dim=(1, 2))
    high_blob_attention = (cam_mass * (blob >= 0.50).to(cam.dtype)).sum(dim=(1, 2))
    return {
        "adaptive": adaptive,
        "margin": margin,
        "blob": blob_loss,
        "blob_overlap": blob_overlap,
        "lung_attention": lung_attention,
        "high_blob_attention": high_blob_attention,
        "inside_mean": inside_mean,
        "outside_mean": outside_mean,
        "correct_class_probability": correct_class_probability.detach(),
        "attention_weight": attention_weight,
        "blank": blank,
    }


def _evaluate_alignment(model, samples, device, target_overlap):
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
            parts = _loss_parts(
                model,
                logits,
                labels,
                lung,
                blob,
                delta=0.1,
                min_attention_weight=0.25,
                target_overlap=target_overlap,
            )
            for key in (
                "blob_overlap",
                "blob",
                "lung_attention",
                "high_blob_attention",
                "margin",
            ):
                values[key] += float(parts[key].detach().mean().cpu())
            values["blank"] += int(parts["blank"].sum().item())
            n += 1
            model.clear_hooks()
    return {key: value / max(n, 1) for key, value in values.items()} | {"n": n}


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=args.weight_decay)
    checkpoint = os.path.join(trial_dir, "best_model.pt")
    epoch_csv = os.path.join(trial_dir, "epochs.csv")
    best_auc = float("-inf")
    best_epoch = 0
    rows = []

    for epoch in range(1, args.epochs + 1):
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
            parts = _loss_parts(
                model,
                logits,
                labels,
                lung,
                blob,
                args.delta,
                args.min_attention_weight,
                target_overlap,
            )
            adaptive_loss = parts["adaptive"].mean()
            blob_loss = parts["blob"].mean()
            total = bce + alpha * adaptive_loss + beta * blob_loss
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
            sums["weighted_adaptive"] += alpha * adaptive_loss.item() * n
            sums["blob"] += blob_loss.item() * n
            sums["weighted_blob"] += beta * blob_loss.item() * n
            sums["total"] += total.item() * n
            for key in (
                "blob_overlap",
                "lung_attention",
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
        row = {
            "epoch": epoch,
            "alpha": alpha,
            "beta": beta,
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
            f"a={alpha:g} b={beta:g} t={target_overlap:g} epoch={epoch:02d} "
            f"val_auc={row['val_auc']:.4f} blob_overlap={row['train_blob_overlap']:.4f} "
            f"w_blob={row['train_weighted_blob']:.5f}"
        )
        if np.isfinite(row["val_auc"]) and row["val_auc"] > best_auc:
            best_auc = row["val_auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    best_metrics = _evaluate_classification(model, val, criterion, device, args.batch_size)
    alignment = _evaluate_alignment(model, val, device, target_overlap)
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
    lines = [
        "=== ZHANG LUNG LOSS + SOFT BLOB OVERLAP LOSS ===",
        f"Cache: {os.path.abspath(args.cache_path)}",
        f"Device: {device}",
        f"Lung mask source for Zhang loss: {args.mask_source}",
        f"Alphas: {args.alphas}",
        f"Betas: {args.betas}",
        f"Target overlaps: {args.target_overlaps}",
        f"Delta: {args.delta}",
        f"Minimum attention weight: {args.min_attention_weight}",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"Optimizer: AdamW, learning rate {LR}, weight decay {args.weight_decay}",
        "Loss: weighted BCE + alpha * Zhang-style lung CAM margin + beta * soft blob overlap hinge",
        "Blob overlap: sum((normalised Grad-CAM / CAM sum) * soft_blob_map)",
        "Blob loss: relu(target_overlap - blob_overlap)",
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


def _summarize(results, run_dir, tolerance):
    reference = next(
        (row for row in results if row["alpha"] == 0.0 and row["beta"] == 0.0),
        results[0],
    )
    threshold = reference["val_auc"] - tolerance
    eligible = [row for row in results if row["val_auc"] >= threshold]
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
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--auc-tolerance", type=float, default=0.02)
    parser.add_argument("--training-seed", type=int, default=SEED)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    for name in ("delta", "min_attention_weight", "auc_tolerance", "weight_decay"):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.min_attention_weight > 1:
        parser.error("--min-attention-weight must be between 0 and 1")
    if args.limit_samples is not None and args.limit_samples < 30:
        parser.error("--limit-samples must be at least 30")

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
    selected = _summarize(results, run_dir, args.auc_tolerance)

    if args.evaluate_test:
        model = NoduleClassifier().to(device)
        model.load_state_dict(torch.load(os.path.join(run_dir, "selected_model.pt"), map_location=device))
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=SoftBlobDataset(train).class_weights().to(device))
        test_metrics = _evaluate_classification(model, test, criterion, device, args.batch_size)
        test_alignment = _evaluate_alignment(model, test, device, selected["target_overlap"])
        lines = [
            "=== HELD-OUT TEST: ZHANG + SOFT BLOB ===",
            f"Selected alpha={selected['alpha']}, beta={selected['beta']}, target={selected['target_overlap']}",
            f"AUC: {test_metrics['auc']:.4f}",
            f"Accuracy: {test_metrics['accuracy']:.4f}",
            f"F1: {test_metrics['f1']:.4f}",
            f"Sensitivity: {test_metrics['sensitivity']:.4f}",
            f"Specificity: {test_metrics['specificity']:.4f}",
            f"CAM mass inside lung: {test_alignment['lung_attention'] * 100:.2f}%",
            f"Soft blob overlap: {test_alignment['blob_overlap']:.4f}",
            f"CAM mass inside high blob response: {test_alignment['high_blob_attention'] * 100:.2f}%",
        ]
        with open(os.path.join(run_dir, "test_results_zhang_soft_blob.txt"), "w") as file:
            file.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        model.remove_hooks()


if __name__ == "__main__":
    main()
