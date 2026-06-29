"""Train a classification-only model with CT + soft-blob + lung-mask input.

This is a side experiment inspired by mask-prior classification models: the
network receives an annotation-free nodule-like soft-blob map as an input
channel, but the loss remains ordinary weighted BCE. Radiologist contours are
not used.
"""

import argparse
import csv
import os
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from config import BATCH_SIZE, EPOCHS, IMG_SIZE, LR, RESULTS_DIR, SEED
from dataset import _map_to_tensor, _mask_to_tensor, _patch_to_tensor, load_nodules_hu, patient_split
from model import NoduleClassifier


def _classification_metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float32)
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


def _validate_soft_blob_samples(samples):
    missing = [i for i, sample in enumerate(samples) if "soft_blob_map" not in sample]
    if missing:
        raise ValueError(
            f"{len(missing)} samples do not contain soft_blob_map. "
            "Use cache/cache_hu_soft_blobs.pkl or rebuild the soft-blob cache."
        )


def _augment_prior_input(ct, blob, lung):
    """Apply mild CT-safe augmentation with shared geometry across channels."""
    if torch.rand(()) < 0.25:
        return torch.cat([ct, blob, lung], dim=0)

    if torch.rand(()) < 0.5:
        ct = TF.hflip(ct)
        blob = TF.hflip(blob)
        lung = TF.hflip(lung)

    angle = float(torch.empty(()).uniform_(-7.0, 7.0).item())
    translate = [
        int(torch.empty(()).uniform_(-0.04, 0.04).item() * IMG_SIZE),
        int(torch.empty(()).uniform_(-0.04, 0.04).item() * IMG_SIZE),
    ]
    scale = float(torch.empty(()).uniform_(0.96, 1.04).item())
    ct = TF.affine(
        ct,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    blob = TF.affine(
        blob,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    lung = TF.affine(
        lung,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    )

    contrast = float(torch.empty(()).uniform_(0.92, 1.08).item())
    brightness = float(torch.empty(()).uniform_(-0.03, 0.03).item())
    ct = (ct - 0.5) * contrast + 0.5 + brightness
    if torch.rand(()) < 0.25:
        ct = ct + torch.randn_like(ct) * 0.01

    return torch.cat([
        ct.clamp(0.0, 1.0),
        blob.clamp(0.0, 1.0),
        (lung > 0.5).float(),
    ], dim=0)


class SoftBlobInputDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.labels = [sample["label"] for sample in samples]
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        ct = _patch_to_tensor(sample["patch"])[1:2]
        blob = _map_to_tensor(sample["soft_blob_map"])
        lung = _mask_to_tensor(sample["mask"])
        if self.augment:
            image = _augment_prior_input(ct, blob, lung)
        else:
            image = torch.cat([ct, blob, lung], dim=0)
        label = torch.tensor(sample["label"], dtype=torch.float32)
        return image, lung, blob, label

    def class_weights(self):
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(1.0)
        return torch.tensor(n_neg / n_pos, dtype=torch.float32)


def _predict(model, samples, criterion, device, batch_size, num_workers=0):
    loader = DataLoader(
        SoftBlobInputDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model.eval()
    labels, probabilities = [], []
    loss_sum = 0.0
    seen = 0
    with torch.no_grad():
        for images, _lung, _blob, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(images).squeeze(1)
            loss_sum += criterion(logits, batch_labels).item() * batch_labels.numel()
            seen += batch_labels.numel()
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
            model.clear_hooks()
    return labels, probabilities, loss_sum / max(seen, 1)


def _gradcam_stats(model, samples, device):
    loader = DataLoader(SoftBlobInputDataset(samples), batch_size=1, shuffle=False)
    lung_pcts = []
    blob_overlaps = []
    model.eval()
    with torch.enable_grad():
        for images, lung, blob, _labels in loader:
            images = images.to(device)
            lung = lung.to(device)
            blob = blob.to(device)
            model.zero_grad(set_to_none=True)
            model.clear_hooks()
            logits = model(images).squeeze(1)
            model.class_scores(logits).sum().backward()
            cam = model.get_gradcam()
            cam_up = F.interpolate(
                cam.unsqueeze(1),
                size=lung.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            cam_mass = cam_up / (cam_up.flatten(start_dim=1).sum(dim=1).view(-1, 1, 1) + 1e-8)
            lung_2d = lung.squeeze(1)
            blob_2d = blob.squeeze(1)
            lung_pcts.append(float((cam_mass * lung_2d).sum().item() * 100.0))
            blob_overlaps.append(float((cam_mass * blob_2d).sum().item()))
            model.clear_hooks()
    return {
        "n": len(lung_pcts),
        "lung_mean_pct": float(np.mean(lung_pcts)) if lung_pcts else float("nan"),
        "lung_std_pct": float(np.std(lung_pcts)) if lung_pcts else float("nan"),
        "blob_mean": float(np.mean(blob_overlaps)) if blob_overlaps else float("nan"),
        "blob_std": float(np.std(blob_overlaps)) if blob_overlaps else float("nan"),
    }


def _write_predictions(path, samples, labels, probabilities):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "patient_id",
            "true_label",
            "prob_malignant",
            "pred_label_at_0p5",
            "correct_at_0p5",
        ])
        for index, (sample, label, probability) in enumerate(zip(samples, labels, probabilities)):
            pred = int(probability >= 0.5)
            writer.writerow([
                index,
                sample.get("patient_id", ""),
                int(label),
                f"{probability:.8f}",
                pred,
                int(pred == int(label)),
            ])


def _save_examples(model, samples, device, run_dir, filename):
    if not samples:
        return
    n_rows = min(4, len(samples))
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 3.8 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    loader = DataLoader(SoftBlobInputDataset(samples), batch_size=1, shuffle=False)
    model.eval()
    with torch.enable_grad():
        for row, (images, lung, blob, labels) in enumerate(loader):
            if row >= n_rows:
                break
            images = images.to(device)
            lung = lung.to(device)
            blob = blob.to(device)
            model.zero_grad(set_to_none=True)
            model.clear_hooks()
            logits = model(images).squeeze(1)
            prob = torch.sigmoid(logits).item()
            model.class_scores(logits).sum().backward()
            cam = model.get_gradcam()
            cam_up = F.interpolate(
                cam.unsqueeze(1),
                size=(IMG_SIZE, IMG_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze().detach().cpu().numpy()
            img_np = images[0, 0].detach().cpu().numpy()
            blob_np = blob[0, 0].detach().cpu().numpy()
            lung_np = lung[0, 0].detach().cpu().numpy()
            label = int(labels[0].item())

            ax = axes[row]
            ax[0].imshow(img_np, cmap="gray")
            ax[0].set_title(f"CT\nlabel={label} p={prob:.2f}")
            ax[1].imshow(blob_np, cmap="magma", vmin=0, vmax=1)
            ax[1].set_title("Soft blob input")
            ax[2].imshow(lung_np, cmap="gray")
            ax[2].set_title("Lung mask input")
            ax[3].imshow(cam_up, cmap="jet", vmin=0, vmax=1)
            ax[3].set_title("Grad-CAM")
            ax[4].imshow(img_np, cmap="gray")
            ax[4].imshow(cam_up, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            ax[4].contour(lung_np, levels=[0.5], colors="lime", linewidths=0.8)
            if np.any(blob_np > 0.5):
                ax[4].contour(blob_np, levels=[0.5], colors="yellow", linewidths=0.8)
            ax[4].set_title("Overlay\nlime=lung yellow=blob")
            for item in ax:
                item.axis("off")
            model.clear_hooks()

    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, filename), dpi=150)
    plt.close(fig)


def _write_split_report(path, train, val, test):
    def lines_for(name, samples):
        patients = sorted({sample["patient_id"] for sample in samples})
        counts = Counter(sample["label"] for sample in samples)
        lines = [
            f"{name}: {len(samples)} samples, {len(patients)} patients",
            f"  benign={counts.get(0, 0)}, malignant={counts.get(1, 0)}",
            "  patients: " + ", ".join(patients),
        ]
        return lines

    train_p = {sample["patient_id"] for sample in train}
    val_p = {sample["patient_id"] for sample in val}
    test_p = {sample["patient_id"] for sample in test}
    overlap = (train_p & val_p) | (train_p & test_p) | (val_p & test_p)
    lines = []
    lines.extend(lines_for("train", train))
    lines.extend(lines_for("val", val))
    lines.extend(lines_for("test", test))
    lines.append(f"patient_overlap_across_splits={sorted(overlap)}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--cache-path", type=str, default="cache/cache_hu_soft_blobs.pkl")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_soft_blob_input")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    samples = load_nodules_hu(args.cache_path)
    _validate_soft_blob_samples(samples)
    if args.limit_samples is not None:
        samples = samples[:args.limit_samples]
    train, val, test = patient_split(samples)
    _write_split_report(os.path.join(run_dir, "split_soft_blob_input.txt"), train, val, test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SoftBlobInputDataset(train, augment=args.augment)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = NoduleClassifier().to(device)
    pos_weight = train_ds.class_weights().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    info_path = os.path.join(run_dir, "soft_blob_input_info.txt")
    with open(info_path, "w") as f:
        f.write("=== SOFT-BLOB INPUT CLASSIFIER ===\n")
        f.write("Goal: test an annotation-free nodule-like spatial prior as model input.\n")
        f.write(f"Cache path: {args.cache_path}\n")
        f.write(f"Device: {device}\n")
        f.write("Input channels: channel0=CT image, channel1=soft_blob_map, channel2=HU lung mask.\n")
        f.write("Loss: weighted BCEWithLogitsLoss only; no Grad-CAM loss, no Zhang loss.\n")
        f.write("Soft-blob maps and lung masks are image-derived inputs, not labels.\n")
        f.write("Radiologist contours are not used by this script.\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Training augmentation: {args.augment}\n")
        if args.augment:
            f.write("Augmentation: shared geometry for CT/blob/lung; intensity jitter/noise on CT only; validation/test unaugmented.\n")
        f.write(f"Loss pos_weight: {pos_weight.item():.6f}\n")
        f.write(f"Train/val/test samples: {len(train)}/{len(val)}/{len(test)}\n")

    epochs_path = os.path.join(run_dir, "soft_blob_input_epochs.csv")
    log_path = os.path.join(run_dir, "soft_blob_input_log.txt")
    best_auc = -float("inf")
    best_epoch = 0
    model_path = os.path.join(run_dir, "soft_blob_input_model.pt")

    with open(epochs_path, "w", newline="") as epoch_f, open(log_path, "w") as log_f:
        writer = csv.writer(epoch_f)
        writer.writerow([
            "epoch",
            "train_bce",
            "val_bce",
            "val_auc",
            "val_accuracy_0p5",
            "val_f1_0p5",
            "val_sensitivity_0p5",
            "val_specificity_0p5",
        ])
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_seen = 0
            for images, _lung, _blob, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images).squeeze(1)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * labels.numel()
                train_seen += labels.numel()
                model.clear_hooks()

            train_bce = train_loss / max(train_seen, 1)
            val_labels, val_probs, val_bce = _predict(
                model, val, criterion, device, args.batch_size, args.num_workers
            )
            val_metrics = _classification_metrics(val_labels, val_probs)
            writer.writerow([
                epoch,
                f"{train_bce:.8f}",
                f"{val_bce:.8f}",
                f"{val_metrics['auc']:.8f}",
                f"{val_metrics['accuracy']:.8f}",
                f"{val_metrics['f1']:.8f}",
                f"{val_metrics['sensitivity']:.8f}",
                f"{val_metrics['specificity']:.8f}",
            ])
            epoch_f.flush()
            line = (
                f"Epoch {epoch:02d} | Train BCE {train_bce:.4f} | "
                f"Val BCE {val_bce:.4f} | Val AUC {val_metrics['auc']:.4f} | "
                f"Val Acc {val_metrics['accuracy']:.4f} | Val F1 {val_metrics['f1']:.4f} | "
                f"Val Sens {val_metrics['sensitivity']:.4f} | Val Spec {val_metrics['specificity']:.4f}\n"
            )
            print(line, end="")
            log_f.write(line)
            log_f.flush()
            if np.isfinite(val_metrics["auc"]) and val_metrics["auc"] > best_auc:
                best_auc = val_metrics["auc"]
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
                print(f"  -> Saved best soft-blob-input model (AUC {best_auc:.4f})")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))

    val_labels, val_probs, val_bce = _predict(
        model, val, criterion, device, args.batch_size, args.num_workers
    )
    val_metrics = _classification_metrics(val_labels, val_probs)
    _write_predictions(
        os.path.join(run_dir, "val_predictions_soft_blob_input.csv"),
        val,
        val_labels,
        val_probs,
    )
    val_alignment = _gradcam_stats(model, val, device)
    with open(info_path, "a") as f:
        f.write("\n=== BEST VALIDATION CHECKPOINT ===\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation AUC during training: {best_auc:.6f}\n")
        f.write(f"Reloaded validation BCE: {val_bce:.6f}\n")
        f.write(f"Reloaded validation AUC: {val_metrics['auc']:.6f}\n")
        f.write(f"Reloaded validation accuracy at 0.5: {val_metrics['accuracy']:.6f}\n")
        f.write(f"Reloaded validation F1 at 0.5: {val_metrics['f1']:.6f}\n")
        f.write(f"Reloaded validation sensitivity at 0.5: {val_metrics['sensitivity']:.6f}\n")
        f.write(f"Reloaded validation specificity at 0.5: {val_metrics['specificity']:.6f}\n")
        f.write(f"Validation Grad-CAM inside lung: {val_alignment['lung_mean_pct']:.2f}% +/- {val_alignment['lung_std_pct']:.2f}%\n")
        f.write(f"Validation Grad-CAM soft blob overlap: {val_alignment['blob_mean']:.4f} +/- {val_alignment['blob_std']:.4f}\n")

    _save_examples(model, val, device, run_dir, "gradcam_examples_soft_blob_input_val.png")

    if args.evaluate_test:
        test_labels, test_probs, test_bce = _predict(
            model, test, criterion, device, args.batch_size, args.num_workers
        )
        test_metrics = _classification_metrics(test_labels, test_probs)
        _write_predictions(
            os.path.join(run_dir, "test_predictions_soft_blob_input.csv"),
            test,
            test_labels,
            test_probs,
        )
        test_alignment = _gradcam_stats(model, test, device)
        result_path = os.path.join(run_dir, "test_results_soft_blob_input.txt")
        with open(result_path, "w") as f:
            f.write("=== HELD-OUT TEST: SOFT-BLOB INPUT CLASSIFIER ===\n")
            f.write(f"Test samples: {len(test)}\n")
            f.write(f"Test BCE loss: {test_bce:.4f}\n")
            f.write(f"Test AUC: {test_metrics['auc']:.4f}\n")
            f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
            f.write(f"Test F1: {test_metrics['f1']:.4f}\n")
            f.write(f"Test Sensitivity: {test_metrics['sensitivity']:.4f}\n")
            f.write(f"Test Specificity: {test_metrics['specificity']:.4f}\n")
            f.write("\n=== TEST GRAD-CAM ALIGNMENT ===\n")
            f.write(f"Test samples evaluated: {test_alignment['n']}\n")
            f.write(f"Mean % inside lung mask: {test_alignment['lung_mean_pct']:.2f}%\n")
            f.write(f"Std inside lung mask: {test_alignment['lung_std_pct']:.2f}%\n")
            f.write(f"Mean soft blob overlap: {test_alignment['blob_mean']:.4f}\n")
            f.write(f"Std soft blob overlap: {test_alignment['blob_std']:.4f}\n")
        _save_examples(model, test, device, run_dir, "gradcam_examples_soft_blob_input_test.png")
        print(f"Saved test results -> {result_path}")


if __name__ == "__main__":
    main()
