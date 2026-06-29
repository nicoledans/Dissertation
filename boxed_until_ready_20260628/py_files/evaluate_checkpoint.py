"""Evaluate one already-selected checkpoint on the held-out patient test split."""

import argparse
import csv
import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from config import BATCH_SIZE, TRAIN_CACHE_PATH
from dataset import LIDCDataset, load_nodules_hu, patient_split
from model import NoduleClassifier


DEFAULT_CHECKPOINT = "results/2026-06-07_adaptive_search/selected_model.pt"
DEFAULT_OUTPUT_DIR = "results/2026-06-07_adaptive_search"
BLANK_CAM_EPS = 1e-8


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _classification(model, test_nodules, device, batch_size):
    loader = DataLoader(
        LIDCDataset(test_nodules), batch_size=batch_size, shuffle=False
    )
    probabilities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for images, _masks, batch_labels in loader:
            logits = model(images.to(device)).squeeze(1)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.tolist())
            model.clear_hooks()

    predictions = np.asarray(probabilities) >= 0.5
    labels_array = np.asarray(labels)
    tn, fp, fn, tp = confusion_matrix(
        labels_array, predictions, labels=[0, 1]
    ).ravel()
    metrics = {
        "auc": float(roc_auc_score(labels_array, probabilities)),
        "accuracy": float(np.mean(predictions == labels_array)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
    }
    return metrics, probabilities, predictions.astype(int).tolist()


def _alignment(model, test_nodules, device):
    loader = DataLoader(LIDCDataset(test_nodules), batch_size=1, shuffle=False)
    inside_fractions = []
    mask_areas = []
    blank_count = 0
    model.eval()
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
                cam.unsqueeze(1),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            mask = masks.squeeze(1)
            inside_fractions.append(
                (cam * mask).sum().item() / (cam.sum().item() + 1e-8)
            )
            mask_areas.append(mask.mean().item())
            model.clear_hooks()

    mean_inside = float(np.mean(inside_fractions))
    mean_area = float(np.mean(mask_areas))
    return {
        "alignment": mean_inside,
        "alignment_std": float(np.std(inside_fractions)),
        "mask_area": mean_area,
        "normalized_alignment": mean_inside / mean_area,
        "blank_count": blank_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a selected checkpoint once on the held-out test split."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    report_path = os.path.join(args.output_dir, "adaptive_test_results.txt")
    predictions_path = os.path.join(args.output_dir, "adaptive_test_predictions.csv")
    if (os.path.exists(report_path) or os.path.exists(predictions_path)) and not args.force:
        parser.error(
            "Test outputs already exist. Refusing to overwrite them; use --force only "
            "if you intentionally need to repeat the same evaluation."
        )
    if not os.path.exists(args.checkpoint):
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    os.makedirs(args.output_dir, exist_ok=True)
    nodules = load_nodules_hu(args.cache_path)
    train, validation, test = patient_split(nodules)
    patient_sets = [
        {sample["patient_id"] for sample in split}
        for split in (train, validation, test)
    ]
    if (
        patient_sets[0] & patient_sets[1]
        or patient_sets[0] & patient_sets[2]
        or patient_sets[1] & patient_sets[2]
    ):
        raise RuntimeError("Patient leakage detected.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleClassifier().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    metrics, probabilities, predictions = _classification(
        model, test, device, args.batch_size
    )
    alignment = _alignment(model, test, device)
    model.remove_hooks()

    with open(predictions_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["patient_id", "label", "probability", "prediction"]
        )
        writer.writeheader()
        for sample, probability, prediction in zip(test, probabilities, predictions):
            writer.writerow(
                {
                    "patient_id": sample["patient_id"],
                    "label": sample["label"],
                    "probability": probability,
                    "prediction": prediction,
                }
            )

    lines = [
        "=== HELD-OUT TEST: SELECTED ADAPTIVE GRADCAM MARGIN MODEL ===",
        f"Checkpoint: {os.path.abspath(args.checkpoint)}",
        f"Checkpoint SHA256: {_sha256(args.checkpoint)}",
        f"Cache: {os.path.abspath(args.cache_path)}",
        f"Device: {device}",
        f"Test samples: {len(test)}",
        f"Test patients: {len(patient_sets[2])}",
        "Patient overlap across train/validation/test: none",
        "",
        f"Test AUC: {metrics['auc']:.4f}",
        f"Test accuracy: {metrics['accuracy']:.4f}",
        f"Test F1: {metrics['f1']:.4f}",
        f"Test sensitivity: {metrics['sensitivity']:.4f}",
        f"Test specificity: {metrics['specificity']:.4f}",
        "",
        f"Mean Grad-CAM inside lung: {alignment['alignment'] * 100:.2f}%",
        f"Grad-CAM inside-lung std: {alignment['alignment_std'] * 100:.2f}%",
        f"Mean lung-mask area: {alignment['mask_area'] * 100:.2f}%",
        f"Normalized alignment: {alignment['normalized_alignment']:.4f}",
        f"Blank Grad-CAMs: {alignment['blank_count']}/{len(test)}",
        "",
        "Reference baseline test AUC: 0.6719",
        "Reference baseline test Grad-CAM inside lung: 24.5%",
    ]
    text = "\n".join(lines)
    with open(report_path, "w") as file:
        file.write(text + "\n")
    print(text)
    print(f"\nSaved: {report_path}")
    print(f"Saved: {predictions_path}")


if __name__ == "__main__":
    main()
