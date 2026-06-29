"""Annotation-free candidate multiple-instance learning for LIDC slices.

The candidate proposals come only from the cached image-derived soft blob map.
Radiologist contours are used only for post-hoc audit columns and PNGs.
"""

import argparse
import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, maximum_filter
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from config import EPOCHS, IMG_SIZE, LR, RESULTS_DIR, SEED
from dataset import _patch_to_tensor, load_nodules_hu, patient_split
from model import NoduleClassifier
from train_zhang_soft_blob import _annotation_match_for_sample, _draw_contours


INVALID_LOGIT = -1.0e6


def _metrics(labels, probabilities):
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


def _hashable_center_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        return image[1]
    return image


def _center_image(sample):
    return np.asarray(_hashable_center_image(sample["patch"]), dtype=np.float32)


def _candidate_centers(blob_map, num_candidates, crop_size, smooth_sigma=1.5):
    blob = np.asarray(blob_map, dtype=np.float32)
    blob = np.nan_to_num(blob, nan=0.0, posinf=0.0, neginf=0.0)
    blob = np.clip(blob, 0.0, 1.0)
    smoothed = gaussian_filter(blob, sigma=float(smooth_sigma))
    if not np.any(smoothed > 0):
        return [], smoothed

    min_distance = max(4, int(round(crop_size * 0.45)))
    local_max = smoothed == maximum_filter(smoothed, size=2 * min_distance + 1)
    active = smoothed > max(1e-4, float(np.percentile(smoothed[smoothed > 0], 75.0)))
    coords = np.argwhere(local_max & active)
    if coords.size == 0:
        coords = np.argwhere(smoothed == smoothed.max())

    scored = [
        (float(smoothed[row, col]), int(row), int(col))
        for row, col in coords
    ]
    scored.sort(reverse=True)

    chosen = []
    for score, row, col in scored:
        if all((row - old_row) ** 2 + (col - old_col) ** 2 >= min_distance ** 2
               for _old_score, old_row, old_col in chosen):
            chosen.append((score, row, col))
        if len(chosen) >= num_candidates:
            break
    return chosen, smoothed


def _fallback_center(blob_map, lung_mask=None):
    blob = np.asarray(blob_map, dtype=np.float32)
    weights = np.clip(np.nan_to_num(blob, nan=0.0), 0.0, None)
    if weights.sum() <= 1e-8 and lung_mask is not None:
        weights = np.asarray(lung_mask).astype(np.float32)
    if weights.sum() <= 1e-8:
        return IMG_SIZE // 2, IMG_SIZE // 2
    rows, cols = np.indices(weights.shape)
    row = int(round(float((rows * weights).sum() / weights.sum())))
    col = int(round(float((cols * weights).sum() / weights.sum())))
    return int(np.clip(row, 0, IMG_SIZE - 1)), int(np.clip(col, 0, IMG_SIZE - 1))


def _crop_with_padding(image, center_row, center_col, crop_size):
    half = crop_size // 2
    top = int(center_row) - half
    left = int(center_col) - half
    bottom = top + crop_size
    right = left + crop_size

    src_top = max(top, 0)
    src_left = max(left, 0)
    src_bottom = min(bottom, image.shape[-2])
    src_right = min(right, image.shape[-1])

    if image.ndim == 2:
        crop = np.zeros((crop_size, crop_size), dtype=np.float32)
        crop_top = src_top - top
        crop_left = src_left - left
        crop[crop_top:crop_top + (src_bottom - src_top),
             crop_left:crop_left + (src_right - src_left)] = image[src_top:src_bottom, src_left:src_right]
    elif image.ndim == 3 and image.shape[0] == 3:
        crop = np.zeros((3, crop_size, crop_size), dtype=np.float32)
        crop_top = src_top - top
        crop_left = src_left - left
        crop[:, crop_top:crop_top + (src_bottom - src_top),
             crop_left:crop_left + (src_right - src_left)] = image[:, src_top:src_bottom, src_left:src_right]
    else:
        raise ValueError(f"Unsupported image shape for crop: {image.shape}")

    return crop, {
        "top": top,
        "left": left,
        "bottom": bottom,
        "right": right,
        "src_top": src_top,
        "src_left": src_left,
        "src_bottom": src_bottom,
        "src_right": src_right,
    }


def _candidate_tensors(sample, num_candidates, crop_size):
    image = np.asarray(sample["patch"], dtype=np.float32)
    center_image = _center_image(sample)
    blob = np.asarray(sample["soft_blob_map"], dtype=np.float32)
    lung = np.asarray(sample.get("mask", np.zeros_like(blob)), dtype=np.float32)

    centers, _smoothed = _candidate_centers(blob, num_candidates, crop_size)
    fallback_used = False
    if not centers:
        row, col = _fallback_center(blob, lung)
        centers = [(0.0, row, col)]
        fallback_used = True

    crops = []
    valid = []
    boxes = []
    scores = []
    center_rows = []
    center_cols = []
    for score, row, col in centers[:num_candidates]:
        crop, box = _crop_with_padding(image, row, col, crop_size)
        crops.append(_patch_to_tensor(crop))
        valid.append(1.0)
        boxes.append(box)
        scores.append(score)
        center_rows.append(row)
        center_cols.append(col)

    while len(crops) < num_candidates:
        crops.append(torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32))
        valid.append(0.0)
        boxes.append({
            "top": 0,
            "left": 0,
            "bottom": 0,
            "right": 0,
            "src_top": 0,
            "src_left": 0,
            "src_bottom": 0,
            "src_right": 0,
        })
        scores.append(0.0)
        center_rows.append(-1)
        center_cols.append(-1)

    box_values = [
        [box["top"], box["left"], box["bottom"], box["right"],
         box["src_top"], box["src_left"], box["src_bottom"], box["src_right"]]
        for box in boxes
    ]
    return {
        "crops": torch.stack(crops, dim=0),
        "valid": torch.tensor(valid, dtype=torch.bool),
        "boxes": torch.tensor(box_values, dtype=torch.int64),
        "candidate_scores": torch.tensor(scores, dtype=torch.float32),
        "centers": torch.tensor(list(zip(center_rows, center_cols)), dtype=torch.int64),
        "fallback": torch.tensor(float(fallback_used), dtype=torch.float32),
        "center_image": center_image,
    }


class CandidateMILDataset(Dataset):
    def __init__(self, samples, num_candidates=5, crop_size=64):
        self.samples = samples
        self.num_candidates = int(num_candidates)
        self.crop_size = int(crop_size)
        self.labels = [int(sample["label"]) for sample in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        data = _candidate_tensors(sample, self.num_candidates, self.crop_size)
        return (
            data["crops"],
            data["valid"],
            torch.tensor(sample["label"], dtype=torch.float32),
            torch.tensor(index, dtype=torch.int64),
            data["boxes"],
            data["candidate_scores"],
            data["centers"],
            data["fallback"],
        )

    def class_weights(self):
        pos = max(sum(self.labels), 1)
        neg = max(len(self.labels) - sum(self.labels), 1)
        return torch.tensor(neg / pos, dtype=torch.float32)


class CandidateMILModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.instance_model = NoduleClassifier()

    def forward(self, crops, valid):
        batch, num_candidates, channels, height, width = crops.shape
        flat = crops.view(batch * num_candidates, channels, height, width)
        instance_logits = self.instance_model(flat).view(batch, num_candidates)
        masked_logits = instance_logits.masked_fill(~valid, INVALID_LOGIT)
        bag_logits, selected = masked_logits.max(dim=1)
        return bag_logits, instance_logits, selected

    def clear_hooks(self):
        self.instance_model.clear_hooks()

    def remove_hooks(self):
        self.instance_model.remove_hooks()


def _evaluate(model, samples, criterion, device, batch_size, num_candidates, crop_size):
    dataset = CandidateMILDataset(samples, num_candidates, crop_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    probabilities, labels = [], []
    selected_indices, fallback_flags = [], []
    candidate_logits_all, candidate_scores_all = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for crops, valid, batch_labels, _indices, _boxes, candidate_scores, _centers, fallback in loader:
            crops = crops.to(device)
            valid = valid.to(device)
            batch_labels = batch_labels.to(device)
            logits, instance_logits, selected = model(crops, valid)
            loss_sum += criterion(logits, batch_labels).item() * batch_labels.numel()
            probabilities.extend(torch.sigmoid(logits).detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().int().tolist())
            selected_indices.extend(selected.detach().cpu().int().tolist())
            fallback_flags.extend(fallback.detach().cpu().tolist())
            candidate_logits_all.extend(instance_logits.detach().cpu().tolist())
            candidate_scores_all.extend(candidate_scores.detach().cpu().tolist())
            model.clear_hooks()
    return {
        "loss": loss_sum / max(len(labels), 1),
        **_metrics(labels, probabilities),
        "labels": labels,
        "probabilities": probabilities,
        "selected_indices": selected_indices,
        "fallback_flags": fallback_flags,
        "candidate_logits": candidate_logits_all,
        "candidate_scores": candidate_scores_all,
    }


def _selected_box_audit(sample, box):
    match = _annotation_match_for_sample(sample)
    if not match:
        return {
            "contour_match": False,
            "selected_box_contour_overlap_pct": float("nan"),
            "selected_center_inside_contour": False,
        }
    majority = np.asarray(match["majority"]).astype(bool)
    selected = np.zeros_like(majority, dtype=bool)
    top, left, _bottom, _right, src_top, src_left, src_bottom, src_right = [int(v) for v in box]
    del top, left, _bottom, _right
    selected[src_top:src_bottom, src_left:src_right] = True
    contour_area = max(int(majority.sum()), 1)
    center_row = (src_top + src_bottom - 1) // 2
    center_col = (src_left + src_right - 1) // 2
    return {
        "contour_match": True,
        "selected_box_contour_overlap_pct": float((selected & majority).sum() / contour_area * 100.0),
        "selected_center_inside_contour": bool(
            0 <= center_row < majority.shape[0]
            and 0 <= center_col < majority.shape[1]
            and majority[center_row, center_col]
        ),
    }


def _write_predictions(path, model, samples, device, batch_size, num_candidates, crop_size, audit_contours):
    dataset = CandidateMILDataset(samples, num_candidates, crop_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []
    model.eval()
    with torch.no_grad():
        for crops, valid, labels, indices, boxes, candidate_scores, centers, fallback in loader:
            crops = crops.to(device)
            valid_device = valid.to(device)
            logits, instance_logits, selected = model(crops, valid_device)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            instance_probs = torch.sigmoid(instance_logits).detach().cpu().numpy()
            for local_idx in range(labels.shape[0]):
                sample_index = int(indices[local_idx].item())
                sample = samples[sample_index]
                selected_idx = int(selected[local_idx].item())
                box = boxes[local_idx, selected_idx].numpy()
                audit = (
                    _selected_box_audit(sample, box)
                    if audit_contours
                    else {
                        "contour_match": False,
                        "selected_box_contour_overlap_pct": float("nan"),
                        "selected_center_inside_contour": False,
                    }
                )
                row = {
                    "sample_index": sample_index,
                    "patient_id": sample["patient_id"],
                    "label": int(labels[local_idx].item()),
                    "probability": float(probabilities[local_idx]),
                    "prediction": int(probabilities[local_idx] >= 0.5),
                    "selected_candidate": selected_idx,
                    "fallback_used": bool(fallback[local_idx].item() >= 0.5),
                    "selected_candidate_blob_score": float(candidate_scores[local_idx, selected_idx].item()),
                    "selected_center_row": int(centers[local_idx, selected_idx, 0].item()),
                    "selected_center_col": int(centers[local_idx, selected_idx, 1].item()),
                    "selected_box_top": int(box[0]),
                    "selected_box_left": int(box[1]),
                    "selected_box_bottom": int(box[2]),
                    "selected_box_right": int(box[3]),
                    **audit,
                }
                for candidate_idx in range(num_candidates):
                    row[f"candidate_{candidate_idx}_valid"] = bool(valid[local_idx, candidate_idx].item())
                    row[f"candidate_{candidate_idx}_blob_score"] = float(candidate_scores[local_idx, candidate_idx].item())
                    row[f"candidate_{candidate_idx}_probability"] = float(instance_probs[local_idx, candidate_idx])
                rows.append(row)
            model.clear_hooks()
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _cam_for_crop(model, crop, device):
    model.eval()
    model.instance_model.zero_grad(set_to_none=True)
    model.clear_hooks()
    tensor = crop.unsqueeze(0).to(device)
    logit = model.instance_model(tensor).squeeze(1)
    probability = torch.sigmoid(logit).detach().item()
    model.instance_model.class_scores(logit).sum().backward()
    cam = model.instance_model.get_gradcam()
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu().numpy()
    model.clear_hooks()
    return cam, probability


def _map_crop_cam_to_full(cam, box, crop_size):
    cam_crop = torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float()
    cam_crop = F.interpolate(
        cam_crop,
        size=(crop_size, crop_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze().numpy()
    full = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    top, left, _bottom, _right, src_top, src_left, src_bottom, src_right = [int(v) for v in box]
    crop_top = src_top - top
    crop_left = src_left - left
    h = src_bottom - src_top
    w = src_right - src_left
    if h > 0 and w > 0:
        full[src_top:src_bottom, src_left:src_right] = cam_crop[crop_top:crop_top + h, crop_left:crop_left + w]
    return full


def _save_examples(model, samples, run_dir, device, num_candidates, crop_size, filename="candidate_mil_examples.png"):
    if not samples:
        return
    dataset = CandidateMILDataset(samples, num_candidates, crop_size)
    rows = min(4, len(samples))
    fig, axes = plt.subplots(rows, 6, figsize=(22, 3.8 * rows), squeeze=False)
    for row_idx in range(rows):
        sample = samples[row_idx]
        crops, valid, label, _index, boxes, candidate_scores, centers, fallback = dataset[row_idx]
        with torch.no_grad():
            bag_logit, instance_logits, selected = model(
                crops.unsqueeze(0).to(device),
                valid.unsqueeze(0).to(device),
            )
        selected_idx = int(selected.item())
        selected_crop = crops[selected_idx]
        selected_box = boxes[selected_idx].numpy()
        selected_cam, selected_probability = _cam_for_crop(model, selected_crop, device)
        mapped_cam = _map_crop_cam_to_full(selected_cam, selected_box, crop_size)

        image = _center_image(sample)
        blob = np.asarray(sample["soft_blob_map"], dtype=np.float32)
        contour_match = _annotation_match_for_sample(sample)
        majority = np.asarray(contour_match["majority"]).astype(bool) if contour_match else None
        union = np.asarray(contour_match["union"]).astype(bool) if contour_match else None

        ax = axes[row_idx]
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title(f"CT label={int(label.item())}")

        ax[1].imshow(image, cmap="gray")
        ax[1].imshow(blob, cmap="magma", alpha=0.55, vmin=0, vmax=1)
        for candidate_idx in range(num_candidates):
            if not bool(valid[candidate_idx].item()):
                continue
            box = boxes[candidate_idx].numpy()
            color = "cyan" if candidate_idx == selected_idx else "white"
            y0, x0, y1, x1 = [int(v) for v in box[:4]]
            ax[1].plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, linewidth=1.0)
        ax[1].set_title("Blob candidates")

        ax[2].imshow(selected_crop[0].detach().cpu().numpy(), cmap="gray")
        ax[2].set_title(f"Selected crop {selected_idx}\np={selected_probability:.2f}")

        ax[3].imshow(selected_crop[0].detach().cpu().numpy(), cmap="gray")
        ax[3].imshow(selected_cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax[3].set_title("Selected-crop Grad-CAM")

        ax[4].imshow(image, cmap="gray")
        ax[4].imshow(mapped_cam, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ax[4].set_title("Mapped Grad-CAM")

        ax[5].imshow(image, cmap="gray")
        if contour_match:
            ax[5].imshow(union, cmap="Blues", alpha=0.25)
            ax[5].imshow(majority, cmap="Greens", alpha=0.42)
            _draw_contours(ax[5], majority, union)
            ax[5].set_title("Radiologist contour\n(audit only)")
        else:
            ax[5].text(0.5, 0.5, "No contour match", ha="center", va="center")
            ax[5].set_title("Radiologist contour\n(audit only)")

        for axis in ax:
            axis.set_xticks([])
            axis.set_yticks([])
        model.clear_hooks()

    fig.tight_layout()
    output_path = os.path.join(run_dir, filename)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"Saved examples -> {output_path}")


def _write_info(run_dir, args, train, val, test, device):
    train_patients = {sample["patient_id"] for sample in train}
    val_patients = {sample["patient_id"] for sample in val}
    test_patients = {sample["patient_id"] for sample in test}
    lines = [
        "Annotation-free candidate MIL experiment",
        f"Cache: {os.path.abspath(args.cache_path)}",
        f"Device: {device}",
        f"Epochs: {args.epochs}",
        f"Batch size: {args.batch_size}",
        f"Learning rate: {args.lr}",
        f"Num candidates: {args.num_candidates}",
        f"Crop size: {args.crop_size}",
        "Candidate source: image-derived soft_blob_map only",
        "Candidate selection: smoothed high-response non-overlapping peaks",
        "Pooling: differentiable max over valid candidate logits",
        "Loss: weighted BCEWithLogitsLoss on bag logit only",
        "No Grad-CAM loss, no Zhang margin, no augmentation",
        "Radiologist contours: audit/PNG only, never training/model selection",
        f"Train samples: {len(train)} patients={len(train_patients)}",
        f"Val samples: {len(val)} patients={len(val_patients)}",
        f"Test samples: {len(test)} patients={len(test_patients)}",
        f"Train/val patient overlap: {len(train_patients & val_patients)}",
        f"Train/test patient overlap: {len(train_patients & test_patients)}",
        f"Val/test patient overlap: {len(val_patients & test_patients)}",
    ]
    with open(os.path.join(run_dir, "candidate_mil_info.txt"), "w") as file:
        file.write("\n".join(lines) + "\n")


def _train(args, train, val, run_dir, device):
    torch.manual_seed(args.training_seed)
    np.random.seed(args.training_seed)
    train_dataset = CandidateMILDataset(train, args.num_candidates, args.crop_size)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = CandidateMILModel().to(device)
    pos_weight = train_dataset.class_weights().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_auc = -np.inf
    best_epoch = 0
    best_path = os.path.join(run_dir, "candidate_mil_model.pt")
    rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for crops, valid, labels, _indices, _boxes, _candidate_scores, _centers, _fallback in train_loader:
            crops = crops.to(device)
            valid = valid.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _instance_logits, _selected = model(crops, valid)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.numel()
            train_count += labels.numel()
            model.clear_hooks()

        train_eval = _evaluate(
            model, train, criterion, device, args.batch_size, args.num_candidates, args.crop_size
        )
        val_eval = _evaluate(
            model, val, criterion, device, args.batch_size, args.num_candidates, args.crop_size
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / max(train_count, 1),
            "train_auc": train_eval["auc"],
            "train_accuracy": train_eval["accuracy"],
            "train_f1": train_eval["f1"],
            "val_loss": val_eval["loss"],
            "val_auc": val_eval["auc"],
            "val_accuracy": val_eval["accuracy"],
            "val_f1": val_eval["f1"],
            "val_sensitivity": val_eval["sensitivity"],
            "val_specificity": val_eval["specificity"],
            "val_fallback_rate": float(np.mean(val_eval["fallback_flags"])) if val_eval["fallback_flags"] else 0.0,
        }
        rows.append(row)
        if np.isfinite(row["val_auc"]) and row["val_auc"] > best_auc:
            best_auc = row["val_auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)
        print(
            f"epoch={epoch:02d} train_auc={row['train_auc']:.4f} "
            f"val_auc={row['val_auc']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_f1={row['val_f1']:.4f}"
        )

    with open(os.path.join(run_dir, "candidate_mil_epochs.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return best_path, best_epoch, best_auc


def _write_test_results(path, best_epoch, best_auc, test_eval):
    lines = [
        "=== HELD-OUT TEST: ANNOTATION-FREE CANDIDATE MIL ===",
        f"Best validation epoch: {best_epoch}",
        f"Best validation AUC: {best_auc:.4f}",
        f"AUC: {test_eval['auc']:.4f}",
        f"Accuracy: {test_eval['accuracy']:.4f}",
        f"F1: {test_eval['f1']:.4f}",
        f"Sensitivity: {test_eval['sensitivity']:.4f}",
        f"Specificity: {test_eval['specificity']:.4f}",
        f"Fallback candidate rate: {float(np.mean(test_eval['fallback_flags'])):.4f}",
    ]
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cache-path", default="cache/cache_hu_soft_blobs.pkl")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--training-seed", type=int, default=SEED)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--skip-contour-audit", action="store_true")
    args = parser.parse_args()
    if args.num_candidates < 1:
        parser.error("--num-candidates must be at least 1")
    if args.crop_size < 8 or args.crop_size > IMG_SIZE:
        parser.error(f"--crop-size must be between 8 and {IMG_SIZE}")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.limit_samples is not None and args.limit_samples < 30:
        parser.error("--limit-samples must be at least 30")
    return args


def main():
    args = _parse_args()
    run_id = args.run_id or datetime.now().strftime("candidate_mil_%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=False)

    samples = load_nodules_hu(args.cache_path)
    missing = [idx for idx, sample in enumerate(samples) if "soft_blob_map" not in sample]
    if missing:
        raise ValueError(
            f"{len(missing)} samples are missing soft_blob_map. "
            "Run build_soft_blob_cache.py first."
        )
    if args.limit_samples is not None:
        rng = np.random.default_rng(args.training_seed)
        keep = rng.choice(len(samples), size=min(args.limit_samples, len(samples)), replace=False)
        samples = [samples[index] for index in sorted(keep)]

    train, val, test = patient_split(samples)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _write_info(run_dir, args, train, val, test, device)

    best_path, best_epoch, best_auc = _train(args, train, val, run_dir, device)

    model = CandidateMILModel().to(device)
    model.load_state_dict(torch.load(best_path, map_location=device))
    pos_weight = CandidateMILDataset(train, args.num_candidates, args.crop_size).class_weights().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    eval_samples = test if args.evaluate_test else val
    split_name = "test" if args.evaluate_test else "val"
    eval_result = _evaluate(
        model, eval_samples, criterion, device, args.batch_size, args.num_candidates, args.crop_size
    )
    result_path = os.path.join(run_dir, f"candidate_mil_{split_name}_results.txt")
    _write_test_results(result_path, best_epoch, best_auc, eval_result)

    prediction_path = os.path.join(run_dir, "candidate_mil_predictions.csv")
    _write_predictions(
        prediction_path,
        model,
        eval_samples,
        device,
        args.batch_size,
        args.num_candidates,
        args.crop_size,
        audit_contours=not args.skip_contour_audit,
    )
    print(f"Saved predictions -> {prediction_path}")

    _save_examples(
        model,
        eval_samples,
        run_dir,
        device,
        args.num_candidates,
        args.crop_size,
        filename="candidate_mil_examples.png",
    )
    model.remove_hooks()


if __name__ == "__main__":
    main()
