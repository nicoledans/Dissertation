"""Train Zhang TS with stochastic possible-prior input weighting.

The prior is used as a residual hint, not a hard mask:
    guided_CT = CT * (1 + alpha * prior)

During training the guide is dropped with probability p, so the model cannot
depend exclusively on it. Loss remains BCE + Zhang TS lung-margin.
"""

import argparse
import csv
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from config import IMG_SIZE, SEED
from dataset import _augment_image_and_mask, _map_to_tensor, _mask_to_tensor, _patch_to_tensor, load_nodules_ts, patient_split
from evaluate_lungx import _load_lungx_rows
from model import NoduleClassifier
from preview_possible_nodule_mask import _make_prior, _possible_nodule_mask


def _metrics(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    pred = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "accuracy": float(np.mean(pred == labels)),
        "f1": float(f1_score(labels, pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _apply_prior(image, prior, alpha):
    guided = image * (1.0 + float(alpha) * prior)
    peak = guided.flatten(start_dim=1).max(dim=1).values.view(-1, 1, 1, 1)
    return (guided / (peak + 1e-8)).clamp(0.0, 1.0)


class PriorDropoutDataset(Dataset):
    def __init__(self, samples, augment=False, alpha=0.5, prior_dropout=0.5, force_prior=None):
        self.samples = samples
        self.labels = [int(sample["label"]) for sample in samples]
        self.augment = augment
        self.alpha = float(alpha)
        self.prior_dropout = float(prior_dropout)
        self.force_prior = force_prior

    def __len__(self):
        return len(self.samples)

    def class_weights(self):
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        return torch.tensor(min(n_neg / max(n_pos, 1), 2.0), dtype=torch.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = _patch_to_tensor(sample["patch"])
        mask = _mask_to_tensor(sample["mask"])
        prior = _map_to_tensor(sample["possible_nodule_prior"])
        if self.augment:
            image, mask, prior = _augment_image_and_mask(image, mask, prior, prior_is_binary=False)

        if self.force_prior is None:
            use_prior = bool(torch.rand(()) >= self.prior_dropout)
        else:
            use_prior = bool(self.force_prior)
        if use_prior:
            image = _apply_prior(image.unsqueeze(0), prior.unsqueeze(0), self.alpha).squeeze(0)
        return image, mask, torch.tensor(float(sample["label"]), dtype=torch.float32), torch.tensor(int(use_prior))


def _gradcam_mass(model, logits, labels, spatial_size):
    scores = model.class_scores(logits, labels)
    cam = model.differentiable_gradcam(scores, normalise=True)
    cam = F.interpolate(cam.unsqueeze(1), size=spatial_size, mode="bilinear", align_corners=False).squeeze(1)
    return cam / (cam.sum(dim=(1, 2), keepdim=True) + 1e-8)


def _zhang_lung_loss(model, logits, labels, masks):
    confidence = (2.0 * torch.abs(torch.sigmoid(logits.detach()) - 0.5)).clamp(0.0, 1.0)
    cam = _gradcam_mass(model, logits, labels, masks.shape[-2:])
    guide = masks.squeeze(1).clamp(0.0, 1.0)
    outside = 1.0 - guide
    inside_mean = (cam * guide).sum(dim=(1, 2)) / (guide.sum(dim=(1, 2)) + 1e-8)
    outside_mean = (cam * outside).sum(dim=(1, 2)) / (outside.sum(dim=(1, 2)) + 1e-8)
    return (torch.relu(outside_mean - inside_mean + 0.10) * confidence).mean()


def _evaluate(model, loader, device, criterion, gradcam=False):
    model.eval()
    labels, probs, cam_inside = [], [], []
    total_loss, total = 0.0, 0
    for images, masks, y, _use_prior in loader:
        images, masks, y = images.to(device), masks.to(device), y.to(device)
        if gradcam:
            model.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            model.class_scores(logits, y).sum().backward()
            cam = model.get_gradcam(normalise=True)
            cam = F.interpolate(cam.unsqueeze(1), size=masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
            cam_mass = cam / (cam.sum(dim=(1, 2), keepdim=True) + 1e-8)
            cam_inside.extend(((cam_mass * masks.squeeze(1)).sum(dim=(1, 2)) * 100.0).detach().cpu().tolist())
            model.clear_hooks()
        else:
            with torch.no_grad():
                logits = model(images).squeeze(1)
            model.clear_hooks()
        total_loss += float(criterion(logits, y).item()) * images.size(0)
        total += images.size(0)
        labels.extend(y.detach().cpu().numpy().astype(int).tolist())
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().tolist())
    metrics = _metrics(labels, probs)
    metrics["loss"] = total_loss / max(total, 1)
    if gradcam:
        metrics["gradcam_inside_ts_pct"] = float(np.mean(cam_inside)) if cam_inside else float("nan")
    return metrics, labels, probs


def _write_predictions(path, labels, probs):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "probability_malignant", "prediction"])
        for label, prob in zip(labels, probs):
            writer.writerow([label, prob, int(prob >= 0.5)])


def _prior_args():
    return SimpleNamespace(
        no_boundary_band=True,
        max_ts_hole_area=100,
        hu_threshold=-600.0,
        prior_dilation_iterations=0,
        blur_sigma=3.0,
        min_area=3,
        max_area=350,
        max_elongation=4.0,
        closing_radius=1,
        neighbourhood_radius=4,
        inner_erosion_radius=1,
        boundary_radius=3,
        max_internal_area=160,
        max_internal_elongation=3.0,
        restrict_to_ts_or_holes=False,
        candidate_lung_erosion_radius=0,
        exclude_ts_edge_radius=0,
        dual_hu_dense=False,
        ground_glass_min_hu=-750.0,
        ground_glass_max_hu=-300.0,
        solid_min_hu=-300.0,
        solid_max_hu=100.0,
        prior_mode="standard",
        component_area_scale=50.0,
        solid_bonus_scale=0.0,
        added_ts_border_downweight=1.0,
        pleural_dense_blob=False,
        pleural_edge_radius=5,
        pleural_inner_erosion_radius=1,
        pleural_min_area=5,
        pleural_max_area=220,
        pleural_max_elongation=2.5,
        pleural_min_mean_hu=-250.0,
        pleural_min_contrast_hu=80.0,
        pleural_contrast_ring_radius=5,
        pleural_keep_large_rim=False,
        pleural_core_hu_threshold=0.0,
        pleural_core_dilation=1,
        pleural_preserve_after_filter=False,
    )


def _lungx_samples(cache_path):
    with open(cache_path, "rb") as file:
        samples = pickle.load(file)
    args = _prior_args()
    output = []
    for sample in samples:
        item_for_prior = {
            "image": sample["image"],
            "mask": sample["ts_mask"],
            "ts_mask": sample["ts_mask"],
            "label": sample["label"],
            "patient_id": sample.get("patient_id", sample.get("scan_id")),
        }
        candidate, parts = _possible_nodule_mask(item_for_prior, args)
        output.append(
            {
                "patch": sample["image"],
                "mask": sample["ts_mask"],
                "label": sample["label"],
                "patient_id": sample.get("patient_id", sample.get("scan_id")),
                "scan_id": sample.get("scan_id"),
                "nodule_number": sample.get("nodule_number"),
                "diagnosis": sample.get("diagnosis"),
                "possible_nodule_mask": candidate.astype(np.uint8),
                "possible_nodule_prior": _make_prior(candidate, parts, args).astype(np.float32),
            }
        )
    return output


def _run_lungx_eval(args, model, device, force_prior, suffix):
    samples = _lungx_samples(args.lungx_cache)
    ds = PriorDropoutDataset(samples, augment=False, alpha=args.alpha, force_prior=force_prior)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    criterion = nn.BCEWithLogitsLoss()
    metrics, labels, probs = _evaluate(model, loader, device, criterion)
    rows = _load_lungx_rows(args.lungx_xlsx)
    pred_path = Path(args.out_dir) / f"lungx_predictions_{suffix}.csv"
    with open(pred_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["scan_id", "nodule_number", "diagnosis", "label", "probability_malignant", "prediction"],
        )
        writer.writeheader()
        for row, prob in zip(rows, probs):
            writer.writerow(
                {
                    "scan_id": row["scan_id"],
                    "nodule_number": row["nodule_number"],
                    "diagnosis": row["diagnosis"],
                    "label": row["label"],
                    "probability_malignant": prob,
                    "prediction": int(prob >= 0.5),
                }
            )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default=r"cache\cache_ts_possible_nodule_prior_best.pkl")
    parser.add_argument("--lungx-cache", default=r"cache\cache_lungx_ts_filled_dil1.pkl")
    parser.add_argument("--lungx-xlsx", default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    parser.add_argument("--out-dir", default=r"results\zhang_ts_prior_dropout_aug")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--prior-dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    samples = load_nodules_ts(args.cache_path)
    train, val, test = patient_split(samples)
    train_ds = PriorDropoutDataset(train, augment=True, alpha=args.alpha, prior_dropout=args.prior_dropout)
    val_plain = PriorDropoutDataset(val, augment=False, alpha=args.alpha, force_prior=False)
    val_guided = PriorDropoutDataset(val, augment=False, alpha=args.alpha, force_prior=True)
    test_plain = PriorDropoutDataset(test, augment=False, alpha=args.alpha, force_prior=False)
    test_guided = PriorDropoutDataset(test, augment=False, alpha=args.alpha, force_prior=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_plain_loader = DataLoader(val_plain, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_guided_loader = DataLoader(val_guided, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_plain_loader = DataLoader(test_plain, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_guided_loader = DataLoader(test_guided, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = NoduleClassifier(input_channels=3).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_ds.class_weights().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_auc = -1.0
    best_path = out_dir / "best_model.pt"
    epoch_rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        loss_sum = bce_sum = lung_sum = guide_sum = seen = 0.0
        for images, masks, labels, use_prior in train_loader:
            images, masks, labels = images.to(device), masks.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images).squeeze(1)
            bce = criterion(logits, labels)
            lung = _zhang_lung_loss(model, logits, labels, masks)
            loss = bce + 0.15 * lung
            loss.backward()
            optimizer.step()
            model.clear_hooks()
            bs = images.size(0)
            seen += bs
            loss_sum += float(loss.item()) * bs
            bce_sum += float(bce.item()) * bs
            lung_sum += float(lung.item()) * bs
            guide_sum += float(use_prior.float().mean().item()) * bs

        val_plain_metrics, _labels, _probs = _evaluate(model, val_plain_loader, device, criterion)
        val_guided_metrics, _labels, _probs = _evaluate(model, val_guided_loader, device, criterion)
        select_auc = max(val_plain_metrics["auc"], val_guided_metrics["auc"])
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "bce_loss": bce_sum / seen,
            "lung_loss": lung_sum / seen,
            "train_prior_use_rate": guide_sum / seen,
            "val_auc_plain": val_plain_metrics["auc"],
            "val_auc_guided": val_guided_metrics["auc"],
            "val_auc_selected": select_auc,
            "seconds": time.time() - start,
        }
        epoch_rows.append(row)
        print(
            f"Epoch {epoch:02d} | loss {row['train_loss']:.4f} | lung {row['lung_loss']:.4f} "
            f"| prior_use {row['train_prior_use_rate']:.2f} | val plain {row['val_auc_plain']:.4f} "
            f"| val guided {row['val_auc_guided']:.4f} | {row['seconds']:.1f}s"
        )
        if select_auc > best_auc:
            best_auc = select_auc
            torch.save(model.state_dict(), best_path)

    with open(out_dir / "epochs.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_plain_metrics, labels_plain, probs_plain = _evaluate(model, test_plain_loader, device, criterion, gradcam=True)
    test_guided_metrics, labels_guided, probs_guided = _evaluate(model, test_guided_loader, device, criterion, gradcam=True)
    _write_predictions(out_dir / "lidc_test_predictions_plain.csv", labels_plain, probs_plain)
    _write_predictions(out_dir / "lidc_test_predictions_guided.csv", labels_guided, probs_guided)

    lungx_plain = _run_lungx_eval(args, model, device, force_prior=False, suffix="plain")
    lungx_guided = _run_lungx_eval(args, model, device, force_prior=True, suffix="guided")

    lines = [
        "=== ZHANG TS PRIOR DROPOUT AUG ===",
        f"Cache: {args.cache_path}",
        f"LUNGx cache: {args.lungx_cache}",
        f"alpha: {args.alpha}",
        f"prior_dropout: {args.prior_dropout}",
        "Loss: BCE + 0.15 * confidence * Zhang TS lung margin",
        "Training input: CT or CT*(1+alpha*prior), stochastically selected per sample.",
        f"Best validation selected AUC: {best_auc:.4f}",
        "",
        "LIDC held-out test:",
        f"  plain AUC: {test_plain_metrics['auc']:.4f}; acc: {test_plain_metrics['accuracy']:.4f}; sens: {test_plain_metrics['sensitivity']:.4f}; spec: {test_plain_metrics['specificity']:.4f}; Grad-CAM TS: {test_plain_metrics['gradcam_inside_ts_pct']:.2f}%",
        f"  guided AUC: {test_guided_metrics['auc']:.4f}; acc: {test_guided_metrics['accuracy']:.4f}; sens: {test_guided_metrics['sensitivity']:.4f}; spec: {test_guided_metrics['specificity']:.4f}; Grad-CAM TS: {test_guided_metrics['gradcam_inside_ts_pct']:.2f}%",
        "",
        "LUNGx:",
        f"  plain AUC: {lungx_plain['auc']:.4f}; acc: {lungx_plain['accuracy']:.4f}; sens: {lungx_plain['sensitivity']:.4f}; spec: {lungx_plain['specificity']:.4f}",
        f"  guided AUC: {lungx_guided['auc']:.4f}; acc: {lungx_guided['accuracy']:.4f}; sens: {lungx_guided['sensitivity']:.4f}; spec: {lungx_guided['specificity']:.4f}",
        "",
        "Reference:",
        "  zhang_ts_aug LIDC AUC 0.7766; LUNGx AUC 0.6682",
        "  inference attention-transfer LUNGx AUC 0.6734",
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    with open("PROGRESS.md", "a", encoding="utf-8") as file:
        file.write(
            "\n\n## Zhang TS Prior Dropout Aug\n"
            f"- Output: `{out_dir}`\n"
            f"- LIDC plain/guided AUC: {test_plain_metrics['auc']:.4f}/{test_guided_metrics['auc']:.4f}\n"
            f"- LUNGx plain/guided AUC: {lungx_plain['auc']:.4f}/{lungx_guided['auc']:.4f}\n"
        )
    model.remove_hooks()


if __name__ == "__main__":
    main()
