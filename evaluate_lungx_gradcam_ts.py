"""Evaluate LUNGx classification and Grad-CAM inside TS lung mask.

LUNGx has nodule centre annotations but no contours. This script measures
Grad-CAM mass inside the TotalSegmentator lung mask only, not nodule overlap.
"""

import argparse
import csv
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import IMG_SIZE
from model import NoduleClassifier


def _image_tensor(image):
    array = np.asarray(image, dtype=np.float32)
    if array.ndim == 2:
        array = np.stack([array, array, array], axis=0)
    elif array.ndim == 3 and array.shape[0] != 3:
        array = np.transpose(array, (2, 0, 1))
    return torch.from_numpy(array).float()


def _mask_array(sample, mask_key):
    mask = np.asarray(sample[mask_key], dtype=np.float32)
    if mask.ndim == 3:
        mask = mask[1] if mask.shape[0] == 3 else mask.squeeze()
    return (mask > 0.5).astype(np.float32)


def _metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    preds = (probs >= 0.5).astype(np.int64)
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tp = int(((preds == 1) & (labels == 1)).sum())
    return {
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _gradcam_for_target(model, image, logit, label, target_mode):
    model.zero_grad(set_to_none=True)
    model.clear_hooks()
    logit = model(image).squeeze(1)
    if target_mode == "predicted":
        score = model.class_scores(logit, labels=None)
    elif target_mode == "label":
        score = model.class_scores(logit, labels=label)
    elif target_mode == "malignant":
        score = logit
    else:
        raise ValueError(target_mode)
    score.sum().backward()
    cam = model.get_gradcam(normalise=True)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    model.clear_hooks()
    return cam.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=r"cache\cache_lungx_ts_filled_dil1.pkl")
    parser.add_argument("--checkpoint", default=r"results\zhang_ts_aug_full\selected_model.pt")
    parser.add_argument("--out-dir", default=r"results\lungx_eval_zhang_ts_aug_full_gradcam_ts")
    parser.add_argument("--mask-key", default="ts_mask")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.cache, "rb") as file:
        samples = pickle.load(file)

    model = NoduleClassifier(input_channels=3).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    rows = []
    labels = []
    probs = []
    pred_inside = []
    label_inside = []
    malignant_inside = []
    mask_areas = []
    blank_pred = 0

    for idx, sample in enumerate(samples):
        image = _image_tensor(sample["image"]).unsqueeze(0).to(device)
        label_value = int(sample["label"])
        label = torch.tensor([label_value], dtype=torch.float32, device=device)
        mask = torch.from_numpy(_mask_array(sample, args.mask_key)).unsqueeze(0).to(device)

        with torch.no_grad():
            logit = model(image).squeeze(1)
            prob = torch.sigmoid(logit).item()

        cam_pred = _gradcam_for_target(model, image, logit, label, "predicted")
        cam_label = _gradcam_for_target(model, image, logit, label, "label")
        cam_mal = _gradcam_for_target(model, image, logit, label, "malignant")

        def inside_fraction(cam):
            total = float(cam.sum().item())
            if total <= 1e-8:
                return float("nan")
            return float((cam * mask).sum().item() / (total + 1e-8))

        pred_frac = inside_fraction(cam_pred)
        label_frac = inside_fraction(cam_label)
        mal_frac = inside_fraction(cam_mal)
        if not np.isfinite(pred_frac):
            blank_pred += 1

        area = float(mask.mean().item())
        labels.append(label_value)
        probs.append(prob)
        pred_inside.append(pred_frac)
        label_inside.append(label_frac)
        malignant_inside.append(mal_frac)
        mask_areas.append(area)
        rows.append(
            {
                "index": idx,
                "scan_id": sample.get("scan_id", sample.get("patient_id", "")),
                "nodule_number": sample.get("nodule_number", ""),
                "label": label_value,
                "probability": prob,
                "prediction": int(prob >= 0.5),
                "correct": int((prob >= 0.5) == bool(label_value)),
                "gradcam_inside_ts_predicted_pct": pred_frac * 100.0 if np.isfinite(pred_frac) else float("nan"),
                "gradcam_inside_ts_label_pct": label_frac * 100.0 if np.isfinite(label_frac) else float("nan"),
                "gradcam_inside_ts_malignant_pct": mal_frac * 100.0 if np.isfinite(mal_frac) else float("nan"),
                "ts_mask_area_pct": area * 100.0,
                "center_image": sample.get("center_image", ""),
                "diagnosis": sample.get("diagnosis", ""),
            }
        )

    metrics = _metrics(labels, probs)
    mean_pred = float(np.nanmean(pred_inside))
    std_pred = float(np.nanstd(pred_inside))
    mean_label = float(np.nanmean(label_inside))
    mean_mal = float(np.nanmean(malignant_inside))
    mean_area = float(np.mean(mask_areas))
    norm_pred = float(mean_pred / (mean_area + 1e-8))

    csv_path = out_dir / "lungx_gradcam_ts_rows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "=== LUNGx Grad-CAM Inside TS Mask Evaluation ===",
        "LUNGx has centre annotations, not contours; this is TS lung-mask CAM alignment only.",
        f"Cache: {os.path.abspath(args.cache)}",
        f"Checkpoint: {os.path.abspath(args.checkpoint)}",
        f"Mask key: {args.mask_key}",
        f"Device: {device}",
        f"Samples: {len(samples)}",
        f"Label counts: {{0: {int((np.asarray(labels)==0).sum())}, 1: {int((np.asarray(labels)==1).sum())}}}",
        "",
        f"AUC: {metrics['auc']:.4f}",
        f"Accuracy @0.5: {metrics['accuracy']:.4f}",
        f"F1 @0.5: {metrics['f1']:.4f}",
        f"Sensitivity @0.5: {metrics['sensitivity']:.4f}",
        f"Specificity @0.5: {metrics['specificity']:.4f}",
        f"Confusion: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}",
        "",
        f"Mean Grad-CAM inside TS mask, predicted-class CAM: {mean_pred * 100:.2f}%",
        f"Grad-CAM inside TS std, predicted-class CAM: {std_pred * 100:.2f}%",
        f"Mean Grad-CAM inside TS mask, true-label CAM: {mean_label * 100:.2f}%",
        f"Mean Grad-CAM inside TS mask, malignant CAM: {mean_mal * 100:.2f}%",
        f"Mean TS mask area: {mean_area * 100:.2f}%",
        f"Normalized alignment, predicted-class CAM: {norm_pred:.4f}",
        f"Blank predicted-class Grad-CAMs: {blank_pred}/{len(samples)}",
        "",
        f"CSV: {csv_path}",
    ]
    summary_path = out_dir / "lungx_gradcam_ts_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
