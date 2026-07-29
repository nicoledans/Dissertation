"""Evaluate progressive-prior Grad-CAM as a no-retraining input guide for Zhang.

For each image:
  1. Get a predicted-class Grad-CAM from the progressive-prior model.
  2. Reweight the CT image as ct * (1 + alpha * cam), then renormalise.
  3. Feed the weighted image to the Zhang model.

This is inference-only. No labels are used to generate the guiding Grad-CAM.
"""

import argparse
import csv
import os
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from audit_staged_gradcam_contour_overlap import _load_cache, _match_samples
from config import IMG_SIZE
from dataset import _patch_to_tensor, load_nodules_ts, patient_split
from evaluate_lungx import (
    _case_dirs,
    _dicom_index,
    _load_hu_slice,
    _load_lungx_rows,
    _prepare_image,
)
from model import NoduleClassifier


def _load_model(path, device):
    model = NoduleClassifier(input_channels=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _metrics(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "accuracy": float(np.mean(preds == labels)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _progressive_cam(progressive_model, image, device):
    progressive_model.zero_grad(set_to_none=True)
    progressive_model.clear_hooks()
    image = image.to(device)
    logit = progressive_model(image).squeeze(1)
    score = progressive_model.class_scores(logit)
    score.sum().backward()
    cam = progressive_model.get_gradcam(normalise=True)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    probability = torch.sigmoid(logit).detach().cpu().numpy()
    progressive_model.clear_hooks()
    return cam.detach(), probability


def _guided_image(image, cam, alpha):
    guided = image * (1.0 + float(alpha) * cam.unsqueeze(1))
    peak = guided.flatten(start_dim=1).max(dim=1).values.view(-1, 1, 1, 1)
    return (guided / (peak + 1e-8)).clamp(0.0, 1.0)


@torch.no_grad()
def _zhang_probs(zhang_model, images):
    logits = zhang_model(images).squeeze(1)
    return torch.sigmoid(logits).detach().cpu().numpy()


def _evaluate_tensor_records(records, alphas, zhang_model, progressive_model, device):
    rows_by_alpha = {alpha: [] for alpha in alphas}
    for index, record in enumerate(records, start=1):
        if index % 50 == 0 or index == len(records):
            print(f"  scored {index}/{len(records)}")
        image = record["image"].to(device)
        cam, prog_prob = _progressive_cam(progressive_model, image, device)
        for alpha in alphas:
            guided = _guided_image(image, cam, alpha)
            prob = float(_zhang_probs(zhang_model, guided)[0])
            row = {
                **record["meta"],
                "alpha": alpha,
                "progressive_probability_malignant": float(prog_prob[0]),
                "zhang_guided_probability_malignant": prob,
                "prediction": int(prob >= 0.5),
            }
            if row.get("label") is not None:
                row["correct"] = int(row["prediction"] == int(row["label"]))
            rows_by_alpha[alpha].append(row)
    return rows_by_alpha


def _lidc_records(cache_path):
    nodules = load_nodules_ts(cache_path)
    _train, _val, test = patient_split(nodules)
    records = []
    for idx, item in enumerate(test):
        records.append(
            {
                "image": _patch_to_tensor(item["patch"]).unsqueeze(0),
                "meta": {
                    "dataset": "lidc_test",
                    "row_index": idx,
                    "patient_id": item["patient_id"],
                    "label": int(item["label"]),
                },
            }
        )
    return records


def _lidc_matched_malignant_records(cache_path):
    samples = _load_cache(cache_path)
    matches = _match_samples(samples, "test", malignant_only=True, max_matches=0)
    records = []
    for idx, match in enumerate(matches):
        sample = match["cache_sample"]
        records.append(
            {
                "image": _patch_to_tensor(sample["image"]).unsqueeze(0),
                "meta": {
                    "dataset": "lidc_matched_malignant",
                    "row_index": idx,
                    "patient_id": match["patient_id"],
                    "scan_id": match["scan_id"],
                    "group_index": match["group_index"],
                    "slice_idx": match["slice_idx"],
                    "label": int(match["label"]),
                },
            }
        )
    return records


def _lungx_records(args):
    image_root = os.path.join(args.lungx_manifest_root, "SPIE-AAPM Lung CT Challenge")
    cases = _case_dirs(image_root)
    records = _load_lungx_rows(args.lungx_xlsx)
    dicom_cache = {}
    output = []
    for record in records:
        case_dir = cases.get(record["scan_id"])
        if case_dir is None:
            raise FileNotFoundError(f"Missing DICOM directory for {record['scan_id']}")
        if record["scan_id"] not in dicom_cache:
            dicom_cache[record["scan_id"]] = _dicom_index(case_dir)
        entries, by_instance = dicom_cache[record["scan_id"]]
        dicom_path = by_instance.get(record["center_image"])
        if dicom_path is None:
            index = max(0, min(record["center_image"] - 1, len(entries) - 1))
            dicom_path = entries[index][2]
        output.append(
            {
                "image": _prepare_image(_load_hu_slice(dicom_path)).unsqueeze(0),
                "meta": {
                    "dataset": "lungx",
                    "scan_id": record["scan_id"],
                    "nodule_number": record["nodule_number"],
                    "center_image": record["center_image"],
                    "diagnosis": record["diagnosis"],
                    "label": record["label"],
                    "dicom_path": dicom_path,
                },
            }
        )
    return output


def _write_prediction_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summarise_dataset(dataset_name, rows_by_alpha):
    lines = [f"## {dataset_name}"]
    summary_rows = []
    for alpha, rows in rows_by_alpha.items():
        labelled = [row for row in rows if row.get("label") is not None]
        labels = [int(row["label"]) for row in labelled]
        probs = [float(row["zhang_guided_probability_malignant"]) for row in labelled]
        metrics = _metrics(labels, probs)
        label_counts = dict(Counter(labels))
        summary_rows.append(
            {
                "dataset": dataset_name,
                "alpha": alpha,
                "n": len(labelled),
                "label_counts": label_counts,
                **metrics,
            }
        )
        lines.extend(
            [
                f"alpha={alpha:g}",
                f"  N: {len(labelled)}; labels: {label_counts}",
                f"  AUC: {metrics['auc']:.4f}",
                f"  Accuracy @0.5: {metrics['accuracy']:.4f}",
                f"  F1 @0.5: {metrics['f1']:.4f}",
                f"  Sensitivity @0.5: {metrics['sensitivity']:.4f}",
                f"  Specificity @0.5: {metrics['specificity']:.4f}",
                f"  Confusion: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}",
            ]
        )
        if dataset_name == "lidc_matched_malignant":
            correct = int(sum(int(row["correct"]) for row in labelled))
            lines.append(f"  Correct malignant: {correct}/{len(labelled)}")
    return summary_rows, lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidc-cache", default=r"cache\cache_ts_filled_dil1.pkl")
    parser.add_argument("--zhang-checkpoint", default=r"results\zhang_ts_aug_full\selected_model.pt")
    parser.add_argument(
        "--progressive-checkpoint",
        default=r"results\zhang_ts_progressive_prior_best_aug\best_model.pt",
    )
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5])
    parser.add_argument(
        "--lungx-manifest-root",
        default=r"C:\repo\manifest-cgqtDj7Y2699835271585651107",
    )
    parser.add_argument(
        "--lungx-xlsx",
        default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx",
    )
    parser.add_argument("--out-dir", default=r"results\attention_transfer_prior_to_zhang")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    zhang_model = _load_model(args.zhang_checkpoint, device)
    progressive_model = _load_model(args.progressive_checkpoint, device)

    datasets = [
        ("lidc_test", _lidc_records(args.lidc_cache)),
        ("lidc_matched_malignant", _lidc_matched_malignant_records(args.lidc_cache)),
        ("lungx", _lungx_records(args)),
    ]

    all_summary_rows = []
    report_lines = [
        "=== ATTENTION TRANSFER: PROGRESSIVE PRIOR -> ZHANG INPUT ===",
        "Guiding CAM is predicted-class Grad-CAM from progressive_prior; labels are not used for guidance.",
        f"Zhang checkpoint: {args.zhang_checkpoint}",
        f"Progressive checkpoint: {args.progressive_checkpoint}",
        f"LIDC cache: {args.lidc_cache}",
        f"Alphas: {args.alphas}",
        "",
        "Reference:",
        "zhang_ts_aug alone: 58/90 malignant; LUNGx AUC 0.6682",
        "simple ensemble zhang+progressive: 71/90 malignant; LUNGx AUC 0.6569",
        "progressive_prior alone: 73/90 malignant; LUNGx AUC 0.5586",
        "",
    ]

    for name, records in datasets:
        print(f"Evaluating {name}: {len(records)} samples")
        rows_by_alpha = _evaluate_tensor_records(
            records,
            args.alphas,
            zhang_model,
            progressive_model,
            device,
        )
        for alpha, rows in rows_by_alpha.items():
            _write_prediction_csv(out_dir / f"{name}_alpha_{alpha:g}_predictions.csv", rows)
        summary_rows, lines = _summarise_dataset(name, rows_by_alpha)
        all_summary_rows.extend(summary_rows)
        report_lines.extend(lines)
        report_lines.append("")

    summary_csv = out_dir / "attention_transfer_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "dataset",
            "alpha",
            "n",
            "label_counts",
            "auc",
            "accuracy",
            "f1",
            "sensitivity",
            "specificity",
            "tp",
            "tn",
            "fp",
            "fn",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summary_rows)

    report_path = out_dir / "attention_transfer_summary.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    print(f"Saved summary: {report_path}")
    print(f"Saved summary CSV: {summary_csv}")

    progress_path = Path("PROGRESS.md")
    with progress_path.open("a", encoding="utf-8") as file:
        file.write(
            "\n\n## Attention Transfer Prior to Zhang\n"
            f"- Output: `{out_dir}`\n"
            f"- Alphas: {args.alphas}\n"
            "- Completed inference-only evaluation on LIDC test, LIDC matched malignant, and LUNGx.\n"
        )

    zhang_model.remove_hooks()
    progressive_model.remove_hooks()


if __name__ == "__main__":
    main()
