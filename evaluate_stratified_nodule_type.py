"""Evaluate model performance stratified by approximate nodule HU type.

Nodule type is derived from a 15px circle around the nodule centre:
  ground_glass: mean HU < -500
  intermediate: -500 <= mean HU < -300
  solid: mean HU >= -300

For LIDC, the circle is centred on the LIDC majority-contour centroid. For
LUNGx, it is centred on the provided nodule centre coordinate. This is
evaluation-only; no retraining is performed.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from audit_staged_gradcam_contour_overlap import _load_cache
from config import IMG_SIZE
from dataset import _patch_to_tensor, patient_split
from model import NoduleClassifier


def _load_model(path, device):
    model = NoduleClassifier(input_channels=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _center_image(sample):
    image = np.asarray(sample["image"], dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image[1]
    return np.clip(image, 0.0, 1.0)


def _circle_mask(shape, center_row, center_col, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - float(center_row)) ** 2 + (xx - float(center_col)) ** 2 <= float(radius) ** 2


def _resize_mask(mask, shape):
    mask = np.asarray(mask, dtype=np.float32)
    if tuple(mask.shape) == tuple(shape):
        return mask > 0.5
    tensor = torch.from_numpy(mask).view(1, 1, *mask.shape)
    resized = F.interpolate(tensor, size=shape, mode="nearest").squeeze().numpy()
    return resized > 0.5


def _mean_hu_in_circle(image, center_row, center_col, radius):
    hu = image.astype(np.float32) * 1400.0 - 1000.0
    roi = _circle_mask(hu.shape, center_row, center_col, radius)
    if not np.any(roi):
        return float("nan")
    return float(np.mean(hu[roi]))


def _type_from_hu(mean_hu):
    if mean_hu < -500:
        return "ground_glass"
    if mean_hu < -300:
        return "intermediate"
    return "solid"


def _lidc_records(cache_path, radius):
    samples = _load_cache(cache_path)
    _train, _val, test = patient_split(samples)
    records = []
    for idx, sample in enumerate(test):
        image = _center_image(sample)
        if "nodule_center_rc" in sample:
            center_row = float(sample["nodule_center_rc"][0])
            center_col = float(sample["nodule_center_rc"][1])
        else:
            center_row, center_col = image.shape[0] / 2.0, image.shape[1] / 2.0
        mean_hu = _mean_hu_in_circle(image, center_row, center_col, radius)
        records.append(
            {
                "dataset": "lidc",
                "row_index": idx,
                "patient_id": sample["patient_id"],
                "slice_idx": sample.get("slice_idx"),
                "label": int(sample["label"]),
                "center_row": center_row,
                "center_col": center_col,
                "mean_hu": mean_hu,
                "nodule_type": _type_from_hu(mean_hu),
                "image": _patch_to_tensor(sample["image"]).unsqueeze(0),
            }
        )
    return records


def _lungx_records(cache_path, radius):
    import pickle

    with open(cache_path, "rb") as file:
        samples = pickle.load(file)
    records = []
    for idx, sample in enumerate(samples):
        image = _center_image(sample)
        center_row = float(sample["center_y_224"])
        center_col = float(sample["center_x_224"])
        mean_hu = _mean_hu_in_circle(image, center_row, center_col, radius)
        records.append(
            {
                "dataset": "lungx",
                "row_index": idx,
                "scan_id": sample.get("scan_id"),
                "nodule_number": sample.get("nodule_number"),
                "label": int(sample["label"]),
                "center_row": center_row,
                "center_col": center_col,
                "mean_hu": mean_hu,
                "nodule_type": _type_from_hu(mean_hu),
                "image": _patch_to_tensor(sample["image"]).unsqueeze(0),
            }
        )
    return records


@torch.no_grad()
def _predict(model, image, device):
    model.clear_hooks()
    logit = model(image.to(device)).squeeze(1)
    prob = float(torch.sigmoid(logit)[0].detach().cpu())
    model.clear_hooks()
    return prob


def _progressive_cam(progressive_model, image, device):
    progressive_model.zero_grad(set_to_none=True)
    progressive_model.clear_hooks()
    image = image.to(device)
    logit = progressive_model(image).squeeze(1)
    score = progressive_model.class_scores(logit)
    score.sum().backward()
    cam = progressive_model.get_gradcam(normalise=True)
    cam = F.interpolate(cam.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(1)
    progressive_model.clear_hooks()
    return cam.detach()


def _guided_image(image, cam, alpha):
    guided = image * (1.0 + float(alpha) * cam.unsqueeze(1))
    peak = guided.flatten(start_dim=1).max(dim=1).values.view(-1, 1, 1, 1)
    return (guided / (peak + 1e-8)).clamp(0.0, 1.0)


def _score_records(records, models, device, attention_alpha):
    zhang = models["zhang_ts_aug"]
    base = models["base_aug"]
    progressive = models["progressive_prior"]
    rows = []
    for index, record in enumerate(records, start=1):
        if index % 50 == 0 or index == len(records):
            print(f"  scored {record['dataset']} {index}/{len(records)}")
        image = record["image"]
        zhang_prob = _predict(zhang, image, device)
        base_prob = _predict(base, image, device)
        cam = _progressive_cam(progressive, image, device)
        guided = _guided_image(image.to(device), cam, attention_alpha)
        attn_prob = _predict(zhang, guided, device)
        row = {key: value for key, value in record.items() if key != "image"}
        row.update(
            {
                "zhang_ts_aug_probability": zhang_prob,
                "base_aug_probability": base_prob,
                "attention_transfer_probability": attn_prob,
            }
        )
        rows.append(row)
    return rows


def _metrics(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    if len(labels) == 0:
        return None
    preds = (probs >= 0.5).astype(int)
    if len(set(labels.tolist())) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(labels, probs))
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "label_counts": dict(Counter(labels.tolist())),
        "auc": auc,
        "accuracy": float(np.mean(preds == labels)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def _summarise(rows):
    models = [
        ("zhang_ts_aug", "zhang_ts_aug_probability"),
        ("attention_transfer_alpha0p5", "attention_transfer_probability"),
        ("base_aug", "base_aug_probability"),
    ]
    datasets = ["lidc", "lungx"]
    nodule_types = ["all", "solid", "intermediate", "ground_glass"]
    summary_rows = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for model_name, prob_key in models:
            for nodule_type in nodule_types:
                subset = dataset_rows if nodule_type == "all" else [
                    row for row in dataset_rows if row["nodule_type"] == nodule_type
                ]
                metrics = _metrics(
                    [int(row["label"]) for row in subset],
                    [float(row[prob_key]) for row in subset],
                )
                if metrics is None:
                    continue
                summary_rows.append(
                    {
                        "dataset": dataset,
                        "model": model_name,
                        "nodule_type": nodule_type,
                        "mean_hu": float(np.mean([float(row["mean_hu"]) for row in subset])) if subset else float("nan"),
                        **metrics,
                    }
                )
    return summary_rows


def _fmt(value):
    if value != value:
        return "nan"
    return f"{float(value):.4f}"


def _table_lines(summary_rows, dataset):
    lines = [f"## {dataset.upper()}"]
    counts = {
        row["nodule_type"]: row["n"]
        for row in summary_rows
        if row["dataset"] == dataset and row["model"] == "zhang_ts_aug"
    }
    lines.append(
        "Type counts: "
        + ", ".join(f"{key}={counts.get(key, 0)}" for key in ["all", "solid", "intermediate", "ground_glass"])
    )
    lines.append("Model | All AUC | Solid AUC | Intermediate AUC | GGO AUC")
    lines.append("--- | ---: | ---: | ---: | ---:")
    for model in ["zhang_ts_aug", "attention_transfer_alpha0p5", "base_aug"]:
        by_type = {
            row["nodule_type"]: row
            for row in summary_rows
            if row["dataset"] == dataset and row["model"] == model
        }
        lines.append(
            f"{model} | {_fmt(by_type['all']['auc'])} | {_fmt(by_type['solid']['auc'])} | "
            f"{_fmt(by_type['intermediate']['auc'])} | {_fmt(by_type['ground_glass']['auc'])}"
        )
    return lines


def _write_csv(path, rows):
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidc-cache", default=r"cache\cache_ts_possible_nodule_prior_best.pkl")
    parser.add_argument("--lungx-cache", default=r"cache\cache_lungx_ts_filled_dil1.pkl")
    parser.add_argument("--zhang-checkpoint", default=r"results\zhang_ts_aug\selected_model.pt")
    parser.add_argument("--base-checkpoint", default=r"results\base_aug\baseline_model.pt")
    parser.add_argument("--progressive-checkpoint", default=r"results\zhang_ts_progressive_prior_best_aug\best_model.pt")
    parser.add_argument("--attention-alpha", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--out-dir", default=r"results\stratified_nodule_type_eval")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    models = {
        "zhang_ts_aug": _load_model(args.zhang_checkpoint, device),
        "base_aug": _load_model(args.base_checkpoint, device),
        "progressive_prior": _load_model(args.progressive_checkpoint, device),
    }

    print("Matching LIDC contours and deriving HU types...")
    records = _lidc_records(args.lidc_cache, args.radius)
    print("Loading LUNGx centres and deriving HU types...")
    records.extend(_lungx_records(args.lungx_cache, args.radius))
    print("Scoring models...")
    scored = _score_records(records, models, device, args.attention_alpha)
    summary_rows = _summarise(scored)

    _write_csv(out_dir / "stratified_nodule_type_predictions.csv", scored)
    _write_csv(out_dir / "stratified_nodule_type_summary.csv", summary_rows)
    lines = [
        "=== STRATIFIED NODULE TYPE EVALUATION ===",
        "Type derived from mean reconstructed HU in 15px radius circle.",
        "LIDC circle centre: cached LIDC nodule centre.",
        "LUNGx circle centre: provided LUNGx centre coordinate.",
        f"HU thresholds: ground_glass < -500; intermediate [-500,-300); solid >= -300.",
        f"Zhang checkpoint: {args.zhang_checkpoint}",
        f"Attention transfer alpha: {args.attention_alpha}",
        "",
    ]
    lines.extend(_table_lines(summary_rows, "lidc"))
    lines.append("")
    lines.extend(_table_lines(summary_rows, "lungx"))
    lines.append("")
    lines.append("Note: AUC is nan where a type subgroup contains only one class.")
    text = "\n".join(lines) + "\n"
    (out_dir / "stratified_nodule_type_summary.txt").write_text(text, encoding="utf-8")
    print(text)
    with open("PROGRESS.md", "a", encoding="utf-8") as file:
        file.write(
            "\n\n## Stratified Nodule Type Evaluation\n"
            f"- Output: `{out_dir}`\n"
            "- Evaluated Zhang, attention-transfer, and base by approximate HU nodule type.\n"
        )
    for model in models.values():
        model.remove_hooks()


if __name__ == "__main__":
    main()
