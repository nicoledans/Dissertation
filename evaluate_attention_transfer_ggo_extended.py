"""Evaluate GGO-extended prior-gated attention transfer.

Inference-only experiment:
  1. Generate predicted-class Grad-CAM from the progressive-prior model.
  2. Gate that CAM by the GGO-extended possible-nodule prior.
  3. Reweight CT as CT * (1 + alpha * gated_cam), then renormalise.
  4. Score with the Zhang TS model.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from audit_staged_gradcam_contour_overlap import _load_cache
from config import IMG_SIZE
from dataset import _patch_to_tensor, patient_split
from model import NoduleClassifier
from preview_possible_nodule_mask import _make_prior, _possible_nodule_mask


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


def _mean_hu_in_circle(image, center_row, center_col, radius):
    hu = image.astype(np.float32) * 1400.0 - 1000.0
    roi = _circle_mask(hu.shape, center_row, center_col, radius)
    if not np.any(roi):
        return float("nan")
    return float(np.mean(hu[roi]))


def _type_from_hu(mean_hu):
    if mean_hu < -500.0:
        return "ground_glass"
    if mean_hu < -300.0:
        return "intermediate"
    return "solid"


def _ggo_prior_args():
    return SimpleNamespace(
        prior_mode="standard",
        hu_threshold=-600.0,
        solid_max_hu=100.0,
        ggo_extended=True,
        ggo_min_hu=-800.0,
        ggo_max_hu=-600.0,
        ggo_min_area=5,
        ggo_max_area=100,
        ggo_max_elongation=1.5,
        no_boundary_band=True,
        max_ts_hole_area=100,
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


def _lidc_records(cache_path, radius):
    samples = _load_cache(cache_path)
    _train, _val, test = patient_split(samples)
    records = []
    for idx, sample in enumerate(test):
        image_np = _center_image(sample)
        if "nodule_center_rc" in sample:
            center_row = float(sample["nodule_center_rc"][0])
            center_col = float(sample["nodule_center_rc"][1])
        else:
            center_row = image_np.shape[0] / 2.0
            center_col = image_np.shape[1] / 2.0
        mean_hu = _mean_hu_in_circle(image_np, center_row, center_col, radius)
        prior = np.asarray(sample["possible_nodule_prior"], dtype=np.float32)
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
                "prior_mean": float(prior.mean()),
                "prior_max": float(prior.max()),
                "image": _patch_to_tensor(sample["image"]).unsqueeze(0),
                "prior": torch.from_numpy(prior).float().unsqueeze(0).unsqueeze(0),
            }
        )
    return records


def _lungx_records(cache_path, radius):
    samples = _load_cache(cache_path)
    prior_args = _ggo_prior_args()
    records = []
    for idx, sample in enumerate(samples):
        image_np = _center_image(sample)
        center_row = float(sample["center_y_224"])
        center_col = float(sample["center_x_224"])
        mean_hu = _mean_hu_in_circle(image_np, center_row, center_col, radius)
        candidate, parts = _possible_nodule_mask(sample, prior_args)
        prior = _make_prior(candidate, parts, prior_args)
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
                "prior_mean": float(prior.mean()),
                "prior_max": float(prior.max()),
                "image": _patch_to_tensor(sample["image"]).unsqueeze(0),
                "prior": torch.from_numpy(prior.astype(np.float32)).unsqueeze(0).unsqueeze(0),
            }
        )
    return records


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
    )
    probability = torch.sigmoid(logit).detach().cpu().numpy()
    progressive_model.clear_hooks()
    return cam.detach(), probability


def _predict(model, image, device):
    model.clear_hooks()
    with torch.no_grad():
        logit = model(image.to(device)).squeeze(1)
        prob = float(torch.sigmoid(logit)[0].detach().cpu())
    model.clear_hooks()
    return prob


def _normalise_guide(guide):
    peak = guide.flatten(start_dim=1).max(dim=1).values.view(-1, 1, 1, 1)
    return torch.where(peak > 0, guide / (peak + 1e-8), guide)


def _guided_image(image, guide, alpha):
    guided = image * (1.0 + float(alpha) * guide)
    peak = guided.flatten(start_dim=1).max(dim=1).values.view(-1, 1, 1, 1)
    return (guided / (peak + 1e-8)).clamp(0.0, 1.0)


def _score_records(records, zhang_model, progressive_model, device, alpha):
    rows = []
    for index, record in enumerate(records, start=1):
        if index % 50 == 0 or index == len(records):
            print(f"  scored {record['dataset']} {index}/{len(records)}")
        image = record["image"].to(device)
        prior = record["prior"].to(device)
        zhang_prob = _predict(zhang_model, image, device)
        cam, progressive_prob = _progressive_cam(progressive_model, image, device)
        gated_guide = _normalise_guide(cam * prior)
        guided = _guided_image(image, gated_guide, alpha)
        guided_prob = _predict(zhang_model, guided, device)
        row = {key: value for key, value in record.items() if key not in {"image", "prior"}}
        row.update(
            {
                "alpha": float(alpha),
                "zhang_probability": zhang_prob,
                "progressive_probability": float(progressive_prob[0]),
                "ggo_gated_attention_transfer_probability": guided_prob,
                "gated_guide_mean": float(gated_guide.mean().detach().cpu()),
                "gated_guide_max": float(gated_guide.max().detach().cpu()),
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
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def _summarise(rows):
    models = [
        ("zhang_alone", "zhang_probability"),
        ("ggo_gated_attention_transfer", "ggo_gated_attention_transfer_probability"),
    ]
    datasets = ["lidc", "lungx"]
    nodule_types = ["all", "solid", "intermediate", "ground_glass"]
    summary_rows = []
    for dataset in datasets:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for model_name, prob_key in models:
            for nodule_type in nodule_types:
                subset = (
                    dataset_rows
                    if nodule_type == "all"
                    else [row for row in dataset_rows if row["nodule_type"] == nodule_type]
                )
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
                        "mean_prior_max": float(np.mean([float(row["prior_max"]) for row in subset])) if subset else float("nan"),
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
        if row["dataset"] == dataset and row["model"] == "zhang_alone"
    }
    lines.append(
        "Type counts: "
        + ", ".join(
            f"{key}={counts.get(key, 0)}"
            for key in ["all", "solid", "intermediate", "ground_glass"]
        )
    )
    lines.append("Model | All AUC | Solid AUC | Intermediate AUC | GGO AUC | Acc | Sens | Spec")
    lines.append("--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:")
    for model in ["zhang_alone", "ggo_gated_attention_transfer"]:
        by_type = {
            row["nodule_type"]: row
            for row in summary_rows
            if row["dataset"] == dataset and row["model"] == model
        }
        all_row = by_type.get("all", {})
        lines.append(
            f"{model} | {_fmt(by_type.get('all', {}).get('auc', float('nan')))} | "
            f"{_fmt(by_type.get('solid', {}).get('auc', float('nan')))} | "
            f"{_fmt(by_type.get('intermediate', {}).get('auc', float('nan')))} | "
            f"{_fmt(by_type.get('ground_glass', {}).get('auc', float('nan')))} | "
            f"{_fmt(all_row.get('accuracy', float('nan')))} | "
            f"{_fmt(all_row.get('sensitivity', float('nan')))} | "
            f"{_fmt(all_row.get('specificity', float('nan')))}"
        )
    return lines


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lidc-cache", default=r"cache\cache_ts_possible_nodule_prior_ggo.pkl")
    parser.add_argument("--lungx-cache", default=r"cache\cache_lungx_ts_filled_dil1.pkl")
    parser.add_argument("--zhang-checkpoint", default=r"results\zhang_ts_aug_full\selected_model.pt")
    parser.add_argument(
        "--progressive-checkpoint",
        default=r"results\zhang_ts_progressive_prior_best_aug\best_model.pt",
    )
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--out-dir", default=r"results\attention_transfer_ggo_extended")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    zhang_model = _load_model(args.zhang_checkpoint, device)
    progressive_model = _load_model(args.progressive_checkpoint, device)

    print("Loading LIDC held-out records from GGO prior cache...")
    records = _lidc_records(args.lidc_cache, args.radius)
    print("Loading LUNGx records and generating GGO priors...")
    records.extend(_lungx_records(args.lungx_cache, args.radius))
    print("Scoring GGO-gated attention transfer...")
    scored = _score_records(records, zhang_model, progressive_model, device, args.alpha)
    summary_rows = _summarise(scored)

    _write_csv(out_dir / "attention_transfer_ggo_extended_predictions.csv", scored)
    _write_csv(out_dir / "attention_transfer_ggo_extended_summary.csv", summary_rows)

    lines = [
        "=== ATTENTION TRANSFER WITH GGO-EXTENDED PRIOR ===",
        "Inference only; no retraining.",
        "Guide = normalise(predicted-class progressive_prior Grad-CAM * GGO-extended blurred prior).",
        f"Weighted CT: ct * (1 + {args.alpha} * guide), then max-normalised.",
        f"Zhang checkpoint: {args.zhang_checkpoint}",
        f"Progressive checkpoint: {args.progressive_checkpoint}",
        f"LIDC GGO cache: {args.lidc_cache}",
        f"LUNGx TS cache: {args.lungx_cache}; GGO prior generated per sample with same settings.",
        "Type split uses mean reconstructed HU in 15px nodule-centre circle.",
        "",
        "Reference values:",
        "Original attention transfer alpha=0.5: LUNGx all AUC 0.6734, GGO 0.7035, intermediate 0.6667, solid 0.3333.",
        "Zhang alone LUNGx reference: overall AUC 0.6682, GGO AUC 0.6797.",
        "",
    ]
    lines.extend(_table_lines(summary_rows, "lidc"))
    lines.append("")
    lines.extend(_table_lines(summary_rows, "lungx"))
    lines.append("")
    lines.append("Note: Solid LUNGx type has very small N, so its AUC is unstable.")
    summary_text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)

    with open("PROGRESS.md", "a", encoding="utf-8") as file:
        file.write(
            "\n\n## Attention Transfer with GGO-Extended Prior\n"
            f"- Output: `{out_dir}`\n"
            "- Evaluated Zhang alone vs GGO-prior-gated progressive Grad-CAM attention transfer on LIDC and LUNGx.\n"
        )

    zhang_model.remove_hooks()
    progressive_model.remove_hooks()


if __name__ == "__main__":
    main()
