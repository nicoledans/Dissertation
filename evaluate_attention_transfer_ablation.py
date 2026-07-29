"""Ablate attention-transfer guide source for Zhang inference.

No training is performed. Each variant creates a guide map at alpha=0.5, uses it
to reweight the CT input, and feeds the result into the Zhang TS model.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from audit_staged_gradcam_contour_overlap import _load_cache, _match_samples
from config import IMG_SIZE, SEED
from dataset import _patch_to_tensor
from evaluate_attention_transfer_prior_to_zhang import _lungx_records
from model import NoduleClassifier


def _load_model(path, device):
    model = NoduleClassifier(input_channels=3).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


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


def _gradcam(model, image, device):
    model.zero_grad(set_to_none=True)
    model.clear_hooks()
    image = image.to(device)
    logit = model(image).squeeze(1)
    score = model.class_scores(logit)
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


def _random_guide_like(image):
    guide = torch.randn(
        (image.shape[0], image.shape[-2], image.shape[-1]),
        dtype=image.dtype,
        device=image.device,
    ).abs()
    return guide / (guide.flatten(start_dim=1).max(dim=1).values[:, None, None] + 1e-8)


def _guided_image(image, guide, alpha):
    guided = image * (1.0 + float(alpha) * guide.unsqueeze(1))
    peak = guided.flatten(start_dim=1).max(dim=1).values.view(-1, 1, 1, 1)
    return (guided / (peak + 1e-8)).clamp(0.0, 1.0)


@torch.no_grad()
def _zhang_prob(zhang_model, image):
    return float(torch.sigmoid(zhang_model(image).squeeze(1))[0].detach().cpu())


def _lidc_matched_malignant_records(cache_path):
    samples = _load_cache(cache_path)
    matches = _match_samples(samples, "test", malignant_only=True, max_matches=0)
    records = []
    for index, match in enumerate(matches):
        sample = match["cache_sample"]
        records.append(
            {
                "image": _patch_to_tensor(sample["image"]).unsqueeze(0),
                "meta": {
                    "dataset": "lidc_matched_malignant",
                    "row_index": index,
                    "patient_id": match["patient_id"],
                    "scan_id": match["scan_id"],
                    "group_index": match["group_index"],
                    "slice_idx": match["slice_idx"],
                    "label": int(match["label"]),
                },
            }
        )
    return records


def _guide_for_variant(variant, image, device, guide_models):
    if variant == "progressive_prior_guide":
        return _gradcam(guide_models["progressive"], image, device)
    if variant == "base_aug_guide":
        return _gradcam(guide_models["base"], image, device)
    if variant == "random_noise_guide":
        return _random_guide_like(image.to(device))
    raise ValueError(f"Unknown variant: {variant}")


def _evaluate_records(records, variant, zhang_model, guide_models, device, alpha):
    rows = []
    for index, record in enumerate(records, start=1):
        if index % 50 == 0 or index == len(records):
            print(f"  {variant}: scored {index}/{len(records)}")
        image = record["image"].to(device)
        guide = _guide_for_variant(variant, image, device, guide_models)
        prob = _zhang_prob(zhang_model, _guided_image(image, guide, alpha))
        row = {
            **record["meta"],
            "variant": variant,
            "alpha": alpha,
            "zhang_guided_probability_malignant": prob,
            "prediction": int(prob >= 0.5),
        }
        if row.get("label") is not None:
            row["correct"] = int(row["prediction"] == int(row["label"]))
        rows.append(row)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--lidc-cache", default=r"cache\cache_ts_filled_dil1.pkl")
    parser.add_argument("--zhang-checkpoint", default=r"results\zhang_ts_aug_full\selected_model.pt")
    parser.add_argument("--progressive-checkpoint", default=r"results\zhang_ts_progressive_prior_best_aug\best_model.pt")
    parser.add_argument("--base-checkpoint", default=r"results\base_aug\baseline_model.pt")
    parser.add_argument("--lungx-manifest-root", default=r"C:\repo\manifest-cgqtDj7Y2699835271585651107")
    parser.add_argument("--lungx-xlsx", default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    parser.add_argument("--out-dir", default=r"results\attention_transfer_ablation")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    zhang_model = _load_model(args.zhang_checkpoint, device)
    guide_models = {
        "progressive": _load_model(args.progressive_checkpoint, device),
        "base": _load_model(args.base_checkpoint, device),
    }

    lidc_records = _lidc_matched_malignant_records(args.lidc_cache)
    lungx_records = _lungx_records(args)
    variants = ["progressive_prior_guide", "random_noise_guide", "base_aug_guide"]

    summary_rows = []
    report_lines = [
        "=== ATTENTION TRANSFER ABLATION ===",
        "No retraining. Zhang input is reweighted by guide map at alpha=0.5.",
        "Variant 1: progressive_prior_best Grad-CAM guide.",
        "Variant 2: random abs Gaussian guide.",
        "Variant 3: base_aug Grad-CAM guide.",
        "",
        "variant | LUNGx AUC | LUNGx sensitivity | LUNGx specificity | LIDC malignant correct/90",
    ]
    for variant in variants:
        print(f"Evaluating {variant} on LIDC matched malignant")
        lidc_rows = _evaluate_records(lidc_records, variant, zhang_model, guide_models, device, args.alpha)
        print(f"Evaluating {variant} on LUNGx")
        lungx_rows = _evaluate_records(lungx_records, variant, zhang_model, guide_models, device, args.alpha)
        _write_csv(out_dir / f"{variant}_lidc_matched_malignant_predictions.csv", lidc_rows)
        _write_csv(out_dir / f"{variant}_lungx_predictions.csv", lungx_rows)

        lungx_metrics = _metrics(
            [int(row["label"]) for row in lungx_rows],
            [float(row["zhang_guided_probability_malignant"]) for row in lungx_rows],
        )
        lidc_correct = sum(int(row["correct"]) for row in lidc_rows)
        summary = {
            "variant": variant,
            "alpha": args.alpha,
            "lungx_auc": lungx_metrics["auc"],
            "lungx_accuracy": lungx_metrics["accuracy"],
            "lungx_f1": lungx_metrics["f1"],
            "lungx_sensitivity": lungx_metrics["sensitivity"],
            "lungx_specificity": lungx_metrics["specificity"],
            "lidc_malignant_correct": lidc_correct,
            "lidc_malignant_total": len(lidc_rows),
        }
        summary_rows.append(summary)
        report_lines.append(
            f"{variant} | {summary['lungx_auc']:.4f} | {summary['lungx_sensitivity']:.4f} | "
            f"{summary['lungx_specificity']:.4f} | {lidc_correct}/{len(lidc_rows)}"
        )

    best = max(summary_rows, key=lambda row: row["lungx_auc"])
    report_lines.extend(
        [
            "",
            f"Best LUNGx AUC variant: {best['variant']} ({best['lungx_auc']:.4f})",
            "Interpretation: if progressive_prior_guide beats random_noise_guide and base_aug_guide, the improvement is more likely driven by the prior model's spatial guide rather than generic image sharpening.",
        ]
    )
    with open(out_dir / "attention_transfer_ablation_summary.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "attention_transfer_ablation_summary.txt").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(report_lines))

    with open("PROGRESS.md", "a", encoding="utf-8") as file:
        file.write(
            "\n\n## Attention Transfer Ablation\n"
            f"- Output: `{out_dir}`\n"
            f"- Best LUNGx AUC variant: {best['variant']} ({best['lungx_auc']:.4f})\n"
        )

    zhang_model.remove_hooks()
    guide_models["progressive"].remove_hooks()
    guide_models["base"].remove_hooks()


if __name__ == "__main__":
    main()
