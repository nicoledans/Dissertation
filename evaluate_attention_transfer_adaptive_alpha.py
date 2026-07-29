"""Evaluate adaptive-alpha GGO-prior-gated attention transfer on LUNGx."""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from evaluate_attention_transfer_ggo_extended import (
    _fmt,
    _guided_image,
    _load_model,
    _lungx_records,
    _normalise_guide,
    _progressive_cam,
    _write_csv,
)


def _prior_region_mean_hu(image_tensor, prior_tensor, threshold):
    image = image_tensor.squeeze(0)[0].detach().cpu().numpy().astype(np.float32)
    prior = prior_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
    hu = image * 1400.0 - 1000.0
    region = prior > float(threshold)
    if np.any(region):
        return float(np.mean(hu[region]))
    total = float(prior.sum())
    if total > 0:
        return float(np.sum(hu * prior) / (total + 1e-8))
    return float(np.mean(hu))


def _alpha_from_hu(mean_hu, ggo_alpha, intermediate_alpha, solid_alpha):
    if mean_hu < -500.0:
        return float(ggo_alpha), "ground_glass"
    if mean_hu < -300.0:
        return float(intermediate_alpha), "intermediate"
    return float(solid_alpha), "solid"


def _predict(model, image, device):
    model.clear_hooks()
    with torch.no_grad():
        logit = model(image.to(device)).squeeze(1)
        prob = float(torch.sigmoid(logit)[0].detach().cpu())
    model.clear_hooks()
    return prob


def _score_records(records, zhang_model, progressive_model, device, args):
    rows = []
    for index, record in enumerate(records, start=1):
        if index % 25 == 0 or index == len(records):
            print(f"  scored lungx {index}/{len(records)}")
        image = record["image"].to(device)
        prior = record["prior"].to(device)
        prior_mean_hu = _prior_region_mean_hu(image, prior, args.prior_region_threshold)
        alpha, prior_type = _alpha_from_hu(
            prior_mean_hu,
            args.ggo_alpha,
            args.intermediate_alpha,
            args.solid_alpha,
        )
        zhang_prob = _predict(zhang_model, image, device)
        cam, progressive_prob = _progressive_cam(progressive_model, image, device)
        gated_guide = _normalise_guide(cam * prior)
        guided = _guided_image(image, gated_guide, alpha)
        adaptive_prob = _predict(zhang_model, guided, device)
        row = {key: value for key, value in record.items() if key not in {"image", "prior"}}
        row.update(
            {
                "prior_region_mean_hu": prior_mean_hu,
                "prior_region_type": prior_type,
                "adaptive_alpha": alpha,
                "zhang_probability": zhang_prob,
                "progressive_probability": float(progressive_prob[0]),
                "adaptive_attention_transfer_probability": adaptive_prob,
                "prediction": int(adaptive_prob >= 0.5),
                "correct": int((adaptive_prob >= 0.5) == int(record["label"])),
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
    summary_rows = []
    for type_key in ["all", "solid", "intermediate", "ground_glass"]:
        subset = rows if type_key == "all" else [row for row in rows if row["nodule_type"] == type_key]
        metrics = _metrics(
            [int(row["label"]) for row in subset],
            [float(row["adaptive_attention_transfer_probability"]) for row in subset],
        )
        if metrics is None:
            continue
        summary_rows.append(
            {
                "dataset": "lungx",
                "model": "adaptive_alpha_ggo_attention_transfer",
                "nodule_type": type_key,
                "mean_nodule_hu": float(np.mean([float(row["mean_hu"]) for row in subset])) if subset else float("nan"),
                "mean_prior_region_hu": float(np.mean([float(row["prior_region_mean_hu"]) for row in subset])) if subset else float("nan"),
                "mean_alpha": float(np.mean([float(row["adaptive_alpha"]) for row in subset])) if subset else float("nan"),
                **metrics,
            }
        )
    return summary_rows


def _table_lines(summary_rows):
    by_type = {row["nodule_type"]: row for row in summary_rows}
    counts = {key: by_type.get(key, {}).get("n", 0) for key in ["all", "solid", "intermediate", "ground_glass"]}
    lines = [
        "## LUNGx",
        "Type counts: "
        + ", ".join(f"{key}={counts.get(key, 0)}" for key in ["all", "solid", "intermediate", "ground_glass"]),
        "Model | All AUC | Solid AUC | Intermediate AUC | GGO AUC | Acc | Sens | Spec",
        "--- | ---: | ---: | ---: | ---: | ---: | ---: | ---:",
    ]
    all_row = by_type.get("all", {})
    lines.append(
        f"adaptive_alpha_ggo_attention_transfer | {_fmt(by_type.get('all', {}).get('auc', float('nan')))} | "
        f"{_fmt(by_type.get('solid', {}).get('auc', float('nan')))} | "
        f"{_fmt(by_type.get('intermediate', {}).get('auc', float('nan')))} | "
        f"{_fmt(by_type.get('ground_glass', {}).get('auc', float('nan')))} | "
        f"{_fmt(all_row.get('accuracy', float('nan')))} | "
        f"{_fmt(all_row.get('sensitivity', float('nan')))} | "
        f"{_fmt(all_row.get('specificity', float('nan')))}"
    )
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lungx-cache", default=r"cache\cache_lungx_ts_filled_dil1.pkl")
    parser.add_argument("--zhang-checkpoint", default=r"results\zhang_ts_aug_full\selected_model.pt")
    parser.add_argument(
        "--progressive-checkpoint",
        default=r"results\zhang_ts_progressive_prior_best_aug\best_model.pt",
    )
    parser.add_argument("--ggo-alpha", type=float, default=0.7)
    parser.add_argument("--intermediate-alpha", type=float, default=0.5)
    parser.add_argument("--solid-alpha", type=float, default=0.3)
    parser.add_argument(
        "--prior-region-threshold",
        type=float,
        default=0.1,
        help="Mean HU is estimated from prior pixels above this soft-prior value; falls back to weighted mean if empty.",
    )
    parser.add_argument("--radius", type=float, default=15.0)
    parser.add_argument("--out-dir", default=r"results\attention_transfer_adaptive_alpha")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    zhang_model = _load_model(args.zhang_checkpoint, device)
    progressive_model = _load_model(args.progressive_checkpoint, device)

    print("Loading LUNGx records and generating GGO priors...")
    records = _lungx_records(args.lungx_cache, args.radius)
    print("Scoring adaptive-alpha GGO attention transfer...")
    scored = _score_records(records, zhang_model, progressive_model, device, args)
    summary_rows = _summarise(scored)

    _write_csv(out_dir / "adaptive_alpha_predictions.csv", scored)
    _write_csv(out_dir / "adaptive_alpha_summary.csv", summary_rows)

    prior_type_counts = Counter(row["prior_region_type"] for row in scored)
    alpha_counts = Counter(float(row["adaptive_alpha"]) for row in scored)
    lines = [
        "=== ADAPTIVE-ALPHA ATTENTION TRANSFER WITH GGO-EXTENDED PRIOR ===",
        "Inference only; no retraining.",
        "Guide = normalise(predicted-class progressive_prior Grad-CAM * GGO-extended blurred prior).",
        "Alpha is selected from estimated HU in the blurred-prior region:",
        f"  GGO mean HU < -500: alpha={args.ggo_alpha}",
        f"  intermediate -500 to -300: alpha={args.intermediate_alpha}",
        f"  solid > -300: alpha={args.solid_alpha}",
        f"Prior-region HU threshold: prior > {args.prior_region_threshold}; weighted fallback if empty.",
        f"Zhang checkpoint: {args.zhang_checkpoint}",
        f"Progressive checkpoint: {args.progressive_checkpoint}",
        f"LUNGx cache: {args.lungx_cache}",
        "",
        "Reference values:",
        "GGO-gated alpha=0.5: LUNGx all AUC 0.6824, GGO 0.7100.",
        "Original attention transfer alpha=0.5: LUNGx all AUC 0.6734, GGO 0.7035.",
        "Zhang alone: LUNGx all AUC 0.6682, GGO 0.6797.",
        "",
        f"Prior-region type counts used for alpha: {dict(prior_type_counts)}",
        f"Alpha counts: {dict(alpha_counts)}",
        "",
    ]
    lines.extend(_table_lines(summary_rows))
    summary_text = "\n".join(lines) + "\n"
    (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
    print(summary_text)

    with open("PROGRESS.md", "a", encoding="utf-8") as file:
        file.write(
            "\n\n## Adaptive-Alpha Attention Transfer with GGO-Extended Prior\n"
            f"- Output: `{out_dir}`\n"
            "- Evaluated LUNGx only; alpha chosen from prior-region HU estimate.\n"
        )

    zhang_model.remove_hooks()
    progressive_model.remove_hooks()


if __name__ == "__main__":
    main()
