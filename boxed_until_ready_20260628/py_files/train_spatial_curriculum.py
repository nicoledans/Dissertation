"""Train a Grad-CAM lung-compliance curriculum.

This experiment follows a two-phase schedule:
1. Spatial dominance: mostly penalise Grad-CAM mass outside the lung mask.
2. Classification dominance: anneal toward mostly weighted BCE.

The trigger for phase 2 is validation Grad-CAM compliance, not a fixed epoch.
"""

import argparse
import csv
import os
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from config import BATCH_SIZE, IMG_SIZE, LR, RESULTS_DIR, SEED, TRAIN_CACHE_PATH
from dataset import LIDCDataset, load_nodules_hu, patient_split
from model import NoduleClassifier

BLANK_CAM_EPS = 1e-8


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


def _predict(model, samples, criterion, device, batch_size):
    loader = DataLoader(LIDCDataset(samples), batch_size=batch_size, shuffle=False)
    model.eval()
    labels, probabilities = [], []
    loss_sum = 0.0
    seen = 0
    with torch.no_grad():
        for images, _masks, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(images).squeeze(1)
            loss_sum += criterion(logits, batch_labels).item() * batch_labels.numel()
            seen += batch_labels.numel()
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())
            model.clear_hooks()
    return labels, probabilities, loss_sum / max(seen, 1)


def _gradcam_compliance(model, samples, device, use_labels=True):
    """Return mean Grad-CAM mass inside lung and related stats."""
    loader = DataLoader(LIDCDataset(samples), batch_size=1, shuffle=False)
    inside_values = []
    outside_values = []
    blank_count = 0
    model.eval()
    with torch.enable_grad():
        for images, masks, labels in loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            model.zero_grad(set_to_none=True)
            model.clear_hooks()
            logits = model(images).squeeze(1)
            scores = model.class_scores(logits, labels if use_labels else None)
            scores.sum().backward()
            raw_cam = model.get_gradcam(normalise=False)
            blank_count += int(raw_cam.detach().amax().item() <= BLANK_CAM_EPS)
            cam = model.normalise_gradcam(raw_cam)
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            cam_mass = cam / (cam.flatten(start_dim=1).sum(dim=1).view(-1, 1, 1) + 1e-8)
            lung = masks.squeeze(1)
            inside = (cam_mass * lung).sum(dim=(1, 2))
            outside = (cam_mass * (1.0 - lung)).sum(dim=(1, 2))
            inside_values.extend(inside.detach().cpu().tolist())
            outside_values.extend(outside.detach().cpu().tolist())
            model.clear_hooks()
    inside_arr = np.asarray(inside_values, dtype=np.float32)
    outside_arr = np.asarray(outside_values, dtype=np.float32)
    return {
        "n": int(len(inside_values)),
        "inside_mean": float(inside_arr.mean()) if len(inside_arr) else float("nan"),
        "inside_std": float(inside_arr.std()) if len(inside_arr) else float("nan"),
        "outside_mean": float(outside_arr.mean()) if len(outside_arr) else float("nan"),
        "outside_std": float(outside_arr.std()) if len(outside_arr) else float("nan"),
        "blank_rate": blank_count / max(len(inside_values), 1),
        "blank_count": blank_count,
    }


def _spatial_penalty(model, logits, labels, masks):
    """Differentiable outside-lung Grad-CAM mass for the ground-truth class."""
    raw_cam = model.differentiable_gradcam(
        model.class_scores(logits, labels),
        normalise=False,
    )
    blank = raw_cam.detach().flatten(start_dim=1).amax(dim=1) <= BLANK_CAM_EPS
    cam = model.normalise_gradcam(raw_cam)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=masks.shape[-2:],
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)
    cam_mass = cam / (cam.flatten(start_dim=1).sum(dim=1).view(-1, 1, 1) + 1e-8)
    lung = masks.squeeze(1)
    outside_mass = (cam_mass * (1.0 - lung)).sum(dim=(1, 2))
    inside_mass = (cam_mass * lung).sum(dim=(1, 2))
    return outside_mass, inside_mass, blank


def _phase_weights(epoch, phase2_start, args):
    if phase2_start is None:
        return args.phase1_spatial_weight, args.phase1_bce_weight, "phase1_spatial_dominance", False
    anneal_index = epoch - phase2_start
    if anneal_index < args.anneal_epochs:
        frac = anneal_index / max(args.anneal_epochs - 1, 1)
        spatial = args.phase1_spatial_weight + frac * (
            args.phase2_spatial_weight - args.phase1_spatial_weight
        )
        bce = args.phase1_bce_weight + frac * (
            args.phase2_bce_weight - args.phase1_bce_weight
        )
        return spatial, bce, "phase2_annealing", anneal_index == args.anneal_epochs - 1
    return args.phase2_spatial_weight, args.phase2_bce_weight, "phase2_classification_dominance", False


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


def _write_info(path, args, train, val, test, device, pos_weight):
    def stats(samples):
        labels = Counter(sample["label"] for sample in samples)
        patients = {sample["patient_id"] for sample in samples}
        return len(samples), len(patients), labels.get(0, 0), labels.get(1, 0), patients

    train_s, val_s, test_s = stats(train), stats(val), stats(test)
    overlap = (train_s[4] & val_s[4]) | (train_s[4] & test_s[4]) | (val_s[4] & test_s[4])
    lines = [
        "=== GRAD-CAM LUNG COMPLIANCE CURRICULUM ===",
        f"Cache path: {args.cache_path}",
        f"Device: {device}",
        "Loss: lambda_bce * weighted_BCE + lambda_spatial * outside_lung_GradCAM_mass",
        "Grad-CAM training target: ground-truth class.",
        "Spatial penalty: fraction of normalised Grad-CAM mass outside HU lung mask.",
        f"Phase 1 weights: lambda_spatial={args.phase1_spatial_weight}, lambda_bce={args.phase1_bce_weight}",
        f"Phase 1 trigger: validation inside-lung compliance >= {args.compliance_threshold * 100:.1f}% for {args.compliance_patience} consecutive epochs",
        f"Phase 1 max epochs: {args.phase1_max_epochs if args.phase1_max_epochs is not None else 'none'}",
        f"Phase 2 anneal epochs: {args.anneal_epochs}",
        f"Phase 2 final weights: lambda_spatial={args.phase2_spatial_weight}, lambda_bce={args.phase2_bce_weight}",
        f"Early stopping patience after phase 2 starts: {args.early_stop_patience}",
        f"Early stopping only after anneal completes: {args.early_stop_after_anneal_only}",
        f"Safety max epochs: {args.max_epochs}",
        f"Batch size: {args.batch_size}",
        f"Learning rate: {args.lr}",
        f"Optimizer: AdamW, weight_decay={args.weight_decay}",
        f"Loss pos_weight: {pos_weight.item():.6f}",
        f"Train: {train_s[0]} samples, {train_s[1]} patients, benign={train_s[2]}, malignant={train_s[3]}",
        f"Val: {val_s[0]} samples, {val_s[1]} patients, benign={val_s[2]}, malignant={val_s[3]}",
        f"Test: {test_s[0]} samples, {test_s[1]} patients, benign={test_s[2]}, malignant={test_s[3]}",
        f"Patient overlap across splits: {sorted(overlap)}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--phase1-spatial-weight", type=float, default=0.85)
    parser.add_argument("--phase1-bce-weight", type=float, default=0.15)
    parser.add_argument("--phase2-spatial-weight", type=float, default=0.10)
    parser.add_argument("--phase2-bce-weight", type=float, default=0.90)
    parser.add_argument("--compliance-threshold", type=float, default=0.90)
    parser.add_argument("--compliance-patience", type=int, default=3)
    parser.add_argument(
        "--phase1-max-epochs",
        type=int,
        default=None,
        help="Force Phase 2 to start after this many Phase 1 epochs even if compliance has not triggered.",
    )
    parser.add_argument("--anneal-epochs", type=int, default=20)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument(
        "--early-stop-after-anneal-only",
        action="store_true",
        help="Do not count early-stopping patience until the annealing phase has completed.",
    )
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_epochs < 1:
        parser.error("--batch-size and --max-epochs must be positive")
    if args.anneal_epochs < 1 or args.compliance_patience < 1 or args.early_stop_patience < 1:
        parser.error("--anneal-epochs, --compliance-patience, and --early-stop-patience must be positive")
    if args.phase1_max_epochs is not None and args.phase1_max_epochs < 1:
        parser.error("--phase1-max-epochs must be positive when supplied")
    for name in (
        "lr",
        "weight_decay",
        "phase1_spatial_weight",
        "phase1_bce_weight",
        "phase2_spatial_weight",
        "phase2_bce_weight",
        "compliance_threshold",
    ):
        value = getattr(args, name)
        if not np.isfinite(value) or value < 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.compliance_threshold > 1:
        parser.error("--compliance-threshold must be between 0 and 1")
    if args.limit_samples is not None and args.limit_samples < 30:
        parser.error("--limit-samples must be at least 30")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_spatial_curriculum")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=False)

    samples = load_nodules_hu(args.cache_path)
    if args.limit_samples is not None:
        samples = samples[:args.limit_samples]
    train, val, test = patient_split(samples)

    train_ds = LIDCDataset(train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    pos_weight = train_ds.class_weights()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    model = NoduleClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_model_path = os.path.join(run_dir, "spatial_curriculum_model.pt")
    info_path = os.path.join(run_dir, "spatial_curriculum_info.txt")
    _write_info(info_path, args, train, val, test, device, pos_weight)

    best_auc = -float("inf")
    best_epoch = 0
    stale_epochs = 0
    phase2_start = None
    anneal_complete_epoch = None
    compliance_streak = 0
    stop_reason = "max_epochs_reached"

    epoch_path = os.path.join(run_dir, "spatial_curriculum_epochs.csv")
    with open(epoch_path, "w", newline="") as f:
        writer = None
        for epoch in range(1, args.max_epochs + 1):
            lambda_spatial, lambda_bce, phase, anneal_completed_now = _phase_weights(
                epoch, phase2_start, args
            )
            model.train()
            sums = Counter()
            seen = 0
            for images, masks, labels in train_loader:
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images).squeeze(1)
                bce = criterion(logits, labels)
                outside_mass, inside_mass, blank = _spatial_penalty(model, logits, labels, masks)
                spatial_penalty = outside_mass.mean()
                total = lambda_bce * bce + lambda_spatial * spatial_penalty
                total.backward()
                optimizer.step()
                model.clear_hooks()

                n = labels.numel()
                seen += n
                sums["bce"] += bce.item() * n
                sums["spatial_penalty"] += spatial_penalty.item() * n
                sums["total_loss"] += total.item() * n
                sums["inside_compliance"] += inside_mass.detach().sum().item()
                sums["outside_mass"] += outside_mass.detach().sum().item()
                sums["blank"] += blank.sum().item()

            val_labels, val_probs, val_bce = _predict(
                model, val, criterion, device, args.batch_size
            )
            val_metrics = _metrics(val_labels, val_probs)
            val_compliance = _gradcam_compliance(model, val, device, use_labels=True)
            val_inside = val_compliance["inside_mean"]

            if phase2_start is None:
                if val_inside >= args.compliance_threshold:
                    compliance_streak += 1
                else:
                    compliance_streak = 0
                max_phase1_reached = (
                    args.phase1_max_epochs is not None
                    and epoch >= args.phase1_max_epochs
                )
                if compliance_streak >= args.compliance_patience or max_phase1_reached:
                    phase2_start = epoch + 1
                    phase2_triggered_now = True
                else:
                    phase2_triggered_now = False
            else:
                phase2_triggered_now = False

            if anneal_completed_now:
                anneal_complete_epoch = epoch

            improved = np.isfinite(val_metrics["auc"]) and val_metrics["auc"] > best_auc
            if improved:
                best_auc = val_metrics["auc"]
                best_epoch = epoch
                stale_epochs = 0
                torch.save(model.state_dict(), best_model_path)
            elif phase2_start is not None and (
                not args.early_stop_after_anneal_only
                or anneal_complete_epoch is not None
                or anneal_completed_now
            ):
                stale_epochs += 1

            row = {
                "epoch": epoch,
                "phase": phase,
                "phase2_triggered": int(phase2_triggered_now),
                "anneal_complete": int(anneal_completed_now),
                "lambda_spatial": lambda_spatial,
                "lambda_bce": lambda_bce,
                "train_bce": sums["bce"] / max(seen, 1),
                "train_spatial_penalty": sums["spatial_penalty"] / max(seen, 1),
                "train_total_loss": sums["total_loss"] / max(seen, 1),
                "train_inside_compliance": sums["inside_compliance"] / max(seen, 1),
                "train_outside_mass": sums["outside_mass"] / max(seen, 1),
                "train_blank_rate": sums["blank"] / max(seen, 1),
                "val_bce": val_bce,
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_sensitivity": val_metrics["sensitivity"],
                "val_specificity": val_metrics["specificity"],
                "val_inside_compliance": val_inside,
                "val_outside_mass": val_compliance["outside_mean"],
                "val_blank_rate": val_compliance["blank_rate"],
                "compliance_streak": compliance_streak,
                "best_epoch": best_epoch,
                "best_val_auc": best_auc,
                "stale_epochs_after_phase2": stale_epochs,
            }
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
            writer.writerow(row)
            f.flush()

            print(
                f"epoch={epoch:03d} phase={phase} "
                f"sp={lambda_spatial:.3f} bce_w={lambda_bce:.3f} "
                f"train_inside={row['train_inside_compliance'] * 100:.1f}% "
                f"val_inside={val_inside * 100:.1f}% streak={compliance_streak} "
                f"val_auc={val_metrics['auc']:.4f}"
            )
            if phase2_triggered_now:
                print(f"  -> Phase 2 will start at epoch {phase2_start}")
            if anneal_completed_now:
                print(f"  -> Annealing completed at epoch {epoch}")

            if phase2_start is not None and stale_epochs >= args.early_stop_patience:
                stop_reason = f"early_stopping_patience_{args.early_stop_patience}"
                break

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    val_labels, val_probs, val_bce = _predict(model, val, criterion, device, args.batch_size)
    val_metrics = _metrics(val_labels, val_probs)
    val_compliance = _gradcam_compliance(model, val, device, use_labels=True)
    _write_predictions(
        os.path.join(run_dir, "val_predictions_spatial_curriculum.csv"),
        val,
        val_labels,
        val_probs,
    )

    with open(info_path, "a") as f:
        f.write("\n=== TRAINING OUTCOME ===\n")
        f.write(f"Stop reason: {stop_reason}\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation AUC: {best_auc:.6f}\n")
        f.write(f"Phase 2 start epoch: {phase2_start}\n")
        f.write(f"Anneal complete epoch: {anneal_complete_epoch}\n")
        f.write(f"Reloaded validation BCE: {val_bce:.6f}\n")
        f.write(f"Reloaded validation AUC: {val_metrics['auc']:.6f}\n")
        f.write(f"Reloaded validation accuracy: {val_metrics['accuracy']:.6f}\n")
        f.write(f"Reloaded validation F1: {val_metrics['f1']:.6f}\n")
        f.write(f"Reloaded validation sensitivity: {val_metrics['sensitivity']:.6f}\n")
        f.write(f"Reloaded validation specificity: {val_metrics['specificity']:.6f}\n")
        f.write(f"Reloaded validation inside-lung compliance: {val_compliance['inside_mean'] * 100:.2f}%\n")
        f.write(f"Reloaded validation blank Grad-CAM rate: {val_compliance['blank_rate'] * 100:.2f}%\n")

    if args.evaluate_test:
        test_labels, test_probs, test_bce = _predict(
            model, test, criterion, device, args.batch_size
        )
        test_metrics = _metrics(test_labels, test_probs)
        test_compliance = _gradcam_compliance(model, test, device, use_labels=True)
        _write_predictions(
            os.path.join(run_dir, "test_predictions_spatial_curriculum.csv"),
            test,
            test_labels,
            test_probs,
        )
        lines = [
            "=== HELD-OUT TEST: GRAD-CAM LUNG COMPLIANCE CURRICULUM ===",
            f"Test samples: {len(test_labels)}",
            f"Test BCE loss: {test_bce:.4f}",
            f"Test AUC: {test_metrics['auc']:.4f}",
            f"Test Accuracy: {test_metrics['accuracy']:.4f}",
            f"Test F1: {test_metrics['f1']:.4f}",
            f"Test Sensitivity: {test_metrics['sensitivity']:.4f}",
            f"Test Specificity: {test_metrics['specificity']:.4f}",
            "",
            "=== TEST GRAD-CAM LUNG COMPLIANCE ===",
            f"Test samples evaluated: {test_compliance['n']}",
            f"Mean % inside lung mask: {test_compliance['inside_mean'] * 100:.2f}%",
            f"Std inside lung mask: {test_compliance['inside_std'] * 100:.2f}%",
            f"Mean % outside lung mask: {test_compliance['outside_mean'] * 100:.2f}%",
            f"Blank Grad-CAMs: {test_compliance['blank_count']}/{test_compliance['n']} ({test_compliance['blank_rate'] * 100:.2f}%)",
        ]
        result_path = os.path.join(run_dir, "test_results_spatial_curriculum.txt")
        with open(result_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print("\n" + "\n".join(lines))
        print(f"Saved test results -> {result_path}")


if __name__ == "__main__":
    main()
