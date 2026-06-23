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
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from config import BATCH_SIZE, EPOCHS, LR, RESULTS_DIR, SEED, TRAIN_CACHE_PATH
from dataset import LIDCDataset, load_nodules_hu, patient_split
from model import NoduleClassifier


DEFAULT_ALPHAS = [0.00, 0.01, 0.05, 0.10, 0.15, 0.30]
BLANK_CAM_EPS = 1e-8


def _validate_alphas(alphas):
    if not alphas or any(not np.isfinite(alpha) or alpha < 0 for alpha in alphas):
        raise argparse.ArgumentTypeError("alphas must contain non-negative values")
    if 0.0 not in alphas:
        raise argparse.ArgumentTypeError("alphas must include 0.0 as the no-penalty reference")
    if len(set(alphas)) != len(alphas):
        raise argparse.ArgumentTypeError("alphas must not contain duplicate values")
    return sorted(alphas)


def _parse_alphas(value):
    try:
        return _validate_alphas([float(item.strip()) for item in value.split(",")])
    except ValueError as exc:
        raise argparse.ArgumentTypeError("alphas must be comma-separated numbers") from exc


def _alpha_tag(alpha):
    return f"{alpha:.10g}".replace(".", "p")


def _make_run_id():
    return datetime.now().strftime("alpha_grid_%Y%m%d_%H%M%S")


def _classification_metrics(labels, probabilities):
    predictions = [int(probability >= 0.5) for probability in probabilities]
    try:
        auc = roc_auc_score(labels, probabilities)
    except ValueError:
        auc = float("nan")
    accuracy = np.mean(np.asarray(predictions) == np.asarray(labels))
    f1 = f1_score(labels, predictions, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "auc": float(auc),
        "accuracy": float(accuracy),
        "f1": float(f1),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def _evaluate_classification(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for images, _masks, batch_labels in loader:
            logits = model(images.to(device)).squeeze(1)
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.tolist())
            model.clear_hooks()
    return _classification_metrics(labels, probabilities)


def _evaluate_alignment(model, nodules, device):
    model.eval()
    inside_values = []
    mask_areas = []
    normalized_values = []
    blank_count = 0
    loader = DataLoader(LIDCDataset(nodules), batch_size=1, shuffle=False)

    with torch.enable_grad():
        for images, masks, _labels in loader:
            images = images.to(device)
            masks = masks.to(device)
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

            mask_2d = masks.squeeze(1)
            inside = (cam * mask_2d).sum().item() / (cam.sum().item() + 1e-8)
            mask_area = mask_2d.mean().item()
            inside_values.append(inside)
            mask_areas.append(mask_area)
            normalized_values.append(inside / mask_area if mask_area > 0 else float("nan"))
            model.clear_hooks()

    inside_mean = float(np.mean(inside_values)) if inside_values else float("nan")
    mask_area_mean = float(np.mean(mask_areas)) if mask_areas else float("nan")
    finite_normalized = [value for value in normalized_values if np.isfinite(value)]
    normalized_alignment = (
        float(np.mean(finite_normalized)) if finite_normalized else float("nan")
    )
    return {
        "alignment": inside_mean,
        "mask_area": mask_area_mean,
        "normalized_alignment": normalized_alignment,
        "blank_count": blank_count,
        "blank_rate": blank_count / max(len(inside_values), 1),
    }


def _train_alpha(alpha, train_nods, val_nods, device, epochs, batch_size, trial_dir):
    # Reset both seeds for every alpha so initialization and batch order match.
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    train_ds = LIDCDataset(train_nods)
    val_loader = DataLoader(LIDCDataset(val_nods), batch_size=batch_size, shuffle=False)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    model = NoduleClassifier().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_ds.class_weights().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = float("-inf")
    best_epoch = 0
    checkpoint = os.path.join(trial_dir, "best_model.pt")
    epoch_rows = []

    for epoch in range(1, epochs + 1):
        model.train()
        outside_sum = 0.0
        blank_count = 0
        sample_count = 0

        for images, masks, labels in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            logits = model(images).squeeze(1)
            bce_loss = criterion(logits, labels)
            scores = model.class_scores(logits, labels)
            raw_cam = model.differentiable_gradcam(scores, normalise=False)
            blank = raw_cam.detach().flatten(start_dim=1).amax(dim=1) <= BLANK_CAM_EPS
            cam = model.normalise_gradcam(raw_cam)
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            mask_2d = masks.squeeze(1)
            outside = (cam * (1.0 - mask_2d)).sum(dim=(1, 2)) / (
                cam.sum(dim=(1, 2)) + 1e-8
            )
            loss = bce_loss + alpha * outside.mean()
            loss.backward()
            optimizer.step()
            model.clear_hooks()

            outside_sum += outside.detach().sum().item()
            blank_count += int(blank.sum().item())
            sample_count += labels.numel()

        metrics = _evaluate_classification(model, val_loader, device)
        row = {
            "epoch": epoch,
            "val_auc": metrics["auc"],
            "val_accuracy": metrics["accuracy"],
            "train_outside": outside_sum / max(sample_count, 1),
            "train_blank_rate": blank_count / max(sample_count, 1),
        }
        epoch_rows.append(row)
        print(
            f"alpha={alpha:.3f} | epoch={epoch:02d} | "
            f"val_auc={metrics['auc']:.4f} | "
            f"outside={row['train_outside']:.4f} | "
            f"blank={row['train_blank_rate'] * 100:.1f}%"
        )

        if np.isfinite(metrics["auc"]) and metrics["auc"] > best_auc:
            best_auc = metrics["auc"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint)

    with open(os.path.join(trial_dir, "epochs.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=epoch_rows[0].keys())
        writer.writeheader()
        writer.writerows(epoch_rows)

    if not os.path.exists(checkpoint):
        raise RuntimeError(
            f"alpha={alpha} produced no finite validation AUC, so no checkpoint was saved."
        )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    best_metrics = _evaluate_classification(model, val_loader, device)
    alignment = _evaluate_alignment(model, val_nods, device)
    model.remove_hooks()
    del model

    return {
        "alpha": alpha,
        "best_epoch": best_epoch,
        "val_auc": best_metrics["auc"],
        "val_accuracy": best_metrics["accuracy"],
        "val_f1": best_metrics["f1"],
        "val_sensitivity": best_metrics["sensitivity"],
        "val_specificity": best_metrics["specificity"],
        "val_alignment": alignment["alignment"],
        "val_mask_area": alignment["mask_area"],
        "val_normalized_alignment": alignment["normalized_alignment"],
        "val_blank_rate": alignment["blank_rate"],
        "final_train_outside": epoch_rows[-1]["train_outside"],
        "final_train_blank_rate": epoch_rows[-1]["train_blank_rate"],
        "checkpoint": checkpoint,
    }


def _write_results(results, run_dir):
    csv_path = os.path.join(run_dir, "alpha_sensitivity.csv")
    fields = [key for key in results[0] if key != "checkpoint"]
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in fields})

    lines = [
        "=== ALPHA SENSITIVITY ===",
        "No automatic alpha selection is applied.",
        "Use this table/plot to decide the AUC-vs-attention trade-off manually.",
        "",
        "alpha | val_auc | val_acc | alignment | norm_align | blank_rate",
    ]
    for result in results:
        lines.append(
            f"{result['alpha']:>5.3f} | {result['val_auc']:.4f} | "
            f"{result['val_accuracy']:.4f} | "
            f"{result['val_alignment'] * 100:>8.2f}% | "
            f"{result['val_normalized_alignment']:>10.3f} | "
            f"{result['val_blank_rate'] * 100:>9.2f}%"
        )
    lines.extend([
        "",
        "The held-out test set was not evaluated in this sensitivity run.",
        "After choosing an alpha, rerun train.py once with --evaluate-test.",
    ])
    text = "\n".join(lines)
    print("\n" + text)
    with open(os.path.join(run_dir, "alpha_sensitivity.txt"), "w") as file:
        file.write(text + "\n")

    alphas = [result["alpha"] for result in results]
    aucs = [result["val_auc"] for result in results]
    alignments = [result["val_alignment"] * 100 for result in results]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(alphas, aucs, marker="o")
    axes[0].set(xlabel="Alpha", ylabel="Best validation AUC", title="Alpha vs AUC")

    axes[1].plot(alphas, alignments, marker="o", color="seagreen")
    axes[1].set(xlabel="Alpha", ylabel="Validation Grad-CAM inside mask (%)",
                title="Alpha vs alignment")

    scatter = axes[2].scatter(aucs, alignments, c=alphas, cmap="viridis", s=80)
    axes[2].set(xlabel="Best validation AUC",
                ylabel="Validation Grad-CAM inside mask (%)",
                title="Classification-alignment trade-off")
    fig.colorbar(scatter, ax=axes[2], label="Alpha")

    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, "alpha_sensitivity.png"), dpi=150)
    plt.close(fig)


def _write_run_info(run_dir, args, train_nods, val_nods, test_nods, device):
    def patient_ids(samples):
        return sorted({sample["patient_id"] for sample in samples})

    train_patients = patient_ids(train_nods)
    val_patients = patient_ids(val_nods)
    test_patients = patient_ids(test_nods)
    overlap = (
        set(train_patients) & set(val_patients)
        or set(train_patients) & set(test_patients)
        or set(val_patients) & set(test_patients)
    )
    if overlap:
        raise RuntimeError(f"Patient leakage detected across splits: {sorted(overlap)}")

    info = [
        "=== ALPHA SEARCH RUN INFO ===",
        f"Cache: {os.path.abspath(args.cache_path)}",
        f"Seed: {SEED}",
        f"Device: {device}",
        f"Alphas: {args.alphas}",
        f"Epochs per alpha: {args.epochs}",
        f"Batch size: {args.batch_size}",
        "Data used for sensitivity table: validation only",
        "Automatic alpha selection: no",
        "Held-out test evaluated: no",
        "",
        f"Train: {len(train_nods)} samples, {len(train_patients)} patients",
        f"Validation: {len(val_nods)} samples, {len(val_patients)} patients",
        f"Test held out: {len(test_nods)} samples, {len(test_patients)} patients",
        "Patient overlap across splits: none",
    ]
    with open(os.path.join(run_dir, "alpha_search_info.txt"), "w") as file:
        file.write("\n".join(info) + "\n")

    with open(os.path.join(run_dir, "alpha_split.txt"), "w") as file:
        for name, patients in (
            ("TRAIN", train_patients),
            ("VALIDATION", val_patients),
            ("TEST_HELD_OUT", test_patients),
        ):
            file.write(f"[{name}]\n")
            file.write("\n".join(patients) + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validation-only fixed-alpha sensitivity study for HU Grad-CAM supervision."
    )
    parser.add_argument("--run-id", default=None,
                        help="Output folder name under results/ (default: timestamped alpha_grid folder)")
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH,
                        help=f"HU cache path (default: {TRAIN_CACHE_PATH})")
    parser.add_argument("--alphas", type=_parse_alphas,
                        default=DEFAULT_ALPHAS,
                        help="Comma-separated grid including 0.0")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Epochs per alpha (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--keep-checkpoints", action="store_true",
                        help="Keep every trial checkpoint for later manual inspection")
    args = parser.parse_args()
    try:
        args.alphas = _validate_alphas(args.alphas)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    run_id = args.run_id or _make_run_id()
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=False)

    nodules = load_nodules_hu(args.cache_path)
    train_nods, val_nods, test_nods = patient_split(nodules)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _write_run_info(run_dir, args, train_nods, val_nods, test_nods, device)

    print(f"[ALPHA GRID] Output: {run_dir}")
    print(f"Device: {device}")
    print(f"Alphas: {args.alphas}")
    print("Held-out test set will not be evaluated.")
    print("No automatic alpha selection will be applied.")

    results = []
    for alpha in args.alphas:
        trial_dir = os.path.join(run_dir, f"alpha_{_alpha_tag(alpha)}")
        os.makedirs(trial_dir)
        print(f"\n=== alpha={alpha:.3f} ===")
        results.append(
            _train_alpha(
                alpha, train_nods, val_nods, device,
                args.epochs, args.batch_size, trial_dir,
            )
        )

    _write_results(results, run_dir)

    if not args.keep_checkpoints:
        for result in results:
            checkpoint = result["checkpoint"]
            if os.path.exists(checkpoint):
                os.remove(checkpoint)

    print("Alpha sensitivity complete.")
    print("Choose an alpha from alpha_sensitivity.txt/png, then rerun train.py for final testing.")


if __name__ == "__main__":
    main()
