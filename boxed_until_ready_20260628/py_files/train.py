import os
import argparse
from collections import Counter
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

from config import (
    EPOCHS, BATCH_SIZE, LR, ALPHA, SEED,
    IMG_SIZE, RESULTS_DIR, TRAIN_CACHE_PATH,
)
from dataset import LIDCDataset, load_nodules_hu, patient_split
from model import NoduleClassifier

BLANK_CAM_EPS = 1e-8


def _test_evaluate(model, test_nods, device, run_dir, batch_size):
    if not test_nods:
        print("No test samples — skipping test evaluation.")
        return

    model.eval()
    ds = LIDCDataset(test_nods)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, _masks, labels in loader:
            images = images.to(device)
            logits = model(images).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            model.clear_hooks()

    preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = float("nan")

    acc = sum(p == l for p, l in zip(preds_bin, all_labels)) / max(len(all_labels), 1)

    try:
        f1 = f1_score(all_labels, preds_bin, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds_bin, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    except Exception:
        f1 = sensitivity = specificity = float("nan")

    print("\nComputing test Grad-CAM mask alignment...")
    gcam_mean, gcam_std, gcam_n, blank_n = _gradcam_mask_stats(model, test_nods, device)

    def _f(v):
        return f"{v:.4f}" if not np.isnan(v) else "nan"

    lines = [
        "=== TEST SET EVALUATION (HU Grad-CAM supervision) ===",
        f"Test samples:           {len(all_labels)}",
        f"Test AUC:               {_f(auc)}",
        f"Test Accuracy:          {_f(acc)}",
        f"Test F1:                {_f(f1)}",
        f"Test Sensitivity:       {_f(sensitivity)}",
        f"Test Specificity:       {_f(specificity)}",
        "",
        "=== TEST GRAD-CAM MASK ALIGNMENT (HU mask) ===",
        f"Test samples evaluated: {gcam_n}",
        f"Mean % inside mask:     {gcam_mean:.1f}%",
        f"Std:                    {gcam_std:.1f}%",
        f"Blank Grad-CAMs:        {blank_n}/{gcam_n} ({blank_n / max(gcam_n, 1) * 100:.1f}%)",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    out_path = os.path.join(run_dir, "test_results_hu.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"Saved → {out_path}")


def _make_run_id(results_dir):
    latest_file = os.path.join(results_dir, "latest_run.txt")
    if os.path.exists(latest_file):
        return open(latest_file).read().strip()
    today = datetime.now().strftime("%Y-%m-%d")
    return f"{today}_run1"


def _gradcam_mask_stats(model, val_nods, device):
    model.eval()
    pct_list = []
    blank_count = 0
    ds = LIDCDataset(val_nods)
    if len(ds) == 0:
        return float("nan"), float("nan"), 0, 0
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    with torch.enable_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            model.zero_grad(set_to_none=True)
            model.clear_hooks()
            logits = model(images).squeeze(1)
            model.class_scores(logits).sum().backward()
            raw_cam = model.get_gradcam(normalise=False)
            blank_count += int(raw_cam.amax().item() <= BLANK_CAM_EPS)
            cam = model.normalise_gradcam(raw_cam)
            cam_up = F.interpolate(
                cam.unsqueeze(1),
                size=(masks.shape[2], masks.shape[3]),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            mask_2d = masks.squeeze(1)
            inside = (cam_up * mask_2d).sum().item()
            total = cam_up.sum().item() + 1e-8
            pct_list.append(inside / total * 100.0)
            model.clear_hooks()
    arr = np.array(pct_list)
    return float(arr.mean()), float(arr.std()), len(arr), blank_count


def _write_run_info(run_dir, cache_path, alpha, epochs, batch_size,
                    train_nods, val_nods, test_nods):
    def _stats(nodules):
        labels = Counter(n["label"] for n in nodules)
        patients = {n["patient_id"] for n in nodules}
        return len(nodules), len(patients), labels.get(0, 0), labels.get(1, 0), patients

    train = _stats(train_nods)
    val = _stats(val_nods)
    test = _stats(test_nods)
    lines = [
        "=== EXPLANATION-SUPERVISED TRAINING RUN ===",
        f"Cache path: {cache_path}",
        "Mask type: HU lung mask",
        "Training loss: weighted BCE + alpha * outside-lung Grad-CAM fraction",
        "Grad-CAM target during training: ground-truth class",
        "Blank Grad-CAM handling: counted and reported; no extra loss",
        f"Alpha: {alpha}",
        f"Epochs: {epochs}",
        f"Batch size: {batch_size}",
        f"Learning rate: {LR}",
        "",
        "Split type: patient-level 70/15/15",
        f"Train: {train[0]} samples, {train[1]} patients, {train[2]} benign, {train[3]} malignant",
        f"Val: {val[0]} samples, {val[1]} patients, {val[2]} benign, {val[3]} malignant",
        f"Test: {test[0]} samples, {test[1]} patients, {test[2]} benign, {test[3]} malignant",
        f"Train/val patient overlap: {len(train[4] & val[4])}",
        f"Train/test patient overlap: {len(train[4] & test[4])}",
        f"Val/test patient overlap: {len(val[4] & test[4])}",
    ]
    path = os.path.join(run_dir, "info_hu.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved -> {path}")


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                        help="Shared run ID for grouping results (auto-generated if omitted)")
    parser.add_argument("--cache-path", type=str, default=None,
                        help=f"Path to HU cache file (default: {TRAIN_CACHE_PATH})")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"Grad-CAM outside-mask penalty weight (default: {ALPHA})")
    parser.add_argument("--evaluate-test", action="store_true",
                        help="Evaluate the held-out test set after training")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.alpha < 0:
        parser.error("--alpha must be non-negative")

    run_id = args.run_id or _make_run_id(RESULTS_DIR)
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "latest_run.txt"), "w") as f:
        f.write(run_id)

    print(f"[RUN] {run_id}  →  {run_dir}")
    print(f"      To compare after all scripts: python compare_all.py --run-id {run_id}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cache_path = args.cache_path or TRAIN_CACHE_PATH
    nodules = load_nodules_hu(cache_path=cache_path)
    print(f"Found {len(nodules)} samples.")

    train_nods, val_nods, test_nods = patient_split(nodules)
    _write_run_info(
        run_dir, cache_path, args.alpha, args.epochs, args.batch_size,
        train_nods, val_nods, test_nods,
    )

    if not train_nods:
        print("No training samples / cache too small")
        return

    train_ds = LIDCDataset(train_nods)
    val_ds = LIDCDataset(val_nods)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleClassifier().to(device)

    pos_weight = train_ds.class_weights().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log_path = os.path.join(run_dir, "train_log.txt")
    best_model_path = os.path.join(run_dir, "best_model.pt")
    best_auc = float("-inf")

    with open(log_path, "w") as log_f:
        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_blank = 0
            epoch_samples = 0
            epoch_penalty_sum = 0.0
            for batch_idx, (images, masks, labels) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model(images).squeeze(1)
                bce_loss = criterion(logits, labels)

                # Generate class-specific Grad-CAM for the ground-truth class.
                # create_graph=True inside differentiable_gradcam keeps the
                # outside-lung fraction differentiable during training.
                class_scores = model.class_scores(logits, labels)
                raw_cam = model.differentiable_gradcam(class_scores, normalise=False)
                blank = raw_cam.detach().flatten(start_dim=1).amax(dim=1) <= BLANK_CAM_EPS
                cam = model.normalise_gradcam(raw_cam).unsqueeze(1)
                cam_resized = F.interpolate(
                    cam,
                    size=(masks.shape[2], masks.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                mask_2d = masks.squeeze(1)

                penalty = (cam_resized * (1.0 - mask_2d)).sum(dim=(1, 2)) / (
                    cam_resized.sum(dim=(1, 2)) + 1e-8
                )

                gradcam_loss = penalty.mean()
                total_loss = bce_loss + args.alpha * gradcam_loss

                total_loss.backward()
                optimizer.step()
                model.clear_hooks()

                epoch_blank += int(blank.sum().item())
                epoch_samples += labels.numel()
                epoch_penalty_sum += float(penalty.detach().sum().item())

                print(
                    f"Epoch {epoch:02d} | Batch {batch_idx:04d} | "
                    f"BCE {bce_loss.item():.4f} | "
                    f"GradCAMOutside {gradcam_loss.item():.4f} | "
                    f"Blank {int(blank.sum().item())}/{labels.numel()} | "
                    f"Total {total_loss.item():.4f}"
                )

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, masks, labels in val_loader:
                    images = images.to(device)
                    logits = model(images).squeeze(1)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs.tolist())
                    all_labels.extend(labels.numpy().tolist())
                    model.clear_hooks()

            preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
            acc = sum(p == l for p, l in zip(preds_bin, all_labels)) / max(len(all_labels), 1)
            try:
                auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                auc = float("nan")

            blank_rate = epoch_blank / max(epoch_samples, 1) * 100.0
            mean_penalty = epoch_penalty_sum / max(epoch_samples, 1)
            log_line = (
                f"Epoch {epoch:02d} | Val Acc {acc:.4f} | Val AUC {auc:.4f} | "
                f"Train GradCAMOutside {mean_penalty:.4f} | "
                f"Train Blank CAM {epoch_blank}/{epoch_samples} ({blank_rate:.1f}%)\n"
            )
            log_f.write(log_line)
            log_f.flush()
            print(log_line, end="")

            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Saved best model (AUC {best_auc:.4f})")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # ── Quantitative Grad-CAM mask alignment ─────────────────────────────────
    constraint_note = "(Mask alignment is reported separately from classification performance.)"
    print("\nComputing Grad-CAM mask alignment stats...")
    mean_pct, std_pct, n_samples, blank_count = _gradcam_mask_stats(model, val_nods, device)
    stats_lines = [
        "=== GRAD-CAM MASK ALIGNMENT (HU mask constraint) ===",
        f"Val samples evaluated: {n_samples}",
        f"Mean % activation inside mask: {mean_pct:.1f}%",
        f"Std:                           {std_pct:.1f}%",
        f"Blank Grad-CAMs:               {blank_count}/{n_samples} "
        f"({blank_count / max(n_samples, 1) * 100:.1f}%)",
        constraint_note,
    ]
    stats_text = "\n".join(stats_lines)
    print(stats_text)
    with open(os.path.join(run_dir, "gradcam_stats_train.txt"), "w") as f:
        f.write(stats_text + "\n")

    # ── Grad-CAM visualisation (4 examples) ──────────────────────────────────
    if not val_nods:
        print("No validation samples — skipping Grad-CAM visualisation")
        return

    mask_color = "lime"
    mask_viz_label = "HU mask"
    viz_fname = "gradcam_examples.png"

    model.eval()
    collected = 0
    n_rows = min(4, len(val_nods))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for images, masks, labels in DataLoader(LIDCDataset(val_nods), batch_size=1):
        if collected >= 4:
            break
        images = images.to(device)
        masks = masks.to(device)
        model.zero_grad(set_to_none=True)
        model.clear_hooks()
        logits = model(images).squeeze(1)
        model.class_scores(logits).sum().backward()
        cam = model.get_gradcam()
        cam_up = F.interpolate(
            cam.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().detach().numpy()

        img_np = images[0, 0].cpu().detach().numpy()
        mask_np = masks[0, 0].cpu().detach().numpy()
        label_val = labels[0].item()

        inside = float((cam_up * (mask_np > 0.5)).sum() / (cam_up.sum() + 1e-8) * 100)

        ax = axes[collected]
        ax[0].imshow(img_np, cmap="gray")
        ax[0].set_title(f"CT slice (label={int(label_val)})")
        ax[0].axis("off")

        ax[1].imshow(mask_np, cmap="gray")
        ax[1].set_title(mask_viz_label)
        ax[1].axis("off")

        ax[2].imshow(cam_up, cmap="jet")
        ax[2].set_title("Grad-CAM heatmap")
        ax[2].axis("off")

        ax[3].imshow(img_np, cmap="gray")
        ax[3].imshow(cam_up, cmap="jet", alpha=0.5)
        ax[3].contour(mask_np, levels=[0.5], colors=mask_color, linewidths=1)
        ax[3].set_title(f"Overlay ({mask_color}=mask) | {inside:.0f}% inside")
        ax[3].axis("off")

        model.clear_hooks()
        collected += 1

    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, viz_fname), dpi=150)
    plt.close(fig)
    print(f"Saved Grad-CAM examples → {run_dir}/{viz_fname}")

    # ── Held-out test evaluation ──────────────────────────────────────────────
    if args.evaluate_test:
        print("\n--- Test Set Evaluation ---")
        _test_evaluate(model, test_nods, device, run_dir, args.batch_size)
    else:
        print("\nHeld-out test evaluation skipped. Use --evaluate-test for a final run.")


if __name__ == "__main__":
    train()
