import os
import argparse
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import (
    EPOCHS, BATCH_SIZE, LR, ALPHA, SEED,
    IMG_SIZE, RESULTS_DIR,
)
from dataset import LIDCDataset, load_nodules, patient_split
from model import NoduleClassifier


def _gradcam_mask_stats(model, val_nods, device):
    """Compute % of Grad-CAM activation energy inside TS mask over full val set."""
    model.eval()
    pct_list = []
    ds = LIDCDataset(val_nods, use_ts_mask=True)
    if len(ds) == 0:
        return float("nan"), float("nan"), 0
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    with torch.enable_grad():
        for images, masks, _, _ in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images).squeeze(1)
            logits.sum().backward()
            cam = model.get_gradcam()
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
    return float(arr.mean()), float(arr.std()), len(arr)


def train_ts():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                        help="Shared run ID for grouping results (auto-timestamp if omitted)")
    parser.add_argument("--cache-path", type=str, default=None,
                        help="Path to cache.pkl (default: results/cache.pkl)")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(RESULTS_DIR, "latest_run.txt"), "w") as f:
        f.write(run_id)

    print(f"[RUN] {run_id}  →  {run_dir}")
    print(f"      To compare after all scripts: python compare_all.py --run-id {run_id}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    nodules = load_nodules(cache_path=args.cache_path)

    ts_count = sum(1 for n in nodules if n.get("ts_mask") is not None)
    if ts_count == 0:
        print("No TotalSegmentator masks found in cache. Re-run build_cache.py without --no-ts.")
        return

    print(f"Found {ts_count}/{len(nodules)} samples with TS masks.")

    train_nods, val_nods, _ = patient_split(nodules)

    if not train_nods:
        print("No training samples / cache too small")
        return

    train_ds = LIDCDataset(train_nods, use_ts_mask=True)
    val_ds = LIDCDataset(val_nods, use_ts_mask=True)

    if len(train_ds) == 0:
        print("No training samples with TS masks.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleClassifier().to(device)

    pos_weight = train_ds.class_weights().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    log_path = os.path.join(run_dir, "train_ts_log.txt")
    best_auc = 0.0
    best_model_path = os.path.join(run_dir, "best_model_ts.pt")

    with open(log_path, "w") as log_f:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            for batch_idx, (images, masks, labels, _) in enumerate(train_loader):
                images = images.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model(images).squeeze(1)
                bce_loss = criterion(logits, labels)

                # ── Differentiable attention penalty via layer4 activations ──────
                acts = model._activations  # (B, C, H_feat, W_feat)
                attn = torch.relu(acts).mean(dim=1, keepdim=True)
                b_size = attn.shape[0]
                flat = attn.view(b_size, -1)
                attn_min = flat.min(dim=1)[0].view(b_size, 1, 1, 1)
                attn_max = flat.max(dim=1)[0].view(b_size, 1, 1, 1)
                attn = (attn - attn_min) / (attn_max - attn_min + 1e-8)

                attn_resized = F.interpolate(
                    attn,
                    size=(masks.shape[2], masks.shape[3]),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

                mask_2d = masks.squeeze(1)

                penalty = (attn_resized * (1.0 - mask_2d)).sum(dim=(1, 2)) / (
                    attn_resized.sum(dim=(1, 2)) + 1e-8
                )

                scale = 2.0 * (torch.sigmoid(logits.detach()) - 0.5).abs()

                total_loss = bce_loss + ALPHA * (scale * penalty).mean()

                total_loss.backward()
                optimizer.step()
                model.clear_hooks()

                print(
                    f"Epoch {epoch:02d} | Batch {batch_idx:04d} | "
                    f"BCE {bce_loss.item():.4f} | "
                    f"Attn {penalty.mean().item():.4f} | "
                    f"Total {total_loss.item():.4f}"
                )

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, masks, labels, _ in val_loader:
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

            log_line = f"Epoch {epoch:02d} | Val Acc {acc:.4f} | Val AUC {auc:.4f}\n"
            log_f.write(log_line)
            log_f.flush()
            print(log_line, end="")

            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), best_model_path)
                print(f"  -> Saved best TS model (AUC {best_auc:.4f})")

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))

    # ── Quantitative Grad-CAM mask alignment ─────────────────────────────────
    print("\nComputing Grad-CAM mask alignment stats...")
    mean_pct, std_pct, n_samples = _gradcam_mask_stats(model, val_nods, device)
    stats_lines = [
        "=== GRAD-CAM MASK ALIGNMENT (TotalSegmentator constraint) ===",
        f"Val samples evaluated: {n_samples}",
        f"Mean % activation inside mask: {mean_pct:.1f}%",
        f"Std:                           {std_pct:.1f}%",
        "(Higher than HU model = TS mask provides tighter/better constraint)",
    ]
    stats_text = "\n".join(stats_lines)
    print(stats_text)
    with open(os.path.join(run_dir, "gradcam_stats_ts.txt"), "w") as f:
        f.write(stats_text + "\n")

    # ── Grad-CAM visualisation (4 examples) ──────────────────────────────────
    if not val_nods:
        print("No validation samples — skipping Grad-CAM visualisation")
        return

    model.eval()
    collected = 0
    n_rows = min(4, len(val_nods))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for images, masks, labels, _ in DataLoader(
        LIDCDataset(val_nods, use_ts_mask=True), batch_size=1
    ):
        if collected >= 4:
            break
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images).squeeze(1)
        logits.sum().backward()
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
        ax[1].set_title("TS mask")
        ax[1].axis("off")

        ax[2].imshow(cam_up, cmap="jet")
        ax[2].set_title("Grad-CAM heatmap")
        ax[2].axis("off")

        ax[3].imshow(img_np, cmap="gray")
        ax[3].imshow(cam_up, cmap="jet", alpha=0.5)
        ax[3].contour(mask_np, levels=[0.5], colors="cyan", linewidths=1)
        ax[3].set_title(f"Overlay (cyan=TS mask) | {inside:.0f}% inside")
        ax[3].axis("off")

        model.clear_hooks()
        collected += 1

    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, "gradcam_examples_ts.png"), dpi=150)
    plt.close(fig)
    print(f"Saved TS Grad-CAM examples → {run_dir}/gradcam_examples_ts.png")


if __name__ == "__main__":
    train_ts()
