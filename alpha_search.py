import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from config import BATCH_SIZE, LR, SEED, RESULTS_DIR
from dataset import LIDCDataset, load_nodules, patient_split
from model import NoduleClassifier

# [ABLATION STUDY] Alpha sensitivity analysis
# Tests model sensitivity to attention penalty weight
ALPHA_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
EPOCHS = 10


def train_with_alpha(alpha, train_nods, val_nods, device, pos_weight):
    # [ABLATION STUDY] Single training run with given alpha
    # Identical training loop to train.py; returns best val AUC
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_ds = LIDCDataset(train_nods)
    val_ds = LIDCDataset(val_nods)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = NoduleClassifier().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_auc = 0.0
    model_path = os.path.join(RESULTS_DIR, f"best_model_alpha_{alpha}.pt")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch_idx, (images, masks, labels, _) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images).squeeze(1)

            # [ROSS ET AL. 2017] classification + lambda x explanation
            bce_loss = criterion(logits, labels)
            bce_loss.backward(retain_graph=True)
            cam = model.get_gradcam()

            cam_resized = F.interpolate(
                cam.unsqueeze(1),
                size=(masks.shape[2], masks.shape[3]),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

            mask_2d = masks.squeeze(1)

            # [ROSS ET AL. 2017] Penalise attention at irrelevant regions
            penalty = (cam_resized * (1.0 - mask_2d)).sum(dim=(1, 2)) / (
                cam_resized.sum(dim=(1, 2)) + 1e-8
            )

            # [ZHANG ET AL. 2022] ad-CSL adaptive scaling
            scale = 2.0 * (torch.sigmoid(logits) - 0.5).abs()

            # [ROSS ET AL. 2017] classification + lambda x explanation
            total_loss = bce_loss + alpha * (scale * penalty).mean()

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            model.clear_hooks()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, _masks, labels, _ in val_loader:
                images = images.to(device)
                logits = model(images).squeeze(1)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())
                model.clear_hooks()

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = float("nan")

        print(f"  alpha={alpha} | Epoch {epoch:02d} | Val AUC {auc:.4f}")

        if not np.isnan(auc) and auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), model_path)

    return best_auc


def run_alpha_search():
    # [ABLATION STUDY] Alpha sensitivity analysis
    # Tests model sensitivity to attention penalty weight
    os.makedirs(RESULTS_DIR, exist_ok=True)

    nodules = load_nodules()
    train_nods, val_nods, _ = patient_split(nodules)

    if not train_nods:
        print("No training samples — run build_cache.py first.")
        return None, {}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds_tmp = LIDCDataset(train_nods)
    pos_weight = train_ds_tmp.class_weights().to(device)

    results = {}
    for alpha in ALPHA_GRID:
        print(f"\n=== Training with alpha={alpha} ===")
        best_auc = train_with_alpha(alpha, train_nods, val_nods, device, pos_weight)
        results[alpha] = best_auc
        print(f"Alpha {alpha} -> Best Val AUC: {best_auc:.4f}")

    # Find optimal alpha
    valid = {a: v for a, v in results.items() if v == v}
    if not valid:
        print("No valid AUC values — all NaN.")
        return None, results

    optimal_alpha = max(valid, key=valid.get)
    baseline_auc = results.get(0.0, float("nan"))

    # Print sensitivity table
    header = (
        "\n=== ALPHA SENSITIVITY RESULTS ===\n"
        f"{'Alpha':>6} | {'Best Val AUC':>12} | {'vs Baseline':>11}\n"
        f"{'------':>6}-+-{'------------':>12}-+-{'----------':>11}"
    )
    print(header)

    table_lines = [header]
    for a in ALPHA_GRID:
        auc_v = results.get(a, float("nan"))
        auc_str = f"{auc_v:.4f}" if auc_v == auc_v else " nan "
        diff = auc_v - baseline_auc if (auc_v == auc_v and baseline_auc == baseline_auc) else float("nan")
        diff_str = f"{diff:+.4f}" if diff == diff else "  nan "
        row = f"{a:>6.2f} | {auc_str:>12} | {diff_str:>11}"
        print(row)
        table_lines.append(row)

    best_line = f"\nBEST:  alpha={optimal_alpha:.2f} -> AUC {valid[optimal_alpha]:.4f}"
    print(best_line)
    table_lines.append(best_line)

    # Save results text
    results_path = os.path.join(RESULTS_DIR, "alpha_search_results.txt")
    with open(results_path, "w") as f:
        f.write("\n".join(table_lines) + "\n")
    print(f"\nSaved results to {results_path}")

    # Plot alpha vs val AUC sensitivity curve
    alphas_plot = [a for a in ALPHA_GRID if results.get(a, float("nan")) == results.get(a, float("nan"))]
    aucs_plot = [results[a] for a in alphas_plot]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(alphas_plot, aucs_plot, marker="o", linewidth=2, color="steelblue", label="Val AUC")

    # Mark best alpha with red dot
    best_idx = alphas_plot.index(optimal_alpha)
    ax.scatter([optimal_alpha], [valid[optimal_alpha]], color="red", zorder=5, s=100, label=f"Best α={optimal_alpha}")
    ax.annotate(
        f"α={optimal_alpha}\nAUC={valid[optimal_alpha]:.4f}",
        xy=(optimal_alpha, valid[optimal_alpha]),
        xytext=(optimal_alpha + 0.05, valid[optimal_alpha] - 0.02),
        fontsize=9,
        color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    ax.set_xlabel("Alpha (attention penalty weight)", fontsize=12)
    ax.set_ylabel("Best Validation AUC", fontsize=12)
    ax.set_title("Alpha Sensitivity Analysis", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "alpha_sensitivity.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved sensitivity plot to {plot_path}")

    return optimal_alpha, results


if __name__ == "__main__":
    run_alpha_search()
