# =============================================================================
# baseline.py
# =============================================================================
# PURPOSE
# -------
# Trains the "baseline" variant of the LIDC-IDRI lung nodule malignancy
# classifier — a ResNet-based binary classifier with NO lung-mask constraint.
# The baseline is the control condition in the dissertation experiments: it
# shows what accuracy and Grad-CAM behaviour look like when the model is free
# to attend to any region of the image, not just the lung parenchyma.
#
# WHAT IS THE BASELINE?
# ----------------------
# NoduleClassifier (ResNet backbone, defined in model.py) is trained with a
# standard binary cross-entropy loss on (image, label) pairs.  The lung mask
# stored in each cache sample IS loaded and passed through the DataLoader but
# is NOT used in the loss function — the mask column (_masks) is intentionally
# discarded during training.
# This is deliberate: the baseline measures Grad-CAM / lung-mask alignment
# without any mask incentive.  It may still attend partly inside the lung
# because the crop contains lung anatomy, but any alignment is not enforced
# by the loss.  Experiment 2 (train.py) adds a mask-alignment penalty and
# tests whether that increases inside-mask attention.
#
# WHAT DOES THIS SCRIPT PRODUCE?
# --------------------------------
# All output is written to results/<run_id>/:
#   baseline_log.txt              — per-epoch val accuracy and AUC.
#   baseline_model.pt             — state dict of the best-AUC checkpoint.
#   split_baseline.txt            — patient IDs and class counts per split.
#   gradcam_stats_baseline.txt    — mean/std % of Grad-CAM inside lung mask
#                                   over the full validation set.
#   gradcam_examples_baseline.png — 4-row figure: CT | mask | heatmap | overlay.
#   test_results_baseline.txt     — AUC, accuracy, F1, sensitivity, specificity
#                                   on the held-out test set.
# results/latest_run.txt is also updated so compare_all.py can locate this
# run automatically without requiring --run-id.
#
# OVERALL FLOW
# ------------
#   1. Parse CLI arguments (--run-id, --cache-path, --epochs, --batch-size).
#   2. Create the results sub-directory; seed both PyTorch and NumPy RNGs.
#   3. Load the cache via load_nodules_hu; split into train / val / test;
#      write the split report to disk.
#   4. Build weighted BCE loss to counteract class imbalance (more benign
#      samples than malignant in LIDC).
#   5. Train for EPOCHS epochs:
#        - Forward pass → BCE loss → backward → Adam step.
#        - After each epoch: compute val AUC; save checkpoint if best so far.
#   6. Reload the best-AUC checkpoint.
#   7. Compute Grad-CAM mask alignment stats over the full validation set.
#   8. Render 4 Grad-CAM visualisation examples (CT | mask | heatmap | overlay).
#   9. Run final evaluation on the held-out test set.
# =============================================================================

import os           # file-system operations: makedirs, path joins, existence checks
import argparse     # parse command-line flags so the script is reconfigurable
                    # without editing source code
import csv          # write per-sample prediction audit files
from collections import Counter     # count benign / malignant labels per split
                                    # using a hash-map tally — cleaner than two
                                    # separate list comprehensions
from datetime import datetime       # generate date-stamped run IDs

import torch                            # core PyTorch — tensors, autograd, device management
import torch.nn.functional as F         # functional API; used only for F.interpolate
                                        # to upsample coarse Grad-CAM maps to image resolution
from torch.utils.data import DataLoader # batches the dataset into mini-batches for
                                        # training and evaluation loops

import numpy as np  # array maths — metric aggregation and legacy RNG seeding

import matplotlib
matplotlib.use("Agg")
# Force the non-interactive "Agg" (Anti-Grain Geometry) backend before importing
# pyplot.  Without this line, matplotlib tries to connect to a display, which
# fails on headless servers (SSH sessions, Slurm jobs without X11 forwarding)
# and raises a RuntimeError.  Agg renders to in-memory PNG buffers instead
# of a screen, which is all we need for saving figures to disk.
import matplotlib.pyplot as plt         # figure / axes API for the Grad-CAM grid

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
# roc_auc_score : area under the ROC curve — the primary model selection metric.
#   Threshold-independent, so it measures discrimination ability across the
#   full range of prediction thresholds, not just at 0.5.
# f1_score      : harmonic mean of precision and recall at threshold 0.5.
#   Captures positive-class (malignant) performance without being misled by
#   class imbalance the way raw accuracy can be.
# confusion_matrix : TP/TN/FP/FN counts — needed to derive sensitivity and
#   specificity, the clinically meaningful metrics for cancer screening.

# ---------------------------------------------------------------------------
# Config / module imports
# ---------------------------------------------------------------------------
from config import EPOCHS, BATCH_SIZE, LR, SEED, RESULTS_DIR, IMG_SIZE, TRAIN_CACHE_PATH
# EPOCHS          : default number of training epochs (overridable via CLI).
# BATCH_SIZE      : default samples per gradient update (overridable via CLI).
# LR              : Adam learning rate.
# SEED            : fixed integer seed for reproducibility.
# RESULTS_DIR     : root directory where per-run result sub-folders are created.
# IMG_SIZE        : spatial dimension of the square input images (224).
#                   Used when upsampling Grad-CAM maps to match the input size.
# TRAIN_CACHE_PATH: default path to the pre-built cache pickle (overridable via CLI).

from dataset import LIDCDataset, load_nodules_hu, patient_split
# LIDCDataset    : torch Dataset wrapping the list of sample dicts produced by
#                  build_cache.py.  Returns (image_tensor, mask_tensor, label) tuples.
# load_nodules_hu: loads the cache pickle and filters to samples with HU masks.
# patient_split  : splits the sample list into train / val / test by patient ID,
#                  ensuring no patient appears in more than one split (prevents
#                  data leakage from patients with multiple nodules).

from model import NoduleClassifier
# NoduleClassifier: ResNet-based binary classifier with Grad-CAM hooks.
#   Exposes .get_gradcam() to retrieve the last computed Grad-CAM activation
#   map and .clear_hooks() to release hook state between forward passes.


# =============================================================================
# GRAD-CAM MASK ALIGNMENT STATS
# =============================================================================
# PURPOSE
# -------
# Measures how much of the model's visual attention (expressed by Grad-CAM)
# falls inside the lung mask.  This is the core quantitative metric for
# comparing the baseline (no constraint) against the mask-penalised model
# (train.py): a higher percentage means the model looks more at the lung.
#
# HOW GRAD-CAM WORKS (briefly)
# -----------------------------
# Grad-CAM (Gradient-weighted Class Activation Mapping; Selvaraju et al. 2017)
# computes a spatial attention heatmap by:
#   1. Running a forward pass to get the predicted logit.
#   2. Backpropagating the logit to the last convolutional layer.
#   3. Global-average-pooling the per-channel gradients to get per-channel
#      importance weights.
#   4. Taking a weighted sum of the feature maps using those weights.
# The result is a coarse heatmap (same spatial resolution as the last conv
# feature map, typically 7×7 for ResNet on 224×224 input) indicating which
# image regions influenced the prediction most.
#
# WHY model.eval() WITH torch.enable_grad()?
# -------------------------------------------
# eval() disables dropout and switches batch norm to use running statistics —
# both necessary for deterministic, production-quality predictions.
# But Grad-CAM requires gradient flow through the network even in eval mode,
# so torch.enable_grad() explicitly enables autograd inside this function,
# regardless of any outer torch.no_grad() context.
# =============================================================================
def _gradcam_mask_stats(model, val_nods, device):
    """Compute % of Grad-CAM activation energy inside lung mask over full val set."""
    model.eval()
    pct_list = []
    ds = LIDCDataset(val_nods)
    if len(ds) == 0:
        # Guard against an empty split (e.g. cache too small to produce a val set).
        # Returns (nan, nan, 0) so the caller can print "nan" rather than crashing
        # or producing a misleading zero.
        return float("nan"), float("nan"), 0
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    # batch_size=1: Grad-CAM is computed per image; batching offers no speed
    # benefit here and keeping it at 1 keeps tensor shapes simple throughout.
    # shuffle=False: order doesn't affect the aggregate mean / std.
    with torch.enable_grad():
        # enable_grad() ensures backward() can run even if this function is
        # called inside a torch.no_grad() outer context (e.g. from an eval block).
        for images, masks, _ in loader:
            # Third element (labels) is unused — we only need images for the
            # forward pass and masks to compute the alignment percentage.
            images = images.to(device)
            masks = masks.to(device)

            model.zero_grad(set_to_none=True)
            # Clear any stale gradients from a previous sample before the
            # forward pass.  set_to_none=True frees the gradient tensors
            # entirely (rather than zeroing in-place), saving GPU memory.
            model.clear_hooks()
            # Clear any feature-map activations saved by forward hooks from
            # the previous iteration, ensuring get_gradcam() below returns
            # fresh data for this sample only.

            logits = model(images).squeeze(1)
            # Forward pass: images shape (1, 3, H, W) -> logits shape (1,).
            # squeeze(1) removes the output feature dimension, leaving a
            # 1-element 1-D tensor — a single logit per image.
            # The forward pass also triggers any registered forward hooks
            # inside the model, saving the feature maps needed for Grad-CAM.

            model.class_scores(logits).sum().backward()
            # .sum() converts the 1-element tensor to a scalar as required
            # by .backward() (which expects a scalar "loss").  With batch_size=1
            # this is equivalent to .item() but stays on the computational graph.
            # .backward() computes d(logits)/d(feature_maps), firing the
            # gradient hooks that Grad-CAM relies on to compute importance weights.

            cam = model.get_gradcam()
            # Retrieves the Grad-CAM heatmap computed during the forward +
            # backward pass above.  Shape: (1, H', W') where H' and W' are the
            # spatial dimensions of the last convolutional feature map
            # (typically 7×7 for ResNet on 224×224 input).

            cam_up = F.interpolate(
                cam.unsqueeze(1),
                size=(masks.shape[2], masks.shape[3]),
                mode="bilinear", align_corners=False,
            ).squeeze(1)
            # cam.unsqueeze(1): (1, H', W') → (1, 1, H', W') — adds a channel
            #   dim required by F.interpolate (expects B × C × H × W input).
            # size=(masks.shape[2], masks.shape[3]): upsample to the full image
            #   resolution (H × W = 224 × 224) so the heatmap aligns pixel-by-pixel
            #   with the mask.  masks is (B=1, C=1, H, W) from the DataLoader,
            #   so shape[2]=H and shape[3]=W.
            # mode="bilinear": smooth upsampling — appropriate for a continuous
            #   activation map (Grad-CAM values are real-valued, not binary).
            # .squeeze(1): removes the channel dim, giving (1, H, W).

            mask_2d = masks.squeeze(1)
            # masks shape: (1, 1, H, W) — squeezing channel dim (index 1) gives
            # (1, H, W), matching cam_up's shape for element-wise multiplication.

            inside = (cam_up * mask_2d).sum().item()
            # Element-wise product zeroes out cam_up pixels outside the lung mask.
            # .sum() gives total Grad-CAM energy inside the lung region.
            # .item() converts the PyTorch scalar to a Python float.

            total = cam_up.sum().item() + 1e-8
            # Total Grad-CAM energy across the whole image.
            # + 1e-8: epsilon guard prevents division-by-zero on a blank heatmap
            # (e.g. if all gradients are exactly zero due to dead ReLUs).

            pct_list.append(inside / total * 100.0)
            # Percentage of activation energy inside the lung mask.
            # 0 % = all attention is outside the lung.
            # 100 % = all attention is inside the lung.

            model.clear_hooks()
            # Release gradient and activation tensors stored by the model's hooks.
            # Without clearing after each sample, tensors accumulate in GPU memory —
            # for 1000+ samples this would cause OOM.

    arr = np.array(pct_list)
    return float(arr.mean()), float(arr.std()), len(arr)
    # Returns (mean %, std %, n_samples) so the caller can report both the
    # average alignment and its variability across samples.


def _classification_metrics(labels, probs):
    """Return threshold-free and threshold-0.5 classification metrics."""
    preds_bin = [1 if p >= 0.5 else 0 for p in probs]
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    acc = sum(p == l for p, l in zip(preds_bin, labels)) / max(len(labels), 1)
    try:
        f1 = f1_score(labels, preds_bin, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(labels, preds_bin, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    except Exception:
        f1 = sensitivity = specificity = float("nan")

    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def _predict_probabilities(model, nods, device, batch_size, criterion=None, num_workers=0):
    """Run inference and optionally return mean BCE loss for a split."""
    model.eval()
    ds = LIDCDataset(nods)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    all_probs, all_labels = [], []
    total_loss = 0.0
    n_loss = 0
    with torch.no_grad():
        for images, _masks, labels in loader:
            images = images.to(device)
            labels_device = labels.to(device)
            logits = model(images).squeeze(1)
            if criterion is not None:
                total_loss += criterion(logits, labels_device).item() * images.size(0)
                n_loss += images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            model.clear_hooks()

    mean_loss = total_loss / n_loss if n_loss else float("nan")
    return all_labels, all_probs, mean_loss


def _write_predictions_csv(path, nods, labels, probs):
    """Save per-sample probabilities so later error analysis does not need reruns."""
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
        for index, (sample, label, prob) in enumerate(zip(nods, labels, probs)):
            pred = 1 if prob >= 0.5 else 0
            writer.writerow([
                index,
                sample.get("patient_id", ""),
                int(label),
                f"{prob:.8f}",
                pred,
                int(pred == int(label)),
            ])


# =============================================================================
# TEST SET EVALUATION
# =============================================================================
# PURPOSE
# -------
# Evaluates the final (best-AUC) model on the held-out test set and writes a
# summary text file.  Separated from the training loop so it is called cleanly
# after reloading the best checkpoint.
#
# METRICS REPORTED
# -----------------
# AUC         : threshold-independent discrimination; primary metric.
# Accuracy    : fraction of correct predictions at threshold 0.5.
# F1          : harmonic mean of precision and recall at threshold 0.5.
# Sensitivity : TP / (TP + FN) — fraction of malignant nodules correctly
#               identified ("true positive rate").  Critical for screening.
# Specificity : TN / (TN + FP) — fraction of benign nodules correctly
#               rejected ("true negative rate").  Limits unnecessary biopsies.
# Grad-CAM %  : mean % of activation inside the lung mask on the test set
#               (same computation as validation, for completeness).
# =============================================================================
def _test_evaluate(model, test_nods, device, run_dir, batch_size, criterion=None, num_workers=0):
    # test_nods  : list of sample dicts for the held-out test patients.
    # device     : "cuda" or "cpu" — must match where the model lives.
    # run_dir    : path to the current run's results directory; output is written here.
    # batch_size : inference batch size (passed in from CLI to match training config).
    if not test_nods:
        print("No test samples - skipping test evaluation.")
        return

    model.eval()
    ds = LIDCDataset(test_nods)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    # shuffle=False: evaluation order is irrelevant to metrics and keeping
    # it deterministic makes debugging easier.

    all_preds, all_labels = [], []
    total_loss = 0.0
    n_loss = 0
    with torch.no_grad():
        # no_grad() disables the autograd engine entirely — no gradient graph
        # is built, halving memory usage and speeding up inference.
        for images, _masks, labels in loader:
            # _masks: underscore prefix signals intentional discard —
            # the baseline doesn't use masks for prediction.
            images = images.to(device)
            labels_device = labels.to(device)
            logits = model(images).squeeze(1)
            if criterion is not None:
                total_loss += criterion(logits, labels_device).item() * images.size(0)
                n_loss += images.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            # sigmoid maps raw logits (unbounded) to probabilities in [0, 1].
            # .cpu() moves from GPU to host memory; .numpy() converts for
            # scikit-learn compatibility.
            all_preds.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
            model.clear_hooks()
            # Forward hooks (for feature map saving) fire even without a
            # backward pass.  Clearing them prevents stale hook state from
            # interfering with the Grad-CAM call later in this function.

    preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
    # Binary predictions at the standard 0.5 decision threshold.

    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = float("nan")
        # Raised when all test labels are the same class (one class absent).
        # Can occur when the test split is very small.

    acc = sum(p == l for p, l in zip(preds_bin, all_labels)) / max(len(all_labels), 1)
    # max(..., 1) guards against division-by-zero when test_nods is non-empty
    # but the DataLoader somehow yields zero samples (degenerate edge case).

    try:
        f1 = f1_score(all_labels, preds_bin, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(all_labels, preds_bin, labels=[0, 1]).ravel()
        # labels=[0, 1]: forces a full 2×2 matrix even when only one class is
        # present in the test set, making the 4-variable unpack always safe.
        # Without this argument, a single-class test set produces a 1×1 matrix
        # whose .ravel() gives one element and the unpack raises ValueError.
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        # sensitivity = recall on the positive (malignant) class.
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        # specificity = recall on the negative (benign) class.
    except Exception:
        f1 = sensitivity = specificity = float("nan")
        # Broad except: catches any remaining degenerate input (e.g. empty arrays).

    test_loss = total_loss / n_loss if n_loss else float("nan")
    _write_predictions_csv(
        os.path.join(run_dir, "test_predictions_baseline.csv"),
        test_nods,
        all_labels,
        all_preds,
    )

    print("\nComputing test Grad-CAM mask alignment...")
    gcam_mean, gcam_std, gcam_n = _gradcam_mask_stats(model, test_nods, device)
    # Runs the same mask-alignment computation on the test set for a complete
    # picture of alignment on held-out data, not just validation.

    def _f(v):
        return f"{v:.4f}" if not np.isnan(v) else "nan"
    # Local helper: format a float to 4 decimal places, or "nan" as a string.
    # Makes all numeric lines in the output file consistent in width and format.

    lines = [
        "=== TEST SET EVALUATION (Baseline) ===",
        f"Test samples:           {len(all_labels)}",
        f"Test BCE loss:          {_f(test_loss)}",
        f"Test AUC:               {_f(auc)}",
        f"Test Accuracy:          {_f(acc)}",
        f"Test F1:                {_f(f1)}",
        f"Test Sensitivity:       {_f(sensitivity)}",
        f"Test Specificity:       {_f(specificity)}",
        "",
        "=== TEST GRAD-CAM MASK ALIGNMENT (Baseline) ===",
        f"Test samples evaluated: {gcam_n}",
        f"Mean % inside mask:     {gcam_mean:.1f}%",
        f"Std:                    {gcam_std:.1f}%",
    ]
    text = "\n".join(lines)
    print("\n" + text)
    out_path = os.path.join(run_dir, "test_results_baseline.txt")
    with open(out_path, "w") as f:
        f.write(text + "\n")
    print(f"Saved -> {out_path}")


# =============================================================================
# RUN ID GENERATION
# =============================================================================
def _make_run_id(results_dir):
    # Generates a unique run ID of the form "YYYY-MM-DD_runN" where N is the
    # smallest positive integer whose corresponding sub-directory does not yet
    # exist in results_dir.  Running the script multiple times on the same day
    # therefore creates separate, non-overwriting result directories:
    # 2026-06-05_run1, 2026-06-05_run2, etc.
    today = datetime.now().strftime("%Y-%m-%d")
    n = 1
    while os.path.exists(os.path.join(results_dir, f"{today}_run{n}")):
        n += 1
    return f"{today}_run{n}"


# =============================================================================
# SPLIT STATISTICS HELPERS
# =============================================================================
# PURPOSE
# -------
# Produce a human-readable audit trail of exactly which patients and samples
# went into each split.  Written to split_baseline.txt in the run directory
# so that:
#   1. The split can be reproduced or compared across runs.
#   2. Patient-level overlap across splits can be verified to be zero
#      (a non-zero overlap would indicate a data leakage bug in patient_split).
# =============================================================================
def _split_stats(name, nodules):
    # name    : display name for the split ("train", "val", or "test").
    # nodules : list of sample dicts for this split.
    # Returns a dict summarising the split for use in _write_split_report.
    labels = Counter(n["label"] for n in nodules)
    # Counter maps each label value (0 or 1) to its count in this split.
    # Using Counter rather than two separate sum() calls avoids iterating
    # the list twice and makes it easy to handle unexpected label values.
    patients = sorted({n["patient_id"] for n in nodules})
    # Set comprehension deduplicates patient IDs (a patient may have multiple
    # nodules, but we want to count unique patients).  sorted() ensures a
    # stable order for diff-friendly text output.
    n_pos = labels.get(1, 0)
    n_neg = labels.get(0, 0)
    # .get(key, 0): returns 0 if the label is entirely absent from this split
    # (e.g. a tiny test split with only benign samples).
    pos_pct = (n_pos / len(nodules) * 100.0) if nodules else 0.0
    # Fraction of malignant samples in this split.  The LIDC dataset has
    # roughly 30-35 % malignant samples; large deviations here would suggest
    # the split is unbalanced and may bias evaluation.
    return {
        "name": name,
        "samples": len(nodules),
        "patients": len(patients),
        "benign": n_neg,
        "malignant": n_pos,
        "malignant_pct": pos_pct,
        "patient_ids": patients,
    }


def _write_split_report(run_dir, cache_path, train_nods, val_nods, test_nods):
    # Computes per-split statistics, checks patient-level overlap between
    # splits, and writes a plain-text report to split_baseline.txt.
    # Also prints the summary section to stdout so it's visible in the log.
    train = _split_stats("train", train_nods)
    val   = _split_stats("val",   val_nods)
    test  = _split_stats("test",  test_nods)

    train_p = set(train["patient_ids"])
    val_p   = set(val["patient_ids"])
    test_p  = set(test["patient_ids"])
    # Convert to sets for O(1) intersection checks.

    lines = [
        "=== BASELINE DATA SPLIT ===",
        f"Cache path: {cache_path}",
        "Split type: patient-level 70/15/15",
        "",
    ]
    for split in (train, val, test):
        lines.extend([
            f"[{split['name']}]",
            f"Samples: {split['samples']}",
            f"Patients: {split['patients']}",
            f"Benign: {split['benign']}",
            f"Malignant: {split['malignant']}",
            f"Malignant %: {split['malignant_pct']:.1f}",
            "",
        ])

    lines.extend([
        "[patient overlap]",
        f"Train/val overlap: {len(train_p & val_p)}",
        f"Train/test overlap: {len(train_p & test_p)}",
        f"Val/test overlap: {len(val_p & test_p)}",
        # All three values should be 0.  A non-zero value means the same
        # patient appears in two splits — a data leakage bug that would
        # artificially inflate validation / test metrics.
    ])

    path = os.path.join(run_dir, "split_baseline.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
        f.write("\n[train patient ids]\n")
        f.write("\n".join(train["patient_ids"]) + "\n")
        f.write("\n[val patient ids]\n")
        f.write("\n".join(val["patient_ids"]) + "\n")
        f.write("\n[test patient ids]\n")
        f.write("\n".join(test["patient_ids"]) + "\n")
    # The full patient ID lists are appended after the summary so the file can
    # be used to reconstruct or audit the split without re-running the script.

    print("\n".join(lines))
    print(f"Saved -> {path}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def train_baseline():

    # ------------------------------------------------------------------
    # Command-line argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", type=str, default=None,
                        help="Shared run ID for grouping results (auto-generated if omitted)")
    # --run-id allows baseline.py and train.py to write into the same run
    # directory so compare_all.py can compare them side-by-side.
    parser.add_argument("--cache-path", type=str, default=None,
                        help=f"Path to cache file (default: {TRAIN_CACHE_PATH})")
    # --cache-path lets you point at a non-default cache, e.g. a small test
    # cache (cache/cache_200.pkl) for quick smoke tests.
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help=f"Number of training epochs (default: {EPOCHS})")
    # --epochs overrides the config value so you can run a short smoke test
    # (--epochs 2) without editing config.py.
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size (default: {BATCH_SIZE})")
    # --batch-size is forwarded to both training and test evaluation DataLoaders
    # so memory usage is consistent across the run.
    parser.add_argument("--lr", type=float, default=LR,
                        help=f"Learning rate (default: {LR})")
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"],
                        help="Optimizer to use: adam preserves old baseline; adamw enables decoupled weight decay.")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay for optimizer regularisation (default: 0.0)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers. Keep 0 on Windows unless you know multiprocessing is stable.")
    parser.add_argument("--augment", action="store_true",
                        help="Apply conservative CT-safe augmentation to training samples only.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Run directory setup
    # ------------------------------------------------------------------
    run_id = args.run_id or _make_run_id(RESULTS_DIR)
    run_dir = os.path.join(RESULTS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    # exist_ok=True: if the directory already exists (e.g. --run-id points to
    # an existing run from train.py), don't error — write results alongside
    # the existing files.

    with open(os.path.join(RESULTS_DIR, "latest_run.txt"), "w") as f:
        f.write(run_id)
    # Record this run ID so compare_all.py can discover it automatically
    # without the user passing --run-id manually.  Overwriting on every run
    # means the file always reflects the most recently started run.

    print(f"[RUN] {run_id}  ->  {run_dir}")
    print(f"      To compare after all scripts: python compare_all.py --run-id {run_id}")

    # ------------------------------------------------------------------
    # Reproducibility: seed both RNGs
    # ------------------------------------------------------------------
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    # Seeding both ensures:
    #   - Weight initialisation in NoduleClassifier is deterministic.
    #   - DataLoader shuffle (train_loader) produces the same batch order.
    #   - Any NumPy random calls in dataset code are reproducible.
    # Note: for full CUDA reproducibility, torch.backends.cudnn.deterministic=True
    # would also be needed, but it significantly slows training and is omitted.

    # ------------------------------------------------------------------
    # Load data, split, and write the split audit report
    # ------------------------------------------------------------------
    cache_path = args.cache_path or TRAIN_CACHE_PATH
    # Resolve the cache path once here so it can be passed to _write_split_report
    # for inclusion in the split audit file.
    nodules = load_nodules_hu(cache_path=cache_path)
    # Loads the cache pickle and returns only samples that have a valid HU mask.

    train_nods, val_nods, test_nods = patient_split(nodules)
    # patient_split assigns whole patients to each split — if a patient has
    # two nodules, both go to the same split, preventing cross-split leakage.

    _write_split_report(run_dir, cache_path, train_nods, val_nods, test_nods)
    # Writes split_baseline.txt and prints a summary.  Done before training
    # so the report is available even if training crashes early.

    if not train_nods:
        print("No training samples / cache too small")
        return
        # Abort before creating DataLoaders rather than crashing inside the
        # training loop with a confusing "division by zero" or empty-batch error.

    # ------------------------------------------------------------------
    # Datasets and DataLoaders
    # ------------------------------------------------------------------
    train_ds = LIDCDataset(train_nods, augment=args.augment)
    val_ds   = LIDCDataset(val_nods)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    # shuffle=True: randomises sample order within each epoch so the model
    # does not overfit to the ordering of patients in the cache.
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    # shuffle=False: validation order is irrelevant to metrics; fixed order
    # makes it easier to track per-sample predictions across epochs.

    # ------------------------------------------------------------------
    # Model, loss, and optimiser
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoduleClassifier().to(device)
    # Move all model parameters to the GPU (if available) so all forward
    # and backward passes run on the accelerator.

    pos_weight = train_ds.class_weights().to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # BCEWithLogitsLoss fuses sigmoid + binary cross-entropy into one
    # numerically stable operation, avoiding float overflow in the naive
    # sigmoid → log chain.
    # pos_weight: scalar w that multiplies the loss for positive (malignant)
    # samples by w.  When benign >> malignant (typical in LIDC), setting
    # w = n_benign / n_malignant upweights rare positives so the model does
    # not learn to always predict benign.  class_weights() computes this ratio
    # from the training split.

    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
    # Adam: adaptive moment estimation.  Adjusts the effective learning rate
    # per parameter, making it robust to the sparse gradients and noisy losses
    # common when training on small medical image datasets.
    # AdamW decouples weight decay from the gradient update, which is usually
    # the cleaner choice when regularising a pretrained backbone.

    run_info_path = os.path.join(run_dir, "baseline_info.txt")
    with open(run_info_path, "w") as f:
        f.write("=== BASELINE TRAINING CONFIG ===\n")
        f.write("Goal: AUC-first classifier training; no attention penalty in loss.\n")
        f.write(f"Cache path: {cache_path}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs requested: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Num workers: {args.num_workers}\n")
        f.write(f"Training augmentation: {args.augment}\n")
        if args.augment:
            f.write("Augmentation details: training-only; validation/test unaugmented; unchanged p=0.25; hflip p=0.5; rotate +/-7 deg; translate +/-4%; scale 0.96-1.04; contrast 0.92-1.08; brightness +/-0.03; noise sigma 0.01 with p=0.25.\n")
            f.write("Attention penalty: none; this is a weighted-BCE classification-only baseline.\n")
        f.write(f"Loss: BCEWithLogitsLoss(pos_weight={pos_weight.item():.6f})\n")
        f.write(f"Train/val/test samples: {len(train_nods)}/{len(val_nods)}/{len(test_nods)}\n")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    log_path = os.path.join(run_dir, "baseline_log.txt")
    epochs_csv_path = os.path.join(run_dir, "baseline_epochs.csv")
    best_auc = 0.0
    best_epoch = 0
    best_model_path = os.path.join(run_dir, "baseline_model.pt")

    with open(log_path, "w") as log_f, open(epochs_csv_path, "w", newline="") as epoch_csv_f:
        epoch_writer = csv.writer(epoch_csv_f)
        epoch_writer.writerow([
            "epoch",
            "train_bce",
            "val_bce",
            "val_auc",
            "val_accuracy_0p5",
            "val_f1_0p5",
            "val_sensitivity_0p5",
            "val_specificity_0p5",
        ])
        for epoch in range(1, args.epochs + 1):

            # ── Training phase ────────────────────────────────────────────
            model.train()
            # train() re-enables dropout and switches batch norm back to
            # computing statistics from the current mini-batch (rather than
            # using running statistics as in eval mode).
            train_loss_sum = 0.0
            train_seen = 0
            for batch_idx, (images, _masks, labels) in enumerate(train_loader):
                # _masks: intentionally unused — baseline trains without any
                # mask signal.  The leading underscore is a Python convention
                # to indicate the variable is deliberately discarded.
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                # Clear accumulated gradients from the previous step.  Must
                # be called before .backward() or gradients accumulate across
                # batches, giving incorrect parameter updates.

                logits = model(images).squeeze(1)
                # Forward pass: (B, 3, H, W) -> (B,) after squeezing the
                # output feature dimension.

                total_loss = criterion(logits, labels)
                # BCEWithLogitsLoss: compares per-sample logits to binary
                # float labels {0.0, 1.0}.
                train_loss_sum += total_loss.item() * images.size(0)
                train_seen += images.size(0)

                total_loss.backward()
                # Compute gradients of total_loss w.r.t. all model parameters
                # via backpropagation.

                optimizer.step()
                # Update parameters using the computed gradients and Adam's
                # running moment estimates.

                model.clear_hooks()
                # Release activation tensors stored by forward hooks.  During
                # training, forward hooks accumulate feature maps for Grad-CAM,
                # but those are not needed here — clearing immediately frees memory.

                print(
                    f"Epoch {epoch:02d} | Batch {batch_idx:04d} | "
                    f"BCE {total_loss.item():.4f}"
                )
                # :02d/:04d: zero-padding so log lines sort correctly when
                # there are more than 9 epochs or 999 batches.

            # ── Validation phase ─────────────────────────────────────────
            model.eval()
            all_preds, all_labels = [], []
            val_loss_sum = 0.0
            val_seen = 0
            with torch.no_grad():
                # no_grad() prevents building a gradient graph during
                # validation, saving memory and speeding up inference.
                for images, _masks, labels in val_loader:
                    images = images.to(device)
                    labels_device = labels.to(device)
                    logits = model(images).squeeze(1)
                    val_loss_sum += criterion(logits, labels_device).item() * images.size(0)
                    val_seen += images.size(0)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs.tolist())
                    all_labels.extend(labels.numpy().tolist())
                    model.clear_hooks()
                    # Forward hooks still fire during eval forward passes.
                    # Clearing them here keeps hook state clean for the
                    # Grad-CAM steps that follow training.

            preds_bin = [1 if p >= 0.5 else 0 for p in all_preds]
            acc = sum(p == l for p, l in zip(preds_bin, all_labels)) / max(len(all_labels), 1)
            train_loss_mean = train_loss_sum / max(train_seen, 1)
            val_loss_mean = val_loss_sum / max(val_seen, 1)
            try:
                auc = roc_auc_score(all_labels, all_preds)
            except ValueError:
                auc = float("nan")
                # Raised when the validation set contains only one class —
                # possible in early epochs when the model predicts all benign.
                # Treating AUC as NaN causes the checkpoint-saving condition
                # below to skip this epoch cleanly.

            val_metrics = _classification_metrics(all_labels, all_preds)
            log_line = (
                f"Epoch {epoch:02d} | Train BCE {train_loss_mean:.4f} | "
                f"Val BCE {val_loss_mean:.4f} | Val AUC {auc:.4f} | "
                f"Val Acc {acc:.4f} | Val F1 {val_metrics['f1']:.4f} | "
                f"Val Sens {val_metrics['sensitivity']:.4f} | "
                f"Val Spec {val_metrics['specificity']:.4f}\n"
            )
            log_f.write(log_line)
            epoch_writer.writerow([
                epoch,
                f"{train_loss_mean:.8f}",
                f"{val_loss_mean:.8f}",
                f"{auc:.8f}",
                f"{acc:.8f}",
                f"{val_metrics['f1']:.8f}",
                f"{val_metrics['sensitivity']:.8f}",
                f"{val_metrics['specificity']:.8f}",
            ])
            log_f.flush()
            epoch_csv_f.flush()
            # flush() writes the line to disk immediately rather than waiting
            # for the file buffer to fill.  This lets you `tail -f baseline_log.txt`
            # to watch training progress in real time.
            print(log_line, end="")

            # ── Model checkpointing ───────────────────────────────────────
            if not np.isnan(auc) and auc > best_auc:
                best_auc = auc
                best_epoch = epoch
                torch.save(model.state_dict(), best_model_path)
                # state_dict() saves only learned parameters (weights and biases),
                # not the model architecture.  This is standard PyTorch practice:
                # architecture comes from instantiating NoduleClassifier, and
                # parameters are loaded from the state dict.
                print(f"  -> Saved best baseline model (AUC {best_auc:.4f})")

    # ------------------------------------------------------------------
    # Reload the best checkpoint
    # ------------------------------------------------------------------
    # After the training loop the model's weights reflect the LAST epoch,
    # which may not be the best one (the model could have overfit in later
    # epochs).  Reloading the best-AUC checkpoint ensures all subsequent
    # analysis (Grad-CAM stats, visualisations, test evaluation) reflects
    # peak validation performance.
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        # map_location=device: ensures a checkpoint saved on GPU is correctly
        # loaded on a CPU-only machine (and vice versa).  Without this, loading
        # a GPU checkpoint on CPU raises a RuntimeError.

    val_labels, val_probs, val_loss = _predict_probabilities(
        model,
        val_nods,
        device,
        args.batch_size,
        criterion=criterion,
        num_workers=args.num_workers,
    )
    val_metrics = _classification_metrics(val_labels, val_probs)
    _write_predictions_csv(
        os.path.join(run_dir, "val_predictions_baseline.csv"),
        val_nods,
        val_labels,
        val_probs,
    )
    with open(run_info_path, "a") as f:
        f.write("\n=== BEST VALIDATION CHECKPOINT ===\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best validation AUC during training: {best_auc:.6f}\n")
        f.write(f"Reloaded validation BCE: {val_loss:.6f}\n")
        f.write(f"Reloaded validation AUC: {val_metrics['auc']:.6f}\n")
        f.write(f"Reloaded validation accuracy at 0.5: {val_metrics['acc']:.6f}\n")
        f.write(f"Reloaded validation F1 at 0.5: {val_metrics['f1']:.6f}\n")
        f.write(f"Reloaded validation sensitivity at 0.5: {val_metrics['sensitivity']:.6f}\n")
        f.write(f"Reloaded validation specificity at 0.5: {val_metrics['specificity']:.6f}\n")

    # ------------------------------------------------------------------
    # Grad-CAM mask alignment — validation set
    # ------------------------------------------------------------------
    print("\nComputing Grad-CAM mask alignment stats...")
    mean_pct, std_pct, n_samples = _gradcam_mask_stats(model, val_nods, device)
    stats_lines = [
        "=== GRAD-CAM MASK ALIGNMENT (Baseline - no constraint) ===",
        f"Val samples evaluated: {n_samples}",
        f"Mean % activation inside mask: {mean_pct:.1f}%",
        f"Std:                           {std_pct:.1f}%",
        "(No mask constraint was applied during baseline training.)",
        # A baseline model can still attend inside the lung because lung tissue
        # is predictive, but the loss did not force that behaviour. Experiment 2
        # (train.py) tests whether mask supervision increases this percentage.
    ]
    stats_text = "\n".join(stats_lines)
    print(stats_text)
    with open(os.path.join(run_dir, "gradcam_stats_baseline.txt"), "w") as f:
        f.write(stats_text + "\n")

    # ------------------------------------------------------------------
    # Grad-CAM visualisation — 4 examples from the validation set
    # ------------------------------------------------------------------
    # Produces a PNG with 4 rows × 4 columns:
    #   Col 0: raw CT slice (greyscale)
    #   Col 1: lung mask (greyscale, white = lung)
    #   Col 2: Grad-CAM heatmap (jet colourmap, warm = high attention)
    #   Col 3: CT + Grad-CAM overlay with red contour at mask boundary and
    #          the per-sample % of Grad-CAM inside the mask in the subplot title.
    if not val_nods:
        print("No validation samples - skipping Grad-CAM visualisation")
        return
        # If there are no validation samples, Grad-CAM examples cannot be drawn.
        # The normal fixed-HU cache has a non-empty validation split.

    model.eval()
    collected = 0
    n_rows = min(4, len(val_nods))
    fig, axes = plt.subplots(n_rows, 4, figsize=(16, 4 * n_rows))
    # n_rows × 4 subplots: up to 4 example nodules, 4 panel types each.
    # figsize=(16, 4*n_rows): 16 inches wide, 4 inches per row — each panel
    # is approximately 4×4 inches, readable at 150 dpi.
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    # plt.subplots returns a 1-D axes array when n_rows == 1, but 2-D when
    # n_rows > 1.  np.newaxis promotes the 1-D case to 2-D so that
    # axes[collected] always returns a length-4 array of axes regardless of
    # how many rows there are.

    for images, masks, labels in DataLoader(LIDCDataset(val_nods), batch_size=1):
        # A fresh DataLoader with batch_size=1 iterates one sample at a time.
        # We stop after 4 examples via the break below.
        if collected >= 4:
            break
        images = images.to(device)
        masks = masks.to(device)

        model.zero_grad(set_to_none=True)
        model.clear_hooks()
        # Reset state before this sample (same defensive pattern as in
        # _gradcam_mask_stats) to prevent any residual gradients or saved
        # activations from a previous iteration corrupting the Grad-CAM output.

        logits = model(images).squeeze(1)
        model.class_scores(logits).sum().backward()
        # Grad-CAM forward + backward — same pattern as _gradcam_mask_stats.
        # No torch.no_grad() wrapper; gradients are required for Grad-CAM.

        cam = model.get_gradcam()
        cam_up = F.interpolate(
            cam.unsqueeze(1), size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().detach().numpy()
        # .squeeze() without arguments removes ALL size-1 dimensions.
        # With B=1 and C=1, the result is a plain (H, W) = (224, 224) array
        # ready for imshow.
        # .detach(): breaks from the computational graph before converting to
        # NumPy.  Required because the tensor still has a grad_fn from backward.

        img_np   = images[0, 0].cpu().detach().numpy()
        # images shape: (1, 3, H, W) - [0, 0] selects batch 0, channel 0 -> (H, W).
        mask_np  = masks[0, 0].cpu().detach().numpy()
        # Same indexing for the mask.
        label_val = labels[0].item()
        # .item() extracts a Python scalar from a 0-dim or 1-element tensor.

        inside = float((cam_up * (mask_np > 0.5)).sum() / (cam_up.sum() + 1e-8) * 100)
        # Per-sample mask alignment percentage for the subplot title.
        # mask_np > 0.5 re-binarises the mask in case interpolation introduced
        # fractional values (though the mask from the DataLoader should already
        # be binary {0, 1}).

        ax = axes[collected]
        ax[0].imshow(img_np, cmap="gray")
        ax[0].set_title(f"CT slice (label={int(label_val)})")
        # label=0 → benign, label=1 → malignant.
        ax[0].axis("off")

        ax[1].imshow(mask_np, cmap="gray")
        ax[1].set_title("Lung mask (unused)")
        # "unused" makes it explicit that the baseline never incorporated the
        # mask during training.
        ax[1].axis("off")

        ax[2].imshow(cam_up, cmap="jet")
        ax[2].set_title("Grad-CAM heatmap")
        # jet colourmap: warm (red/yellow) = high attention, cool (blue) = low.
        ax[2].axis("off")

        ax[3].imshow(img_np, cmap="gray")
        ax[3].imshow(cam_up, cmap="jet", alpha=0.5)
        # Overlay the heatmap on the CT with 50 % transparency so the anatomy
        # is visible through the attention map.
        ax[3].contour(mask_np, levels=[0.5], colors="red", linewidths=1)
        # Draw the lung mask boundary as a red contour at the 0.5 iso-level,
        # letting the viewer immediately see whether warm Grad-CAM regions
        # align with the red lung outline.
        ax[3].set_title(f"Overlay (red=mask) | {inside:.0f}% inside")
        ax[3].axis("off")

        model.clear_hooks()
        collected += 1

    plt.tight_layout()
    fig.savefig(os.path.join(run_dir, "gradcam_examples_baseline.png"), dpi=150)
    # dpi=150: 150 dots per inch gives a 2400×(600*n_rows) pixel image —
    # large enough to read subplot titles without being excessively large on disk.
    plt.close(fig)
    # close() releases the figure from matplotlib's internal figure registry,
    # freeing the associated memory.  Without this, repeatedly creating figures
    # leaks memory and eventually triggers "Too many open figures" warnings.
    print(f"Saved Grad-CAM examples -> {run_dir}/gradcam_examples_baseline.png")

    # ------------------------------------------------------------------
    # Test set evaluation
    # ------------------------------------------------------------------
    print("\n--- Test Set Evaluation ---")
    _test_evaluate(
        model,
        test_nods,
        device,
        run_dir,
        args.batch_size,
        criterion=criterion,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    train_baseline()
    # Standard Python idiom: only run train_baseline() when this script is
    # executed directly (python baseline.py), not when it is imported as a
    # module by compare_all.py or a unit test.
