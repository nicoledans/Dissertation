"""Compare trained-model Grad-CAMs with LIDC radiologist nodule contours.

The completed HU cache used by the existing experiments predates the metadata
fields that identify the exact scan and nodule group. This script therefore
reconstructs eligible nodules from each held-out patient's raw LIDC scan using
the same label, slice-selection, and image-preprocessing rules as build_cache.py,
then matches the reconstructed image to the cached image.

Radiologist contours are used only for post-hoc evaluation and visualisation.
They are never passed to either model.
"""

import argparse
import csv
import hashlib
import math
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pylidc as pl
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from skimage.draw import polygon

from config import (
    HU_MAX,
    HU_MIN,
    IMG_SIZE,
    MALIGNANCY_THRESHOLD,
    MIN_RADIOLOGISTS,
    TRAIN_CACHE_PATH,
)
from dataset import patient_split
from model import NoduleClassifier


DEFAULT_BASELINE = "results/2026-06-06_run1/baseline_model.pt"
DEFAULT_HU_MODEL = "results/2026-06-06_run1/best_model.pt"
DEFAULT_OUTPUT_DIR = "results/2026-06-06_run1/nodule_gradcam_audit"


def _image_hash(image):
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _middle_slice_idx(annotation_group):
    indices = sorted(
        {
            contour.image_k_position
            for annotation in annotation_group
            for contour in annotation.contours
        }
    )
    if not indices:
        raise ValueError("Annotation group has no contour slices.")
    return int(indices[len(indices) // 2])


def _prepare_image(raw_slice):
    image = np.clip(raw_slice.astype(np.float32), HU_MIN, HU_MAX)
    image = (image - HU_MIN) / (HU_MAX - HU_MIN)
    factors = (IMG_SIZE / image.shape[0], IMG_SIZE / image.shape[1])
    return zoom(image, factors, order=1).astype(np.float32)


def _reader_mask(annotation, slice_idx, shape):
    """Rasterize one radiologist's contours on one source CT slice."""
    mask = np.zeros(shape, dtype=bool)
    for contour in annotation.contours:
        if int(contour.image_k_position) != int(slice_idx):
            continue
        coordinates = contour.to_matrix(include_k=False)
        rows, cols = polygon(
            coordinates[:, 0],
            coordinates[:, 1],
            shape=shape,
        )
        inclusion = str(getattr(contour, "inclusion", "TRUE")).upper()
        if inclusion == "FALSE":
            mask[rows, cols] = False
        else:
            mask[rows, cols] = True
    return mask


def _annotation_masks(annotation_group, slice_idx, shape):
    reader_masks = [
        _reader_mask(annotation, slice_idx, shape)
        for annotation in annotation_group
    ]
    reader_masks = [mask for mask in reader_masks if np.any(mask)]
    if not reader_masks:
        raise ValueError("No radiologist contour exists on selected slice.")

    votes = np.stack(reader_masks).sum(axis=0)
    union = votes >= 1
    majority = votes >= math.ceil(len(reader_masks) / 2)
    if not np.any(majority):
        majority = union

    factors = (IMG_SIZE / shape[0], IMG_SIZE / shape[1])
    union_224 = zoom(union.astype(np.uint8), factors, order=0).astype(bool)
    majority_224 = zoom(majority.astype(np.uint8), factors, order=0).astype(bool)
    return union_224, majority_224, len(reader_masks)


def _eligible_candidates(scan):
    volume = scan.to_volume()
    candidates = []
    for group_index, annotation_group in enumerate(scan.cluster_annotations()):
        if len(annotation_group) < MIN_RADIOLOGISTS:
            continue
        mean_rating = float(np.mean([a.malignancy for a in annotation_group]))
        if mean_rating == MALIGNANCY_THRESHOLD:
            continue
        label = int(mean_rating > MALIGNANCY_THRESHOLD)
        slice_idx = _middle_slice_idx(annotation_group)
        slice_idx = int(np.clip(slice_idx, 0, volume.shape[2] - 1))
        raw_slice = volume[:, :, slice_idx]
        image = _prepare_image(raw_slice)
        union, majority, readers = _annotation_masks(
            annotation_group, slice_idx, raw_slice.shape
        )
        candidates.append(
            {
                "image": image,
                "hash": _image_hash(image),
                "label": label,
                "patient_id": scan.patient_id,
                "scan_id": str(scan.id),
                "group_index": group_index,
                "slice_idx": slice_idx,
                "mean_rating": mean_rating,
                "reader_count": readers,
                "union": union,
                "majority": majority,
            }
        )
    return candidates


def _load_test_samples(cache_path):
    with open(cache_path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{cache_path} is a checkpoint, not a completed cache.")
    _, _, test = patient_split(samples)
    return test


def _match_test_samples(test_samples, max_matches):
    """Match held-out cached images to reconstructed raw LIDC candidates."""
    by_patient = {}
    for sample in test_samples:
        by_patient.setdefault(sample["patient_id"], []).append(sample)

    matches = []
    for patient_number, (patient_id, cached_samples) in enumerate(
        sorted(by_patient.items()), start=1
    ):
        if len(matches) >= max_matches:
            break
        print(
            f"Matching patient {patient_number}/{len(by_patient)}: "
            f"{patient_id} ({len(matches)}/{max_matches} matches)"
        )
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
        candidates = []
        for scan in scans:
            try:
                candidates.extend(_eligible_candidates(scan))
            except Exception as error:
                print(f"  WARNING: could not reconstruct scan {scan.id}: {error}")

        for cached in cached_samples:
            cached_hash = _image_hash(np.asarray(cached["image"], dtype=np.float32))
            eligible = [
                candidate
                for candidate in candidates
                if candidate["label"] == cached["label"]
                and candidate["hash"] == cached_hash
            ]
            if not eligible:
                # Float-library differences can very rarely change the exact
                # byte hash, so retain a strict numerical fallback.
                eligible = [
                    candidate
                    for candidate in candidates
                    if candidate["label"] == cached["label"]
                    and np.allclose(
                        candidate["image"],
                        cached["image"],
                        rtol=0,
                        atol=1e-6,
                    )
                ]
            if eligible:
                candidate = eligible[0]
                match = dict(candidate)
                match["lung_mask"] = np.asarray(cached["mask"]).astype(bool)
                matches.append(match)
                candidates.remove(candidate)
                if len(matches) >= max_matches:
                    break
    return matches


def _load_model(checkpoint, device):
    model = NoduleClassifier().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def _gradcam(model, image, device):
    tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
    tensor = tensor.repeat(1, 3, 1, 1).to(device)
    model.zero_grad(set_to_none=True)
    model.clear_hooks()
    logit = model(tensor).squeeze(1)
    probability = torch.sigmoid(logit).item()
    prediction = int(probability >= 0.5)
    model.class_scores(logit).sum().backward()
    cam = model.get_gradcam()
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach().cpu().numpy()
    model.clear_hooks()
    return cam, probability, prediction


def _cam_metrics(cam, majority_mask, lung_mask):
    total = float(cam.sum()) + 1e-8
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    nodule_pixels = np.argwhere(majority_mask)
    center = nodule_pixels.mean(axis=0)
    peak_distance = float(np.linalg.norm(np.asarray(peak) - center))
    return {
        "nodule_energy_pct": float((cam * majority_mask).sum() / total * 100),
        "lung_energy_pct": float((cam * lung_mask).sum() / total * 100),
        "peak_inside_nodule": bool(majority_mask[peak]),
        "peak_distance_px": peak_distance,
    }


def _draw_contour(axis, mask, color="lime", linewidth=1.5):
    axis.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=linewidth)


def _save_figure(records, output_path):
    columns = 5
    fig, axes = plt.subplots(
        len(records),
        columns,
        figsize=(18, 3.7 * len(records)),
        squeeze=False,
    )
    headers = [
        "CT slice",
        "Radiologist nodule area",
        "Baseline Grad-CAM",
        "HU-supervised Grad-CAM",
        "Direct comparison",
    ]
    for column, header in enumerate(headers):
        axes[0, column].set_title(header, fontsize=11, fontweight="bold")

    for row, record in enumerate(records):
        image = record["image"]
        majority = record["majority"]
        union = record["union"]
        baseline_cam = record["baseline_cam"]
        hu_cam = record["hu_cam"]
        reference_label = "Malignant" if record["label"] else "Benign"
        baseline_label = "Malignant" if record["baseline_prediction"] else "Benign"
        hu_label = "Malignant" if record["hu_prediction"] else "Benign"

        axes[row, 0].imshow(image, cmap="gray")
        _draw_contour(axes[row, 0], majority)

        axes[row, 1].imshow(image, cmap="gray")
        axes[row, 1].imshow(union, cmap="Blues", alpha=0.28)
        axes[row, 1].imshow(majority, cmap="Greens", alpha=0.45)
        _draw_contour(axes[row, 1], majority)

        axes[row, 2].imshow(image, cmap="gray")
        axes[row, 2].imshow(baseline_cam, cmap="jet", alpha=0.48, vmin=0, vmax=1)
        _draw_contour(axes[row, 2], majority)

        axes[row, 3].imshow(image, cmap="gray")
        axes[row, 3].imshow(hu_cam, cmap="jet", alpha=0.48, vmin=0, vmax=1)
        _draw_contour(axes[row, 3], majority)

        difference = hu_cam - baseline_cam
        axes[row, 4].imshow(image, cmap="gray")
        axes[row, 4].imshow(difference, cmap="coolwarm", alpha=0.55, vmin=-1, vmax=1)
        _draw_contour(axes[row, 4], majority)

        axes[row, 0].set_ylabel(
            f"{record['patient_id']}\n"
            f"Radiologists: {reference_label} ({record['mean_rating']:.2f})\n"
            f"Baseline: {baseline_label} ({record['baseline_probability']:.1%})\n"
            f"HU model: {hu_label} ({record['hu_probability']:.1%})",
            fontsize=9,
        )
        axes[row, 2].text(
            0.02,
            0.02,
            f"Nodule attention: {record['baseline_nodule_energy_pct']:.2f}%",
            transform=axes[row, 2].transAxes,
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
        axes[row, 3].text(
            0.02,
            0.02,
            f"Nodule attention: {record['hu_nodule_energy_pct']:.2f}%",
            transform=axes[row, 3].transAxes,
            color="white",
            fontsize=8,
            bbox={"facecolor": "black", "alpha": 0.65, "pad": 2},
        )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])

    fig.suptitle(
        "Held-out test samples: radiologist nodule contours vs model Grad-CAM\n"
        "Green outline = majority-vote nodule; blue area = any-reader nodule; "
        "red Grad-CAM = stronger attention",
        fontsize=13,
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare held-out Grad-CAMs with LIDC radiologist nodule contours."
    )
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--baseline-checkpoint", default=DEFAULT_BASELINE)
    parser.add_argument("--hu-checkpoint", default=DEFAULT_HU_MODEL)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-matches", type=int, default=8)
    args = parser.parse_args()

    for path in (args.cache_path, args.baseline_checkpoint, args.hu_checkpoint):
        if not os.path.exists(path):
            parser.error(f"Required file not found: {path}")
    if args.max_matches < 1:
        parser.error("--max-matches must be at least 1.")

    os.makedirs(args.output_dir, exist_ok=True)
    test_samples = _load_test_samples(args.cache_path)
    matches = _match_test_samples(test_samples, args.max_matches)
    if not matches:
        raise RuntimeError(
            "No cache images could be matched to raw LIDC nodules. "
            "The cache may have been built with different preprocessing rules."
        )
    print(f"Matched {len(matches)} held-out samples.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = _load_model(args.baseline_checkpoint, device)
    hu_model = _load_model(args.hu_checkpoint, device)

    records = []
    for index, match in enumerate(matches, start=1):
        print(f"Computing Grad-CAM {index}/{len(matches)}")
        baseline_cam, baseline_probability, baseline_prediction = _gradcam(
            baseline, match["image"], device
        )
        hu_cam, hu_probability, hu_prediction = _gradcam(
            hu_model, match["image"], device
        )
        baseline_metrics = _cam_metrics(
            baseline_cam, match["majority"], match["lung_mask"]
        )
        hu_metrics = _cam_metrics(hu_cam, match["majority"], match["lung_mask"])
        record = dict(match)
        record.update(
            {
                "baseline_cam": baseline_cam,
                "hu_cam": hu_cam,
                "baseline_probability": baseline_probability,
                "baseline_prediction": baseline_prediction,
                "hu_probability": hu_probability,
                "hu_prediction": hu_prediction,
                **{f"baseline_{k}": v for k, v in baseline_metrics.items()},
                **{f"hu_{k}": v for k, v in hu_metrics.items()},
            }
        )
        records.append(record)

    baseline.remove_hooks()
    hu_model.remove_hooks()

    figure_path = os.path.join(args.output_dir, "annotation_vs_gradcam.png")
    csv_path = os.path.join(args.output_dir, "annotation_vs_gradcam.csv")
    _save_figure(records, figure_path)

    fields = [
        "patient_id",
        "scan_id",
        "group_index",
        "slice_idx",
        "label",
        "mean_rating",
        "reader_count",
        "baseline_probability",
        "baseline_prediction",
        "baseline_nodule_energy_pct",
        "baseline_lung_energy_pct",
        "baseline_peak_inside_nodule",
        "baseline_peak_distance_px",
        "hu_probability",
        "hu_prediction",
        "hu_nodule_energy_pct",
        "hu_lung_energy_pct",
        "hu_peak_inside_nodule",
        "hu_peak_distance_px",
    ]
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record[field] for field in fields})

    def mean(field):
        return float(np.mean([record[field] for record in records]))

    report = [
        "=== RADIOLOGIST NODULE CONTOUR VS GRAD-CAM AUDIT ===",
        f"Held-out samples matched: {len(records)}",
        "Radiologist target: majority-vote contour on selected slice",
        "",
        f"Baseline mean Grad-CAM inside nodule: {mean('baseline_nodule_energy_pct'):.2f}%",
        f"HU model mean Grad-CAM inside nodule: {mean('hu_nodule_energy_pct'):.2f}%",
        f"Baseline mean Grad-CAM inside lung: {mean('baseline_lung_energy_pct'):.2f}%",
        f"HU model mean Grad-CAM inside lung: {mean('hu_lung_energy_pct'):.2f}%",
        f"Baseline peak inside nodule: {sum(r['baseline_peak_inside_nodule'] for r in records)}/{len(records)}",
        f"HU model peak inside nodule: {sum(r['hu_peak_inside_nodule'] for r in records)}/{len(records)}",
        "",
        "Important: this small visual audit is descriptive, not a full statistical evaluation.",
    ]
    report_path = os.path.join(args.output_dir, "annotation_vs_gradcam_summary.txt")
    with open(report_path, "w") as file:
        file.write("\n".join(report) + "\n")

    print("\n" + "\n".join(report))
    print(f"\nSaved figure: {figure_path}")
    print(f"Saved measurements: {csv_path}")
    print(f"Saved summary: {report_path}")


if __name__ == "__main__":
    main()
