"""Audit soft blob maps against LIDC contours without using contours for training."""

import argparse
import csv
import os
import pickle
from collections import defaultdict

import numpy as np
import pylidc as pl
import torch
import torch.nn.functional as F

from config import IMG_SIZE, TRAIN_CACHE_PATH
from dataset import patient_split
from model import NoduleClassifier
from visualize_nodule_gradcam import _eligible_candidates, _image_hash


def _load_cache(path):
    with open(path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{path} is not a completed cache list.")
    return samples


def _hashable_center_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        return image[1]
    return image


def _candidate_lookup_for_patient(patient_id):
    candidates = []
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).all()
    for scan in scans:
        try:
            candidates.extend(_eligible_candidates(scan))
        except Exception as error:
            print(f"  WARNING: could not reconstruct scan {scan.id}: {error}")
    by_key = defaultdict(list)
    for candidate in candidates:
        by_key[(candidate["label"], candidate["hash"])].append(candidate)
    return by_key


def _lung_mask_for_sample(sample, mask_source):
    if mask_source == "ts":
        mask = sample.get("ts_mask")
    elif mask_source == "hu":
        mask = sample.get("mask")
    else:
        method = str(sample.get("soft_blob_method", ""))
        mask = sample.get("ts_mask") if "ts_mask" in method else sample.get("mask")
        if mask is None:
            mask = sample.get("ts_mask", sample.get("mask"))
    if mask is None:
        raise ValueError(f"Sample {sample.get('patient_id', '')} has no {mask_source} lung mask.")
    return np.asarray(mask).astype(bool)


def _match_samples(samples, split, max_matches, mask_source):
    if split != "all":
        train, val, test = patient_split(samples)
        samples = {"train": train, "val": val, "test": test}[split]
    by_patient = defaultdict(list)
    for sample in samples:
        by_patient[sample["patient_id"]].append(sample)

    matches = []
    for patient_number, (patient_id, patient_samples) in enumerate(sorted(by_patient.items()), start=1):
        if max_matches and len(matches) >= max_matches:
            break
        print(
            f"Matching patient {patient_number}/{len(by_patient)}: "
            f"{patient_id} ({len(matches)} matches)"
        )
        lookup = _candidate_lookup_for_patient(patient_id)
        for sample in patient_samples:
            if "soft_blob_map" not in sample:
                continue
            key = (int(sample["label"]), _image_hash(_hashable_center_image(sample["image"])))
            eligible = lookup.get(key, [])
            if not eligible:
                continue
            candidate = eligible.pop(0)
            match = dict(candidate)
            match["cache_sample"] = sample
            match["lung_mask"] = _lung_mask_for_sample(sample, mask_source)
            matches.append(match)
            if max_matches and len(matches) >= max_matches:
                break
    return matches


def _center_disk(mask, radius_px):
    coords = np.argwhere(mask)
    center = coords.mean(axis=0)
    rr, cc = np.ogrid[:mask.shape[0], :mask.shape[1]]
    disk = (rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= float(radius_px) ** 2
    return disk, center


def _annotation_type(match, blob, vesselness):
    majority = match["majority"]
    lung = match["lung_mask"]
    contour = np.argwhere(majority)
    touches_lung_edge = False
    if contour.size and np.any(lung):
        from scipy.ndimage import binary_erosion
        inner = binary_erosion(lung, iterations=4, border_value=0)
        touches_lung_edge = bool(np.any(majority & lung & ~inner))
    raw_mean = float(np.mean(match["image"][majority])) if np.any(majority) else float("nan")
    approx_hu = raw_mean * 1400.0 - 1000.0
    texture = float(match.get("mean_texture", float("nan")))
    margin = float(match.get("mean_margin", float("nan")))
    spiculation = float(match.get("mean_spiculation", float("nan")))
    tags = []
    if np.isfinite(texture):
        tags.append("subsolid" if texture <= 3.0 else "solid")
    else:
        tags.append("subsolid" if approx_hu < -300.0 else "solid")
    if touches_lung_edge:
        tags.append("pleural")
    if np.isfinite(margin) and margin <= 3.0:
        tags.append("irregular")
    if np.isfinite(spiculation) and spiculation >= 3.0:
        tags.append("spiculated")
    if np.any(majority) and float(vesselness[majority].mean()) >= 0.25:
        tags.append("vessel_attached")
    if "irregular" not in tags and np.any(majority) and float(blob[majority].std()) >= 0.20:
        tags.append("irregular")
    return ";".join(tags)


def _map_metrics(match, top_percentile, center_radius_px):
    sample = match["cache_sample"]
    blob = np.asarray(sample["soft_blob_map"], dtype=np.float32)
    solid = np.asarray(sample.get("solid_like_map", np.zeros_like(blob)), dtype=np.float32)
    subsolid = np.asarray(sample.get("subsolid_like_map", np.zeros_like(blob)), dtype=np.float32)
    vesselness = np.asarray(sample.get("vesselness_map", np.zeros_like(blob)), dtype=np.float32)
    majority = np.asarray(match["majority"]).astype(bool)
    lung = np.asarray(match["lung_mask"]).astype(bool)
    if not np.any(majority):
        raise ValueError("Matched sample has empty majority contour.")

    lung_values = blob[lung] if np.any(lung) else blob.ravel()
    high_threshold = float(np.percentile(lung_values, top_percentile)) if lung_values.size else 1.0
    nonzero = blob > 1e-4
    high = blob >= high_threshold
    center_region, center = _center_disk(majority, center_radius_px)
    outside_contour = lung & ~majority
    return {
        "patient_id": match["patient_id"],
        "scan_id": match["scan_id"],
        "group_index": match["group_index"],
        "slice_idx": match["slice_idx"],
        "label": match["label"],
        "mean_rating": match["mean_rating"],
        "mean_texture": match.get("mean_texture", float("nan")),
        "mean_margin": match.get("mean_margin", float("nan")),
        "mean_spiculation": match.get("mean_spiculation", float("nan")),
        "reader_count": match["reader_count"],
        "nodule_type_tags": _annotation_type(match, blob, vesselness),
        "contour_area_px": int(majority.sum()),
        "blob_recall_nonzero_pct": float((nonzero & majority).sum() / majority.sum() * 100.0),
        "blob_recall_top_pct": float((high & majority).sum() / majority.sum() * 100.0),
        "center_inside_high_response": bool(np.any(center_region & high)),
        "center_mean_blob": float(blob[center_region].mean()),
        "mean_blob_inside_contour": float(blob[majority].mean()),
        "mean_blob_outside_contour": float(blob[outside_contour].mean()) if np.any(outside_contour) else float("nan"),
        "mean_solid_inside_contour": float(solid[majority].mean()),
        "mean_subsolid_inside_contour": float(subsolid[majority].mean()),
        "mean_vesselness_inside_contour": float(vesselness[majority].mean()),
        "high_response_threshold": high_threshold,
        "center_row": float(center[0]),
        "center_col": float(center[1]),
    }


def _load_model(checkpoint, device):
    model = NoduleClassifier().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def _image_tensor(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        return torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
    if image.ndim == 3 and image.shape[0] == 3:
        return torch.from_numpy(image)
    raise ValueError(f"Unsupported image shape {image.shape}")


def _gradcam_metrics(model, match, device):
    sample = match["cache_sample"]
    image = _image_tensor(sample["image"]).unsqueeze(0).to(device)
    blob = torch.from_numpy(np.asarray(sample["soft_blob_map"], dtype=np.float32)).to(device)
    lung = torch.from_numpy(np.asarray(match["lung_mask"], dtype=np.float32)).to(device)
    contour = torch.from_numpy(np.asarray(match["majority"], dtype=np.float32)).to(device)

    model.zero_grad(set_to_none=True)
    model.clear_hooks()
    logit = model(image).squeeze(1)
    probability = torch.sigmoid(logit).item()
    model.class_scores(logit).sum().backward()
    raw_cam = model.get_gradcam(normalise=False)
    cam = model.normalise_gradcam(raw_cam)
    cam = F.interpolate(
        cam.unsqueeze(1),
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
    ).squeeze().detach()
    cam_mass = cam / (cam.sum() + 1e-8)
    model.clear_hooks()
    return {
        "probability": probability,
        "prediction": int(probability >= 0.5),
        "cam_mass_inside_lung_pct": float((cam_mass * lung).sum().cpu() * 100.0),
        "cam_mass_inside_candidate_weighted": float((cam_mass * blob).sum().cpu()),
        "cam_mass_inside_high_candidate_pct": float((cam_mass * (blob >= 0.50).float()).sum().cpu() * 100.0),
        "cam_mass_inside_true_contour_pct": float((cam_mass * contour).sum().cpu() * 100.0),
    }


def _write_summary(rows, path):
    def mean(field):
        values = [row[field] for row in rows if isinstance(row.get(field), (int, float, np.floating))]
        values = [value for value in values if np.isfinite(value)]
        return float(np.mean(values)) if values else float("nan")

    tag_counts = defaultdict(int)
    tag_recall = defaultdict(list)
    for row in rows:
        for tag in str(row["nodule_type_tags"]).split(";"):
            tag_counts[tag] += 1
            tag_recall[tag].append(row["blob_recall_top_pct"])

    lines = [
        "=== SOFT BLOB MAP VS LIDC CONTOUR AUDIT ===",
        "LIDC contours are audit-only and were not used to generate maps or train the loss.",
        f"Matched samples: {len(rows)}",
        f"Mean contour recall touched by non-zero response: {mean('blob_recall_nonzero_pct'):.2f}%",
        f"Mean contour recall touched by top-percentile response: {mean('blob_recall_top_pct'):.2f}%",
        f"Centre inclusion in high response: {sum(row['center_inside_high_response'] for row in rows)}/{len(rows)}",
        f"Mean blob value inside contour: {mean('mean_blob_inside_contour'):.4f}",
        f"Mean blob value outside contour: {mean('mean_blob_outside_contour'):.4f}",
    ]
    if "cam_mass_inside_lung_pct" in rows[0]:
        lines.extend(
            [
                "",
                "Grad-CAM audit:",
                f"Mean CAM mass inside lung: {mean('cam_mass_inside_lung_pct'):.2f}%",
                f"Mean CAM weighted soft candidate overlap: {mean('cam_mass_inside_candidate_weighted'):.4f}",
                f"Mean CAM mass inside true contour: {mean('cam_mass_inside_true_contour_pct'):.2f}%",
            ]
        )
    lines.append("")
    lines.append("Failure-case grouping by approximate type tag:")
    for tag in sorted(tag_counts):
        lines.append(
            f"{tag}: n={tag_counts[tag]}, mean top-response contour recall="
            f"{float(np.mean(tag_recall[tag])):.2f}%"
        )
    with open(path, "w") as file:
        file.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default=TRAIN_CACHE_PATH)
    parser.add_argument("--out-dir", default="results/soft_blob_audit")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--mask-source", choices=["auto", "hu", "ts"], default="auto")
    parser.add_argument("--max-matches", type=int, default=0)
    parser.add_argument("--top-percentile", type=float, default=90.0)
    parser.add_argument("--center-radius-px", type=float, default=4.0)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    if not (0.0 < args.top_percentile < 100.0):
        parser.error("--top-percentile must be between 0 and 100")
    if args.center_radius_px <= 0:
        parser.error("--center-radius-px must be positive")
    if args.checkpoint and not os.path.exists(args.checkpoint):
        parser.error(f"Checkpoint not found: {args.checkpoint}")

    samples = _load_cache(args.cache_path)
    matches = _match_samples(samples, args.split, args.max_matches, args.mask_source)
    if not matches:
        raise RuntimeError("No soft-blob cache samples could be matched to LIDC contours.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, device) if args.checkpoint else None
    rows = []
    for index, match in enumerate(matches, start=1):
        print(f"Auditing {index}/{len(matches)}")
        row = _map_metrics(match, args.top_percentile, args.center_radius_px)
        if model is not None:
            row.update(_gradcam_metrics(model, match, device))
        rows.append(row)
    if model is not None:
        model.remove_hooks()

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "soft_blob_lidc_audit.csv")
    fields = list(rows[0].keys())
    with open(csv_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary_path = os.path.join(args.out_dir, "soft_blob_lidc_audit_summary.txt")
    _write_summary(rows, summary_path)
    print(f"\nSaved audit CSV: {csv_path}")
    print(f"Saved audit summary: {summary_path}")


if __name__ == "__main__":
    main()
