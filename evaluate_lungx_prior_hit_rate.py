"""Evaluate possible-nodule prior hit rate on LUNGx centre annotations.

LUNGx provides nodule centre points, not contours. This script therefore uses
an approximate circular ROI around the provided centre. It is evaluation-only:
no training is performed.
"""

import argparse
import csv
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation

from build_cache import _get_lung_mask
from config import HU_MAX, HU_MIN, IMG_SIZE
from evaluate_lungx import _case_dirs, _dicom_index, _load_hu_slice, _load_lungx_rows
from preview_possible_nodule_mask import _make_prior, _possible_nodule_mask
from train_tristream import _resize_224


def _normalise_hu(raw_slice):
    image = np.clip(raw_slice.astype(np.float32), HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def _circle(shape, cy, cx, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius ** 2


def _candidate_args(args):
    return SimpleNamespace(
        no_boundary_band=True,
        max_ts_hole_area=100,
        hu_threshold=-600.0,
        prior_dilation_iterations=0,
        blur_sigma=3.0,
        min_area=3,
        max_area=350,
        max_elongation=4.0,
        closing_radius=1,
        neighbourhood_radius=4,
        inner_erosion_radius=1,
        boundary_radius=3,
        max_internal_area=160,
        max_internal_elongation=3.0,
        restrict_to_ts_or_holes=False,
        candidate_lung_erosion_radius=0,
        exclude_ts_edge_radius=0,
        dual_hu_dense=False,
        ground_glass_min_hu=-750.0,
        ground_glass_max_hu=-300.0,
        solid_min_hu=-300.0,
        solid_max_hu=100.0,
        prior_mode="standard",
        component_area_scale=50.0,
        solid_bonus_scale=0.0,
        added_ts_border_downweight=1.0,
        pleural_dense_blob=False,
        pleural_edge_radius=5,
        pleural_inner_erosion_radius=1,
        pleural_min_area=5,
        pleural_max_area=220,
        pleural_max_elongation=2.5,
        pleural_min_mean_hu=-250.0,
        pleural_min_contrast_hu=80.0,
        pleural_contrast_ring_radius=5,
        pleural_keep_large_rim=False,
        pleural_core_hu_threshold=0.0,
        pleural_core_dilation=1,
        pleural_preserve_after_filter=False,
    )


def _sample_from_lungx_slice(raw_slice):
    image = _resize_224(_normalise_hu(raw_slice), order=1).astype(np.float32)
    lung_native = _get_lung_mask(raw_slice).astype(bool)
    lung = (_resize_224(lung_native, order=0) > 0.5).astype(bool)
    # LUNGx has no TS cache here. Use the same annotation-free HU lung mask as
    # both HU and TS support so the candidate logic can run without contours.
    return {"image": image, "mask": lung, "ts_mask": lung}


def _load_lungx_cache(path):
    if not path:
        return None
    with open(path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{path} is not a completed cache list.")
    lookup = {}
    for sample in samples:
        key = (str(sample["scan_id"]).upper(), int(sample["nodule_number"]))
        lookup[key] = sample
    return lookup


def _centre_to_224(record, raw_slice):
    scale_x = IMG_SIZE / float(raw_slice.shape[1])
    scale_y = IMG_SIZE / float(raw_slice.shape[0])
    cx = (float(record["center_x_1based"]) - 1.0) * scale_x
    cy = (float(record["center_y_1based"]) - 1.0) * scale_y
    return cy, cx


def _overlay_circle(ax, cy, cx, radius, color="yellow", linewidth=1.2):
    circle = plt.Circle((cx, cy), radius, fill=False, color=color, linewidth=linewidth)
    ax.add_patch(circle)
    ax.plot([cx], [cy], marker="+", color=color, markersize=7, markeredgewidth=1.4)


def _write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_grid(records, out_path, seed, count):
    rng = np.random.default_rng(seed)
    selected = list(rng.choice(records, size=min(count, len(records)), replace=False))
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, 3 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, record in enumerate(selected):
        image = record["image"]
        candidate = record["candidate_dil1"]
        prior = record["prior"]
        cy = record["center_y_224"]
        cx = record["center_x_224"]

        panels = [
            ("CT + centre ROI", image, "gray"),
            ("Binary candidate + ROI", candidate.astype(float), "gray"),
            ("Soft prior + ROI", prior, "magma"),
        ]
        for col_idx, (title, data, cmap) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
            _overlay_circle(ax, cy, cx, 15)
            ax.set_xticks([])
            ax.set_yticks([])
            if row_idx == 0:
                ax.set_title(title, fontsize=10)
            if col_idx == 0:
                status = "hit" if record["hit_15px"] else ("near" if record["near_miss_15_30px"] else "miss")
                ax.set_ylabel(
                    f"{record['scan_id']} n{record['nodule_number']}\n{status}, area {record['candidate_area_pct']:.1f}%",
                    fontsize=8,
                )
    fig.suptitle(
        "LUNGx possible-nodule prior audit. Yellow circle = approximate 15px centre ROI, not contour.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", default=r"C:\repo\manifest-cgqtDj7Y2699835271585651107")
    parser.add_argument("--xlsx", default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    parser.add_argument("--out-dir", default=r"results\prior_lungx_hit_rate")
    parser.add_argument(
        "--lungx-cache",
        default="",
        help="Optional LUNGx cache with real TS masks from build_lungx_ts_cache.py.",
    )
    parser.add_argument("--roi-radius", type=float, default=15.0)
    parser.add_argument("--near-radius", type=float, default=30.0)
    parser.add_argument("--candidate-dilation", type=int, default=1)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = os.path.join(args.manifest_root, "SPIE-AAPM Lung CT Challenge")
    cases = _case_dirs(image_root)
    records = _load_lungx_rows(args.xlsx)
    prior_args = _candidate_args(args)
    dicom_cache = {}
    lungx_cache = _load_lungx_cache(args.lungx_cache)
    rows = []
    visual_records = []

    for record in records:
        case_dir = cases.get(record["scan_id"])
        if case_dir is None:
            raise FileNotFoundError(f"Missing DICOM directory for {record['scan_id']}")
        if record["scan_id"] not in dicom_cache:
            dicom_cache[record["scan_id"]] = _dicom_index(case_dir)
        entries, by_instance = dicom_cache[record["scan_id"]]
        dicom_path = by_instance.get(record["center_image"])
        if dicom_path is None:
            dicom_path = entries[max(0, min(record["center_image"] - 1, len(entries) - 1))][2]

        raw_slice = _load_hu_slice(dicom_path)
        cached = None
        if lungx_cache is not None:
            cached = lungx_cache.get((record["scan_id"].upper(), int(record["nodule_number"])))
            if cached is None:
                raise KeyError(
                    f"{record['scan_id']} nodule {record['nodule_number']} missing from {args.lungx_cache}"
                )
        if cached is not None:
            sample = {
                "image": np.asarray(cached["image"], dtype=np.float32),
                "mask": np.asarray(cached["mask"]).astype(bool),
                "ts_mask": np.asarray(cached["ts_mask"]).astype(bool),
            }
            cx = float(cached.get("center_x_224"))
            cy = float(cached.get("center_y_224"))
        else:
            sample = _sample_from_lungx_slice(raw_slice)
            cy, cx = _centre_to_224(record, raw_slice)
        candidate, parts = _possible_nodule_mask(sample, prior_args)
        prior = _make_prior(candidate, parts, prior_args)
        candidate_dil = binary_dilation(candidate, iterations=int(args.candidate_dilation))
        roi15 = _circle(candidate.shape, cy, cx, args.roi_radius)
        roi30 = _circle(candidate.shape, cy, cx, args.near_radius)
        hit = bool(np.any(candidate_dil & roi15))
        within30 = bool(np.any(candidate_dil & roi30))
        near_miss = bool((not hit) and within30)
        miss = bool(not within30)
        row = {
            "scan_id": record["scan_id"],
            "nodule_number": record["nodule_number"],
            "center_image": record["center_image"],
            "center_x_1based": record["center_x_1based"],
            "center_y_1based": record["center_y_1based"],
            "center_x_224": cx,
            "center_y_224": cy,
            "diagnosis": record["diagnosis"],
            "label": record["label"],
            "hit_15px": int(hit),
            "near_miss_15_30px": int(near_miss),
            "miss_beyond_30px": int(miss),
            "candidate_area_pct": float(candidate_dil.mean() * 100.0),
            "soft_prior_mean": float(prior.mean()),
            "dicom_path": dicom_path,
        }
        rows.append(row)
        visual_records.append(
            {
                **row,
                "image": sample["image"],
                "candidate_dil1": candidate_dil,
                "prior": prior,
            }
        )

    total = len(rows)
    hits = sum(row["hit_15px"] for row in rows)
    near = sum(row["near_miss_15_30px"] for row in rows)
    miss = sum(row["miss_beyond_30px"] for row in rows)
    mean_area = float(np.mean([row["candidate_area_pct"] for row in rows]))
    median_area = float(np.median([row["candidate_area_pct"] for row in rows]))

    _write_csv(out_dir / "prior_lungx_evaluation.csv", rows)
    _make_grid(visual_records, out_dir / "prior_lungx_examples.png", args.seed, args.examples)

    lung_support_note = (
        f"Using real LUNGx TS masks from cache: {args.lungx_cache}"
        if args.lungx_cache
        else (
            "Because this repo has no TotalSegmentator masks for LUNGx in this run, "
            "the lung support here uses the same annotation-free HU fallback mask used in prior LUNGx overlay evaluations."
        )
    )

    lines = [
        "=== LUNGx POSSIBLE NODULE PRIOR HIT RATE ===",
        "Evaluation only; no training was run.",
        "Important: LUNGx provides centre points, not nodule contours.",
        "Hit metrics use an approximate 15px radius ROI around the centre point.",
        "Near miss means candidate exists within 30px of centre but not inside 15px ROI.",
        lung_support_note,
        "",
        f"Manifest root: {args.manifest_root}",
        f"Annotation sheet: {args.xlsx}",
        f"LUNGx TS cache: {args.lungx_cache or 'none; HU fallback lung support used'}",
        f"Total evaluated: {total}",
        "",
        "Prior settings matched cache_ts_possible_nodule_prior_best:",
        "  no boundary band: True",
        "  max TS hole area: 100",
        "  HU threshold: -600",
        "  prior dilation before blur: 0",
        "  blur sigma: 3",
        "  min area: 3, max area: 350, max elongation: 4",
        "  closing radius: 1",
        "  neighbourhood radius: 4",
        "  binary candidate dilation for hit metric: 1",
        "",
        f"Hit within 15px: {hits}/{total} ({hits / total * 100.0:.2f}%)",
        f"Near miss 15-30px: {near}/{total} ({near / total * 100.0:.2f}%)",
        f"Miss beyond 30px: {miss}/{total} ({miss / total * 100.0:.2f}%)",
        f"Mean candidate area: {mean_area:.2f}%",
        f"Median candidate area: {median_area:.2f}%",
        "",
        "LIDC reference for cache_ts_possible_nodule_prior_best with binary dilation=1:",
        "  Hit rate: 1502/1559 (96.34%)",
        "  100% contour covered: 1270/1559 (81.46%)",
        "  Mean contour coverage: 0.9314",
        "  Mean candidate area: 7.92%",
        "  Note: LIDC stats are contour-based; LUNGx stats are approximate centre-circle estimates.",
        "",
        f"Saved CSV: {out_dir / 'prior_lungx_evaluation.csv'}",
        f"Saved examples: {out_dir / 'prior_lungx_examples.png'}",
    ]
    report_path = out_dir / "prior_lungx_evaluation.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    with open("PROGRESS.md", "a", encoding="utf-8") as file:
        file.write(
            "\n\n## LUNGx Possible Nodule Prior Hit Rate\n"
            f"- Output: `{out_dir}`\n"
            f"- Hit within 15px: {hits}/{total} ({hits / total * 100.0:.2f}%)\n"
            f"- Near miss 15-30px: {near}/{total} ({near / total * 100.0:.2f}%)\n"
            f"- Miss beyond 30px: {miss}/{total} ({miss / total * 100.0:.2f}%)\n"
            f"- LUNGx uses approximate centre-circle ROIs, not contours; {lung_support_note}\n"
        )


if __name__ == "__main__":
    main()
