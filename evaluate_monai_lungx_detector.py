"""Evaluate MONAI lung nodule detector predictions on LUNGx centre points.

LUNGx has centre annotations, not contours. This script computes detector hit
rate by distance from each annotated centre to the nearest MONAI predicted box
centre, and creates visual examples on the annotated DICOM slice.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydicom

from config import HU_MAX, HU_MIN, IMG_SIZE
from evaluate_lungx import _case_dirs, _dicom_index, _load_hu_slice, _load_lungx_rows


def _scan_id_from_image(value):
    stem = Path(str(value)).name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = Path(stem).stem
    return stem.upper()


def _load_predictions(path, score_threshold, topk):
    with open(path, encoding="utf-8") as file:
        raw = json.load(file)
    by_scan = {}
    for item in raw:
        scan_id = _scan_id_from_image(item.get("image", ""))
        boxes = np.asarray(item.get("box", []), dtype=np.float32)
        scores = np.asarray(item.get("label_scores", []), dtype=np.float32)
        if boxes.size == 0:
            by_scan[scan_id] = []
            continue
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)
        keep = scores >= float(score_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        order = np.argsort(-scores)
        if topk > 0:
            order = order[: int(topk)]
        by_scan[scan_id] = [
            {"box": boxes[idx].tolist(), "score": float(scores[idx])}
            for idx in order
        ]
    return by_scan


def _dicom_meta(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    ipp = np.asarray(getattr(ds, "ImagePositionPatient"), dtype=np.float64)
    iop = np.asarray(getattr(ds, "ImageOrientationPatient"), dtype=np.float64)
    row_cosine = iop[:3]
    col_cosine = iop[3:]
    normal = np.cross(row_cosine, col_cosine)
    spacing = np.asarray(getattr(ds, "PixelSpacing", [1.0, 1.0]), dtype=np.float64)
    return {
        "ipp": ipp,
        "row_cosine": row_cosine,
        "col_cosine": col_cosine,
        "normal": normal,
        "row_spacing": float(spacing[0]),
        "col_spacing": float(spacing[1]),
    }


def _dicom_center_lps(record, dicom_path):
    meta = _dicom_meta(dicom_path)
    col = float(record["center_x_1based"]) - 1.0
    row = float(record["center_y_1based"]) - 1.0
    return (
        meta["ipp"]
        + row * meta["row_spacing"] * meta["row_cosine"]
        + col * meta["col_spacing"] * meta["col_cosine"]
    )


def _lps_to_ras(point_lps):
    point_lps = np.asarray(point_lps, dtype=np.float64)
    return np.asarray([-point_lps[0], -point_lps[1], point_lps[2]], dtype=np.float64)


def _ras_to_lps(point_ras):
    point_ras = np.asarray(point_ras, dtype=np.float64)
    return np.asarray([-point_ras[0], -point_ras[1], point_ras[2]], dtype=np.float64)


def _box_center_ras(box, coordinate_mode="monai_lungx"):
    box = np.asarray(box, dtype=np.float64)
    # MONAI bundle postprocessing converts boxes to cccwhd after world transform.
    # If an xyzxyz box appears, use its midpoint as a fallback.
    if box.size >= 6:
        if np.all(box[3:6] > 0) and np.any(box[3:6] < 80):
            center = box[:3]
        else:
            center = (box[:3] + box[3:6]) / 2.0
        if coordinate_mode == "ras":
            return center
        if coordinate_mode == "monai_lungx":
            # The MONAI detector JSON for dcm2niix-exported LUNGx volumes is
            # in a world-axis convention that maps to DICOM RAS by swapping the
            # first two axes and flipping both signs.
            return np.asarray([-center[1], -center[0], center[2]], dtype=np.float64)
        raise ValueError(f"Unknown coordinate mode: {coordinate_mode}")
    raise ValueError(f"Unexpected box shape: {box}")


def _project_lps_to_slice(point_lps, dicom_path):
    meta = _dicom_meta(dicom_path)
    delta = np.asarray(point_lps, dtype=np.float64) - meta["ipp"]
    row = float(np.dot(delta, meta["row_cosine"]) / meta["row_spacing"])
    col = float(np.dot(delta, meta["col_cosine"]) / meta["col_spacing"])
    normal_mm = float(np.dot(delta, meta["normal"]))
    return row, col, normal_mm


def _normalise_hu(raw_slice):
    return np.clip((np.clip(raw_slice.astype(np.float32), HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)


def _resize_point_to_224(row, col, raw_shape):
    return (
        float(row) * IMG_SIZE / float(raw_shape[0]),
        float(col) * IMG_SIZE / float(raw_shape[1]),
    )


def _draw_examples(rows, out_path, count):
    rows = sorted(rows, key=lambda row: (not row["hit_15mm"], row["nearest_distance_mm"]))
    selected = rows[: min(count, len(rows))]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 2, figsize=(8, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)
    for row_idx, row in enumerate(selected):
        raw_slice = _load_hu_slice(row["dicom_path"])
        image = _normalise_hu(raw_slice)
        ax0, ax1 = axes[row_idx]
        ax0.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax1.imshow(image, cmap="gray", vmin=0, vmax=1)

        actual_row = float(row["center_y_1based"]) - 1.0
        actual_col = float(row["center_x_1based"]) - 1.0
        ax0.scatter([actual_col], [actual_row], c="yellow", marker="+", s=70, linewidths=2)
        ax1.scatter([actual_col], [actual_row], c="yellow", marker="+", s=70, linewidths=2, label="actual")

        height, width = image.shape
        visible_predictions = [
            det
            for det in row["slice_predictions"]
            if 0 <= det["row"] < height and 0 <= det["col"] < width
        ]
        nearest_visible = min(visible_predictions, key=lambda det: det["distance_mm"], default=None)
        for det in visible_predictions:
            is_nearest = det is nearest_visible
            color = "cyan" if is_nearest else "red"
            size = 72 if is_nearest else 38
            ax1.scatter(
                [det["col"]],
                [det["row"]],
                edgecolors=color,
                facecolors="none",
                marker="o",
                s=size,
                linewidths=1.8 if is_nearest else 1.2,
            )
            ax1.text(
                det["col"] + 3,
                det["row"] + 3,
                f"{det['score']:.2f}",
                color=color,
                fontsize=7,
                bbox=dict(facecolor="black", alpha=0.35, pad=1) if is_nearest else None,
            )

        status = "HIT" if row["hit_15mm"] else ("NEAR" if row["near_30mm"] else "MISS")
        title = (
            f"{row['scan_id']} n{row['nodule_number']} {status} "
            f"d={row['nearest_distance_mm']:.1f}mm score={row['nearest_score']:.3f}"
        )
        ax0.set_title("Actual LUNGx centre", fontsize=9)
        ax1.set_title(title, fontsize=9)
        for ax in (ax0, ax1):
            ax.axis("off")
    fig.suptitle(
        "MONAI detector predictions vs LUNGx centre annotations\n"
        "Yellow = actual centre; cyan = nearest visible prediction; red = other visible predictions",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys() if key != "slice_predictions"})
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-json", default=r"C:\repo\lung_nodule_ct_detection\eval\result_luna16_fold0.json")
    parser.add_argument("--manifest-root", default=r"C:\repo\manifest-cgqtDj7Y2699835271585651107")
    parser.add_argument("--xlsx", default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    parser.add_argument("--out-dir", default=r"results\monai_lungx_detector_eval")
    parser.add_argument("--score-threshold", type=float, default=0.02)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--hit-radius-mm", type=float, default=15.0)
    parser.add_argument("--near-radius-mm", type=float, default=30.0)
    parser.add_argument("--slice-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--box-coordinate-mode", choices=["monai_lungx", "ras"], default="monai_lungx")
    parser.add_argument(
        "--only-predicted-scans",
        action="store_true",
        help="Evaluate only scans present in the prediction JSON; useful for smoke tests.",
    )
    args = parser.parse_args()

    if not Path(args.pred_json).exists():
        raise FileNotFoundError(f"Prediction JSON not found: {args.pred_json}")

    predictions = _load_predictions(args.pred_json, args.score_threshold, args.topk)
    image_root = os.path.join(args.manifest_root, "SPIE-AAPM Lung CT Challenge")
    cases = _case_dirs(image_root)
    records = _load_lungx_rows(args.xlsx)
    dicom_cache = {}
    rows = []

    for record in records:
        scan_id = record["scan_id"].upper()
        if args.only_predicted_scans and scan_id not in predictions:
            continue
        case_dir = cases.get(scan_id)
        if case_dir is None:
            continue
        if scan_id not in dicom_cache:
            dicom_cache[scan_id] = _dicom_index(case_dir)
        entries, by_instance = dicom_cache[scan_id]
        dicom_path = by_instance.get(record["center_image"])
        if dicom_path is None:
            dicom_path = entries[max(0, min(record["center_image"] - 1, len(entries) - 1))][2]
        actual_ras = _lps_to_ras(_dicom_center_lps(record, dicom_path))

        detections = predictions.get(scan_id, [])
        distances = []
        slice_predictions = []
        for det in detections:
            center_ras = _box_center_ras(det["box"], args.box_coordinate_mode)
            distance = float(np.linalg.norm(center_ras - actual_ras))
            distances.append((distance, det, center_ras))
            row, col, normal_mm = _project_lps_to_slice(_ras_to_lps(center_ras), dicom_path)
            if abs(normal_mm) <= args.slice_tolerance_mm:
                slice_predictions.append(
                    {
                        "row": row,
                        "col": col,
                        "normal_mm": normal_mm,
                        "score": float(det["score"]),
                        "distance_mm": distance,
                    }
                )

        if distances:
            nearest_distance, nearest_det, _nearest_ras = min(distances, key=lambda item: item[0])
            nearest_score = float(nearest_det["score"])
        else:
            nearest_distance = float("inf")
            nearest_score = float("nan")

        rows.append(
            {
                "scan_id": scan_id,
                "nodule_number": record["nodule_number"],
                "label": record["label"],
                "diagnosis": record["diagnosis"],
                "center_image": record["center_image"],
                "center_x_1based": record["center_x_1based"],
                "center_y_1based": record["center_y_1based"],
                "num_predictions_kept": len(detections),
                "num_predictions_on_slice": len(slice_predictions),
                "nearest_distance_mm": nearest_distance,
                "nearest_score": nearest_score,
                "hit_15mm": int(nearest_distance <= args.hit_radius_mm),
                "near_30mm": int(nearest_distance <= args.near_radius_mm),
                "dicom_path": dicom_path,
                "slice_predictions": slice_predictions,
            }
        )

    total = len(rows)
    hits = sum(row["hit_15mm"] for row in rows)
    near = sum((not row["hit_15mm"]) and row["near_30mm"] for row in rows)
    miss = total - hits - near
    finite_distances = [row["nearest_distance_mm"] for row in rows if np.isfinite(row["nearest_distance_mm"])]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "monai_lungx_detector_eval.csv", rows)
    _draw_examples(rows, out_dir / "monai_lungx_detector_examples.png", args.examples)

    lines = [
        "=== MONAI LUNGx DETECTOR CENTRE-HIT EVALUATION ===",
        "LUNGx has centre points, not contours; this is detector-centre distance evaluation.",
        f"Prediction JSON: {args.pred_json}",
        f"Score threshold: {args.score_threshold}",
        f"Top-K predictions per scan: {args.topk}",
        f"Total LUNGx nodules evaluated: {total}",
        f"Hit within {args.hit_radius_mm:g}mm: {hits}/{total} ({hits / max(total, 1) * 100:.2f}%)",
        f"Near miss {args.hit_radius_mm:g}-{args.near_radius_mm:g}mm: {near}/{total} ({near / max(total, 1) * 100:.2f}%)",
        f"Miss beyond {args.near_radius_mm:g}mm: {miss}/{total} ({miss / max(total, 1) * 100:.2f}%)",
        f"Mean nearest distance: {float(np.mean(finite_distances)):.2f}mm" if finite_distances else "Mean nearest distance: nan",
        f"Median nearest distance: {float(np.median(finite_distances)):.2f}mm" if finite_distances else "Median nearest distance: nan",
        "",
        f"CSV: {out_dir / 'monai_lungx_detector_eval.csv'}",
        f"Examples PNG: {out_dir / 'monai_lungx_detector_examples.png'}",
    ]
    (out_dir / "monai_lungx_detector_eval_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
