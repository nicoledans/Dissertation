"""Audit MONAI detector box coordinates against LUNGx DICOM annotations.

This script derives the DICOM voxel-to-LPS affine from headers, inspects the
MONAI preprocessing/postprocessing convention, and compares candidate
interpretations of the saved MONAI JSON boxes. It is intentionally an audit:
the header-derived interpretation is reported separately from the older
empirical swap/flip interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom

from config import HU_MAX, HU_MIN
from evaluate_lungx import _load_lungx_rows


@dataclass
class DicomGeometry:
    affine_lps: np.ndarray
    row_cosine: np.ndarray
    col_cosine: np.ndarray
    normal: np.ndarray
    row_spacing: float
    col_spacing: float
    slice_spacing: float
    first_ipp: np.ndarray
    last_ipp: np.ndarray
    iop: np.ndarray


def _case_dirs(manifest_root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in manifest_root.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.upper()
        if name.startswith("LUNGX-CT") or name.startswith("CT-TRAINING-"):
            if any(path.rglob("*.dcm")):
                out[name] = path
    return out


def _dicom_series(case_dir: Path) -> list[tuple[int, Path, pydicom.dataset.FileDataset]]:
    series = []
    for path in case_dir.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            instance = int(getattr(ds, "InstanceNumber"))
        except Exception:
            continue
        series.append((instance, path, ds))
    series.sort(key=lambda item: item[0])
    if not series:
        raise FileNotFoundError(f"No DICOM slices found under {case_dir}")
    return series


def _derive_dicom_affine_lps(series: list[tuple[int, Path, pydicom.dataset.FileDataset]]) -> DicomGeometry:
    first = series[0][2]
    last = series[-1][2]
    iop = np.asarray(first.ImageOrientationPatient, dtype=np.float64)
    row_cosine = iop[:3]
    col_cosine = iop[3:]
    normal_cross = np.cross(row_cosine, col_cosine)
    first_ipp = np.asarray(first.ImagePositionPatient, dtype=np.float64)
    last_ipp = np.asarray(last.ImagePositionPatient, dtype=np.float64)
    spacing = np.asarray(first.PixelSpacing, dtype=np.float64)
    row_spacing = float(spacing[0])
    col_spacing = float(spacing[1])
    if len(series) > 1:
        slice_step = (last_ipp - first_ipp) / float(len(series) - 1)
        slice_spacing = float(np.linalg.norm(slice_step))
        normal = slice_step / (slice_spacing + 1e-12)
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0))
        normal = normal_cross

    affine = np.eye(4, dtype=np.float64)
    # DICOM pixel array is indexed [row, col, slice].
    affine[:3, 0] = row_cosine * row_spacing
    affine[:3, 1] = col_cosine * col_spacing
    affine[:3, 2] = normal * slice_spacing
    affine[:3, 3] = first_ipp
    return DicomGeometry(
        affine_lps=affine,
        row_cosine=row_cosine,
        col_cosine=col_cosine,
        normal=normal,
        row_spacing=row_spacing,
        col_spacing=col_spacing,
        slice_spacing=slice_spacing,
        first_ipp=first_ipp,
        last_ipp=last_ipp,
        iop=iop,
    )


def _normalise_hu(raw: np.ndarray) -> np.ndarray:
    return np.clip((np.clip(raw.astype(np.float32), HU_MIN, HU_MAX) - HU_MIN) / (HU_MAX - HU_MIN), 0, 1)


def _load_hu_slice(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return image * slope + intercept


def _scan_id_from_image(value: str) -> str:
    name = Path(str(value)).name
    if name.endswith(".nii.gz"):
        return name[:-7].upper()
    return Path(name).stem.upper()


def _load_predictions(path: Path) -> dict[str, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for item in data:
        scan_id = _scan_id_from_image(item.get("image", ""))
        boxes = item.get("box", [])
        scores = item.get("label_scores", item.get("score", []))
        out[scan_id] = [
            {"center": np.asarray(box[:3], dtype=np.float64), "size": np.asarray(box[3:6], dtype=np.float64), "score": float(score)}
            for box, score in zip(boxes, scores)
        ]
    return out


def _lps_to_ras(point: np.ndarray) -> np.ndarray:
    return np.asarray([-point[0], -point[1], point[2]], dtype=np.float64)


def _ras_to_lps(point: np.ndarray) -> np.ndarray:
    return np.asarray([-point[0], -point[1], point[2]], dtype=np.float64)


def _annotation_lps(record: dict, affine_lps: np.ndarray) -> np.ndarray:
    row = float(record["center_y_1based"]) - 1.0
    col = float(record["center_x_1based"]) - 1.0
    slice_idx = float(record["center_image"]) - 1.0
    return (affine_lps @ np.asarray([row, col, slice_idx, 1.0], dtype=np.float64))[:3]


def _project_lps_to_dicom_row_col(point_lps: np.ndarray, ds: pydicom.dataset.FileDataset) -> tuple[float, float, float]:
    ipp = np.asarray(ds.ImagePositionPatient, dtype=np.float64)
    iop = np.asarray(ds.ImageOrientationPatient, dtype=np.float64)
    row_cosine = iop[:3]
    col_cosine = iop[3:]
    normal = np.cross(row_cosine, col_cosine)
    spacing = np.asarray(ds.PixelSpacing, dtype=np.float64)
    delta = point_lps - ipp
    row = float(np.dot(delta, row_cosine) / spacing[0])
    col = float(np.dot(delta, col_cosine) / spacing[1])
    normal_mm = float(np.dot(delta, normal))
    return row, col, normal_mm


def _box_modes() -> dict[str, callable]:
    return {
        # Header-derived if AffineBoxToWorldCoordinated was called with
        # affine_lps_to_ras=True after RAS preprocessing: saved coords are LPS.
        "header_saved_lps": lambda center: center,
        # Header-derived if saved coords are RAS.
        "header_saved_ras": lambda center: _ras_to_lps(center),
        # Old empirical mode, kept only to explain why it looked good in CT001.
        "legacy_swap_flip": lambda center: np.asarray([center[1], center[0], center[2]], dtype=np.float64),
        "legacy_swap_no_flip": lambda center: _ras_to_lps(np.asarray([center[1], center[0], center[2]], dtype=np.float64)),
    }


def _distance_summary(values: list[float]) -> str:
    finite = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return "n=0 mean=nan median=nan"
    return (
        f"n={finite.size} mean={float(np.mean(finite)):.2f} "
        f"median={float(np.median(finite)):.2f} min={float(np.min(finite)):.2f} max={float(np.max(finite)):.2f}"
    )


def _draw_overlays(rows: list[dict], out_path: Path, mode_name: str, examples: int) -> None:
    selected = rows[: max(0, examples)]
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), 2, figsize=(8, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.expand_dims(axes, axis=0)
    for idx, row in enumerate(selected):
        image = _normalise_hu(_load_hu_slice(Path(row["dicom_path"])))
        ax0, ax1 = axes[idx]
        ax0.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax1.imshow(image, cmap="gray", vmin=0, vmax=1)
        ar = float(row["center_y_1based"]) - 1.0
        ac = float(row["center_x_1based"]) - 1.0
        for ax in (ax0, ax1):
            ax.scatter([ac], [ar], marker="+", c="yellow", s=100, linewidths=2)
            ax.add_patch(plt.Circle((ac, ar), 15, edgecolor="yellow", facecolor="none", linewidth=1.2))
            ax.axis("off")
        best = row.get(f"{mode_name}_best")
        if best:
            ax1.scatter([best["col"]], [best["row"]], edgecolors="cyan", facecolors="none", s=90, linewidths=2)
            ax1.text(best["col"] + 3, best["row"] + 3, f"{best['score']:.2f}", color="cyan", fontsize=8)
        ax0.set_title(f"{row['scan_id']} annotation", fontsize=9)
        ax1.set_title(f"{mode_name}: d={row.get(mode_name + '_nearest_mm', float('nan')):.1f}mm", fontsize=9)
    fig.suptitle(f"Header-derived MONAI overlay audit: {mode_name}\nYellow=LUNGx centre; cyan=nearest predicted box centre", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-root", default=r"C:\repo\manifest-cgqtDj7Y2699835271585651107")
    parser.add_argument("--nifti-dir", default=r"C:\repo\Dissertation\monai_detector_inputs\lungx_nifti")
    parser.add_argument("--pred-json", required=True)
    parser.add_argument("--xlsx", default=r"C:\repo\Dissertation\data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    parser.add_argument("--config-json", default=r"C:\repo\lung_nodule_ct_detection\configs\inference.json")
    parser.add_argument("--out-dir", default=r"results\monai_coordinate_audit")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--slice-tolerance-mm", type=float, default=25.0)
    parser.add_argument("--score-threshold", type=float, default=0.02)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = _load_lungx_rows(Path(args.xlsx))
    cases = _case_dirs(Path(args.manifest_root))
    predictions = _load_predictions(Path(args.pred_json))
    config = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
    modes = _box_modes()

    rows: list[dict] = []
    geometry_lines: list[str] = []

    for record in records:
        scan_id = str(record["scan_id"]).upper()
        if scan_id not in predictions:
            continue
        case_dir = cases.get(scan_id)
        if case_dir is None:
            continue
        series = _dicom_series(case_dir)
        by_instance = {instance: (path, ds) for instance, path, ds in series}
        if int(record["center_image"]) not in by_instance:
            continue
        dicom_path, annot_ds = by_instance[int(record["center_image"])]
        geom = _derive_dicom_affine_lps(series)
        annot_lps = _annotation_lps(record, geom.affine_lps)
        annot_ras = _lps_to_ras(annot_lps)

        nifti_path = Path(args.nifti_dir) / f"{scan_id}.nii.gz"
        nifti_note = "missing"
        annot_voxel = [float("nan")] * 3
        if nifti_path.exists():
            nii = nib.load(str(nifti_path))
            annot_voxel = nib.affines.apply_affine(np.linalg.inv(nii.affine), annot_ras).tolist()
            nifti_note = f"shape={nii.shape} axcodes={nib.aff2axcodes(nii.affine)}"

        row_out = {
            "scan_id": scan_id,
            "nodule_number": record["nodule_number"],
            "label": record["label"],
            "center_image": record["center_image"],
            "center_x_1based": record["center_x_1based"],
            "center_y_1based": record["center_y_1based"],
            "dicom_path": str(dicom_path),
            "annotation_lps_x": annot_lps[0],
            "annotation_lps_y": annot_lps[1],
            "annotation_lps_z": annot_lps[2],
            "annotation_ras_x": annot_ras[0],
            "annotation_ras_y": annot_ras[1],
            "annotation_ras_z": annot_ras[2],
            "annotation_nifti_i": annot_voxel[0],
            "annotation_nifti_j": annot_voxel[1],
            "annotation_nifti_k": annot_voxel[2],
        }

        for mode_name, to_lps in modes.items():
            best = None
            for det in predictions[scan_id]:
                if det["score"] < args.score_threshold:
                    continue
                center_lps = to_lps(det["center"])
                distance_mm = float(np.linalg.norm(center_lps - annot_lps))
                proj_row, proj_col, normal_mm = _project_lps_to_dicom_row_col(center_lps, annot_ds)
                candidate = {
                    "distance_mm": distance_mm,
                    "row": proj_row,
                    "col": proj_col,
                    "normal_mm": normal_mm,
                    "score": det["score"],
                }
                if best is None or candidate["distance_mm"] < best["distance_mm"]:
                    best = candidate
            if best is None:
                row_out[f"{mode_name}_nearest_mm"] = float("inf")
                row_out[f"{mode_name}_nearest_score"] = float("nan")
                row_out[f"{mode_name}_on_slice"] = 0
            else:
                row_out[f"{mode_name}_nearest_mm"] = best["distance_mm"]
                row_out[f"{mode_name}_nearest_score"] = best["score"]
                row_out[f"{mode_name}_nearest_row"] = best["row"]
                row_out[f"{mode_name}_nearest_col"] = best["col"]
                row_out[f"{mode_name}_nearest_normal_mm"] = best["normal_mm"]
                row_out[f"{mode_name}_on_slice"] = int(abs(best["normal_mm"]) <= args.slice_tolerance_mm)
                row_out[f"{mode_name}_best"] = best

        rows.append(row_out)

        if len(geometry_lines) < 80:
            geometry_lines.extend(
                [
                    f"## {scan_id}",
                    f"DICOM ImageOrientationPatient: {geom.iop.tolist()}",
                    f"DICOM first IPP: {geom.first_ipp.tolist()}",
                    f"DICOM last IPP: {geom.last_ipp.tolist()}",
                    f"PixelSpacing row/col: {geom.row_spacing}, {geom.col_spacing}",
                    f"Derived slice spacing: {geom.slice_spacing:.6f}",
                    "Derived DICOM affine LPS [row, col, slice, 1] -> [L, P, S, 1]:",
                    np.array2string(geom.affine_lps, precision=6),
                    f"NIfTI: {nifti_note}",
                    f"Annotation LPS: {annot_lps.tolist()}",
                    f"Annotation RAS: {annot_ras.tolist()}",
                    f"Annotation NIfTI voxel: {annot_voxel}",
                    "",
                ]
            )

    csv_rows = [{k: v for k, v in row.items() if not k.endswith("_best")} for row in rows]
    csv_path = out_dir / "coordinate_audit.csv"
    if csv_rows:
        fieldnames = sorted({k for row in csv_rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

    lines = [
        "=== MONAI COORDINATE TRANSFORM AUDIT ===",
        "This derives DICOM LPS coordinates from headers and compares saved MONAI JSON box interpretations.",
        "",
        "MONAI config facts:",
        f"whether_raw_luna16: {config.get('whether_raw_luna16')}",
        "preprocessing LoadImaged affine_lps_to_ras: true (from config)",
        "preprocessing Orientationd axcodes: RAS",
        "postprocessing AffineBoxToWorldCoordinated affine_lps_to_ras: true",
        "postprocessing ConvertBoxModed: xyzxyz -> cccwhd",
        "",
        "Important interpretation:",
        "If image metadata is already RAS after preprocessing, AffineBoxToWorldCoordinated(... affine_lps_to_ras=True)",
        "flips the affine again via diag([-1,-1,1,1]). Therefore saved JSON centres should be treated as LPS",
        "unless the bundle internally stores a different affine than the visible MetaTensor affine.",
        "",
        f"Prediction JSON: {args.pred_json}",
        f"Cases evaluated: {len(rows)}",
        "",
        "Nearest predicted-centre distance to LUNGx annotation by coordinate mode:",
    ]
    for mode_name in modes:
        values = [row.get(f"{mode_name}_nearest_mm", float("inf")) for row in rows]
        hits15 = sum(1 for value in values if np.isfinite(value) and value <= 15)
        lines.append(f"{mode_name}: {_distance_summary(values)}; hit<=15mm {hits15}/{max(len(values),1)}")
    lines.extend(
        [
            "",
            "Header-derived preferred mode:",
            "  header_saved_lps if using the current config with postprocess affine_lps_to_ras=true.",
            "  header_saved_ras if postprocess affine_lps_to_ras is changed to false.",
            "",
            "Legacy mode note:",
            "  legacy_swap_flip is empirical and should not be used as the final transform unless validated against",
            "  the bundle's internal image-coordinate box order across multiple cases.",
            "",
            f"CSV: {csv_path}",
            "",
            "=== DICOM/NIfTI GEOMETRY EXAMPLES ===",
            *geometry_lines,
        ]
    )
    (out_dir / "coordinate_audit_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    rows_sorted = sorted(rows, key=lambda row: row.get("header_saved_lps_nearest_mm", float("inf")))
    _draw_overlays(rows_sorted, out_dir / "header_saved_lps_examples.png", "header_saved_lps", args.examples)
    _draw_overlays(rows_sorted, out_dir / "legacy_swap_flip_examples.png", "legacy_swap_flip", args.examples)
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
