"""Export CT scans to NIfTI and create a MONAI bundle validation datalist.

This is for MONAI lung nodule detector inference. It writes one .nii.gz file
per full CT scan and a JSON datalist:

    {"validation": [{"image": "SCAN_ID.nii.gz"}, ...]}

The exported files are DICOM-derived HU volumes. Use the MONAI bundle setting
`whether_raw_luna16=true` for fresh, non-resampled exports.
"""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _save_nifti(volume, spacing, out_path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError("nibabel is required for Python fallback NIfTI export.") from exc

    affine = np.diag([float(spacing[0]), float(spacing[1]), float(spacing[2]), 1.0])
    image = nib.Nifti1Image(volume.astype(np.float32), affine=affine)
    nib.save(image, str(out_path))


def _write_datalist(nifti_paths, out_json, relative_to):
    rel_root = Path(relative_to).resolve()
    data = {
        "validation": [
            {"image": str(Path(path).resolve().relative_to(rel_root)).replace("\\", "/")}
            for path in sorted(nifti_paths)
        ]
    }
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def _run_dcm2niix(dicom_dir, out_dir, name):
    exe = shutil.which("dcm2niix")
    if exe is None:
        return None
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        exe,
        "-z",
        "y",
        "-f",
        name,
        "-o",
        str(out_dir),
        str(dicom_dir),
    ]
    subprocess.run(command, check=True)
    matches = sorted(out_dir.glob(f"{name}*.nii.gz"))
    if not matches:
        raise RuntimeError(f"dcm2niix ran but produced no NIfTI for {dicom_dir}")
    preferred = out_dir / f"{name}.nii.gz"
    if preferred.exists():
        return preferred
    if len(matches) == 1:
        matches[0].rename(preferred)
        return preferred
    raise RuntimeError(
        f"dcm2niix produced multiple NIfTIs for {dicom_dir}: {[str(p) for p in matches]}"
    )


def _lungx_dicom_dirs(manifest_root):
    from evaluate_lungx import _case_dirs

    image_root = Path(manifest_root) / "SPIE-AAPM Lung CT Challenge"
    cases = _case_dirs(str(image_root))
    return [(scan_id, Path(path)) for scan_id, path in sorted(cases.items())]


def _lungx_python_export(scan_id, case_dir, out_path):
    from build_lungx_ts_cache import _load_volume, _scan_meta
    from evaluate_lungx import _dicom_index

    entries, _by_instance = _dicom_index(str(case_dir))
    volume = _load_volume(entries)
    meta = _scan_meta(scan_id, entries)
    zvals = np.sort(np.asarray(meta.slice_zvals, dtype=np.float32))
    z_spacing = float(np.median(np.diff(zvals))) if len(zvals) > 1 else float(meta.slice_thickness)
    spacing = (float(meta.pixel_spacing), float(meta.pixel_spacing), abs(z_spacing))
    _save_nifti(volume, spacing, out_path)


def _export_lungx(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nifti_paths = []
    scans = _lungx_dicom_dirs(args.manifest_root)
    for index, (scan_id, dicom_dir) in enumerate(scans, start=1):
        if args.limit_scans and len(nifti_paths) >= args.limit_scans:
            break
        out_path = out_dir / f"{scan_id}.nii.gz"
        if out_path.exists() and not args.overwrite:
            print(f"[{index}/{len(scans)}] exists: {out_path}")
            nifti_paths.append(out_path)
            continue
        print(f"[{index}/{len(scans)}] exporting {scan_id}")
        if args.prefer_dcm2niix:
            converted = _run_dcm2niix(dicom_dir, out_dir, scan_id)
            if converted is None:
                print("  dcm2niix not found; using Python fallback")
                _lungx_python_export(scan_id, dicom_dir, out_path)
            else:
                out_path = converted
        else:
            _lungx_python_export(scan_id, dicom_dir, out_path)
        nifti_paths.append(out_path)
    return nifti_paths


def _lidc_scans(limit):
    import pylidc as pl

    scans = pl.query(pl.Scan).all()
    scans = sorted(scans, key=lambda scan: scan.patient_id)
    if limit:
        scans = scans[:limit]
    return scans


def _lidc_python_export(scan, out_path):
    volume = scan.to_volume().astype(np.float32)
    pixel_spacing = float(scan.pixel_spacing)
    zvals = np.sort(np.asarray(scan.slice_zvals, dtype=np.float32))
    z_spacing = float(np.median(np.diff(zvals))) if len(zvals) > 1 else float(scan.slice_thickness)
    spacing = (pixel_spacing, pixel_spacing, abs(z_spacing))
    _save_nifti(volume, spacing, out_path)


def _lidc_dicom_dir(scan):
    for attr in ["get_path_to_dicom_files", "path"]:
        value = getattr(scan, attr, None)
        if value is None:
            continue
        if callable(value):
            try:
                value = value()
            except TypeError:
                continue
        if value and Path(value).exists():
            return Path(value)
    return None


def _export_lidc(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nifti_paths = []
    scans = _lidc_scans(args.limit_scans)
    for index, scan in enumerate(scans, start=1):
        scan_id = scan.patient_id
        out_path = out_dir / f"{scan_id}.nii.gz"
        if out_path.exists() and not args.overwrite:
            print(f"[{index}/{len(scans)}] exists: {out_path}")
            nifti_paths.append(out_path)
            continue
        print(f"[{index}/{len(scans)}] exporting {scan_id}")
        converted = None
        if args.prefer_dcm2niix:
            dicom_dir = _lidc_dicom_dir(scan)
            if dicom_dir is not None:
                converted = _run_dcm2niix(dicom_dir, out_dir, scan_id)
            else:
                print("  LIDC DICOM directory not available from pylidc; using Python fallback")
        if converted is None:
            _lidc_python_export(scan, out_path)
        else:
            out_path = converted
        nifti_paths.append(out_path)
    return nifti_paths


def _default_args():
    return SimpleNamespace(
        manifest_root=r"C:\repo\manifest-cgqtDj7Y2699835271585651107",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["lidc", "lungx"], required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument(
        "--manifest-root",
        default=_default_args().manifest_root,
        help="LUNGx manifest root; used only with --source lungx.",
    )
    parser.add_argument("--limit-scans", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--prefer-dcm2niix",
        action="store_true",
        help="Use dcm2niix if installed. Recommended for best DICOM orientation handling.",
    )
    args = parser.parse_args()

    if args.source == "lungx":
        nifti_paths = _export_lungx(args)
    else:
        nifti_paths = _export_lidc(args)

    data = _write_datalist(nifti_paths, args.datalist, args.output_dir)
    print("")
    print("MONAI NIfTI export complete")
    print(f"Source: {args.source}")
    print(f"NIfTI folder: {args.output_dir}")
    print(f"Datalist: {args.datalist}")
    print(f"Validation items: {len(data['validation'])}")
    print("Datalist key: validation")
    print("For fresh DICOM-derived NIfTI, set MONAI bundle configs/inference.json:")
    print('  "whether_raw_luna16": true')


if __name__ == "__main__":
    main()
