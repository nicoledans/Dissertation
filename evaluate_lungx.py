"""Evaluate a trained LIDC nodule classifier on the LUNGx challenge data."""

import argparse
import csv
import os
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter

import numpy as np
import pydicom
import torch
from scipy.ndimage import zoom
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from config import HU_MAX, HU_MIN, IMG_SIZE
from model import NoduleClassifier


XLSX_NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _xlsx_col_to_idx(cell_ref):
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for char in letters:
        index = index * 26 + ord(char.upper()) - 64
    return index - 1


def _read_xlsx_first_sheet(path):
    """Read a simple first-sheet .xlsx without pandas/openpyxl."""
    with zipfile.ZipFile(path) as archive:
        shared_strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall("a:si", XLSX_NS):
                shared_strings.append(
                    "".join(text.text or "" for text in item.findall(".//a:t", XLSX_NS))
                )

        root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in root.findall(".//a:row", XLSX_NS):
            values = []
            for cell in row.findall("a:c", XLSX_NS):
                index = _xlsx_col_to_idx(cell.attrib["r"])
                while len(values) <= index:
                    values.append("")
                value_node = cell.find("a:v", XLSX_NS)
                if value_node is None:
                    continue
                value = value_node.text or ""
                if cell.attrib.get("t") == "s":
                    value = shared_strings[int(value)]
                values[index] = value
            rows.append(values)
    return rows


def _parse_center(value):
    text = str(value).strip()
    if "," in text:
        x_text, y_text = [part.strip() for part in text.split(",", 1)]
        return float(x_text), float(y_text)
    digits = "".join(ch for ch in text if ch.isdigit())
    if len(digits) == 6:
        return float(digits[:3]), float(digits[3:])
    raise ValueError(f"Cannot parse nodule centre: {value!r}")


def _label_from_text(value):
    if value is None or str(value).strip() == "":
        return None
    text = str(value).lower()
    if "cancer" in text or "malignant" in text:
        return 1
    if "benign" in text:
        return 0
    return None


def _load_lungx_rows(xlsx_path):
    rows = _read_xlsx_first_sheet(xlsx_path)
    header = [str(value).strip() for value in rows[0]]
    records = []
    for raw in rows[1:]:
        if not raw or not raw[0] or str(raw[0]).startswith("*"):
            continue
        row = {header[index]: raw[index] if index < len(raw) else "" for index in range(len(header))}
        scan_id = str(row.get("Scan Number", "")).strip()
        if not scan_id:
            continue
        if not row.get("Nodule Center x,y Position*") or not row.get("Nodule Center Image"):
            continue
        nodule_number = row.get("Nodule Number", "1") or "1"
        x, y = _parse_center(row["Nodule Center x,y Position*"])
        label = _label_from_text(row.get("Final Diagnosis", row.get("Diagnosis", "")))
        records.append(
            {
                "scan_id": scan_id.upper(),
                "nodule_number": int(float(nodule_number)),
                "center_x_1based": x,
                "center_y_1based": y,
                "center_image": int(float(row["Nodule Center Image"])),
                "label": label,
                "diagnosis": row.get("Final Diagnosis", row.get("Diagnosis", "")),
            }
        )
    return records


def _case_dirs(root):
    mapping = {}
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if os.path.isdir(path):
            mapping[name.upper()] = path
    return mapping


def _dicom_index(case_dir):
    entries = []
    for current_root, _dirs, files in os.walk(case_dir):
        for filename in files:
            if not filename.lower().endswith(".dcm"):
                continue
            path = os.path.join(current_root, filename)
            ds = pydicom.dcmread(path, stop_before_pixels=True)
            instance = int(getattr(ds, "InstanceNumber", len(entries) + 1))
            z_pos = None
            if hasattr(ds, "ImagePositionPatient"):
                z_pos = float(ds.ImagePositionPatient[2])
            entries.append((instance, z_pos, path))
    if not entries:
        raise FileNotFoundError(f"No DICOM files found under {case_dir}")
    entries.sort(key=lambda item: item[0])
    by_instance = {instance: path for instance, _z, path in entries}
    return entries, by_instance


def _load_hu_slice(path):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return image * slope + intercept


def _prepare_image(raw_slice):
    image = np.clip(raw_slice.astype(np.float32), HU_MIN, HU_MAX)
    image = (image - HU_MIN) / (HU_MAX - HU_MIN)
    factors = (IMG_SIZE / image.shape[0], IMG_SIZE / image.shape[1])
    image = zoom(image, factors, order=1).astype(np.float32)
    tensor = torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1)
    return tensor


def _load_model(checkpoint, device):
    model = NoduleClassifier(input_channels=3).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()
    return model


def _metrics(labels, probs):
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(labels, probs)),
        "accuracy": float(np.mean(preds == labels)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-root",
        default=r"C:\repo\manifest-cgqtDj7Y2699835271585651107",
    )
    parser.add_argument(
        "--xlsx",
        default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx",
    )
    parser.add_argument(
        "--checkpoint",
        default=r"results\zhang_hu_aug_full\selected_model.pt",
    )
    parser.add_argument("--out-dir", default=r"results\lungx_eval_zhang_hu_aug_full")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    image_root = os.path.join(args.manifest_root, "SPIE-AAPM Lung CT Challenge")
    cases = _case_dirs(image_root)
    records = _load_lungx_rows(args.xlsx)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_model(args.checkpoint, device)

    dicom_cache = {}
    rows = []
    tensors = []
    pending = []
    for record in records:
        case_dir = cases.get(record["scan_id"])
        if case_dir is None:
            raise FileNotFoundError(f"Missing DICOM directory for {record['scan_id']}")
        if record["scan_id"] not in dicom_cache:
            dicom_cache[record["scan_id"]] = _dicom_index(case_dir)
        entries, by_instance = dicom_cache[record["scan_id"]]
        dicom_path = by_instance.get(record["center_image"])
        if dicom_path is None:
            index = max(0, min(record["center_image"] - 1, len(entries) - 1))
            dicom_path = entries[index][2]
        tensor = _prepare_image(_load_hu_slice(dicom_path))
        tensors.append(tensor)
        pending.append({**record, "dicom_path": dicom_path})

    probabilities = []
    with torch.no_grad():
        for start in range(0, len(tensors), args.batch_size):
            batch = torch.stack(tensors[start:start + args.batch_size]).to(device)
            logits = model(batch).squeeze(1)
            probabilities.extend(torch.sigmoid(logits).cpu().numpy().tolist())
    model.remove_hooks()

    for record, probability in zip(pending, probabilities):
        rows.append(
            {
                **record,
                "probability_malignant": probability,
                "prediction": int(probability >= 0.5),
            }
        )

    pred_path = os.path.join(args.out_dir, "lungx_predictions.csv")
    fieldnames = [
        "scan_id",
        "nodule_number",
        "center_x_1based",
        "center_y_1based",
        "center_image",
        "diagnosis",
        "label",
        "probability_malignant",
        "prediction",
        "dicom_path",
    ]
    with open(pred_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    labelled = [row for row in rows if row["label"] is not None]
    lines = [
        "=== LUNGx EXTERNAL EVALUATION ===",
        f"Manifest root: {args.manifest_root}",
        f"Annotation sheet: {args.xlsx}",
        f"Checkpoint: {args.checkpoint}",
        f"Device: {device}",
        f"Nodules scored: {len(rows)}",
        f"Labelled nodules: {len(labelled)}",
        f"Label counts: {dict(Counter(row['label'] for row in labelled))}",
    ]
    if labelled:
        labels = [row["label"] for row in labelled]
        probs = [row["probability_malignant"] for row in labelled]
        metrics = _metrics(labels, probs)
        lines.extend(
            [
                "",
                f"AUC: {metrics['auc']:.4f}",
                f"Accuracy @0.5: {metrics['accuracy']:.4f}",
                f"F1 @0.5: {metrics['f1']:.4f}",
                f"Sensitivity @0.5: {metrics['sensitivity']:.4f}",
                f"Specificity @0.5: {metrics['specificity']:.4f}",
            ]
        )
    else:
        lines.append("No truth labels found; wrote predictions only.")

    summary_path = os.path.join(args.out_dir, "lungx_eval_summary.txt")
    with open(summary_path, "w") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nSaved predictions: {pred_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
