import argparse
import csv
import os
import pickle
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms.functional as TF
from scipy.ndimage import binary_dilation, zoom
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from build_cache import _get_lung_mask
from config import HU_MAX, HU_MIN, IMG_SIZE, SEED
from dataset import load_nodules_ts, patient_split
from evaluate_lungx import (
    _case_dirs,
    _dicom_index,
    _load_hu_slice,
    _load_lungx_rows,
)


def _sample_key(sample):
    return (
        sample.get("patient_id"),
        sample.get("nodule_id"),
        sample.get("slice_idx"),
    )


def _load_cache(path):
    with open(path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{path} is not a completed cache list.")
    return samples


def _ts_lookup(path):
    lookup = {}
    for sample in _load_cache(path):
        key = _sample_key(sample)
        if key[0] is not None and key[1] is not None and sample.get("ts_mask") is not None:
            lookup[key] = sample["ts_mask"]
    return lookup


def _mask_to_tensor(mask):
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0)
    tensor = TF.resize(tensor, [IMG_SIZE, IMG_SIZE], antialias=False)
    return (tensor > 0.5).float()


def _image_to_tensor(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] == 3:
        image = image[1]
    image = np.clip(image, 0.0, 1.0)
    tensor = torch.from_numpy(image).unsqueeze(0)
    tensor = TF.resize(tensor, [IMG_SIZE, IMG_SIZE], antialias=True)
    return tensor.clamp(0.0, 1.0)


def _dilate_tensor_mask(mask, iterations):
    if iterations <= 0:
        return mask
    array = mask.squeeze(0).numpy().astype(bool)
    array = binary_dilation(array, iterations=int(iterations))
    return torch.from_numpy(array.astype(np.float32)).unsqueeze(0)


def _augment_triplet(ct, candidate, lung):
    if torch.rand(()) < 0.5:
        ct = TF.hflip(ct)
        candidate = TF.hflip(candidate)
        lung = TF.hflip(lung)
    if torch.rand(()) < 0.5:
        ct = TF.vflip(ct)
        candidate = TF.vflip(candidate)
        lung = TF.vflip(lung)

    angle = float(torch.empty(()).uniform_(-15.0, 15.0).item())
    translate = [0, 0]
    scale = 1.0
    ct = TF.affine(
        ct,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )
    candidate = TF.affine(
        candidate,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    )
    lung = TF.affine(
        lung,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0.0,
    )
    return ct.clamp(0.0, 1.0), (candidate > 0.5).float(), (lung > 0.5).float()


class TriStreamDataset(Dataset):
    def __init__(self, samples, augment=False, ablation="full", candidate_dilation=1):
        self.samples = samples
        self.labels = [int(sample["label"]) for sample in samples]
        self.augment = augment
        self.ablation = ablation
        self.candidate_dilation = candidate_dilation

    def __len__(self):
        return len(self.samples)

    def class_weights(self):
        n_pos = sum(self.labels)
        n_neg = len(self.labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            return torch.tensor(1.0)
        return torch.tensor(n_neg / n_pos, dtype=torch.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        ct = _image_to_tensor(sample["image"])
        candidate = _mask_to_tensor(sample["possible_nodule_mask"])
        candidate = _dilate_tensor_mask(candidate, self.candidate_dilation)
        lung = _mask_to_tensor(sample["ts_mask"])

        if self.augment:
            ct, candidate, lung = _augment_triplet(ct, candidate, lung)

        stream1 = ct.repeat(3, 1, 1)
        if self.ablation == "nocandidate":
            stream2 = stream1.clone()
        else:
            stream2 = (ct * 0.75 + candidate * 0.25).clamp(0.0, 1.0).repeat(3, 1, 1)
        if self.ablation == "nomask":
            stream3 = stream1.clone()
        else:
            stream3 = (ct * 0.85 + lung * 0.15).clamp(0.0, 1.0).repeat(3, 1, 1)
        label = torch.tensor(float(sample["label"]), dtype=torch.float32)
        return stream1, stream2, stream3, lung, label


class _ResNetFeature(nn.Module):
    def __init__(self, with_hooks=False):
        super().__init__()
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._activations = None
        self._gradients = None
        if with_hooks:
            self._forward_hook = self.features[-1].register_forward_hook(self._save_activations)
            self._backward_hook = self.features[-1].register_full_backward_hook(self._save_gradients)
        else:
            self._forward_hook = None
            self._backward_hook = None

    def _save_activations(self, _module, _input, output):
        self._activations = output

    def _save_gradients(self, _module, _grad_input, grad_output):
        self._gradients = grad_output[0]

    def forward(self, x):
        fmap = self.features(x)
        return torch.flatten(self.pool(fmap), 1)

    def gradcam(self):
        grads = self._gradients
        acts = self._activations
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=1))
        flat = cam.flatten(start_dim=1)
        cam_min = flat.min(dim=1).values.view(-1, 1, 1)
        cam_max = flat.max(dim=1).values.view(-1, 1, 1)
        return (cam - cam_min) / (cam_max - cam_min + 1e-8)

    def remove_hooks(self):
        if self._forward_hook is not None:
            self._forward_hook.remove()
        if self._backward_hook is not None:
            self._backward_hook.remove()


class TriStreamNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stream1 = _ResNetFeature(with_hooks=True)
        self.stream2 = _ResNetFeature()
        self.stream3 = _ResNetFeature()
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6144, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
        )

    def forward(self, x1, x2, x3):
        features = torch.cat(
            [self.stream1(x1), self.stream2(x2), self.stream3(x3)],
            dim=1,
        )
        return self.head(features).squeeze(1)

    def remove_hooks(self):
        self.stream1.remove_hooks()


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


def _evaluate(model, loader, device, criterion=None, gradcam=False):
    model.eval()
    labels = []
    probs = []
    total_loss = 0.0
    total = 0
    cam_inside = []
    for x1, x2, x3, lung, label in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)
        x3 = x3.to(device)
        lung = lung.to(device)
        label = label.to(device)
        if gradcam:
            model.zero_grad(set_to_none=True)
            logits = model(x1, x2, x3)
            scores = logits * (label * 2.0 - 1.0)
            scores.sum().backward()
            cam = model.stream1.gradcam()
            cam = F.interpolate(
                cam.unsqueeze(1),
                size=lung.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
            mass = cam.flatten(start_dim=1).sum(dim=1) + 1e-8
            inside = (cam * lung.squeeze(1)).flatten(start_dim=1).sum(dim=1) / mass
            cam_inside.extend((inside.detach().cpu().numpy() * 100.0).tolist())
        else:
            with torch.no_grad():
                logits = model(x1, x2, x3)
        if criterion is not None:
            loss = criterion(logits, label)
            total_loss += float(loss.item()) * label.numel()
            total += label.numel()
        labels.extend(label.detach().cpu().numpy().astype(int).tolist())
        probs.extend(torch.sigmoid(logits.detach()).cpu().numpy().tolist())
    metrics = _metrics(labels, probs)
    if total:
        metrics["loss"] = total_loss / total
    if cam_inside:
        metrics["gradcam_inside_ts_pct"] = float(np.mean(cam_inside))
    return metrics, labels, probs


def _write_predictions(path, labels, probs):
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["label", "probability_malignant", "prediction"])
        for label, prob in zip(labels, probs):
            writer.writerow([label, prob, int(prob >= 0.5)])


def _prepare_samples(candidate_cache, ts_cache):
    ts_by_key = _ts_lookup(ts_cache)
    samples = []
    for sample in _load_cache(candidate_cache):
        key = _sample_key(sample)
        if key in ts_by_key:
            item = dict(sample)
            item["ts_mask"] = ts_by_key[key]
            if item.get("possible_nodule_mask") is not None:
                samples.append(item)
    return samples


def _train_one(args, run_id, ablation):
    run_dir = os.path.join(args.results_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    samples = _prepare_samples(args.candidate_cache, args.ts_cache)
    train, val, test = patient_split(samples, seed=args.seed)
    train_ds = TriStreamDataset(
        train,
        augment=True,
        ablation=ablation,
        candidate_dilation=args.candidate_dilation,
    )
    val_ds = TriStreamDataset(val, augment=False, ablation=ablation, candidate_dilation=args.candidate_dilation)
    test_ds = TriStreamDataset(test, augment=False, ablation=ablation, candidate_dilation=args.candidate_dilation)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TriStreamNet().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=train_ds.class_weights().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_auc = -1.0
    best_path = os.path.join(run_dir, "best_model.pt")

    info_path = os.path.join(run_dir, "tristream_info.txt")
    with open(info_path, "w") as file:
        file.write(f"Run: {run_id}\n")
        file.write(f"Ablation: {ablation}\n")
        file.write(f"Candidate cache: {args.candidate_cache}\n")
        file.write(f"TS cache: {args.ts_cache}\n")
        file.write(f"Candidate dilation: {args.candidate_dilation}\n")
        file.write(f"Train/val/test: {len(train)}/{len(val)}/{len(test)}\n")
        file.write(f"Loss: BCEWithLogitsLoss(pos_weight={train_ds.class_weights().item():.6f})\n")
        file.write("Architecture: three ImageNet-pretrained ResNet-50 feature backbones, concat 6144 -> 512 -> 1.\n")

    epoch_rows = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total_loss = 0.0
        total = 0
        for x1, x2, x3, _lung, label in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            label = label.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x1, x2, x3)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * label.numel()
            total += label.numel()
        train_loss = total_loss / max(total, 1)
        val_metrics, _labels, _probs = _evaluate(model, val_loader, device, criterion)
        elapsed = time.time() - start
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auc": val_metrics["auc"],
                "val_accuracy": val_metrics["accuracy"],
                "epoch_seconds": elapsed,
            }
        )
        print(
            f"{run_id} epoch {epoch:02d}: train_loss={train_loss:.4f} "
            f"val_auc={val_metrics['auc']:.4f} time={elapsed:.1f}s",
            flush=True,
        )
        if val_metrics["auc"] > best_auc:
            best_auc = val_metrics["auc"]
            torch.save(model.state_dict(), best_path)

    with open(os.path.join(run_dir, "epochs.csv"), "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(epoch_rows[0].keys()))
        writer.writeheader()
        writer.writerows(epoch_rows)

    model.load_state_dict(torch.load(best_path, map_location=device))
    test_metrics, test_labels, test_probs = _evaluate(model, test_loader, device, criterion, gradcam=True)
    _write_predictions(os.path.join(run_dir, "lidc_test_predictions.csv"), test_labels, test_probs)
    with open(os.path.join(run_dir, "test_results.txt"), "w") as file:
        file.write("=== LIDC HELD-OUT TEST ===\n")
        for key in ["auc", "accuracy", "f1", "sensitivity", "specificity", "loss", "gradcam_inside_ts_pct"]:
            file.write(f"{key}: {test_metrics[key]:.4f}\n")

    lungx_metrics = _evaluate_lungx(args, model, device, ablation, run_dir)
    model.remove_hooks()
    return test_metrics, lungx_metrics


def _normalise_hu(raw_slice):
    image = np.clip(raw_slice.astype(np.float32), HU_MIN, HU_MAX)
    return ((image - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def _resize_224(array, order):
    factors = (IMG_SIZE / array.shape[0], IMG_SIZE / array.shape[1])
    return zoom(array.astype(np.float32), factors, order=order)


def _lungx_candidate_from_hu(raw_slice, lung_mask):
    hu = raw_slice.astype(np.float32)
    dense = hu > -600.0
    holes = binary_dilation(lung_mask.astype(bool), iterations=1) & dense
    return holes.astype(np.float32)


def _lungx_streams(raw_slice, ablation):
    image = _resize_224(_normalise_hu(raw_slice), order=1)
    ct = torch.from_numpy(np.clip(image, 0.0, 1.0)).unsqueeze(0)
    lung_native = _get_lung_mask(raw_slice).astype(bool)
    lung = torch.from_numpy((_resize_224(lung_native, order=0) > 0.5).astype(np.float32)).unsqueeze(0)
    candidate_native = _lungx_candidate_from_hu(raw_slice, lung_native)
    candidate = torch.from_numpy((_resize_224(candidate_native, order=0) > 0.5).astype(np.float32)).unsqueeze(0)
    candidate = _dilate_tensor_mask(candidate, 1)
    stream1 = ct.repeat(3, 1, 1)
    stream2 = stream1.clone() if ablation == "nocandidate" else (ct * 0.75 + candidate * 0.25).clamp(0, 1).repeat(3, 1, 1)
    stream3 = stream1.clone() if ablation == "nomask" else (ct * 0.85 + lung * 0.15).clamp(0, 1).repeat(3, 1, 1)
    return stream1, stream2, stream3


def _evaluate_lungx(args, model, device, ablation, run_dir):
    image_root = os.path.join(args.lungx_manifest_root, "SPIE-AAPM Lung CT Challenge")
    cases = _case_dirs(image_root)
    records = _load_lungx_rows(args.lungx_xlsx)
    dicom_cache = {}
    rows = []
    model.eval()
    with torch.no_grad():
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
            x1, x2, x3 = _lungx_streams(_load_hu_slice(dicom_path), ablation)
            logits = model(
                x1.unsqueeze(0).to(device),
                x2.unsqueeze(0).to(device),
                x3.unsqueeze(0).to(device),
            )
            rows.append({**record, "probability_malignant": float(torch.sigmoid(logits).item()), "dicom_path": dicom_path})

    labelled = [row for row in rows if row["label"] is not None]
    labels = [int(row["label"]) for row in labelled]
    probs = [float(row["probability_malignant"]) for row in labelled]
    metrics = _metrics(labels, probs)
    with open(os.path.join(run_dir, "lungx_predictions.csv"), "w", newline="") as file:
        fieldnames = list(rows[0].keys()) + ["prediction"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "prediction": int(row["probability_malignant"] >= 0.5)})
    with open(os.path.join(run_dir, "lungx_eval_summary.txt"), "w") as file:
        file.write("=== LUNGx EXTERNAL EVALUATION ===\n")
        file.write("Note: LUNGx has no saved TS/candidate cache here; stream overlays use annotation-free HU fallback masks.\n")
        file.write(f"Label counts: {dict(Counter(labels))}\n")
        for key in ["auc", "accuracy", "f1", "sensitivity", "specificity"]:
            file.write(f"{key}: {metrics[key]:.4f}\n")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-cache", default="cache/cache_ts_possible_nodule_prior_v4_original_ts_restricted.pkl")
    parser.add_argument("--ts-cache", default="cache/cache_ts_filled_dil1.pkl")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--candidate-dilation", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-all", action="store_true")
    parser.add_argument("--run-id", default="tristream_dil1_aug")
    parser.add_argument("--ablation", choices=["full", "nocandidate", "nomask"], default="full")
    parser.add_argument("--lungx-manifest-root", default=r"C:\repo\manifest-cgqtDj7Y2699835271585651107")
    parser.add_argument("--lungx-xlsx", default=r"data\lungx_annotations\TestSet_NoduleData_PublicRelease_wTruth.xlsx")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    runs = (
        [
            ("tristream_dil1_aug", "full"),
            ("tristream_nocandidate_aug", "nocandidate"),
            ("tristream_nomask_aug", "nomask"),
        ]
        if args.run_all
        else [(args.run_id, args.ablation)]
    )
    results = []
    for run_id, ablation in runs:
        lidc, lungx = _train_one(args, run_id, ablation)
        results.append((run_id, lidc, lungx))

    comparison_path = os.path.join(args.results_root, "tristream_comparison.csv")
    with open(comparison_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["model", "lidc_auc", "lungx_auc"])
        for run_id, lidc, lungx in results:
            writer.writerow([run_id, f"{lidc['auc']:.4f}", f"{lungx['auc']:.4f}"])
        writer.writerow(["base_aug_existing", "0.7507", "not tested"])
        writer.writerow(["zhang_ts_aug_existing", "0.7766", "0.6682"])

    progress_path = "PROGRESS.md"
    with open(progress_path, "a") as file:
        file.write("\n## TriStream Dil1 Experiments\n\n")
        for run_id, lidc, lungx in results:
            file.write(f"- {run_id}: LIDC AUC {lidc['auc']:.4f}; LUNGx AUC {lungx['auc']:.4f}\n")
    print("\nFinal comparison:")
    print("Model | LIDC AUC | LUNGx AUC")
    for run_id, lidc, lungx in results:
        print(f"{run_id} | {lidc['auc']:.4f} | {lungx['auc']:.4f}")
    print("base_aug (existing) | 0.7507 | not tested")
    print("zhang_ts_aug (existing) | 0.7766 | 0.6682")


if __name__ == "__main__":
    main()
