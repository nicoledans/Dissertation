"""Create a compact original-vs-GGO-extended prior preview."""

import argparse
import os
import pickle
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from preview_possible_nodule_mask import (
    _candidate_lookup_for_patient,
    _center_image,
    _contour_for_sample,
    _mask,
)


def _load_cache(path):
    with open(path, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{path} is not a completed cache list.")
    return samples


def _load_ggo_indices(joined_csv, count):
    import csv

    rows = []
    with open(joined_csv, newline="") as file:
        for row in csv.DictReader(file):
            if row.get("nodule_type") == "ground_glass":
                rows.append(row)
    rows.sort(
        key=lambda row: (
            float(row.get("coverage", 0.0)),
            -float(row.get("candidate_area_pct", 0.0)),
        )
    )
    if not rows:
        return []
    chosen = []
    step = max(len(rows) // max(count, 1), 1)
    for row in rows[::step]:
        chosen.append(int(row["cache_index"]))
        if len(chosen) >= count:
            break
    return chosen


def _overlay_mask(image, mask, color=(0.0, 0.9, 1.0), alpha=0.35):
    rgb = np.stack([image, image, image], axis=-1)
    mask = mask.astype(bool)
    color_arr = np.asarray(color, dtype=np.float32)
    rgb[mask] = rgb[mask] * (1.0 - alpha) + color_arr * alpha
    return np.clip(rgb, 0.0, 1.0)


def _show_contour(ax, mask, color="yellow", width=1.0):
    if mask is not None and np.any(mask):
        ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-cache", default=r"cache\cache_ts_possible_nodule_prior_best.pkl")
    parser.add_argument("--ggo-cache", default=r"cache\cache_ts_possible_nodule_prior_ggo.pkl")
    parser.add_argument(
        "--joined-csv",
        default=r"results\prior_ggo_extended\ggo_extended_prior_eval_joined.csv",
    )
    parser.add_argument("--out-path", default=r"results\prior_ggo_extended\preview_examples.png")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()

    original_samples = _load_cache(args.original_cache)
    ggo_samples = _load_cache(args.ggo_cache)
    indices = _load_ggo_indices(args.joined_csv, args.count)
    if not indices:
        indices = list(range(min(args.count, len(ggo_samples))))

    by_patient = defaultdict(list)
    for index in indices:
        by_patient[ggo_samples[index]["patient_id"]].append(index)

    records = []
    for patient_id, patient_indices in sorted(by_patient.items()):
        lookup = _candidate_lookup_for_patient(patient_id)
        for index in patient_indices:
            sample = ggo_samples[index]
            contour = _contour_for_sample(sample, lookup)
            if contour is None:
                continue
            image = _center_image(sample)
            ts_mask = _mask(sample, "ts_mask")
            original_candidate = _mask(original_samples[index], "possible_nodule_mask")
            ggo_candidate = _mask(ggo_samples[index], "possible_nodule_mask")
            soft_prior = np.asarray(ggo_samples[index]["possible_nodule_prior"], dtype=np.float32)
            records.append(
                {
                    "index": index,
                    "patient_id": sample["patient_id"],
                    "image": image,
                    "ts_mask": ts_mask,
                    "contour": contour,
                    "original_candidate": original_candidate,
                    "ggo_candidate": ggo_candidate,
                    "soft_prior": soft_prior,
                }
            )

    cols = 5
    rows = len(records)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, max(rows, 1) * 2.6))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)
    titles = [
        "CT + nodule contour",
        "TS mask",
        "Original prior candidate",
        "GGO-extended candidate",
        "GGO soft prior",
    ]
    for row_idx, record in enumerate(records):
        panels = [
            (record["image"], "gray", None),
            (record["image"], "gray", record["ts_mask"]),
            (_overlay_mask(record["image"], record["original_candidate"], (0.0, 0.8, 1.0)), None, None),
            (_overlay_mask(record["image"], record["ggo_candidate"], (0.0, 1.0, 0.25)), None, None),
            (record["soft_prior"], "magma", None),
        ]
        for col_idx, (image, cmap, mask) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(image, cmap=cmap)
            if mask is not None:
                _show_contour(ax, mask, color="cyan", width=0.8)
            _show_contour(ax, record["contour"], color="yellow", width=1.0)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(titles[col_idx], fontsize=9)
            if col_idx == 0:
                ax.text(
                    0.0,
                    -0.08,
                    f"{record['patient_id']} idx={record['index']}",
                    transform=ax.transAxes,
                    fontsize=7,
                    va="top",
                )
    fig.suptitle(
        "GGO-extended possible-nodule prior preview\nYellow = LIDC nodule contour; cyan/green overlays are annotation-free candidates",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(args.out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {args.out_path}")
    print(f"Examples: {rows}")


if __name__ == "__main__":
    main()
