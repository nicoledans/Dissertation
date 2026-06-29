"""Preview annotation-free MIL candidate crops before training."""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import load_nodules_hu, patient_split
from train_candidate_mil import (
    _candidate_tensors,
    _center_image,
    _annotation_match_for_sample,
    _draw_contours,
)


def _sample_subset(samples, split, count, seed):
    if split != "all":
        train, val, test = patient_split(samples)
        samples = {"train": train, "val": val, "test": test}[split]
    rng = np.random.default_rng(seed)
    if len(samples) <= count:
        return samples
    indices = rng.choice(len(samples), size=count, replace=False)
    return [samples[index] for index in sorted(indices)]


def _draw_box(axis, box, color, linewidth=1.2):
    top, left, bottom, right = [int(v) for v in box[:4]]
    axis.plot(
        [left, right, right, left, left],
        [top, top, bottom, bottom, top],
        color=color,
        linewidth=linewidth,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-path", default="cache/cache_hu_soft_blobs.pkl")
    parser.add_argument("--out", default="results/candidate_mil_crop_preview.png")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=5)
    parser.add_argument("--crop-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show-contours", action="store_true")
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.num_candidates < 1:
        parser.error("--num-candidates must be at least 1")
    if args.crop_size < 8:
        parser.error("--crop-size must be at least 8")

    samples = load_nodules_hu(args.cache_path)
    missing = [index for index, sample in enumerate(samples) if "soft_blob_map" not in sample]
    if missing:
        raise ValueError(
            f"{len(missing)} samples are missing soft_blob_map. "
            "Run build_soft_blob_cache.py first."
        )
    chosen = _sample_subset(samples, args.split, args.samples, args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    columns = 2 + args.num_candidates
    fig, axes = plt.subplots(
        len(chosen),
        columns,
        figsize=(3.2 * columns, 3.2 * len(chosen)),
        squeeze=False,
    )
    for row, sample in enumerate(chosen):
        image = _center_image(sample)
        blob = np.asarray(sample["soft_blob_map"], dtype=np.float32)
        data = _candidate_tensors(sample, args.num_candidates, args.crop_size)
        boxes = data["boxes"].numpy()
        valid = data["valid"].numpy().astype(bool)
        scores = data["candidate_scores"].numpy()
        centers = data["centers"].numpy()

        contour_match = _annotation_match_for_sample(sample) if args.show_contours else None
        majority = np.asarray(contour_match["majority"]).astype(bool) if contour_match else None
        union = np.asarray(contour_match["union"]).astype(bool) if contour_match else None

        ax = axes[row]
        ax[0].imshow(image, cmap="gray")
        ax[0].set_title(
            f"{sample['patient_id']}\nlabel={sample['label']}"
        )
        if contour_match:
            _draw_contours(ax[0], majority, union)

        ax[1].imshow(image, cmap="gray")
        ax[1].imshow(blob, cmap="magma", alpha=0.55, vmin=0, vmax=1)
        for idx in range(args.num_candidates):
            if valid[idx]:
                _draw_box(ax[1], boxes[idx], "cyan" if idx == 0 else "white")
        if contour_match:
            _draw_contours(ax[1], majority, union)
        ax[1].set_title("Soft blob + boxes")

        for idx in range(args.num_candidates):
            crop = data["crops"][idx, 0].numpy()
            ax[2 + idx].imshow(crop, cmap="gray", vmin=0, vmax=1)
            if valid[idx]:
                row_c, col_c = centers[idx]
                ax[2 + idx].set_title(
                    f"cand {idx}\nscore={scores[idx]:.3f}\nrc=({row_c},{col_c})"
                )
            else:
                ax[2 + idx].set_title(f"cand {idx}\ninvalid/pad")

        for axis in ax:
            axis.set_xticks([])
            axis.set_yticks([])

    fig.suptitle(
        "Annotation-free candidate MIL crop preview\n"
        "Candidate boxes come from soft_blob_map only; contours are optional audit-only overlays.",
        y=0.995,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    print(f"Saved candidate crop preview -> {args.out}")


if __name__ == "__main__":
    main()
