"""Combine the five dil1 before/after examples into one presentation image."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=r"results\dil1_supervisor_example\five_more")
    parser.add_argument("--out-path", default=r"results\dil1_supervisor_example\dil1_five_examples_combined.png")
    args = parser.parse_args()

    paths = sorted(Path(args.input_dir).glob("dil1_before_after_idx_*.png"))
    if not paths:
        raise FileNotFoundError(f"No dil1 example PNGs found in {args.input_dir}")

    images = [np.asarray(Image.open(path).convert("RGB")) for path in paths]
    target_width = min(image.shape[1] for image in images)
    resized = []
    for image in images:
        if image.shape[1] != target_width:
            ratio = target_width / image.shape[1]
            new_height = int(round(image.shape[0] * ratio))
            image = np.asarray(Image.fromarray(image).resize((target_width, new_height), Image.LANCZOS))
        resized.append(image)

    fig, axes = plt.subplots(len(resized), 1, figsize=(12, 3.0 * len(resized)))
    if len(resized) == 1:
        axes = [axes]
    for ax, image, path in zip(axes, resized, paths):
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_ylabel(path.stem.replace("dil1_before_after_", ""), fontsize=8)
    fig.suptitle(
        "Five Examples Fixed By TS Mask Cleanup (fill holes + closing radius 1 + dilation 1)",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.985], h_pad=0.05)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
