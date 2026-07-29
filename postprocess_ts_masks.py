"""Post-process saved TotalSegmentator masks without rerunning TS.

This reads an existing cache pickle, applies conservative 2-D cleanup to each
sample's TS mask, and writes a new cache.  It is intended for inspecting whether
small holes/boundary gaps explain mask-vs-nodule overlap failures.
"""

import argparse
import os
import pickle
from datetime import datetime

import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes


def _disk(radius):
    if radius <= 0:
        return None
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _postprocess_mask(mask, closing_radius, dilation_radius):
    mask = np.asarray(mask).astype(bool)
    if mask.ndim == 3 and mask.shape[0] == 3:
        processed = []
        before_channels = []
        for channel in mask:
            channel_processed, channel_before = _postprocess_mask(
                channel,
                closing_radius,
                dilation_radius,
            )
            processed.append(channel_processed)
            before_channels.append(channel_before)
        return (
            np.stack(processed, axis=0).astype(np.uint8),
            np.stack(before_channels, axis=0).astype(bool),
        )
    if mask.ndim != 2:
        raise ValueError(f"Expected 2-D mask or 3-channel 2.5-D mask, got {mask.shape}")

    before = mask.copy()
    mask = binary_fill_holes(mask)
    if closing_radius > 0:
        mask = binary_closing(mask, structure=_disk(closing_radius))
        mask = binary_fill_holes(mask)
    if dilation_radius > 0:
        mask = binary_dilation(mask, structure=_disk(dilation_radius))
    return mask.astype(np.uint8), before


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output-cache", required=True)
    parser.add_argument("--mask-key", default="ts_mask")
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument("--dilation-radius", type=int, default=0)
    args = parser.parse_args()

    if args.closing_radius < 0:
        parser.error("--closing-radius must be non-negative")
    if args.dilation_radius < 0:
        parser.error("--dilation-radius must be non-negative")

    with open(args.input_cache, "rb") as file:
        samples = pickle.load(file)
    if not isinstance(samples, list):
        raise ValueError(f"{args.input_cache} is not a completed cache list.")

    changed = 0
    total_added = 0
    total_removed = 0
    missing = 0

    for sample in samples:
        mask = sample.get(args.mask_key)
        if mask is None:
            missing += 1
            continue
        new_mask, before = _postprocess_mask(
            mask,
            closing_radius=args.closing_radius,
            dilation_radius=args.dilation_radius,
        )
        before_bool = before.astype(bool)
        after_bool = np.asarray(new_mask).astype(bool)
        added = int((after_bool & ~before_bool).sum())
        removed = int((before_bool & ~after_bool).sum())
        if added or removed:
            changed += 1
            total_added += added
            total_removed += removed
        sample[args.mask_key] = new_mask.astype(np.uint8)

    os.makedirs(os.path.dirname(args.output_cache) or ".", exist_ok=True)
    with open(args.output_cache, "wb") as file:
        pickle.dump(samples, file, protocol=pickle.HIGHEST_PROTOCOL)

    info_path = os.path.splitext(args.output_cache)[0] + "_postprocess_info.txt"
    lines = [
        "TotalSegmentator mask post-process",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Input cache: {args.input_cache}",
        f"Output cache: {args.output_cache}",
        f"Mask key: {args.mask_key}",
        f"Samples: {len(samples)}",
        f"Missing masks: {missing}",
        f"Changed masks: {changed}",
        f"Closing radius: {args.closing_radius}",
        f"Dilation radius: {args.dilation_radius}",
        f"Total pixels added: {total_added}",
        f"Total pixels removed: {total_removed}",
    ]
    with open(info_path, "w") as file:
        file.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"Info: {info_path}")


if __name__ == "__main__":
    main()
