# =============================================================================
# build_cache.py
# =============================================================================
# PURPOSE
# -------
# This script pre-processes the LIDC-IDRI CT dataset and saves every nodule
# sample to a single pickle file ("the cache").  All downstream scripts
# (baseline.py, train.py, compare_all.py) load this cache instead of touching
# the raw DICOM files, which makes training fast and reproducible.
#
# WHY PRE-PROCESS RATHER THAN PROCESS ON-THE-FLY?
# ------------------------------------------------
# Each LIDC scan is a full 3-D CT volume stored as hundreds of DICOM files.
# Loading + masking + resizing one scan takes several seconds.  A training
# loop that did this on every batch would spend most of its time on I/O, not
# learning.  By doing all of that work once and writing the result to disk, we
# pay the cost once and reuse the result indefinitely.
#
# WHAT DOES THE CACHE CONTAIN?
# -----------------------------
# A Python list of dicts, one per accepted nodule.  Each dict has:
#   "image"      â€” 224Ã—224 float32 array, HU values clipped to [-1000, 400]
#                  and linearly scaled to [0, 1].
#   "mask"       â€” 224Ã—224 uint8 array, HU-threshold-based lung foreground mask.
#   "ts_mask"    â€” 224Ã—224 uint8 array, TotalSegmentator-based lung mask
#                  (None when --no-ts is passed).
#   "label"      â€” int, 0 = benign, 1 = malignant (from radiologist consensus).
#   "patient_id" â€” string, LIDC patient ID (e.g. "LIDC-IDRI-0001").
#
# TWO MASK TYPES â€” WHY?
# ----------------------
# Experiment 1 uses the HU-threshold mask (fast, no extra dependency).
# Experiment 2 uses the TotalSegmentator (TS) mask (more anatomically accurate,
# requires GPU + extra packages).  Storing both in the same cache lets you run
# either experiment from one build.
#
# OVERALL FLOW
# ------------
#   1.  Parse command-line arguments (sample size, --no-ts, --fresh, path).
#   2.  Load (or start) a checkpoint so a crash can be resumed.
#   3.  Shuffle all LIDC scans with the configured random seed for an
#       unbiased subset when SAMPLE_SIZE < total available scans.
#   4.  For each scan:
#         a.  Load the 3-D CT volume.
#         b.  Run TotalSegmentator on the whole volume (once per scan).
#         c.  For each nodule group with enough radiologist annotations:
#               - Compute average malignancy; skip ambiguous (==3) nodules.
#               - Find the middle annotated slice across all radiologists.
#               - Extract that 2-D slice, compute the HU lung mask on it.
#               - Clip + normalise HU, resize image + masks to 224Ã—224.
#               - Append the sample dict to the list.
#         d.  Write a checkpoint after every scan.
#   5.  Pickle the complete list to disk.
# =============================================================================

import os           # file-system operations: makedirs, path joins, existence checks
import pickle       # serialise the sample list to a binary file
import hashlib      # stable image hashes for exact-duplicate detection
import tempfile     # create a throw-away directory for TotalSegmentator's temp files
import shutil       # delete the temp directory tree after TS finishes
import argparse     # parse command-line flags so the script is re-configurable
                    # without editing source code
from datetime import datetime   # timestamp entries in the human-readable status file

import numpy as np              # array maths â€” the entire image/mask pipeline runs
                                # on NumPy arrays
import pylidc as pl             # Python wrapper for the LIDC-IDRI dataset; provides
                                # SQLAlchemy-backed access to DICOM scans and
                                # radiologist annotations

from scipy.ndimage import (
    # label: assigns a unique integer ID to every connected region in a binary mask.
    #   Used in the HU lung segmentation to separate background air, left lung,
    #   right lung, and other internal structures.
    label,

    # binary_dilation: morphologically expands a binary mask by a structuring element.
    #   Used after lung segmentation to add a small margin around the lung boundary,
    #   ensuring nodules that sit at the pleural edge are not clipped.
    binary_dilation,

    # binary_fill_holes: fills enclosed background regions inside a binary mask.
    #   Used in the TS pipeline to close the inter-lobe gaps that TotalSegmentator
    #   sometimes leaves between adjacent lung lobes.
    binary_fill_holes,

    # center_of_mass: returns the (row, col) centroid of a boolean mask, weighted
    #   by pixel values.  Used to decide which extracted lung region is the left
    #   lung (lower column index) and which is the right lung.
    center_of_mass,

    # zoom: resamples an array to a new shape using interpolation.
    #   Used to resize 512Ã—512 (or other native resolution) slices to 224Ã—224,
    #   the standard input size for ResNet-based models.
    zoom,
)

from skimage.morphology import disk as morpho_disk
# disk(r) builds a circular (disk-shaped) 2-D structuring element of radius r.
# This is the correct shape for isotropic dilation on a CT slice where in-plane
# pixel spacing is the same in X and Y.  A square structuring element would
# over-dilate in the diagonal directions.

# ---------------------------------------------------------------------------
# Config import
# ---------------------------------------------------------------------------
# All numeric hyper-parameters live in config.py so there is one place to
# change them.  We import them with private aliases (_CFG_*) and then assign
# them to module-level names below so the rest of this file never references
# the config module directly after the import block.  That pattern makes it
# easy to spot every constant this module depends on in one place.
from config import (
    SAMPLE_SIZE as _CFG_SAMPLE_SIZE,            # how many nodule samples to collect
    LUNG_HU_THRESHOLD as _CFG_LUNG_HU_THRESHOLD,# HU cut-off for air vs tissue
    MASK_DILATION as _CFG_MASK_DILATION,        # dilation radius in pixels
    CACHE_PATH as _CFG_CACHE_PATH,              # output pickle file path
    MIN_RADIOLOGISTS as _CFG_MIN_RADIOLOGISTS,  # minimum annotation consensus
    MALIGNANCY_THRESHOLD as _CFG_MALIGNANCY_THRESHOLD,  # score that separates classes
    SEED as _CFG_SEED,                          # RNG seed for scan shuffling
    HU_MIN as _CFG_HU_MIN,                      # lower HU clip bound for normalisation
    HU_MAX as _CFG_HU_MAX,                      # upper HU clip bound for normalisation
)

# ---------------------------------------------------------------------------
# Optional heavy dependencies (TotalSegmentator path only)
# ---------------------------------------------------------------------------
# nibabel, torch, and totalsegmentator are only needed when building TS masks.
# Wrapping the import in try/except lets users who only want HU masks run the
# script without installing those packages (via --no-ts).
# _TS_INSTALLED is checked later so we can print a clear error message rather
# than an unhelpful ImportError traceback.
try:
    import nibabel as nib
    # nibabel reads and writes NIfTI files (.nii.gz), the standard 3-D medical
    # image format expected by TotalSegmentator.

    import torch as _torch
    # PyTorch is used only to query torch.cuda.is_available() so we know whether
    # to pass device="gpu" or device="cpu" to TotalSegmentator.

    from totalsegmentator.python_api import totalsegmentator as _run_ts
    # TotalSegmentator is a pre-trained nnU-Net model that segments ~100
    # anatomical structures from CT scans, including all five lung lobes.
    # We call its Python API so we don't have to shell out to a subprocess.
    _TS_INSTALLED = True
except ImportError:
    _TS_INSTALLED = False

# ---------------------------------------------------------------------------
# Module-level constants â€” single source of truth
# ---------------------------------------------------------------------------
# Re-assigning the config values to plain names means every function below
# reads e.g. LUNG_HU_THRESHOLD instead of _CFG_LUNG_HU_THRESHOLD, keeping
# the code readable while still centralising the values in config.py.
LUNG_HU_THRESHOLD = _CFG_LUNG_HU_THRESHOLD
# Hounsfield Unit threshold for air.  HU is a standardised scale:
#   -1000 HU = air, 0 HU = water, +400 HU = dense bone.
# Lung parenchyma and airway lumen are air-filled and fall below -500 HU.
# Tissue, blood, and fat are all above -500 HU.
# Thresholding at -500 therefore separates lung/air from soft tissue.

MASK_DILATION = _CFG_MASK_DILATION
# After segmenting the lungs, we dilate the binary mask by this many pixels
# using a disk structuring element.  Value = 3 means the boundary expands by
# ~3 pixels in every direction.  This matters because nodules at the pleural
# surface (where the lung meets the chest wall) may be partially outside the
# raw segmented region; the dilation ensures they are included.

MALIGNANCY_THRESHOLD = _CFG_MALIGNANCY_THRESHOLD
# The LIDC dataset records each radiologist's malignancy rating on a scale
# of 1 (definitely benign) to 5 (definitely malignant).  We average the
# ratings across all radiologists who annotated a nodule:
#   avg < 3  â†’ label 0 (benign)
#   avg > 3  â†’ label 1 (malignant)
#   avg == 3 â†’ excluded (ambiguous, would add noise to both classes)
# This is the standard LIDC binary classification protocol used in published
# literature (e.g. Armato et al., 2011).

MIN_RADIOLOGISTS = _CFG_MIN_RADIOLOGISTS
# LIDC nodules are annotated by 1â€“4 radiologists.  We require at least
# MIN_RADIOLOGISTS (= 2) independent annotations before accepting a nodule,
# so that the malignancy average is based on more than one opinion.
# Nodules seen by only one radiologist are discarded.

CACHE_PATH = _CFG_CACHE_PATH
# Default output path for the finished pickle file.

SEED = _CFG_SEED
# Fixed integer seed for NumPy's random number generator.  Ensures that the
# scan shuffle order is identical every time, making the cached dataset
# deterministically reproducible from the same raw DICOM files.

HU_MIN = _CFG_HU_MIN
# Lower bound of the HU window used for normalisation (-1000 HU = air).
# Pixels below this value are clipped to HU_MIN before scaling.

HU_MAX = _CFG_HU_MAX
# Upper bound of the HU window (+400 HU â‰ˆ dense cortical bone).
# The window [-1000, 400] is a standard lung window that retains the full
# dynamic range relevant for lung nodule detection without letting extreme
# implant artefacts (+3000 HU) distort the normalised scale.


# =============================================================================
# HU-THRESHOLD LUNG MASK
# =============================================================================
# Physics-based lung foreground mask.
# Replicates the pipeline developed and visualised in hu_mask.ipynb.
#
# PURPOSE
# -------
# The HU mask marks every pixel that belongs to the lung parenchyma.
# During training it is used as an attention signal (analogous to Grad-CAM):
# the model is encouraged to attend to lung tissue rather than background.
#
# WHY A PHYSICS-BASED APPROACH?
# ------------------------------
# CT HU values are physically meaningful and scanner-independent (unlike RGB
# pixel values in natural images).  A simple threshold therefore reliably
# separates air from tissue without any learned weights, making it fast,
# interpretable, and guaranteed to generalise across scanners.
#
# PIPELINE OVERVIEW (7 steps)
# ----------------------------
#  Step 1  Threshold at LUNG_HU_THRESHOLD (-500 HU) to find air pixels.
#  Step 2  Build the largest filled patient-body mask from non-air pixels.
#  Step 3  Keep only air cavities inside that filled body.
#  Step 4  Remove any remaining border-touching air.
#  Step 5  Keep the 2 largest internal body-contained air regions.
#  Step 6  Assign left vs right by centroid column position.
#  Step 7  Combine and dilate with disk(MASK_DILATION).
# =============================================================================

# This follows logic in hu_mask.ipynb
def _get_lung_mask(slice_2d):
    """Build an orientation-agnostic HU lung-air mask for one CT slice."""
    air = slice_2d < LUNG_HU_THRESHOLD

    # Find the patient body, fill its holes, then only consider air cavities
    # inside that filled body. This rejects CT-table air without assuming the
    # table is at the bottom/top/left/right of the image.
    body = slice_2d >= LUNG_HU_THRESHOLD
    body_labeled, n_body = label(body)
    if n_body == 0:
        return np.zeros(slice_2d.shape, dtype=np.uint8)

    body_sizes = np.bincount(body_labeled.ravel())[1:]
    body_label = int(np.argmax(body_sizes) + 1)
    body_filled = binary_fill_holes(body_labeled == body_label)

    internal_air = air & body_filled
    labeled, _ = label(internal_air)

    # Safety net for unusual crops where the filled body touches an edge.
    border_labels = set()
    border_labels.update(labeled[0, :])
    border_labels.update(labeled[-1, :])
    border_labels.update(labeled[:, 0])
    border_labels.update(labeled[:, -1])
    border_labels.discard(0)

    if border_labels:
        labeled = labeled.copy()
        for lbl in border_labels:
            labeled[labeled == lbl] = 0

    labeled2, num2 = label(labeled > 0)
    if num2 == 0:
        return np.zeros(slice_2d.shape, dtype=np.uint8)

    region_sizes = np.bincount(labeled2.ravel())[1:]
    keep_labels = np.argsort(region_sizes)[::-1][:min(2, len(region_sizes))] + 1
    lung_regions = [labeled2 == lbl for lbl in keep_labels]

    if len(lung_regions) == 2:
        cx = [center_of_mass(m)[1] for m in lung_regions]
        lungs = [
            lung_regions[int(np.argmin(cx))],
            lung_regions[int(np.argmax(cx))],
        ]
    else:
        lungs = lung_regions

    combined = lungs[0].copy()
    for lung in lungs[1:]:
        combined |= lung

    return binary_dilation(combined, structure=morpho_disk(MASK_DILATION)).astype(np.uint8)



# =============================================================================
# TOTALSEGMENTATOR LUNG MASK
# =============================================================================
# This function runs TotalSegmentator on the full 3-D CT volume and returns
# a 3-D binary mask that covers all five lung lobes.
#
# WHY 3-D INSTEAD OF 2-D?
# ------------------------
# TotalSegmentator is a 3-D nnU-Net: it uses the full volumetric context
# (adjacent slices) to predict each voxel's label.  Running it on individual
# 2-D slices would break that context and produce poor results.  We therefore
# run it once per scan on the whole volume, then extract the relevant 2-D
# slice inside the main processing loop.
#
# PIPELINE
# ---------
#  1.  Convert the NumPy volume to a NIfTI file with the correct physical
#      affine (pixel spacing and slice thickness).
#  2.  Run TotalSegmentator, requesting only the 5 lung lobe labels.
#  3.  Load each lobe's segmentation and add them into one combined mask.
#  4.  Fill holes (inter-lobe gaps) and dilate.
#  5.  Return the 3-D uint8 mask.
# =============================================================================

# Canonical label names for the five lung lobes as used by TotalSegmentator.
# These must exactly match the output filename stems that TS writes.
_TS_LUNG_LABELS = [
    "lung_upper_lobe_left", "lung_lower_lobe_left",
    "lung_upper_lobe_right", "lung_middle_lobe_right",
    "lung_lower_lobe_right",
]

# Total Segmentator Mask
def _get_ts_vol_mask(vol, scan, device):
    # vol:    3-D float32 NumPy array of shape (rows, cols, slices) containing
    #         raw HU values for the entire CT scan.
    # scan:   pylidc Scan object â€” provides pixel spacing and slice z-positions.
    # device: string "gpu" or "cpu", passed directly to TotalSegmentator.

    patient_id = scan.patient_id
    # Store patient_id now so we can use it in error messages even if the scan
    # object is somehow no longer accessible inside the except block.

    tmpdir = tempfile.mkdtemp(prefix=f"ts_{patient_id}_")
    # Create a unique temporary directory for this scan's intermediate files.
    # tempfile.mkdtemp guarantees the name doesn't collide with any existing
    # directory, which matters when multiple scans run concurrently (future use).
    # The prefix makes it easy to identify in the OS temp folder if debugging.

    try:
        nifti_in = os.path.join(tmpdir, "ct.nii.gz")
        # Path where we will write the CT volume as a compressed NIfTI file.
        # TotalSegmentator expects a NIfTI (not raw NumPy or DICOM) as input.

        nifti_out = os.path.join(tmpdir, "seg")
        os.makedirs(nifti_out, exist_ok=True)
        # TotalSegmentator writes one .nii.gz file per label into this directory.
        # We create it explicitly; exist_ok=True is a no-op if it already exists.

        # ------------------------------------------------------------------
        # Build the NIfTI affine matrix
        # ------------------------------------------------------------------
        # A NIfTI affine is a 4Ã—4 matrix that maps voxel indices (i, j, k) to
        # physical coordinates in millimetres.  TotalSegmentator uses this to
        # operate at the correct physical scale â€” if we used an identity affine
        # (1 mmÂ³ isotropic), it would assume all voxels are 1mmÃ—1mmÃ—1mm, which
        # is wrong for most LIDC scans and causes fragmented or incorrect lobe
        # predictions.
        px = float(scan.pixel_spacing)
        # pixel_spacing: in-plane (X and Y) voxel size in mm.
        # LIDC CT scans are square in-plane, so a single value describes both
        # dimensions.  Typical values range from 0.5 mm to 0.9 mm.

        zvals = np.sort(scan.slice_zvals)
        # slice_zvals: array of actual z-axis positions (in mm) for every
        # DICOM slice in this scan, derived from the ImagePositionPatient tag.
        # We sort because some DICOM datasets store slices in reverse order.
        sl = float(np.median(np.diff(zvals))) if len(zvals) > 1 else float(scan.slice_thickness)
        # np.diff(zvals) gives the gap between consecutive slice positions.
        # np.median is used rather than np.mean to be robust against occasional
        # missing slices or duplicate z-values in the DICOM header.
        # scan.slice_thickness is the *nominal* thickness recorded in the DICOM
        # header â€” it is not always equal to the actual slice-to-slice spacing
        # (e.g. scans with overlapping reconstruction have thickness > spacing).
        # Using actual z-positions avoids this ambiguity.
        # Fallback to slice_thickness only when there is a single slice
        # (no diff possible), which should never occur in practice.

        affine = np.diag([px, px, sl, 1.0])
        # np.diag builds a 4Ã—4 diagonal matrix from [px, px, sl, 1.0].
        # Diagonal entries map voxel steps to mm:
        #   affine[0,0] = px  â†’ 1 voxel step in X = px mm
        #   affine[1,1] = px  â†’ 1 voxel step in Y = px mm
        #   affine[2,2] = sl  â†’ 1 voxel step in Z = sl mm
        #   affine[3,3] = 1   â†’ homogeneous coordinate (always 1)
        # Off-diagonal entries are 0 (no rotation or shear), which is correct
        # for axial CT scans where the volume axes align with the scanner axes.

        nib_img = nib.Nifti1Image(vol.astype(np.float32), affine=affine)
        # nib.Nifti1Image wraps the NumPy array and the affine into a NIfTI
        # object.  vol is cast to float32 because NIfTI and most medical image
        # tools expect 32-bit floats for CT data.
        nib.save(nib_img, nifti_in)
        # Write to disk.  .nii.gz is gzip-compressed; nibabel handles this
        # automatically when the filename ends in ".gz".

        # ------------------------------------------------------------------
        # Run TotalSegmentator
        # ------------------------------------------------------------------
        _run_ts(
            nifti_in, nifti_out,
            task="total",       # use the full 104-class model (includes lung lobes)
            device=device,      # "gpu" or "cpu" â€” GPU is ~20Ã— faster here
            fast=True,          # use the lower-resolution fast mode; full-res adds
                                # little accuracy for lung lobe segmentation but
                                # doubles runtime and memory usage
            quiet=True,         # suppress TS's verbose progress bars; our own
                                # print statements are enough
            roi_subset=[        # only compute the 5 lung lobe labels â€” skipping
                                # the other ~99 structures cuts runtime significantly
                "lung_upper_lobe_left", "lung_lower_lobe_left",
                "lung_upper_lobe_right", "lung_middle_lobe_right",
                "lung_lower_lobe_right",
            ],
        )
        # After this call, nifti_out/ contains up to 5 files:
        #   lung_upper_lobe_left.nii.gz, lung_lower_lobe_left.nii.gz, etc.
        # Each file is a binary volume with 1 where that lobe was predicted.

        # ------------------------------------------------------------------
        # Merge the five lobe masks into one combined lung mask
        # ------------------------------------------------------------------
        combined = None
        for lname in _TS_LUNG_LABELS:
            seg_path = os.path.join(nifti_out, f"{lname}.nii.gz")
            if os.path.exists(seg_path):
                seg = nib.load(seg_path).get_fdata()
                # get_fdata() returns a float64 NumPy array.
                # Values are 0.0 (not this lobe) or 1.0 (this lobe).
                combined = seg if combined is None else combined + seg
                # On the first found file, initialise combined.
                # On subsequent files, add â€” summing multiple binary arrays
                # gives a "vote count" per voxel.  Any voxel belonging to at
                # least one lobe will have combined > 0.

        if combined is None:
            # TS produced no output files â€” this shouldn't happen if the scan
            # is a valid thoracic CT, but we guard against it rather than
            # crashing the entire cache build.
            return None

        # ------------------------------------------------------------------
        # Post-process the 3-D mask
        # ------------------------------------------------------------------
        ts_3d = binary_fill_holes((combined > 0).astype(bool))
        # (combined > 0) is True wherever any lobe was predicted â€” converts
        # the summed float array back to a clean binary mask.
        # binary_fill_holes operates in 3-D here (the input is 3-D), closing
        # any enclosed cavities â€” specifically the inter-lobe fissure gaps that
        # TS sometimes leaves between the upper and lower lobes.  Without this
        # step, the mask would have small "holes" at lobe boundaries, which
        # could exclude nodules that sit in a fissure.

        ts_3d = binary_dilation(ts_3d, iterations=MASK_DILATION)
        # Expand the 3-D mask by MASK_DILATION voxels in every direction.
        # Uses the default ball-shaped structuring element in 3-D, which is
        # the 3-D analogue of the disk used in the HU mask.
        # This adds the same pleural margin as the HU mask, keeping both masks
        # consistent in their treatment of pleural-surface nodules.

        return ts_3d.astype(np.uint8)
        # Return as uint8 (0 / 1) to match the dtype of the HU mask and save
        # memory (bool and uint8 are both 1 byte per element, but uint8 is
        # more widely compatible with zoom and other array operations).

    except Exception as e:
        # Catch any error from nibabel, TotalSegmentator, or file I/O.
        # We print a warning but do NOT re-raise â€” the scan is skipped rather
        # than aborting the entire cache build.  This is important because LIDC
        # has ~1000 scans and some may have unusual anatomy or header issues
        # that confuse TS; losing a few scans is acceptable.
        print(f"  TS WARNING for {patient_id}: {e} â€” skipping TS mask for this scan")
        return None

    finally:
        # The finally block runs whether or not an exception was raised.
        # This guarantees the temporary directory is always deleted, even if
        # TS crashes partway through.  Temp files for a full CT scan can be
        # several hundred MB; cleaning up is essential when processing 1000+
        # scans.  ignore_errors=True prevents a secondary exception if the
        # directory was already removed.
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# SLICE SELECTION
# =============================================================================
def _get_middle_slice_idx(ann_group):
    # ann_group: list of pylidc Annotation objects that all describe the SAME
    #            nodule (one per radiologist who annotated it).
    #
    # PURPOSE
    # -------
    # Each radiologist annotated the nodule on a subset of CT slices and drew
    # a contour on each annotated slice.  Different radiologists may have
    # started and stopped their annotations on different slices, so the union
    # of all their annotated slices defines the full axial extent of the nodule.
    # We want the slice that is most centrally located within that extent,
    # because:
    #   1. The nodule's cross-section is largest near the centre, giving the
    #      model the most informative view.
    #   2. Using the consensus extent (all radiologists) rather than one
    #      radiologist's extent (ann_group[0] only) is consistent with the
    #      fact that our label is also a consensus (average malignancy across
    #      all radiologists).  Both the label and the slice selection now
    #      reflect the full group of annotations.

    k_indices = sorted(set(
        c.image_k_position for ann in ann_group for c in ann.contours
    ))
    # This generator expression iterates over every radiologist's annotation
    # (ann) and, within each annotation, over every contour (c) that
    # radiologist drew.  c.image_k_position is the z-index (slice number)
    # within the scan volume where that contour lives.
    # set() deduplicates â€” multiple radiologists often annotate the same
    # central slices, so without deduplication those slices would appear
    # multiple times and bias the "middle" index.
    # sorted() gives us a list in ascending slice order.
    # The full expression collects the sorted, deduplicated set of all slice
    # indices covered by any annotation from any radiologist.

    return k_indices[len(k_indices) // 2]
    # Integer floor division gives the 0-based index of the central element.
    # For an even number of annotated slices (e.g. 6), this returns the lower
    # of the two middle elements (index 2 of 0..5), which is a standard
    # convention for "middle of even list" and is consistent across all nodules.


def _get_consensus_nodule_center(ann_group):
    """Return the mean in-plane nodule center across radiologist annotations."""
    centers = []
    for annotation in ann_group:
        bbox = annotation.bbox()
        row = 0.5 * (bbox[0].start + bbox[0].stop - 1)
        col = 0.5 * (bbox[1].start + bbox[1].stop - 1)
        centers.append((row, col))
    if not centers:
        raise ValueError("Nodule group contains no annotation bounding boxes.")
    return tuple(float(value) for value in np.mean(centers, axis=0))


def _extract_square(array_2d, center, side_pixels, pad_value):
    """Extract a fixed square around center, padding beyond image edges."""
    side_pixels = int(side_pixels)
    if side_pixels < 4:
        raise ValueError(f"Crop side must be at least 4 pixels; got {side_pixels}.")

    center_row, center_col = center
    row_start = int(round(center_row - side_pixels / 2))
    col_start = int(round(center_col - side_pixels / 2))
    row_end = row_start + side_pixels
    col_end = col_start + side_pixels

    output = np.full(
        (side_pixels, side_pixels), pad_value, dtype=array_2d.dtype
    )
    src_row_start = max(row_start, 0)
    src_col_start = max(col_start, 0)
    src_row_end = min(row_end, array_2d.shape[0])
    src_col_end = min(col_end, array_2d.shape[1])
    if src_row_start >= src_row_end or src_col_start >= src_col_end:
        raise ValueError("Nodule-centred crop does not overlap the source slice.")

    dst_row_start = src_row_start - row_start
    dst_col_start = src_col_start - col_start
    dst_row_end = dst_row_start + (src_row_end - src_row_start)
    dst_col_end = dst_col_start + (src_col_end - src_col_start)
    output[dst_row_start:dst_row_end, dst_col_start:dst_col_end] = array_2d[
        src_row_start:src_row_end, src_col_start:src_col_end
    ]
    return output, (row_start, col_start, row_end, col_end)


# =============================================================================
# CHECKPOINT UTILITIES
# =============================================================================
# Processing 1000+ LIDC scans takes hours.  If the process is killed (power
# loss, OOM, Ctrl-C), we want to resume from where we stopped rather than
# starting over.  We achieve this by writing a checkpoint pickle after every
# scan.  The checkpoint stores:
#   "samples"       â€” the list of sample dicts collected so far.
#   "next_scan_idx" â€” the index of the first scan NOT yet processed.
# On resume, we skip all scans with index < next_scan_idx.

def _save_checkpoint(samples, next_scan_idx, next_nod_idx, path):
    # samples: current list of collected sample dicts.
    # next_scan_idx: scan index to resume from.
    # next_nod_idx: nodule-group index within that scan to resume from.
    #   0 means "start from the beginning of next_scan_idx" (used when a
    #   scan completes fully).  N > 0 means "skip the first N nodule groups
    #   of next_scan_idx because they were already processed".
    # path: file path for the checkpoint pickle.
    with open(path, "wb") as f:
        pickle.dump({
            "samples": samples,
            "next_scan_idx": next_scan_idx,
            "next_nod_idx": next_nod_idx,
        }, f)
    # "wb" = write binary â€” required for pickle.
    # We overwrite the checkpoint file on every save rather than appending,
    # so the file is always a complete, self-consistent snapshot.


def _write_status(samples, cache_out, sample_size, complete=False):
    # Writes a human-readable plain-text status file to cache/cache_status.txt.
    # This lets you check progress without opening a Python interpreter â€”
    # just `cat cache/cache_status.txt` in the terminal.
    #
    # samples:     current list of collected sample dicts.
    # cache_out:   the final cache output path (for display only).
    # sample_size: target number of samples (for the "X / Y" progress display).
    # complete:    True when called after the final pickle write; changes the
    #              status line and timestamp label.

    n_mal = sum(s["label"] for s in samples)
    # s["label"] is 0 or 1, so summing gives the count of malignant samples.

    n_ben = len(samples) - n_mal
    # Benign count = total âˆ’ malignant.

    ts_count = sum(1 for s in samples if s.get("ts_mask") is not None)
    # Count samples that have a TS mask.  .get() is used instead of direct
    # key access because older checkpoint files written before TS support
    # was added may not have the "ts_mask" key â€” .get() returns None
    # rather than raising KeyError in that case.

    status = "COMPLETE" if complete else "IN PROGRESS"
    label = "Last cached: " if complete else "Last updated:"
    # Two slightly different labels for the timestamp line to make it
    # obvious whether the build finished cleanly or is still running.

    lines = [
        f"Status:       {status}",
        f"{label}  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Cache path:   {cache_out}",
        f"Collected:    {len(samples)} / {sample_size} samples",
        f"  Benign:     {n_ben}",
        f"  Malignant:  {n_mal}",
        f"  TS masks:   {ts_count}/{len(samples)}",
    ]
    with open(os.path.join("cache", "cache_status.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # "w" = write text (overwrites each time), so the file always reflects
    # the current state, not a growing log.


def _load_checkpoint(path):
    # Returns the checkpoint dict {"samples": [...], "next_scan_idx": N}
    # if the checkpoint file exists, or None if it doesn't.
    # None is the signal to main() to start from scratch.
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


# =============================================================================
# MAIN â€” ORCHESTRATES THE ENTIRE CACHE BUILD
# =============================================================================
def main():

    # ------------------------------------------------------------------
    # Command-line argument parsing
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-size", type=int, default=_CFG_SAMPLE_SIZE,
        help="Override SAMPLE_SIZE from config (e.g. 200 for a quick demo run)"
        # Useful for smoke-testing the pipeline end-to-end quickly without
        # waiting for the full 2000-sample build.
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Ignore any existing checkpoint and start from scratch"
        # Without --fresh, any existing checkpoint is resumed automatically.
        # Use --fresh when you change the processing pipeline and need to
        # rebuild the cache from scratch, or to test with clean state.
    )
    parser.add_argument(
        "--no-ts", action="store_true",
        help="Skip TotalSegmentator â€” cache HU masks only (faster, for Exp 1)"
        # Allows building the cache on machines without a GPU or without
        # TotalSegmentator installed.  The TS mask entries will be None.
    )
    parser.add_argument(
        "--cache-path", type=str, default=None,
        help="Override cache output path (default: cache/cache.pkl)"
        # Lets you maintain multiple cache files side-by-side (e.g. a
        # small test cache and the full production cache).
    )
    parser.add_argument(
        "--nodule-crop-mm",
        type=float,
        default=None,
        help=(
            "Build nodule-centred square crops with this physical side length "
            "in mm instead of full CT slices (recommended: 128)"
        ),
    )
    args = parser.parse_args()
    if args.nodule_crop_mm is not None and args.nodule_crop_mm <= 0:
        parser.error("--nodule-crop-mm must be positive")

    SAMPLE_SIZE = args.sample_size
    # Local variable shadows the module-level constant when --sample-size
    # is passed.  Kept as a local so the module-level SAMPLE_SIZE stays
    # equal to the config value everywhere outside main().

    cache_out = args.cache_path if args.cache_path else CACHE_PATH
    # Use the CLI override if provided; otherwise fall back to config default.

    checkpoint_out = cache_out.replace(".pkl", "_checkpoint.pkl")
    # Derive the checkpoint path from the cache path so they are always
    # co-located and unambiguous.  E.g. "cache/cache.pkl" â†’
    # "cache/cache_checkpoint.pkl".

    # ------------------------------------------------------------------
    # Directory setup
    # ------------------------------------------------------------------
    os.makedirs("cache", exist_ok=True)
    # Ensure the "cache/" directory exists for cache_status.txt.
    # exist_ok=True suppresses the error if it already exists.

    os.makedirs(os.path.dirname(os.path.abspath(cache_out)), exist_ok=True)
    # Ensure the output directory for the cache file itself exists.
    # os.path.abspath converts a relative path to absolute so that
    # os.path.dirname never returns an empty string (which would make
    # makedirs fail).

    # ------------------------------------------------------------------
    # Device selection for TotalSegmentator
    # ------------------------------------------------------------------
    if args.no_ts:
        device = "cpu"
        # We still need to set device to something; "cpu" is harmless here
        # because _get_ts_vol_mask will never be called in --no-ts mode.
        print("[NO-TS] Skipping TotalSegmentator â€” HU masks only.")
    else:
        if not _TS_INSTALLED:
            # TS packages failed to import at module load time.  Print a
            # clear actionable message and exit cleanly.
            print("ERROR: TotalSegmentator is not installed. Run with --no-ts to build HU-only cache.")
            return
        device = "gpu" if _torch.cuda.is_available() else "cpu"
        # torch.cuda.is_available() returns True only if a CUDA-capable GPU
        # is present AND the correct CUDA drivers are installed.
        if device == "gpu":
            print(f"[GPU] TotalSegmentator will use CUDA: {_torch.cuda.get_device_name(0)}")
            # get_device_name(0) returns the human-readable name of the first
            # GPU (e.g. "NVIDIA GeForce RTX 3090"), useful for confirming
            # the right hardware is being used.
        else:
            print("[CPU] No CUDA detected â€” TotalSegmentator running on CPU (slow).")
            # TS on CPU is functional but can take 5â€“10 minutes per scan
            # compared to ~20 seconds on a modern GPU.  The warning prepares
            # the user for a long wait.

    print(f"[CACHE] Output â†’ {cache_out}")
    if args.nodule_crop_mm is None:
        print("[INPUT] Full axial CT slices.")
    else:
        print(
            f"[INPUT] Consensus nodule-centred crops: "
            f"{args.nodule_crop_mm:g} mm square."
        )

    # ------------------------------------------------------------------
    # Checkpoint resume logic
    # ------------------------------------------------------------------
    checkpoint = None if args.fresh else _load_checkpoint(checkpoint_out)
    # If --fresh was passed, we ignore any existing checkpoint by forcing
    # checkpoint to None.  Otherwise we try to load one from disk.

    if checkpoint:
        samples = checkpoint["samples"]
        start_scan_idx = checkpoint["next_scan_idx"]
        start_nod_idx = checkpoint.get("next_nod_idx", 0)
        # .get() with default 0 provides backwards compatibility: checkpoint
        # files written before per-nodule saving was added won't have the
        # "next_nod_idx" key, and defaulting to 0 is correct (start the
        # resuming scan from its first nodule group).
        print(f"[RESUME] Loaded checkpoint: {len(samples)} samples already collected, "
              f"resuming from scan {start_scan_idx}, nodule group {start_nod_idx}")
    else:
        samples = []
        start_scan_idx = 0
        start_nod_idx = 0
        # Fresh start: empty sample list, begin at the first scan.
        if args.fresh and os.path.exists(checkpoint_out):
            os.remove(checkpoint_out)
            # If --fresh was explicitly requested AND a checkpoint file exists,
            # delete it so a future run without --fresh doesn't accidentally
            # resume from the stale checkpoint.
            print("[FRESH] Deleted existing checkpoint, starting from scratch.")

    seen_image_hashes = {
        hashlib.sha256(np.ascontiguousarray(sample["image"]).tobytes()).hexdigest()
        for sample in samples
    }

    # ------------------------------------------------------------------
    # Load and shuffle all scans
    # ------------------------------------------------------------------
    scans = pl.query(pl.Scan).all()
    # pl.query(pl.Scan).all() returns a list of all pylidc Scan objects,
    # one per CT scan in the configured LIDC-IDRI database.
    # The order returned by the database is deterministic but reflects
    # the order scans were inserted â€” which is correlated with patient ID
    # and can introduce subtle biases (e.g. early LIDC patients may have
    # been acquired on older scanners).

    rng = np.random.default_rng(SEED)
    # Create a NumPy Generator (the modern NumPy RNG API, recommended over
    # the legacy np.random.seed() approach because it is stateless â€” the
    # same SEED always produces the same sequence regardless of what other
    # code has called np.random previously).

    scans = [scans[i] for i in rng.permutation(len(scans))]
    # rng.permutation(N) returns a shuffled array of integers 0..N-1.
    # We use list comprehension to reorder scans according to this permutation.
    # This ensures that if SAMPLE_SIZE < len(scans), the collected subset is
    # a random, unbiased sample of all available scans rather than just the
    # first N patients in the database.  Combined with the fixed SEED, the
    # shuffle is reproducible: every run with the same SEED produces the
    # same ordering.

    # ------------------------------------------------------------------
    # Main scan loop
    # ------------------------------------------------------------------
    for scan_idx, scan in enumerate(scans):
        # scan_idx: position in the shuffled scan list (0-based).
        # scan:     pylidc Scan object for this iteration.

        if scan_idx < start_scan_idx:
            # Skip scans already processed in a previous run (checkpoint
            # resume).  We iterate from 0 rather than slicing scans[start:]
            # so that scan_idx remains the correct absolute position for
            # the next checkpoint write.
            continue

        if len(samples) >= SAMPLE_SIZE:
            # Stop early once we have enough samples.  Checked here (before
            # loading the volume) to avoid doing unnecessary I/O.
            break

        print(f"Scan {scan_idx:04d} | patient {scan.patient_id} | cached so far: {len(samples)}/{SAMPLE_SIZE}")
        # :04d zero-pads the scan index to 4 digits so log lines sort
        # correctly when there are more than 99 scans.

        try:
            vol = scan.to_volume()
            # to_volume() reads all DICOM slices for this scan from disk,
            # assembles them into a 3-D NumPy array of shape (rows, cols, slices),
            # and applies any necessary rescale slope/intercept so that the
            # values are in Hounsfield Units.  This is the most expensive
            # operation per scan â€” it involves disk I/O for ~50â€“400 DICOM files.

            nod_groups = scan.cluster_annotations()
            # cluster_annotations() groups the individual radiologist
            # Annotation objects by spatial proximity: annotations from
            # different radiologists that overlap in 3-D space are grouped
            # together as describing the same physical nodule.
            # Returns a list of lists: [[ann1_rad1, ann1_rad2, ...], [ann2_rad1, ...], ...]
            # Each inner list is one nodule, containing one Annotation per
            # radiologist who marked it.

            # TotalSegmentator mask â€” computed once per scan (None if --no-ts)
            if args.no_ts:
                ts_vol_mask = None
                # Sentinel value: downstream code checks `is not None` before
                # trying to extract a 2-D slice.
            else:
                ts_vol_mask = _get_ts_vol_mask(vol, scan, device)
                # Returns a 3-D uint8 mask of the same spatial shape as vol,
                # or None if TS failed (see _get_ts_vol_mask).
                # Computed once here so the expensive TS inference is not
                # repeated for every nodule in this scan.

            # ----------------------------------------------------------
            # Nodule loop â€” one iteration per annotated nodule group
            # ----------------------------------------------------------
            for nod_idx, ann_group in enumerate(nod_groups):
                if scan_idx == start_scan_idx and nod_idx < start_nod_idx:
                    # Skip nodule groups that were already processed in a
                    # previous run.  This condition is only True for the one
                    # scan we were interrupted on (start_scan_idx); for all
                    # later scans scan_idx > start_scan_idx so the condition
                    # is always False and every nodule is processed normally.
                    # The volume and TS mask for this scan are re-computed
                    # above â€” that cost is unavoidable, but it only happens
                    # once for the single interrupted scan.
                    continue

                if len(samples) >= SAMPLE_SIZE:
                    # Check inside the nodule loop too, in case the last
                    # few nodules in a scan push us over the target.
                    break

                if len(ann_group) < MIN_RADIOLOGISTS:
                    # Discard nodules with fewer than MIN_RADIOLOGISTS
                    # (= 2) independent annotations.  A single radiologist's
                    # opinion is not reliable enough for a consensus label.
                    continue

                avg_mal = float(np.mean([a.malignancy for a in ann_group]))
                # Collect each radiologist's malignancy score (integer 1â€“5)
                # and average them.  np.mean returns float64; we cast to
                # Python float for clean comparisons.

                if avg_mal == MALIGNANCY_THRESHOLD:
                    # Exclude nodules with an average malignancy of exactly 3.
                    # These are genuinely ambiguous â€” radiologists disagree
                    # about whether the nodule leans benign or malignant.
                    # Including them with either label would introduce
                    # mislabelled training examples and degrade accuracy.
                    # This is the standard LIDC exclusion criterion.
                    continue

                label_val = 1 if avg_mal > MALIGNANCY_THRESHOLD else 0
                # Binary label:
                #   avg_mal > 3 â†’ malignant (1)
                #   avg_mal < 3 â†’ benign (0)
                # Because the == 3 case was already excluded above, this
                # covers all remaining cases.

                try:
                    # -------------------------------------------------------
                    # Slice selection
                    # -------------------------------------------------------
                    slice_idx = _get_middle_slice_idx(ann_group)
                    # Returns the z-index (slice number) of the middle annotated
                    # slice across all radiologists in the group (see function).

                    slice_idx = int(np.clip(slice_idx, 0, vol.shape[2] - 1))
                    # Safety clamp: image_k_position values come from the DICOM
                    # header and should always be valid slice indices, but an
                    # off-by-one or header inconsistency could push the index
                    # out of bounds.  np.clip ensures 0 â‰¤ slice_idx â‰¤ last slice.
                    # int() converts from numpy scalar to Python int for clean
                    # use as an array index.

                    raw_slice = vol[:, :, slice_idx].astype(np.float32)
                    # Extract the 2-D slice at the chosen z-index.
                    # vol[:, :, slice_idx] gives a (rows, cols) view;
                    # .astype(np.float32) copies it into a new float32 array
                    # to avoid accidentally modifying the original volume and
                    # to match the expected dtype for downstream processing.

                    # -------------------------------------------------------
                    # HU lung mask (computed on raw, unnormalised HU values)
                    # -------------------------------------------------------
                    lung_mask = _get_lung_mask(raw_slice)
                    # _get_lung_mask must receive raw HU values because its
                    # threshold (-500 HU) only makes physical sense in HU space.
                    # If we called it after normalisation, the threshold would
                    # be meaningless.  The order (mask before clip/normalise)
                    # is therefore intentional.

                    # -------------------------------------------------------
                    # HU clipping and normalisation
                    # -------------------------------------------------------
                    consensus_center = _get_consensus_nodule_center(ann_group)
                    crop_bounds = None
                    crop_side_pixels = None
                    if args.nodule_crop_mm is not None:
                        pixel_spacing = float(scan.pixel_spacing)
                        crop_side_pixels = max(
                            4, int(round(args.nodule_crop_mm / pixel_spacing))
                        )
                        raw_slice, crop_bounds = _extract_square(
                            raw_slice,
                            consensus_center,
                            crop_side_pixels,
                            HU_MIN,
                        )
                        lung_mask, _ = _extract_square(
                            lung_mask,
                            consensus_center,
                            crop_side_pixels,
                            0,
                        )

                    raw_slice = np.clip(raw_slice, HU_MIN, HU_MAX)
                    # Clamp extreme HU values.
                    #   Below HU_MIN (-1000): deep-space-black outside the FOV â€”
                    #     no diagnostic value, just padding artefacts.
                    #   Above HU_MAX (+400):  metal implants, reconstruction
                    #     artefacts (+1000 to +3000 HU) â€” would dominate the
                    #     normalised range and wash out the diagnostically
                    #     relevant [-1000, 400] window.
                    # After clipping, all values are in [-1000, 400].

                    raw_slice = (raw_slice - HU_MIN) / (HU_MAX - HU_MIN)
                    # Min-max normalisation to [0, 1]:
                    #   (x - min) / (max - min)
                    #   = (x - (-1000)) / (400 - (-1000))
                    #   = (x + 1000) / 1400
                    # -1000 HU (air) â†’ 0.0
                    # +400  HU (bone) â†’ 1.0
                    # This puts all scans on the same numerical scale regardless
                    # of scanner calibration differences, which is important for
                    # stable neural network training.

                    # -------------------------------------------------------
                    # Resize to 224Ã—224
                    # -------------------------------------------------------
                    zh = 224.0 / raw_slice.shape[0]
                    zw = 224.0 / raw_slice.shape[1]
                    # Compute the zoom factor for each axis.
                    # shape[0] = rows (height), shape[1] = cols (width).
                    # For a 512Ã—512 scan: zh = zw = 224/512 â‰ˆ 0.4375.
                    # Separating zh and zw handles non-square scans correctly.

                    image_224 = zoom(raw_slice, (zh, zw), order=1).astype(np.float32)
                    # order=1 = bilinear interpolation.
                    #   Bilinear is the standard choice for image downsampling:
                    #   it is smooth (no aliasing from nearest-neighbour) but
                    #   does not over-smooth (unlike bicubic/order=3).
                    # .astype(np.float32) ensures the output stays in float32
                    # (zoom may return float64).

                    mask_224 = zoom(lung_mask.astype(np.float32), (zh, zw), order=0)
                    mask_224 = (mask_224 > 0.5).astype(np.uint8)
                    # order=0 = nearest-neighbour interpolation for the mask.
                    #   Masks are binary (0/1); bilinear interpolation would
                    #   create fractional values at boundaries.  Nearest-neighbour
                    #   preserves the binary nature.
                    # We first convert to float32 before zooming because scipy's
                    # zoom is more numerically stable on float inputs than uint8.
                    # The > 0.5 threshold then re-binarises (nearest-neighbour
                    # on a 0/1 mask already guarantees values near 0 or 1, so
                    # 0.5 is an unambiguous threshold).

                    # -------------------------------------------------------
                    # TS mask: extract 2-D slice and resize
                    # -------------------------------------------------------
                    if ts_vol_mask is not None:
                        ts_slice = ts_vol_mask[:, :, slice_idx].astype(np.float32)
                        if args.nodule_crop_mm is not None:
                            ts_slice, _ = _extract_square(
                                ts_slice,
                                consensus_center,
                                crop_side_pixels,
                                0,
                            )
                        # Extract the same 2-D slice from the pre-computed 3-D
                        # TS mask.  Same slice_idx as the image â€” they must match.
                        ts_mask_224 = zoom(ts_slice, (zh, zw), order=0)
                        ts_mask_224 = (ts_mask_224 > 0.5).astype(np.uint8)
                        # Same resize logic as the HU mask: nearest-neighbour
                        # followed by re-binarisation.
                    else:
                        ts_mask_224 = None
                        # --no-ts was passed OR TS failed for this scan.
                        # Storing None rather than a zero array makes it
                        # explicit downstream that no TS mask is available
                        # (a zero array could be confused with a valid mask
                        # that happens to predict no lung tissue).

                    if not np.any(mask_224):
                        print(
                            f"  WARNING: skipped nodule with empty HU lung mask "
                            f"after cropping in {scan.patient_id}, group {nod_idx}"
                        )
                        _save_checkpoint(
                            samples, scan_idx, nod_idx + 1, checkpoint_out
                        )
                        continue

                    # -------------------------------------------------------
                    # Append the completed sample
                    # -------------------------------------------------------
                    image_hash = hashlib.sha256(
                        np.ascontiguousarray(image_224).tobytes()
                    ).hexdigest()
                    if image_hash in seen_image_hashes:
                        print(
                            f"  WARNING: skipped exact duplicate model input for "
                            f"{scan.patient_id}, nodule group {nod_idx}"
                        )
                        _save_checkpoint(
                            samples, scan_idx, nod_idx + 1, checkpoint_out
                        )
                        continue

                    _save_checkpoint(samples, scan_idx, nod_idx + 1, checkpoint_out)
                    # Checkpoint is saved BEFORE appending the sample.
                    # This ordering is deliberate: if the process is killed in
                    # the window between these two lines, the checkpoint already
                    # records nod_idx + 1, so on resume this nodule is skipped
                    # and the sample is lost (one missing entry).
                    # The alternative â€” append first, checkpoint second â€” risks
                    # a duplicate: the sample lands in memory but the checkpoint
                    # isn't updated, so on resume the nodule is re-processed and
                    # appended again.  A duplicate inflates that sample's weight
                    # during training; a single missing entry does not.
                    # In practice the window is microseconds and either outcome
                    # is extremely unlikely, but missing is safer than duplicating.
                    samples.append({
                        "image":      image_224,     # normalised CT slice, float32 224Ã—224
                        "mask":       mask_224,      # HU lung mask, uint8 224Ã—224
                        "ts_mask":    ts_mask_224,   # TS lung mask, uint8 224Ã—224 (or None)
                        "label":      label_val,     # 0=benign, 1=malignant
                        "patient_id": scan.patient_id,  # e.g. "LIDC-IDRI-0001"
                        "scan_id":     str(getattr(scan, "id", scan.patient_id)),
                        "nodule_id":   f"{scan.patient_id}:{getattr(scan, 'id', 'scan')}:{nod_idx}",
                        "slice_idx":   slice_idx,
                        "nodule_center_rc": consensus_center,
                        "crop_mode":   "nodule_centered" if args.nodule_crop_mm is not None else "full_slice",
                        "crop_side_mm": args.nodule_crop_mm,
                        "crop_side_pixels": crop_side_pixels,
                        "crop_bounds_rc": crop_bounds,
                    })
                    seen_image_hashes.add(image_hash)

                except Exception as e:
                    # A single nodule failing (e.g. a contour with no valid
                    # k-positions, or a degenerate zoom factor) should not
                    # abort the rest of the scan.  Skip and log.
                    print(f"  WARNING: skipped nodule in {scan.patient_id}: {e}")
                    continue

        except Exception as e:
            # A whole-scan failure (e.g. corrupted DICOM files, to_volume()
            # error) skips the entire scan.  The scan is not counted in
            # next_scan_idx, so a fresh run would encounter the same error â€”
            # this is intentional: we don't silently hide broken scans.
            print(f"  WARNING: skipped scan {scan.patient_id}: {e}")

        # ------------------------------------------------------------------
        # End-of-scan checkpoint
        # ------------------------------------------------------------------
        _save_checkpoint(samples, scan_idx + 1, 0, checkpoint_out)
        # Advances next_scan_idx past this scan and resets next_nod_idx to 0.
        # Without this, the last per-nodule save for this scan would store
        # (scan_idx, last_nod_idx + 1).  On resume that would cause the script
        # to reload this scan's volume and TS mask, skip all its nodules, then
        # move on â€” correct but wasteful.  Saving (scan_idx + 1, 0) here means
        # resume jumps straight to the next scan with no redundant work.
        _write_status(samples, cache_out, SAMPLE_SIZE)
        # Update the human-readable status file so progress is visible
        # even while the script is running.

    # ------------------------------------------------------------------
    # Write the final cache file
    # ------------------------------------------------------------------
    with open(cache_out, "wb") as f:
        pickle.dump(samples, f)
    # Pickle serialises the entire list of sample dicts (including NumPy
    # arrays) to a single binary file.  pickle.HIGHEST_PROTOCOL is used
    # by default, which is the most compact and fastest format available
    # in the running Python version.

    # Keep checkpoint so a future run with a larger SAMPLE_SIZE can resume from here
    # (use --fresh to discard it and start over)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    n_mal = sum(s["label"] for s in samples)
    n_ben = len(samples) - n_mal
    ts_count = sum(1 for s in samples if s.get("ts_mask") is not None)
    print(f"\nDone. Cached {len(samples)} nodule slices to {cache_out}")
    print(f"  {n_ben} benign, {n_mal} malignant | TS masks: {ts_count}/{len(samples)}")
    print(f"Now run baseline.py / train.py â€” pass --cache-path {cache_out} if not using default")

    _write_status(samples, cache_out, SAMPLE_SIZE, complete=True)
    # Write the final status file with complete=True so the status line
    # reads "COMPLETE" rather than "IN PROGRESS".
    print("Updated cache/cache_status.txt")


if __name__ == "__main__":
    main()
    # Standard Python idiom: only run main() when this script is executed
    # directly (python build_cache.py), not when it is imported as a module
    # by another script.  This allows functions like _get_lung_mask to be
    # imported and unit-tested without triggering the full cache build.
