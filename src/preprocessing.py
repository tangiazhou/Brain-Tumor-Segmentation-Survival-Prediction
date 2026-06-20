"""
Stage 1: Preprocessing
- Load pre-sliced HDF5 files (keys: 'image', 'mask')
- Each file is one 2D slice: image shape (240, 240, 4), mask shape (240, 240, 3)
- Intensity normalization (z-score per channel, non-zero voxels only)
- Filtering of near-empty slices

File naming convention: volume_{vol_id}_slice_{slice_id}.h5
  vol_id   : 1 – 369
  slice_id : 0 – 154
"""

import os
import re
import numpy as np
import h5py


# Channel indices inside the 'image' array (4 MRI modalities)
CHANNEL_NAMES = ["t1ce", "t1", "flair", "t2"]

# Channel indices inside the 'mask' array (3 tumor sub-regions)
MASK_CHANNEL_NAMES = ["NCR_NET", "ED", "ET"]      # assumed order


# ---------------------------------------------------------------------------
# Single-slice loading
# ---------------------------------------------------------------------------

def load_h5_slice(path: str):
    """
    Load one HDF5 slice file.

    Returns:
        image : float32 array of shape (240, 240, 4)  — 4 MRI modalities
        mask  : uint8  array of shape (240, 240, 3)   — 3 tumor sub-regions
    """
    with h5py.File(path, "r") as f:
        image = f["image"][()].astype(np.float32)
        mask = f["mask"][()].astype(np.uint8)
    return image, mask


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Collapse multi-channel mask to a single binary whole-tumor mask.
    Any channel > 0 counts as tumor.

    Args:
        mask: (H, W, C) or (H, W) uint8 array.

    Returns:
        (H, W) uint8 binary mask.
    """
    if mask.ndim == 3:
        return (mask.max(axis=-1) > 0).astype(np.uint8)
    return (mask > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_slice(image: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each modality channel independently,
    computed only over non-zero (brain) pixels.

    Args:
        image: (H, W, 4) float32 array.

    Returns:
        Normalized (H, W, 4) float32 array; zero pixels stay 0.
    """
    normalized = np.zeros_like(image)
    for c in range(image.shape[-1]):
        ch = image[..., c]
        brain = ch > 0
        if brain.sum() == 0:
            continue
        mu = ch[brain].mean()
        sigma = ch[brain].std()
        out = np.zeros_like(ch)
        out[brain] = (ch[brain] - mu) / (sigma + 1e-8)
        normalized[..., c] = out
    return normalized


# ---------------------------------------------------------------------------
# Slice filtering
# ---------------------------------------------------------------------------

def has_tumor(mask: np.ndarray, min_tumor_fraction: float = 0.001) -> bool:
    """Return True if the binary mask contains enough tumor pixels."""
    binary = binarize_mask(mask)
    return binary.sum() / binary.size >= min_tumor_fraction


def has_brain(image: np.ndarray, min_brain_fraction: float = 0.01) -> bool:
    """Return True if at least one modality has enough non-zero pixels."""
    return (image > 0).any(axis=-1).mean() >= min_brain_fraction


# ---------------------------------------------------------------------------
# Dataset-level helpers
# ---------------------------------------------------------------------------

def parse_filename(filename: str):
    """
    Extract (volume_id, slice_id) from 'volume_{v}_slice_{s}.h5'.
    Returns (int, int) or (None, None) if pattern doesn't match.
    """
    m = re.match(r"volume_(\d+)_slice_(\d+)\.h5", os.path.basename(filename))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def list_slices(slices_dir: str, volume_ids: list = None) -> list:
    """
    Return sorted list of .h5 file paths in slices_dir.
    Optionally filter to a specific list of volume IDs.

    Args:
        slices_dir:  Directory containing volume_*_slice_*.h5 files.
        volume_ids:  If given, only return files for these volume numbers.

    Returns:
        List of absolute file paths, sorted by (volume_id, slice_id).
    """
    files = []
    for fname in os.listdir(slices_dir):
        if not fname.endswith(".h5"):
            continue
        vol_id, slice_id = parse_filename(fname)
        if vol_id is None:
            continue
        if volume_ids is not None and vol_id not in volume_ids:
            continue
        files.append((vol_id, slice_id, os.path.join(slices_dir, fname)))
    files.sort()
    return [f[2] for f in files]


def load_volume_slices(slices_dir: str, volume_id: int,
                        normalize: bool = True,
                        tumor_only: bool = True,
                        target_index: dict = None) -> list:
    """
    Load all slices for a single volume, with optional filtering.

    Args:
        slices_dir:    Directory containing .h5 files.
        volume_id:     Integer volume ID (1-369).
        normalize:     Apply z-score normalization per channel.
        tumor_only:    Only return slices that contain tumor pixels.
        target_index:  Optional dict {(vol_id, slice_id): target} from
                       build_target_index(). If provided, uses the
                       pre-computed target column from meta_data.csv
                       for filtering instead of recomputing from mask.

    Returns:
        List of dicts: [{'image': arr, 'mask': arr, 'slice_id': int}, ...]
    """
    paths = list_slices(slices_dir, volume_ids=[volume_id])
    results = []
    for path in paths:
        _, slice_id = parse_filename(path)
        image, mask = load_h5_slice(path)

        if not has_brain(image):
            continue

        if tumor_only:
            if target_index is not None:
                # Use pre-computed target from meta_data.csv (faster + reliable)
                if target_index.get((volume_id, slice_id), 0) == 0:
                    continue
            else:
                # Fall back to computing from mask array
                if not has_tumor(mask):
                    continue

        if normalize:
            image = normalize_slice(image)

        results.append({
            "image": image,       # (240, 240, 4)
            "mask": mask,         # (240, 240, 3)
            "slice_id": slice_id,
            "volume_id": volume_id,
            "path": path,
        })
    return results


def get_channel(image: np.ndarray, modality: str) -> np.ndarray:
    """
    Extract a single modality channel from an image slice.

    Args:
        image:    (H, W, 4) array.
        modality: One of 't1', 't1ce', 't2', 'flair'.

    Returns:
        (H, W) float32 array.
    """
    idx = CHANNEL_NAMES.index(modality)
    return image[..., idx]


def build_target_index(meta_csv_path: str) -> dict:
    """
    Build a lookup dict from meta_data.csv for fast tumor-slice filtering.

    Returns:
        dict: {(volume_id, slice_id): target} where target is 0 or 1.
    """
    import pandas as pd
    df = pd.read_csv(meta_csv_path)
    index = {}
    for _, row in df.iterrows():
        m = re.search(r'volume_(\d+)_slice_(\d+)', str(row['slice_path']))
        if m:
            index[(int(m.group(1)), int(m.group(2)))] = int(row['target'])
    return index
