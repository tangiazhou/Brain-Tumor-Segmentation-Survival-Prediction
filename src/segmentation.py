"""
Stage 2: Tumor Segmentation
- Intensity thresholding (Otsu + fixed)
- Region growing (flood fill from seed)
- Morphological post-processing (closing, hole filling)
- Multi-modal fusion (FLAIR + T1ce + T2)
- Watershed segmentation

Channel order (verified from channel_check.png):
    index 0 = T1ce
    index 1 = T1
    index 2 = FLAIR
    index 3 = T2
"""

import numpy as np
from skimage.filters import threshold_otsu, sobel
from skimage.morphology import (closing, remove_small_objects, disk)
from skimage.segmentation import flood, watershed
from scipy.ndimage import binary_fill_holes, label

# Verified channel indices
T1CE_IDX  = 0
T1_IDX    = 1
FLAIR_IDX = 2
T2_IDX    = 3


# ---------------------------------------------------------------------------
# Thresholding
# ---------------------------------------------------------------------------

def threshold_otsu_segment(image: np.ndarray) -> np.ndarray:
    """
    Binary segmentation using Otsu's global threshold.
    Only operates on non-zero (brain) pixels to avoid skull-strip background.
    """
    brain_mask = image > 0
    if brain_mask.sum() == 0:
        return np.zeros_like(image, dtype=bool)
    thresh = threshold_otsu(image[brain_mask])
    return (image > thresh) & brain_mask


def threshold_fixed(image: np.ndarray, low: float = 1.5,
                    high: float = None) -> np.ndarray:
    """
    Fixed threshold: pixels above `low` stddevs above mean of brain region.
    """
    brain_mask = image > 0
    if brain_mask.sum() == 0:
        return np.zeros_like(image, dtype=bool)
    brain_pixels = image[brain_mask]
    mu, sigma = brain_pixels.mean(), brain_pixels.std()
    lower_bound = mu + low * sigma
    seg = (image > lower_bound) & brain_mask
    if high is not None:
        seg &= (image < mu + high * sigma)
    return seg


# ---------------------------------------------------------------------------
# Region Growing
# ---------------------------------------------------------------------------

def _find_seed(image: np.ndarray, binary_mask: np.ndarray):
    labeled, n = label(binary_mask)
    if n == 0:
        return None
    sizes = [(labeled == i).sum() for i in range(1, n + 1)]
    largest = np.argmax(sizes) + 1
    region = labeled == largest
    vals = image * region
    seed = np.unravel_index(vals.argmax(), vals.shape)
    return seed


def region_growing(image: np.ndarray, seed=None,
                   tolerance: float = 0.5) -> np.ndarray:
    """Simple flood-fill region growing."""
    if seed is None:
        otsu_mask = threshold_otsu_segment(image)
        seed = _find_seed(image, otsu_mask)
    if seed is None:
        return np.zeros_like(image, dtype=bool)
    flooded = flood(image, seed, tolerance=tolerance)
    brain_mask = image > 0
    return flooded & brain_mask


# ---------------------------------------------------------------------------
# Morphological post-processing
# ---------------------------------------------------------------------------

def morphological_cleanup(binary_mask: np.ndarray,
                           close_radius: int = 3,
                           min_size: int = 50,
                           fill_holes: bool = True,
                           max_components: int = 2) -> np.ndarray:
    """
    Clean up a binary segmentation mask:
      1. Binary closing to connect nearby blobs.
      2. Remove small spurious objects.
      3. Fill internal holes.
      4. Keep only the top N largest connected components to eliminate
         vascular false positives and scattered noise.

    Args:
        binary_mask:     Input boolean 2D array.
        close_radius:    Disk radius for morphological closing.
        min_size:        Minimum object size in pixels to keep.
        fill_holes:      Whether to fill holes inside objects.
        max_components:  Maximum number of connected components to keep.
                         Brain tumors are typically 1-2 connected regions.
    """
    mask = binary_mask.astype(bool)
    selem = disk(close_radius)
    mask = closing(mask, selem)
    mask = remove_small_objects(mask, max_size=min_size)
    if fill_holes:
        mask = binary_fill_holes(mask)

    # Keep only the top N largest connected components
    labeled, n = label(mask)
    if n > max_components:
        sizes = [(labeled == i).sum() for i in range(1, n + 1)]
        top_n = np.argsort(sizes)[-max_components:] + 1
        mask  = np.isin(labeled, top_n)

    return mask.astype(np.uint8)


# ---------------------------------------------------------------------------
# Multi-modal segmentation (FLAIR + T1ce + T2 fusion)
# ---------------------------------------------------------------------------

def segment_multimodal(image: np.ndarray) -> np.ndarray:
    """
    Combine FLAIR, T1ce, and T2 for whole-tumor coverage:
    - FLAIR: captures peritumoral edema (outer ring)
    - T1ce:  captures enhancing tumor core (bright focal spot)
    - T2:    captures additional edema and necrotic regions

    False positive reduction:
    - Top-2 components: keep only the 2 largest connected regions,
      eliminating scattered vascular false positives

    Args:
        image: (H, W, 4) normalized array
               channels: T1ce=0, T1=1, FLAIR=2, T2=3
    """
    flair = image[..., FLAIR_IDX]
    t1ce  = image[..., T1CE_IDX]
    t2    = image[..., T2_IDX]

    flair_mask = threshold_fixed(flair, low=1.5)
    t1ce_mask  = threshold_fixed(t1ce,  low=1.5)
    t2_mask    = threshold_fixed(t2,    low=1.5)

    combined = (flair_mask | t1ce_mask | t2_mask).astype(np.uint8)

    # max_components=2 keeps up to 2 largest regions (tumor can have
    # separated necrotic core + enhancing rim)
    return morphological_cleanup(combined, max_components=2)


# ---------------------------------------------------------------------------
# Watershed segmentation
# ---------------------------------------------------------------------------

def segment_watershed(image: np.ndarray) -> np.ndarray:
    """
    Watershed segmentation using multi-modal thresholding as markers.
    Falls back to segment_multimodal if markers are insufficient.

    Args:
        image: (H, W, 4) normalized array
               channels: T1ce=0, T1=1, FLAIR=2, T2=3
    """
    flair = image[..., FLAIR_IDX]
    t1ce  = image[..., T1CE_IDX]

    t1ce_fg  = threshold_fixed(t1ce,  low=1.5).astype(bool)
    flair_fg = threshold_fixed(flair, low=2.0).astype(bool)
    sure_fg  = (t1ce_fg | flair_fg)

    sure_bg = (~threshold_fixed(flair, low=0.3).astype(bool)) & (flair == 0)

    if sure_fg.sum() < 10:
        return segment_multimodal(image)

    markers = np.zeros(flair.shape, dtype=np.int32)
    markers[sure_bg] = 1
    markers[sure_fg] = 2

    gradient = sobel(flair)
    result   = watershed(gradient, markers)

    mask = (result == 2).astype(np.uint8)
    return morphological_cleanup(mask, max_components=2)