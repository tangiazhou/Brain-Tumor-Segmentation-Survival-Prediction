"""
Stage 3: Feature Extraction
Extracts geometric, shape, texture, and intensity features from segmented
tumor regions. Returns a flat dict suitable for DataFrame rows.
"""

import numpy as np
from skimage.measure import regionprops, label
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import entropy as scipy_entropy


# ---------------------------------------------------------------------------
# Geometric & Shape Features
# ---------------------------------------------------------------------------

def geometric_features(mask: np.ndarray, pixel_spacing: float = 1.0) -> dict:
    """
    Compute area, perimeter, and bounding box dimensions from binary mask.

    Args:
        mask:           2D binary mask (0/1).
        pixel_spacing:  Physical size of one pixel (mm). Defaults to 1.0.

    Returns:
        dict with keys: area_px, area_mm2, perimeter, bbox_height, bbox_width
    """
    labeled = label(mask)
    props = regionprops(labeled)

    if not props:
        return {k: 0.0 for k in
                ["area_px", "area_mm2", "perimeter", "bbox_height", "bbox_width"]}

    # Use the largest region
    largest = max(props, key=lambda r: r.area)
    minr, minc, maxr, maxc = largest.bbox

    return {
        "area_px": float(largest.area),
        "area_mm2": float(largest.area) * pixel_spacing ** 2,
        "perimeter": float(largest.perimeter),
        "bbox_height": float(maxr - minr),
        "bbox_width": float(maxc - minc),
    }


def shape_features(mask: np.ndarray) -> dict:
    """
    Compute shape descriptors: compactness, eccentricity, solidity, extent.

    Compactness = 4π·area / perimeter²  (circle = 1, more irregular < 1)
    """
    labeled = label(mask)
    props = regionprops(labeled)

    if not props:
        return {k: 0.0 for k in
                ["compactness", "eccentricity", "solidity", "extent"]}

    largest = max(props, key=lambda r: r.area)
    perimeter = largest.perimeter
    area = largest.area

    compactness = (4 * np.pi * area / perimeter ** 2) if perimeter > 0 else 0.0

    return {
        "compactness": float(compactness),
        "eccentricity": float(largest.eccentricity),
        "solidity": float(largest.solidity),
        "extent": float(largest.extent),
    }


# ---------------------------------------------------------------------------
# Texture Features (GLCM)
# ---------------------------------------------------------------------------

def _quantize(image: np.ndarray, n_levels: int = 32) -> np.ndarray:
    """Quantize a float image to [0, n_levels-1] integers."""
    img_min, img_max = image.min(), image.max()
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.uint8)
    quantized = ((image - img_min) / (img_max - img_min) * (n_levels - 1))
    return quantized.astype(np.uint8)


def glcm_features(image: np.ndarray, mask: np.ndarray,
                  distances: list = None, angles: list = None,
                  n_levels: int = 32) -> dict:
    """
    Compute GLCM-based texture features on the masked tumor region.

    Features: contrast, dissimilarity, homogeneity, energy, correlation, ASM.
    Averaged over provided distances and angles.

    Args:
        image:    2D float image (single MRI modality slice).
        mask:     2D binary mask selecting the tumor region.
        distances: GLCM pixel-pair distances. Default: [1, 2].
        angles:   GLCM directions (rad). Default: 0, 45, 90, 135 deg.
        n_levels: Quantization levels.
    """
    if distances is None:
        distances = [1, 2]
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Crop to tumor bounding box for efficiency
    rows, cols = np.where(mask > 0)
    if len(rows) == 0:
        props = ["contrast", "dissimilarity", "homogeneity",
                 "energy", "correlation", "ASM"]
        return {f"glcm_{p}": 0.0 for p in props}

    rmin, rmax = rows.min(), rows.max() + 1
    cmin, cmax = cols.min(), cols.max() + 1
    roi = image[rmin:rmax, cmin:cmax]
    roi_q = _quantize(roi, n_levels)

    glcm = graycomatrix(roi_q, distances=distances, angles=angles,
                        levels=n_levels, symmetric=True, normed=True)

    props_names = ["contrast", "dissimilarity", "homogeneity",
                   "energy", "correlation", "ASM"]
    features = {}
    for prop in props_names:
        val = graycoprops(glcm, prop).mean()
        features[f"glcm_{prop}"] = float(val)

    return features


def entropy_feature(image: np.ndarray, mask: np.ndarray,
                    n_levels: int = 32) -> dict:
    """Shannon entropy of intensity histogram within tumor region."""
    pixels = image[mask > 0]
    if len(pixels) == 0:
        return {"entropy": 0.0}
    quantized = _quantize(pixels.reshape(-1, 1).squeeze(), n_levels)
    hist, _ = np.histogram(quantized, bins=n_levels, range=(0, n_levels - 1))
    hist = hist / (hist.sum() + 1e-8)
    ent = scipy_entropy(hist + 1e-8)
    return {"entropy": float(ent)}


# ---------------------------------------------------------------------------
# Intensity Statistics
# ---------------------------------------------------------------------------

def intensity_features(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute intensity statistics over masked tumor pixels:
    mean, std (variance), min, max, median, skewness, kurtosis.
    """
    pixels = image[mask > 0].astype(float)
    if len(pixels) == 0:
        return {k: 0.0 for k in
                ["intensity_mean", "intensity_std", "intensity_var",
                 "intensity_min", "intensity_max", "intensity_median",
                 "intensity_skew", "intensity_kurt"]}

    mean = pixels.mean()
    std = pixels.std()
    centered = pixels - mean
    skew = (centered ** 3).mean() / (std ** 3 + 1e-8)
    kurt = (centered ** 4).mean() / (std ** 4 + 1e-8)

    return {
        "intensity_mean": float(mean),
        "intensity_std": float(std),
        "intensity_var": float(std ** 2),
        "intensity_min": float(pixels.min()),
        "intensity_max": float(pixels.max()),
        "intensity_median": float(np.median(pixels)),
        "intensity_skew": float(skew),
        "intensity_kurt": float(kurt),
    }


# ---------------------------------------------------------------------------
# Aggregate all features
# ---------------------------------------------------------------------------

def extract_all_features(image: np.ndarray, mask: np.ndarray,
                          pixel_spacing: float = 1.0) -> dict:
    """
    Extract the full feature vector from a single 2D MRI slice and its mask.

    Args:
        image:         2D normalized MRI slice (single modality).
        mask:          2D binary segmentation mask.
        pixel_spacing: Physical pixel size in mm.

    Returns:
        Flat dict of all features.
    """
    features = {}
    features.update(geometric_features(mask, pixel_spacing))
    features.update(shape_features(mask))
    features.update(glcm_features(image, mask))
    features.update(entropy_feature(image, mask))
    features.update(intensity_features(image, mask))
    return features
