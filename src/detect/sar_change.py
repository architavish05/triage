"""Sentinel-1 SAR change detection for flood mapping.

Includes radiometric calibration, Lee speckle filtering, decibel (dB) conversion,
and bimodal Otsu / adaptive thresholding on post-event backscatter and pre/post difference.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


@dataclass
class SARWaterDetectionResult:
    """Output of SAR water change detection."""
    water_mask: np.ndarray          # Boolean mask (True where flood/water detected)
    water_confidence: np.ndarray    # Float array in [0.0, 1.0]
    post_db: np.ndarray             # Preprocessed post-event dB grid
    pre_db: np.ndarray              # Preprocessed pre-event dB grid
    diff_db: np.ndarray             # post_db - pre_db difference grid
    post_threshold: float           # Threshold applied to post-event dB
    diff_threshold: float           # Threshold applied to diff dB


def to_db(linear_array: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Convert linear SAR intensity to decibels (dB).

    dB = 10 * log10(max(intensity, eps))
    """
    clipped = np.maximum(linear_array, eps)
    return 10.0 * np.log10(clipped)


def lee_filter(img: np.ndarray, window_size: int = 5, num_looks: float = 1.0) -> np.ndarray:
    """Apply Lee speckle filter on SAR intensity/dB raster.

    Uses local mean and variance to adaptively smooth speckle while preserving edges.

    Args:
        img: 2D numpy array of SAR backscatter.
        window_size: Size of sliding local window (must be odd).
        num_looks: Equivalent number of looks (ENL) of the SAR product.

    Returns:
        Filtered 2D array.
    """
    if window_size % 2 == 0:
        window_size += 1

    mean_local = uniform_filter(img, size=window_size)
    mean_sqr_local = uniform_filter(img ** 2, size=window_size)
    var_local = np.maximum(mean_sqr_local - (mean_local ** 2), 0.0)

    # Theoretical noise variance for multiplicative speckle
    noise_var = (mean_local ** 2) / max(num_looks, 1.0)

    # Filter weighting factor W
    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(var_local > 0, np.maximum(0.0, 1.0 - (noise_var / var_local)), 0.0)
    weight = np.clip(weight, 0.0, 1.0)

    filtered = mean_local + weight * (img - mean_local)
    return filtered.astype(np.float32)


def otsu_threshold(data: np.ndarray, num_bins: int = 256) -> float:
    """Compute Otsu threshold maximizing between-class variance.

    Args:
        data: 1D or 2D array of continuous values.
        num_bins: Number of histogram bins.

    Returns:
        Optimal separation threshold value.
    """
    clean_data = data[np.isfinite(data)]
    if clean_data.size == 0:
        return -16.0

    counts, bin_edges = np.histogram(clean_data, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = clean_data.size

    weight_bg = np.cumsum(counts) / total
    weight_fg = 1.0 - weight_bg

    mean_bg = np.cumsum(counts * bin_centers) / np.maximum(np.cumsum(counts), 1)
    total_mean = np.sum(counts * bin_centers) / total
    mean_fg = (total_mean - weight_bg * mean_bg) / np.maximum(weight_fg, 1e-9)

    between_variance = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
    max_val = np.max(between_variance)
    candidates = np.where(between_variance >= max_val - 1e-6)[0]
    best_idx = candidates[len(candidates) // 2]
    return float(bin_centers[best_idx])


def preprocess_sar(
    array: np.ndarray,
    already_db: bool = False,
    window_size: int = 5,
) -> np.ndarray:
    """Preprocess SAR backscatter: ensure dB scale and filter speckle."""
    db_array = array if already_db else to_db(array)
    return lee_filter(db_array, window_size=window_size)


def detect_water_sar(
    post_sar: np.ndarray,
    pre_sar: np.ndarray,
    already_db: bool = True,
    post_threshold: Optional[float] = None,
    diff_threshold: float = -3.5,
    window_size: int = 5,
) -> SARWaterDetectionResult:
    """Detect water and calculate continuous flood confidence via SAR change.

    Water surface acts as a specular reflector:
      Rule: (post_db <= post_threshold) AND (post_db - pre_db <= diff_threshold)

    Confidence is computed continuously:
      Confidence scales higher as post_db drops deeper below threshold and
      the negative drop (pre - post) deepens.

    Args:
        post_sar: Post-flood SAR backscatter grid.
        pre_sar: Pre-flood SAR backscatter grid (same orbit).
        already_db: True if input is already calibrated to decibels.
        post_threshold: dB cutoff for water. If None, computed adaptively via Otsu.
        diff_threshold: Minimum required backscatter decrease in dB (e.g. -3.5 dB).
        window_size: Window size for speckle filter.

    Returns:
        SARWaterDetectionResult containing binary mask, continuous confidence, and grids.
    """
    post_db = preprocess_sar(post_sar, already_db=already_db, window_size=window_size)
    pre_db = preprocess_sar(pre_sar, already_db=already_db, window_size=window_size)

    diff_db = post_db - pre_db

    # Determine post-flood water threshold
    if post_threshold is None:
        computed_otsu = otsu_threshold(post_db)
        # Cap threshold within physically reasonable range for water (-22 to -14 dB)
        post_threshold = float(np.clip(computed_otsu, -22.0, -14.0))
        logger.info(f"Otsu-derived SAR water threshold: {post_threshold:.2f} dB")
    else:
        logger.info(f"Using fixed SAR water threshold: {post_threshold:.2f} dB")

    # Hard binary detection criteria
    is_post_low = post_db <= post_threshold
    is_diff_strong = diff_db <= diff_threshold
    water_mask = is_post_low & is_diff_strong

    # Continuous confidence calculation:
    # 1. Post score: 0 at threshold, 1.0 at threshold - 8 dB
    post_margin = (post_threshold - post_db) / 8.0
    post_score = np.clip(post_margin, 0.0, 1.0)

    # 2. Diff score: 0 at diff_threshold, 1.0 at diff_threshold - 6 dB
    diff_margin = (diff_threshold - diff_db) / 6.0
    diff_score = np.clip(diff_margin, 0.0, 1.0)

    # Fused confidence in [0, 1]: geometric mean of both evidences
    confidence = np.sqrt(post_score * diff_score)
    # Ensure pixels failing hard binary rule have low/zero confidence
    confidence = np.where(water_mask, np.clip(confidence, 0.3, 1.0), 0.0).astype(np.float32)

    return SARWaterDetectionResult(
        water_mask=water_mask,
        water_confidence=confidence,
        post_db=post_db,
        pre_db=pre_db,
        diff_db=diff_db,
        post_threshold=post_threshold,
        diff_threshold=diff_threshold,
    )
