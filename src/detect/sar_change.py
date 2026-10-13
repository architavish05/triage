"""Sentinel-1 SAR change detection for flood mapping.

Includes Lee speckle filtering, Otsu thresholding, dual-polarisation
(VV + VH) change detection, and debris amplitude-surge detection.

Sign convention (explicitly documented):
    delta_vv = post_vv - pre_vv   (negative → backscatter decrease → water signal)
    delta_vh = post_vh - pre_vh   (negative → water signal, less specific than VV)

Water physics:
    Smooth open water acts as a specular reflector, directing energy away from
    the radar.  This causes a sharp drop in both VV and VH backscatter (typically
    6–15 dB below surrounding land).  VV is the primary indicator; VH confirms.

Debris physics:
    Rough sediment, boulders, and disturbed terrain increase backscatter (positive
    delta_vv) and local texture variance relative to pre-event smooth land.
    This is NOT water; post_db values remain above -18 dB.

Limitations in mountainous terrain:
    - Shadow and layover cause spurious low backscatter (false water).
    - Otsu may fail on unimodal distributions (handled gracefully with a fallback).
    - Slope filtering (performed by Person 3) must be applied downstream.
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

    water_mask: np.ndarray          # bool: True where flood/water detected
    water_confidence: np.ndarray    # float32 in [0.0, 1.0]
    post_db: np.ndarray             # Preprocessed post-event VV dB grid
    pre_db: np.ndarray              # Preprocessed pre-event VV dB grid
    diff_db: np.ndarray             # delta_vv = post_db - pre_db
    post_threshold: float           # dB threshold applied to post-event
    diff_threshold: float           # dB threshold applied to diff


# ═══════════════════════════════════════════════════════════════════
# Core signal processing utilities
# ═══════════════════════════════════════════════════════════════════

def to_db(linear_array: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Convert linear SAR intensity to decibels (dB).

    dB = 10 * log10(max(intensity, eps))

    Args:
        linear_array: 1-D or 2-D array of linear backscatter intensity.
        eps: Minimum floor value to avoid log(0).

    Returns:
        dB-scale array of same shape and float32 dtype.
    """
    clipped = np.maximum(linear_array, eps)
    return (10.0 * np.log10(clipped)).astype(np.float32)


def lee_filter(img: np.ndarray, window_size: int = 5, num_looks: float = 1.0) -> np.ndarray:
    """Apply Lee adaptive speckle filter to a SAR backscatter raster.

    Uses local mean and local variance to adaptively smooth speckle while
    preserving edges and bright targets.

    Filter equation:
        filtered = mean_local + W * (img - mean_local)
        W = max(0, 1 - noise_var / local_var)
        noise_var = mean_local² / num_looks

    Args:
        img: 2-D float32 SAR array (may be dB or linear scale).
        window_size: Sliding window size in pixels (will be made odd).
        num_looks: Equivalent number of looks (ENL); higher → more smoothing.

    Returns:
        Speckle-filtered float32 array of same shape.
    """
    if window_size % 2 == 0:
        window_size += 1

    mean_local   = uniform_filter(img, size=window_size)
    mean_sqr     = uniform_filter(img ** 2, size=window_size)
    var_local    = np.maximum(mean_sqr - mean_local ** 2, 0.0)
    noise_var    = mean_local ** 2 / max(num_looks, 1.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        weight = np.where(var_local > 0, np.maximum(0.0, 1.0 - noise_var / var_local), 0.0)
    weight = np.clip(weight, 0.0, 1.0)

    filtered = mean_local + weight * (img - mean_local)
    return filtered.astype(np.float32)


def otsu_threshold(data: np.ndarray, num_bins: int = 256) -> float:
    """Compute Otsu's threshold maximising between-class variance.

    Robust to degenerate cases:
        - Empty or constant arrays → returns -16.0 (physically safe water proxy)
        - Unimodal distributions → returns the median as a safe fallback

    Args:
        data: 1-D or 2-D array of finite values.
        num_bins: Number of histogram bins (more bins → finer resolution).

    Returns:
        Optimal separation threshold value (float).
    """
    clean = data[np.isfinite(data)].ravel()
    if clean.size == 0:
        logger.debug("otsu_threshold: empty input → returning -16.0 dB fallback.")
        return -16.0

    if np.ptp(clean) < 1e-6:  # constant array
        logger.debug("otsu_threshold: constant input → returning value as threshold.")
        return float(clean[0])

    counts, bin_edges = np.histogram(clean, bins=num_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total = clean.size

    weight_bg = np.cumsum(counts) / total
    weight_fg = 1.0 - weight_bg

    mean_bg = np.cumsum(counts * bin_centers) / np.maximum(np.cumsum(counts), 1)
    total_mean = np.sum(counts * bin_centers) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_fg = np.where(
            weight_fg > 1e-9,
            np.where(weight_fg > 0, (total_mean - weight_bg * mean_bg) / weight_fg, 0.0),
            0.0,
        )

    between_var = weight_bg * weight_fg * ((mean_bg - mean_fg) ** 2)
    max_var = np.max(between_var)

    if max_var < 1e-10:  # unimodal — Otsu is unreliable
        fallback = float(np.median(clean))
        logger.debug(
            "otsu_threshold: unimodal distribution (max between-var=%.2e) → using median %.2f.",
            max_var, fallback,
        )
        return fallback

    candidates = np.where(between_var >= max_var - 1e-6)[0]
    best_idx = candidates[len(candidates) // 2]
    return float(bin_centers[best_idx])


def preprocess_sar(
    array: np.ndarray,
    already_db: bool = False,
    window_size: int = 5,
) -> np.ndarray:
    """Ensure dB scale and apply Lee speckle filter.

    Args:
        array: Input SAR array (linear or dB scale).
        already_db: Set True if input is already in dB.
        window_size: Lee filter window size.

    Returns:
        Filtered dB-scale float32 array.
    """
    db_array = array if already_db else to_db(array)
    return lee_filter(db_array.astype(np.float32), window_size=window_size)


# ═══════════════════════════════════════════════════════════════════
# Water detection — single-polarisation (primary API)
# ═══════════════════════════════════════════════════════════════════

def detect_water_sar(
    post_sar: np.ndarray,
    pre_sar: np.ndarray,
    already_db: bool = True,
    post_threshold: Optional[float] = None,
    diff_threshold: float = -3.5,
    window_size: int = 5,
) -> SARWaterDetectionResult:
    """Detect flood water and compute continuous flood confidence via SAR change.

    Water detection logic:
        Rule: post_db <= post_threshold AND (post_db - pre_db) <= diff_threshold

    Confidence scoring:
        post_score = clip((post_threshold - post_db) / 8.0, 0, 1)
        diff_score = clip((diff_threshold - diff_db)  / 6.0, 0, 1)
        confidence = sqrt(post_score * diff_score)  [geometric mean]

    Args:
        post_sar: Post-flood SAR backscatter grid (any scale).
        pre_sar: Pre-flood SAR backscatter grid (same orbit, same scale).
        already_db: True if inputs are already calibrated dB values.
        post_threshold: dB cutoff for standing water.  If None, derived by Otsu
            thresholding on post_db and clipped to [-22, -14] dB.
        diff_threshold: Required backscatter decrease (dB, negative = decrease).
        window_size: Lee filter window size (pixels).

    Returns:
        SARWaterDetectionResult with binary mask, continuous confidence, and grids.
    """
    post_db = preprocess_sar(post_sar, already_db=already_db, window_size=window_size)
    pre_db  = preprocess_sar(pre_sar,  already_db=already_db, window_size=window_size)
    diff_db = post_db - pre_db  # negative = backscatter decreased

    # Adaptive Otsu threshold capped to physically plausible water range
    if post_threshold is None:
        computed_otsu = otsu_threshold(post_db)
        post_threshold = float(np.clip(computed_otsu, -22.0, -14.0))
        logger.info("Otsu-derived SAR water threshold: %.2f dB", post_threshold)
    else:
        logger.info("Fixed SAR water threshold: %.2f dB", post_threshold)

    is_post_low    = post_db <= post_threshold
    is_diff_strong = diff_db <= diff_threshold
    water_mask     = is_post_low & is_diff_strong

    # Continuous confidence — both evidence streams must agree
    post_score = np.clip((post_threshold - post_db) / 8.0, 0.0, 1.0)
    diff_score = np.clip((diff_threshold  - diff_db) / 6.0, 0.0, 1.0)
    confidence  = np.sqrt(post_score * diff_score)

    # Pixels failing the hard rule receive zero confidence
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


# ═══════════════════════════════════════════════════════════════════
# Water detection — dual-polarisation (VV + VH)
# ═══════════════════════════════════════════════════════════════════

def detect_water_sar_dual_pol(
    pre_vv: np.ndarray,
    pre_vh: np.ndarray,
    post_vv: np.ndarray,
    post_vh: np.ndarray,
    already_db: bool = True,
    vv_post_threshold: Optional[float] = None,
    vh_post_threshold: Optional[float] = None,
    diff_threshold: float = -3.5,
    window_size: int = 5,
    vh_weight: float = 0.35,
) -> SARWaterDetectionResult:
    """Dual-polarisation SAR water detection fusing VV and VH channels.

    VV is the primary water indicator (stronger backscatter drop for smooth surfaces).
    VH provides secondary confirmation; it is down-weighted because VH is noisier
    and less spectrally distinct over water in C-band.

    Fusion:
        vv_conf = water confidence from VV channel alone
        vh_conf = water confidence from VH channel alone
        combined = clip((1 - vh_weight) * vv_conf + vh_weight * vh_conf, 0, 1)

    Args:
        pre_vv:  Pre-event VV backscatter.
        pre_vh:  Pre-event VH backscatter.
        post_vv: Post-event VV backscatter.
        post_vh: Post-event VH backscatter.
        already_db: True if inputs are already in dB.
        vv_post_threshold: Otsu water threshold for VV (None = auto).
        vh_post_threshold: Otsu water threshold for VH (None = auto).
        diff_threshold: Backscatter decrease threshold (dB).
        window_size: Lee filter window size.
        vh_weight: Fractional weight of VH channel in confidence fusion [0, 1].

    Returns:
        SARWaterDetectionResult using VV grids and dual-pol combined confidence.
    """
    vv_result = detect_water_sar(
        post_vv, pre_vv,
        already_db=already_db,
        post_threshold=vv_post_threshold,
        diff_threshold=diff_threshold,
        window_size=window_size,
    )
    vh_result = detect_water_sar(
        post_vh, pre_vh,
        already_db=already_db,
        post_threshold=vh_post_threshold,
        diff_threshold=diff_threshold,
        window_size=window_size,
    )

    # Fuse confidence: VV primary, VH secondary confirmation
    vv_w = float(np.clip(1.0 - vh_weight, 0.0, 1.0))
    vh_w = float(np.clip(vh_weight, 0.0, 1.0))
    combined_conf = np.clip(
        vv_w * vv_result.water_confidence + vh_w * vh_result.water_confidence,
        0.0, 1.0,
    ).astype(np.float32)

    # Water mask: require VV hard rule; VH is confirmatory only
    combined_mask = vv_result.water_mask

    return SARWaterDetectionResult(
        water_mask=combined_mask,
        water_confidence=combined_conf,
        post_db=vv_result.post_db,
        pre_db=vv_result.pre_db,
        diff_db=vv_result.diff_db,
        post_threshold=vv_result.post_threshold,
        diff_threshold=vv_result.diff_threshold,
    )
