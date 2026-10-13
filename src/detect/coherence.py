"""Debris detection via InSAR coherence loss or amplitude/texture change fallback.

Detects debris flows, boulder deposits, and ground surface disturbance using:
1. InSAR coherence loss (when complex SLC scenes are available).
2. Amplitude surge and texture variance anomaly fallback (when GRD data is used).
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


@dataclass
class DebrisDetectionResult:
    """Output of debris flow and deposit detection."""
    debris_mask: np.ndarray         # Boolean mask (True where debris detected)
    debris_confidence: np.ndarray   # Float array in [0.0, 1.0]
    method_used: str                # 'coherence_loss' or 'amplitude_texture_fallback'
    metric_grid: np.ndarray         # Coherence loss or texture anomaly values
    threshold: float                # Threshold applied to metric


def compute_complex_coherence(
    master_slc: np.ndarray,
    slave_slc: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute interferometric spatial coherence from two complex SLC arrays.

    gamma = |sum(s1 * conj(s2))| / sqrt(sum(|s1|^2) * sum(|s2|^2))

    Args:
        master_slc: Complex 2D array of reference acquisition.
        slave_slc: Complex 2D array of secondary acquisition.
        window_size: Moving window dimension (odd integer).

    Returns:
        Coherence raster with values in [0.0, 1.0].
    """
    if window_size % 2 == 0:
        window_size += 1

    cross = master_slc * np.conj(slave_slc)
    pwr_master = np.abs(master_slc) ** 2
    pwr_slave = np.abs(slave_slc) ** 2

    # Local neighborhood sums via uniform filter
    sum_cross = uniform_filter(np.real(cross), size=window_size) + 1j * uniform_filter(np.imag(cross), size=window_size)
    sum_pwr_m = uniform_filter(pwr_master, size=window_size)
    sum_pwr_s = uniform_filter(pwr_slave, size=window_size)

    denom = np.sqrt(np.maximum(sum_pwr_m * sum_pwr_s, 1e-12))
    coherence = np.abs(sum_cross) / denom
    return np.clip(coherence, 0.0, 1.0).astype(np.float32)


def detect_debris_from_coherence(
    pre_coherence: np.ndarray,
    co_event_coherence: np.ndarray,
    loss_threshold: float = 0.35,
    co_event_max_threshold: float = 0.30,
) -> DebrisDetectionResult:
    """Detect debris via InSAR temporal decorrelation (coherence loss).

    Stable pre-event ground retains coherence. Intense debris deposition or scouring
    destroys radar phase correlation, causing a dramatic drop in co-event coherence.

    Args:
        pre_coherence: Baseline pre-event coherence grid.
        co_event_coherence: Pre-to-post co-event coherence grid.
        loss_threshold: Minimum coherence loss (pre - co_event) indicating disturbance.
        co_event_max_threshold: Upper bound for co-event coherence in disturbed zones.

    Returns:
        DebrisDetectionResult with debris mask and confidence in [0, 1].
    """
    coherence_loss = pre_coherence - co_event_coherence
    mask = (coherence_loss >= loss_threshold) & (co_event_coherence <= co_event_max_threshold)

    # Confidence scales with the magnitude of coherence loss
    confidence = np.clip((coherence_loss - loss_threshold) / (1.0 - loss_threshold), 0.0, 1.0)
    confidence = np.where(mask, np.clip(confidence, 0.35, 1.0), 0.0).astype(np.float32)

    return DebrisDetectionResult(
        debris_mask=mask,
        debris_confidence=confidence,
        method_used="coherence_loss",
        metric_grid=coherence_loss,
        threshold=loss_threshold,
    )


def compute_local_std(array: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute local standard deviation as a textural roughness descriptor."""
    if window_size % 2 == 0:
        window_size += 1
    mean_val = uniform_filter(array, size=window_size)
    mean_sq = uniform_filter(array ** 2, size=window_size)
    variance = np.maximum(mean_sq - (mean_val ** 2), 0.0)
    return np.sqrt(variance).astype(np.float32)


def detect_debris_amplitude_fallback(
    post_db: np.ndarray,
    pre_db: np.ndarray,
    diff_threshold: float = 2.0,
    texture_threshold: float = 1.2,
    window_size: int = 5,
) -> DebrisDetectionResult:
    """Fallback debris detection using SAR amplitude surge and texture roughness.

    Used when SLC data is unavailable. Debris flows deposit rough gravel, boulders,
    and uneven sediment deposits, significantly elevating diffuse backscatter and
    local spatial variance relative to pre-event smooth terrain.

    Args:
        post_db: Filtered post-flood SAR dB grid.
        pre_db: Filtered pre-flood SAR dB grid.
        diff_threshold: Minimum positive dB increase (roughness surge).
        texture_threshold: Minimum increase in local texture standard deviation.
        window_size: Moving window size for texture computation.

    Returns:
        DebrisDetectionResult with debris mask and confidence in [0, 1].
    """
    diff_db = post_db - pre_db
    post_texture = compute_local_std(post_db, window_size=window_size)
    pre_texture = compute_local_std(pre_db, window_size=window_size)
    texture_diff = post_texture - pre_texture

    # Debris signatures:
    # 1. Backscatter surge due to rough boulder/sediment scattering (diff_db >= diff_threshold)
    # 2. Local textural disturbance (texture_diff >= texture_threshold)
    is_amplitude_surge = diff_db >= diff_threshold
    is_texture_rough = texture_diff >= texture_threshold

    # Debris must NOT be specular water (post_db > -18 dB)
    is_non_water = post_db > -18.0

    debris_mask = (is_amplitude_surge | is_texture_rough) & is_non_water

    # Continuous confidence calculation
    amp_score = np.clip((diff_db - diff_threshold) / 5.0, 0.0, 1.0)
    tex_score = np.clip((texture_diff - texture_threshold) / 3.0, 0.0, 1.0)
    combined_score = np.maximum(amp_score, tex_score)

    confidence = np.where(debris_mask, np.clip(combined_score, 0.3, 1.0), 0.0).astype(np.float32)

    return DebrisDetectionResult(
        debris_mask=debris_mask,
        debris_confidence=confidence,
        method_used="amplitude_texture_fallback",
        metric_grid=np.maximum(diff_db, texture_diff),
        threshold=diff_threshold,
    )
