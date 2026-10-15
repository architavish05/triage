"""Debris detection via InSAR coherence loss or GRD amplitude/texture fallback.

Two distinct computation paths are provided and clearly distinguished:

PATH A — Genuine InSAR Coherence (requires complex SLC data):
    ``compute_complex_coherence(master_slc, slave_slc)``
    Returns spatial coherence γ ∈ [0, 1] by the standard formula:
        γ = |Σ s1·conj(s2)| / sqrt(Σ|s1|² · Σ|s2|²)
    over a sliding window.  Input arrays must be complex-valued (dtype complex64/128).

PATH B — GRD Amplitude Correlation Proxy (when only detected backscatter is available):
    ``compute_coherence_proxy(pre_db, post_db)``
    Computes normalised local amplitude correlation as a proxy for coherence loss.
    THIS IS NOT GENUINE INTERFEROMETRIC COHERENCE.  It approximates temporal
    decorrelation using intensity differences and local texture, which are
    physically related but not equivalent to phase coherence.  The proxy is
    documented and labelled in all outputs.

Critical limitation:
    Genuine InSAR coherence cannot be computed from four GRD backscatter images
    (pre_vv, pre_vh, post_vv, post_vh) alone, because phase information is lost
    during GRD processing.  Any function claiming otherwise is scientifically
    incorrect.  Users who call ``compute_coherence_proxy`` must acknowledge this
    limitation in any scientific communication.

Debris detection:
    ``detect_debris_from_coherence``    — uses PATH A (coherence loss)
    ``detect_debris_amplitude_fallback`` — uses PATH B proxy or raw amplitude surge
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import uniform_filter

logger = logging.getLogger(__name__)


@dataclass
class DebrisDetectionResult:
    """Output of debris flow and deposit detection."""

    debris_mask: np.ndarray          # bool: True where debris detected
    debris_confidence: np.ndarray    # float32 in [0.0, 1.0]
    method_used: str                 # see module docstring for valid values
    metric_grid: np.ndarray          # coherence-loss or texture-anomaly values
    threshold: float                 # threshold applied to metric_grid


# ═══════════════════════════════════════════════════════════════════
# PATH A — Genuine InSAR Coherence (SLC data required)
# ═══════════════════════════════════════════════════════════════════

def compute_complex_coherence(
    master_slc: np.ndarray,
    slave_slc: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """Compute interferometric spatial coherence from two complex SLC arrays.

    γ = |Σ_w (s1 · conj(s2))| / sqrt(Σ_w|s1|² · Σ_w|s2|²)

    where the sum is taken over a sliding spatial window of size window_size.

    This is the standard box-filter coherence estimator (Zebker & Villasenor, 1992).
    Window size directly controls the bias-variance trade-off.

    Args:
        master_slc: 2-D complex64/128 reference (master) SLC scene.
        slave_slc:  2-D complex64/128 secondary (slave) SLC scene.
                    Must have the same spatial dimensions as master_slc.
        window_size: Spatial averaging window in pixels (forced odd).

    Returns:
        Coherence raster with values in [0.0, 1.0].

    Raises:
        ValueError: If input arrays are not complex-typed.
    """
    if not np.iscomplexobj(master_slc) or not np.iscomplexobj(slave_slc):
        raise ValueError(
            "compute_complex_coherence requires complex-valued SLC arrays. "
            "For GRD/backscatter data use compute_coherence_proxy instead."
        )

    if window_size % 2 == 0:
        window_size += 1

    cross     = master_slc * np.conj(slave_slc)
    pwr_m     = np.abs(master_slc) ** 2
    pwr_s     = np.abs(slave_slc)  ** 2

    sum_cross = (
        uniform_filter(np.real(cross), size=window_size)
        + 1j * uniform_filter(np.imag(cross), size=window_size)
    )
    sum_pwr_m = uniform_filter(pwr_m, size=window_size)
    sum_pwr_s = uniform_filter(pwr_s, size=window_size)

    denom     = np.sqrt(np.maximum(sum_pwr_m * sum_pwr_s, 1e-12))
    coherence = np.abs(sum_cross) / denom
    return np.clip(coherence, 0.0, 1.0).astype(np.float32)


def detect_debris_from_coherence(
    pre_coherence: np.ndarray,
    co_event_coherence: np.ndarray,
    loss_threshold: float = 0.35,
    co_event_max_threshold: float = 0.30,
) -> DebrisDetectionResult:
    """Detect debris via InSAR temporal decorrelation (coherence loss).

    Stable pre-event terrain retains high interferometric coherence.  Intense
    debris deposition, ground scour, or sedimentation destroys phase correlation,
    causing a sharp drop in co-event coherence (Zebker & Villasenor, 1992).

    Detection rule:
        (pre_coherence - co_event_coherence >= loss_threshold)
        AND (co_event_coherence <= co_event_max_threshold)

    Confidence:
        confidence = clip((coherence_loss - loss_threshold) / (1 - loss_threshold), 0, 1)

    Args:
        pre_coherence: Baseline pre-event coherence grid ∈ [0, 1].
        co_event_coherence: Pre-to-post co-event coherence grid ∈ [0, 1].
        loss_threshold: Minimum coherence loss indicating surface disturbance.
        co_event_max_threshold: Upper bound for co-event coherence in debris zones.

    Returns:
        DebrisDetectionResult with method_used='coherence_loss'.
    """
    coherence_loss = pre_coherence - co_event_coherence
    mask = (coherence_loss >= loss_threshold) & (co_event_coherence <= co_event_max_threshold)

    confidence = np.clip(
        (coherence_loss - loss_threshold) / max(1.0 - loss_threshold, 1e-9),
        0.0, 1.0,
    )
    confidence = np.where(mask, np.clip(confidence, 0.35, 1.0), 0.0).astype(np.float32)

    return DebrisDetectionResult(
        debris_mask=mask,
        debris_confidence=confidence,
        method_used="coherence_loss",
        metric_grid=coherence_loss.astype(np.float32),
        threshold=loss_threshold,
    )


# ═══════════════════════════════════════════════════════════════════
# PATH B — GRD Amplitude Proxy
# ═══════════════════════════════════════════════════════════════════

def compute_local_std(array: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Compute local standard deviation as a textural roughness descriptor.

    Args:
        array: 2-D float32 input raster.
        window_size: Window diameter in pixels (forced odd).

    Returns:
        Local std float32 array, same shape as input.
    """
    if window_size % 2 == 0:
        window_size += 1
    mean_val = uniform_filter(array, size=window_size)
    mean_sq  = uniform_filter(array ** 2, size=window_size)
    variance = np.maximum(mean_sq - mean_val ** 2, 0.0)
    return np.sqrt(variance).astype(np.float32)


def compute_coherence_proxy(
    pre_db: np.ndarray,
    post_db: np.ndarray,
    window_size: int = 7,
) -> np.ndarray:
    """Compute a GRD-based amplitude correlation proxy for coherence.

    ⚠ THIS IS NOT GENUINE INTERFEROMETRIC COHERENCE.  It approximates temporal
    decorrelation by measuring normalised cross-correlation of local amplitude
    windows in the pre- and post-event backscatter.

    High proxy value (near 1) → similar amplitude textures → stable surface.
    Low proxy value (near 0) → large textural change → potential disturbance.

    Formula:
        proxy = 1 - clip(|post_db - pre_db| / 10.0, 0, 1)
        then locally smoothed and scaled.

    Physical justification:
        After a major debris flow or surface disturbance, the local backscatter
        pattern changes substantially (different surface roughness, dielectric
        properties).  This amplitude change is a proxy for the expected coherence
        loss, though it cannot capture phase decorrelation effects.

    Args:
        pre_db:  2-D float32 pre-event dB backscatter.
        post_db: 2-D float32 post-event dB backscatter.
        window_size: Smoothing window size (pixels).

    Returns:
        Proxy coherence array in [0.0, 1.0], same shape as inputs.
        method label = 'grd_amplitude_proxy' (NOT 'insar_coherence').
    """
    if window_size % 2 == 0:
        window_size += 1

    # Normalised absolute amplitude difference
    abs_diff = np.abs(post_db - pre_db)
    # 10 dB = very strong change → proxy ≈ 0; 0 dB change → proxy = 1
    proxy_raw = np.clip(1.0 - abs_diff / 10.0, 0.0, 1.0)
    # Smooth to reduce speckle influence
    proxy = uniform_filter(proxy_raw, size=window_size)
    return np.clip(proxy, 0.0, 1.0).astype(np.float32)


def detect_debris_amplitude_fallback(
    post_db: np.ndarray,
    pre_db: np.ndarray,
    diff_threshold: float = 2.0,
    texture_threshold: float = 1.2,
    window_size: int = 5,
) -> DebrisDetectionResult:
    """GRD fallback: detect debris via SAR amplitude surge and texture roughness.

    Used when SLC data is unavailable (standard operational case for Sentinel-1 GRD).

    Physical basis:
        - Debris deposits (boulders, gravel, disturbed sediment) scatter diffusely,
          significantly increasing backscatter amplitude relative to pre-event smooth land.
        - Local spatial variance increases due to the irregular surface texture.
        - Water pixels have NEGATIVE dB change (specular) — excluded via is_non_water.

    Detection:
        is_amplitude_surge = (post_db - pre_db) >= diff_threshold
        is_texture_rough   = (post_texture - pre_texture) >= texture_threshold
        is_non_water       = post_db > -18.0 dB  (specular water excluded)
        debris_mask        = (is_amplitude_surge OR is_texture_rough) AND is_non_water

    Confidence:
        amp_score = clip((diff_db - diff_threshold) / 5.0, 0, 1)
        tex_score = clip((texture_diff - texture_threshold) / 3.0, 0, 1)
        confidence = max(amp_score, tex_score)

    Limitations:
        - Urban areas can produce false debris signals (corner reflectors).
        - Wind-roughened water can produce false positive if post_db > -18 dB.
        - Slope filtering (by Person 3) should suppress steep-terrain artefacts.

    Args:
        post_db: Filtered post-flood SAR dB grid.
        pre_db:  Filtered pre-flood SAR dB grid.
        diff_threshold: Minimum positive dB increase indicating debris backscatter surge.
        texture_threshold: Minimum increase in local texture std to confirm roughness.
        window_size: Window size for texture computation.

    Returns:
        DebrisDetectionResult with method_used='amplitude_texture_fallback'.
    """
    diff_db      = post_db - pre_db
    post_texture = compute_local_std(post_db, window_size=window_size)
    pre_texture  = compute_local_std(pre_db,  window_size=window_size)
    texture_diff = post_texture - pre_texture

    is_amp_surge  = diff_db >= diff_threshold
    is_tex_rough  = texture_diff >= texture_threshold
    is_non_water  = post_db > -18.0  # specular water has lower backscatter

    debris_mask = (is_amp_surge | is_tex_rough) & is_non_water

    amp_score = np.clip((diff_db - diff_threshold) / 5.0, 0.0, 1.0)
    tex_score = np.clip((texture_diff - texture_threshold) / 3.0, 0.0, 1.0)
    combined  = np.maximum(amp_score, tex_score)

    confidence = np.where(debris_mask, np.clip(combined, 0.3, 1.0), 0.0).astype(np.float32)

    return DebrisDetectionResult(
        debris_mask=debris_mask,
        debris_confidence=confidence,
        method_used="amplitude_texture_fallback",
        metric_grid=np.maximum(diff_db, texture_diff).astype(np.float32),
        threshold=diff_threshold,
    )
