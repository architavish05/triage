"""Unit tests for coherence loss and amplitude/texture debris detection."""

import numpy as np
from p2_ai.detect.coherence import (
    compute_complex_coherence,
    detect_debris_amplitude_fallback,
    detect_debris_from_coherence,
)


def test_compute_complex_coherence():
    """Verify coherence equals 1.0 for identical scenes and near 0 for uncorrelated noise."""
    shape = (32, 32)
    s1 = np.ones(shape, dtype=np.complex64)
    s2 = np.ones(shape, dtype=np.complex64)
    coh_identical = compute_complex_coherence(s1, s2, window_size=3)
    assert np.allclose(coh_identical, 1.0, atol=1e-3)

    rng = np.random.default_rng(99)
    noise1 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    noise2 = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    coh_noise = compute_complex_coherence(noise1, noise2, window_size=5)
    assert np.mean(coh_noise) < 0.45


def test_detect_debris_from_coherence():
    """Verify coherence loss detects ground surface decorrelation."""
    shape = (30, 30)
    pre_coh = np.full(shape, 0.85, dtype=np.float32)
    co_event_coh = np.full(shape, 0.85, dtype=np.float32)

    # Inject debris disturbance in corner (coherence drops to 0.15)
    co_event_coh[5:15, 5:15] = 0.15

    result = detect_debris_from_coherence(pre_coh, co_event_coh, loss_threshold=0.35, co_event_max_threshold=0.30)
    assert result.debris_mask[10, 10]
    assert not result.debris_mask[25, 25]
    assert result.debris_confidence[10, 10] > 0.5
    assert np.all(result.debris_confidence >= 0.0)
    assert np.all(result.debris_confidence <= 1.0)


def test_detect_debris_amplitude_fallback():
    """Verify amplitude surge and texture roughness detect debris without SLC."""
    shape = (40, 40)
    pre_db = np.full(shape, -14.0, dtype=np.float32)
    post_db = np.full(shape, -14.0, dtype=np.float32)

    # Debris flow deposit causes rough surface backscatter increase (+4.5 dB)
    post_db[12:22, 12:22] = -9.5

    result = detect_debris_amplitude_fallback(post_db, pre_db, diff_threshold=2.0)
    assert result.debris_mask[15, 15]
    assert not result.debris_mask[0, 0]
    assert result.debris_confidence[15, 15] > 0.3
    assert result.method_used == "amplitude_texture_fallback"
