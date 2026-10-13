"""Unit tests for SAR change detection and preprocessing."""

import numpy as np
from src.detect.sar_change import (
    detect_water_sar,
    lee_filter,
    otsu_threshold,
    preprocess_sar,
    to_db,
)


def test_to_db():
    """Verify linear intensity conversion to decibels."""
    linear = np.array([1.0, 0.1, 0.01, 1e-4], dtype=np.float32)
    db = to_db(linear)
    assert np.isclose(db[0], 0.0, atol=1e-3)
    assert np.isclose(db[1], -10.0, atol=1e-3)
    assert np.isclose(db[2], -20.0, atol=1e-3)
    assert np.isclose(db[3], -40.0, atol=1e-3)


def test_lee_filter_noise_reduction():
    """Verify Lee filter preserves signal structure while damping speckle variance."""
    rng = np.random.default_rng(42)
    clean_signal = np.ones((64, 64), dtype=np.float32) * -12.0
    noisy_signal = clean_signal + rng.normal(0, 3.0, size=(64, 64)).astype(np.float32)

    filtered = lee_filter(noisy_signal, window_size=5)
    assert filtered.shape == (64, 64)
    # Variance of filtered array should be noticeably lower than raw noisy input
    assert np.var(filtered) < np.var(noisy_signal)
    # Mean should be approximately preserved
    assert np.isclose(np.mean(filtered), np.mean(clean_signal), atol=0.5)


def test_otsu_threshold():
    """Verify Otsu threshold cleanly separates two distinct modes."""
    cluster1 = np.random.normal(-24.0, 1.0, size=500)
    cluster2 = np.random.normal(-10.0, 1.0, size=500)
    bimodal = np.concatenate([cluster1, cluster2])

    thresh = otsu_threshold(bimodal)
    assert -20.0 < thresh < -14.0


def test_detect_water_sar():
    """Verify SAR water change detection identifies flooded zones and computes confidence [0, 1]."""
    shape = (40, 40)
    pre_sar = np.full(shape, -11.0, dtype=np.float32)
    post_sar = np.full(shape, -11.0, dtype=np.float32)

    # Inject flood in center (10x10) with sharp dB drop to -23 dB
    post_sar[15:25, 15:25] = -23.0

    result = detect_water_sar(post_sar, pre_sar, already_db=True, post_threshold=-16.0, diff_threshold=-4.0)

    assert result.water_mask.shape == shape
    # Flood pixels in center should be True
    assert np.all(result.water_mask[17:23, 17:23])
    # Background should be False
    assert not np.any(result.water_mask[0:10, 0:10])

    # Confidence must be in [0, 1]
    assert np.all(result.water_confidence >= 0.0)
    assert np.all(result.water_confidence <= 1.0)
    assert np.mean(result.water_confidence[17:23, 17:23]) > 0.7
