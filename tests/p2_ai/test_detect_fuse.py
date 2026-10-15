"""Unit tests for multi-sensor detection fusion."""

import numpy as np
from p2_ai.detect.coherence import detect_debris_amplitude_fallback
from p2_ai.detect.fuse import (
    CLASS_BACKGROUND,
    CLASS_DEBRIS,
    CLASS_WATER,
    fuse_detections,
)
from p2_ai.detect.sar_change import detect_water_sar


def test_fuse_detections_water_and_debris():
    """Verify fusion produces valid class map (0, 1, 2) and [0, 1] confidence."""
    shape = (50, 50)
    pre_sar = np.full(shape, -12.0, dtype=np.float32)
    post_sar = np.full(shape, -12.0, dtype=np.float32)

    # Water zone in left half: backscatter drops to -22 dB
    post_sar[10:30, 5:20] = -22.0
    # Debris zone in right half: backscatter surges to -7 dB
    post_sar[10:30, 30:45] = -7.0

    water_res = detect_water_sar(post_sar, pre_sar, already_db=True, post_threshold=-16.0, diff_threshold=-4.0)
    debris_res = detect_debris_amplitude_fallback(post_sar, pre_sar, diff_threshold=2.0)

    fused = fuse_detections(water_res, debris_res)

    assert fused.shape == shape
    # Water region must be CLASS_WATER (1)
    assert fused.class_map[20, 10] == CLASS_WATER
    # Debris region must be CLASS_DEBRIS (2)
    assert fused.class_map[20, 35] == CLASS_DEBRIS
    # Background must be CLASS_BACKGROUND (0)
    assert fused.class_map[0, 0] == CLASS_BACKGROUND

    # Confidence raster must be bounded in [0, 1]
    assert np.all(fused.confidence >= 0.0)
    assert np.all(fused.confidence <= 1.0)
    assert fused.summary_counts["water_pixels"] > 0
    assert fused.summary_counts["debris_pixels"] > 0


def test_fuse_with_optical_reinforcement():
    """Verify optical NDWI strengthens water confidence."""
    shape = (30, 30)
    pre_sar = np.full(shape, -12.0, dtype=np.float32)
    post_sar = np.full(shape, -12.0, dtype=np.float32)
    post_sar[10:20, 10:20] = -21.0

    water_res = detect_water_sar(post_sar, pre_sar, already_db=True, post_threshold=-16.0, diff_threshold=-4.0)
    debris_res = detect_debris_amplitude_fallback(post_sar, pre_sar)

    # Synthetic NDWI confirming open water
    ndwi = np.full(shape, -0.3, dtype=np.float32)
    ndwi[10:20, 10:20] = 0.55

    fused_no_opt = fuse_detections(water_res, debris_res, optical_ndwi=None)
    fused_opt = fuse_detections(water_res, debris_res, optical_ndwi=ndwi)

    # Reinforced confidence should be higher or equal
    assert np.mean(fused_opt.confidence[12:18, 12:18]) >= np.mean(fused_no_opt.confidence[12:18, 12:18])
