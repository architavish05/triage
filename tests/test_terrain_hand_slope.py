"""Unit tests for slope, HAND, and impossible flood filtering."""

import numpy as np
from src.detect.fuse import CLASS_BACKGROUND, CLASS_DEBRIS, CLASS_WATER
from src.terrain.hand_slope import (
    compute_hand,
    compute_slope,
    filter_impossible_flood,
)


def test_compute_slope():
    """Verify slope computation on a uniform ramp."""
    # Ramp with dz = 30m over dx = 30m -> slope = 45 degrees
    rows, cols = 30, 30
    x_coords = np.arange(cols, dtype=np.float32)
    dem = np.tile(x_coords * 30.0, (rows, 1))

    slope = compute_slope(dem, res_meters=30.0)
    # Interior pixels should be 45 degrees
    assert np.allclose(slope[5:25, 5:25], 45.0, atol=1.0)


def test_compute_hand_valley():
    """Verify HAND is minimal near river channel and high on mountain slopes."""
    # Symmetrical V-shaped valley with river at center column 25
    rows, cols = 40, 50
    col_idx = np.arange(cols)
    dist_from_river = np.abs(col_idx - 25).astype(np.float32)
    # Valley floor at 500m, rises by 15m per cell away from river
    dem = np.tile(500.0 + dist_from_river * 15.0, (rows, 1))

    hand, drainage_mask = compute_hand(dem, res_meters=30.0)

    # Valley floor center channel cell should have 0 HAND
    assert np.all(hand[:, 25] == 0.0)
    assert np.all(hand[:, 24:27] <= 15.0)
    # Ridge lines at edges (cols 0 and 49) should have high HAND (> 200m)
    assert np.all(hand[:, 0] > 200.0)
    assert np.all(hand[:, 49] > 200.0)


def test_filter_impossible_flood():
    """Verify impossible flood pixels on steep slopes, high HAND, and radar shadows are purged."""
    shape = (40, 40)
    class_map = np.full(shape, CLASS_BACKGROUND, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)

    slope = np.full(shape, 3.0, dtype=np.float32)      # Gentle flat valley floor (3 deg)
    hand = np.full(shape, 4.0, dtype=np.float32)       # Low HAND (4m above river)
    shadow = np.zeros(shape, dtype=bool)

    # 1. Valid flood in valley floor (r=10..15, c=10..15)
    class_map[10:16, 10:16] = CLASS_WATER
    confidence[10:16, 10:16] = 0.85

    # 2. Impossible water on steep cliff (r=20..25, c=10..15, slope=28 deg)
    class_map[20:26, 10:16] = CLASS_WATER
    confidence[20:26, 10:16] = 0.80
    slope[20:26, 10:16] = 28.0

    # 3. Impossible water on high ridge (r=10..16, c=25..30, HAND=120m)
    class_map[10:16, 25:31] = CLASS_WATER
    confidence[10:16, 25:31] = 0.75
    hand[10:16, 25:31] = 120.0

    # 4. False water inside radar shadow (r=25..30, c=25..30)
    class_map[25:31, 25:31] = CLASS_WATER
    confidence[25:31, 25:31] = 0.70
    shadow[25:31, 25:31] = True

    result = filter_impossible_flood(
        class_map=class_map,
        confidence=confidence,
        slope=slope,
        hand=hand,
        shadow_mask=shadow,
        max_flood_slope_deg=10.0,
        max_flood_hand_m=22.0,
    )

    # Valid flood at (12, 12) must remain CLASS_WATER
    assert result.filtered_class_map[12, 12] == CLASS_WATER
    assert result.filtered_confidence[12, 12] > 0.5

    # Steep cliff (22, 12) must be eliminated (downgraded to 0)
    assert result.filtered_class_map[22, 12] == CLASS_BACKGROUND
    assert result.filtered_confidence[22, 12] == 0.0

    # High ridge (12, 28) must be eliminated
    assert result.filtered_class_map[12, 28] == CLASS_BACKGROUND
    assert result.filtered_confidence[12, 28] == 0.0

    # Shadow water (28, 28) must be eliminated
    assert result.filtered_class_map[28, 28] == CLASS_BACKGROUND
    assert result.filtered_confidence[28, 28] == 0.0

    assert result.suppressed_water_pixels == 36 * 3  # All 3 invalid 6x6 patches suppressed
