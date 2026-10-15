"""Unit tests for radar shadow, layover, and cant_see occlusion masking."""

import numpy as np
from p3_terrain.terrain.shadow_mask import (
    build_cant_see_mask,
    compute_radar_shadow_layover,
    get_orbit_look_vector,
)


def test_orbit_look_vector():
    """Verify 3D line-of-sight unit vector for ascending and descending geometries."""
    # Descending orbit looks West (negative x)
    dx_desc, dy_desc, dz_desc = get_orbit_look_vector("DESCENDING", incidence_angle_deg=35.0)
    assert dx_desc < 0.0
    assert dz_desc < 0.0
    norm_desc = np.sqrt(dx_desc ** 2 + dy_desc ** 2 + dz_desc ** 2)
    assert np.isclose(norm_desc, 1.0, atol=1e-4)

    # Ascending orbit looks East (positive x)
    dx_asc, dy_asc, dz_asc = get_orbit_look_vector("ASCENDING", incidence_angle_deg=35.0)
    assert dx_asc > 0.0
    assert dz_asc < 0.0
    norm_asc = np.sqrt(dx_asc ** 2 + dy_asc ** 2 + dz_asc ** 2)
    assert np.isclose(norm_asc, 1.0, atol=1e-4)


def test_radar_shadow_and_layover_on_ridge():
    """Verify radar shadow occurs on back-slopes and layover on steep fore-slopes."""
    # Create an East-West triangular mountain ridge (cols=60, peak at col=30)
    # Descending orbit looks West (from East towards West)
    # Fore-slope faces East (cols 30..60), back-slope faces West (cols 0..30)
    rows, cols = 40, 60
    dem = np.zeros((rows, cols), dtype=np.float32)
    for c in range(cols):
        # Peak elevation 1500m at center col 30, base at 200m
        dist_from_peak = abs(c - 30)
        dem[:, c] = 1500.0 - dist_from_peak * 45.0  # Steep mountain flank (45m per 30m pixel > 50 deg slope!)

    shadow_mask, layover_mask = compute_radar_shadow_layover(
        dem,
        res_meters=30.0,
        orbit_direction="DESCENDING",
        incidence_angle_deg=38.0,
    )

    # Since descending radar is East looking West:
    # Terrain behind peak (to the West, cols < 30) must have radar shadow
    assert np.any(shadow_mask[:, :28])
    # Terrain facing radar (to the East, cols > 30) must trigger layover
    assert np.any(layover_mask[:, 32:])


def test_cant_see_mask_fusion():
    """Verify cant_see fuses shadow, layover, cloud, and nodata into comprehensive occlusion."""
    shape = (50, 50)
    shadow = np.zeros(shape, dtype=bool)
    layover = np.zeros(shape, dtype=bool)
    cloud = np.zeros(shape, dtype=bool)

    shadow[10:20, 10:20] = True
    layover[25:35, 25:35] = True
    cloud[40:48, 40:48] = True

    result = build_cant_see_mask(shadow, layover, cloud_mask=cloud)

    assert result.cant_see_mask.shape == shape
    # Verify all three occluded regions are flagged in cant_see
    assert np.all(result.cant_see_mask[12:18, 12:18])
    assert np.all(result.cant_see_mask[27:33, 27:33])
    assert np.all(result.cant_see_mask[42:46, 42:46])
    # Unaffected open ground is False
    assert not result.cant_see_mask[0, 0]

    assert result.cant_see_breakdown["radar_shadow_pixels"] == 100
    assert result.cant_see_breakdown["radar_layover_pixels"] == 100
    assert result.cant_see_breakdown["cloud_pixels"] == 64
