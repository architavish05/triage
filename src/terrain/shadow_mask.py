"""Radar shadow, layover, and occlusion (cant_see) mask calculations.

Uses DEM elevation gradients and Sentinel-1 orbit geometry (incidence angle and look direction)
to determine areas where radar cannot illuminate or where radar backscatter is geometrically distorted,
combined with optical cloud masks to produce the unified responder 'cant_see' layer.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RadarOcclusionResult:
    """Radar shadow, layover, and multi-sensor cant_see occlusion masks."""
    shadow_mask: np.ndarray        # True where terrain is in radar shadow
    layover_mask: np.ndarray       # True where terrain causes radar layover
    cant_see_mask: np.ndarray      # Consolidated occlusion mask (shadow | layover | cloud | nodata)
    cant_see_breakdown: Dict[str, int]


def get_orbit_look_vector(
    orbit_direction: str = "DESCENDING",
    incidence_angle_deg: float = 38.0,
) -> Tuple[float, float, float]:
    """Compute normalized 3D radar look vector from satellite towards ground target.

    In WGS84 ground coordinates (dx: East, dy: North, dz: Up):
    - Descending orbit (satellite tracks South): looks West (azimuth ~ 260 deg, range pointing West).
    - Ascending orbit (satellite tracks North): looks East (azimuth ~ 100 deg, range pointing East).
    """
    theta_rad = np.radians(incidence_angle_deg)
    # Ground range distance unit = sin(theta), vertical z unit = -cos(theta)
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    direction_upper = orbit_direction.upper()
    if "DESC" in direction_upper:
        # Looking West: negative X (dx < 0), slight South component
        dx = -sin_theta * 0.98
        dy = -sin_theta * 0.17
    else:
        # Looking East: positive X (dx > 0), slight North component
        dx = sin_theta * 0.98
        dy = sin_theta * 0.17

    dz = -cos_theta
    norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    return (float(dx / norm), float(dy / norm), float(dz / norm))


def compute_radar_shadow_layover(
    dem: np.ndarray,
    res_meters: float = 30.0,
    orbit_direction: str = "DESCENDING",
    incidence_angle_deg: float = 38.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate radar shadow and layover from DEM and orbit geometry.

    Radar Shadow:
      Occurs when back-slope facing away from the radar antenna is steeper than
      the grazing angle (90 - incidence_angle), or when terrain is occluded along
      the line of sight by a foreground peak.

    Radar Layover:
      Occurs when fore-slope facing towards the radar is steeper than the
      incidence angle (top of peak is closer to radar than its base).

    Args:
        dem: 2D elevation grid in meters.
        res_meters: Ground pixel resolution in meters (e.g. 30m for GLO-30).
        orbit_direction: 'DESCENDING' (looks West) or 'ASCENDING' (looks East).
        incidence_angle_deg: Radar incidence angle from nadir in degrees (typical S1: 30-45 deg).

    Returns:
        Tuple of (shadow_mask, layover_mask) as boolean 2D numpy arrays.
    """
    rows, cols = dem.shape
    dx, dy, _ = get_orbit_look_vector(orbit_direction, incidence_angle_deg)

    # Compute elevation gradients along x (cols) and y (rows)
    # dem gradients: dy_dem (row, North-South), dx_dem (col, East-West)
    grad_y, grad_x = np.gradient(dem, res_meters)

    # Direction along radar ground range (pointing away from sensor along LOS):
    ground_norm = np.sqrt(dx ** 2 + dy ** 2)
    ux = dx / max(ground_norm, 1e-9)
    uy = dy / max(ground_norm, 1e-9)

    # Elevation gradient along line of sight (range direction):
    dz_dr = grad_x * ux + grad_y * uy
    range_slope_deg = np.degrees(np.arctan(dz_dr))

    grazing_deg = 90.0 - incidence_angle_deg

    # 1. Local geometric criteria:
    # Layover: fore-slope rising towards peak exceeds incidence angle
    local_layover = range_slope_deg > incidence_angle_deg

    # Local shadow: back-slope descending away from sensor steeper than grazing angle
    local_shadow = range_slope_deg < -grazing_deg

    # 2. Ray-tracing cast shadow along range direction:
    # A ridge crest blocks sight lines behind it until terrain elevation rises above the ray
    cast_shadow = np.zeros((rows, cols), dtype=bool)
    tan_grazing = np.tan(np.radians(grazing_deg))

    # Trace rays along rows (East-West profile)
    # For descending orbit: radar looks West (from East to West), so scan East-to-West (cols-1 down to 0)
    # For ascending orbit: radar looks East (from West to East), so scan West-to-East (0 up to cols-1)
    if "DESC" in orbit_direction.upper():
        col_range = range(cols - 2, -1, -1)
        col_step = 1  # Previous pixel toward radar is col + 1
    else:
        col_range = range(1, cols)
        col_step = -1  # Previous pixel toward radar is col - 1

    ray_height = dem.copy()
    for r in range(rows):
        current_los = dem[r, cols - 1 if "DESC" in orbit_direction.upper() else 0]
        for c in col_range:
            prev_c = c + col_step
            # Line of sight descends along terrain at grazing slope
            los_from_prev = ray_height[r, prev_c] - (tan_grazing * res_meters)
            if dem[r, c] < los_from_prev:
                cast_shadow[r, c] = True
                ray_height[r, c] = los_from_prev
            else:
                ray_height[r, c] = dem[r, c]

    shadow_mask = local_shadow | cast_shadow
    layover_mask = local_layover & (~shadow_mask)

    logger.debug(
        f"Shadow pixels: {np.count_nonzero(shadow_mask)}, "
        f"Layover pixels: {np.count_nonzero(layover_mask)}"
    )
    return shadow_mask, layover_mask


def build_cant_see_mask(
    shadow_mask: np.ndarray,
    layover_mask: np.ndarray,
    cloud_mask: Optional[np.ndarray] = None,
    nodata_mask: Optional[np.ndarray] = None,
) -> RadarOcclusionResult:
    """Fuse all sources of sensor occlusion into responder cant_see mask.

    Areas where rescue teams must be warned that satellite sensors could not observe:
    - Radar shadow (mountain peaks blocking SAR illumination)
    - Radar layover (steep fore-slope distortion)
    - Cloud obscuration (optical view blocked)
    - Sensor nodata boundaries

    Args:
        shadow_mask: 2D boolean array of radar shadow.
        layover_mask: 2D boolean array of radar layover.
        cloud_mask: Optional 2D boolean array of optical cloud cover.
        nodata_mask: Optional 2D boolean array of missing data.

    Returns:
        RadarOcclusionResult containing unified cant_see mask and breakdown.
    """
    shape = shadow_mask.shape
    cant_see = shadow_mask | layover_mask

    if cloud_mask is not None and cloud_mask.shape == shape:
        cant_see = cant_see | cloud_mask
    if nodata_mask is not None and nodata_mask.shape == shape:
        cant_see = cant_see | nodata_mask

    breakdown = {
        "radar_shadow_pixels": int(np.count_nonzero(shadow_mask)),
        "radar_layover_pixels": int(np.count_nonzero(layover_mask)),
        "cloud_pixels": int(np.count_nonzero(cloud_mask)) if cloud_mask is not None else 0,
        "nodata_pixels": int(np.count_nonzero(nodata_mask)) if nodata_mask is not None else 0,
        "total_cant_see_pixels": int(np.count_nonzero(cant_see)),
        "total_pixels": int(cant_see.size),
        "occluded_percent": float(round((np.count_nonzero(cant_see) / cant_see.size) * 100.0, 2)),
    }

    return RadarOcclusionResult(
        shadow_mask=shadow_mask,
        layover_mask=layover_mask,
        cant_see_mask=cant_see,
        cant_see_breakdown=breakdown,
    )
