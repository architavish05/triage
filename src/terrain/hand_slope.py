"""Slope and Height Above Nearest Drainage (HAND) hydrological filtering.

Calculates terrain slope and HAND relative elevation above drainage networks to remove
physically impossible flood detections on steep valley flanks and high mountain ridges.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from src.detect.fuse import CLASS_BACKGROUND, CLASS_DEBRIS, CLASS_WATER

logger = logging.getLogger(__name__)


@dataclass
class TerrainFilterResult:
    """Output of terrain-filtered hazard map."""
    filtered_class_map: np.ndarray        # Cleaned 2D class map (0=bg, 1=water, 2=debris)
    filtered_confidence: np.ndarray       # Terrain-adjusted per-pixel confidence [0, 1]
    slope_deg: np.ndarray                 # Terrain slope raster in degrees
    hand_meters: np.ndarray               # Height Above Nearest Drainage raster in meters
    drainage_mask: np.ndarray             # Identified stream/drainage channel cells
    suppressed_water_pixels: int          # Count of false water pixels removed
    suppressed_debris_pixels: int         # Count of false debris pixels removed
    stats: Dict[str, float]


def compute_slope(dem: np.ndarray, res_meters: float = 30.0) -> np.ndarray:
    """Compute terrain slope in degrees from 2D elevation model.

    slope_deg = arctan(sqrt((dz/dx)^2 + (dz/dy)^2)) * (180 / pi)
    """
    grad_y, grad_x = np.gradient(dem, res_meters)
    slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
    return np.degrees(slope_rad).astype(np.float32)


def extract_drainage_network(
    dem: np.ndarray,
    slope: np.ndarray,
    elevation_window: int = 15,
) -> np.ndarray:
    """Extract stream channel / valley drainage network from DEM and slope.

    Identifies local elevation minima with gentle slopes that collect surface runoff.
    """
    if elevation_window % 2 == 0:
        elevation_window += 1
    # Local neighborhood minimum elevation
    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(dem, size=elevation_window)
    # Valley bottom cells: elevation is within 5m of local minimum AND slope < 8 deg
    valley_bottom = (dem - local_min <= 5.0) & (slope < 8.0)

    # In case whole terrain is flat, ensure at least lowest elevation cells are marked
    if np.count_nonzero(valley_bottom) == 0:
        min_elev = np.min(dem)
        valley_bottom = dem <= min_elev + 5.0

    return valley_bottom


def compute_hand(
    dem: np.ndarray,
    drainage_mask: Optional[np.ndarray] = None,
    slope: Optional[np.ndarray] = None,
    res_meters: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Height Above Nearest Drainage (HAND) in meters.

    For every grid cell, HAND represents elevation difference relative to the
    nearest hydrological drainage channel cell.

    Args:
        dem: 2D elevation grid in meters.
        drainage_mask: Optional boolean mask of river/stream channels.
        slope: Optional slope in degrees.
        res_meters: Ground cell resolution in meters.

    Returns:
        Tuple of (hand_grid, drainage_mask).
    """
    if slope is None:
        slope = compute_slope(dem, res_meters=res_meters)

    if drainage_mask is None or np.count_nonzero(drainage_mask) == 0:
        drainage_mask = extract_drainage_network(dem, slope)

    # Compute Euclidean nearest drainage cell coordinates
    _, (nearest_r, nearest_c) = distance_transform_edt(~drainage_mask, return_indices=True)

    # Drainage elevation at the nearest channel cell
    drainage_elev = dem[nearest_r, nearest_c]

    # HAND = max(0, cell_elevation - drainage_elevation)
    hand = np.maximum(0.0, dem - drainage_elev).astype(np.float32)

    return hand, drainage_mask


def filter_impossible_flood(
    class_map: np.ndarray,
    confidence: np.ndarray,
    slope: np.ndarray,
    hand: np.ndarray,
    shadow_mask: np.ndarray,
    max_flood_slope_deg: float = 10.0,
    max_flood_hand_m: float = 22.0,
) -> TerrainFilterResult:
    """Filter out or down-weight physically impossible flood and debris detections.

    Physical Hydrology Rules:
    1. Standing or slow-moving flood water cannot exist on steep terrain (> max_flood_slope_deg).
    2. Flood water cannot rise beyond valley drainage envelopes (> max_flood_hand_m).
    3. Radar shadow creates dark specular-like backscatter that mimics water;
       any detection inside radar shadow is occluded and rejected as false alarm.

    Args:
        class_map: Initial 2D classification map (0=bg, 1=water, 2=debris).
        confidence: Initial per-pixel confidence array [0, 1].
        slope: Terrain slope raster in degrees.
        hand: Height Above Nearest Drainage in meters.
        shadow_mask: 2D boolean mask of radar shadow areas.
        max_flood_slope_deg: Maximum slope permissible for flood inundation (default 10 deg).
        max_flood_hand_m: Maximum HAND permissible for flood inundation (default 22m).

    Returns:
        TerrainFilterResult with cleaned class map, attenuated confidence, and filter stats.
    """
    filtered_class = class_map.copy()
    filtered_conf = confidence.copy()

    # Rule 1 & 2: Impossible flood water
    is_water = (class_map == CLASS_WATER)
    impossible_water_slope = is_water & (slope > max_flood_slope_deg)
    impossible_water_hand = is_water & (hand > max_flood_hand_m)
    impossible_water_shadow = is_water & shadow_mask

    impossible_water = impossible_water_slope | impossible_water_hand | impossible_water_shadow
    suppressed_water = int(np.count_nonzero(impossible_water))

    # Remove impossible water
    filtered_class[impossible_water] = CLASS_BACKGROUND
    filtered_conf[impossible_water] = 0.0

    # Rule 3: Debris in radar shadow (cannot observe inside shadow)
    is_debris = (class_map == CLASS_DEBRIS)
    impossible_debris_shadow = is_debris & shadow_mask
    suppressed_debris = int(np.count_nonzero(impossible_debris_shadow))

    filtered_class[impossible_debris_shadow] = CLASS_BACKGROUND
    filtered_conf[impossible_debris_shadow] = 0.0

    # Smooth confidence attenuation in transitional slope/HAND zones for valid water:
    valid_water = (filtered_class == CLASS_WATER)
    # Slope penalty between 6 deg and max_flood_slope_deg
    slope_penalty = np.clip((max_flood_slope_deg - slope) / 4.0, 0.2, 1.0)
    # HAND penalty between 12m and max_flood_hand_m
    hand_penalty = np.clip((max_flood_hand_m - hand) / 10.0, 0.2, 1.0)
    attenuation = np.minimum(slope_penalty, hand_penalty)

    filtered_conf[valid_water] = (filtered_conf[valid_water] * attenuation[valid_water]).astype(np.float32)

    stats = {
        "suppressed_water_pixels": float(suppressed_water),
        "suppressed_debris_pixels": float(suppressed_debris),
        "mean_water_slope": float(np.mean(slope[filtered_class == CLASS_WATER])) if np.any(filtered_class == CLASS_WATER) else 0.0,
        "mean_water_hand": float(np.mean(hand[filtered_class == CLASS_WATER])) if np.any(filtered_class == CLASS_WATER) else 0.0,
    }

    logger.info(
        f"Terrain filtering complete: suppressed {suppressed_water} false water pixels "
        f"and {suppressed_debris} false debris pixels (radar shadow/slope/HAND)."
    )

    # Return drainage mask placeholder or empty if needed
    drainage_mask = (hand <= 2.0) & (slope <= 5.0)

    return TerrainFilterResult(
        filtered_class_map=filtered_class,
        filtered_confidence=filtered_conf,
        slope_deg=slope,
        hand_meters=hand,
        drainage_mask=drainage_mask,
        suppressed_water_pixels=suppressed_water,
        suppressed_debris_pixels=suppressed_debris,
        stats=stats,
    )
