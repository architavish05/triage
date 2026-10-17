"""Terrain analysis modules: slope, HAND, radar shadow/layover, and cant_see occlusion."""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from p3_terrain.terrain.hand_slope import (
    TerrainFilterResult,
    compute_hand,
    compute_slope,
    extract_drainage_network,
    filter_and_export_terrain,
    filter_impossible_flood,
    save_raster,
)
from p3_terrain.terrain.shadow_mask import (
    RadarOcclusionResult,
    build_cant_see_mask,
    compute_radar_shadow_layover,
    get_orbit_look_vector,
)


def run_terrain(
    dem: np.ndarray,
    class_map: np.ndarray,
    confidence: np.ndarray,
    orbit_direction: str = "ASCENDING",
    res_meters: float = 30.0,
    max_flood_slope_deg: float = 10.0,
    max_flood_hand_m: float = 22.0,
    cloud_skipped: bool = False,
) -> Dict[str, Any]:
    """Contract interface: Compute slope, HAND, radar shadow/layover, cant_see mask, and filter impossible flood."""
    slope = compute_slope(dem, res_meters=res_meters)
    hand, drainage = compute_hand(dem, slope=slope, res_meters=res_meters)
    shadow_mask, layover_mask = compute_radar_shadow_layover(
        dem, res_meters=res_meters, orbit_direction=orbit_direction
    )
    cloud_mask = np.zeros(class_map.shape, dtype=bool) if cloud_skipped else None
    occlusion = build_cant_see_mask(
        shadow_mask=shadow_mask,
        layover_mask=layover_mask,
        cloud_mask=cloud_mask,
    )
    filtered = filter_impossible_flood(
        class_map=class_map,
        confidence=confidence,
        slope=slope,
        hand=hand,
        shadow_mask=shadow_mask,
        max_flood_slope_deg=max_flood_slope_deg,
        max_flood_hand_m=max_flood_hand_m,
    )
    return {
        "slope": slope,
        "hand": hand,
        "drainage": drainage,
        "shadow_mask": shadow_mask,
        "layover_mask": layover_mask,
        "cant_see_mask": occlusion.cant_see_mask,
        "occlusion": occlusion,
        "filtered": filtered,
        "class_map": filtered.filtered_class_map,
        "confidence": filtered.filtered_confidence,
    }


__all__ = [
    "run_terrain",
    "compute_slope",
    "compute_hand",
    "extract_drainage_network",
    "filter_impossible_flood",
    "filter_and_export_terrain",
    "save_raster",
    "TerrainFilterResult",
    "get_orbit_look_vector",
    "compute_radar_shadow_layover",
    "build_cant_see_mask",
    "RadarOcclusionResult",
]
