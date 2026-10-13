"""Terrain analysis modules: slope, HAND, radar shadow/layover, and cant_see occlusion."""

from src.terrain.hand_slope import (
    TerrainFilterResult,
    compute_hand,
    compute_slope,
    extract_drainage_network,
    filter_impossible_flood,
)
from src.terrain.shadow_mask import (
    RadarOcclusionResult,
    build_cant_see_mask,
    compute_radar_shadow_layover,
    get_orbit_look_vector,
)

__all__ = [
    "compute_slope",
    "compute_hand",
    "extract_drainage_network",
    "filter_impossible_flood",
    "TerrainFilterResult",
    "get_orbit_look_vector",
    "compute_radar_shadow_layover",
    "build_cant_see_mask",
    "RadarOcclusionResult",
]
