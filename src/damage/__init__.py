"""Damage assessment modules: overlay hazard predictions on OSM infrastructure."""

from src.damage.overlay import (
    DamageOverlayResult,
    assess_bridge_damage,
    assess_building_damage,
    assess_road_damage,
    compute_damage_overlay,
    coord_to_pixel,
    sample_points_along_line,
    sample_polygon_grid,
    save_damage_geojson,
)

__all__ = [
    "coord_to_pixel",
    "sample_points_along_line",
    "sample_polygon_grid",
    "assess_building_damage",
    "assess_road_damage",
    "assess_bridge_damage",
    "compute_damage_overlay",
    "save_damage_geojson",
    "DamageOverlayResult",
]
