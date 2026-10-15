"""Damage assessment modules: overlay hazard predictions on OSM infrastructure."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from p4_app.damage.overlay import (
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


def run_damage(
    osm_data: Any,
    class_map: np.ndarray,
    confidence: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    damage_threshold: float = 0.35,
    output_dir: Optional[Path] = None,
) -> DamageOverlayResult:
    """Contract interface: Compute infrastructure damage overlay and optionally save GeoJSONs."""
    res = compute_damage_overlay(
        osm_data=osm_data,
        class_map=class_map,
        confidence=confidence,
        transform=transform,
        damage_threshold=damage_threshold,
    )
    if output_dir is not None:
        save_damage_geojson(res, Path(output_dir))
    return res


__all__ = [
    "run_damage",
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
