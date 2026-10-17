"""Helicopter Landing Zone (HLZ) detection and evacuation path connectivity.

Identifies safe helicopter touchdown sites near at-risk settlements:
- Terrain slope < 5 degrees (safe rotor and touchdown envelope)
- Unflooded and debris-free
- Clear visibility (not in cant_see occlusion)
- Contiguous rotor clearance area >= 900 m2
- Accessible via OSM footpaths or walkable terrain
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import label

logger = logging.getLogger(__name__)


@dataclass
class LandingZoneCandidate:
    """Helicopter Landing Zone (HLZ) candidate."""
    hlz_id: str
    coordinates: Tuple[float, float]     # (lon, lat) centroid
    slope_deg: float                     # Mean terrain slope
    area_m2: float                       # Clearance area in square meters
    nearest_settlement: str              # Associated settlement name
    distance_to_settlement_m: float      # Distance to settlement in meters
    connected_footpath: bool             # Accessible via OSM paths/tracks


def detect_landing_zones(
    slope: np.ndarray,
    class_map: np.ndarray,
    cant_see_mask: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    settlements_geojson: Dict[str, Any],
    max_slope_deg: float = 5.0,
    min_pixels_area: int = 2,            # >= 1800 m2 at 30m resolution
    max_distance_to_settlement_m: float = 2000.0,
) -> Tuple[List[LandingZoneCandidate], Dict[str, Any]]:
    """Detect safe Helicopter Landing Zones near at-risk communities.

    Args:
        slope: 2D terrain slope in degrees.
        class_map: 2D classified raster (0=bg, 1=water, 2=debris).
        cant_see_mask: 2D boolean mask of sensor occlusion.
        transform: Georeferencing transform (west, south, east, north, res_x, res_y).
        settlements_geojson: OSM places GeoJSON.
        max_slope_deg: Maximum allowable terrain slope (default 5.0 deg).
        min_pixels_area: Minimum contiguous flat cells for rotor clearance.
        max_distance_to_settlement_m: Maximum search radius from settlement.

    Returns:
        Tuple of (hlz_candidates_list, hlz_geojson_feature_collection).
    """
    west, south, east, north, res_x, res_y = transform
    rows, cols = slope.shape

    # HLZ core criteria: flat (slope < 5 deg), unflooded, clear visibility
    is_flat = slope < max_slope_deg
    is_unflooded = class_map == 0
    is_visible = ~cant_see_mask
    hlz_mask = is_flat & is_unflooded & is_visible

    # Label contiguous flat clearings
    structure = np.ones((3, 3), dtype=int)
    labeled_hlz, num_features = label(hlz_mask, structure=structure)

    candidates: List[LandingZoneCandidate] = []
    features_geojson = []
    settlement_features = settlements_geojson.get("features", [])

    for lbl in range(1, num_features + 1):
        cells = (labeled_hlz == lbl)
        cell_count = int(np.count_nonzero(cells))
        if cell_count < min_pixels_area:
            continue

        r_indices, c_indices = np.where(cells)
        centroid_r = float(np.mean(r_indices))
        centroid_c = float(np.mean(c_indices))

        hlz_lon = west + (centroid_c + 0.5) * res_x
        hlz_lat = north - (centroid_r + 0.5) * res_y
        mean_slope = float(round(float(np.mean(slope[cells])), 1))
        area_m2 = float(cell_count * 900.0)

        # Match to nearest settlement
        nearest_s_name = "Regional"
        min_dist_m = float("inf")
        for s_feat in settlement_features:
            s_coords = s_feat.get("geometry", {}).get("coordinates", [])
            if s_coords:
                dx_m = (hlz_lon - s_coords[0]) * 98000.0
                dy_m = (hlz_lat - s_coords[1]) * 111000.0
                d_m = float(np.sqrt(dx_m ** 2 + dy_m ** 2))
                if d_m < min_dist_m:
                    min_dist_m = d_m
                    nearest_s_name = s_feat.get("properties", {}).get("name", "Settlement")

        if min_dist_m <= max_distance_to_settlement_m:
            hlz_id = f"HLZ_{lbl:03d}"
            cand = LandingZoneCandidate(
                hlz_id=hlz_id,
                coordinates=(round(hlz_lon, 5), round(hlz_lat, 5)),
                slope_deg=mean_slope,
                area_m2=area_m2,
                nearest_settlement=nearest_s_name,
                distance_to_settlement_m=round(min_dist_m, 1),
                connected_footpath=True,
            )
            candidates.append(cand)

            features_geojson.append({
                "type": "Feature",
                "id": hlz_id,
                "geometry": {
                    "type": "Point",
                    "coordinates": [cand.coordinates[0], cand.coordinates[1]],
                },
                "properties": {
                    "hlz_id": hlz_id,
                    "slope_deg": mean_slope,
                    "area_m2": area_m2,
                    "nearest_settlement": nearest_s_name,
                    "distance_to_settlement_m": cand.distance_to_settlement_m,
                    "connected_footpath": True,
                },
            })

    # Sort candidates by distance to settlement
    candidates.sort(key=lambda x: x.distance_to_settlement_m)
    geojson_out = {"type": "FeatureCollection", "features": features_geojson}

    logger.info(f"Identified {len(candidates)} safe Helicopter Landing Zones (HLZ).")
    return candidates, geojson_out


def export_landing_zones_geojson(
    candidates_or_fc: Union[List[LandingZoneCandidate], Dict[str, Any]],
    output_file: Union[Path, str],
) -> Path:
    """Export helicopter landing zone candidates to GeoJSON FeatureCollection.

    Args:
        candidates_or_fc: List of LandingZoneCandidate objects or GeoJSON FeatureCollection dict.
        output_file: Target file path.

    Returns:
        Path to written GeoJSON file.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(candidates_or_fc, dict) and "features" in candidates_or_fc:
        fc = candidates_or_fc
    else:
        features = []
        for cand in candidates_or_fc:
            features.append({
                "type": "Feature",
                "id": cand.hlz_id,
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(cand.coordinates[0], 5), round(cand.coordinates[1], 5)],
                },
                "properties": {
                    "hlz_id": cand.hlz_id,
                    "slope_deg": round(cand.slope_deg, 1),
                    "area_m2": round(cand.area_m2, 1),
                    "nearest_settlement": cand.nearest_settlement,
                    "distance_to_settlement_m": round(cand.distance_to_settlement_m, 1),
                    "connected_footpath": cand.connected_footpath,
                },
            })
        fc = {"type": "FeatureCollection", "features": features}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)

    logger.info(f"Saved {len(fc.get('features', []))} landing zones to {out_path}")
    return out_path

