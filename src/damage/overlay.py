"""Damage overlay assessment: Intersecting hazard confidence with OSM assets.

Calculates asset-level hit probability (hit_prob) and damage classification for:
- OSM buildings (polygons/points)
- OSM highways and roads (LineStrings)
- OSM bridges (LineStrings/Points)
Produces enriched GeoJSON FeatureCollections and exact quantitative damage tallies.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.detect.fuse import CLASS_BACKGROUND, CLASS_DEBRIS, CLASS_WATER

logger = logging.getLogger(__name__)


@dataclass
class DamageOverlayResult:
    """Damage overlay assessment product containing enriched GeoJSONs and counts."""
    damaged_buildings: Dict[str, Any]    # GeoJSON FeatureCollection
    damaged_roads: Dict[str, Any]        # GeoJSON FeatureCollection
    damaged_bridges: Dict[str, Any]      # GeoJSON FeatureCollection
    summary: Dict[str, Any]             # Structured quantitative summary


def coord_to_pixel(
    lon: float,
    lat: float,
    transform: Tuple[float, float, float, float, float, float],
    shape: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """Convert geographic coordinates (WGS84 lon, lat) to raster (row, col) index.

    Transform tuple format: (west, south, east, north, res_x, res_y)
    """
    west, south, east, north, res_x, res_y = transform
    rows, cols = shape

    if not (west <= lon <= east and south <= lat <= north):
        return None

    col = int(np.clip((lon - west) / max(res_x, 1e-12), 0, cols - 1))
    row = int(np.clip((north - lat) / max(res_y, 1e-12), 0, rows - 1))
    return row, col


def sample_points_along_line(
    coords: List[List[float]],
    num_samples_per_segment: int = 5,
) -> List[Tuple[float, float]]:
    """Linearly interpolate sample coordinates along a LineString."""
    samples = []
    if not coords:
        return samples

    for i in range(len(coords) - 1):
        x1, y1 = coords[i][:2]
        x2, y2 = coords[i + 1][:2]
        for t in np.linspace(0.0, 1.0, num_samples_per_segment, endpoint=False):
            samples.append((float(x1 + t * (x2 - x1)), float(y1 + t * (y2 - y1))))
    samples.append((float(coords[-1][0]), float(coords[-1][1])))
    return samples


def sample_polygon_grid(
    poly_coords: List[List[float]],
    transform: Tuple[float, float, float, float, float, float],
    shape: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Extract raster pixel indices covered by a polygon footprint."""
    pixels = set()
    ring = poly_coords[0] if isinstance(poly_coords[0][0], (list, tuple)) else poly_coords

    # Bounding box of polygon
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    p_min = coord_to_pixel(min_lon, max_lat, transform, shape)
    p_max = coord_to_pixel(max_lon, min_lat, transform, shape)

    if p_min is None or p_max is None:
        # Fallback: test vertices directly
        for pt in ring:
            pix = coord_to_pixel(pt[0], pt[1], transform, shape)
            if pix:
                pixels.add(pix)
        return list(pixels)

    r_start, c_start = min(p_min[0], p_max[0]), min(p_min[1], p_max[1])
    r_end, c_end = max(p_min[0], p_max[0]), max(p_min[1], p_max[1])

    # Ray-casting point-in-polygon on raster grid
    for r in range(r_start, r_end + 1):
        for c in range(c_start, c_end + 1):
            west, _, _, north, res_x, res_y = transform
            px_lon = west + (c + 0.5) * res_x
            px_lat = north - (r + 0.5) * res_y

            # Point in polygon test
            inside = False
            n_pts = len(ring)
            for i in range(n_pts):
                j = (i - 1 + n_pts) % n_pts
                xi, yi = ring[i][:2]
                xj, yj = ring[j][:2]
                if ((yi > px_lat) != (yj > px_lat)) and (
                    px_lon < (xj - xi) * (px_lat - yi) / (yj - yi + 1e-12) + xi
                ):
                    inside = not inside
            if inside:
                pixels.add((r, c))

    if not pixels:
        # If building is smaller than 1 pixel, assign centroid pixel
        mid_lon = (min_lon + max_lon) / 2.0
        mid_lat = (min_lat + max_lat) / 2.0
        centroid_pix = coord_to_pixel(mid_lon, mid_lat, transform, shape)
        if centroid_pix:
            pixels.add(centroid_pix)

    return list(pixels)


def assess_building_damage(
    buildings_geojson: Dict[str, Any],
    class_map: np.ndarray,
    confidence: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    damage_threshold: float = 0.35,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Intersect building polygons with hazard layers and compute hit_prob."""
    shape = class_map.shape
    features = buildings_geojson.get("features", [])
    output_features = []

    flood_hit = 0
    debris_hit = 0
    total_hit = 0

    for feat in features:
        geom = feat.get("geometry", {})
        props = dict(feat.get("properties", {}))
        g_type = geom.get("type", "")
        coords = geom.get("coordinates", [])

        pixels = []
        if g_type == "Polygon" and coords:
            pixels = sample_polygon_grid(coords, transform, shape)
        elif g_type == "Point" and coords:
            pix = coord_to_pixel(coords[0], coords[1], transform, shape)
            if pix:
                pixels = [pix]

        if not pixels:
            props["hit_prob"] = 0.0
            props["hazard_type"] = "none"
            props["is_damaged"] = False
        else:
            conf_vals = [float(confidence[r, c]) for r, c in pixels]
            classes = [int(class_map[r, c]) for r, c in pixels]

            # Max confidence determines hit probability
            hit_prob = float(round(float(np.max(conf_vals)), 3))
            props["hit_prob"] = hit_prob

            # Dominant hazard type
            water_count = classes.count(CLASS_WATER)
            debris_count = classes.count(CLASS_DEBRIS)

            if hit_prob >= damage_threshold:
                props["is_damaged"] = True
                total_hit += 1
                if debris_count > water_count:
                    props["hazard_type"] = "debris"
                    debris_hit += 1
                else:
                    props["hazard_type"] = "flood"
                    flood_hit += 1
            else:
                props["is_damaged"] = False
                props["hazard_type"] = "none"

        output_features.append({
            "type": "Feature",
            "id": feat.get("id"),
            "geometry": geom,
            "properties": props,
        })

    summary = {
        "total_buildings": len(features),
        "damaged_buildings_count": total_hit,
        "flood_damaged_buildings": flood_hit,
        "debris_damaged_buildings": debris_hit,
    }

    return {"type": "FeatureCollection", "features": output_features}, summary


def assess_road_damage(
    highways_geojson: Dict[str, Any],
    class_map: np.ndarray,
    confidence: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    damage_threshold: float = 0.35,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Intersect highway lines with hazard layers and compute hit_prob and impassability."""
    shape = class_map.shape
    features = highways_geojson.get("features", [])
    output_features = []

    total_roads = len(features)
    impassable_roads = 0
    total_length_km = 0.0
    damaged_length_km = 0.0

    for feat in features:
        geom = feat.get("geometry", {})
        props = dict(feat.get("properties", {}))
        g_type = geom.get("type", "")
        coords = geom.get("coordinates", [])

        if g_type != "LineString" or not coords:
            props["hit_prob"] = 0.0
            props["is_impassable"] = False
            output_features.append({"type": "Feature", "id": feat.get("id"), "geometry": geom, "properties": props})
            continue

        sample_pts = sample_points_along_line(coords)
        conf_vals = []
        class_vals = []

        for pt_lon, pt_lat in sample_pts:
            pix = coord_to_pixel(pt_lon, pt_lat, transform, shape)
            if pix:
                conf_vals.append(float(confidence[pix[0], pix[1]]))
                class_vals.append(int(class_map[pix[0], pix[1]]))

        # Estimate segment length in kilometers (~111 km per deg approx)
        lons = [p[0] for p in coords]
        lats = [p[1] for p in coords]
        dx_deg = max(lons) - min(lons)
        dy_deg = max(lats) - min(lats)
        est_len_km = round(np.sqrt((dx_deg * 98.0) ** 2 + (dy_deg * 111.0) ** 2), 2)
        total_length_km += est_len_km

        if not conf_vals:
            props["hit_prob"] = 0.0
            props["is_impassable"] = False
            props["length_km"] = est_len_km
            props["damaged_length_km"] = 0.0
        else:
            max_hit = float(round(float(np.max(conf_vals)), 3))
            props["hit_prob"] = max_hit
            props["length_km"] = est_len_km

            # Proportion of road samples exceeding hazard threshold
            damaged_samples = sum(1 for c in conf_vals if c >= damage_threshold)
            damage_ratio = damaged_samples / max(len(conf_vals), 1)
            seg_damaged_km = round(est_len_km * damage_ratio, 2)
            props["damaged_length_km"] = seg_damaged_km

            if max_hit >= damage_threshold:
                props["is_impassable"] = True
                impassable_roads += 1
                damaged_length_km += seg_damaged_km
                props["hazard_type"] = "debris" if class_vals.count(CLASS_DEBRIS) > class_vals.count(CLASS_WATER) else "flood"
            else:
                props["is_impassable"] = False
                props["hazard_type"] = "none"

        output_features.append({
            "type": "Feature",
            "id": feat.get("id"),
            "geometry": geom,
            "properties": props,
        })

    summary = {
        "total_roads_count": total_roads,
        "impassable_roads_count": impassable_roads,
        "total_roads_km": round(total_length_km, 2),
        "damaged_roads_km": round(damaged_length_km, 2),
    }

    return {"type": "FeatureCollection", "features": output_features}, summary


def assess_bridge_damage(
    bridges_geojson: Dict[str, Any],
    class_map: np.ndarray,
    confidence: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    damage_threshold: float = 0.35,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Intersect critical river bridge structures with hazard layers."""
    shape = class_map.shape
    features = bridges_geojson.get("features", [])
    output_features = []

    damaged_bridges = 0

    for feat in features:
        geom = feat.get("geometry", {})
        props = dict(feat.get("properties", {}))
        g_type = geom.get("type", "")
        coords = geom.get("coordinates", [])

        pts = []
        if g_type == "Point" and coords:
            pts = [(coords[0], coords[1])]
        elif g_type == "LineString" and coords:
            pts = sample_points_along_line(coords, num_samples_per_segment=3)

        conf_vals = []
        class_vals = []
        for pt_lon, pt_lat in pts:
            pix = coord_to_pixel(pt_lon, pt_lat, transform, shape)
            if pix:
                conf_vals.append(float(confidence[pix[0], pix[1]]))
                class_vals.append(int(class_map[pix[0], pix[1]]))

        if not conf_vals:
            props["hit_prob"] = 0.0
            props["is_damaged"] = False
            props["bridge_status"] = "operational"
        else:
            hit_prob = float(round(float(np.max(conf_vals)), 3))
            props["hit_prob"] = hit_prob
            if hit_prob >= damage_threshold:
                props["is_damaged"] = True
                props["bridge_status"] = "washed_out_or_damaged"
                damaged_bridges += 1
            else:
                props["is_damaged"] = False
                props["bridge_status"] = "operational"

        output_features.append({
            "type": "Feature",
            "id": feat.get("id"),
            "geometry": geom,
            "properties": props,
        })

    summary = {
        "total_bridges": len(features),
        "damaged_bridges_count": damaged_bridges,
    }

    return {"type": "FeatureCollection", "features": output_features}, summary


def compute_damage_overlay(
    osm_data: Any,
    class_map: np.ndarray,
    confidence: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    damage_threshold: float = 0.35,
) -> DamageOverlayResult:
    """Execute complete infrastructure damage assessment across all OSM asset layers.

    Args:
        osm_data: OSMData dataclass containing buildings, highways, bridges, etc.
        class_map: 2D classified hazard array (0=bg, 1=water, 2=debris).
        confidence: 2D per-pixel confidence grid in [0, 1].
        transform: Georeferencing transform tuple (west, south, east, north, res_x, res_y).
        damage_threshold: Minimum hazard confidence classifying an asset as hit/damaged.

    Returns:
        DamageOverlayResult containing enriched GeoJSONs and total summary metrics.
    """
    logger.info("Computing asset damage overlay for buildings, highways, and bridges...")

    buildings_res, bldg_summary = assess_building_damage(
        osm_data.buildings,
        class_map,
        confidence,
        transform,
        damage_threshold=damage_threshold,
    )

    roads_res, roads_summary = assess_road_damage(
        osm_data.highways,
        class_map,
        confidence,
        transform,
        damage_threshold=damage_threshold,
    )

    bridges_res, bridge_summary = assess_bridge_damage(
        osm_data.bridges,
        class_map,
        confidence,
        transform,
        damage_threshold=damage_threshold,
    )

    consolidated_summary = {
        **bldg_summary,
        **roads_summary,
        **bridge_summary,
        "damage_threshold_used": damage_threshold,
    }

    logger.info(
        f"Damage overlay complete: {bldg_summary['damaged_buildings_count']} buildings, "
        f"{roads_summary['impassable_roads_count']} impassable road segments ({roads_summary['damaged_roads_km']} km), "
        f"{bridge_summary['damaged_bridges_count']} bridges damaged."
    )

    return DamageOverlayResult(
        damaged_buildings=buildings_res,
        damaged_roads=roads_res,
        damaged_bridges=bridges_res,
        summary=consolidated_summary,
    )


def save_damage_geojson(damage_result: DamageOverlayResult, output_dir: Path) -> Dict[str, Path]:
    """Persist enriched damage GeoJSON layers and summary JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "buildings": output_dir / "damaged_buildings.geojson",
        "roads": output_dir / "damaged_roads.geojson",
        "bridges": output_dir / "damaged_bridges.geojson",
        "summary": output_dir / "damage_summary.json",
    }

    with open(paths["buildings"], "w", encoding="utf-8") as f:
        json.dump(damage_result.damaged_buildings, f, indent=2)

    with open(paths["roads"], "w", encoding="utf-8") as f:
        json.dump(damage_result.damaged_roads, f, indent=2)

    with open(paths["bridges"], "w", encoding="utf-8") as f:
        json.dump(damage_result.damaged_bridges, f, indent=2)

    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(damage_result.summary, f, indent=2)

    logger.debug(f"Saved damage GeoJSONs and summary to {output_dir}")
    return paths
