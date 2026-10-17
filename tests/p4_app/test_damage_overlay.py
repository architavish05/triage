"""Unit tests for Phase 4 damage overlay assessment and GeoJSON output."""

import json
from pathlib import Path
import numpy as np
from p4_app.damage.overlay import (
    assess_bridge_damage,
    assess_building_damage,
    assess_road_damage,
    compute_damage_overlay,
    coord_to_pixel,
    sample_points_along_line,
    save_damage_geojson,
)
from p2_ai.detect.fuse import CLASS_BACKGROUND, CLASS_DEBRIS, CLASS_WATER
from p1_data.fetch.osm import generate_synthetic_osm


def test_coord_to_pixel():
    """Verify geographic coordinate to raster pixel transformation."""
    # bbox: west=85.0, south=27.0, east=86.0, north=28.0, 100x100 grid
    transform = (85.0, 27.0, 86.0, 28.0, 0.01, 0.01)
    shape = (100, 100)

    # Top-left corner (85.0, 28.0)
    assert coord_to_pixel(85.0, 28.0, transform, shape) == (0, 0)
    # Bottom-right corner (86.0, 27.0)
    assert coord_to_pixel(86.0, 27.0, transform, shape) == (99, 99)
    # Center (85.5, 27.5)
    assert coord_to_pixel(85.5, 27.5, transform, shape) == (50, 50)
    # Out of bounds
    assert coord_to_pixel(84.0, 27.5, transform, shape) is None


def test_sample_points_along_line():
    """Verify point interpolation along a road LineString."""
    coords = [[85.0, 27.0], [85.1, 27.1]]
    pts = sample_points_along_line(coords, num_samples_per_segment=4)
    assert len(pts) == 5
    assert pts[0] == (85.0, 27.0)
    assert pts[-1] == (85.1, 27.1)


def test_assess_building_damage():
    """Verify building damage intersection and hit_prob classification."""
    shape = (50, 50)
    transform = (85.0, 27.0, 85.5, 27.5, 0.01, 0.01)
    class_map = np.full(shape, CLASS_BACKGROUND, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)

    # Put a flood hazard in pixel (20, 20)
    class_map[18:25, 18:25] = CLASS_WATER
    confidence[18:25, 18:25] = 0.82

    # Building 1 located in the flood zone
    bldg_in_flood = {
        "type": "Feature",
        "id": "bldg_1",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[85.20, 27.29], [85.22, 27.29], [85.22, 27.31], [85.20, 27.31], [85.20, 27.29]]],
        },
        "properties": {"name": "School"},
    }

    # Building 2 located on safe high ground
    bldg_safe = {
        "type": "Feature",
        "id": "bldg_2",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[85.02, 27.45], [85.04, 27.45], [85.04, 27.47], [85.02, 27.47], [85.02, 27.45]]],
        },
        "properties": {"name": "Safe House"},
    }

    geojson = {"type": "FeatureCollection", "features": [bldg_in_flood, bldg_safe]}
    res_fc, summary = assess_building_damage(geojson, class_map, confidence, transform, damage_threshold=0.35)

    feat_flood = res_fc["features"][0]
    feat_safe = res_fc["features"][1]

    assert feat_flood["properties"]["is_damaged"] is True
    assert feat_flood["properties"]["hit_prob"] > 0.7
    assert feat_flood["properties"]["hazard_type"] == "flood"

    assert feat_safe["properties"]["is_damaged"] is False
    assert feat_safe["properties"]["hit_prob"] == 0.0

    assert summary["damaged_buildings_count"] == 1
    assert summary["total_buildings"] == 2


def test_assess_road_and_bridge_damage():
    """Verify road and bridge damage calculation."""
    shape = (50, 50)
    transform = (85.0, 27.0, 85.5, 27.5, 0.01, 0.01)
    class_map = np.full(shape, CLASS_BACKGROUND, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)

    # Debris flow crossing the highway at center
    class_map[22:28, 22:28] = CLASS_DEBRIS
    confidence[22:28, 22:28] = 0.90

    # Road crossing center
    road_cut = {
        "type": "Feature",
        "id": "hwy_1",
        "geometry": {"type": "LineString", "coordinates": [[85.1, 27.25], [85.3, 27.25]]},
        "properties": {"name": "Araniko Highway"},
    }
    roads_fc, r_sum = assess_road_damage(
        {"type": "FeatureCollection", "features": [road_cut]},
        class_map,
        confidence,
        transform,
        damage_threshold=0.35,
    )
    assert roads_fc["features"][0]["properties"]["is_impassable"] is True
    assert roads_fc["features"][0]["properties"]["hit_prob"] >= 0.8
    assert r_sum["impassable_roads_count"] == 1

    # Bridge at the crossing
    bridge_cut = {
        "type": "Feature",
        "id": "bridge_1",
        "geometry": {"type": "Point", "coordinates": [85.24, 27.25]},
        "properties": {"name": "Friendship Bridge"},
    }
    bridge_fc, b_sum = assess_bridge_damage(
        {"type": "FeatureCollection", "features": [bridge_cut]},
        class_map,
        confidence,
        transform,
        damage_threshold=0.35,
    )
    assert bridge_fc["features"][0]["properties"]["is_damaged"] is True
    assert bridge_fc["features"][0]["properties"]["bridge_status"] == "washed_out_or_damaged"
    assert b_sum["damaged_bridges_count"] == 1


def test_compute_damage_overlay_end_to_end(tmp_path):
    """Verify compute_damage_overlay integrates with OSMData and saves valid GeoJSON layers."""
    bbox = (85.1, 27.8, 85.5, 28.2)
    osm_data = generate_synthetic_osm(bbox)

    shape = (80, 80)
    w, s, e, n = bbox
    res_x = (e - w) / shape[1]
    res_y = (n - s) / shape[0]
    transform = (w, s, e, n, res_x, res_y)

    class_map = np.zeros(shape, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)

    # Inject water and debris hazard in center river corridor
    class_map[30:50, 35:45] = CLASS_WATER
    confidence[30:50, 35:45] = 0.88
    class_map[20:30, 35:45] = CLASS_DEBRIS
    confidence[20:30, 35:45] = 0.79

    damage_result = compute_damage_overlay(
        osm_data=osm_data,
        class_map=class_map,
        confidence=confidence,
        transform=transform,
        damage_threshold=0.35,
    )

    assert "total_buildings" in damage_result.summary
    assert "total_roads_km" in damage_result.summary
    assert "damaged_bridges_count" in damage_result.summary

    # Save to directory and verify GeoJSON files
    out_dir = tmp_path / "damage_outputs"
    paths = save_damage_geojson(damage_result, out_dir)

    assert "assets_damage" in paths
    assert (out_dir / "assets_damage.geojson").exists()
    assert (out_dir / "damage_summary.json").exists()

    for key, p in paths.items():
        assert p.exists()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data is not None
