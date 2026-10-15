"""Comprehensive test suite for Person 3 deliverables: Phases 3, 6, and 7.

Run with:
    py -3.10 -m pytest src/rescue/tests_p3.py -v
"""

import csv
import json
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

# Phase 3 imports
from src.terrain.hand_slope import (
    compute_hand,
    compute_slope,
    filter_and_export_terrain,
    filter_impossible_flood,
    save_raster,
)
from src.terrain.shadow_mask import (
    build_cant_see_mask,
    compute_radar_shadow_layover,
    get_orbit_look_vector,
)

# Phase 6 imports
from src.flowpath.trace import (
    DownstreamTrace,
    SettlementArrival,
    buffer_flow_path,
    calculate_settlement_arrival_times,
    compute_d8_flow,
    export_downstream_trace_json,
    export_flowpath_geojson,
    fill_depressions,
    trace_downstream_path,
)

# Phase 7 imports
from src.network.cutoff import SettlementIsolation
from src.rescue.dry_islands import check_settlement_on_dry_island, identify_dry_islands
from src.rescue.escape_time import compute_high_ground_mask, evaluate_settlement_escape_time
from src.rescue.landing_zones import (
    LandingZoneCandidate,
    detect_landing_zones,
    export_landing_zones_geojson,
)
from src.rescue.trapped_risk import (
    RescuePriorityItem,
    compute_rescue_priorities,
    export_rescue_priority_json,
    export_trapped_csv,
)


# =====================================================================
# Phase 3: Terrain Filtering & Occlusion Tests
# =====================================================================

class TestPhase3Terrain:

    def test_save_raster_pil_and_roundtrip(self, tmp_path):
        """Verify save_raster writes valid TIFF readable by PIL/rasterio."""
        arr = np.random.uniform(0.0, 1.0, (32, 32)).astype(np.float32)
        out_file = tmp_path / "test_raster.tif"
        save_raster(out_file, arr)

        assert out_file.exists()
        loaded = np.array(Image.open(out_file))
        assert loaded.shape == (32, 32)
        assert loaded.dtype == np.uint8

    def test_filter_and_export_terrain(self, tmp_path):
        """Verify terrain filtering purges slopes >10 deg and HAND >22m, and exports rasters."""
        shape = (30, 30)
        class_map = np.zeros(shape, dtype=np.uint8)
        conf = np.zeros(shape, dtype=np.float32)
        slope = np.full(shape, 2.0, dtype=np.float32)
        hand = np.full(shape, 5.0, dtype=np.float32)
        shadow = np.zeros(shape, dtype=bool)

        # 1. Valid water
        class_map[5:10, 5:10] = 1
        conf[5:10, 5:10] = 0.80

        # 2. Impossible steep slope water (slope = 15 deg > 10 deg)
        class_map[15:20, 5:10] = 1
        conf[15:20, 5:10] = 0.85
        slope[15:20, 5:10] = 15.0

        # 3. Impossible high HAND water (hand = 35m > 22m)
        class_map[5:10, 15:20] = 1
        conf[5:10, 15:20] = 0.90
        hand[5:10, 15:20] = 35.0

        # 4. Water in radar shadow
        class_map[15:20, 15:20] = 1
        conf[15:20, 15:20] = 0.75
        shadow[15:20, 15:20] = True

        res, cant_see_p, conf_p = filter_and_export_terrain(
            class_map=class_map,
            confidence=conf,
            slope=slope,
            hand=hand,
            shadow_mask=shadow,
            output_dir=tmp_path,
        )

        assert cant_see_p.exists()
        assert conf_p.exists()
        # 3 patches of 5x5 = 75 pixels suppressed
        assert res.suppressed_water_pixels == 75
        assert res.filtered_class_map[7, 7] == 1
        assert res.filtered_class_map[17, 7] == 0
        assert res.filtered_class_map[7, 17] == 0
        assert res.filtered_class_map[17, 17] == 0


# =====================================================================
# Phase 6: Flowpath Tracing & Arrival Estimation Tests
# =====================================================================

class TestPhase6Flowpath:

    def test_export_flowpath_geojson_structure(self, tmp_path):
        """Verify export_flowpath_geojson generates contract-compliant GeoJSON."""
        trace = DownstreamTrace(
            path_pixels=[(0, 0), (1, 1), (2, 2)],
            path_coords=[(85.32, 28.10), (85.33, 28.11), (85.34, 28.12)],
            cumulative_dist_km=[0.0, 1.2, 2.5],
            elevations_m=[2200.0, 1900.0, 1600.0],
            cumulative_time_mins=[0.0, 3.2, 7.1],
            total_length_km=2.5,
            total_time_hours=0.12,
        )

        arrivals = [
            SettlementArrival(
                rank_order=1,
                settlement_id="place/10",
                name="Downstream Village",
                place_type="village",
                coordinates=(85.34, 28.12),
                distance_from_source_km=2.5,
                elevation_m=1600.0,
                arrival_time_mins=7.1,
                arrival_time_hours=0.12,
                warning_lead_time_urgency="IMMEDIATE (<30m)",
            )
        ]

        out_geojson = tmp_path / "flowpath.geojson"
        export_flowpath_geojson(trace, out_geojson, settlement_arrivals=arrivals)

        assert out_geojson.exists()
        with open(out_geojson, "r", encoding="utf-8") as f:
            fc = json.load(f)

        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) == 2  # 1 LineString + 1 Point
        line = fc["features"][0]
        assert line["geometry"]["type"] == "LineString"
        assert len(line["geometry"]["coordinates"]) == 3
        assert line["properties"]["length_km"] == 2.5
        assert len(line["properties"]["settlement_arrivals"]) == 1

        pt = fc["features"][1]
        assert pt["geometry"]["type"] == "Point"
        assert pt["properties"]["name"] == "Downstream Village"
        assert pt["properties"]["warning_lead_time_urgency"] == "IMMEDIATE (<30m)"


# =====================================================================
# Phase 7: Rescue Intelligence & HLZ Tests
# =====================================================================

class TestPhase7Rescue:

    def test_hlz_detection_slope_and_clearance(self, tmp_path):
        """Verify HLZ detection enforces slope < 5 deg and exports GeoJSON."""
        shape = (30, 30)
        transform = (85.2, 27.8, 85.4, 28.0, 0.00666, 0.00666)
        slope = np.full(shape, 12.0, dtype=np.float32)  # Steep terrain
        cls = np.zeros(shape, dtype=np.uint8)
        cant_see = np.zeros(shape, dtype=bool)

        # Flat meadow at rows 10..13, cols 10..13 (slope = 3 deg)
        slope[10:14, 10:14] = 3.0

        settlements_fc = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "id": "place/1",
                "geometry": {"type": "Point", "coordinates": [85.28, 27.92]},
                "properties": {"name": "Test Settlement"},
            }],
        }

        candidates, hlz_fc = detect_landing_zones(
            slope=slope,
            class_map=cls,
            cant_see_mask=cant_see,
            transform=transform,
            settlements_geojson=settlements_fc,
            max_slope_deg=5.0,
            min_pixels_area=2,
        )

        assert len(candidates) >= 1
        for cand in candidates:
            assert cand.slope_deg < 5.0
            assert cand.area_m2 >= 1800.0

        out_hlz = tmp_path / "landing_zones.geojson"
        export_landing_zones_geojson(candidates, out_hlz)
        assert out_hlz.exists()

    def test_trapped_csv_export_schema(self, tmp_path):
        """Verify export_trapped_csv produces exact CONTRACT.md CSV schema."""
        item = RescuePriorityItem(
            rank=1,
            settlement_id="place/3",
            settlement_name="Tatopani",
            place_type="village",
            coordinates=(85.3875, 28.0625),
            trapped_risk=0.45,
            cutoff_prob=0.95,
            buildings_at_risk=4,
            stranded_population_estimate=20,
            population_data_type="estimate",
            household_size_assumption=5.0,
            priority_score=1.71,
            is_on_dry_island=False,
            escape_possible=True,
            nearest_hospital="Barhabise Primary Health Centre",
            recommended_action="EMERGENCY SUPPLY RESUPPLY (CUT-OFF COMMUNITY)",
        )

        out_csv = tmp_path / "trapped.csv"
        export_trapped_csv([item], out_csv)

        assert out_csv.exists()
        with open(out_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert row["rank"] == "1"
        assert row["settlement_name"] == "Tatopani"
        assert row["population_data_type"] == "estimate"
        assert row["priority_score"] == "1.71"
        assert row["coordinates"] == "[85.3875, 28.0625]"
        assert row["is_on_dry_island"] == "False"
        assert row["escape_possible"] == "True"


# =====================================================================
# End-to-End Integration against data/mock_run/
# =====================================================================

class TestMockRunDeliverables:

    def test_mock_run_rasters_exist_and_valid(self):
        """Verify cant_see.tif and flood_conf.tif exist and have correct formats."""
        run_dir = Path("data/mock_run")
        cant_see_path = run_dir / "cant_see.tif"
        conf_path = run_dir / "flood_conf.tif"

        assert cant_see_path.exists()
        assert conf_path.exists()

        cant_see_im = Image.open(cant_see_path)
        conf_im = Image.open(conf_path)

        assert cant_see_im.size == (128, 128)
        assert conf_im.size == (128, 128)

        cant_see_arr = np.array(cant_see_im)
        conf_arr = np.array(conf_im)

        assert cant_see_arr.shape == (128, 128)
        assert conf_arr.shape == (128, 128)
        assert set(np.unique(cant_see_arr)).issubset({0, 255})
        assert conf_arr.max() <= 255

    def test_mock_run_flowpath_geojson(self):
        """Verify flowpath.geojson has valid LineString and settlement arrival times."""
        flowpath_path = Path("data/mock_run/flowpath.geojson")
        assert flowpath_path.exists()

        with open(flowpath_path, "r", encoding="utf-8") as f:
            fc = json.load(f)

        assert fc["type"] == "FeatureCollection"
        assert len(fc["features"]) >= 1
        line = fc["features"][0]
        assert line["geometry"]["type"] == "LineString"
        assert len(line["geometry"]["coordinates"]) == 29
        assert line["properties"]["length_km"] >= 1.0

    def test_mock_run_rescue_outputs(self):
        """Verify trapped.csv and landing_zones.geojson meet all requirements."""
        run_dir = Path("data/mock_run")
        trapped_path = run_dir / "trapped.csv"
        hlz_path = run_dir / "landing_zones.geojson"
        if not hlz_path.exists():
            hlz_path = run_dir / "geojson" / "landing_zones.geojson"

        assert trapped_path.exists()
        assert hlz_path.exists()

        # Check trapped.csv
        with open(trapped_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        # Rank 1 must be Tatopani (highest cut-off score)
        assert rows[0]["settlement_name"] == "Tatopani"
        assert float(rows[0]["priority_score"]) > float(rows[1]["priority_score"])
        # Invariant: population must be labelled 'estimate'
        for r in rows:
            assert r["population_data_type"] == "estimate"

        # Check landing_zones.geojson
        with open(hlz_path, "r", encoding="utf-8") as f:
            hlz_fc = json.load(f)

        assert len(hlz_fc["features"]) >= 1
        for feat in hlz_fc["features"]:
            p = feat["properties"]
            assert float(p["slope_deg"]) < 5.0
            assert float(p["area_m2"]) >= 1800.0
