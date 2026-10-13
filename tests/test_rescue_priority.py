"""Unit tests for Phase 7 rescue intelligence and priority ranking."""

import numpy as np
import pytest

from src.detect.fuse import CLASS_BACKGROUND, CLASS_WATER
from src.network.cutoff import SettlementIsolation
from src.rescue.dry_islands import check_settlement_on_dry_island, identify_dry_islands
from src.rescue.escape_time import compute_high_ground_mask, evaluate_settlement_escape_time
from src.rescue.landing_zones import detect_landing_zones
from src.rescue.trapped_risk import compute_rescue_priorities


def test_dry_islands_detection():
    """Verify that unflooded patches enclosed by flood water are identified as dry islands."""
    # 40x40 grid, entire perimeter is safe high ground (HAND=50m)
    shape = (40, 40)
    class_map = np.full(shape, CLASS_BACKGROUND, dtype=np.uint8)
    hand = np.full(shape, 35.0, dtype=np.float32)

    # In the center (HAND=5m), create a ring of water (flood moat) enclosing an unflooded dry patch (16:24, 16:24)
    hand[10:30, 10:30] = 5.0
    # Ring of flood water
    class_map[12:28, 12:15] = CLASS_WATER  # West wall
    class_map[12:28, 25:28] = CLASS_WATER  # East wall
    class_map[12:15, 12:28] = CLASS_WATER  # North wall
    class_map[25:28, 12:28] = CLASS_WATER  # South wall

    dry_island_mask, island_props = identify_dry_islands(class_map, hand, safe_hand_threshold=25.0)

    # The enclosed center dry ground (r=18..22, c=18..22) must be True in dry_island_mask
    assert np.all(dry_island_mask[18:22, 18:22])
    # The outer unflooded ground (r=2, c=2) must be False (connected to safe mainland)
    assert not dry_island_mask[2, 2]
    assert len(island_props) >= 1


def test_escape_time_evaluation():
    """Verify escape-time check flags insufficient evacuation lead time."""
    shape = (50, 50)
    transform = (85.0, 27.0, 85.5, 27.5, 0.01, 0.01)

    # High ground is located 1500m away (50 pixels at 30m)
    high_ground_mask = np.zeros(shape, dtype=bool)
    high_ground_mask[0:5, :] = True  # High ground far in the North

    # Settlement in the South at (85.25, 27.10)
    # Walking 1500m at 0.7 m/s takes ~35 minutes
    # Case A: Flood arrives in 10 minutes -> Cannot escape!
    check_fail = evaluate_settlement_escape_time(
        settlement_lon=85.25,
        settlement_lat=27.10,
        settlement_name="Trapped Village",
        arrival_time_mins=10.0,
        high_ground_mask=high_ground_mask,
        transform=transform,
        walking_speed_m_per_sec=0.70,
        res_meters=30.0,
    )
    assert check_fail.escape_possible is False
    assert check_fail.time_margin_mins < 0.0

    # Case B: Flood arrives in 90 minutes -> Escape is possible!
    check_ok = evaluate_settlement_escape_time(
        settlement_lon=85.25,
        settlement_lat=27.10,
        settlement_name="Warned Village",
        arrival_time_mins=90.0,
        high_ground_mask=high_ground_mask,
        transform=transform,
        walking_speed_m_per_sec=0.70,
        res_meters=30.0,
    )
    assert check_ok.escape_possible is True
    assert check_ok.time_margin_mins > 0.0


def test_landing_zones_detection():
    """Verify helicopter landing zone detection requires flat, unflooded, clear ground."""
    shape = (40, 40)
    transform = (85.0, 27.0, 85.4, 27.4, 0.01, 0.01)
    slope = np.full(shape, 18.0, dtype=np.float32)        # Mountainous slopes (18 deg)
    class_map = np.zeros(shape, dtype=np.uint8)
    cant_see = np.zeros(shape, dtype=bool)

    # Flat unflooded meadow near center (slope = 2.5 deg, 4x4 pixels)
    slope[20:25, 20:25] = 2.5

    settlements_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "s1",
                "geometry": {"type": "Point", "coordinates": [85.22, 27.18]},
                "properties": {"name": "Larcha"},
            }
        ],
    }

    candidates, geojson_fc = detect_landing_zones(
        slope=slope,
        class_map=class_map,
        cant_see_mask=cant_see,
        transform=transform,
        settlements_geojson=settlements_fc,
        max_slope_deg=5.0,
    )

    assert len(candidates) >= 1
    assert candidates[0].slope_deg < 5.0
    assert candidates[0].nearest_settlement == "Larcha"
    assert len(geojson_fc["features"]) >= 1


def test_rescue_priority_ranking():
    """Verify multi-factor rescue priority scoring and stranded estimates."""
    transform = (85.0, 27.0, 85.5, 27.5, 0.01, 0.01)
    dry_island_mask = np.zeros((50, 50), dtype=bool)
    # Settlement A is on a dry island
    dry_island_mask[20:30, 20:30] = True

    # Settlement records from Phase 5
    rec_a = SettlementIsolation(
        settlement_id="s_a",
        name="Tatopani (Dry Island)",
        place_type="village",
        coordinates=(85.25, 27.25),
        population_est=1500,
        cutoff_prob=0.95,
        cutoff_prob_hospital=0.95,
        cutoff_prob_town=0.90,
        nearest_hospital_name="Regional Hospital",
        pre_event_distance_km=14.0,
    )

    rec_b = SettlementIsolation(
        settlement_id="s_b",
        name="Safe Hilltop",
        place_type="village",
        coordinates=(85.40, 27.40),
        population_est=800,
        cutoff_prob=0.10,
        cutoff_prob_hospital=0.10,
        cutoff_prob_town=0.05,
        nearest_hospital_name="Regional Hospital",
        pre_event_distance_km=4.0,
    )

    bldgs_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "b1",
                "geometry": {"type": "Point", "coordinates": [85.25, 27.25]},
                "properties": {"settlement": "Tatopani (Dry Island)", "is_damaged": True},
            },
            {
                "type": "Feature",
                "id": "b2",
                "geometry": {"type": "Point", "coordinates": [85.25, 27.25]},
                "properties": {"settlement": "Tatopani (Dry Island)", "is_damaged": True},
            },
        ],
    }

    priorities = compute_rescue_priorities(
        isolation_records=[rec_b, rec_a],
        damaged_buildings_geojson=bldgs_fc,
        dry_island_mask=dry_island_mask,
        transform=transform,
        household_size=5.0,
    )

    # Rank 1 must be Tatopani (cut off, on dry island, damaged buildings)
    assert priorities[0].settlement_name == "Tatopani (Dry Island)"
    assert priorities[0].rank == 1
    assert priorities[0].is_on_dry_island is True
    assert priorities[0].priority_score > priorities[1].priority_score

    # Check stranded population estimate
    assert priorities[0].population_data_type == "estimate"
    assert priorities[0].stranded_population_estimate == int(round(priorities[0].buildings_at_risk * 5.0))
