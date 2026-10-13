"""Unit tests for Phase 5 road graph and Monte Carlo cut-off analysis."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.network.cutoff import (
    RoadGraph,
    build_road_graph,
    run_monte_carlo_cutoff,
)
from src.network.priority import (
    build_cutoff_table,
    export_cutoff_geojson,
    export_cutoff_table_csv,
)


def test_road_graph_basics():
    """Verify RoadGraph node creation, nearest node query, and Dijkstra shortest path."""
    g = RoadGraph()
    g.add_edge("A", "B", length_km=5.0, hit_prob=0.0)
    g.add_edge("B", "C", length_km=3.0, hit_prob=0.0)

    assert len(g.nodes) == 3
    assert len(g.edge_list) == 2

    # Shortest path from A to {C} should be 8.0 km
    dist, target = g.shortest_path_distance("A", {"C"})
    assert dist == 8.0
    assert target == "C"


def test_monte_carlo_cutoff_probabilities():
    """Verify Monte Carlo simulation correctly models 0%, 50%, and 100% failure rates."""
    # Create simple linear network:
    # Hospital (node H) -- [Road 1] -- Town (node T) -- [Road 2: 50% hit] -- Village (node V) -- [Road 3: 100% hit] -- Isolated (node I)
    damaged_roads_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "r1",
                "geometry": {"type": "LineString", "coordinates": [[85.1, 27.0], [85.2, 27.0]]},
                "properties": {"hit_prob": 0.0},
            },
            {
                "type": "Feature",
                "id": "r2",
                "geometry": {"type": "LineString", "coordinates": [[85.2, 27.0], [85.3, 27.0]]},
                "properties": {"hit_prob": 0.5},
            },
            {
                "type": "Feature",
                "id": "r3",
                "geometry": {"type": "LineString", "coordinates": [[85.3, 27.0], [85.4, 27.0]]},
                "properties": {"hit_prob": 1.0},
            },
        ],
    }

    hospitals_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "hosp_1",
                "geometry": {"type": "Point", "coordinates": [85.1, 27.0]},
                "properties": {"name": "Central Hospital"},
            }
        ],
    }

    settlements_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "s_town",
                "geometry": {"type": "Point", "coordinates": [85.2, 27.0]},
                "properties": {"name": "Safe Town", "place": "town", "population": "10000"},
            },
            {
                "type": "Feature",
                "id": "s_village",
                "geometry": {"type": "Point", "coordinates": [85.3, 27.0]},
                "properties": {"name": "Mid Village", "place": "village", "population": "1200"},
            },
            {
                "type": "Feature",
                "id": "s_isolated",
                "geometry": {"type": "Point", "coordinates": [85.4, 27.0]},
                "properties": {"name": "Cutoff Hamlet", "place": "hamlet", "population": "300"},
            },
        ],
    }

    graph = build_road_graph(damaged_roads_fc)
    results = run_monte_carlo_cutoff(
        road_graph=graph,
        settlements_geojson=settlements_fc,
        hospitals_geojson=hospitals_fc,
        num_simulations=200,
        random_seed=42,
    )

    by_name = {r.name: r for r in results}

    # Safe Town has 0% failure road to hospital
    assert by_name["Safe Town"].cutoff_prob_hospital == 0.0

    # Cutoff Hamlet road to village has hit_prob = 1.0, so 100% of simulations should disconnect it
    assert by_name["Cutoff Hamlet"].cutoff_prob_hospital == 1.0

    # Mid Village road has hit_prob = 0.5, so cut-off probability should be around 0.5 (+- 0.08)
    assert 0.40 <= by_name["Mid Village"].cutoff_prob_hospital <= 0.60


def test_priority_table_and_exports(tmp_path):
    """Verify table formatting, CSV export, and GeoJSON export."""
    damaged_roads_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "r1",
                "geometry": {"type": "LineString", "coordinates": [[85.1, 27.0], [85.2, 27.0]]},
                "properties": {"hit_prob": 0.8},
            }
        ],
    }
    hospitals_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "h1",
                "geometry": {"type": "Point", "coordinates": [85.1, 27.0]},
                "properties": {"name": "Valley Hospital"},
            }
        ],
    }
    settlements_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "s1",
                "geometry": {"type": "Point", "coordinates": [85.2, 27.0]},
                "properties": {"name": "Isolated Village", "place": "village"},
            }
        ],
    }

    graph = build_road_graph(damaged_roads_fc)
    records = run_monte_carlo_cutoff(graph, settlements_fc, hospitals_fc, num_simulations=50)

    table = build_cutoff_table(records)
    assert len(table) == 1
    assert table[0]["rank"] == 1
    assert table[0]["settlement_name"] == "Isolated Village"

    # Test CSV export
    csv_file = tmp_path / "cutoff_table.csv"
    export_cutoff_table_csv(table, csv_file)
    assert csv_file.exists()

    # Test GeoJSON export
    geojson_file = tmp_path / "cutoff_settlements.geojson"
    export_cutoff_geojson(records, geojson_file)
    assert geojson_file.exists()
    with open(geojson_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["type"] == "FeatureCollection"
        assert len(data["features"]) == 1
