"""Unit tests for Phase 6 flood path tracing and settlement arrival estimation."""

import numpy as np
import pytest

from p3_terrain.flowpath.trace import (
    buffer_flow_path,
    calculate_settlement_arrival_times,
    compute_d8_flow,
    fill_depressions,
    trace_downstream_path,
)


def test_fill_depressions():
    """Verify Priority-Flood algorithm eliminates interior sinks and depressions."""
    # 5x5 DEM with a deep sink in the center (row 2, col 2)
    dem = np.array([
        [100.0, 100.0, 100.0, 100.0, 100.0],
        [100.0,  50.0,  50.0,  50.0, 100.0],
        [100.0,  50.0,  10.0,  50.0, 100.0],  # Center sink at 10m surrounded by 50m
        [100.0,  50.0,  50.0,  50.0, 100.0],
        [100.0, 100.0,  30.0, 100.0, 100.0],  # Outlet at bottom center (30m)
    ], dtype=np.float32)

    filled = fill_depressions(dem)
    # The center sink should be raised to spill level (at least 30m outlet + epsilon)
    assert filled[2, 2] >= 30.0
    # Boundary outlet should be preserved
    assert filled[4, 2] == 30.0


def test_compute_d8_flow():
    """Verify D8 flow direction points strictly downhill."""
    # North-to-South sloping plane (elevation decreases as row increases)
    rows, cols = 10, 10
    row_idx = np.arange(rows, dtype=np.float32)[:, None]
    dem = np.tile(1000.0 - row_idx * 50.0, (1, cols))  # Slopes South across 10 cols

    d8_dir, flow_accum = compute_d8_flow(dem, res_meters=30.0)

    # South offset is index 2 in D8_OFFSETS
    assert np.all(d8_dir[:8, :] == 2)
    # Flow accumulation should increase down the slope
    assert flow_accum[8, 5] > flow_accum[1, 5]


def test_trace_downstream_path_and_arrival():
    """Verify downstream trajectory tracing accumulates distance and arrival time monotonically."""
    rows, cols = 40, 40
    # Valley sloping from North-West (r=0, c=0) to South-East (r=39, c=39)
    r_grid, c_grid = np.ogrid[:rows, :cols]
    dem = 2000.0 - (r_grid * 20.0 + c_grid * 20.0).astype(np.float32)

    transform = (85.0, 27.0, 85.4, 27.4, 0.01, 0.01)
    d8_dir, _ = compute_d8_flow(dem, res_meters=30.0)

    # Start at high peak (r=2, c=2)
    start_lon = 85.0 + 2.5 * 0.01
    start_lat = 27.4 - 2.5 * 0.01

    trace = trace_downstream_path(d8_dir, dem, (start_lon, start_lat), transform, res_meters=30.0)

    assert len(trace.path_coords) > 10
    assert trace.total_length_km > 0.5
    assert trace.total_time_hours > 0.0

    # Distances and times must be monotonically non-decreasing
    dists = trace.cumulative_dist_km
    times = trace.cumulative_time_mins
    for i in range(len(dists) - 1):
        assert dists[i + 1] >= dists[i]
        assert times[i + 1] >= times[i]

    # Test buffering
    buffer_mask = buffer_flow_path(trace.path_coords, transform, (rows, cols), buffer_meters=200.0)
    assert np.any(buffer_mask)
    # Initial point should be inside buffer
    assert buffer_mask[2, 2] is np.True_


def test_calculate_settlement_arrival_times():
    """Verify settlements along the flood path are listed in chronological arrival order."""
    # Create synthetic trace
    path_coords = [(85.10, 27.30), (85.15, 27.25), (85.20, 27.20), (85.25, 27.15)]
    cum_dist = [0.0, 7.5, 15.0, 22.5]
    elevations = [2500.0, 1800.0, 1200.0, 800.0]
    cum_time = [0.0, 18.0, 38.0, 62.0]  # Arrival in minutes

    from p3_terrain.flowpath.trace import DownstreamTrace
    trace = DownstreamTrace(
        path_pixels=[(0, 0), (1, 1), (2, 2), (3, 3)],
        path_coords=path_coords,
        cumulative_dist_km=cum_dist,
        elevations_m=elevations,
        cumulative_time_mins=cum_time,
        total_length_km=22.5,
        total_time_hours=1.03,
    )

    settlements_fc = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "s_far",
                "geometry": {"type": "Point", "coordinates": [85.25, 27.15]},  # Downstream (at 62 mins)
                "properties": {"name": "Lower Town", "place": "town"},
            },
            {
                "type": "Feature",
                "id": "s_near",
                "geometry": {"type": "Point", "coordinates": [85.15, 27.25]},  # Upstream (at 18 mins)
                "properties": {"name": "Upper Village", "place": "village"},
            },
        ],
    }

    arrivals = calculate_settlement_arrival_times(trace, settlements_fc, buffer_meters=500.0)

    assert len(arrivals) == 2
    # Upper Village must be rank 1 (arrives earlier at 18 mins)
    assert arrivals[0].name == "Upper Village"
    assert arrivals[0].rank_order == 1
    assert arrivals[0].arrival_time_mins == 18.0
    assert arrivals[0].warning_lead_time_urgency == "IMMEDIATE (<30m)"

    # Lower Town must be rank 2 (arrives later at 62 mins)
    assert arrivals[1].name == "Lower Town"
    assert arrivals[1].rank_order == 2
    assert arrivals[1].arrival_time_mins == 62.0
    assert arrivals[1].warning_lead_time_urgency == "HIGH (30-90m)"
