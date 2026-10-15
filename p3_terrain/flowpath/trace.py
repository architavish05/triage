"""Hydrological flood path modeling and downstream settlement arrival time tracing.

Implements DEM depression filling, D8 steepest-descent flow direction, flow accumulation,
downstream trajectory tracing from initiation points (e.g. glacier collapse / outburst),
corridor buffering, and physics-based flood surge arrival time estimation.
"""

from dataclasses import asdict, dataclass
import heapq
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# D8 Direction Encoding (dx, dy): 8 neighbors
D8_OFFSETS = [
    (0, 1),    # 0: East
    (1, 1),    # 1: South-East
    (1, 0),    # 2: South
    (1, -1),   # 3: South-West
    (0, -1),   # 4: West
    (-1, -1),  # 5: North-West
    (-1, 0),   # 6: North
    (-1, 1),   # 7: North-East
]

D8_DISTANCES = [1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0), 1.0, np.sqrt(2.0)]


@dataclass
class DownstreamTrace:
    """Downstream flood trajectory and profile."""
    path_pixels: List[Tuple[int, int]]       # [(r, c), ...]
    path_coords: List[Tuple[float, float]]   # [(lon, lat), ...]
    cumulative_dist_km: List[float]          # Distance from initiation point
    elevations_m: List[float]                # Channel bed elevation profile
    cumulative_time_mins: List[float]        # Cumulative arrival time in minutes
    total_length_km: float
    total_time_hours: float


@dataclass
class SettlementArrival:
    """Settlement along the flood path with chronological arrival metrics."""
    rank_order: int
    settlement_id: str
    name: str
    place_type: str
    coordinates: Tuple[float, float]
    distance_from_source_km: float
    elevation_m: float
    arrival_time_mins: float
    arrival_time_hours: float
    warning_lead_time_urgency: str  # 'IMMEDIATE (<30m)', 'HIGH (30-90m)', 'MODERATE (>90m)'


def fill_depressions(dem: np.ndarray, epsilon: float = 0.01) -> np.ndarray:
    """Fill depressions and sinks in DEM using Priority-Flood algorithm.

    Guarantees every interior cell has a strictly decreasing or flat path
    to the boundary, allowing unbroken hydrological routing.
    """
    rows, cols = dem.shape
    filled = np.full((rows, cols), np.inf, dtype=np.float32)
    visited = np.zeros((rows, cols), dtype=bool)

    # Priority queue stores (elevation, r, c)
    pq: List[Tuple[float, int, int]] = []

    # Initialize priority queue with boundary cells
    for r in range(rows):
        for c in (0, cols - 1):
            filled[r, c] = dem[r, c]
            visited[r, c] = True
            heapq.heappush(pq, (float(dem[r, c]), r, c))
    for c in range(1, cols - 1):
        for r in (0, rows - 1):
            filled[r, c] = dem[r, c]
            visited[r, c] = True
            heapq.heappush(pq, (float(dem[r, c]), r, c))

    # Spill elevation outward-in
    while pq:
        h, r, c = heapq.heappop(pq)
        for dr, dc in D8_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr, nc]:
                visited[nr, nc] = True
                spill_h = max(float(dem[nr, nc]), h + epsilon)
                filled[nr, nc] = spill_h
                heapq.heappush(pq, (spill_h, nr, nc))

    return filled


def compute_d8_flow(
    filled_dem: np.ndarray,
    res_meters: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute D8 flow direction and accumulation grids.

    Args:
        filled_dem: 2D depression-free elevation raster.
        res_meters: Pixel resolution in meters.

    Returns:
        Tuple of (d8_dir, flow_accum):
          d8_dir: 2D array of integers [0..7] indicating steepest descent neighbor index. -1 for sink/boundary.
          flow_accum: 2D array of upstream contributing cell counts.
    """
    rows, cols = filled_dem.shape
    d8_dir = np.full((rows, cols), -1, dtype=np.int8)

    # 1. Determine steepest descent neighbor for each cell
    for r in range(rows):
        for c in range(cols):
            center_z = filled_dem[r, c]
            max_slope = -1.0
            best_dir = -1

            for idx, ((dr, dc), dist_factor) in enumerate(zip(D8_OFFSETS, D8_DISTANCES)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols:
                    dz = center_z - filled_dem[nr, nc]
                    if dz > 0:
                        slope = dz / (dist_factor * res_meters)
                        if slope > max_slope:
                            max_slope = slope
                            best_dir = idx

            d8_dir[r, c] = best_dir

    # 2. Compute flow accumulation via topological sorting (descending elevation)
    flat_indices = np.argsort(filled_dem.ravel())[::-1]
    flow_accum = np.ones((rows, cols), dtype=np.float32)

    for idx in flat_indices:
        r = idx // cols
        c = idx % cols
        direction = d8_dir[r, c]
        if direction >= 0:
            dr, dc = D8_OFFSETS[direction]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                flow_accum[nr, nc] += flow_accum[r, c]

    return d8_dir, flow_accum


def trace_downstream_path(
    d8_dir: np.ndarray,
    filled_dem: np.ndarray,
    start_coords: Tuple[float, float],
    transform: Tuple[float, float, float, float, float, float],
    res_meters: float = 30.0,
    max_steps: int = 5000,
) -> DownstreamTrace:
    """Trace flood path downstream from an initiation point and calculate arrival times.

    Physics-based GLOF/debris flow surge velocity:
      v(x) = min(16.0, max(3.5, 6.5 * sqrt(S / 0.02))) in m/s (approx 15 - 55 km/h).
      Arrival time is integrated along the descent path: dt = dx / v.

    Args:
        d8_dir: D8 flow directions [0..7].
        filled_dem: Elevation raster in meters.
        start_coords: (lon, lat) of initiation point.
        transform: (west, south, east, north, res_x, res_y).
        res_meters: Cell resolution in meters.
        max_steps: Safety threshold for path traversal.

    Returns:
        DownstreamTrace object with coordinates, distances, and cumulative arrival times.
    """
    west, south, east, north, res_x, res_y = transform
    rows, cols = filled_dem.shape

    # Find starting cell
    start_col = int(np.clip((start_coords[0] - west) / max(res_x, 1e-9), 0, cols - 1))
    start_row = int(np.clip((north - start_coords[1]) / max(res_y, 1e-9), 0, rows - 1))

    curr_r, curr_c = start_row, start_col
    visited = set()

    path_pixels = [(curr_r, curr_c)]
    start_lon = west + (curr_c + 0.5) * res_x
    start_lat = north - (curr_r + 0.5) * res_y
    path_coords = [(start_lon, start_lat)]
    cum_dist_km = [0.0]
    elevations = [float(filled_dem[curr_r, curr_c])]
    cum_time_mins = [0.0]

    for _ in range(max_steps):
        visited.add((curr_r, curr_c))
        direction = d8_dir[curr_r, curr_c]
        if direction < 0:
            break  # Reached sink or boundary

        dr, dc = D8_OFFSETS[direction]
        nr, nc = curr_r + dr, curr_c + dc

        if not (0 <= nr < rows and 0 <= nc < cols) or (nr, nc) in visited:
            break

        step_dist_m = D8_DISTANCES[direction] * res_meters
        step_dist_km = step_dist_m / 1000.0
        prev_z = filled_dem[curr_r, curr_c]
        next_z = filled_dem[nr, nc]
        dz = max(0.1, prev_z - next_z)

        # Local channel bed gradient S
        slope_frac = dz / max(step_dist_m, 1.0)
        # Debris surge velocity in m/s
        v_surge = float(np.clip(6.5 * np.sqrt(max(0.002, slope_frac) / 0.02), 3.5, 16.0))
        # Step travel time in minutes
        dt_mins = (step_dist_m / v_surge) / 60.0

        curr_r, curr_c = nr, nc
        p_lon = west + (curr_c + 0.5) * res_x
        p_lat = north - (curr_r + 0.5) * res_y

        path_pixels.append((curr_r, curr_c))
        path_coords.append((p_lon, p_lat))
        cum_dist_km.append(round(cum_dist_km[-1] + step_dist_km, 3))
        elevations.append(round(float(next_z), 1))
        cum_time_mins.append(round(cum_time_mins[-1] + dt_mins, 2))

    total_dist = cum_dist_km[-1] if cum_dist_km else 0.0
    total_hours = round(cum_time_mins[-1] / 60.0, 2) if cum_time_mins else 0.0

    logger.info(
        f"Traced downstream flood path: {len(path_coords)} cells, "
        f"{total_dist:.2f} km total distance, {total_hours:.2f} hours estimated travel time."
    )

    return DownstreamTrace(
        path_pixels=path_pixels,
        path_coords=path_coords,
        cumulative_dist_km=cum_dist_km,
        elevations_m=elevations,
        cumulative_time_mins=cum_time_mins,
        total_length_km=total_dist,
        total_time_hours=total_hours,
    )


def buffer_flow_path(
    path_coords: List[Tuple[float, float]],
    transform: Tuple[float, float, float, float, float, float],
    shape: Tuple[int, int],
    buffer_meters: float = 300.0,
) -> np.ndarray:
    """Create a 2D boolean raster mask buffering the flood trajectory."""
    from scipy.ndimage import distance_transform_edt
    west, south, east, north, res_x, res_y = transform
    rows, cols = shape
    res_m = 30.0  # approximate

    path_mask = np.zeros((rows, cols), dtype=bool)
    for lon, lat in path_coords:
        col = int(np.clip((lon - west) / max(res_x, 1e-9), 0, cols - 1))
        row = int(np.clip((north - lat) / max(res_y, 1e-9), 0, rows - 1))
        path_mask[row, col] = True

    # Distance transform from path
    dist_cells = distance_transform_edt(~path_mask)
    buffer_cells = buffer_meters / res_m
    return dist_cells <= buffer_cells


def calculate_settlement_arrival_times(
    trace: DownstreamTrace,
    settlements_geojson: Dict[str, Any],
    buffer_meters: float = 800.0,
) -> List[SettlementArrival]:
    """Identify settlements within the flood corridor and compute chronological arrival times.

    Args:
        trace: DownstreamTrace result from initiation point.
        settlements_geojson: OSM places GeoJSON FeatureCollection.
        buffer_meters: Proximity threshold to flood path in meters.

    Returns:
        List of SettlementArrival items ordered chronologically from earliest to latest impact.
    """
    settlement_features = settlements_geojson.get("features", [])
    arrivals: List[SettlementArrival] = []

    if not trace.path_coords:
        return arrivals

    path_array = np.array(trace.path_coords)  # [[lon, lat], ...]

    for feat in settlement_features:
        s_id = str(feat.get("id", ""))
        props = feat.get("properties", {})
        coords = feat.get("geometry", {}).get("coordinates", [])
        if not coords:
            continue

        s_lon, s_lat = coords[0], coords[1]
        name = props.get("name", f"Settlement {s_id}")
        place_type = props.get("place", "village")

        # Compute distance to all points on the path (in approx meters)
        dx_m = (path_array[:, 0] - s_lon) * 98000.0
        dy_m = (path_array[:, 1] - s_lat) * 111000.0
        distances_m = np.sqrt(dx_m ** 2 + dy_m ** 2)

        closest_idx = int(np.argmin(distances_m))
        min_dist_m = distances_m[closest_idx]

        # Check if settlement lies within flood corridor
        if min_dist_m <= buffer_meters:
            arr_mins = float(trace.cumulative_time_mins[closest_idx])
            arr_hours = round(arr_mins / 60.0, 2)
            dist_km = float(trace.cumulative_dist_km[closest_idx])
            elev = float(trace.elevations_m[closest_idx])

            if arr_mins < 30.0:
                urgency = "IMMEDIATE (<30m)"
            elif arr_mins <= 90.0:
                urgency = "HIGH (30-90m)"
            else:
                urgency = "MODERATE (>90m)"

            arrivals.append(SettlementArrival(
                rank_order=0,
                settlement_id=s_id,
                name=name,
                place_type=place_type,
                coordinates=(s_lon, s_lat),
                distance_from_source_km=dist_km,
                elevation_m=elev,
                arrival_time_mins=arr_mins,
                arrival_time_hours=arr_hours,
                warning_lead_time_urgency=urgency,
            ))

    # Sort strictly by arrival time (earliest to latest)
    arrivals.sort(key=lambda x: x.arrival_time_mins)
    for rank, item in enumerate(arrivals, start=1):
        item.rank_order = rank

    logger.info(f"Identified {len(arrivals)} settlements along downstream flood corridor.")
    return arrivals


def export_flowpath_geojson(
    trace: DownstreamTrace,
    output_file: Union[Path, str],
    settlement_arrivals: Optional[List[SettlementArrival]] = None,
) -> Path:
    """Export traced flood surge trajectory and settlement arrival times to GeoJSON.

    Args:
        trace: DownstreamTrace result.
        output_file: Target file path for GeoJSON.
        settlement_arrivals: Optional list of chronological settlement arrival items.

    Returns:
        Path to written GeoJSON file.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    features = []

    # 1. Main LineString feature of flood routing trajectory
    line_feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": [[round(lon, 10), round(lat, 10)] for lon, lat in trace.path_coords],
        },
        "properties": {
            "name": "Flood surge route",
            "length_km": round(trace.total_length_km, 2),
            "total_time_hours": round(trace.total_time_hours, 2),
            "settlement_arrivals": [asdict(a) for a in settlement_arrivals] if settlement_arrivals else [],
        },
    }
    features.append(line_feature)

    # 2. Point features for settlements along the flood corridor
    if settlement_arrivals:
        for arr in settlement_arrivals:
            pt_feat = {
                "type": "Feature",
                "id": arr.settlement_id,
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(arr.coordinates[0], 6), round(arr.coordinates[1], 6)],
                },
                "properties": {
                    "rank_order": arr.rank_order,
                    "name": arr.name,
                    "place_type": arr.place_type,
                    "distance_from_source_km": round(arr.distance_from_source_km, 2),
                    "elevation_m": round(arr.elevation_m, 1),
                    "arrival_time_mins": round(arr.arrival_time_mins, 1),
                    "arrival_time_hours": round(arr.arrival_time_hours, 2),
                    "warning_lead_time_urgency": arr.warning_lead_time_urgency,
                },
            }
            features.append(pt_feat)

    fc = {
        "type": "FeatureCollection",
        "features": features,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fc, f, indent=2)

    logger.info(f"Saved flowpath GeoJSON with {len(features)} features to {out_path}")
    return out_path


def export_downstream_trace_json(
    trace: DownstreamTrace,
    output_file: Union[Path, str],
) -> Path:
    """Export raw downstream trace coordinates to JSON."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coords = [[round(lon, 10), round(lat, 10)] for lon, lat in trace.path_coords]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(coords, f)
    logger.info(f"Saved downstream trace JSON to {out_path}")
    return out_path


def main(run_dir_str: str = "data/mock_run") -> None:
    """CLI runner to execute D8 depression filling and flowpath tracing against a run directory."""
    try:
        from p1_data.fetch.dem import generate_synthetic_dem
    except ImportError:
        from src.fetch.dem import generate_synthetic_dem

    run_dir = Path(run_dir_str)
    if not run_dir.exists():
        raise FileNotFoundError(f"Directory not found: {run_dir}")

    grid_path = run_dir / "grid.json"
    settlements_path = run_dir / "settlements.geojson"

    bbox = (85.2, 27.8, 85.45, 28.15)
    shape = (128, 128)
    transform = (85.2, 27.8, 85.45, 28.15, 0.001953125, 0.002734375)

    if grid_path.exists():
        with open(grid_path, "r", encoding="utf-8") as f:
            grid = json.load(f)
            bbox = tuple(grid.get("bbox", bbox))
            shape = tuple(grid.get("shape", shape))
            transform = tuple(grid.get("transform", transform))

    # Generate or fetch elevation raster matching the scene
    dem_data = generate_synthetic_dem(bbox, shape=shape)
    filled_dem = fill_depressions(dem_data.elevation)
    d8_dir, flow_accum = compute_d8_flow(filled_dem, res_meters=30.0)

    # Initiate breach at top-center (consistent with pipeline)
    start_lon = (bbox[0] + bbox[2]) / 2.0
    start_lat = bbox[3] - 0.05
    trace = trace_downstream_path(
        d8_dir=d8_dir,
        filled_dem=filled_dem,
        start_coords=(start_lon, start_lat),
        transform=transform,
        res_meters=30.0,
    )

    settlement_arrivals: List[SettlementArrival] = []
    if settlements_path.exists():
        with open(settlements_path, "r", encoding="utf-8") as f:
            settlements_fc = json.load(f)
        settlement_arrivals = calculate_settlement_arrival_times(
            trace=trace,
            settlements_geojson=settlements_fc,
            buffer_meters=10000.0,  # Buffer to capture impacted settlements along regional corridor
        )

    # Export flowpath.geojson and downstream_trace.json
    flowpath_path = run_dir / "flowpath.geojson"
    export_flowpath_geojson(trace, flowpath_path, settlement_arrivals=settlement_arrivals)

    geojson_dir = run_dir / "geojson"
    if geojson_dir.exists():
        trace_json_path = geojson_dir / "downstream_trace.json"
        export_downstream_trace_json(trace, trace_json_path)

    print(f"[Flowpath] Successfully processed {run_dir}:")
    print(f"  - Trajectory waypoints: {len(trace.path_coords)}")
    print(f"  - Total length: {trace.total_length_km:.2f} km")
    print(f"  - Total time: {trace.total_time_hours:.2f} hours")
    print(f"  - Impacted settlements: {len(settlement_arrivals)}")
    print(f"  - Output GeoJSON: {flowpath_path}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data/mock_run"
    main(target)

