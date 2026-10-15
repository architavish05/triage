"""Spatial grid and coordinate transformation helpers conforming to CONTRACT.md."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def coord_to_pixel(
    lon: float,
    lat: float,
    transform: Tuple[float, float, float, float, float, float],
    shape: Tuple[int, int],
) -> Optional[Tuple[int, int]]:
    """Convert WGS84 geographic coordinate (lon, lat) to raster pixel (row, col).

    Args:
        lon: Longitude in degrees.
        lat: Latitude in degrees.
        transform: 6-tuple (west, south, east, north, res_x, res_y).
        shape: 2-tuple (height, width) of the raster.

    Returns:
        (row, col) pixel indices if inside bounds, else None.
    """
    west, south, east, north, res_x, res_y = transform
    height, width = shape

    if not (west <= lon <= east and south <= lat <= north):
        return None

    col = int((lon - west) / res_x)
    row = int((north - lat) / res_y)

    col = min(max(0, col), width - 1)
    row = min(max(0, row), height - 1)

    return (row, col)


def pixel_to_coord(
    row: int,
    col: int,
    transform: Tuple[float, float, float, float, float, float],
) -> Tuple[float, float]:
    """Convert raster pixel (row, col) to WGS84 geographic coordinate (lon, lat).

    Args:
        row: Row index (0-indexed from top).
        col: Column index (0-indexed from left).
        transform: 6-tuple (west, south, east, north, res_x, res_y).

    Returns:
        (lon, lat) center coordinates of the pixel in degrees.
    """
    west, south, east, north, res_x, res_y = transform
    lon = west + (col + 0.5) * res_x
    lat = north - (row + 0.5) * res_y
    return (lon, lat)


def load_grid_spec(path: Union[str, Path]) -> Dict[str, Any]:
    """Load grid specification from a JSON file (e.g. data/mock_run/grid.json)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_grid_spec(
    path: Union[str, Path],
    bbox: Tuple[float, float, float, float],
    shape: Tuple[int, int],
    resolution_meters: float = 30.0,
    transform: Optional[Tuple[float, float, float, float, float, float]] = None,
    crs: str = "EPSG:4326",
) -> Path:
    """Save grid specification to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if transform is None:
        west, south, east, north = bbox
        h, w = shape
        res_x = (east - west) / w
        res_y = (north - south) / h
        transform = (west, south, east, north, res_x, res_y)
    data = {
        "bbox": list(bbox),
        "shape": list(shape),
        "resolution_meters": resolution_meters,
        "crs": crs,
        "transform": list(transform),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path
