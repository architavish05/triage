"""Slope and Height Above Nearest Drainage (HAND) hydrological filtering.

Calculates terrain slope and HAND relative elevation above drainage networks to remove
physically impossible flood detections on steep valley flanks and high mountain ridges.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.ndimage import distance_transform_edt, uniform_filter

from p2_ai.detect.fuse import CLASS_BACKGROUND, CLASS_DEBRIS, CLASS_WATER

logger = logging.getLogger(__name__)


@dataclass
class TerrainFilterResult:
    """Output of terrain-filtered hazard map."""
    filtered_class_map: np.ndarray        # Cleaned 2D class map (0=bg, 1=water, 2=debris)
    filtered_confidence: np.ndarray       # Terrain-adjusted per-pixel confidence [0, 1]
    slope_deg: np.ndarray                 # Terrain slope raster in degrees
    hand_meters: np.ndarray               # Height Above Nearest Drainage raster in meters
    drainage_mask: np.ndarray             # Identified stream/drainage channel cells
    suppressed_water_pixels: int          # Count of false water pixels removed
    suppressed_debris_pixels: int         # Count of false debris pixels removed
    stats: Dict[str, float]


def compute_slope(dem: np.ndarray, res_meters: float = 30.0) -> np.ndarray:
    """Compute terrain slope in degrees from 2D elevation model.

    slope_deg = arctan(sqrt((dz/dx)^2 + (dz/dy)^2)) * (180 / pi)
    """
    grad_y, grad_x = np.gradient(dem, res_meters)
    slope_rad = np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2))
    return np.degrees(slope_rad).astype(np.float32)


def extract_drainage_network(
    dem: np.ndarray,
    slope: np.ndarray,
    elevation_window: int = 15,
) -> np.ndarray:
    """Extract stream channel / valley drainage network from DEM and slope.

    Identifies local elevation minima with gentle slopes that collect surface runoff.
    """
    if elevation_window % 2 == 0:
        elevation_window += 1
    # Local neighborhood minimum elevation
    from scipy.ndimage import minimum_filter
    local_min = minimum_filter(dem, size=elevation_window)
    # Valley bottom cells: elevation is within 5m of local minimum AND slope < 8 deg
    valley_bottom = (dem - local_min <= 5.0) & (slope < 8.0)

    # In case whole terrain is flat, ensure at least lowest elevation cells are marked
    if np.count_nonzero(valley_bottom) == 0:
        min_elev = np.min(dem)
        valley_bottom = dem <= min_elev + 5.0

    return valley_bottom


def compute_hand(
    dem: np.ndarray,
    drainage_mask: Optional[np.ndarray] = None,
    slope: Optional[np.ndarray] = None,
    res_meters: float = 30.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Height Above Nearest Drainage (HAND) in meters.

    For every grid cell, HAND represents elevation difference relative to the
    nearest hydrological drainage channel cell.

    Args:
        dem: 2D elevation grid in meters.
        drainage_mask: Optional boolean mask of river/stream channels.
        slope: Optional slope in degrees.
        res_meters: Ground cell resolution in meters.

    Returns:
        Tuple of (hand_grid, drainage_mask).
    """
    if slope is None:
        slope = compute_slope(dem, res_meters=res_meters)

    if drainage_mask is None or np.count_nonzero(drainage_mask) == 0:
        drainage_mask = extract_drainage_network(dem, slope)

    # Compute Euclidean nearest drainage cell coordinates
    _, (nearest_r, nearest_c) = distance_transform_edt(~drainage_mask, return_indices=True)

    # Drainage elevation at the nearest channel cell
    drainage_elev = dem[nearest_r, nearest_c]

    # HAND = max(0, cell_elevation - drainage_elevation)
    hand = np.maximum(0.0, dem - drainage_elev).astype(np.float32)

    return hand, drainage_mask


def filter_impossible_flood(
    class_map: np.ndarray,
    confidence: np.ndarray,
    slope: np.ndarray,
    hand: np.ndarray,
    shadow_mask: np.ndarray,
    max_flood_slope_deg: float = 10.0,
    max_flood_hand_m: float = 22.0,
) -> TerrainFilterResult:
    """Filter out or down-weight physically impossible flood and debris detections.

    Physical Hydrology Rules:
    1. Standing or slow-moving flood water cannot exist on steep terrain (> max_flood_slope_deg).
    2. Flood water cannot rise beyond valley drainage envelopes (> max_flood_hand_m).
    3. Radar shadow creates dark specular-like backscatter that mimics water;
       any detection inside radar shadow is occluded and rejected as false alarm.

    Args:
        class_map: Initial 2D classification map (0=bg, 1=water, 2=debris).
        confidence: Initial per-pixel confidence array [0, 1].
        slope: Terrain slope raster in degrees.
        hand: Height Above Nearest Drainage in meters.
        shadow_mask: 2D boolean mask of radar shadow areas.
        max_flood_slope_deg: Maximum slope permissible for flood inundation (default 10 deg).
        max_flood_hand_m: Maximum HAND permissible for flood inundation (default 22m).

    Returns:
        TerrainFilterResult with cleaned class map, attenuated confidence, and filter stats.
    """
    filtered_class = class_map.copy()
    filtered_conf = confidence.copy()

    # Rule 1 & 2: Impossible flood water
    is_water = (class_map == CLASS_WATER)
    impossible_water_slope = is_water & (slope > max_flood_slope_deg)
    impossible_water_hand = is_water & (hand > max_flood_hand_m)
    impossible_water_shadow = is_water & shadow_mask

    impossible_water = impossible_water_slope | impossible_water_hand | impossible_water_shadow
    suppressed_water = int(np.count_nonzero(impossible_water))

    # Remove impossible water
    filtered_class[impossible_water] = CLASS_BACKGROUND
    filtered_conf[impossible_water] = 0.0

    # Rule 3: Debris in radar shadow (cannot observe inside shadow)
    is_debris = (class_map == CLASS_DEBRIS)
    impossible_debris_shadow = is_debris & shadow_mask
    suppressed_debris = int(np.count_nonzero(impossible_debris_shadow))

    filtered_class[impossible_debris_shadow] = CLASS_BACKGROUND
    filtered_conf[impossible_debris_shadow] = 0.0

    # Smooth confidence attenuation in transitional slope/HAND zones for valid water:
    valid_water = (filtered_class == CLASS_WATER)
    # Slope penalty between 6 deg and max_flood_slope_deg
    slope_penalty = np.clip((max_flood_slope_deg - slope) / 4.0, 0.2, 1.0)
    # HAND penalty between 12m and max_flood_hand_m
    hand_penalty = np.clip((max_flood_hand_m - hand) / 10.0, 0.2, 1.0)
    attenuation = np.minimum(slope_penalty, hand_penalty)

    filtered_conf[valid_water] = (filtered_conf[valid_water] * attenuation[valid_water]).astype(np.float32)

    stats = {
        "suppressed_water_pixels": float(suppressed_water),
        "suppressed_debris_pixels": float(suppressed_debris),
        "mean_water_slope": float(np.mean(slope[filtered_class == CLASS_WATER])) if np.any(filtered_class == CLASS_WATER) else 0.0,
        "mean_water_hand": float(np.mean(hand[filtered_class == CLASS_WATER])) if np.any(filtered_class == CLASS_WATER) else 0.0,
    }

    logger.info(
        f"Terrain filtering complete: suppressed {suppressed_water} false water pixels "
        f"and {suppressed_debris} false debris pixels (radar shadow/slope/HAND)."
    )

    # Return drainage mask placeholder or empty if needed
    drainage_mask = (hand <= 2.0) & (slope <= 5.0)

    return TerrainFilterResult(
        filtered_class_map=filtered_class,
        filtered_confidence=filtered_conf,
        slope_deg=slope,
        hand_meters=hand,
        drainage_mask=drainage_mask,
        suppressed_water_pixels=suppressed_water,
        suppressed_debris_pixels=suppressed_debris,
        stats=stats,
    )


def save_raster(
    path: Union[Path, str],
    array: np.ndarray,
    transform: Optional[Tuple[float, float, float, float, float, float]] = None,
    crs: str = "EPSG:4326",
) -> Path:
    """Save 2D array as TIFF/GeoTIFF raster.

    Uses rasterio if available for geospatial metadata,
    otherwise falls back to PIL.Image.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rasterio
        from rasterio.transform import from_bounds

        rows, cols = array.shape
        dtype = array.dtype
        if dtype == bool:
            arr_to_write = (array.astype(np.uint8) * 255)
            dtype_str = "uint8"
        elif np.issubdtype(dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
            arr_to_write = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
            dtype_str = "uint8"
        else:
            arr_to_write = array
            dtype_str = str(dtype)

        rio_transform = None
        if transform is not None:
            west, south, east, north = transform[0], transform[1], transform[2], transform[3]
            rio_transform = from_bounds(west, south, east, north, cols, rows)

        with rasterio.open(
            out_path,
            "w",
            driver="GTiff",
            height=rows,
            width=cols,
            count=1,
            dtype=dtype_str,
            crs=crs if rio_transform is not None else None,
            transform=rio_transform,
        ) as dst:
            dst.write(arr_to_write, 1)

        logger.info(f"Saved raster to {out_path} using rasterio")
        return out_path
    except ImportError:
        pass

    from PIL import Image

    if array.dtype == bool:
        img_arr = (array.astype(np.uint8) * 255)
    elif np.issubdtype(array.dtype, np.floating) and float(np.nanmax(array)) <= 1.0:
        img_arr = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    else:
        img_arr = array.astype(np.uint8)

    im = Image.fromarray(img_arr)
    im.save(out_path, format="TIFF")
    logger.info(f"Saved raster to {out_path} using PIL")
    return out_path


def filter_and_export_terrain(
    class_map: np.ndarray,
    confidence: np.ndarray,
    slope: np.ndarray,
    hand: np.ndarray,
    shadow_mask: np.ndarray,
    output_dir: Union[Path, str],
    transform: Optional[Tuple[float, float, float, float, float, float]] = None,
    max_flood_slope_deg: float = 10.0,
    max_flood_hand_m: float = 22.0,
) -> Tuple[TerrainFilterResult, Path, Path]:
    """Execute terrain filtering and export cant_see.tif and flood_conf.tif.

    Args:
        class_map: Initial classified raster (0=bg, 1=water, 2=debris).
        confidence: Per-pixel confidence map [0.0, 1.0] or [0..255].
        slope: Terrain slope in degrees.
        hand: Height Above Nearest Drainage in meters.
        shadow_mask: Radar shadow boolean mask.
        output_dir: Target folder to save cant_see.tif and flood_conf.tif.
        transform: Optional georeferencing transform.
        max_flood_slope_deg: Maximum slope for flood inundation (default 10 deg).
        max_flood_hand_m: Maximum HAND for flood inundation (default 22m).

    Returns:
        Tuple of (TerrainFilterResult, cant_see_path, flood_conf_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conf_float = confidence.astype(np.float32)
    if float(np.nanmax(conf_float)) > 1.0:
        conf_float = conf_float / 255.0

    filter_res = filter_impossible_flood(
        class_map=class_map,
        confidence=conf_float,
        slope=slope,
        hand=hand,
        shadow_mask=shadow_mask,
        max_flood_slope_deg=max_flood_slope_deg,
        max_flood_hand_m=max_flood_hand_m,
    )

    cant_see_path = out_dir / "cant_see.tif"
    flood_conf_path = out_dir / "flood_conf.tif"

    save_raster(cant_see_path, shadow_mask, transform=transform)
    save_raster(flood_conf_path, filter_res.filtered_confidence, transform=transform)

    return filter_res, cant_see_path, flood_conf_path


def main(run_dir_str: str = "data/mock_run") -> None:
    """CLI entrypoint to run terrain filtering against a run directory."""
    import json
    from PIL import Image

    run_dir = Path(run_dir_str)
    if not run_dir.exists():
        raise FileNotFoundError(f"Directory not found: {run_dir}")

    # Load inputs
    slope_path = run_dir / "slope.tif"
    hand_path = run_dir / "hand.tif"
    cls_path = run_dir / "class.tif"
    raw_conf_path = run_dir / "flood_raw_conf.tif"
    if not raw_conf_path.exists():
        raw_conf_path = run_dir / "flood_conf.tif"
    cant_see_path = run_dir / "cant_see.tif"
    grid_path = run_dir / "grid.json"

    slope = np.array(Image.open(slope_path)).astype(np.float32)
    hand = np.array(Image.open(hand_path)).astype(np.float32)
    cls = np.array(Image.open(cls_path)).astype(np.uint8)
    raw_conf = np.array(Image.open(raw_conf_path)).astype(np.float32) / 255.0

    if cant_see_path.exists():
        cant_see = np.array(Image.open(cant_see_path)) > 0
    else:
        cant_see = np.zeros(cls.shape, dtype=bool)

    transform = None
    if grid_path.exists():
        with open(grid_path, "r", encoding="utf-8") as f:
            grid = json.load(f)
            transform = tuple(grid.get("transform", ()))

    res, c_path, f_path = filter_and_export_terrain(
        class_map=cls,
        confidence=raw_conf,
        slope=slope,
        hand=hand,
        shadow_mask=cant_see,
        output_dir=run_dir,
        transform=transform,
        max_flood_slope_deg=10.0,
        max_flood_hand_m=22.0,
    )

    print(f"[Terrain] Successfully processed {run_dir}:")
    print(f"  - Suppressed water pixels: {res.suppressed_water_pixels}")
    print(f"  - Suppressed debris pixels: {res.suppressed_debris_pixels}")
    print(f"  - Output cant_see: {c_path}")
    print(f"  - Output flood_conf: {f_path}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data/mock_run"
    main(target)

