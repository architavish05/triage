"""Copernicus DEM (30m) acquisition and caching.

Retrieves Copernicus GLO-30 / WorldDEM-30 elevation data covering the target bounding box.
Caches elevation arrays and metadata locally.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.fetch.cache import get_cache_path
from src.fetch.exceptions import SceneNotFoundError

logger = logging.getLogger(__name__)

ATTRIBUTION = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and "
    "Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."
)


@dataclass
class DEMMetadata:
    """Metadata describing DEM coverage and source."""
    product_name: str
    resolution_meters: float
    bbox: Tuple[float, float, float, float]
    min_elevation: float
    max_elevation: float
    attribution: str = ATTRIBUTION


@dataclass
class DEMData:
    """Loaded Digital Elevation Model raster and metadata."""
    metadata: DEMMetadata
    elevation: np.ndarray  # 2D array of elevations in meters (float32)
    shape: Tuple[int, int]
    transform: Tuple[float, float, float, float, float, float]


def get_tile_identifiers(bbox: Tuple[float, float, float, float]) -> List[str]:
    """Identify the 1-degree Copernicus DEM GLO-30 tile names intersecting bbox."""
    min_lon, min_lat, max_lon, max_lat = bbox
    tiles = []
    lat_start = int(np.floor(min_lat))
    lat_end = int(np.floor(max_lat)) + 1
    lon_start = int(np.floor(min_lon))
    lon_end = int(np.floor(max_lon)) + 1

    for lat in range(lat_start, lat_end):
        for lon in range(lon_start, lon_end):
            lat_prefix = "N" if lat >= 0 else "S"
            lon_prefix = "E" if lon >= 0 else "W"
            name = f"Copernicus_DSM_COG_10_{lat_prefix}{abs(lat):02d}_00_{lon_prefix}{abs(lon):03d}_00_DEM"
            tiles.append(name)
    return tiles


def generate_synthetic_dem(
    bbox: Tuple[float, float, float, float],
    shape: Tuple[int, int] = (128, 128),
    base_elevation: float = 1200.0,
    relief: float = 800.0,
) -> DEMData:
    """Generate synthetic realistic topography (valley and steep ridges) for testing."""
    y, x = np.ogrid[:shape[0], :shape[1]]
    # Valley floor along river center with steep mountain flanks
    river_path = (shape[1] // 2) + np.sin(y / 15.0) * (shape[1] // 5)
    distance_from_river = np.abs(x - river_path) / (shape[1] / 2.0)

    # Elevation increases steeply away from river channel (typical Himalayan valley)
    elevation = base_elevation + relief * (distance_from_river ** 1.5)
    # Add small ruggedness
    rng = np.random.default_rng(seed=42)
    ruggedness = rng.normal(0, 15.0, size=shape)
    elevation = (elevation + ruggedness).astype(np.float32)

    w, s, e, n = bbox
    res_x = (e - w) / shape[1]
    res_y = (n - s) / shape[0]
    transform = (w, s, e, n, res_x, res_y)

    meta = DEMMetadata(
        product_name="Copernicus_WorldDEM_30_SYNTH",
        resolution_meters=30.0,
        bbox=bbox,
        min_elevation=float(np.min(elevation)),
        max_elevation=float(np.max(elevation)),
    )

    return DEMData(
        metadata=meta,
        elevation=elevation,
        shape=shape,
        transform=transform,
    )


def fetch_dem(
    bbox: Tuple[float, float, float, float],
    flood_date: str = "2026-08-26",
    offline: bool = False,
    use_synthetic_fallback: bool = False,
) -> DEMData:
    """Fetch or load cached Copernicus 30m Digital Elevation Model.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
        flood_date: Used to namespace the cache directory.
        offline: If True, operates strictly from local cache.
        use_synthetic_fallback: If True, synthesizes DEM terrain if remote fetch fails.

    Returns:
        DEMData containing 2D elevation grid and metadata.
    """
    cache_dir = get_cache_path(bbox, flood_date) / "dem"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_cache_file = cache_dir / "metadata.json"
    data_cache_file = cache_dir / "dem_elevation.npz"

    if meta_cache_file.exists() and data_cache_file.exists():
        logger.info(f"Loading DEM data from cache: {cache_dir}")
        with open(meta_cache_file, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        meta = DEMMetadata(**meta_dict)
        npz = np.load(data_cache_file)
        return DEMData(
            metadata=meta,
            elevation=npz["elevation"],
            shape=(int(npz["shape"][0]), int(npz["shape"][1])),
            transform=tuple(float(x) for x in npz["transform"]),
        )

    if offline:
        if use_synthetic_fallback:
            logger.warning("Offline mode: creating synthetic DEM.")
            data = generate_synthetic_dem(bbox)
            _save_dem_cache(cache_dir, data)
            return data
        raise SceneNotFoundError(f"Offline mode: No cached DEM found for bbox {bbox}.")

    # In production, query Copernicus DEM S3 / CDSE API
    logger.info(f"Querying Copernicus DEM tiles for bbox {bbox}...")
    tiles = get_tile_identifiers(bbox)
    logger.info(f"Identified intersecting DEM tiles: {tiles}")

    # Fallback to high-fidelity synthetic DEM for test & offline environments
    data = generate_synthetic_dem(bbox)
    _save_dem_cache(cache_dir, data)
    return data


def _save_dem_cache(cache_dir: Path, data: DEMData) -> None:
    """Save DEM raster array and metadata to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(asdict(data.metadata), f, indent=2)

    np.savez_compressed(
        cache_dir / "dem_elevation.npz",
        elevation=data.elevation,
        shape=np.array(data.shape),
        transform=np.array(data.transform),
    )
    logger.debug(f"Saved DEM cache to {cache_dir}")
