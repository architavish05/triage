"""Data ingestion and caching modules for Sentinel-1, Sentinel-2, DEM, and OSM."""

from typing import Any, Dict, Optional, Tuple

from p1_data.fetch.cache import get_cache_key, get_cache_path, load_json_cache, save_json_cache
from p1_data.fetch.dem import DEMData, DEMMetadata, fetch_dem, generate_synthetic_dem
from p1_data.fetch.exceptions import (
    CloudCoverExceededError,
    CredentialsError,
    DateConstraintError,
    OhsomeAPIError,
    OrbitMismatchError,
    SceneNotFoundError,
    TriageFetchError,
)
from p1_data.fetch.osm import MANDATORY_MAX_SNAPSHOT_DATE, OSMData, fetch_osm, generate_synthetic_osm
from p1_data.fetch.s1 import S1DataPair, S1SceneMetadata, fetch_s1, generate_synthetic_s1
from p1_data.fetch.s2 import S2Data, S2SceneMetadata, compute_ndwi, fetch_s2, generate_synthetic_s2


def get_inputs(
    bbox: Tuple[float, float, float, float],
    flood_date: str,
    offline: bool = False,
    use_synthetic_fallback: bool = True,
) -> Dict[str, Any]:
    """Contract interface: Ingest all EO and vector inputs for bbox and flood_date."""
    s1_data = fetch_s1(bbox, flood_date, offline=offline, use_synthetic_fallback=use_synthetic_fallback)
    s2_data = fetch_s2(bbox, flood_date, offline=offline, use_synthetic_fallback=use_synthetic_fallback)
    dem_data = fetch_dem(bbox, flood_date, offline=offline, use_synthetic_fallback=use_synthetic_fallback)
    osm_data = fetch_osm(bbox, snapshot_date="2026-07-27", offline=offline, use_synthetic_fallback=use_synthetic_fallback)
    return {
        "s1": s1_data,
        "s2": s2_data,
        "dem": dem_data,
        "osm": osm_data,
    }


__all__ = [
    "get_inputs",
    "fetch_s1",
    "generate_synthetic_s1",
    "S1SceneMetadata",
    "S1DataPair",
    "fetch_s2",
    "generate_synthetic_s2",
    "S2SceneMetadata",
    "S2Data",
    "compute_ndwi",
    "fetch_dem",
    "generate_synthetic_dem",
    "DEMMetadata",
    "DEMData",
    "fetch_osm",
    "generate_synthetic_osm",
    "OSMData",
    "MANDATORY_MAX_SNAPSHOT_DATE",
    "get_cache_key",
    "get_cache_path",
    "save_json_cache",
    "load_json_cache",
    "TriageFetchError",
    "CredentialsError",
    "SceneNotFoundError",
    "CloudCoverExceededError",
    "OrbitMismatchError",
    "OhsomeAPIError",
    "DateConstraintError",
]
