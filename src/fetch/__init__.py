"""Data ingestion and caching modules for Sentinel-1, Sentinel-2, DEM, and OSM."""

from src.fetch.cache import get_cache_key, get_cache_path, load_json_cache, save_json_cache
from src.fetch.dem import DEMData, DEMMetadata, fetch_dem, generate_synthetic_dem
from src.fetch.exceptions import (
    CloudCoverExceededError,
    CredentialsError,
    DateConstraintError,
    OhsomeAPIError,
    OrbitMismatchError,
    SceneNotFoundError,
    TriageFetchError,
)
from src.fetch.osm import MANDATORY_MAX_SNAPSHOT_DATE, OSMData, fetch_osm, generate_synthetic_osm
from src.fetch.s1 import S1DataPair, S1SceneMetadata, fetch_s1, generate_synthetic_s1
from src.fetch.s2 import S2Data, S2SceneMetadata, compute_ndwi, fetch_s2, generate_synthetic_s2

__all__ = [
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
