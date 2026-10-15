"""Uniform caching utilities for satellite and geospatial data."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")


def get_cache_key(bbox: Tuple[float, float, float, float], date_str: str) -> str:
    """Generate a consistent cache directory key for a given bounding box and date.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        date_str: ISO date string, e.g. '2026-08-26'

    Returns:
        String key representing the geographic and temporal scope.
    """
    bbox_rounded = tuple(round(coord, 4) for coord in bbox)
    raw_str = f"{date_str}_{bbox_rounded[0]}_{bbox_rounded[1]}_{bbox_rounded[2]}_{bbox_rounded[3]}"
    # Return human-readable prefix with hash suffix for uniqueness & valid path
    digest = hashlib.md5(raw_str.encode("utf-8")).hexdigest()[:8]
    return f"{date_str}_{bbox_rounded[0]}_{bbox_rounded[1]}_{bbox_rounded[2]}_{bbox_rounded[3]}_{digest}"


def get_cache_path(bbox: Tuple[float, float, float, float], date_str: str, base_dir: Optional[Path] = None) -> Path:
    """Return and create the directory path for cached files."""
    if base_dir is None:
        env_dir = os.getenv("TRIAGE_CACHE_DIR")
        base_dir = Path(env_dir) if env_dir else DEFAULT_CACHE_DIR
    target = base_dir / get_cache_key(bbox, date_str)
    target.mkdir(parents=True, exist_ok=True)
    return target


def save_json_cache(bbox: Tuple[float, float, float, float], date_str: str, filename: str, data: Dict[str, Any]) -> Path:
    """Save serializable data to JSON in the cache directory."""
    folder = get_cache_path(bbox, date_str)
    file_path = folder / filename
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.debug(f"Cached JSON saved to {file_path}")
    return file_path


def load_json_cache(bbox: Tuple[float, float, float, float], date_str: str, filename: str) -> Optional[Dict[str, Any]]:
    """Load JSON data from cache if it exists, otherwise return None."""
    folder = get_cache_path(bbox, date_str)
    file_path = folder / filename
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            logger.debug(f"Loaded cached JSON from {file_path}")
            return json.load(f)
    return None
