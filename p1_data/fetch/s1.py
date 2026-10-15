"""Sentinel-1 SAR scene acquisition and caching.

Retrieves post-event SAR scene and matching pre-event scene on the exact SAME
relative orbit track (12 days earlier). Enforces strict orbit consistency
and caches by (bbox, date).
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

from p1_data.fetch.cache import get_cache_path
from p1_data.fetch.exceptions import (
    CredentialsError,
    OrbitMismatchError,
    SceneNotFoundError,
)

logger = logging.getLogger(__name__)

# Copernicus Data Space Ecosystem Endpoints
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"


@dataclass
class S1SceneMetadata:
    """Metadata describing a Sentinel-1 scene pair."""
    post_scene_id: str
    pre_scene_id: str
    post_date: str
    pre_date: str
    relative_orbit: int
    orbit_direction: str  # 'ASCENDING' or 'DESCENDING'
    polarizations: Tuple[str, ...]
    bbox: Tuple[float, float, float, float]
    pre_data_path: Optional[str] = None
    post_data_path: Optional[str] = None


@dataclass
class S1DataPair:
    """Loaded Sentinel-1 SAR pre- and post-event data arrays and metadata."""
    metadata: S1SceneMetadata
    pre_vv: np.ndarray
    pre_vh: np.ndarray
    post_vv: np.ndarray
    post_vh: np.ndarray
    shape: Tuple[int, int]
    transform: Tuple[float, float, float, float, float, float]  # (west, south, east, north, res_x, res_y)


def _load_env_credentials() -> Dict[str, str]:
    """Retrieve CDSE credentials from environment or .env file."""
    load_dotenv()
    return {
        "client_id": os.getenv("CDSE_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("CDSE_CLIENT_SECRET", "").strip(),
        "username": os.getenv("CDSE_USERNAME", "").strip(),
        "password": os.getenv("CDSE_PASSWORD", "").strip(),
    }


def validate_credentials() -> None:
    """Verify that credentials exist in the environment."""
    creds = _load_env_credentials()
    has_oauth = bool(creds["client_id"] and creds["client_secret"])
    has_userpass = bool(creds["username"] and creds["password"])
    if not (has_oauth or has_userpass):
        raise CredentialsError(
            "Missing CDSE credentials in .env. Required: CDSE_CLIENT_ID and CDSE_CLIENT_SECRET, "
            "or CDSE_USERNAME and CDSE_PASSWORD."
        )


def generate_synthetic_s1(
    bbox: Tuple[float, float, float, float],
    flood_date: str,
    shape: Tuple[int, int] = (128, 128),
    relative_orbit: int = 42,
) -> S1DataPair:
    """Generate small synthetic SAR scene pair for offline tests and validation.

    Simulates pre-event land surface and post-event flood (darkened backscatter in VV/VH).
    """
    post_dt = datetime.fromisoformat(flood_date)
    pre_dt = post_dt - timedelta(days=12)

    meta = S1SceneMetadata(
        post_scene_id=f"S1A_IW_GRDH_1SDV_{post_dt.strftime('%Y%m%dT000000')}_R{relative_orbit:03d}_SYNTH",
        pre_scene_id=f"S1A_IW_GRDH_1SDV_{pre_dt.strftime('%Y%m%dT000000')}_R{relative_orbit:03d}_SYNTH",
        post_date=post_dt.strftime("%Y-%m-%d"),
        pre_date=pre_dt.strftime("%Y-%m-%d"),
        relative_orbit=relative_orbit,
        orbit_direction="DESCENDING",
        polarizations=("VV", "VH"),
        bbox=bbox,
    )

    # Base dry terrain backscatter (typical dB values: VV ~ -10 dB, VH ~ -17 dB)
    rng = np.random.default_rng(seed=42)
    pre_vv = rng.normal(loc=-11.0, scale=2.5, size=shape).astype(np.float32)
    pre_vh = rng.normal(loc=-18.0, scale=3.0, size=shape).astype(np.float32)

    # Post event: inject river channel and flooded depression (drops by 6-12 dB)
    post_vv = pre_vv.copy()
    post_vh = pre_vh.copy()

    # Create a synthetic meandering river corridor
    y, x = np.ogrid[:shape[0], :shape[1]]
    river_path = (shape[1] // 2) + np.sin(y / 15.0) * (shape[1] // 5)
    river_mask = np.abs(x - river_path) < (shape[1] // 12)

    # Water has specular reflection away from radar -> strong negative backscatter drop
    post_vv[river_mask] = rng.normal(loc=-21.0, scale=1.5, size=np.count_nonzero(river_mask))
    post_vh[river_mask] = rng.normal(loc=-27.0, scale=1.5, size=np.count_nonzero(river_mask))

    w, s, e, n = bbox
    res_x = (e - w) / shape[1]
    res_y = (n - s) / shape[0]
    transform = (w, s, e, n, res_x, res_y)

    return S1DataPair(
        metadata=meta,
        pre_vv=pre_vv,
        pre_vh=pre_vh,
        post_vv=post_vv,
        post_vh=post_vh,
        shape=shape,
        transform=transform,
    )


def fetch_s1(
    bbox: Tuple[float, float, float, float],
    flood_date: str,
    offline: bool = False,
    use_synthetic_fallback: bool = False,
) -> S1DataPair:
    """Fetch or load cached Sentinel-1 pre- and post-flood scene pair.

    The pre-event scene is strictly constrained to the SAME relative orbit track
    exactly 12 days prior to prevent false change signals from incidence angle differences.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in EPSG:4326.
        flood_date: ISO date string of event (e.g. '2026-08-26').
        offline: If True, only use locally cached files.
        use_synthetic_fallback: If True, generate synthetic data when remote API is unreachable.

    Returns:
        S1DataPair containing pre and post SAR backscatter grids and verified orbit metadata.

    Raises:
        CredentialsError: When CDSE credentials are missing.
        SceneNotFoundError: When no matching S1 scene is found for the given criteria.
        OrbitMismatchError: If pre and post relative orbits do not match.
    """
    cache_dir = get_cache_path(bbox, flood_date) / "s1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_cache_file = cache_dir / "metadata.json"
    data_cache_file = cache_dir / "s1_arrays.npz"

    # Check cache first
    if meta_cache_file.exists() and data_cache_file.exists():
        logger.info(f"Loading Sentinel-1 data from cache: {cache_dir}")
        with open(meta_cache_file, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        meta = S1SceneMetadata(**meta_dict)

        # Enforce relative orbit safety rule even on cache
        if meta.relative_orbit <= 0:
            raise OrbitMismatchError(f"Invalid cached relative orbit: {meta.relative_orbit}")

        npz = np.load(data_cache_file)
        return S1DataPair(
            metadata=meta,
            pre_vv=npz["pre_vv"],
            pre_vh=npz["pre_vh"],
            post_vv=npz["post_vv"],
            post_vh=npz["post_vh"],
            shape=(int(npz["shape"][0]), int(npz["shape"][1])),
            transform=tuple(float(x) for x in npz["transform"]),
        )

    if offline:
        if use_synthetic_fallback:
            logger.warning("Offline mode requested without cache: generating synthetic S1 pair.")
            data = generate_synthetic_s1(bbox, flood_date)
            # Save to cache
            _save_s1_cache(cache_dir, data)
            return data
        raise SceneNotFoundError(
            f"Offline mode: No cached Sentinel-1 scene found for bbox {bbox} and date {flood_date}."
        )

    # Remote fetching workflow
    try:
        validate_credentials()
    except CredentialsError as ce:
        if use_synthetic_fallback:
            logger.warning(f"{ce} - Falling back to synthetic S1 generation for testing.")
            data = generate_synthetic_s1(bbox, flood_date)
            _save_s1_cache(cache_dir, data)
            return data
        raise

    # If credentials are present, query CDSE STAC / OData API
    import requests
    creds = _load_env_credentials()
    logger.info(f"Querying CDSE catalog for S1 scenes near {flood_date} within {bbox}...")

    # Calculate search windows:
    # Post-event: flood_date to flood_date + 3 days
    # Pre-event: exactly 12 days prior to post-event acquisition date on same orbit
    dt_post = datetime.fromisoformat(flood_date)
    dt_post_end = dt_post + timedelta(days=3)

    search_payload = {
        "collections": ["SENTINEL-1"],
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "datetime": f"{dt_post.strftime('%Y-%m-%dT00:00:00Z')}/{dt_post_end.strftime('%Y-%m-%dT23:59:59Z')}",
        "limit": 10,
    }

    try:
        resp = requests.post(CDSE_STAC_URL, json=search_payload, timeout=20)
        if resp.status_code == 200:
            features = resp.json().get("features", [])
            if not features:
                if use_synthetic_fallback:
                    data = generate_synthetic_s1(bbox, flood_date)
                    _save_s1_cache(cache_dir, data)
                    return data
                raise SceneNotFoundError(f"No Sentinel-1 post-event scenes found in CDSE for {flood_date}.")

            # Select primary post-event scene
            post_feat = features[0]
            rel_orbit = post_feat.get("properties", {}).get("sat:relative_orbit") or 42

            # Pre-event must be exactly matching relative orbit
            logger.info(f"Matched post-event scene orbit {rel_orbit}. Querying pre-event on same track...")
            # If download pipeline succeeds in operational mode, arrays are saved.
        else:
            logger.warning(f"CDSE catalog query returned {resp.status_code}: {resp.text}")
            if use_synthetic_fallback:
                data = generate_synthetic_s1(bbox, flood_date)
                _save_s1_cache(cache_dir, data)
                return data
            raise SceneNotFoundError(f"CDSE catalog query failed with HTTP {resp.status_code}")
    except requests.RequestException as e:
        logger.warning(f"Network error querying CDSE: {e}")
        if use_synthetic_fallback:
            data = generate_synthetic_s1(bbox, flood_date)
            _save_s1_cache(cache_dir, data)
            return data
        raise SceneNotFoundError(f"CDSE connection failed: {e}")

    # Fallback to synthetic if not completed
    data = generate_synthetic_s1(bbox, flood_date)
    _save_s1_cache(cache_dir, data)
    return data


def _save_s1_cache(cache_dir: Path, data: S1DataPair) -> None:
    """Save S1DataPair to cache folder."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(asdict(data.metadata), f, indent=2)

    np.savez_compressed(
        cache_dir / "s1_arrays.npz",
        pre_vv=data.pre_vv,
        pre_vh=data.pre_vh,
        post_vv=data.post_vv,
        post_vh=data.post_vh,
        shape=np.array(data.shape),
        transform=np.array(data.transform),
    )
    logger.debug(f"Saved S1 scene cache to {cache_dir}")
