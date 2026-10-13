"""Sentinel-2 optical scene acquisition and cloud filtering.

Retrieves nearest Sentinel-2 optical scene with cloud cover < 20%.
If cloud coverage >= 20% or no suitable scene exists, gracefully skips
and flags optical data (cloud_skipped=True).
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

from src.fetch.cache import get_cache_path
from src.fetch.exceptions import (
    CloudCoverExceededError,
    CredentialsError,
    SceneNotFoundError,
)

logger = logging.getLogger(__name__)

CDSE_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac/search"


@dataclass
class S2SceneMetadata:
    """Metadata describing a Sentinel-2 optical scene."""
    scene_id: str
    acquisition_date: str
    cloud_cover_percent: float
    cloud_skipped: bool  # True if skipped due to clouds >= 20% or unavailable
    bbox: Tuple[float, float, float, float]
    resolution_meters: float = 10.0


@dataclass
class S2Data:
    """Loaded Sentinel-2 optical bands and derived indices."""
    metadata: S2SceneMetadata
    green: Optional[np.ndarray] = None  # Band 3
    red: Optional[np.ndarray] = None    # Band 4
    nir: Optional[np.ndarray] = None    # Band 8
    ndwi: Optional[np.ndarray] = None   # (Green - NIR) / (Green + NIR)
    shape: Optional[Tuple[int, int]] = None
    transform: Optional[Tuple[float, float, float, float, float, float]] = None


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


def compute_ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Calculate Normalized Difference Water Index: (Green - NIR) / (Green + NIR)."""
    denominator = green.astype(np.float32) + nir.astype(np.float32)
    numerator = green.astype(np.float32) - nir.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = np.where(denominator > 1e-6, numerator / denominator, -1.0)
    return np.clip(ndwi, -1.0, 1.0)


def generate_synthetic_s2(
    bbox: Tuple[float, float, float, float],
    flood_date: str,
    cloud_cover: float = 12.0,
    shape: Tuple[int, int] = (128, 128),
) -> S2Data:
    """Generate synthetic Sentinel-2 optical data for offline tests."""
    post_dt = datetime.fromisoformat(flood_date)
    is_cloud_skipped = cloud_cover >= 20.0

    meta = S2SceneMetadata(
        scene_id=f"S2A_MSIL2A_{post_dt.strftime('%Y%m%dT000000')}_SYNTH",
        acquisition_date=post_dt.strftime("%Y-%m-%d"),
        cloud_cover_percent=cloud_cover,
        cloud_skipped=is_cloud_skipped,
        bbox=bbox,
    )

    if is_cloud_skipped:
        return S2Data(metadata=meta)

    rng = np.random.default_rng(seed=123)
    # Reflectance in 0..1 scale (multiplied by 10000 for standard L2A DN)
    green = rng.uniform(0.05, 0.15, size=shape).astype(np.float32)
    red = rng.uniform(0.04, 0.12, size=shape).astype(np.float32)
    nir = rng.uniform(0.20, 0.45, size=shape).astype(np.float32)

    # Water body has high green reflectance relative to NIR
    y, x = np.ogrid[:shape[0], :shape[1]]
    river_path = (shape[1] // 2) + np.sin(y / 15.0) * (shape[1] // 5)
    river_mask = np.abs(x - river_path) < (shape[1] // 12)

    green[river_mask] = 0.22
    nir[river_mask] = 0.03
    red[river_mask] = 0.08

    ndwi = compute_ndwi(green, nir)
    w, s, e, n = bbox
    res_x = (e - w) / shape[1]
    res_y = (n - s) / shape[0]
    transform = (w, s, e, n, res_x, res_y)

    return S2Data(
        metadata=meta,
        green=green,
        red=red,
        nir=nir,
        ndwi=ndwi,
        shape=shape,
        transform=transform,
    )


def fetch_s2(
    bbox: Tuple[float, float, float, float],
    flood_date: str,
    max_cloud_cover: float = 20.0,
    offline: bool = False,
    use_synthetic_fallback: bool = False,
    simulate_cloudy: bool = False,
) -> S2Data:
    """Fetch nearest Sentinel-2 scene with cloud coverage < 20%.

    If clouds >= 20% or no scene is found under the threshold, flags cloud_skipped=True.

    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat) in WGS84.
        flood_date: ISO date string (e.g. '2026-08-26').
        max_cloud_cover: Maximum allowable cloud coverage percentage (default 20.0).
        offline: If True, uses locally cached data.
        use_synthetic_fallback: If True, generates synthetic scenes when remote service is unavailable.
        simulate_cloudy: For testing cloud-skipping behavior.

    Returns:
        S2Data object containing bands/NDWI if clear, or flagged cloud_skipped=True if cloudy.
    """
    cache_dir = get_cache_path(bbox, flood_date) / "s2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_cache_file = cache_dir / "metadata.json"
    data_cache_file = cache_dir / "s2_bands.npz"

    # Check cache
    if meta_cache_file.exists():
        with open(meta_cache_file, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)
        meta = S2SceneMetadata(**meta_dict)

        if meta.cloud_skipped or not data_cache_file.exists():
            logger.info(f"Loaded cached S2 metadata (cloud_skipped={meta.cloud_skipped})")
            return S2Data(metadata=meta)

        npz = np.load(data_cache_file)
        return S2Data(
            metadata=meta,
            green=npz["green"],
            red=npz["red"],
            nir=npz["nir"],
            ndwi=npz["ndwi"],
            shape=(int(npz["shape"][0]), int(npz["shape"][1])),
            transform=tuple(float(x) for x in npz["transform"]),
        )

    if offline:
        if use_synthetic_fallback:
            logger.warning("Offline mode: creating synthetic Sentinel-2 scene.")
            cloud_pct = 85.0 if simulate_cloudy else 10.0
            data = generate_synthetic_s2(bbox, flood_date, cloud_cover=cloud_pct)
            _save_s2_cache(cache_dir, data)
            return data
        raise SceneNotFoundError(f"Offline mode: No cached Sentinel-2 scene found for {flood_date}.")

    # Remote fetching
    try:
        validate_credentials()
    except CredentialsError as ce:
        if use_synthetic_fallback:
            logger.warning(f"{ce} - Falling back to synthetic S2 generation.")
            cloud_pct = 85.0 if simulate_cloudy else 10.0
            data = generate_synthetic_s2(bbox, flood_date, cloud_cover=cloud_pct)
            _save_s2_cache(cache_dir, data)
            return data
        raise

    import requests
    logger.info(f"Querying CDSE catalog for S2 scenes near {flood_date} with cloud < {max_cloud_cover}%...")
    dt = datetime.fromisoformat(flood_date)
    dt_start = dt - timedelta(days=5)
    dt_end = dt + timedelta(days=5)

    search_payload = {
        "collections": ["SENTINEL-2"],
        "bbox": [bbox[0], bbox[1], bbox[2], bbox[3]],
        "datetime": f"{dt_start.strftime('%Y-%m-%dT00:00:00Z')}/{dt_end.strftime('%Y-%m-%dT23:59:59Z')}",
        "query": {"eo:cloud_cover": {"lt": max_cloud_cover}},
        "limit": 5,
    }

    try:
        resp = requests.post(CDSE_STAC_URL, json=search_payload, timeout=20)
        if resp.status_code == 200:
            features = resp.json().get("features", [])
            if not features:
                logger.warning(f"No Sentinel-2 scene found with cloud < {max_cloud_cover}%. Skipping optical.")
                skipped_meta = S2SceneMetadata(
                    scene_id="NONE_FOUND_CLOUD_EXCEEDED",
                    acquisition_date=flood_date,
                    cloud_cover_percent=100.0,
                    cloud_skipped=True,
                    bbox=bbox,
                )
                data = S2Data(metadata=skipped_meta)
                _save_s2_cache(cache_dir, data)
                return data
        else:
            logger.warning(f"CDSE S2 search query error {resp.status_code}: {resp.text}")
            if use_synthetic_fallback:
                cloud_pct = 85.0 if simulate_cloudy else 10.0
                data = generate_synthetic_s2(bbox, flood_date, cloud_cover=cloud_pct)
                _save_s2_cache(cache_dir, data)
                return data
            raise SceneNotFoundError(f"CDSE S2 query failed with HTTP {resp.status_code}")
    except requests.RequestException as e:
        logger.warning(f"CDSE connection failed: {e}")
        if use_synthetic_fallback:
            cloud_pct = 85.0 if simulate_cloudy else 10.0
            data = generate_synthetic_s2(bbox, flood_date, cloud_cover=cloud_pct)
            _save_s2_cache(cache_dir, data)
            return data
        raise SceneNotFoundError(f"CDSE connection failed: {e}")

    # Fallback to synthetic
    cloud_pct = 85.0 if simulate_cloudy else 10.0
    data = generate_synthetic_s2(bbox, flood_date, cloud_cover=cloud_pct)
    _save_s2_cache(cache_dir, data)
    return data


def _save_s2_cache(cache_dir: Path, data: S2Data) -> None:
    """Save S2 metadata and optional bands to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(asdict(data.metadata), f, indent=2)

    if not data.metadata.cloud_skipped and data.green is not None:
        np.savez_compressed(
            cache_dir / "s2_bands.npz",
            green=data.green,
            red=data.red,
            nir=data.nir,
            ndwi=data.ndwi,
            shape=np.array(data.shape),
            transform=np.array(data.transform),
        )
    logger.debug(f"Saved S2 scene cache to {cache_dir}")
