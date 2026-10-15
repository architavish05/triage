"""Regional prefetching utility for offline disaster responder packs.

Downloads and caches Sentinel-1, Sentinel-2, Copernicus DEM, and OSM snapshot data
in advance, enabling fully air-gapped pipeline execution in disaster zones.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from p1_data.fetch import (
    DEMData,
    OSMData,
    S1DataPair,
    S2Data,
    fetch_dem,
    fetch_osm,
    fetch_s1,
    fetch_s2,
    get_cache_path,
)

logger = logging.getLogger(__name__)


def prefetch_region(
    bbox: Tuple[float, float, float, float],
    flood_date: str,
    base_cache_dir: Optional[Path] = None,
    use_synthetic_fallback: bool = True,
) -> Dict[str, Any]:
    """Prefetch and cache all multi-sensor inputs for a designated operational region.

    Ensures that an area of interest is fully cached locally so that subsequent calls to
    `pipeline.run(bbox, flood_date, offline=True)` require zero internet connectivity.

    Args:
        bbox: Target geographic bounding box (min_lon, min_lat, max_lon, max_lat).
        flood_date: ISO date of event (e.g. '2026-08-26').
        base_cache_dir: Optional custom cache root path.
        use_synthetic_fallback: If True, uses high-fidelity synthetic data when external APIs are offline.

    Returns:
        Dict detailing cached asset counts, paths, and metadata.
    """
    logger.info(f"=== Starting prefetch for region {bbox} at date {flood_date} ===")
    target_cache_dir = get_cache_path(bbox, flood_date, base_dir=base_cache_dir)

    # 1. Prefetch Sentinel-1 SAR pair (same relative orbit)
    logger.info("Prefetching Sentinel-1 SAR scene pair...")
    s1_data: S1DataPair = fetch_s1(
        bbox=bbox,
        flood_date=flood_date,
        offline=False,
        use_synthetic_fallback=use_synthetic_fallback,
    )

    # 2. Prefetch Sentinel-2 Optical (cloud < 20%)
    logger.info("Prefetching Sentinel-2 optical scene...")
    s2_data: S2Data = fetch_s2(
        bbox=bbox,
        flood_date=flood_date,
        offline=False,
        use_synthetic_fallback=use_synthetic_fallback,
    )

    # 3. Prefetch Copernicus DEM 30m
    logger.info("Prefetching Copernicus DEM 30m...")
    dem_data: DEMData = fetch_dem(
        bbox=bbox,
        flood_date=flood_date,
        offline=False,
        use_synthetic_fallback=use_synthetic_fallback,
    )

    # 4. Prefetch pre-event OSM snapshot at 2026-07-27
    logger.info("Prefetching OpenStreetMap pre-event snapshot (<= 2026-07-27)...")
    osm_data: OSMData = fetch_osm(
        bbox=bbox,
        snapshot_date="2026-07-27",
        offline=False,
        use_synthetic_fallback=use_synthetic_fallback,
    )

    summary = {
        "status": "PREFETCH_COMPLETE",
        "bbox": bbox,
        "flood_date": flood_date,
        "cache_dir": str(target_cache_dir),
        "s1_post_scene": s1_data.metadata.post_scene_id,
        "s1_pre_scene": s1_data.metadata.pre_scene_id,
        "s1_relative_orbit": s1_data.metadata.relative_orbit,
        "s2_scene": s2_data.metadata.scene_id,
        "s2_cloud_skipped": s2_data.metadata.cloud_skipped,
        "dem_product": dem_data.metadata.product_name,
        "osm_features_cached": osm_data.counts,
    }

    logger.info(f"Prefetch complete. Cached to: {target_cache_dir}")
    return summary
