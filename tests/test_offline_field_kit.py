"""Unit tests for Phase 10 offline field kit, MBTiles, and air-gapped execution."""

import json
from pathlib import Path
import sqlite3
import numpy as np
import pytest

from src.offline.export_bundle import (
    create_mbtiles_from_raster,
    export_responder_bundle,
)
from src.offline.prefetch import prefetch_region
from src.pipeline import run


def test_prefetch_region(tmp_path, monkeypatch):
    """Verify prefetch_region downloads/caches all required multi-sensor layers."""
    monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
    bbox = (85.20, 27.80, 85.35, 28.00)
    flood_date = "2026-08-26"

    summary = prefetch_region(bbox, flood_date, use_synthetic_fallback=True)

    assert summary["status"] == "PREFETCH_COMPLETE"
    assert summary["s1_relative_orbit"] > 0
    assert summary["osm_features_cached"]["buildings"] > 0

    # Verify cache directory exists on disk
    cache_path = Path(summary["cache_dir"])
    assert cache_path.exists()
    assert (cache_path / "s1").exists()
    assert (cache_path / "dem").exists()


def test_create_mbtiles_database(tmp_path):
    """Verify MBTiles database creation adhering to OGC/Mapbox MBTiles spec."""
    shape = (60, 60)
    transform = (85.0, 27.0, 85.5, 27.5, 0.01, 0.01)

    class_map = np.zeros(shape, dtype=np.uint8)
    class_map[20:30, 20:30] = 1  # Water
    class_map[35:45, 10:20] = 2  # Debris
    confidence = np.full(shape, 0.85, dtype=np.float32)

    mbtiles_file = tmp_path / "test_hazard.mbtiles"
    res_path = create_mbtiles_from_raster(class_map, confidence, transform, mbtiles_file)

    assert res_path.exists()

    # Inspect SQLite schema and records
    conn = sqlite3.connect(str(res_path))
    cur = conn.cursor()

    cur.execute("SELECT name, value FROM metadata;")
    metadata = dict(cur.fetchall())
    assert metadata["format"] == "png"
    assert "Triage" in metadata["name"]

    cur.execute("SELECT zoom_level, tile_column, tile_row, LENGTH(tile_data) FROM tiles;")
    tiles = cur.fetchall()
    assert len(tiles) >= 1
    # Tile blob size should be positive
    assert tiles[0][3] > 100
    conn.close()


def test_air_gapped_pipeline_execution(tmp_path, monkeypatch):
    """Verify pipeline.run(..., offline=True) executes with network strictly disabled."""
    monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
    bbox = (85.20, 27.80, 85.35, 28.00)
    flood_date = "2026-08-26"

    # Step 1: Prefetch data into cache
    prefetch_region(bbox, flood_date, use_synthetic_fallback=True)

    # Step 2: Strictly block all outbound network requests via monkeypatch
    import requests
    def mock_network_disabled(*args, **kwargs):
        raise RuntimeError("CRITICAL: Network connection attempted while in offline mode!")

    monkeypatch.setattr(requests, "get", mock_network_disabled)
    monkeypatch.setattr(requests, "post", mock_network_disabled)

    # Step 3: Run pipeline with offline=True
    run_out_dir = tmp_path / "offline_run_output"
    results = run(bbox, flood_date, offline=True, output_dir=run_out_dir)

    assert results is not None
    assert results["class_map"].shape is not None
    assert results["damage"].summary["damaged_buildings_count"] >= 0

    # Step 4: Export full responder pack and verify ZIP bundle
    bundle_dir = run_out_dir / "bundle"
    pack_outputs = export_responder_bundle(results, bundle_dir, create_zip=True)

    assert pack_outputs["mbtiles"].exists()
    assert pack_outputs["manifest"].exists()
    assert pack_outputs["zip_pack"].exists()
    assert pack_outputs["zip_pack"].stat().st_size > 500
