"""Integrated fetch test for Phase 1 data pipeline."""

from p1_data.fetch import fetch_dem, fetch_osm, fetch_s1, fetch_s2


def test_phase1_integrated_fetch(tmp_path, monkeypatch):
    """Test full Phase 1 ingestion across S1, S2, DEM, and OSM for arbitrary bbox & date."""
    monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))

    # Bhote Koshi / Trishuli river corridor test bounding box
    bbox = (85.20, 27.80, 85.45, 28.15)
    flood_date = "2026-08-26"

    # Ingest S1 SAR pre/post pair
    s1_data = fetch_s1(bbox, flood_date, offline=False, use_synthetic_fallback=True)
    assert s1_data.metadata.relative_orbit > 0
    assert s1_data.metadata.post_date == "2026-08-26"
    assert s1_data.metadata.pre_date == "2026-08-14"  # Exactly 12 days earlier on same relative orbit

    # Ingest S2 Optical
    s2_data = fetch_s2(bbox, flood_date, offline=False, use_synthetic_fallback=True)
    assert s2_data.metadata.acquisition_date == "2026-08-26"

    # Ingest Copernicus DEM 30m
    dem_data = fetch_dem(bbox, flood_date, offline=False, use_synthetic_fallback=True)
    assert dem_data.metadata.resolution_meters == 30.0
    assert dem_data.elevation.ndim == 2

    # Ingest pre-event OSM (strictly <= 2026-07-27)
    osm_data = fetch_osm(bbox, snapshot_date="2026-07-27", offline=False, use_synthetic_fallback=True)
    counts = osm_data.counts
    assert counts["buildings"] > 0
    assert counts["highways"] > 0
    assert counts["bridges"] > 0
    assert counts["hospitals"] > 0
    assert counts["places"] > 0

    # Verify everything cached under the expected directory structure
    event_cache_dirs = [d for d in tmp_path.iterdir() if d.name.startswith(flood_date)]
    osm_cache_dirs = [d for d in tmp_path.iterdir() if d.name.startswith("2026-07-27")]
    assert len(event_cache_dirs) == 1
    assert len(osm_cache_dirs) == 1

    event_subfolders = [f.name for f in event_cache_dirs[0].iterdir() if f.is_dir()]
    assert "s1" in event_subfolders
    assert "s2" in event_subfolders
    assert "dem" in event_subfolders

    osm_subfolders = [f.name for f in osm_cache_dirs[0].iterdir() if f.is_dir()]
    assert "osm" in osm_subfolders
