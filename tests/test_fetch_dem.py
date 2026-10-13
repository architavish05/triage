"""Unit tests for Copernicus DEM fetch and caching module."""

from src.fetch.dem import fetch_dem, generate_synthetic_dem, get_tile_identifiers


def test_tile_identifiers():
    """Verify calculation of intersecting 1-degree Copernicus DEM tiles."""
    bbox = (85.2, 27.8, 85.9, 28.3)
    tiles = get_tile_identifiers(bbox)
    assert len(tiles) >= 2
    assert "Copernicus_DSM_COG_10_N27_00_E085_00_DEM" in tiles
    assert "Copernicus_DSM_COG_10_N28_00_E085_00_DEM" in tiles


def test_generate_synthetic_dem():
    """Verify synthetic DEM provides realistic elevation gradients and metadata."""
    bbox = (85.0, 27.5, 85.5, 28.0)
    dem = generate_synthetic_dem(bbox, shape=(64, 64), base_elevation=1000.0, relief=900.0)

    assert dem.shape == (64, 64)
    assert dem.elevation.shape == (64, 64)
    assert dem.metadata.min_elevation >= 800.0
    assert dem.metadata.max_elevation <= 3000.0
    assert "Copernicus WorldDEM-30" in dem.metadata.attribution


def test_dem_caching(tmp_path, monkeypatch):
    """Verify DEM array is saved and reloaded from cache."""
    monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
    bbox = (85.1, 27.6, 85.3, 27.8)

    # First fetch writes to cache
    dem1 = fetch_dem(bbox, offline=False, use_synthetic_fallback=True)
    assert dem1 is not None

    # Second fetch loads from cache in strict offline mode
    dem2 = fetch_dem(bbox, offline=True, use_synthetic_fallback=False)
    assert dem2.metadata.product_name == dem1.metadata.product_name
    assert dem2.elevation.shape == dem1.elevation.shape
