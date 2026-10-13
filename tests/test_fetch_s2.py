"""Unit tests for Sentinel-2 optical fetch and cloud thresholding module."""

import pytest
from src.fetch.exceptions import CredentialsError
from src.fetch.s2 import compute_ndwi, fetch_s2, generate_synthetic_s2


def test_generate_synthetic_s2_and_ndwi():
    """Verify Sentinel-2 generation, NDWI calculation, and water signal."""
    bbox = (85.2, 27.8, 85.4, 28.0)
    flood_date = "2026-08-26"
    data = generate_synthetic_s2(bbox, flood_date, cloud_cover=12.0, shape=(64, 64))

    assert not data.metadata.cloud_skipped
    assert data.green is not None
    assert data.nir is not None
    assert data.ndwi is not None
    assert data.ndwi.shape == (64, 64)

    # Water pixels must have positive NDWI (typically > 0.3)
    assert data.ndwi.max() > 0.2


def test_s2_cloud_skipping():
    """Verify scenes with cloud cover >= 20% are flagged as cloud_skipped."""
    bbox = (85.2, 27.8, 85.4, 28.0)
    flood_date = "2026-08-26"
    # When cloud is 45%, optical bands should be skipped and flagged
    data = generate_synthetic_s2(bbox, flood_date, cloud_cover=45.0)

    assert data.metadata.cloud_skipped
    assert data.green is None
    assert data.ndwi is None


def test_s2_missing_credentials(monkeypatch):
    """Verify CredentialsError is raised when CDSE credentials are empty and fallback disabled."""
    monkeypatch.setenv("CDSE_CLIENT_ID", "")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "")
    monkeypatch.setenv("CDSE_USERNAME", "")
    monkeypatch.setenv("CDSE_PASSWORD", "")

    bbox = (12.0, 44.0, 12.5, 44.5)
    with pytest.raises(CredentialsError):
        fetch_s2(bbox, "2026-08-26", offline=False, use_synthetic_fallback=False)


def test_s2_caching(tmp_path, monkeypatch):
    """Verify S2 caching across multiple calls."""
    monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
    bbox = (85.2, 27.8, 85.3, 27.9)
    flood_date = "2026-08-26"

    # Fetch and cache
    data1 = fetch_s2(bbox, flood_date, offline=False, use_synthetic_fallback=True)
    assert data1 is not None

    # Load from cache in offline mode
    data2 = fetch_s2(bbox, flood_date, offline=True, use_synthetic_fallback=False)
    assert data2.metadata.scene_id == data1.metadata.scene_id
