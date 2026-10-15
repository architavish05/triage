"""Unit tests for Sentinel-1 fetch and cache module."""

import pytest
from p1_data.fetch.exceptions import CredentialsError, OrbitMismatchError
from p1_data.fetch.s1 import fetch_s1, generate_synthetic_s1


def test_generate_synthetic_s1():
    """Verify synthetic Sentinel-1 generates matching pre/post orbit pair with water backscatter drop."""
    bbox = (85.2, 27.8, 85.4, 28.0)
    flood_date = "2026-08-26"
    data = generate_synthetic_s1(bbox, flood_date, shape=(64, 64), relative_orbit=42)

    assert data.metadata.relative_orbit == 42
    assert data.metadata.post_date == "2026-08-26"
    assert data.metadata.pre_date == "2026-08-14"  # Exactly 12 days earlier
    assert data.shape == (64, 64)
    assert data.pre_vv.shape == (64, 64)
    assert data.post_vv.shape == (64, 64)

    # Average post-flood backscatter in water zone should be substantially lower than pre-flood
    diff = data.post_vv - data.pre_vv
    assert diff.min() < -5.0  # Significant dB drop in flood corridor


def test_s1_missing_credentials(monkeypatch):
    """Verify CredentialsError is raised when CDSE credentials are missing."""
    monkeypatch.setenv("CDSE_CLIENT_ID", "")
    monkeypatch.setenv("CDSE_CLIENT_SECRET", "")
    monkeypatch.setenv("CDSE_USERNAME", "")
    monkeypatch.setenv("CDSE_PASSWORD", "")

    bbox = (10.0, 45.0, 10.5, 45.5)
    with pytest.raises(CredentialsError):
        fetch_s1(bbox, "2026-08-26", offline=False, use_synthetic_fallback=False)


def test_s1_caching_and_fallback(tmp_path, monkeypatch):
    """Verify S1 data caches properly and reloads on second fetch."""
    monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
    bbox = (85.2, 27.8, 85.3, 27.9)
    flood_date = "2026-08-26"

    # First fetch with synthetic fallback creates cache
    data1 = fetch_s1(bbox, flood_date, offline=False, use_synthetic_fallback=True)
    assert data1 is not None

    # Second fetch should load from cache even in strict offline mode
    data2 = fetch_s1(bbox, flood_date, offline=True, use_synthetic_fallback=False)
    assert data2.metadata.post_scene_id == data1.metadata.post_scene_id
    assert data2.pre_vv.shape == data1.pre_vv.shape
