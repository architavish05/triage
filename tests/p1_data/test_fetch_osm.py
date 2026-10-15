"""Unit tests for OpenStreetMap pre-event snapshot fetcher via ohsome."""

import pytest
from p1_data.fetch.exceptions import DateConstraintError
from p1_data.fetch.osm import (
    MANDATORY_MAX_SNAPSHOT_DATE,
    fetch_osm,
    generate_synthetic_osm,
    validate_snapshot_date,
)


def test_date_constraint_enforcement():
    """Verify that any date later than 2026-07-27 raises DateConstraintError."""
    # Valid dates
    validate_snapshot_date("2026-07-27")
    validate_snapshot_date("2026-06-01")

    # Invalid post-event dates violate hard rule
    with pytest.raises(DateConstraintError):
        validate_snapshot_date("2026-07-28")

    with pytest.raises(DateConstraintError):
        validate_snapshot_date("2026-08-26")


def test_generate_synthetic_osm():
    """Verify synthetic OSM includes all essential infrastructure layers."""
    bbox = (85.1, 27.8, 85.5, 28.2)
    osm = generate_synthetic_osm(bbox)

    assert osm.snapshot_date == MANDATORY_MAX_SNAPSHOT_DATE
    counts = osm.counts
    assert counts["buildings"] > 0
    assert counts["highways"] > 0
    assert counts["bridges"] > 0
    assert counts["hospitals"] > 0
    assert counts["places"] > 0
    assert "OpenStreetMap" in osm.attribution


def test_osm_caching(tmp_path, monkeypatch):
    """Verify OSM snapshot caching and offline retrieval."""
    monkeypatch.setenv("TRIAGE_CACHE_DIR", str(tmp_path))
    bbox = (85.2, 27.7, 85.4, 27.9)

    # First fetch caches features
    osm1 = fetch_osm(bbox, snapshot_date="2026-07-27", offline=False, use_synthetic_fallback=True)
    assert osm1.counts["buildings"] > 0

    # Second fetch loads from cache without network
    osm2 = fetch_osm(bbox, snapshot_date="2026-07-27", offline=True, use_synthetic_fallback=False)
    assert osm2.counts["buildings"] == osm1.counts["buildings"]
    assert osm2.counts["places"] == osm1.counts["places"]
