"""Unit tests for Phase 9 bilingual reporting, radio SMS summary, and pipeline run."""

from pathlib import Path
import pytest

from p1_data.pipeline import run
from p4_app.report.build_report import (
    format_template_context,
    generate_situation_reports,
)
from p4_app.report.sms_summary import ATTRIBUTION_SHORT, generate_sms_summary
from p3_terrain.rescue.trapped_risk import RescuePriorityItem


def test_format_template_context_and_render(tmp_path):
    """Verify situation report generation in English and Nepali using fixed Jinja templates."""
    flood_date = "2026-08-26"
    bbox = (85.2, 27.8, 85.4, 28.1)
    damage_summary = {
        "damaged_buildings_count": 42,
        "damaged_roads_km": 14.5,
        "damaged_bridges_count": 3,
    }
    priority_items = [
        RescuePriorityItem(
            rank=1,
            settlement_id="s1",
            settlement_name="Barhabise",
            place_type="town",
            coordinates=(85.3, 27.9),
            trapped_risk=0.85,
            cutoff_prob=0.92,
            buildings_at_risk=28,
            stranded_population_estimate=140,
            population_data_type="estimate",
            household_size_assumption=5.0,
            priority_score=21.9,
            is_on_dry_island=True,
            escape_possible=False,
            nearest_hospital="Primary Health Centre",
            recommended_action="IMMEDIATE HELICOPTER EXTRACTION",
        )
    ]

    out_dir = tmp_path / "reports"
    reports = generate_situation_reports(
        flood_date=flood_date,
        bbox=bbox,
        damage_summary=damage_summary,
        priority_items=priority_items,
        output_dir=out_dir,
        dry_islands_count=2,
    )

    assert "html_en" in reports
    assert "html_ne" in reports
    assert reports["html_en"].exists()
    assert reports["html_ne"].exists()

    with open(reports["html_en"], "r", encoding="utf-8") as f:
        content_en = f.read()
        # Verify exact numbers from pipeline outputs
        assert "42" in content_en  # damaged buildings
        assert "14.5 km" in content_en  # roads cut
        assert "Barhabise" in content_en
        assert "140" in content_en  # stranded estimate
        assert "Contains modified Copernicus Sentinel data 2026." in content_en
        assert "Copernicus WorldDEM-30" in content_en
        assert "OpenStreetMap" in content_en

    with open(reports["html_ne"], "r", encoding="utf-8") as f:
        content_ne = f.read()
        assert "ट्रायज विपद् वस्तुस्थिति प्रतिवेदन" in content_ne
        assert "42" in content_ne
        assert "Barhabise" in content_ne
        assert "Contains modified Copernicus Sentinel data 2026." in content_ne


def test_sms_summary_content(tmp_path):
    """Verify compact SMS / VHF radio summary formatting and attribution."""
    damage_summary = {
        "damaged_buildings_count": 12,
        "damaged_roads_km": 5.2,
        "damaged_bridges_count": 1,
    }
    priority_items = [
        RescuePriorityItem(
            rank=1,
            settlement_id="s1",
            settlement_name="Tatopani",
            place_type="village",
            coordinates=(85.35, 27.95),
            trapped_risk=0.90,
            cutoff_prob=0.95,
            buildings_at_risk=10,
            stranded_population_estimate=50,
            population_data_type="estimate",
            household_size_assumption=5.0,
            priority_score=8.55,
            is_on_dry_island=True,
            escape_possible=False,
            nearest_hospital="Health Post",
            recommended_action="IMMEDIATE AIRDROP",
        )
    ]

    sms_file = tmp_path / "sms.txt"
    text = generate_sms_summary(
        flood_date="2026-08-26",
        bbox=(85.2, 27.8, 85.4, 28.1),
        damage_summary=damage_summary,
        priority_items=priority_items,
        output_path=sms_file,
    )

    assert "TRIAGE DISASTER SITREP: 2026-08-26" in text
    assert "12 bldgs damaged" in text
    assert "Tatopani" in text
    assert "Cutoff: 95%" in text
    assert ATTRIBUTION_SHORT in text
    assert sms_file.exists()


def test_pipeline_run_end_to_end(tmp_path):
    """Verify run(bbox, flood_date) pipeline executes completely and produces all outputs."""
    bbox = (85.20, 27.80, 85.40, 28.10)
    flood_date = "2026-08-26"

    results = run(bbox=bbox, flood_date=flood_date, offline=False, output_dir=tmp_path / "run_out")

    assert results is not None
    assert "class_map" in results
    assert "confidence" in results
    assert "damage" in results
    assert "isolation_records" in results
    assert "downstream_trace" in results
    assert "rescue_priorities" in results
    assert "landing_zones" in results
    assert "report_paths" in results
    assert "sms_summary" in results

    # Verify report files were created on disk
    assert results["report_paths"]["html_en"].exists()
    assert results["report_paths"]["html_ne"].exists()
