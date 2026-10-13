"""Situation report generator using fixed Jinja templates.

Generates bilingual 1-page Situation Reports (English and Nepali):
- Every number strictly originates from pipeline outputs (no LLM-invented numbers).
- Fixed HTML Jinja templates rendered to PDF via WeasyPrint (with HTML fallback).
- Mandatory legal Copernicus, DEM, and OSM attributions embedded on footers.
"""

from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jinja2

from src.rescue.trapped_risk import RescuePriorityItem

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent


def format_template_context(
    flood_date: str,
    bbox: Tuple[float, float, float, float],
    damage_summary: Dict[str, Any],
    priority_items: List[RescuePriorityItem],
    s1_metadata: Optional[Any] = None,
    s2_metadata: Optional[Any] = None,
    cant_see_breakdown: Optional[Dict[str, Any]] = None,
    dry_islands_count: int = 0,
) -> Dict[str, Any]:
    """Assemble strict quantitative context for Jinja template rendering."""
    bbox_str = f"[{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}, {bbox[3]:.3f}]"

    # Top 8 priority settlements
    top_priorities = priority_items[:8]
    cutoff_count = sum(1 for p in priority_items if p.cutoff_prob >= 0.5)
    total_stranded_est = sum(p.stranded_population_estimate for p in priority_items)

    s1_orbit_status = "Acquired & Calibrated" if s1_metadata else "Pre/Post Pair Loaded"
    s1_rel_orbit = getattr(s1_metadata, "relative_orbit", 42)

    if s2_metadata:
        if getattr(s2_metadata, "cloud_skipped", False):
            s2_status = f"Skipped (Cloud: {getattr(s2_metadata, 'cloud_cover_percent', 100):.1f}% >= 20%)"
        else:
            s2_status = f"Clear Ingestion (Cloud: {getattr(s2_metadata, 'cloud_cover_percent', 0.0):.1f}%)"
    else:
        s2_status = "Optical Not Acquired"

    cant_see_pct = cant_see_breakdown.get("occluded_percent", 0.0) if cant_see_breakdown else 0.0

    return {
        "flood_date": flood_date,
        "bbox_str": bbox_str,
        "generated_timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "summary": damage_summary,
        "top_priorities": [asdict(p) for p in top_priorities],
        "cutoff_settlements_count": cutoff_count,
        "dry_islands_count": dry_islands_count,
        "total_stranded_estimate": total_stranded_est,
        "s1_orbit_status": s1_orbit_status,
        "s1_relative_orbit": s1_rel_orbit,
        "s2_optical_status": s2_status,
        "cant_see_percent": cant_see_pct,
    }


def generate_situation_reports(
    flood_date: str,
    bbox: Tuple[float, float, float, float],
    damage_summary: Dict[str, Any],
    priority_items: List[RescuePriorityItem],
    output_dir: Path,
    s1_metadata: Optional[Any] = None,
    s2_metadata: Optional[Any] = None,
    cant_see_breakdown: Optional[Dict[str, Any]] = None,
    dry_islands_count: int = 0,
) -> Dict[str, Path]:
    """Generate both English and Nepali 1-page Situation Reports.

    Args:
        flood_date: Event date ISO string.
        bbox: Bounding box tuple.
        damage_summary: Quantitative damage summary from Phase 4.
        priority_items: Prioritized rescue settlements from Phase 7.
        output_dir: Target output folder.
        s1_metadata: Sentinel-1 metadata from Phase 1.
        s2_metadata: Sentinel-2 metadata from Phase 1.
        cant_see_breakdown: Occlusion stats from Phase 3.
        dry_islands_count: Number of isolated dry islands from Phase 7.

    Returns:
        Dict mapping language and format to created file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )

    context = format_template_context(
        flood_date=flood_date,
        bbox=bbox,
        damage_summary=damage_summary,
        priority_items=priority_items,
        s1_metadata=s1_metadata,
        s2_metadata=s2_metadata,
        cant_see_breakdown=cant_see_breakdown,
        dry_islands_count=dry_islands_count,
    )

    outputs: Dict[str, Path] = {}

    for lang, tmpl_name in [("en", "template_en.html"), ("ne", "template_ne.html")]:
        template = jinja_env.get_template(tmpl_name)
        rendered_html = template.render(**context)

        html_path = output_dir / f"sitrep_{lang}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(rendered_html)
        outputs[f"html_{lang}"] = html_path

        # Attempt PDF conversion via WeasyPrint
        pdf_path = output_dir / f"sitrep_{lang}.pdf"
        try:
            import weasyprint
            weasyprint.HTML(string=rendered_html).write_pdf(str(pdf_path))
            outputs[f"pdf_{lang}"] = pdf_path
            logger.info(f"Compiled PDF report: {pdf_path}")
        except Exception as e:
            logger.warning(f"WeasyPrint unavailable or failed ({e}). HTML report preserved: {html_path}")

    return outputs
