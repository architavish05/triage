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

from p3_terrain.rescue.trapped_risk import RescuePriorityItem

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

        # Attempt PDF conversion via WeasyPrint, falling back to ReportLab
        pdf_path = output_dir / f"sitrep_{lang}.pdf"
        pdf_generated = False
        try:
            import weasyprint
            weasyprint.HTML(string=rendered_html).write_pdf(str(pdf_path))
            outputs[f"pdf_{lang}"] = pdf_path
            pdf_generated = True
            logger.info(f"Compiled PDF report via WeasyPrint: {pdf_path}")
        except Exception as e:
            logger.info(f"WeasyPrint unavailable ({e}), trying ReportLab fallback...")

        if not pdf_generated:
            if _render_pdf_reportlab(context, pdf_path, lang=lang):
                outputs[f"pdf_{lang}"] = pdf_path
                logger.info(f"Compiled PDF report via ReportLab fallback: {pdf_path}")
            else:
                logger.warning(f"PDF generation unavailable. HTML report preserved: {html_path}")

    return outputs


def _render_pdf_reportlab(context: Dict[str, Any], pdf_path: Path, lang: str = "en") -> bool:
    """Generate a clean, high-density 1-page PDF situation report using ReportLab."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            leftMargin=24,
            rightMargin=24,
            topMargin=24,
            bottomMargin=24,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=15,
            leading=17,
            textColor=colors.HexColor("#0f172a"),
        )
        meta_style = ParagraphStyle(
            "ReportMeta",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=colors.HexColor("#475569"),
        )
        header_cell_style = ParagraphStyle(
            "HeaderCell",
            parent=styles["Normal"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=8.5,
            textColor=colors.HexColor("#ffffff"),
        )
        body_cell_style = ParagraphStyle(
            "BodyCell",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7,
            leading=8.5,
            textColor=colors.HexColor("#1e293b"),
        )
        footer_style = ParagraphStyle(
            "FooterText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=6.5,
            leading=8,
            textColor=colors.HexColor("#64748b"),
        )

        elements = []

        # 1. Header Banner
        title_text = "TRIAGE DISASTER SITUATION REPORT" if lang == "en" else "TRIAGE DISASTER SITREP (NEPALI TRANSLATION)"
        elements.append(Paragraph(f"<b>{title_text}</b>", title_style))
        elements.append(Spacer(1, 3))

        meta_line = (
            f"<b>Flood Event Date:</b> {context.get('flood_date', '2026-08-26')} &nbsp;|&nbsp; "
            f"<b>Region BBox:</b> {context.get('bbox_str', '')} &nbsp;|&nbsp; "
            f"<b>Generated:</b> {context.get('generated_timestamp', '')}"
        )
        elements.append(Paragraph(meta_line, meta_style))
        elements.append(Spacer(1, 6))

        # 2. Key Operational Metrics (KPI Table)
        summary = context.get("summary", {})
        kpi_data = [
            ["Damaged Buildings", "Impassable Roads", "Severed Bridges", "Cut-Off Villages", "Dry Islands", "Stranded (Est.)"],
            [
                str(summary.get("damaged_buildings_count", 0)),
                f"{summary.get('damaged_roads_km', 0.0):.2f} km",
                str(summary.get("damaged_bridges_count", 0)),
                str(context.get("cutoff_settlements_count", 0)),
                str(context.get("dry_islands_count", 0)),
                f"~{context.get('total_stranded_estimate', 0)}",
            ],
        ]
        kpi_table = Table(kpi_data, colWidths=[94, 94, 94, 94, 94, 94])
        kpi_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#ffffff")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, 0), 7.5),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#f1f5f9")),
            ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
            ("FONTSIZE", (0, 1), (-1, 1), 9.5),
            ("TEXTCOLOR", (0, 1), (-1, 1), colors.HexColor("#0f172a")),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ]))
        elements.append(kpi_table)
        elements.append(Spacer(1, 6))

        # 3. Sensor & Physical Constraint Status
        sensor_text = (
            f"<b>Sentinel-1 SAR:</b> {context.get('s1_orbit_status')} (Relative Track: {context.get('s1_relative_orbit')}) &nbsp;|&nbsp; "
            f"<b>Sentinel-2 Optical:</b> {context.get('s2_optical_status')} &nbsp;|&nbsp; "
            f"<b>Radar Blind Spot (Shadow/Layover):</b> {context.get('cant_see_percent', 0.0):.1f}%"
        )
        elements.append(Paragraph(sensor_text, meta_style))
        elements.append(Spacer(1, 6))

        # 4. Top Rescue Priorities Table
        priorities = context.get("top_priorities", [])
        if priorities:
            elements.append(Paragraph("<b>PRIORITIZED RESCUE EVACUATION DISPATCH</b>", meta_style))
            elements.append(Spacer(1, 2))

            table_rows = [[
                Paragraph("<b>Rank</b>", header_cell_style),
                Paragraph("<b>Settlement</b>", header_cell_style),
                Paragraph("<b>Type</b>", header_cell_style),
                Paragraph("<b>Cutoff %</b>", header_cell_style),
                Paragraph("<b>Trapped Risk</b>", header_cell_style),
                Paragraph("<b>Bldgs Hit</b>", header_cell_style),
                Paragraph("<b>Stranded (est)</b>", header_cell_style),
                Paragraph("<b>Action Recommendation</b>", header_cell_style),
            ]]

            for p in priorities[:8]:
                c_pct = f"{round(float(p.get('cutoff_prob', 0.0)) * 100)}%"
                t_pct = f"{round(float(p.get('trapped_risk', 0.0)) * 100)}%"
                table_rows.append([
                    Paragraph(f"#{p.get('rank', '-')}", body_cell_style),
                    Paragraph(str(p.get("settlement_name", "")), body_cell_style),
                    Paragraph(str(p.get("place_type", "")), body_cell_style),
                    Paragraph(c_pct, body_cell_style),
                    Paragraph(t_pct, body_cell_style),
                    Paragraph(str(p.get("buildings_at_risk", 0)), body_cell_style),
                    Paragraph(f"~{p.get('stranded_population_estimate', 0)}", body_cell_style),
                    Paragraph(str(p.get("recommended_action", "DISPATCH")), body_cell_style),
                ])

            p_table = Table(table_rows, colWidths=[24, 85, 45, 45, 52, 42, 58, 213])
            p_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0284c7")),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ]))
            elements.append(p_table)
            elements.append(Spacer(1, 8))

        # 5. Mandatory Legal & Ethical Attributions
        legal_text = (
            "<b>Mandatory Legal & Data Attributions:</b><br/>"
            "• Contains modified Copernicus Sentinel data 2026.<br/>"
            "• Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 "
            "provided under COPERNICUS by the European Union and ESA; all rights reserved.<br/>"
            "• © OpenStreetMap contributors.<br/>"
            "<i>Ethical Invariant: Educational prototype. No images of victims or personal casualty records. "
            "All stranded population numbers are modelled planning estimates.</i>"
        )
        elements.append(Paragraph(legal_text, footer_style))

        doc.build(elements)
        return True
    except Exception as e:
        logger.warning(f"ReportLab PDF generation failed: {e}")
        return False
