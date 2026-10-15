"""Plain-text top-10 summary for SMS / satellite messenger / VHF radio transmission.

Outputs compact, high-density situational briefings for bandwidth-constrained responders.
Includes mandatory Copernicus and OSM attribution tag on footer.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from p3_terrain.rescue.landing_zones import LandingZoneCandidate
from p3_terrain.rescue.trapped_risk import RescuePriorityItem

ATTRIBUTION_SHORT = (
    "Contains modified Copernicus Sentinel data 2026. "
    "Copernicus WorldDEM-30 © DLR/Airbus/ESA. © OpenStreetMap contributors."
)


def generate_sms_summary(
    flood_date: str,
    bbox: Tuple[float, float, float, float],
    damage_summary: Dict[str, Any],
    priority_items: List[RescuePriorityItem],
    landing_zones: Optional[List[LandingZoneCandidate]] = None,
    output_path: Optional[Path] = None,
) -> str:
    """Generate compact plain-text SITREP for SMS or VHF radio broadcast.

    Args:
        flood_date: ISO date of event.
        bbox: Bounding box tuple.
        damage_summary: Quantitative damage summary dict.
        priority_items: Rescue priority ranking list.
        landing_zones: Optional list of identified HLZs.
        output_path: Optional path to save text file.

    Returns:
        Formatted multi-line text string.
    """
    lines = []
    lines.append(f"=== [TRIAGE DISASTER SITREP: {flood_date}] ===")
    lines.append(f"ROI BBOX: [{bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f}, {bbox[3]:.2f}]")

    # High-level tallies
    b_hit = damage_summary.get("damaged_buildings_count", 0)
    r_km = damage_summary.get("damaged_roads_km", 0.0)
    br_hit = damage_summary.get("damaged_bridges_count", 0)
    cutoff_count = sum(1 for p in priority_items if p.cutoff_prob >= 0.5)
    total_est = sum(p.stranded_population_estimate for p in priority_items)

    lines.append(
        f"HAZARD TOTALS: {b_hit} bldgs damaged | {r_km} km roads cut | "
        f"{br_hit} bridges broken | {cutoff_count} cut-off communities | "
        f"~{total_est} stranded (ESTIMATE)"
    )
    lines.append("")
    lines.append("--- TOP 10 RESCUE PRIORITIES (RANKED BY TRAPPED RISK & ROAD SEVERANCE) ---")

    for item in priority_items[:10]:
        c_pct = int(round(item.cutoff_prob * 100))
        t_pct = int(round(item.trapped_risk * 100))
        lines.append(
            f"{item.rank}. {item.settlement_name} [{item.coordinates[0]:.3f}, {item.coordinates[1]:.3f}] "
            f"| Cutoff: {c_pct}% | TrappedRisk: {t_pct}% | Bldgs: {item.buildings_at_risk} "
            f"| Stranded: ~{item.stranded_population_estimate} (est) | {item.recommended_action}"
        )

    if landing_zones:
        lines.append("")
        lines.append(f"--- HELICOPTER LANDING ZONES ({len(landing_zones)} SAFE SITES IDENTIFIED) ---")
        for hlz in landing_zones[:5]:
            lines.append(
                f"- {hlz.hlz_id}: [{hlz.coordinates[0]:.4f}, {hlz.coordinates[1]:.4f}] "
                f"near {hlz.nearest_settlement} ({hlz.distance_to_settlement_m:.0f}m away, slope {hlz.slope_deg} deg)"
            )

    lines.append("")
    lines.append(f"LEGAL ATTR: {ATTRIBUTION_SHORT}")
    lines.append("DISCLAIMER: Educational prototype. No images of victims. All pop counts are estimates.")

    text_content = "\n".join(lines)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_content)

    return text_content
