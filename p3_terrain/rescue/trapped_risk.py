"""Trapped risk assessment, stranded population estimation, and rescue prioritization.

Computes:
1. trapped_risk per settlement (dry island isolation + escape time failure + local damage).
2. stranded_estimate = buildings_at_risk * household_size_assumption (explicitly labelled 'estimate').
3. Final priority ranking = trapped_risk * cutoff_prob * buildings_at_risk.
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from p4_app.network.cutoff import SettlementIsolation
from p3_terrain.rescue.dry_islands import check_settlement_on_dry_island
from p3_terrain.rescue.escape_time import EscapeTimeCheck

logger = logging.getLogger(__name__)

DEFAULT_HOUSEHOLD_SIZE = 5.0  # Regional average rural household size assumption


@dataclass
class RescuePriorityItem:
    """Consolidated rescue prioritization and population risk assessment."""
    rank: int
    settlement_id: str
    settlement_name: str
    place_type: str
    coordinates: Tuple[float, float]
    trapped_risk: float                     # Composite trapped index in [0.0, 1.0]
    cutoff_prob: float                     # Road isolation probability in [0.0, 1.0]
    buildings_at_risk: int                 # Count of buildings flooded or stranded
    stranded_population_estimate: int       # Labelled 'estimate'
    population_data_type: str              # Strictly 'estimate'
    household_size_assumption: float       # Assumption factor (default 5.0)
    priority_score: float                  # trapped_risk * cutoff_prob * buildings_at_risk
    is_on_dry_island: bool
    escape_possible: bool
    nearest_hospital: str
    recommended_action: str


def compute_rescue_priorities(
    isolation_records: List[SettlementIsolation],
    damaged_buildings_geojson: Dict[str, Any],
    dry_island_mask: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    escape_checks: Optional[Dict[str, EscapeTimeCheck]] = None,
    household_size: float = DEFAULT_HOUSEHOLD_SIZE,
) -> List[RescuePriorityItem]:
    """Calculate multi-factor rescue priority ranking across all settlements.

    Priority Formula:
      priority_score = trapped_risk * cutoff_prob * buildings_at_risk

    Where:
      - trapped_risk in [0.0, 1.0]: Combines dry island surrounding, foot escape failure,
        and localized asset destruction.
      - cutoff_prob in [0.0, 1.0]: Monte Carlo road network severing probability.
      - buildings_at_risk: Physical structures compromised or isolated.
      - stranded_population_estimate: Labelled 'estimate' = buildings_at_risk * household_size.

    Args:
        isolation_records: Settlement isolation list from Phase 5.
        damaged_buildings_geojson: Enriched building GeoJSON from Phase 4.
        dry_island_mask: 2D boolean mask from dry island detection.
        transform: Georeferencing transform.
        escape_checks: Optional dictionary mapping settlement name to EscapeTimeCheck.
        household_size: Household multiplier assumption (default 5.0).

    Returns:
        List of RescuePriorityItem objects ordered from highest to lowest rescue priority.
    """
    building_features = damaged_buildings_geojson.get("features", [])

    # Group buildings by settlement
    bldgs_by_settlement: Dict[str, List[Dict[str, Any]]] = {}
    for b_feat in building_features:
        s_name = b_feat.get("properties", {}).get("settlement", "")
        if s_name:
            bldgs_by_settlement.setdefault(s_name, []).append(b_feat)

    items: List[RescuePriorityItem] = []

    for s_rec in isolation_records:
        name = s_rec.name
        lon, lat = s_rec.coordinates

        # 1. Check if settlement is on a dry island surrounded by hazard
        is_dry_island = check_settlement_on_dry_island(lon, lat, dry_island_mask, transform)

        # 2. Check escape feasibility
        escape_info = escape_checks.get(name) if escape_checks else None
        escape_possible = escape_info.escape_possible if escape_info else True

        # 3. Count buildings at risk (damaged OR on dry island)
        associated_bldgs = bldgs_by_settlement.get(name, [])
        if not associated_bldgs:
            # Spatial fallback: count buildings within 500m
            for b_feat in building_features:
                coords = b_feat.get("geometry", {}).get("coordinates", [])
                if coords:
                    if isinstance(coords[0], (int, float)):
                        bx, by = float(coords[0]), float(coords[1])
                    elif isinstance(coords[0], (list, tuple)) and isinstance(coords[0][0], (int, float)):
                        bx, by = float(coords[0][0]), float(coords[0][1])
                    elif isinstance(coords[0], (list, tuple)) and isinstance(coords[0][0], (list, tuple)):
                        bx, by = float(coords[0][0][0]), float(coords[0][0][1])
                    else:
                        continue
                    dx = (bx - lon) * 98000.0
                    dy = (by - lat) * 111000.0
                    if dx ** 2 + dy ** 2 <= 500.0 ** 2:
                        associated_bldgs.append(b_feat)

        damaged_bldgs_count = sum(1 for b in associated_bldgs if b.get("properties", {}).get("is_damaged", False))

        # If on dry island, all buildings in the cluster are at risk
        if is_dry_island:
            buildings_at_risk = max(damaged_bldgs_count, len(associated_bldgs), 5)
        else:
            buildings_at_risk = max(damaged_bldgs_count, 1 if s_rec.cutoff_prob >= 0.8 else 0)

        # 4. Stranded population estimate (strictly labelled 'estimate')
        stranded_est = int(round(buildings_at_risk * household_size))

        # 5. Calculate composite trapped_risk [0.0, 1.0]
        risk_components = []
        if is_dry_island:
            risk_components.append(0.40)  # Isolated by perimeter flood
        if not escape_possible:
            risk_components.append(0.35)  # Flood arrives before high ground reached
        if damaged_bldgs_count > 0:
            risk_components.append(0.25)  # Physical destruction verified

        # Baseline risk from road severance
        base_risk = 0.20 if s_rec.cutoff_prob >= 0.5 else 0.05
        trapped_risk = float(round(min(1.0, sum(risk_components) + base_risk), 2))

        # 6. Final priority score: trapped_risk * cutoff_prob * buildings_at_risk
        priority_score = float(round(trapped_risk * s_rec.cutoff_prob * max(buildings_at_risk, 1), 2))

        # Recommended tactical action
        if priority_score >= 10.0 or (is_dry_island and not escape_possible):
            action = "IMMEDIATE HELICOPTER EXTRACTION / AIRDROP REQUIRED"
        elif priority_score >= 3.0:
            action = "HIGH PRIORITY RESCUE / EVACUATION CORRIDOR RECON"
        elif s_rec.cutoff_prob >= 0.7:
            action = "EMERGENCY SUPPLY RESUPPLY (CUT-OFF COMMUNITY)"
        else:
            action = "MONITORING / STANDBY"

        items.append(RescuePriorityItem(
            rank=0,
            settlement_id=s_rec.settlement_id,
            settlement_name=name,
            place_type=s_rec.place_type,
            coordinates=s_rec.coordinates,
            trapped_risk=trapped_risk,
            cutoff_prob=s_rec.cutoff_prob,
            buildings_at_risk=buildings_at_risk,
            stranded_population_estimate=stranded_est,
            population_data_type="estimate",
            household_size_assumption=household_size,
            priority_score=priority_score,
            is_on_dry_island=is_dry_island,
            escape_possible=escape_possible,
            nearest_hospital=s_rec.nearest_hospital_name,
            recommended_action=action,
        ))

    # Rank settlements descending by priority_score
    items.sort(key=lambda x: (x.priority_score, x.stranded_population_estimate), reverse=True)
    for rank, item in enumerate(items, start=1):
        item.rank_order = rank
        item.rank = rank

    logger.info(f"Prioritized {len(items)} settlements for tactical rescue operations.")
    return items


def export_rescue_priority_json(priority_items: List[RescuePriorityItem], output_file: Path) -> Path:
    """Export rescue priority rankings to JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    serialized = [asdict(item) for item in priority_items]
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)
    logger.debug(f"Saved rescue priority JSON to {output_file}")
    return output_file
