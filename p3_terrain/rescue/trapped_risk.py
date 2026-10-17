"""Trapped risk assessment, stranded population estimation, and rescue prioritization.

Computes:
1. trapped_risk per settlement (dry island isolation + escape time failure + local damage).
2. stranded_estimate = buildings_at_risk * household_size_assumption (explicitly labelled 'estimate').
3. Final priority ranking = trapped_risk * cutoff_prob * buildings_at_risk.
"""

from dataclasses import asdict, dataclass
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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


def export_rescue_priority_json(priority_items: List[RescuePriorityItem], output_file: Union[Path, str]) -> Path:
    """Export rescue priority rankings to JSON."""
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [asdict(item) for item in priority_items]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2)
    logger.debug(f"Saved rescue priority JSON to {out_path}")
    return out_path


def export_trapped_csv(
    priority_items: List[RescuePriorityItem],
    output_file: Union[Path, str],
) -> Path:
    """Export rescue prioritization table to trapped.csv matching the contract schema.

    Args:
        priority_items: List of RescuePriorityItem objects.
        output_file: Target CSV file path.

    Returns:
        Path to written CSV file.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "rank",
        "settlement_id",
        "settlement_name",
        "place_type",
        "coordinates",
        "trapped_risk",
        "cutoff_prob",
        "buildings_at_risk",
        "stranded_population_estimate",
        "population_data_type",
        "household_size_assumption",
        "priority_score",
        "is_on_dry_island",
        "escape_possible",
        "nearest_hospital",
        "recommended_action",
        "rank_order",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in priority_items:
            coords_str = f"[{item.coordinates[0]}, {item.coordinates[1]}]"
            writer.writerow({
                "rank": item.rank,
                "settlement_id": item.settlement_id,
                "settlement_name": item.settlement_name,
                "place_type": item.place_type,
                "coordinates": coords_str,
                "trapped_risk": item.trapped_risk,
                "cutoff_prob": item.cutoff_prob,
                "buildings_at_risk": item.buildings_at_risk,
                "stranded_population_estimate": item.stranded_population_estimate,
                "population_data_type": item.population_data_type,
                "household_size_assumption": item.household_size_assumption,
                "priority_score": item.priority_score,
                "is_on_dry_island": item.is_on_dry_island,
                "escape_possible": item.escape_possible,
                "nearest_hospital": item.nearest_hospital,
                "recommended_action": item.recommended_action,
                "rank_order": getattr(item, "rank_order", item.rank),
            })

    logger.info(f"Saved {len(priority_items)} rescue priority items to {out_path}")
    return out_path


export_rescue_priorities_csv = export_trapped_csv


def main(run_dir_str: str = "data/mock_run") -> None:
    """CLI runner to execute dry island extraction, HLZ detection, and rescue prioritization."""
    from PIL import Image
    try:
        from p3_terrain.rescue.dry_islands import identify_dry_islands
        from p3_terrain.rescue.escape_time import compute_high_ground_mask, evaluate_settlement_escape_time
        from p3_terrain.rescue.landing_zones import detect_landing_zones, export_landing_zones_geojson
    except ImportError:
        from src.rescue.dry_islands import identify_dry_islands
        from src.rescue.escape_time import compute_high_ground_mask, evaluate_settlement_escape_time
        from src.rescue.landing_zones import detect_landing_zones, export_landing_zones_geojson

    run_dir = Path(run_dir_str)
    if not run_dir.exists():
        raise FileNotFoundError(f"Directory not found: {run_dir}")

    # Load rasters
    cls = np.array(Image.open(run_dir / "class.tif")).astype(np.uint8)
    hand = np.array(Image.open(run_dir / "hand.tif")).astype(np.float32)
    slope = np.array(Image.open(run_dir / "slope.tif")).astype(np.float32)
    cant_see_path = run_dir / "cant_see.tif"
    cant_see = np.array(Image.open(cant_see_path)) > 0 if cant_see_path.exists() else np.zeros(cls.shape, dtype=bool)

    # Load grid transform
    grid_path = run_dir / "grid.json"
    transform = (85.2, 27.8, 85.45, 28.15, 0.001953125, 0.002734375)
    if grid_path.exists():
        with open(grid_path, "r", encoding="utf-8") as f:
            transform = tuple(json.load(f).get("transform", transform))

    # Load settlements
    settlements_path = run_dir / "settlements.geojson"
    settlements_fc = {"type": "FeatureCollection", "features": []}
    if settlements_path.exists():
        with open(settlements_path, "r", encoding="utf-8") as f:
            settlements_fc = json.load(f)

    # 1. Dry islands
    dry_island_mask, island_props = identify_dry_islands(cls, hand, safe_hand_threshold=25.0)

    # 2. High ground mask & escape checks
    high_ground_mask = compute_high_ground_mask(hand, slope, cls)
    escape_checks = {}
    for feat in settlements_fc.get("features", []):
        s_name = feat.get("properties", {}).get("name", "Settlement")
        s_coords = feat.get("geometry", {}).get("coordinates", [0.0, 0.0])
        check = evaluate_settlement_escape_time(
            settlement_lon=s_coords[0],
            settlement_lat=s_coords[1],
            settlement_name=s_name,
            arrival_time_mins=120.0,
            high_ground_mask=high_ground_mask,
            transform=transform,
        )
        escape_checks[s_name] = check

    # 3. Flat HLZs (< 5 deg, area >= 1800 m2)
    hlz_candidates, hlz_fc = detect_landing_zones(
        slope=slope,
        class_map=cls,
        cant_see_mask=cant_see,
        transform=transform,
        settlements_geojson=settlements_fc,
        max_slope_deg=5.0,
        min_pixels_area=2,
    )

    # 4. Load cutoff records and buildings
    s_by_name = {f.get("properties", {}).get("name"): f for f in settlements_fc.get("features", [])}
    cutoff_path = run_dir / "cutoff.csv"
    isolation_records: List[SettlementIsolation] = []
    if cutoff_path.exists():
        with open(cutoff_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                s_name = r.get("name", "")
                s_feat = s_by_name.get(s_name, {})
                s_id = s_feat.get("id", f"place/{len(isolation_records)+1}")
                s_coords = tuple(s_feat.get("geometry", {}).get("coordinates", [0.0, 0.0]))
                isolation_records.append(SettlementIsolation(
                    settlement_id=s_id,
                    name=s_name,
                    place_type=r.get("place", "village"),
                    coordinates=s_coords,
                    population_est=int(float(r.get("population_est", 1000))),
                    cutoff_prob=float(r.get("cutoff_prob", 0.0)),
                    cutoff_prob_hospital=float(r.get("cutoff_prob_hospital", 0.0)),
                    cutoff_prob_town=float(r.get("cutoff_prob_town", 0.0)),
                    nearest_hospital_name=r.get("nearest_hospital", "Regional Hospital"),
                    pre_event_distance_km=float(r.get("pre_event_distance_km", 10.0)),
                ))

    bldgs_path = run_dir / "geojson" / "damaged_buildings.geojson"
    bldgs_fc = {"type": "FeatureCollection", "features": []}
    if bldgs_path.exists():
        with open(bldgs_path, "r", encoding="utf-8") as f:
            bldgs_fc = json.load(f)

    # 5. Compute rescue priorities
    priority_items = compute_rescue_priorities(
        isolation_records=isolation_records,
        damaged_buildings_geojson=bldgs_fc,
        dry_island_mask=dry_island_mask,
        transform=transform,
        escape_checks=escape_checks,
        household_size=5.0,
    )

    # 6. Export outputs
    trapped_csv_path = run_dir / "trapped.csv"
    rescue_csv_path = run_dir / "rescue_priorities.csv"
    export_trapped_csv(priority_items, trapped_csv_path)
    export_trapped_csv(priority_items, rescue_csv_path)

    geojson_dir = run_dir / "geojson"
    if geojson_dir.exists():
        export_rescue_priority_json(priority_items, geojson_dir / "rescue_priorities.json")
        export_landing_zones_geojson(hlz_fc, geojson_dir / "landing_zones.geojson")
    export_landing_zones_geojson(hlz_fc, run_dir / "landing_zones.geojson")

    print(f"[Rescue] Successfully processed {run_dir}:")
    print(f"  - Dry islands detected: {len(island_props)}")
    print(f"  - Helicopter landing zones (< 5 deg): {len(hlz_candidates)}")
    print(f"  - Settlements ranked: {len(priority_items)}")
    for item in priority_items:
        print(f"    Rank {item.rank}: {item.settlement_name} ({item.place_type}) - "
              f"Priority Score: {item.priority_score:.2f}, Cutoff: {item.cutoff_prob*100:.1f}%, "
              f"Trapped Risk: {item.trapped_risk:.2f}, Stranded Est: {item.stranded_population_estimate} (estimate)")
    print(f"  - Output trapped CSV: {trapped_csv_path}")
    print(f"  - Output HLZ GeoJSON: {run_dir / 'landing_zones.geojson'}")


if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "data/mock_run"
    main(target)

