"""Settlement isolation ranking and tabular export.

Formats Monte Carlo cut-off results into prioritized decision-support tables,
CSV exports, and GeoJSON layers for disaster responders.
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from p4_app.network.cutoff import SettlementIsolation

logger = logging.getLogger(__name__)


def build_cutoff_table(isolation_records: List[SettlementIsolation]) -> List[Dict[str, Any]]:
    """Format settlement isolation records into a ranked decision table."""
    table = []
    for rank, item in enumerate(isolation_records, start=1):
        table.append({
            "rank": rank,
            "name": item.name,
            "settlement_name": item.name,
            "place": item.place_type,
            "place_type": item.place_type,
            "population_est": item.population_est,
            "cutoff_prob": item.cutoff_prob,
            "cutoff_prob_hospital": item.cutoff_prob_hospital,
            "cutoff_prob_town": item.cutoff_prob_town,
            "nearest_hospital": item.nearest_hospital_name,
            "pre_event_distance_km": item.pre_event_distance_km,
            "longitude": item.coordinates[0],
            "latitude": item.coordinates[1],
            "settlement_id": item.settlement_id,
        })
    return table


def export_cutoff_table_csv(ranked_table: List[Dict[str, Any]], output_path: Path) -> Path:
    """Save ranked settlement cut-off table to CSV with contract and mock-run compatible schema."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    std_columns = [
        "rank", "name", "place", "population_est",
        "cutoff_prob", "cutoff_prob_hospital", "cutoff_prob_town",
        "nearest_hospital", "pre_event_distance_km",
    ]
    if not ranked_table:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            f.write(",".join(std_columns) + "\n")
        return output_path

    first = ranked_table[0]
    if all(k in first for k in std_columns):
        fieldnames = std_columns
    else:
        fieldnames = list(first.keys())

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ranked_table)

    logger.debug(f"Exported cut-off table CSV to {output_path}")
    return output_path


def export_cutoff_geojson(isolation_records: List[SettlementIsolation], output_path: Optional[Path] = None) -> Dict[str, Any]:
    """Generate GeoJSON FeatureCollection of settlements with isolation metrics."""
    features = []
    for rank, item in enumerate(isolation_records, start=1):
        feat = {
            "type": "Feature",
            "id": item.settlement_id,
            "geometry": {
                "type": "Point",
                "coordinates": [item.coordinates[0], item.coordinates[1]],
            },
            "properties": {
                "rank": rank,
                "name": item.name,
                "place": item.place_type,
                "population_est": item.population_est,
                "cutoff_prob": item.cutoff_prob,
                "cutoff_prob_hospital": item.cutoff_prob_hospital,
                "cutoff_prob_town": item.cutoff_prob_town,
                "nearest_hospital": item.nearest_hospital_name,
                "pre_event_distance_km": item.pre_event_distance_km,
            },
        }
        features.append(feat)

    geojson = {"type": "FeatureCollection", "features": features}

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        # Also write cutoff.csv in parent run directory and same folder for contract compliance
        table = build_cutoff_table(isolation_records)
        export_cutoff_table_csv(table, p.parent / "cutoff.csv")
        if p.parent.name == "geojson":
            export_cutoff_table_csv(table, p.parent.parent / "cutoff.csv")

        logger.debug(f"Exported cut-off GeoJSON to {output_path}")

    return geojson
