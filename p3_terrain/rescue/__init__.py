"""Rescue intelligence and prioritization modules."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from p3_terrain.rescue.dry_islands import (
    check_settlement_on_dry_island,
    identify_dry_islands,
)
from p3_terrain.rescue.escape_time import (
    EscapeTimeCheck,
    compute_high_ground_mask,
    evaluate_settlement_escape_time,
)
from p3_terrain.rescue.landing_zones import (
    LandingZoneCandidate,
    detect_landing_zones,
    export_landing_zones_geojson,
)
from p3_terrain.rescue.trapped_risk import (
    DEFAULT_HOUSEHOLD_SIZE,
    RescuePriorityItem,
    compute_rescue_priorities,
    export_rescue_priorities_csv,
    export_rescue_priority_json,
    export_trapped_csv,
)


def run_rescue(
    isolation_records: List[Any],
    damaged_buildings_geojson: Dict[str, Any],
    dry_island_mask: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    escape_checks: Optional[Dict[str, Any]] = None,
    household_size: float = DEFAULT_HOUSEHOLD_SIZE,
    output_file: Optional[Path] = None,
) -> List[RescuePriorityItem]:
    """Contract interface: Compute multi-factor rescue priority ranking and optionally export JSON."""
    priorities = compute_rescue_priorities(
        isolation_records=isolation_records,
        damaged_buildings_geojson=damaged_buildings_geojson,
        dry_island_mask=dry_island_mask,
        transform=transform,
        escape_checks=escape_checks,
        household_size=household_size,
    )
    if output_file is not None:
        export_rescue_priority_json(priorities, Path(output_file))
    return priorities


__all__ = [
    "run_rescue",
    "identify_dry_islands",
    "check_settlement_on_dry_island",
    "compute_high_ground_mask",
    "evaluate_settlement_escape_time",
    "EscapeTimeCheck",
    "detect_landing_zones",
    "export_landing_zones_geojson",
    "LandingZoneCandidate",
    "compute_rescue_priorities",
    "export_rescue_priority_json",
    "export_trapped_csv",
    "export_rescue_priorities_csv",
    "RescuePriorityItem",
    "DEFAULT_HOUSEHOLD_SIZE",
]
