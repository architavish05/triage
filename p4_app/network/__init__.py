"""Network cut-off and topological isolation analysis modules."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from p4_app.network.cutoff import (
    RoadGraph,
    SettlementIsolation,
    build_road_graph,
    run_monte_carlo_cutoff,
)
from p4_app.network.priority import (
    build_cutoff_table,
    export_cutoff_geojson,
    export_cutoff_table_csv,
)


def run_cutoff(
    damaged_roads: Dict[str, Any],
    settlements_geojson: Dict[str, Any],
    hospitals_geojson: Dict[str, Any],
    num_simulations: int = 200,
    random_seed: int = 42,
    output_file: Optional[Path] = None,
) -> Tuple[List[SettlementIsolation], List[Dict[str, Any]]]:
    """Contract interface: Build road graph, simulate edge failure, and generate cutoff table."""
    road_graph = build_road_graph(damaged_roads)
    records = run_monte_carlo_cutoff(
        road_graph=road_graph,
        settlements_geojson=settlements_geojson,
        hospitals_geojson=hospitals_geojson,
        num_simulations=num_simulations,
        random_seed=random_seed,
    )
    table = build_cutoff_table(records)
    if output_file is not None:
        export_cutoff_geojson(records, Path(output_file))
    return records, table


__all__ = [
    "run_cutoff",
    "RoadGraph",
    "SettlementIsolation",
    "build_road_graph",
    "run_monte_carlo_cutoff",
    "build_cutoff_table",
    "export_cutoff_table_csv",
    "export_cutoff_geojson",
]
