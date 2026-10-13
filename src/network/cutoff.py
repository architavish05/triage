"""OSM road network graph modeling and Monte Carlo cut-off analysis.

Builds a topological road graph from OSM highways and assesses settlement isolation
using Monte Carlo edge failure sampling (N=200) weighted by hazard hit_prob.
Computes cut-off probability to nearest towns and emergency healthcare facilities (hospitals).
"""

from collections import deque
from dataclasses import asdict, dataclass
import heapq
import json
import logging
from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SettlementIsolation:
    """Cut-off analysis metrics for an individual settlement."""
    settlement_id: str
    name: str
    place_type: str
    coordinates: Tuple[float, float]     # (lon, lat)
    population_est: int
    cutoff_prob: float                  # Combined cut-off probability in [0, 1]
    cutoff_prob_hospital: float         # Probability isolated from healthcare in [0, 1]
    cutoff_prob_town: float             # Probability isolated from regional towns in [0, 1]
    nearest_hospital_name: str
    pre_event_distance_km: float        # Baseline road distance to nearest hospital


class RoadGraph:
    """Topological graph representing road network with failure probabilities."""

    def __init__(self) -> None:
        self.nodes: Dict[str, Tuple[float, float]] = {}  # node_id -> (lon, lat)
        self.adj: Dict[str, Dict[str, Dict[str, Any]]] = {}  # u -> {v: {length_km, hit_prob, ...}}
        self.edge_list: List[Tuple[str, str, float, float]] = []  # (u, v, length_km, hit_prob)

    def add_node(self, node_id: str, lon: float, lat: float) -> None:
        """Add node if not present."""
        if node_id not in self.nodes:
            self.nodes[node_id] = (lon, lat)
            self.adj[node_id] = {}

    def add_edge(self, u: str, v: str, length_km: float, hit_prob: float, road_id: str = "") -> None:
        """Add bidirectional road edge with length and hazard hit probability."""
        if u == v:
            return
        self.add_node(u, *self.nodes.get(u, (0.0, 0.0)))
        self.add_node(v, *self.nodes.get(v, (0.0, 0.0)))

        edge_data = {"length_km": length_km, "hit_prob": hit_prob, "road_id": road_id}
        self.adj[u][v] = edge_data
        self.adj[v][u] = edge_data
        self.edge_list.append((u, v, length_km, hit_prob))

    def find_nearest_node(self, lon: float, lat: float) -> Optional[str]:
        """Find the nearest graph node to given coordinates."""
        if not self.nodes:
            return None
        best_node = None
        min_dist_sq = float("inf")
        for node_id, (n_lon, n_lat) in self.nodes.items():
            d_sq = ((lon - n_lon) * 98.0) ** 2 + ((lat - n_lat) * 111.0) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_node = node_id
        return best_node

    def shortest_path_distance(self, source: str, targets: Set[str]) -> Tuple[float, Optional[str]]:
        """Dijkstra's shortest path distance from source to any of the target nodes."""
        if not targets or source not in self.adj:
            return float("inf"), None
        if source in targets:
            return 0.0, source

        distances: Dict[str, float] = {source: 0.0}
        pq = [(0.0, source)]

        while pq:
            d, u = heapq.heappop(pq)
            if u in targets:
                return d, u
            if d > distances.get(u, float("inf")):
                continue

            for v, data in self.adj[u].items():
                new_d = d + data["length_km"]
                if new_d < distances.get(v, float("inf")):
                    distances[v] = new_d
                    heapq.heappush(pq, (new_d, v))

        return float("inf"), None


def _calc_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Approximate distance in kilometers between two coordinates."""
    dx = (lon2 - lon1) * 98.0
    dy = (lat2 - lat1) * 111.0
    return float(np.sqrt(dx ** 2 + dy ** 2))


def build_road_graph(damaged_roads_geojson: Dict[str, Any]) -> RoadGraph:
    """Construct RoadGraph from damaged roads GeoJSON FeatureCollection."""
    graph = RoadGraph()
    features = damaged_roads_geojson.get("features", [])

    for feat in features:
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        coords = geom.get("coordinates", [])
        if geom.get("type") != "LineString" or len(coords) < 2:
            continue

        hit_prob = float(props.get("hit_prob", 0.0))
        road_id = str(feat.get("id", ""))

        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            u = f"{round(p1[0], 5)}_{round(p1[1], 5)}"
            v = f"{round(p2[0], 5)}_{round(p2[1], 5)}"
            graph.add_node(u, p1[0], p1[1])
            graph.add_node(v, p2[0], p2[1])

            seg_len = max(0.01, _calc_distance_km(p1[0], p1[1], p2[0], p2[1]))
            graph.add_edge(u, v, length_km=seg_len, hit_prob=hit_prob, road_id=road_id)

    logger.info(f"Built road graph with {len(graph.nodes)} nodes and {len(graph.edge_list)} edges.")
    return graph


def run_monte_carlo_cutoff(
    road_graph: RoadGraph,
    settlements_geojson: Dict[str, Any],
    hospitals_geojson: Dict[str, Any],
    num_simulations: int = 200,
    random_seed: int = 42,
) -> List[SettlementIsolation]:
    """Execute Monte Carlo edge removal (N=200) to compute settlement cut-off probability.

    In each realization, road segments fail independently with probability hit_prob.
    A settlement is marked cut-off if all paths to regional towns or healthcare
    facilities are broken.

    Args:
        road_graph: RoadGraph containing road topology and hit probabilities.
        settlements_geojson: OSM places GeoJSON FeatureCollection.
        hospitals_geojson: OSM hospitals GeoJSON FeatureCollection.
        num_simulations: Monte Carlo iterations (default N=200).
        random_seed: Seed for repeatable simulations.

    Returns:
        List of SettlementIsolation records sorted by cutoff_prob descending.
    """
    rng = random.Random(random_seed)
    settlements_features = settlements_geojson.get("features", [])
    hospitals_features = hospitals_geojson.get("features", [])

    # Map hospitals to graph nodes
    hospital_nodes: Set[str] = set()
    hospital_names: Dict[str, str] = {}
    for h_feat in hospitals_features:
        coords = h_feat.get("geometry", {}).get("coordinates", [])
        name = h_feat.get("properties", {}).get("name", "Hospital/Health Post")
        if coords:
            n = road_graph.find_nearest_node(coords[0], coords[1])
            if n:
                hospital_nodes.add(n)
                hospital_names[n] = name

    # Map towns to graph nodes
    town_nodes: Set[str] = set()
    for s_feat in settlements_features:
        place_type = s_feat.get("properties", {}).get("place", "village")
        if place_type in ("town", "city"):
            coords = s_feat.get("geometry", {}).get("coordinates", [])
            if coords:
                n = road_graph.find_nearest_node(coords[0], coords[1])
                if n:
                    town_nodes.add(n)

    # If no towns identified, treat hospitals as destination hubs
    if not town_nodes:
        town_nodes = set(hospital_nodes)

    # Prepare settlement mappings and baseline pre-event distances
    settlement_info: List[Dict[str, Any]] = []
    for s_feat in settlements_features:
        s_id = str(s_feat.get("id", ""))
        props = s_feat.get("properties", {})
        coords = s_feat.get("geometry", {}).get("coordinates", [0.0, 0.0])
        name = props.get("name", f"Settlement {s_id}")
        place_type = props.get("place", "village")
        pop_str = props.get("population", "")
        try:
            pop_est = int(pop_str) if pop_str else (5000 if place_type == "town" else 800)
        except ValueError:
            pop_est = 800

        n = road_graph.find_nearest_node(coords[0], coords[1])
        base_dist, nearest_h = road_graph.shortest_path_distance(n, hospital_nodes) if n else (float("inf"), None)
        h_name = hospital_names.get(nearest_h, "None reachable") if nearest_h else "None reachable"

        settlement_info.append({
            "id": s_id,
            "name": name,
            "place": place_type,
            "coords": (coords[0], coords[1]),
            "pop": pop_est,
            "node": n,
            "base_dist": round(base_dist, 2) if np.isfinite(base_dist) else 999.0,
            "h_name": h_name,
            "disconnected_hospital_count": 0,
            "disconnected_town_count": 0,
        })

    # Run Monte Carlo simulations
    logger.info(f"Running Monte Carlo cut-off analysis (N={num_simulations} simulations)...")
    edges = road_graph.edge_list

    for sim_idx in range(num_simulations):
        # Build active adjacency for this realization
        active_adj: Dict[str, Set[str]] = {node: set() for node in road_graph.nodes}
        for u, v, _, hit_prob in edges:
            # Edge survives if uniform random >= hit_prob
            if rng.random() >= hit_prob:
                active_adj[u].add(v)
                active_adj[v].add(u)

        # Precompute connected components in this realization via BFS
        visited: Set[str] = set()
        components: List[Set[str]] = []
        node_to_comp: Dict[str, int] = {}

        for node in road_graph.nodes:
            if node not in visited:
                comp_idx = len(components)
                comp: Set[str] = set()
                q = deque([node])
                visited.add(node)
                while q:
                    curr = q.popleft()
                    comp.add(curr)
                    node_to_comp[curr] = comp_idx
                    for neighbor in active_adj[curr]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append(neighbor)
                components.append(comp)

        # Check hospital and town reachability for each component
        comp_has_hospital = [bool(comp & hospital_nodes) for comp in components]
        comp_has_town = [bool(comp & town_nodes) for comp in components]

        # Evaluate reachability for each settlement
        for s in settlement_info:
            s_node = s["node"]
            if s_node is None or s_node not in node_to_comp:
                s["disconnected_hospital_count"] += 1
                s["disconnected_town_count"] += 1
            else:
                c_idx = node_to_comp[s_node]
                if not comp_has_hospital[c_idx]:
                    s["disconnected_hospital_count"] += 1
                if not comp_has_town[c_idx]:
                    s["disconnected_town_count"] += 1

    # Compile settlement isolation metrics
    results: List[SettlementIsolation] = []
    for s in settlement_info:
        p_hosp = float(round(s["disconnected_hospital_count"] / num_simulations, 3))
        p_town = float(round(s["disconnected_town_count"] / num_simulations, 3))
        p_comb = float(round(max(p_hosp, p_town), 3))

        results.append(SettlementIsolation(
            settlement_id=s["id"],
            name=s["name"],
            place_type=s["place"],
            coordinates=s["coords"],
            population_est=s["pop"],
            cutoff_prob=p_comb,
            cutoff_prob_hospital=p_hosp,
            cutoff_prob_town=p_town,
            nearest_hospital_name=s["h_name"],
            pre_event_distance_km=s["base_dist"],
        ))

    # Sort descending by cutoff_prob, then by population
    results.sort(key=lambda x: (x.cutoff_prob, x.population_est), reverse=True)
    return results
