"""Escape-time verification: flood arrival vs foot transit time to safe high ground.

Compares predicted flood wave arrival times against evacuation hiking times
required to reach safe high ground (adequate HAND and walkable slope).
"""

from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt

logger = logging.getLogger(__name__)


@dataclass
class EscapeTimeCheck:
    """Evacuation feasibility check for a settlement."""
    settlement_name: str
    arrival_time_mins: float
    distance_to_high_ground_m: float
    required_escape_time_mins: float
    escape_possible: bool
    time_margin_mins: float             # Positive = surplus lead time, Negative = deficit


def compute_high_ground_mask(
    hand: np.ndarray,
    slope: np.ndarray,
    class_map: np.ndarray,
    safe_hand_m: float = 18.0,
    max_walkable_slope_deg: float = 28.0,
) -> np.ndarray:
    """Identify walkable safe high ground refuge terrain.

    High ground must be:
    1. Elevated sufficiently above flood drainage (HAND >= safe_hand_m).
    2. Not impassably steep / cliff (slope <= max_walkable_slope_deg).
    3. Unflooded and free of debris (class_map == 0).
    """
    is_elevated = hand >= safe_hand_m
    is_walkable = slope <= max_walkable_slope_deg
    is_unflooded = class_map == 0
    return is_elevated & is_walkable & is_unflooded


def evaluate_settlement_escape_time(
    settlement_lon: float,
    settlement_lat: float,
    settlement_name: str,
    arrival_time_mins: float,
    high_ground_mask: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    walking_speed_m_per_sec: float = 0.70,  # ~2.5 km/h over rough mountain terrain
    res_meters: float = 30.0,
) -> EscapeTimeCheck:
    """Check whether a settlement has sufficient lead time to reach high ground on foot.

    Args:
        settlement_lon: Longitude of settlement.
        settlement_lat: Latitude of settlement.
        settlement_name: Name of settlement.
        arrival_time_mins: Predicted flood surge arrival time in minutes from Phase 6.
        high_ground_mask: 2D boolean mask of reachable safe high ground refuge.
        transform: Georeferencing transform.
        walking_speed_m_per_sec: Average evacuation walking speed (default 0.7 m/s).
        res_meters: Pixel resolution in meters.

    Returns:
        EscapeTimeCheck detailing distances, escape time, feasibility, and time margin.
    """
    west, south, east, north, res_x, res_y = transform
    rows, cols = high_ground_mask.shape

    c = int(np.clip((settlement_lon - west) / max(res_x, 1e-9), 0, cols - 1))
    r = int(np.clip((north - settlement_lat) / max(res_y, 1e-9), 0, rows - 1))

    # If settlement is ALREADY on safe high ground
    if high_ground_mask[r, c]:
        return EscapeTimeCheck(
            settlement_name=settlement_name,
            arrival_time_mins=arrival_time_mins,
            distance_to_high_ground_m=0.0,
            required_escape_time_mins=0.0,
            escape_possible=True,
            time_margin_mins=arrival_time_mins,
        )

    # Compute Euclidean distance transform to nearest safe high ground cell
    dist_cells = distance_transform_edt(~high_ground_mask)
    dist_m = float(dist_cells[r, c] * res_meters)

    # Required escape time in minutes
    req_escape_mins = float(round((dist_m / max(walking_speed_m_per_sec, 0.1)) / 60.0, 1))

    # Margin = arrival_time - required_escape_time
    margin_mins = float(round(arrival_time_mins - req_escape_mins, 1))
    escape_possible = margin_mins >= 0.0

    return EscapeTimeCheck(
        settlement_name=settlement_name,
        arrival_time_mins=arrival_time_mins,
        distance_to_high_ground_m=round(dist_m, 1),
        required_escape_time_mins=req_escape_mins,
        escape_possible=escape_possible,
        time_margin_mins=margin_mins,
    )
