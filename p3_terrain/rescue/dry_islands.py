"""Dry island detection for isolated unflooded assets.

Identifies unflooded land parcels, buildings, and settlement clusters that are completely
surrounded by flood water or debris channels, creating inaccessible topological islands.
"""

from collections import deque
import logging
from typing import Any, Dict, List, Set, Tuple

import numpy as np
from scipy.ndimage import label

from p2_ai.detect.fuse import CLASS_BACKGROUND, CLASS_DEBRIS, CLASS_WATER

logger = logging.getLogger(__name__)


def identify_dry_islands(
    class_map: np.ndarray,
    hand: np.ndarray,
    safe_hand_threshold: float = 25.0,
) -> Tuple[np.ndarray, Dict[int, Dict[str, Any]]]:
    """Identify dry islands: unflooded terrain surrounded by flood/debris.

    A dry island is an unflooded connected component of terrain (class == 0) that:
    1. Does NOT touch the safe regional hinterland / raster boundary where HAND is high.
    2. Is bordered on its perimeter predominantly by flood water or debris.

    Args:
        class_map: 2D classified raster (0=bg, 1=water, 2=debris).
        hand: Height Above Nearest Drainage raster in meters.
        safe_hand_threshold: HAND threshold above which terrain is safe high ground (e.g. 25m).

    Returns:
        Tuple of (dry_island_mask, island_properties_dict):
          dry_island_mask: 2D boolean raster (True where cell is on an isolated dry island).
          island_properties_dict: Metadata for each identified island.
    """
    rows, cols = class_map.shape
    hazard_mask = class_map > CLASS_BACKGROUND
    unflooded_mask = ~hazard_mask

    # Label connected components of unflooded ground (8-connectivity)
    structure = np.ones((3, 3), dtype=int)
    labeled_dry, num_features = label(unflooded_mask, structure=structure)

    # Connected components touching the outer boundary or extending into safe high ground
    # (HAND >= safe_hand_threshold) are considered connected to safe mainland.
    safe_mainland_labels = set()

    # 1. Boundary-touching labels
    boundary_cells = np.concatenate([
        labeled_dry[0, :], labeled_dry[rows - 1, :],
        labeled_dry[:, 0], labeled_dry[:, cols - 1],
    ])
    safe_mainland_labels.update(int(lbl) for lbl in np.unique(boundary_cells) if lbl > 0)

    # 2. Safe high ground touching labels
    high_ground_cells = labeled_dry[hand >= safe_hand_threshold]
    safe_mainland_labels.update(int(lbl) for lbl in np.unique(high_ground_cells) if lbl > 0)

    dry_island_mask = np.zeros((rows, cols), dtype=bool)
    island_props: Dict[int, Dict[str, Any]] = {}

    for lbl in range(1, num_features + 1):
        if lbl not in safe_mainland_labels:
            component_cells = (labeled_dry == lbl)
            size_pixels = int(np.count_nonzero(component_cells))
            if size_pixels >= 4:  # Filter out trivial 1-3 pixel noise
                dry_island_mask[component_cells] = True
                island_props[lbl] = {
                    "label": lbl,
                    "area_pixels": size_pixels,
                    "area_m2": size_pixels * 900.0,  # 30m x 30m pixel
                    "mean_hand_m": float(round(float(np.mean(hand[component_cells])), 1)),
                }

    logger.info(f"Identified {len(island_props)} dry islands cut off by surrounding flood/debris.")
    return dry_island_mask, island_props


def check_settlement_on_dry_island(
    settlement_lon: float,
    settlement_lat: float,
    dry_island_mask: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    search_radius_pixels: int = 3,
) -> bool:
    """Check if a settlement falls within or immediately adjacent to a dry island."""
    west, south, east, north, res_x, res_y = transform
    rows, cols = dry_island_mask.shape

    c = int(np.clip((settlement_lon - west) / max(res_x, 1e-9), 0, cols - 1))
    r = int(np.clip((north - settlement_lat) / max(res_y, 1e-9), 0, rows - 1))

    r_min = max(0, r - search_radius_pixels)
    r_max = min(rows, r + search_radius_pixels + 1)
    c_min = max(0, c - search_radius_pixels)
    c_max = min(cols, c + search_radius_pixels + 1)

    return bool(np.any(dry_island_mask[r_min:r_max, c_min:c_max]))
