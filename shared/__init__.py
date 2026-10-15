"""Shared CONTRACT.md helpers, constants, grid/coordinate utilities, and IO functions."""

from shared.constants import (
    ATTRIBUTION_DEM,
    ATTRIBUTION_OSM,
    ATTRIBUTION_SENTINEL,
    ATTRIBUTION_SHORT,
    CLASS_BACKGROUND,
    CLASS_DEBRIS,
    CLASS_WATER,
    DEFAULT_DAMAGE_THRESHOLD,
    DEFAULT_HOUSEHOLD_SIZE,
    FUSION_WEIGHT_COHERENCE,
    FUSION_WEIGHT_NDWI,
    FUSION_WEIGHT_SAR,
    FUSION_WEIGHT_UNET,
    MANDATORY_MAX_SNAPSHOT_DATE,
    MAX_FLOOD_HAND_M,
    MAX_FLOOD_SLOPE_DEG,
)
from shared.grid import (
    coord_to_pixel,
    load_grid_spec,
    pixel_to_coord,
    save_grid_spec,
)
from shared.io import (
    load_geojson,
    load_hazard_grid,
    load_metadata,
    save_geojson,
    save_hazard_grid,
    save_metadata,
)

__all__ = [
    # Constants
    "CLASS_BACKGROUND",
    "CLASS_WATER",
    "CLASS_DEBRIS",
    "ATTRIBUTION_SENTINEL",
    "ATTRIBUTION_DEM",
    "ATTRIBUTION_OSM",
    "ATTRIBUTION_SHORT",
    "MANDATORY_MAX_SNAPSHOT_DATE",
    "MAX_FLOOD_SLOPE_DEG",
    "MAX_FLOOD_HAND_M",
    "DEFAULT_DAMAGE_THRESHOLD",
    "DEFAULT_HOUSEHOLD_SIZE",
    "FUSION_WEIGHT_UNET",
    "FUSION_WEIGHT_SAR",
    "FUSION_WEIGHT_COHERENCE",
    "FUSION_WEIGHT_NDWI",
    # Grid
    "coord_to_pixel",
    "pixel_to_coord",
    "load_grid_spec",
    "save_grid_spec",
    # IO
    "save_hazard_grid",
    "load_hazard_grid",
    "save_metadata",
    "load_metadata",
    "save_geojson",
    "load_geojson",
]
