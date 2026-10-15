"""CONTRACT.md constants, class definitions, and attribution strings."""

# Class Map Definitions (CONTRACT.md Section 4)
CLASS_BACKGROUND: int = 0  # Background / Unaffected terrain
CLASS_WATER: int = 1       # Standing or flowing flood water
CLASS_DEBRIS: int = 2      # Debris, mudflow, or sediment deposit

# Legal Attributions (CONTRACT.md Section 7)
ATTRIBUTION_SENTINEL: str = "Contains modified Copernicus Sentinel data 2026."
ATTRIBUTION_DEM: str = (
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence "
    "and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; "
    "all rights reserved."
)
ATTRIBUTION_OSM: str = "© OpenStreetMap contributors."
ATTRIBUTION_SHORT: str = "Contains modified Copernicus Sentinel data (2026). © OpenStreetMap contributors."

# Invariants & Constraints (CONTRACT.md Sections 3 & 5)
MANDATORY_MAX_SNAPSHOT_DATE: str = "2026-07-27"
MAX_FLOOD_SLOPE_DEG: float = 10.0
MAX_FLOOD_HAND_M: float = 22.0
DEFAULT_DAMAGE_THRESHOLD: float = 0.35
DEFAULT_HOUSEHOLD_SIZE: float = 5.0

# 4-Way Fusion Weights (CONTRACT.md Section 4)
FUSION_WEIGHT_UNET: float = 0.40
FUSION_WEIGHT_SAR: float = 0.30
FUSION_WEIGHT_COHERENCE: float = 0.15
FUSION_WEIGHT_NDWI: float = 0.15
