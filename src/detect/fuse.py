"""Multi-sensor and multi-detector fusion.

Combines SAR change water detection, InSAR coherence loss / amplitude texture debris detection,
and creates the unified 3-class hazard map (Background=0, Flood=1, Debris=2) with per-pixel confidence [0, 1].
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.detect.coherence import DebrisDetectionResult
from src.detect.sar_change import SARWaterDetectionResult

logger = logging.getLogger(__name__)

CLASS_BACKGROUND = 0
CLASS_WATER = 1
CLASS_DEBRIS = 2


@dataclass
class FusedDetectionResult:
    """Consolidated hazard classification and confidence grid."""
    class_map: np.ndarray             # 2D int array: 0=Background, 1=Water, 2=Debris
    confidence: np.ndarray            # 2D float array in [0.0, 1.0]
    water_confidence: np.ndarray      # 2D float array in [0.0, 1.0]
    debris_confidence: np.ndarray     # 2D float array in [0.0, 1.0]
    water_mask: np.ndarray            # Boolean mask where class == 1
    debris_mask: np.ndarray           # Boolean mask where class == 2
    hazard_mask: np.ndarray           # Boolean mask where class > 0
    shape: Tuple[int, int]
    summary_counts: Dict[str, int]


def fuse_detections(
    water_result: SARWaterDetectionResult,
    debris_result: DebrisDetectionResult,
    optical_ndwi: Optional[np.ndarray] = None,
    unet_prob: Optional[np.ndarray] = None,
    ndwi_threshold: float = 0.2,
) -> FusedDetectionResult:
    """Fuse 4 complementary sensors/detectors: U-Net + SAR rule + coherence + S2 NDWI.

    Classification priority:
    1. Water / Flood (Class 1): Fused U-Net probability + SAR rule-based change + optical NDWI confirmation.
    2. Debris (Class 2): InSAR coherence loss or amplitude/texture roughness surge.
    3. Background (Class 0): Unaffected areas.

    Args:
        water_result: Output from SAR change detection (sar_change.py).
        debris_result: Output from coherence/texture debris detection (coherence.py).
        optical_ndwi: Optional NDWI raster from Sentinel-2 (if clear, cloud_skipped=False).
        unet_prob: Optional U-Net flood segmentation probability raster (unet.py).
        ndwi_threshold: Minimum NDWI value indicating open surface water.

    Returns:
        FusedDetectionResult with class_map, per-pixel confidence, and summary stats.
    """
    shape = water_result.water_confidence.shape
    water_conf = water_result.water_confidence.copy()
    debris_conf = debris_result.debris_confidence.copy()

    # Fuse U-Net AI probability with SAR rule-based confidence
    if unet_prob is not None and unet_prob.shape == shape:
        water_conf = (0.50 * water_conf + 0.50 * unet_prob).astype(np.float32)

    # Incorporate optical NDWI if available (boost confidence where optical confirms water)
    if optical_ndwi is not None and optical_ndwi.shape == shape:
        optical_water = optical_ndwi >= ndwi_threshold
        # Boost water confidence where optical confirms
        water_conf = np.where(optical_water, np.clip(water_conf + 0.20, 0.0, 1.0), water_conf)
        # Suppress debris in unmistakable open water
        debris_conf = np.where(optical_water, 0.0, debris_conf)

    # Resolve overlapping claims: where both water and debris are flagged
    class_map = np.zeros(shape, dtype=np.uint8)
    fused_conf = np.zeros(shape, dtype=np.float32)

    # Debris first
    debris_active = debris_conf >= 0.3
    class_map[debris_active] = CLASS_DEBRIS
    fused_conf[debris_active] = debris_conf[debris_active]

    # Water takes precedence when water confidence >= debris confidence or strong water signal
    water_active = water_conf >= 0.3
    water_wins = water_active & ((water_conf >= debris_conf) | (water_result.water_mask))
    class_map[water_wins] = CLASS_WATER
    fused_conf[water_wins] = water_conf[water_wins]

    water_mask = (class_map == CLASS_WATER)
    debris_mask = (class_map == CLASS_DEBRIS)
    hazard_mask = (class_map > CLASS_BACKGROUND)

    counts = {
        "background_pixels": int(np.count_nonzero(class_map == CLASS_BACKGROUND)),
        "water_pixels": int(np.count_nonzero(water_mask)),
        "debris_pixels": int(np.count_nonzero(debris_mask)),
        "total_hazard_pixels": int(np.count_nonzero(hazard_mask)),
        "total_pixels": int(class_map.size),
    }

    logger.info(
        f"Fused hazard detection: {counts['water_pixels']} water pixels, "
        f"{counts['debris_pixels']} debris pixels across {counts['total_pixels']} grid cells."
    )

    return FusedDetectionResult(
        class_map=class_map,
        confidence=fused_conf,
        water_confidence=water_conf,
        debris_confidence=debris_conf,
        water_mask=water_mask,
        debris_mask=debris_mask,
        hazard_mask=hazard_mask,
        shape=shape,
        summary_counts=counts,
    )
