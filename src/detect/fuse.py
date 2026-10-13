"""Multi-sensor probability fusion for flood and debris mapping.

Implements the four-way fusion formula specified in CONTRACT.md:

    P_fused = 0.40 * P_unet + 0.30 * P_sar + 0.15 * P_coherence + 0.15 * P_ndwi

where each P_* term is an independent flood-evidence probability in [0, 1].

NDWI mapping:
    Raw NDWI (green - nir) / (green + nir) lies in [-1, 1].
    We apply a sigmoid-shaped mapping centred at 0.0 so that:
        NDWI = -1.0  →  P_ndwi ≈ 0.00  (very dry)
        NDWI =  0.0  →  P_ndwi ≈ 0.50  (neutral)
        NDWI = +0.5  →  P_ndwi ≈ 0.88  (likely water)
        NDWI = +1.0  →  P_ndwi ≈ 1.00  (definite water)

Absent-sensor policy:
    When a sensor channel is unavailable (None), its weight is redistributed
    proportionally across the remaining channels to preserve probability scale.
    The redistribution is documented in the return value.

Three-class output:
    0 = Background / Unaffected
    1 = Standing or flowing flood water
    2 = Debris, mudflow, or sediment deposit

Debris evidence:
    Debris is detected via the SAR amplitude surge (positive dB change) signal
    from sar_change.py, or via coherence loss when SLC data is available.
    The U-Net outputs binary flood probability only; it does NOT independently
    predict debris. Debris is therefore determined by the auxiliary SAR/coherence
    evidence, not by the fused flood confidence.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from src.detect.coherence import DebrisDetectionResult
from src.detect.sar_change import SARWaterDetectionResult

logger = logging.getLogger(__name__)

# ── Contract class IDs ────────────────────────────────────────────────────────
CLASS_BACKGROUND: int = 0
CLASS_WATER: int = 1
CLASS_DEBRIS: int = 2

# ── Contract fusion weights ───────────────────────────────────────────────────
_W_UNET: float = 0.40
_W_SAR: float = 0.30
_W_COH: float = 0.15
_W_NDWI: float = 0.15

# Minimum fused confidence to assign a positive class label
_WATER_CONF_THRESHOLD: float = 0.30
_DEBRIS_CONF_THRESHOLD: float = 0.30

# Water class wins over debris when both are above threshold
# (water is more operationally urgent and SAR water signal is more specific)
_WATER_PRIORITY: bool = True


@dataclass
class FusedDetectionResult:
    """Consolidated hazard classification and per-pixel confidence grid."""

    class_map: np.ndarray          # 2D uint8:  0=Background, 1=Water, 2=Debris
    confidence: np.ndarray         # 2D float32 in [0.0, 1.0] – fused flood confidence
    water_confidence: np.ndarray   # 2D float32 – P_fused (flood evidence)
    debris_confidence: np.ndarray  # 2D float32 – debris evidence from SAR/coherence
    water_mask: np.ndarray         # bool mask where class_map == 1
    debris_mask: np.ndarray        # bool mask where class_map == 2
    hazard_mask: np.ndarray        # bool mask where class_map > 0
    shape: Tuple[int, int]
    summary_counts: Dict[str, int]
    fusion_weights_used: Dict[str, float]  # actual weights after absent-sensor redistribution


def _ndwi_to_probability(ndwi: np.ndarray, k: float = 5.0) -> np.ndarray:
    """Map raw NDWI ∈ [-1, 1] to flood probability ∈ [0, 1].

    Uses a sigmoid transform centred at 0.0.  The steepness parameter ``k``
    controls transition sharpness:  k=5 gives a smooth ramp over ≈ ±0.4 NDWI.

    Args:
        ndwi: 2-D float32 array of normalised-difference water index values.
        k: Sigmoid slope parameter (higher = steeper transition).

    Returns:
        Float32 probability array in [0.0, 1.0].
    """
    prob = 1.0 / (1.0 + np.exp(-k * ndwi))
    return np.clip(prob, 0.0, 1.0).astype(np.float32)


def _validate_channel(
    arr: Optional[np.ndarray],
    name: str,
    expected_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Validate and coerce a fusion channel to float32 in [0, 1].

    Returns None if the channel is absent or shape-mismatched (with a warning).
    """
    if arr is None:
        return None

    if arr.shape != expected_shape:
        logger.warning(
            "%s shape %s mismatches expected %s — channel excluded from fusion.",
            name,
            arr.shape,
            expected_shape,
        )
        return None

    arr = np.asarray(arr, dtype=np.float32)

    # Replace NaN/Inf with 0.0 (treat as no-evidence)
    arr = np.where(np.isfinite(arr), arr, 0.0).astype(np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def fuse_detections(
    water_result: SARWaterDetectionResult,
    debris_result: DebrisDetectionResult,
    optical_ndwi: Optional[np.ndarray] = None,
    unet_prob: Optional[np.ndarray] = None,
    ndwi_sigmoid_k: float = 5.0,
) -> FusedDetectionResult:
    """Fuse four complementary flood-evidence channels into a hazard class map.

    Implements the CONTRACT.md formula exactly:

        P_fused = 0.40 * P_unet + 0.30 * P_sar + 0.15 * P_coherence + 0.15 * P_ndwi

    When a channel is unavailable its weight is redistributed proportionally
    among the remaining channels, preserving the relative weighting rationale.

    Args:
        water_result: SAR change-detection result (provides P_sar).
        debris_result: Coherence / amplitude-texture debris result (provides
            P_coherence as debris-evidence probability).
        optical_ndwi: Optional 2-D array of raw Sentinel-2 NDWI values in
            [-1, 1].  Mapped internally to [0, 1] via sigmoid transform.
            Pass None when Sentinel-2 is cloud-covered (cloud_skipped=True).
        unet_prob: Optional 2-D float32 U-Net flood probability in [0, 1].
            Pass None when no trained model weights are available.
        ndwi_sigmoid_k: Steepness of the NDWI-to-probability sigmoid (default 5).

    Returns:
        FusedDetectionResult containing class_map (0/1/2), per-pixel
        confidence, per-class masks, summary pixel counts, and the
        actual fusion weights used after any absent-sensor redistribution.

    Notes:
        * The U-Net produces binary flood probability only.  It does NOT
          independently predict debris.  Debris classification is driven by
          the SAR amplitude-surge / coherence-loss evidence in debris_result.
        * NDWI is NOT added raw to the probability sum.  It is first mapped
          through a documented sigmoid transform.
        * Missing channels are treated as absent evidence (zero contribution),
          not as zero flood probability.
    """
    shape: Tuple[int, int] = water_result.water_confidence.shape

    # ── 1. Gather and validate all four evidence channels ─────────────────────
    p_sar = _validate_channel(water_result.water_confidence, "P_sar", shape)
    # Coherence / amplitude proxy is debris evidence; re-interpret as "disturbance
    # probability" which informs the coherence term in the flood-evidence formula.
    # Justification: strong surface disturbance (coherence loss or texture surge)
    # that is co-located with SAR backscatter decrease indicates flood + debris,
    # so the coherence term contributes to the fused hazard confidence.
    p_coh = _validate_channel(debris_result.debris_confidence, "P_coherence", shape)
    p_unet = _validate_channel(unet_prob, "P_unet", shape)

    # Map NDWI to probability before fusion — never add raw NDWI to probability maps
    if optical_ndwi is not None:
        ndwi_validated = _validate_channel(optical_ndwi, "NDWI", shape)
        p_ndwi = _ndwi_to_probability(ndwi_validated, k=ndwi_sigmoid_k) if ndwi_validated is not None else None
    else:
        p_ndwi = None

    # ── 2. Compute effective weights after absent-sensor redistribution ────────
    available: Dict[str, Tuple[Optional[np.ndarray], float]] = {
        "unet":      (p_unet, _W_UNET),
        "sar":       (p_sar,  _W_SAR),
        "coherence": (p_coh,  _W_COH),
        "ndwi":      (p_ndwi, _W_NDWI),
    }

    present_weight_total: float = sum(
        w for (arr, w) in available.values() if arr is not None
    )
    absent_weight_total: float = sum(
        w for (arr, w) in available.values() if arr is None
    )

    if present_weight_total < 1e-6:
        # Degenerate case: no evidence at all → all background
        logger.error("No valid fusion channels available. Returning all-background result.")
        return FusedDetectionResult(
            class_map=np.zeros(shape, dtype=np.uint8),
            confidence=np.zeros(shape, dtype=np.float32),
            water_confidence=np.zeros(shape, dtype=np.float32),
            debris_confidence=(debris_result.debris_confidence
                               if debris_result.debris_confidence.shape == shape
                               else np.zeros(shape, dtype=np.float32)),
            water_mask=np.zeros(shape, dtype=bool),
            debris_mask=np.zeros(shape, dtype=bool),
            hazard_mask=np.zeros(shape, dtype=bool),
            shape=shape,
            summary_counts={
                "background_pixels": int(np.prod(shape)),
                "water_pixels": 0,
                "debris_pixels": 0,
                "total_hazard_pixels": 0,
                "total_pixels": int(np.prod(shape)),
            },
            fusion_weights_used={k: 0.0 for k in available},
        )

    # Scale up present-channel weights proportionally so they sum to 1.0
    scale: float = 1.0 / present_weight_total if absent_weight_total > 0 else 1.0
    effective_weights: Dict[str, float] = {
        name: (w * scale if arr is not None else 0.0)
        for name, (arr, w) in available.items()
    }

    if absent_weight_total > 1e-6:
        absent_names = [n for n, (arr, _) in available.items() if arr is None]
        logger.info(
            "Absent fusion channels %s (weight %.2f redistributed proportionally).",
            absent_names,
            absent_weight_total,
        )
        logger.info("Effective fusion weights: %s", effective_weights)

    # ── 3. Compute P_fused (flood evidence) ───────────────────────────────────
    p_fused = np.zeros(shape, dtype=np.float64)
    for name, (arr, _) in available.items():
        if arr is not None:
            p_fused += effective_weights[name] * arr.astype(np.float64)

    water_conf = np.clip(p_fused, 0.0, 1.0).astype(np.float32)

    # ── 4. Debris confidence (kept separate from flood evidence) ──────────────
    # Debris confidence = amplitude-surge or coherence-loss evidence.
    # The coherence module already produces this in [0, 1].
    # We DO NOT further weight this by P_fused; debris and water are separate claims.
    debris_conf = debris_result.debris_confidence.copy().astype(np.float32)
    # Suppress debris in clear open-water pixels (high positive NDWI → specular return)
    if p_ndwi is not None:
        open_water = p_ndwi >= 0.75
        debris_conf = np.where(open_water, 0.0, debris_conf).astype(np.float32)

    debris_conf = np.clip(debris_conf, 0.0, 1.0)

    # ── 5. Three-class decision ───────────────────────────────────────────────
    class_map = np.zeros(shape, dtype=np.uint8)
    fused_conf = np.zeros(shape, dtype=np.float32)

    # Assign debris first (lower priority — water will overwrite if both active)
    debris_active = debris_conf >= _DEBRIS_CONF_THRESHOLD
    class_map[debris_active] = CLASS_DEBRIS
    fused_conf[debris_active] = debris_conf[debris_active]

    # Assign water: wins over debris when water evidence is strong
    water_active = water_conf >= _WATER_CONF_THRESHOLD
    if _WATER_PRIORITY:
        # Water wins whenever it is active AND either:
        #   a) water confidence ≥ debris confidence, or
        #   b) the SAR hard binary water mask fires (unambiguous specular reflector)
        water_wins = water_active & (
            (water_conf >= debris_conf) | water_result.water_mask
        )
    else:
        water_wins = water_active

    class_map[water_wins] = CLASS_WATER
    fused_conf[water_wins] = water_conf[water_wins]

    # Ensure confidence is valid everywhere
    fused_conf = np.clip(fused_conf, 0.0, 1.0).astype(np.float32)

    water_mask = class_map == CLASS_WATER
    debris_mask = class_map == CLASS_DEBRIS
    hazard_mask = class_map > CLASS_BACKGROUND

    counts: Dict[str, int] = {
        "background_pixels": int(np.count_nonzero(class_map == CLASS_BACKGROUND)),
        "water_pixels": int(np.count_nonzero(water_mask)),
        "debris_pixels": int(np.count_nonzero(debris_mask)),
        "total_hazard_pixels": int(np.count_nonzero(hazard_mask)),
        "total_pixels": int(class_map.size),
    }

    logger.info(
        "Fusion complete — water: %d px, debris: %d px, background: %d px "
        "(total: %d px).  Weights used: unet=%.2f sar=%.2f coh=%.2f ndwi=%.2f.",
        counts["water_pixels"],
        counts["debris_pixels"],
        counts["background_pixels"],
        counts["total_pixels"],
        effective_weights["unet"],
        effective_weights["sar"],
        effective_weights["coherence"],
        effective_weights["ndwi"],
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
        fusion_weights_used=effective_weights,
    )
