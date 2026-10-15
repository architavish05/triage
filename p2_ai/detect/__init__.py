"""Detection modules: SAR change, InSAR coherence / GRD proxy, U-Net AI, and 4-way fusion.

Person 2 (AI & Detection) — Branch: p2-ai
Owned directories: p2_ai/detect/, p2_ai/notebooks/
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np

from p2_ai.detect.coherence import (
    DebrisDetectionResult,
    compute_coherence_proxy,
    compute_complex_coherence,
    compute_local_std,
    detect_debris_amplitude_fallback,
    detect_debris_from_coherence,
)
from p2_ai.detect.fuse import (
    CLASS_BACKGROUND,
    CLASS_DEBRIS,
    CLASS_WATER,
    FusedDetectionResult,
    fuse_detections,
)
from p2_ai.detect.sar_change import (
    SARWaterDetectionResult,
    detect_water_sar,
    detect_water_sar_dual_pol,
    lee_filter,
    otsu_threshold,
    preprocess_sar,
    to_db,
)
from p2_ai.detect.unet import (
    UNetInferenceEngine,
    UNetPredictionResult,
    compute_segmentation_metrics,
    export_to_onnx,
    normalize_5channel_input,
    predict_unet,
    save_trained_model_stub,
)


def run_detection(
    s1_data: Any,
    s2_data: Optional[Any] = None,
    dem_data: Optional[Any] = None,
    slope: Optional[np.ndarray] = None,
    unet_threshold: float = 0.50,
) -> FusedDetectionResult:
    """Contract interface: Run SAR change, debris/coherence, U-Net, and multi-sensor fusion."""
    if slope is None and dem_data is not None:
        from p3_terrain.terrain.hand_slope import compute_slope
        slope = compute_slope(dem_data.elevation, res_meters=30.0)
    elif slope is None:
        slope = np.zeros(s1_data.post_vv.shape, dtype=np.float32)

    sar_water_res = detect_water_sar(s1_data.post_vv, s1_data.pre_vv, already_db=True)
    debris_res = detect_debris_amplitude_fallback(s1_data.post_vv, s1_data.pre_vv)
    unet_res = predict_unet(
        s1_data.pre_vv,
        s1_data.pre_vh,
        s1_data.post_vv,
        s1_data.post_vh,
        slope,
        threshold=unet_threshold,
    )
    optical_ndwi = None
    if s2_data is not None and not getattr(s2_data.metadata, "cloud_skipped", False):
        optical_ndwi = s2_data.ndwi
    return fuse_detections(
        water_result=sar_water_res,
        debris_result=debris_res,
        optical_ndwi=optical_ndwi,
        unet_prob=unet_res.probability_map,
    )


__all__ = [
    # Top-level contract runner
    "run_detection",
    # SAR change detection
    "to_db",
    "lee_filter",
    "otsu_threshold",
    "preprocess_sar",
    "detect_water_sar",
    "detect_water_sar_dual_pol",
    "SARWaterDetectionResult",
    # Coherence / debris detection
    "compute_complex_coherence",
    "compute_coherence_proxy",
    "detect_debris_from_coherence",
    "compute_local_std",
    "detect_debris_amplitude_fallback",
    "DebrisDetectionResult",
    # U-Net
    "predict_unet",
    "UNetPredictionResult",
    "UNetInferenceEngine",
    "normalize_5channel_input",
    "compute_segmentation_metrics",
    "save_trained_model_stub",
    "export_to_onnx",
    # Fusion
    "fuse_detections",
    "FusedDetectionResult",
    "CLASS_BACKGROUND",
    "CLASS_WATER",
    "CLASS_DEBRIS",
]
