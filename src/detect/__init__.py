"""Detection modules: SAR change, InSAR coherence / GRD proxy, U-Net AI, and 4-way fusion.

Person 2 (AI & Detection) — Branch: p2-ai
Owned directories: src/detect/, notebooks/
"""

from src.detect.coherence import (
    DebrisDetectionResult,
    compute_coherence_proxy,
    compute_complex_coherence,
    compute_local_std,
    detect_debris_amplitude_fallback,
    detect_debris_from_coherence,
)
from src.detect.fuse import (
    CLASS_BACKGROUND,
    CLASS_DEBRIS,
    CLASS_WATER,
    FusedDetectionResult,
    fuse_detections,
)
from src.detect.sar_change import (
    SARWaterDetectionResult,
    detect_water_sar,
    detect_water_sar_dual_pol,
    lee_filter,
    otsu_threshold,
    preprocess_sar,
    to_db,
)
from src.detect.unet import (
    UNetInferenceEngine,
    UNetPredictionResult,
    compute_segmentation_metrics,
    export_to_onnx,
    normalize_5channel_input,
    predict_unet,
    save_trained_model_stub,
)

__all__ = [
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
