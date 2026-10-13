"""Detection modules: SAR change, InSAR coherence loss, U-Net AI, and 4-way multi-sensor fusion."""

from src.detect.coherence import (
    DebrisDetectionResult,
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
    lee_filter,
    otsu_threshold,
    preprocess_sar,
    to_db,
)
from src.detect.unet import (
    UNetInferenceEngine,
    UNetPredictionResult,
    compute_segmentation_metrics,
    normalize_5channel_input,
    predict_unet,
    save_trained_model_stub,
)

__all__ = [
    "to_db",
    "lee_filter",
    "otsu_threshold",
    "preprocess_sar",
    "detect_water_sar",
    "SARWaterDetectionResult",
    "compute_complex_coherence",
    "detect_debris_from_coherence",
    "compute_local_std",
    "detect_debris_amplitude_fallback",
    "DebrisDetectionResult",
    "predict_unet",
    "UNetPredictionResult",
    "UNetInferenceEngine",
    "normalize_5channel_input",
    "compute_segmentation_metrics",
    "save_trained_model_stub",
    "fuse_detections",
    "FusedDetectionResult",
    "CLASS_BACKGROUND",
    "CLASS_WATER",
    "CLASS_DEBRIS",
]
