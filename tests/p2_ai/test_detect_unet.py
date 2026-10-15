"""Unit tests for Phase 8 U-Net deep learning model and 4-way fusion."""

import numpy as np
import pytest

from p2_ai.detect.coherence import detect_debris_amplitude_fallback
from p2_ai.detect.fuse import CLASS_DEBRIS, CLASS_WATER, fuse_detections
from p2_ai.detect.sar_change import detect_water_sar
from p2_ai.detect.unet import (
    compute_segmentation_metrics,
    normalize_5channel_input,
    predict_unet,
    save_trained_model_stub,
)


def test_normalize_5channel_input():
    """Verify normalization of 5 multi-modal input channels."""
    shape = (32, 32)
    pre_vv = np.full(shape, -11.0, dtype=np.float32)
    pre_vh = np.full(shape, -18.0, dtype=np.float32)
    post_vv = np.full(shape, -22.0, dtype=np.float32)
    post_vh = np.full(shape, -27.0, dtype=np.float32)
    slope = np.full(shape, 4.0, dtype=np.float32)

    tensor = normalize_5channel_input(pre_vv, pre_vh, post_vv, post_vh, slope)
    assert tensor.shape == (5, 32, 32)
    assert tensor.dtype == np.float32
    # Normalized channels should be in a reasonable range around 0
    assert np.all(tensor >= -3.5)
    assert np.all(tensor <= 3.5)


def test_compute_segmentation_metrics():
    """Verify IoU, precision, recall, and F1 calculation against ground truth."""
    gt = np.zeros((20, 20), dtype=bool)
    gt[5:15, 5:15] = True  # 100 positive pixels

    pred = np.zeros((20, 20), dtype=bool)
    pred[5:15, 5:15] = True  # Perfect prediction (100 TP, 0 FP, 0 FN)

    m_perf = compute_segmentation_metrics(pred, gt)
    assert m_perf["iou"] == 1.0
    assert m_perf["precision"] == 1.0
    assert m_perf["recall"] == 1.0
    assert m_perf["f1_score"] == 1.0

    # Partial overlap (50 TP, 50 FP, 50 FN)
    pred_partial = np.zeros((20, 20), dtype=bool)
    pred_partial[5:15, 10:20] = True
    m_part = compute_segmentation_metrics(pred_partial, gt)
    assert m_part["precision"] == 0.5
    assert m_part["recall"] == 0.5
    assert np.isclose(m_part["iou"], 50.0 / 150.0, atol=1e-3)


def test_predict_unet_inference():
    """Verify U-Net forward inference predicts flood probability [0, 1]."""
    shape = (48, 48)
    pre_vv = np.full(shape, -11.0, dtype=np.float32)
    pre_vh = np.full(shape, -17.0, dtype=np.float32)
    post_vv = np.full(shape, -11.0, dtype=np.float32)
    post_vh = np.full(shape, -17.0, dtype=np.float32)
    slope = np.full(shape, 2.0, dtype=np.float32)

    # Inject water corridor in center (rows 20..28)
    post_vv[20:28, :] = -23.0
    post_vh[20:28, :] = -28.0

    pred_res = predict_unet(pre_vv, pre_vh, post_vv, post_vh, slope, threshold=0.50)

    assert pred_res.probability_map.shape == shape
    assert np.all(pred_res.probability_map >= 0.0)
    assert np.all(pred_res.probability_map <= 1.0)
    # Flooded corridor probability must be significantly higher than dry background
    assert np.mean(pred_res.probability_map[22:26, :]) > np.mean(pred_res.probability_map[0:5, :])
    assert pred_res.inference_backend in ("numpy_vectorized", "onnxruntime", "pytorch")


def test_four_way_fusion_with_unet():
    """Verify 4-way fusion combining U-Net + SAR rule + Coherence + S2 NDWI."""
    shape = (40, 40)
    pre_sar = np.full(shape, -12.0, dtype=np.float32)
    post_sar = np.full(shape, -12.0, dtype=np.float32)
    post_sar[10:25, 10:25] = -22.0  # SAR water signal

    water_res = detect_water_sar(post_sar, pre_sar, already_db=True, post_threshold=-16.0, diff_threshold=-4.0)
    debris_res = detect_debris_amplitude_fallback(post_sar, pre_sar)

    # U-Net probability
    unet_prob = np.zeros(shape, dtype=np.float32)
    unet_prob[10:25, 10:25] = 0.92

    # S2 NDWI (clear optical water confirmation)
    s2_ndwi = np.full(shape, -0.4, dtype=np.float32)
    s2_ndwi[10:25, 10:25] = 0.65

    fused = fuse_detections(
        water_result=water_res,
        debris_result=debris_res,
        optical_ndwi=s2_ndwi,
        unet_prob=unet_prob,
    )

    assert fused.shape == shape
    assert fused.class_map[18, 18] == CLASS_WATER
    # In center water cell, fused confidence should be very high (> 0.85)
    assert fused.confidence[18, 18] > 0.85
    assert np.all(fused.confidence >= 0.0)
    assert np.all(fused.confidence <= 1.0)


def test_save_model_stub(tmp_path, monkeypatch):
    """Verify model stub weights and metadata saving."""
    monkeypatch.setattr("p2_ai.detect.unet.DEFAULT_MODEL_DIR", tmp_path / "models")
    w_path, meta_path = save_trained_model_stub()
    assert w_path.exists()
    assert meta_path.exists()
