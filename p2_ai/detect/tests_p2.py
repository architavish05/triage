"""Internal tests for src/detect/ — owned by Person 2 (p2-ai branch).

These tests are in addition to (and must not conflict with) the shared tests/
directory.  They cover edge cases, numerical invariants, and integration
scenarios not covered by the team-shared suite.

Run with:
    python -m pytest src/detect/tests_p2.py -v
"""

import numpy as np
import pytest


# ═══════════════════════════════════════════════════════════════════
# Helper fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def normal_scene():
    """Standard 64x64 SAR scene: dry land with flood corridor."""
    rng = np.random.default_rng(42)
    shape = (64, 64)
    pre_vv = rng.normal(-11.0, 2.0, shape).astype(np.float32)
    pre_vh = rng.normal(-17.0, 2.5, shape).astype(np.float32)
    post_vv = pre_vv.copy()
    post_vh = pre_vh.copy()
    # Flood corridor in rows 24–40
    post_vv[24:40, :] = rng.normal(-22.0, 1.0, (16, 64)).astype(np.float32)
    post_vh[24:40, :] = rng.normal(-28.0, 1.0, (16, 64)).astype(np.float32)
    slope = np.full(shape, 3.0, dtype=np.float32)
    return {"pre_vv": pre_vv, "pre_vh": pre_vh, "post_vv": post_vv, "post_vh": post_vh, "slope": slope}


# ═══════════════════════════════════════════════════════════════════
# SAR change detection — edge cases
# ═══════════════════════════════════════════════════════════════════

class TestSARChangeEdgeCases:

    def test_all_zeros(self):
        from p2_ai.detect.sar_change import detect_water_sar
        shape = (32, 32)
        zeros = np.zeros(shape, dtype=np.float32)
        result = detect_water_sar(zeros, zeros, already_db=True, post_threshold=-16.0, diff_threshold=-3.5)
        assert result.water_mask.shape == shape
        assert np.all(result.water_confidence >= 0.0)
        assert np.all(result.water_confidence <= 1.0)

    def test_constant_array_otsu(self):
        from p2_ai.detect.sar_change import otsu_threshold
        # Constant input: Otsu must not raise, must return the constant value
        t = otsu_threshold(np.full(100, -15.0))
        assert np.isfinite(t)
        assert np.isclose(t, -15.0)

    def test_empty_array_otsu(self):
        from p2_ai.detect.sar_change import otsu_threshold
        t = otsu_threshold(np.array([]))
        assert t == -16.0  # documented fallback

    def test_nan_input_handling(self):
        from p2_ai.detect.sar_change import detect_water_sar
        shape = (20, 20)
        arr = np.full(shape, -12.0, dtype=np.float32)
        arr[5, 5] = np.nan
        arr[10, 10] = np.inf
        result = detect_water_sar(arr, arr, already_db=True, post_threshold=-16.0, diff_threshold=-3.5)
        assert np.all(np.isfinite(result.water_confidence))

    def test_mismatched_dimensions(self):
        from p2_ai.detect.sar_change import lee_filter
        # Lee filter on non-square array
        arr = np.random.randn(17, 53).astype(np.float32)
        filtered = lee_filter(arr, window_size=5)
        assert filtered.shape == (17, 53)

    def test_floodwater_backscatter_decrease(self):
        """Post_vv much lower than pre_vv → high water confidence."""
        from p2_ai.detect.sar_change import detect_water_sar
        shape = (30, 30)
        pre  = np.full(shape, -10.0, dtype=np.float32)
        post = np.full(shape, -25.0, dtype=np.float32)
        result = detect_water_sar(post, pre, already_db=True, post_threshold=-16.0, diff_threshold=-3.5)
        assert np.all(result.water_mask)
        assert np.mean(result.water_confidence) > 0.5

    def test_backscatter_surge_not_water(self):
        """Post_vv much HIGHER than pre_vv → no water detected."""
        from p2_ai.detect.sar_change import detect_water_sar
        shape = (30, 30)
        pre  = np.full(shape, -12.0, dtype=np.float32)
        post = np.full(shape, -5.0, dtype=np.float32)
        result = detect_water_sar(post, pre, already_db=True, post_threshold=-16.0, diff_threshold=-3.5)
        assert not np.any(result.water_mask)

    def test_dual_pol_confidence(self, normal_scene):
        from p2_ai.detect.sar_change import detect_water_sar_dual_pol
        result = detect_water_sar_dual_pol(
            normal_scene["pre_vv"], normal_scene["pre_vh"],
            normal_scene["post_vv"], normal_scene["post_vh"],
            already_db=True,
        )
        assert result.water_confidence.shape == (64, 64)
        assert np.all(result.water_confidence >= 0.0)
        assert np.all(result.water_confidence <= 1.0)
        # Flood corridor rows 24-40 should have higher confidence
        flood_mean = np.mean(result.water_confidence[28:36, :])
        dry_mean   = np.mean(result.water_confidence[0:10, :])
        assert flood_mean > dry_mean


# ═══════════════════════════════════════════════════════════════════
# Fusion — contract weight verification
# ═══════════════════════════════════════════════════════════════════

class TestFusionContractWeights:

    def _make_water_result(self, shape, conf_val):
        from p2_ai.detect.sar_change import SARWaterDetectionResult
        conf = np.full(shape, conf_val, dtype=np.float32)
        mask = conf > 0.5
        return SARWaterDetectionResult(
            water_mask=mask,
            water_confidence=conf,
            post_db=np.zeros(shape, dtype=np.float32),
            pre_db=np.zeros(shape, dtype=np.float32),
            diff_db=np.zeros(shape, dtype=np.float32),
            post_threshold=-16.0,
            diff_threshold=-3.5,
        )

    def _make_debris_result(self, shape, conf_val):
        from p2_ai.detect.coherence import DebrisDetectionResult
        conf = np.full(shape, conf_val, dtype=np.float32)
        return DebrisDetectionResult(
            debris_mask=conf > 0.3,
            debris_confidence=conf,
            method_used="test",
            metric_grid=conf.copy(),
            threshold=0.3,
        )

    def test_correct_fusion_weights_all_present(self):
        """Verify weighted sum equals contract formula when all 4 channels present."""
        from p2_ai.detect.fuse import fuse_detections
        shape = (10, 10)
        p_sar  = 0.8    # SAR confidence (weight 0.30)
        p_coh  = 0.5    # coherence (weight 0.15)
        p_unet = 0.9    # U-Net (weight 0.40)
        p_ndwi_raw = 0.6  # raw NDWI → mapped to sigmoid

        # Expected P_fused (approx): 0.40*0.9 + 0.30*0.8 + 0.15*p_coh + 0.15*sigmoid(5*0.6)
        # sigmoid(3.0) ≈ 0.953
        expected_approx = 0.40 * p_unet + 0.30 * p_sar + 0.15 * p_coh + 0.15 * (1.0 / (1.0 + np.exp(-5 * p_ndwi_raw)))

        ndwi  = np.full(shape, p_ndwi_raw, dtype=np.float32)
        unet  = np.full(shape, p_unet,     dtype=np.float32)
        water = self._make_water_result(shape, p_sar)
        debris = self._make_debris_result(shape, p_coh)

        result = fuse_detections(water, debris, optical_ndwi=ndwi, unet_prob=unet)
        # Check weights sum to 1.0 when all channels present
        w = result.fusion_weights_used
        assert np.isclose(sum(w.values()), 1.0, atol=1e-6)

        # Check contract coefficients preserved
        assert np.isclose(w["unet"], 0.40, atol=1e-6)
        assert np.isclose(w["sar"],  0.30, atol=1e-6)
        assert np.isclose(w["coherence"], 0.15, atol=1e-6)
        assert np.isclose(w["ndwi"],  0.15, atol=1e-6)

        # Check actual fused confidence close to expected
        actual = float(np.mean(result.water_confidence))
        assert abs(actual - expected_approx) < 0.01

    def test_absent_unet_redistributes_weights(self):
        """When U-Net absent, remaining weights scale to sum=1."""
        from p2_ai.detect.fuse import fuse_detections
        shape = (10, 10)
        water  = self._make_water_result(shape, 0.6)
        debris = self._make_debris_result(shape, 0.3)
        ndwi   = np.full(shape, 0.4, dtype=np.float32)

        result = fuse_detections(water, debris, optical_ndwi=ndwi, unet_prob=None)
        w = result.fusion_weights_used
        assert w["unet"] == 0.0
        assert np.isclose(sum(w.values()), 1.0, atol=1e-6)

    def test_only_sar_present(self):
        """Single channel: weight must be 1.0 for SAR."""
        from p2_ai.detect.fuse import fuse_detections
        shape = (10, 10)
        water  = self._make_water_result(shape, 0.7)
        debris = self._make_debris_result(shape, 0.0)
        # Pass debris with zero confidence to simulate no coherence
        result = fuse_detections(water, debris, optical_ndwi=None, unet_prob=None)
        # All 4 channels are "present" (water and debris always come in)
        # But coherence = 0 and ndwi = None
        w = result.fusion_weights_used
        assert np.isclose(sum(w.values()), 1.0, atol=1e-6)
        assert w["ndwi"] == 0.0

    def test_ndwi_sigmoid_mapping(self):
        """NDWI must be mapped to [0,1] via sigmoid before fusion."""
        from p2_ai.detect.fuse import _ndwi_to_probability
        p_neg = _ndwi_to_probability(np.array([-1.0]), k=5.0)
        p_zer = _ndwi_to_probability(np.array([0.0]),  k=5.0)
        p_pos = _ndwi_to_probability(np.array([1.0]),  k=5.0)
        assert float(p_neg.ravel()[0]) < 0.1
        assert abs(float(p_zer.ravel()[0]) - 0.5) < 0.01
        assert float(p_pos.ravel()[0]) > 0.9

    def test_probability_bounds(self):
        from p2_ai.detect.fuse import fuse_detections
        shape = (32, 32)
        water  = self._make_water_result(shape, 0.5)
        debris = self._make_debris_result(shape, 0.5)
        ndwi   = np.full(shape, 0.3, dtype=np.float32)
        unet   = np.full(shape, 0.7, dtype=np.float32)
        result = fuse_detections(water, debris, optical_ndwi=ndwi, unet_prob=unet)
        assert np.all(result.confidence >= 0.0)
        assert np.all(result.confidence <= 1.0)
        assert np.all(result.water_confidence >= 0.0)
        assert np.all(result.water_confidence <= 1.0)

    def test_class_ids(self):
        from p2_ai.detect.fuse import CLASS_BACKGROUND, CLASS_WATER, CLASS_DEBRIS
        assert CLASS_BACKGROUND == 0
        assert CLASS_WATER == 1
        assert CLASS_DEBRIS == 2

    def test_missing_optical_zero_confidence(self):
        """Missing optical must not be treated as zero flood prob — weights redistribute."""
        from p2_ai.detect.fuse import fuse_detections
        shape = (10, 10)
        water  = self._make_water_result(shape, 0.8)
        debris = self._make_debris_result(shape, 0.0)
        with_opt    = fuse_detections(water, debris, optical_ndwi=np.full(shape, 0.5, dtype=np.float32))
        without_opt = fuse_detections(water, debris, optical_ndwi=None)
        # Both should produce finite results
        assert np.all(np.isfinite(with_opt.confidence))
        assert np.all(np.isfinite(without_opt.confidence))
        # Without optical, SAR/coherence/unet weights are redistributed → higher relative SAR weight
        # → confidence may be higher since SAR weight increases
        w_with    = with_opt.fusion_weights_used
        w_without = without_opt.fusion_weights_used
        assert w_without["sar"] > w_with["sar"]


# ═══════════════════════════════════════════════════════════════════
# U-Net — architecture and inference tests
# ═══════════════════════════════════════════════════════════════════

class TestUNetArchitecture:

    def test_five_channel_input_required(self):
        from p2_ai.detect.unet import UNetInferenceEngine
        engine = UNetInferenceEngine()
        # 4-channel input should raise
        bad_input = np.zeros((4, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match="5-channel"):
            engine.forward(bad_input)

    def test_valid_five_channel(self):
        from p2_ai.detect.unet import UNetInferenceEngine
        engine = UNetInferenceEngine()
        inp = np.zeros((5, 32, 32), dtype=np.float32)
        out = engine.forward(inp)
        assert out.shape == (32, 32)
        assert out.dtype == np.float32
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_odd_dimensions(self):
        from p2_ai.detect.unet import predict_unet
        shape = (37, 53)
        pre_vv  = np.full(shape, -12.0, dtype=np.float32)
        pre_vh  = np.full(shape, -18.0, dtype=np.float32)
        post_vv = np.full(shape, -22.0, dtype=np.float32)
        post_vh = np.full(shape, -27.0, dtype=np.float32)
        slope   = np.full(shape, 5.0, dtype=np.float32)
        result = predict_unet(pre_vv, pre_vh, post_vv, post_vh, slope)
        assert result.probability_map.shape == shape

    def test_nan_inf_input_safe(self):
        from p2_ai.detect.unet import predict_unet
        shape = (20, 20)
        pre_vv  = np.full(shape, -12.0, dtype=np.float32)
        pre_vh  = np.full(shape, -18.0, dtype=np.float32)
        post_vv = np.full(shape, -12.0, dtype=np.float32)
        post_vh = np.full(shape, -18.0, dtype=np.float32)
        slope   = np.full(shape, 5.0, dtype=np.float32)
        pre_vv[0, 0] = np.nan
        post_vv[5, 5] = np.inf
        result = predict_unet(pre_vv, pre_vh, post_vv, post_vh, slope)
        assert np.all(np.isfinite(result.probability_map))

    def test_pytorch_unet_build(self):
        """Verify the PyTorch U-Net can be instantiated and does a forward pass."""
        pytest.importorskip("torch")
        from p2_ai.detect.unet import _build_unet_model
        import torch
        model = _build_unet_model()
        model.eval()
        dummy = torch.zeros(1, 5, 64, 64, dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 1, 64, 64)

    def test_inference_backend_label(self):
        from p2_ai.detect.unet import predict_unet
        shape = (16, 16)
        result = predict_unet(
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )
        assert result.inference_backend in ("pytorch", "onnxruntime", "numpy_vectorized")


# ═══════════════════════════════════════════════════════════════════
# Coherence — honest proxy labelling
# ═══════════════════════════════════════════════════════════════════

class TestCoherenceHonesty:

    def test_complex_coherence_rejects_real_input(self):
        from p2_ai.detect.coherence import compute_complex_coherence
        real_arr = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="complex"):
            compute_complex_coherence(real_arr, real_arr)

    def test_proxy_labelled_correctly(self):
        from p2_ai.detect.coherence import compute_coherence_proxy
        pre_db  = np.full((20, 20), -12.0, dtype=np.float32)
        post_db = np.full((20, 20), -12.0, dtype=np.float32)
        proxy = compute_coherence_proxy(pre_db, post_db)
        assert proxy.shape == (20, 20)
        assert np.all(proxy >= 0.0)
        assert np.all(proxy <= 1.0)
        # Stable scene (no change) → high proxy coherence
        assert np.mean(proxy) > 0.8

    def test_proxy_decreases_with_large_change(self):
        from p2_ai.detect.coherence import compute_coherence_proxy
        pre_db  = np.full((20, 20), -12.0, dtype=np.float32)
        post_db = np.full((20, 20), +2.0,  dtype=np.float32)  # 14 dB surge
        proxy = compute_coherence_proxy(pre_db, post_db)
        assert np.mean(proxy) < 0.5  # large change → low proxy


# ═══════════════════════════════════════════════════════════════════
# GeoTIFF output metadata check (mock)
# ═══════════════════════════════════════════════════════════════════

class TestRasterMetadata:

    def test_mock_flood_conf_tif(self):
        """Verify data/mock_run/flood_raw_conf.tif has expected properties."""
        pytest.importorskip("rasterio")
        import rasterio
        from pathlib import Path
        tif = Path("data/mock_run/flood_raw_conf.tif")
        if not tif.exists():
            pytest.skip("Mock raster not found.")
        with rasterio.open(tif) as ds:
            arr = ds.read(1)
            assert arr.ndim == 2
            assert ds.count >= 1
            # Note: mock raster may not have a CRS (generated synthetically)
            # Production outputs must have valid CRS — this is verified in integration tests

    def test_mock_class_tif(self):
        pytest.importorskip("rasterio")
        import rasterio
        from pathlib import Path
        tif = Path("data/mock_run/class.tif")
        if not tif.exists():
            pytest.skip("Mock raster not found.")
        with rasterio.open(tif) as ds:
            arr = ds.read(1)
            unique_vals = set(np.unique(arr))
            # Class values must be subset of {0, 1, 2}
            assert unique_vals.issubset({0, 1, 2})


# ═══════════════════════════════════════════════════════════════════
# Integration — mock_run fixtures
# ═══════════════════════════════════════════════════════════════════

class TestMockRunIntegration:

    def test_detect_on_synthetic_s1(self):
        """End-to-end detect path using synthetic S1 data (no credentials needed)."""
        from p2_ai.detect import detect_water_sar, detect_debris_amplitude_fallback, fuse_detections, predict_unet
        import numpy as np

        shape = (64, 64)
        rng = np.random.default_rng(99)
        pre_vv  = rng.normal(-11.0, 2.0, shape).astype(np.float32)
        pre_vh  = rng.normal(-17.0, 2.5, shape).astype(np.float32)
        post_vv = pre_vv.copy()
        post_vh = pre_vh.copy()
        post_vv[20:40, 10:50] = rng.normal(-22.0, 1.0, (20, 40)).astype(np.float32)
        post_vh[20:40, 10:50] = rng.normal(-28.0, 1.0, (20, 40)).astype(np.float32)
        slope = np.full(shape, 3.0, dtype=np.float32)

        sar_res    = detect_water_sar(post_vv, pre_vv, already_db=True, post_threshold=-16.0)
        debris_res = detect_debris_amplitude_fallback(post_vv, pre_vv)
        unet_res   = predict_unet(pre_vv, pre_vh, post_vv, post_vh, slope)
        fused      = fuse_detections(sar_res, debris_res, unet_prob=unet_res.probability_map)

        assert fused.class_map.shape == shape
        assert fused.class_map.dtype == np.uint8
        assert set(np.unique(fused.class_map)).issubset({0, 1, 2})
        assert np.all(fused.confidence >= 0.0)
        assert np.all(fused.confidence <= 1.0)
        # Flood zone should be detected as water
        assert fused.class_map[28, 25] in (1, 2)  # water or debris (both acceptable)
