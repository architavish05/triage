# Person 2 to Person 4 Handoff: AI Flood Detection & Multi-Sensor Fusion

**Author:** Person 2 (AI & Detection Engineer)  
**Target:** Person 4 (Application, SitRep, Dashboard & Validation Engineer)  
**System:** Triage Satellite Disaster Intelligence System  
**Branch:** `p2-ai`  
**Date:** October 2026  

---

## 1. Executive Summary

Person 2 has completed the implementation, testing, and hardening of all AI detection and sensor fusion components specified under Phase 2 and Phase 8 of `CONTRACT.md`. All deliverables reside strictly in `src/detect/` and `notebooks/`.

Key deliverables:
1. **Configurable 5-Channel U-Net (`src/detect/unet.py`)**:
   - PyTorch encoder-decoder architecture (`UNet5Ch`) accepting Pre-VV, Pre-VH, Post-VV, Post-VH, and DEM Slope.
   - 3-tier runtime execution: ONNX Runtime $\to$ PyTorch checkpoint $\to$ Deterministic NumPy heuristic.
   - Production ONNX export utility (`export_to_onnx`) with graph verification.
2. **Contract-Compliant 4-Way Fusion (`src/detect/fuse.py`)**:
   - Exact mathematical adherence to `CONTRACT.md` formula:
     $$P_{\text{fused}} = 0.40 \cdot P_{\text{unet}} + 0.30 \cdot P_{\text{sar}} + 0.15 \cdot P_{\text{coherence}} + 0.15 \cdot P_{\text{ndwi}}$$
   - Non-linear sigmoid mapping for optical NDWI: $P_{\text{ndwi}} = \frac{1}{1 + \exp(-5 \cdot \text{NDWI})}$.
   - Deterministic weight redistribution when optical or coherence sensors are unavailable.
3. **SAR Change & Coherence Engines (`src/detect/sar_change.py`, `src/detect/coherence.py`)**:
   - Robust Otsu thresholding with zero-variance and unimodal safeguards.
   - Dual-pol SAR enhancement (`detect_water_sar_dual_pol`).
   - Honest InSAR coherence handling: explicit separation between genuine complex SLC coherence and GRD amplitude change proxy (`compute_coherence_proxy`).
4. **Reproducible Notebook Workflows**:
   - `notebooks/01_unet_training.ipynb`: Executable PyTorch training loop with BCE+Dice loss and early stopping.
   - `notebooks/02_unseen_eval.ipynb`: Independent held-out evaluation harness complying with anti-fabrication rules (`NOT RUN` until verified Himalayan labels are supplied).

---

## 2. API Contract & Data Interfaces

Person 4's dashboard (`app/`), SitRep generator (`src/report/`), and validation modules (`src/eval/`) interact with Person 2's outputs via `src/pipeline.py` or direct module imports.

### 2.1 Final Fused Output (`FusedDetectionResult`)

When `fuse_detections()` is called, it returns a `FusedDetectionResult` dataclass:

| Attribute | Type | Shape | Value Domain / Meaning |
| :--- | :--- | :--- | :--- |
| `class_map` | `np.ndarray` (uint8) | `(H, W)` | **Hazard Classification:**<br>• `0`: Background (dry land / unaffected)<br>• `1`: Water (flood inundation)<br>• `2`: Debris (mudflow / debris deposit) |
| `confidence` | `np.ndarray` (float32) | `(H, W)` | **Detection Confidence:** `[0.0, 1.0]`. Multi-sensor fused posterior probability of hazard presence. |
| `water_mask` | `np.ndarray` (bool) | `(H, W)` | `True` where `class_map == 1`. |
| `debris_mask`| `np.ndarray` (bool) | `(H, W)` | `True` where `class_map == 2`. |
| `shape` | `Tuple[int, int]` | — | Spatial dimensions `(H, W)`. |
| `fusion_weights_used` | `Dict[str, float]`| — | Exact weights applied during fusion (e.g. `{"unet": 0.40, "sar": 0.30, "coherence": 0.15, "ndwi": 0.15}`). |

### 2.2 U-Net Output (`UNetInferenceResult`)

| Attribute | Type | Shape | Value Domain / Meaning |
| :--- | :--- | :--- | :--- |
| `flood_probability`| `np.ndarray` (float32) | `(H, W)` | Continuous flood probability in `[0.0, 1.0]`. |
| `binary_mask` | `np.ndarray` (bool) | `(H, W)` | Binarized flood mask at threshold (default 0.50). |
| `inference_backend`| `str` | — | One of `"onnxruntime"`, `"pytorch"`, or `"numpy_vectorized"`. |

---

## 3. SitRep & Dashboard Rendering Guidance

### 3.1 Confidence Mapping for Decision Makers
When rendering the Streamlit confidence slider or formatting VHF/SMS broadcasts:
- **High Confidence ($\ge 0.70$)**: Multi-sensor consensus (e.g., U-Net + SAR drop + optical NDWI confirmation). Primary target for immediate evacuation alerts and rescue prioritization.
- **Moderate Confidence ($0.45 \le p < 0.70$)**: Single high-reliability sensor (e.g., strong SAR drop under cloud cover without optical, or optical NDWI without radar). Flagged for reconnaissance.
- **Low Confidence ($< 0.45$)**: Specular reflection artifacts, ambiguous backscatter change, or steep terrain noise. Filtered from automated rescue tasking.

### 3.2 Sensor Availability & Weight Redistribution
When clouds prevent optical Sentinel-2 acquisition (`cloud_skipped=True`), `fuse_detections` redistributes the 15% optical weight proportionally:
- Standard 4-way: `U-Net: 0.40, SAR: 0.30, Coherence: 0.15, NDWI: 0.15`
- Cloud-covered (no NDWI): `U-Net: 0.47, SAR: 0.35, Coherence: 0.18, NDWI: 0.00`
- GRD-only (no SLC Coherence, no NDWI): `U-Net: 0.57, SAR: 0.43`

*Person 4 note:* The active weights are exposed in `fused.fusion_weights_used` and can be directly displayed in the Streamlit "Sensor Attribution" panel.

---

## 4. Known Technical Discrepancies & Test Calibration

### 4.1 Unit Test Calibration Conflict (`tests/test_detect_unet.py`)
In `tests/test_detect_unet.py::test_four_way_fusion_with_unet`:
- Line 108 asserts: `assert fused.confidence[18, 18] > 0.85`.
- **Root Cause**: This legacy test was written against an uncommitted prototype fusion formula (`0.50*sar + 0.50*unet + 0.20_ndwi_boost` which produced ~1.0 clipped).
- **Contract Enforcement**: Section 8 of `CONTRACT.md` mandates the exact weighted sum:
  $$P_{\text{fused}} = 0.40 \cdot 0.92 + 0.30 \cdot 0.77 + 0.15 \cdot 0.0 + 0.15 \cdot 0.963 = 0.772$$
- Per the repository governing rules ("CONTRACT.md IS THE SOURCE OF TRUTH"), Person 2 has strictly preserved the contract equation. The test assertion in `tests/test_detect_unet.py` should be updated by the test owner (P1/P4) from `> 0.85` to `> 0.70`.

### 4.2 InSAR Coherence vs. GRD Amplitude
- True InSAR coherence requires Complex SLC pairs ($\gamma = \frac{|\langle s_1 s_2^* \rangle|}{\sqrt{\langle |s_1|^2 \rangle \langle |s_2|^2 \rangle}}$).
- When operating on Sentinel-1 GRD backscatter products, `compute_complex_coherence()` correctly raises a `ValueError` if non-complex arrays are supplied.
- For GRD-only pipelines, use `compute_coherence_proxy()` which produces an honest amplitude stability proxy labelled `"grd_amplitude_proxy"`.

---

## 5. Validation Against EMSR927

Under the project data isolation policy:
- **Zero Data Leakage**: EMSR927 delineation layers are strictly post-hoc reference vectors. They are **never** ingested into `src/detect/` or used during model inference.
- Person 4's `src/eval/emsr927.py` can directly evaluate `fused.class_map == 1` against the EMSR927 flood polygons using `compute_segmentation_metrics(fused.water_mask, emsr_gt_mask)`.
