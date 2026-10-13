# Triage | Satellite Flood Damage Mapping for Disaster Rescue

**Triage** is an end-to-end satellite disaster intelligence and rescue prioritization engine designed to assist emergency response operations following catastrophic floods and debris flows. 

**Reference Case Study**: The 26 August 2026 glacier-collapse and flash flood along the Bhote Koshi – Trishuli river corridor, Bagmati Province, Nepal.

> **Ethical Note & Disclaimer**: This repository is an educational prototype and scientific research pipeline, not an operational emergency management tool. In accordance with humanitarian and disaster mapping ethics, **no images of victims or personal casualty records are processed or displayed anywhere within this project**. All casualty and population risk figures are modelled planning estimates.

---

## Attributions & Scientific Citations

### Mandatory Legal Data Attributions
- **Contains modified Copernicus Sentinel data 2026.**
- **Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.**
- **© OpenStreetMap contributors.**

### Scientific Benchmark Citations
- **Kuro Siwo**:
  > Bountos, N. I., et al. (2024). *Kuro Siwo: A multi-sensor, multi-modal SAR and Optical flood mapping benchmark dataset*.
- **Sen1Floods11**:
  > Bonafilia, D., et al. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*. CVPR Workshops 2020.

---

## System Architecture & End-to-End Workflow

Given any geographic bounding box (`bbox`) and event date (`flood_date`), Triage produces actionable intelligence across 7 core analytical stages:

```
                  [ Satellite & Vector Ingestion ]
           Sentinel-1 (Pre/Post Orbit) | Sentinel-2 | Copernicus DEM | OSM (<=2026-07-27)
                                      │
                                      ▼
                      [ Multi-Sensor Hazard Detection ]
            SAR Change (dB) + InSAR Coherence + 5-Channel U-Net AI + Optical NDWI
                                      │
                                      ▼
                         [ Terrain & Physics Filter ]
               DEM Slope + HAND Drainage + Radar Shadow & Layover -> cant_see
                                      │
                                      ▼
                        [ Infrastructure Damage Overlay ]
            OSM Buildings, Roads & Bridges Intersected -> Asset hit_prob
                                      │
                                      ▼
                       [ Topological Cut-Off Analysis ]
               Road Network Graph + Monte Carlo Edge Failure (N=200)
                                      │
                                      ▼
                      [ Hydrological Flowpath & Arrival ]
            D8 Steepest Descent + Surge Velocity -> Chronological Timeline
                                      │
                                      ▼
                        [ Tactical Rescue Intelligence ]
        Dry Islands + Foot Escape Feasibility + Helicopter Landing Zones (HLZ)
                                      │
                                      ▼
                       [ Multi-Modal Decision Support ]
          Streamlit App | Bilingual PDF SitReps (EN/NE) | VHF Radio/SMS | MBTiles Pack
```

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/organization/triage.git
cd triage
pip install -r requirements.txt
```

### 2. Configure Credentials (`.env`)
Copy `.env.example` to `.env` and provide your Copernicus Data Space Ecosystem (CDSE) credentials:
```bash
cp .env.example .env
```
Edit `.env`:
```env
CDSE_CLIENT_ID=your_client_id_here
CDSE_CLIENT_SECRET=your_client_secret_here
CDSE_USERNAME=your_username_here
CDSE_PASSWORD=your_password_here
```
*(Note: If credentials are not set, Triage automatically falls back to deterministic simulation/prefetched assets for testing).*

---

## Usage Guide

### 1. Python API Execution
The unified public entry point executes the entire pipeline end-to-end:
```python
from src.pipeline import run

results = run(
    bbox=(85.20, 27.80, 85.45, 28.15),  # (min_lon, min_lat, max_lon, max_lat)
    flood_date="2026-08-26",
    offline=False,
)

print("Damaged Buildings:", results["damage"].summary["damaged_buildings_count"])
print("Top Priority Rescue:", results["rescue_priorities"][0].settlement_name)
```

### 2. Run the Case Study
To run the Trishuli corridor (26 Aug 2026) case study and generate post-hoc Copernicus EMS (EMSR927) validation metrics:
```bash
python run_case_study.py
```

### 3. Launch the Interactive Dashboard
To launch the Streamlit decision-support application:
```bash
streamlit run app/app.py
```
Dashboard features:
- Area & date picker with scenario presets
- Multi-layer interactive Folium map (Flood, Debris, Damaged Assets, Cut-off Communities, HLZ Landing Zones, Flowpath)
- Decision-support tables (Rescue priorities, Downstream arrival timeline)
- Downloads: One-page bilingual situation reports (English & Nepali HTML/PDF), plain-text radio summaries, and GeoJSON bundles.

### 4. Offline Field Kit Execution
For air-gapped disaster operations without internet connectivity:
```python
from src.offline.prefetch import prefetch_region
from src.offline.export_bundle import export_responder_bundle
from src.pipeline import run

# 1. Prefetch data in base camp before deploying to the field
prefetch_region(bbox=(85.20, 27.80, 85.45, 28.15), flood_date="2026-08-26")

# 2. Run in the field with network disabled (strictly reads from local cache)
results = run(bbox=(85.20, 27.80, 85.45, 28.15), flood_date="2026-08-26", offline=True)

# 3. Export complete field bundle (MBTiles raster database, GeoJSONs, SitReps, and ZIP pack)
export_responder_bundle(results, bundle_output_dir=results["output_dir"] / "field_pack")
```

---

## Post-Hoc Validation Protocol (EMSR927)

> **HARD SCIENTIFIC RULE**: Copernicus EMS (EMSR927) or any published damage map is **NEVER** ingested as an input or feature. It is loaded strictly post-hoc in `src/eval/compare_emsr927.py` after final pipeline execution to calculate unbiased IoU, precision, and recall metrics.

```bash
python -c "from src.pipeline import run; from src.eval.compare_emsr927 import compare_with_emsr927; res = run((85.20, 27.80, 85.45, 28.15), '2026-08-26'); compare_with_emsr927(res['class_map'], res['confidence'], res['output_dir'] / 'eval')"
```

---

## Limitations & Scientific Realities

Every automated satellite disaster mapping pipeline possesses physical and operational constraints. Responders must understand the following limitations:

1. **Satellites Cannot Detect Individual People**:
   Spaceborne SAR and optical sensors image ground surface reflectivity, not human beings. Any label indicating "trapped" or "stranded" represents a **modelled vulnerability estimate** based on spatial encirclement and road graph topological failure, not confirmed living presence.
2. **Satellite Revisit Gaps (No Early Warning)**:
   Sentinel-1 has a 6-to-12 day orbital repeat cycle. A satellite disaster system is a **post-disaster impact mapping and response coordination tool**, not an early-warning telemetry system.
3. **Radar Shadow and Layover in Deep Himalayan Valleys**:
   In narrow Himalayan gorges with 40°–60° slopes, side-looking radar experiences geometric shadow (dark backscatter on back-slopes mimicking specular water) and layover (bright compression on fore-slopes). While Triage models these in `shadow_mask.py`, steep terrain creates unavoidable radar blind spots (`cant_see` mask).
4. **Cloud Obscuration Blocks Optical Sensors**:
   Monsoon storms and glacier collapse events often generate heavy cloud cover. When Sentinel-2 cloud cover exceeds 20%, optical imagery is skipped (`cloud_skipped=True`), relying entirely on radar and DEM layers.
5. **OpenStreetMap Data Incompleteness**:
   In remote mountain hamlets, OpenStreetMap building footprints and trails may be incomplete or unmapped prior to the disaster.
6. **Household Size Assumptions**:
   Stranded population counts are calculated as $\text{buildings\_at\_risk} \times \text{household\_size\_assumption}$ (default: 5.0). Actual rural occupancy varies seasonally and diurnally. All numbers are explicitly labelled `"estimate"`.
7. **Training Distribution Gap**:
   Benchmark training datasets (Kuro Siwo, Sen1Floods11) primarily comprise lowland and riverine flood events. Complex Himalayan glacier outbursts and hyperconcentrated debris flows exhibit distinct backscatter dynamics.
8. **Offline Mode Scope**:
   Offline execution (`offline=True`) operates strictly on previously prefetched regional data. Uncached geographic areas cannot be processed without an active internet connection.

---

## License & Attribution Summary

Licensed under the Apache License 2.0. All satellite data processed subject to Copernicus Open Access terms and OpenStreetMap ODbL.
