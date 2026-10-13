# Triage Interface & Data Contract (CONTRACT.md)

This document defines the strict API interfaces, file formats, schemas, and coordinate reference systems for the Triage disaster mapping engine.

All team members must adhere to these contracts. Changing any contract requires explicit consensus among all 4 roles.

---

## 1. Ownership & Module Boundaries

| Role | Branch | Owned Directories | Responsibilities |
| :--- | :--- | :--- | :--- |
| **P1** | `p1-data` | `src/fetch/`, `src/pipeline.py`, `src/offline/`, `data/mock_run/`, `README.md`, `ATTRIBUTIONS.md` | Data ingestion, caching, pipeline orchestrator, offline bundles. |
| **P2** | `p2-ai` | `src/detect/`, `notebooks/` | SAR change detection, InSAR coherence fallback, 5-ch U-Net, 4-way fusion. |
| **P3** | `p3-terrain` | `src/terrain/`, `src/damage/`, `src/network/`, `src/flowpath/`, `src/rescue/` | Slope/HAND, shadow masking, damage overlay, road cut-off, D8 routing, HLZ, rescue ranks. |
| **P4** | `p4-app` | `app/`, `.streamlit/`, `src/report/`, `src/eval/` | Bilingual SitRep templates, VHF/SMS broadcast, Streamlit dashboard, EMSR927 validation. |

---

## 2. Global Pipeline Entrypoint (`src/pipeline.py`)

### Function Signature
```python
def run(
    bbox: Tuple[float, float, float, float],
    flood_date: str,
    offline: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    ...
```

### Inputs
- `bbox`: `(min_lon, min_lat, max_lon, max_lat)` in WGS84 EPSG:4326.
- `flood_date`: ISO date string `YYYY-MM-DD` (e.g. `"2026-08-26"`).
- `offline`: Boolean flag. If `True`, operates strictly from cached assets.
- `output_dir`: Optional target `Path`. Defaults to `data/processed/{flood_date}_{min_lon:.2f}_{min_lat:.2f}`.

### Return Dictionary Keys
- `"bbox"`: `Tuple[float, float, float, float]`
- `"flood_date"`: `str`
- `"class_map"`: `np.ndarray` (2D `uint8`: 0=background, 1=water, 2=debris)
- `"confidence"`: `np.ndarray` (2D `float32` [0.0, 1.0])
- `"cant_see_mask"`: `np.ndarray` (2D `bool`)
- `"slope"`: `np.ndarray` (2D `float32`, degrees)
- `"hand"`: `np.ndarray` (2D `float32`, meters)
- `"transform"`: `(west, south, east, north, res_x, res_y)`
- `"damage"`: `DamageOverlayResult` dataclass
- `"isolation_records"`: `List[IsolationRecord]`
- `"ranked_cutoff_table"`: `List[Dict[str, Any]]`
- `"downstream_trace"`: `DownstreamTraceResult` dataclass
- `"settlement_arrivals"`: `List[SettlementArrival]`
- `"dry_island_mask"`: `np.ndarray` (2D `bool`)
- `"dry_islands_count"`: `int`
- `"landing_zones"`: `List[LandingZoneCandidate]`
- `"landing_zones_geojson"`: `Dict[str, Any]` (GeoJSON FeatureCollection)
- `"rescue_priorities"`: `List[RescuePriorityItem]`
- `"report_paths"`: `Dict[str, Path]` (`"html_en"`, `"html_ne"`, etc.)
- `"sms_summary"`: `str`
- `"output_dir"`: `Path`

---

## 3. Data Ingestion Contracts (`src/fetch/`)

### 1. Sentinel-1 (`fetch_s1`)
- **Enforcement**: Must fetch identical relative orbit track for pre/post pairs.
- **Temporal Baseline**: Pre-event image strictly 12 days prior to post-event.
- **Outputs**:
  - `pre_vv`, `pre_vh`, `post_vv`, `post_vh`: 2D `float32` arrays in decibels (dB).
  - Metadata: `orbit_direction` (`"ASCENDING"` or `"DESCENDING"`), `relative_orbit_track` (`int`).

### 2. Sentinel-2 (`fetch_s2`)
- **Enforcement**: Cloud cover strictly `< 20%`. If cloud cover $\ge 20\%$, returns `cloud_skipped=True`.
- **Outputs**: `green`, `nir` (2D `float32`), `ndwi` (normalized difference water index: `(green - nir) / (green + nir)`).

### 3. Copernicus DEM (`fetch_dem`)
- **Product**: Copernicus WorldDEM-30 (30m spatial resolution).
- **Outputs**: `elevation` (2D `float32` in meters above sea level).

### 4. OpenStreetMap (`fetch_osm`)
- **Snapshot Rule**: Ingest pre-event snapshot strictly $\le \text{2026-07-27}$ via ohsome API. Never ingest post-disaster OSM geometries.
- **Outputs**: GeoJSON FeatureCollections for `buildings`, `highways`, `bridges`, `places`, and `hospitals`.

---

## 4. Detection & AI Contracts (`src/detect/`)

### Class Map Definitions
- `0`: Background / Unaffected terrain
- `1`: Standing or flowing flood water
- `2`: Debris, mudflow, or sediment deposit

### U-Net Model Input Spec
- Tensor shape: `(1, 5, H, W)`
- Channels:
  1. `pre_vv` (normalized dB)
  2. `pre_vh` (normalized dB)
  3. `post_vv` (normalized dB)
  4. `post_vh` (normalized dB)
  5. `slope` (normalized degrees)
- Output: 2D probability map `float32` [0.0, 1.0].

### 4-Way Fusion Weights
$$\text{Confidence} = 0.40 \cdot P_{\text{unet}} + 0.30 \cdot P_{\text{sar}} + 0.15 \cdot P_{\text{coherence}} + 0.15 \cdot P_{\text{ndwi}}$$

---

## 5. Terrain & Physical Hydrology (`src/terrain/`)

- **Slope Constraint**: Purge impossible flood water on slopes $> 10^\circ$.
- **HAND Constraint**: Height Above Nearest Drainage $> 22\text{m}$ filtered as impossible water.
- **Occlusion Modeling**: Consolidated `cant_see_mask = shadow_mask | layover_mask | cloud_mask`.

---

## 6. Output File Structure & Schemas (`data/processed/` & `data/mock_run/`)

```
<run_directory>/
├── geojson/
│   ├── damaged_buildings.geojson
│   ├── damaged_roads.geojson
│   ├── damaged_bridges.geojson
│   ├── cutoff_settlements.geojson
│   ├── damage_summary.json
│   ├── landing_zones.geojson
│   ├── downstream_trace.json
│   └── rescue_priorities.json
├── reports/
│   ├── sitrep_en.html
│   ├── sitrep_ne.html
│   ├── sitrep_sms.txt
│   └── sitrep_en.pdf (optional, generated if WeasyPrint available)
├── hazard_grid.npz (contains class_map, cant_see_mask, bbox)
└── metadata.json
```

### GeoJSON Schema Specifications

#### `damaged_buildings.geojson`
```json
{
  "type": "Feature",
  "id": "way/101",
  "geometry": { "type": "Polygon", "coordinates": [...] },
  "properties": {
    "building": "residential",
    "hit_prob": 0.85,
    "hazard_type": "flood",
    "is_damaged": true
  }
}
```

#### `damaged_roads.geojson`
```json
{
  "type": "Feature",
  "id": "way/202",
  "geometry": { "type": "LineString", "coordinates": [...] },
  "properties": {
    "name": "Araniko Highway",
    "highway": "primary",
    "hit_prob": 0.78,
    "damaged_length_km": 2.45,
    "is_impassable": true
  }
}
```

#### `cutoff_settlements.geojson`
```json
{
  "type": "Feature",
  "id": "place/3",
  "geometry": { "type": "Point", "coordinates": [85.3875, 28.0625] },
  "properties": {
    "name": "Tatopani",
    "place": "village",
    "cutoff_prob": 0.95,
    "cutoff_prob_hospital": 0.95,
    "cutoff_prob_town": 0.95,
    "nearest_hospital": "Barhabise Primary Health Centre",
    "pre_event_distance_km": 36.21
  }
}
```

#### `damage_summary.json`
```json
{
  "total_buildings": 27,
  "damaged_buildings_count": 9,
  "flood_damaged_buildings": 1,
  "debris_damaged_buildings": 8,
  "total_roads_count": 2,
  "impassable_roads_count": 2,
  "total_roads_km": 31.85,
  "damaged_roads_km": 4.44,
  "total_bridges": 1,
  "damaged_bridges_count": 0,
  "damage_threshold_used": 0.35
}
```

---

## 7. Legal & Ethical Invariants

1. **Mandatory Attributions**: All reports, dashboards, and export products must contain:
   - *"Contains modified Copernicus Sentinel data 2026."*
   - *"Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved."*
   - *"© OpenStreetMap contributors."*
2. **Strict Validation Isolation**: Copernicus EMS (EMSR927) data must NEVER be used as a pipeline input. It is strictly reserved for post-hoc validation in `src/eval/compare_emsr927.py`.
3. **No Personal Data**: Zero images of victims or personal casualty records anywhere in the project.
4. **Modelled Estimates**: All stranded population figures must strictly be labeled `(estimate)`.
