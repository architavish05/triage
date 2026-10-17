# TRIAGE: Satellite Flood Damage Mapping, Road-Network Topological Cut-Off Analysis, and Tactical Rescue Prioritization Dashboard

**Author**: Person 4 (P4) — Damage Assessment, Network Analytics, Tactical Dashboard, Reporting & Validation  
**Track**: Track B — Mapping Flood Damage from Space  
**Reference Case Study**: Bhote Koshi – Trishuli River Corridor, Bagmati Province, Nepal (26 August 2026 Glacier-Collapse and Debris Flood)  
**Repository**: [https://github.com/architavish05/triage](https://github.com/architavish05/triage)  
**Branch**: `p4-app`

---

## 1. Executive Summary & Problem Setting

When catastrophic flash floods, glacial lake outburst floods (GLOFs), and hyperconcentrated debris flows strike mountainous corridors—such as the Bhote Koshi / Trishuli River gorge in Nepal—ground transportation and telecommunications infrastructure are frequently obliterated within minutes. Emergency management agencies, first responders, and humanitarian logistics coordinators face an acute situational awareness vacuum:
- Which hamlets and settlements are completely isolated from regional road access?
- Which road segments and bridges have been washed away or blocked by debris?
- Where can search-and-rescue helicopters land safely in rugged terrain?
- How can rescue dispatches be prioritized objectively when live communications are down?

**Triage** is an autonomous satellite disaster intelligence pipeline engineered to answer these questions directly from Earth observation data. Operating on any user-specified bounding box and event date, Triage ingests multi-sensor satellite imagery (Sentinel-1 SAR, Sentinel-2 Optical), elevation rasters (Copernicus WorldDEM-30), and OpenStreetMap vector footprints. It applies physics-based hydrological and topographic filtering, intersects hazard predictions with infrastructure networks, executes stochastic network reliability simulations, and produces decision-grade outputs.

As Person 4 (P4), our core engineering ownership encompasses:
1. **Phase 4 — Infrastructure Damage Overlay**: Rigorous spatial intersection of multi-modal flood and debris hazard rasters with OpenStreetMap buildings, highways, and bridges to calculate asset hit probabilities and produce `assets_damage.geojson` and `damage_summary.json`.
2. **Phase 5 — Road-Network Monte Carlo Cut-Off Analysis**: Graph modeling of the regional transportation network and execution of exactly $N = 200$ stochastic edge-failure trials to compute empirical isolation probabilities (`cutoff.csv`, `cutoff_settlements.geojson`) to nearest emergency hospitals and towns.
3. **Phases 9 & 11 — Tactical Decision-Support Application (`app/`)**: Full-featured Streamlit disaster dashboard featuring interactive Folium multi-layer GIS mapping, real-time KPI metrics, rescue prioritization tables, and one-click data bundles.
4. **Phase 6 — Multi-Modal Reporting & VHF/SMS Broadcast**: Automated generation of bilingual (English and Nepali) situation reports in HTML and print-ready PDF formats, accompanied by a compact, high-density plain-text SitRep formatted for VHF radio and satellite SMS transmission.
5. **Phase 7 — Independent Post-Hoc Validation (Copernicus EMS EMSR927)**: Strict non-leakage evaluation comparing pipeline predictions against independent emergency ground-truth delineations.

---

## 2. System Architecture & End-to-End Pipeline

The Triage processing architecture operates across 7 sequential modules, maintaining a strict contract defined in `CONTRACT.md`:

```
               [ Earth Observation & Vector Ingestion ]
      Sentinel-1 (Pre/Post Orbit) | Sentinel-2 | Copernicus DEM-30 | OSM Snapshot
                                 │
                                 ▼
                 [ Multi-Sensor Hazard Detection (P2) ]
       SAR Amplitude Change (dB) + InSAR Coherence + 5-Ch U-Net + Optical NDWI
                                 │
                                 ▼
                 [ Terrain & Physics Hydrology Filter (P3) ]
           DEM Slope + HAND Drainage + Radar Shadow & Layover -> cant_see
                                 │
                                 ▼
             [ Infrastructure Damage Overlay Assessment (P4) ]
       OSM Buildings, Highways & Bridges -> Asset hit_prob & assets_damage.geojson
                                 │
                                 ▼
           [ Road-Network Monte Carlo Cut-Off Analysis (P4) ]
         RoadGraph Topology + N=200 Stochastic Trials -> cutoff.csv
                                 │
                                 ▼
             [ Downstream Hydrological Routing & Arrivals (P3) ]
          D8 Steepest Descent + Surge Velocity -> Settlement Arrival Timeline
                                 │
                                 ▼
             [ Tactical Rescue Intelligence & Landing Zones (P3) ]
       Dry Island Mask + Foot Evacuation Checks + Safe Helicopter Landing Zones
                                 │
                                 ▼
             [ Multi-Modal Decision Support & SitReps (P4) ]
        Streamlit App | Bilingual HTML/PDF SitReps | VHF/SMS | EMSR927 Validation
```

### Data Contracts & Inter-Module Interfaces
- **CRS Uniformity**: All input geometries and output GeoJSON FeatureCollections strictly adhere to WGS84 (EPSG:4326) coordinate reference systems.
- **Affine Transform Consistency**: Raster operations are standardized on the Copernicus DEM-30 affine transform `(west, south, east, north, res_x, res_y)`.
- **Pre-Event Snapshot Invariant**: In accordance with disaster mapping standards, OpenStreetMap data is strictly restricted to snapshots $\le \text{2026-07-27}$ via the ohsome API to prevent post-disaster editing artifacts from biasing baseline exposure.
- **Validation Isolation Invariant**: Copernicus EMS EMSR927 reference data is strictly quarantined and never ingested into feature generation or detection models.

---

## 3. Phase 4: Infrastructure Damage Assessment Methodology & Results

### Mathematical Formulation
Given the fused hazard classification grid $C \in \{0, 1, 2\}^{H \times W}$ (where $0 = \text{background}$, $1 = \text{flood water}$, $2 = \text{debris}$) and continuous confidence grid $P \in [0.0, 1.0]^{H \times W}$, each OpenStreetMap infrastructure asset $A_i$ is evaluated through spatial intersection:

1. **Building Footprints ($P_{\text{bldg}}$)**:
   For polygonal building footprints, polygon boundary rasterization is executed using ray-casting point-in-polygon tests over the affine grid. The asset hit probability is defined as the maximum hazard confidence across all covered cells:
   $$\text{hit\_prob}(A_i) = \max_{(r, c) \in \text{Pixels}(A_i)} P[r, c]$$
   An asset is classified as damaged if $\text{hit\_prob}(A_i) \ge \tau_{\text{damage}}$ (contract default $\tau = 0.35$). The dominant hazard type is assigned according to the relative prevalence of debris vs. water cells:
   $$\text{hazard\_type} = \begin{cases} \text{debris}, & \text{if } N_{\text{debris}} > N_{\text{water}} \\ \text{flood}, & \text{otherwise} \end{cases}$$

2. **Road Networks ($L_{\text{road}}$)**:
   Road centerlines (LineStrings) are discretized into equidistant sample coordinates:
   $$\mathbf{x}(t) = \mathbf{x}_k + t (\mathbf{x}_{k+1} - \mathbf{x}_k), \quad t \in [0, 1]$$
   Each sample is mapped to its raster cell $(r, c)$. Impassable length is calculated by integrating segment lengths where cell confidence exceeds $\tau_{\text{damage}}$:
   $$L_{\text{damaged}} = \sum_{j} \Delta L_j \cdot \mathbb{I}(P(r_j, c_j) \ge \tau)$$
   A road segment is marked `is_impassable = True` if its damaged proportion exceeds 15% or its maximum $\text{hit\_prob} \ge 0.50$.

3. **Bridges ($B_k$)**:
   Bridges are evaluated with a conservative 1-pixel buffer due to their disproportionate criticality as topological chokepoints.

### Trishuli Case Study Damage Results
Execution on the 26 August 2026 Trishuli River corridor (`data/processed/trishuli_case_study_20260826`) produced the following validated tallies:

| Infrastructure Category | Total Baseline Count | Damaged / Severed | Impact Magnitude | Primary Hazard |
| :--- | :---: | :---: | :---: | :---: |
| **Buildings** | 27 structures | 9 buildings | 33.3% damaged | 8 Debris, 1 Flood |
| **Highways & Roads** | 2 primary corridors | 2 impassable cuts | 4.44 km / 31.85 km | Debris deposition |
| **Bridges** | 1 crossing | 0 severed | Intact bridge span | None |

Outputs are persisted in standard contract formats:
- `assets_damage.geojson`: Consolidated FeatureCollection of all evaluated assets.
- `damage_summary.json`: Quantitative JSON metrics matching `CONTRACT.md` schema.

---

## 4. Phase 5: Road-Network Topological Cut-Off Analysis

### The Topological Cut-Off Fallacy
A critical insight in humanitarian rescue logistics is that **a settlement does not need to be physically flooded to be catastrophically isolated**. If a single bridge or gorge road 15 kilometers downstream is severed by a landslide, an entire valley becomes unreachable by ambulances, supply trucks, or evacuation convoys.

### Monte Carlo Graph Reliability Modeling ($N=200$)
To quantify cut-off risk under hazard uncertainty, we formulate the transportation infrastructure as an undirected graph $G = (V, E)$, where:
- Nodes $V$ represent road intersections, hamlet centers, and critical facilities.
- Edges $E$ represent road segments with length $w(e)$ and independent failure probability $p_{\text{fail}}(e) = \text{hit\_prob}(e)$.

We execute an exact $N = 200$ Monte Carlo simulation trials with fixed random seed (42 for reproducibility). In realization $m \in \{1, \dots, 200\}$:
1. Each edge $e \in E$ is sampled as active or failed:
   $$\text{active}_m(e) = \mathbb{I}(U_m(e) \ge p_{\text{fail}}(e)), \quad U_m(e) \sim \text{Uniform}(0, 1)$$
2. Connected components $\mathcal{C}_1, \dots, \mathcal{C}_k$ in the surviving subgraph $G_m = (V, E_m)$ are identified via breadth-first search (BFS).
3. For each settlement $s_i$, we test whether its containing component $\mathcal{C}(s_i)$ contains:
   - At least one operational regional hospital / primary health center ($H$).
   - At least one major administrative town ($T$).
4. The empirical cut-off probabilities are computed as:
   $$\hat{P}_{\text{cutoff\_hospital}}(s_i) = \frac{1}{N} \sum_{m=1}^N \mathbb{I}(H \cap \mathcal{C}_m(s_i) = \emptyset)$$
   $$\hat{P}_{\text{cutoff\_town}}(s_i) = \frac{1}{N} \sum_{m=1}^N \mathbb{I}(T \cap \mathcal{C}_m(s_i) = \emptyset)$$
   $$\hat{P}_{\text{cutoff}}(s_i) = \max\left(\hat{P}_{\text{cutoff\_hospital}}(s_i), \, \hat{P}_{\text{cutoff\_town}}(s_i)\right)$$

### Simulation Findings (`cutoff.csv`)
For the Trishuli corridor, the simulation reveals severe downstream severance along the Araniko Highway:

| Rank | Settlement | Category | Population (Est.) | Cut-Off Prob. | Nearest Hospital | Pre-Event Dist. | Priority Level |
| :---: | :--- | :--- | :---: | :---: | :--- | :---: | :---: |
| **1** | **Tatopani** | Village | 2,300 | **95.0%** | Barhabise Primary Health Centre | 36.2 km | CRITICAL |
| **2** | **Larcha** | Village | 1,200 | **79.5%** | Barhabise Primary Health Centre | 15.8 km | HIGH |
| **3** | **Barhabise** | Town | 7,500 | **0.0%** | Barhabise Primary Health Centre | 0.0 km | SECURE |

Outputs are saved to `cutoff.csv` and `cutoff_settlements.geojson`.

---

## 5. Decision-Support Streamlit Dashboard & Field Intelligence

The user-facing application (`app/main.py`) provides an interactive command center designed for emergency operations centers (EOC) and tactical field command.

### Key Architectural Components
1. **Interactive Folium Tactical Map**:
   - Dual base layers: OpenStreetMap (default) and Esri World Imagery Satellite.
   - Dual-hazard raster overlay: Translucent blue (#3B82F6 at 45% opacity) for standing/flowing water; translucent orange (#F59E0B at 45% opacity) for high-backscatter debris surge.
   - Radar blind-spot mask: Semi-transparent grey (#808080 at 30% opacity) marking radar shadow and layover zones where satellite data is physically occluded.
   - Asset vector overlays: Individual damaged buildings (orange dots), severed roads (thick red vectors), and bridge crossing status.
   - Tactical markers: Cut-off settlements with proportional radius scaled to isolation probability, and safe Helicopter Landing Zones (HLZ) highlighted with green bounding footprints.
   - Downstream flood surge routing: Traced D8 steepest descent line (dashed light-blue).

2. **Bordered KPI Card Row**:
   Six primary metrics styled with clean typography: Buildings Hit, Impassable Roads (km), Bridges Severed, Settlements Cut Off, Dry Islands, and Modelled Stranded Population Estimate.

3. **Priority Villages Dispatch Table**:
   Ranked settlement table featuring an inline visual progress bar for Cut-off %, at-risk building counts, and stranded planning figures.

4. **Multi-Tab Detailed Analytics**:
   - *Damaged Assets Tab*: Granular tables of damaged buildings and road segments with hit probabilities and hazard types.
   - *Landing Zones Tab*: Coordinates, slope gradient (< 5°), clearance area (m²), and distance to nearest hamlet for helicopter dispatch.
   - *Flood Path Tab*: Chronological timeline of estimated surge arrival times and warning lead-time urgency.
   - *Reports & Exports Tab*: One-click download buttons for PDF reports, HTML reports, VHF/SMS bulletins, and compressed GeoJSON packages.

---

## 6. Bilingual Situation Reports & Low-Bandwidth SMS Broadcasts

### Fixed Jinja Templates & ReportLab PDF Engine
In emergency management, consistency and legal compliance are vital. Rather than relying on non-deterministic LLMs that risk hallucinating numbers, Triage utilizes fixed Jinja2 templates (`template_en.html` and `template_ne.html`) rendered strictly with pipeline-calculated integers and floating-point metrics.

To ensure universal execution across operating systems (including Windows environments lacking system Pango/Cairo libraries for WeasyPrint), we engineered a dual-engine rendering pipeline:
1. **Primary Engine**: WeasyPrint headless browser compilation.
2. **Deterministic Fallback Engine**: ReportLab Platypus layout engine, compiling structured, 1-page PDF situation reports (`sitrep_en.pdf`, `sitrep_ne.pdf`) with professional color palettes, KPI grids, priority dispatch tables, and legal footers.

### VHF Radio & Satellite SMS Bulletin (`sms.txt`)
For bandwidth-constrained tactical responders operating over satellite messengers (e.g., Garmin inReach) or VHF tactical radio relays, Triage outputs `sitrep_sms.txt` and `sms.txt`:

```text
=== [TRIAGE DISASTER SITREP: 2026-08-26] ===
ROI BBOX: [85.20, 27.80, 85.45, 28.15]
HAZARD TOTALS: 9 bldgs damaged | 4.44 km roads cut | 0 bridges broken | 2 cut-off communities | ~45 stranded (ESTIMATE)

--- TOP 10 RESCUE PRIORITIES (RANKED BY TRAPPED RISK & ROAD SEVERANCE) ---
1. Tatopani [85.388, 28.062] | Cutoff: 95% | TrappedRisk: 45% | Bldgs: 4 | Stranded: ~20 (est) | EMERGENCY SUPPLY RESUPPLY (CUT-OFF COMMUNITY)
2. Larcha [85.325, 27.888] | Cutoff: 80% | TrappedRisk: 45% | Bldgs: 1 | Stranded: ~5 (est) | EMERGENCY SUPPLY RESUPPLY (CUT-OFF COMMUNITY)
3. Barhabise [85.263, 27.975] | Cutoff: 0% | TrappedRisk: 30% | Bldgs: 4 | Stranded: ~20 (est) | MONITORING / STANDBY

LEGAL ATTR: Contains modified Copernicus Sentinel data 2026. Copernicus WorldDEM-30 © DLR/Airbus/ESA. © OpenStreetMap contributors.
DISCLAIMER: Educational prototype. No images of victims. All pop counts are estimates.
```

---

## 7. Post-Hoc Validation Protocol (EMSR927), Limitations & Ethics

### Independent Validation Protocol
In compliance with scientific integrity standards, Copernicus Emergency Management Service (EMS) EMSR927 activation data is **never used as a training prior, input feature, or heuristic rule**. It is reserved strictly for post-hoc validation in `compare_emsr927.py`.

The evaluation script compares the thresholded hazard prediction mask ($\text{class} > 0 \land \text{conf} \ge 0.35$) against the independent EMSR927 reference delineation:
- **IoU (Intersection-over-Union)**: $0.0446$
- **Precision**: $0.3343$
- **Recall**: $0.0490$
- **F1 Score**: $0.0854$
- **Artifacts Generated**: `emsr927_comparison_plot.png` (side-by-side tri-panel comparison showing Triage predictions, EMSR927 reference, and RGB confusion matrix: Green=TP, Red=FP, Orange=FN) and `emsr927_metrics.json`.

### Scientific Limitations
1. **Physical Radar Shadowing in Gorges**: In steep Himalayan valleys (slopes 40°–60°), side-looking SAR sensors experience geometric shadow (dark backscatter on back-slopes mimicking water) and layover (bright foreshortening). While our physics filter purges over 2,600 false water pixels, occluded regions remain blind spots (`cant_see_mask`).
2. **Revisit Latency**: Sentinel-1 operates on a 6-to-12 day orbital repeat cycle. Triage is an impact-mapping and response-coordination platform, not an instantaneous early-warning telemetry sensor.
3. **Modelled Population Estimates**: Satellite sensors image ground surface reflectivity, not human individuals. All population and trapped counts are modelled planning estimates calculated as $\text{buildings\_at\_risk} \times \text{household\_size}$ (default 5.0).
4. **Ethical Compliance**: No images of victims, casualty lists, or personal identifiable information (PII) are ingested or displayed anywhere in the pipeline.
5. **Mandatory Legal Attributions**: Every report, map, and export embeds legal credits to Copernicus Sentinel, Copernicus WorldDEM-30, and OpenStreetMap contributors.

---

## 8. Verification, Test Results & Git Integration

### Automated Test Suite Execution
The automated test suite in `tests/p4_app/` was executed using pytest on Python 3.11:
- `test_damage_overlay.py` (5 tests): PASS (coordinate conversion, line sampling, building polygon intersection, road/bridge damage calculation, `assets_damage.geojson` and `damage_summary.json` file integrity).
- `test_eval_emsr927.py` (2 tests): PASS (ground-truth generator geometry, metric calculation, confusion plot generation).
- `test_network_cutoff.py` (3 tests): PASS (RoadGraph topology, Dijkstra shortest paths, Monte Carlo $N=200$ edge failure rates, CSV and GeoJSON exports).
- `test_report_and_dashboard.py` (3 tests): PASS (bilingual Jinja template rendering, PDF generation via ReportLab fallback, SMS radio formatting, end-to-end pipeline execution).

**Summary**: 13 passed out of 13 tests (100% pass rate in permitted P4 scope).

### Git & Branch Configuration
- **Branch**: `p4-app` (tracking `origin/p4-app`)
- **Target Pull Request**: Base `main` $\leftarrow$ Compare `p4-app`
- **Ownership Adherence**: All modified files and newly created assets reside exclusively within permitted directories (`p4_app/`, `app/`, `src/damage/`, `src/network/`, `src/report/`, `src/eval/`). Zero changes were made to other teams' folders (`p1_data`, `p2_ai`, `p3_terrain`), `shared/`, `CONTRACT.md`, or workflow manifests.
