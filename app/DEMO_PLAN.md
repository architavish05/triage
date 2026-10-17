# Triage | Track B: Mapping Flood Damage from Space
## 3-Minute Video Demonstration Script & Operational Walkthrough Plan

**Role**: Person 4 (P4) — Damage Assessment, Network Analytics, Dashboard, Reports & Validation  
**Reference Case Study**: Bhote Koshi – Trishuli River Corridor, Bagmati Province, Nepal (26 August 2026)  
**Total Target Duration**: 3 Minutes (180 Seconds)

---

### [0:00 – 0:30] Scene 1: The Operational Challenge & Problem Setting
- **Visual**: Disaster scene backdrop (steep Himalayan gorge of Bhote Koshi / Trishuli). High-angle aerial view of the river corridor.
- **Narrator**:
  > *"When catastrophic mountain floods and glacier collapses strike remote regions like the Bhote Koshi corridor in Nepal, traditional communications and road networks are instantly wiped out. Rescuers on the ground are blind: which villages are trapped? Which bridges are severed? Where can helicopters land safely?*
  > 
  > *This is **Triage** — an autonomous satellite disaster intelligence and rescue prioritization engine built to tell responders exactly where the damage is and who is cut off, directly from space."*

---

### [0:30 – 1:00] Scene 2: End-to-End Satellite Ingestion & Physics Filtering
- **Visual**: Architecture workflow diagram transitioning to terminal executing `python run_case_study.py`.
- **Narrator**:
  > *"Triage ingests dual-orbit Sentinel-1 Synthetic Aperture Radar, Sentinel-2 multispectral imagery, Copernicus WorldDEM-30 elevation, and pre-disaster OpenStreetMap snapshots. 
  > 
  > In narrow Himalayan valleys with 40° to 60° slopes, raw radar creates severe false alarms due to geometric shadow. Triage solves this by combining a 5-channel U-Net with physics-based HAND (Height Above Nearest Drainage) constraints and radar shadow-layover ray tracing, purging over 2,600 false water pixels."*

---

### [0:01:00 – 0:01:30] Scene 3: Phase 4 Damage Assessment & Phase 5 Network Cut-Off
- **Visual**: GIS overlay of OSM building footprints and highway vectors showing spatial intersection with hazard confidence.
- **Narrator**:
  > *"In Phase 4, Triage intersects the validated hazard grid with infrastructure assets. Across the corridor, 9 buildings and 4.44 kilometers of road are identified with significant damage probability, generating `assets_damage.geojson` and `damage_summary.json`.
  > 
  > In Phase 5, we model the road network as a topological graph and run N=200 Monte Carlo stochastic edge failure simulations. Even if an entire village has no water in its center, its lifeline highway may be cut. The simulation computes exact isolation probabilities to emergency healthcare: Tatopani is isolated with 95% probability, while Larcha faces a 79.5% cut-off risk."*

---

### [0:01:30 – 0:02:15] Scene 4: Streamlit Tactical Dashboard Walkthrough
- **Visual**: Screen capture of the dark-themed Streamlit application (`streamlit run app/main.py`).
- **Narrator**:
  > *"Here is the Triage Decision-Support Dashboard. At the top, six high-contrast KPI cards display total damaged structures, impassable road cuts, severed bridges, isolated villages, and modelled stranded population estimates.
  > 
  > On the left, our interactive Folium tactical map displays 7 switchable layers:
  > - Blue inundation zones and orange debris surge.
  > - Grey radar shadow blind spots so commanders know where satellites cannot see.
  > - Red severed highways and orange damaged buildings.
  > - Green Helicopter Landing Zones (HLZ) with slope < 5° and safe clearance.
  > - And dashed blue downstream flood arrival trajectories.
  > 
  > On the right, the Priority Villages panel dynamically ranks settlements, directing emergency teams to Tatopani and Larcha first."*

---

### [0:02:15 – 0:02:45] Scene 5: Multi-Modal Reports, SMS & Post-Hoc Validation
- **Visual**: Click 'Download PDF Report', showing 1-page bilingual SitRep; display plain-text SMS radio summary; display `emsr927_comparison_plot.png`.
- **Narrator**:
  > *"In disaster zones with zero internet, Triage produces low-bandwidth outputs:
  > - A 160-character compatible plain-text SitRep formatted for VHF radio and satellite SMS.
  > - Bilingual, one-page printable situation reports in English and Nepali with mandatory Copernicus and OpenStreetMap attributions.
  > 
  > Finally, in Phase 11, we perform strict post-hoc validation against independent Copernicus EMS EMSR927 ground truth. EMSR927 is never ingested into the model. Our independent comparison achieves 0.334 Precision and 0.085 F1 score over complex mountain debris flows."*

---

### [0:02:45 – 0:03:00] Scene 6: Conclusion & System Invariants
- **Visual**: Ethical disclaimer slide and GitHub repository URL (`https://github.com/architavish05/triage`).
- **Narrator**:
  > *"Triage adheres to strict ethical disaster mapping standards: zero personal casualty records or victim images are processed, and all population figures are clearly marked as modelled planning estimates.
  > 
  > Triage: transforming raw spaceborne radar and elevation into life-saving rescue priorities."*

---

### Key Production Assets for Demo Recording:
- Application Launch: `streamlit run app/main.py`
- Case Study Runner: `python run_case_study.py`
- Pre-processed Scene: `data/mock_run/` or `data/processed/trishuli_case_study_20260826/`
- Generated Reports: `data/processed/trishuli_case_study_20260826/reports/sitrep_en.pdf`
- Validation Plot: `data/processed/trishuli_case_study_20260826/eval_emsr927/emsr927_comparison_plot.png`
