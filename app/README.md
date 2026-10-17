# Triage Decision-Support Application

Interactive Streamlit disaster response and evacuation prioritization dashboard for satellite-detected flood and debris flow events.

## Launch Instructions

To launch the dashboard:

```bash
streamlit run app/main.py
```

Alternatively:

```bash
streamlit run p4_app/app/app.py
```

To launch with a specific run folder directly:

```bash
streamlit run app/main.py -- data/mock_run/
```

## Features

1. **Scenario Preset & Custom Bounding Box**: Select preset disaster events (e.g., Bhote Koshi / Trishuli Corridor, Nepal) or define custom coordinates and flood dates.
2. **Interactive Folium Tactical Map**: Multi-layer GIS viewer with switchable base maps, flood inundation (blue), debris deposit (orange), radar shadow/layover blind spots (grey), damaged buildings/roads/bridges, cut-off settlements, and safe Helicopter Landing Zones (HLZ).
3. **KPI Metrics Overview**: Instant counts of damaged buildings, impassable road kilometers, broken bridges, isolated communities, dry islands, and estimated stranded population.
4. **Prioritized Rescue Evacuation Table**: Ranked list of communities by trapped risk score, road cutoff probability, and vulnerable population.
5. **Situation Reports & Data Export**: One-click download of bilingual Situation Reports (English and Nepali HTML/PDF), plain-text VHF/SMS bulletins, and compressed GeoJSON data packages.

## Attributions

- Contains modified Copernicus Sentinel data 2026.
- Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.
- © OpenStreetMap contributors.
