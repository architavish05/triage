"""Triage | Flood damage from space.

Streamlit decision-support application for disaster response teams.
"""

import sys
from pathlib import Path
import json

# Ensure repository root is on sys.path when executed via `streamlit run app/app.py`
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import streamlit as st
import folium
from folium import plugins
from streamlit_folium import folium_static

from src.pipeline import run

# Page configuration
st.set_page_config(
    page_title="Triage | Flood damage from space",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
<style>
  .main-title { font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-bottom: 0px; }
  .main-subtitle { font-size: 1.0rem; color: #475569; margin-bottom: 15px; font-weight: 500; }
  .metric-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; text-align: center; }
  .estimate-pill { background: #fef3c7; color: #92400e; font-weight: 700; font-size: 0.75rem; padding: 2px 6px; border-radius: 4px; }
  .footer-text { font-size: 0.8rem; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 10px; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🛰️ Triage | Flood damage from space</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Satellite-based flood damage mapping &amp; rescue prioritization engine</div>', unsafe_allow_html=True)

# Sidebar controls
st.sidebar.header("🎯 Area & Event Selection")

preset = st.sidebar.selectbox(
    "Scenario Preset",
    [
        "Case Study: Bhote Koshi - Trishuli Corridor, Nepal (2026-08-26)",
        "Custom Bounding Box",
    ],
)

if "Bhote Koshi" in preset:
    default_bbox = (85.20, 27.80, 85.45, 28.15)
    default_date = "2026-08-26"
else:
    default_bbox = (85.10, 27.70, 85.50, 28.20)
    default_date = "2026-08-26"

col_w, col_s = st.sidebar.columns(2)
min_lon = col_w.number_input("Min Lon (West)", value=default_bbox[0], format="%.4f")
min_lat = col_s.number_input("Min Lat (South)", value=default_bbox[1], format="%.4f")

col_e, col_n = st.sidebar.columns(2)
max_lon = col_e.number_input("Max Lon (East)", value=default_bbox[2], format="%.4f")
max_lat = col_n.number_input("Max Lat (North)", value=default_bbox[3], format="%.4f")

flood_date = st.sidebar.text_input("Flood Date (YYYY-MM-DD)", value=default_date)
offline_mode = st.sidebar.checkbox("Offline Mode (Cache Only)", value=False)

run_button = st.sidebar.button("🚀 Run Disaster Pipeline", type="primary", use_container_width=True)

# Run pipeline or fetch from session state
if run_button or "pipeline_results" not in st.session_state:
    bbox = (min_lon, min_lat, max_lon, max_lat)
    with st.spinner("Executing end-to-end multi-sensor pipeline..."):
        results = run(bbox=bbox, flood_date=flood_date, offline=offline_mode)
        st.session_state["pipeline_results"] = results

results = st.session_state["pipeline_results"]
damage = results["damage"]
summary = damage.summary
priorities = results["rescue_priorities"]
arrivals = results["settlement_arrivals"]
hlzs = results["landing_zones"]

# Top KPI Summary Cards
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)
kpi_col1.metric("Damaged Buildings", summary.get("damaged_buildings_count", 0))
kpi_col2.metric("Impassable Roads", f"{summary.get('damaged_roads_km', 0.0)} km")
kpi_col3.metric("Severed Bridges", summary.get("damaged_bridges_count", 0))
cutoff_count = sum(1 for p in priorities if p.cutoff_prob >= 0.5)
kpi_col4.metric("Cut-off Communities", cutoff_count)
kpi_col5.metric("Dry Islands", results["dry_islands_count"])
total_stranded = sum(p.stranded_population_estimate for p in priorities)
kpi_col6.metric("Stranded (Estimate)", f"~{total_stranded}")

# Interactive Folium Map
st.subheader("🗺️ Operational Geospatial Intelligence Map")

map_center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]
m = folium.Map(location=map_center, zoom_start=11, tiles="CartoDB positron")

# Layer 1: Damaged Buildings
bldg_layer = folium.FeatureGroup(name="Damaged Buildings", show=True)
for feat in damage.damaged_buildings.get("features", []):
    if feat.get("properties", {}).get("is_damaged", False):
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if coords:
            # Centroid
            if geom.get("type") == "Polygon":
                ring = coords[0]
                p_lon = sum(p[0] for p in ring) / len(ring)
                p_lat = sum(p[1] for p in ring) / len(ring)
            else:
                p_lon, p_lat = coords[0], coords[1]
            folium.CircleMarker(
                location=[p_lat, p_lon],
                radius=4,
                color="#b91c1c",
                fill=True,
                fill_color="#ef4444",
                fill_opacity=0.8,
                tooltip=f"Damaged Building (Hit Prob: {feat['properties']['hit_prob']:.2f})",
            ).add_to(bldg_layer)
bldg_layer.add_to(m)

# Layer 2: Damaged & Impassable Roads
road_layer = folium.FeatureGroup(name="Damaged Roads", show=True)
for feat in damage.damaged_roads.get("features", []):
    if feat.get("properties", {}).get("is_impassable", False):
        coords = feat.get("geometry", {}).get("coordinates", [])
        if coords:
            latlons = [[p[1], p[0]] for p in coords]
            folium.PolyLine(
                locations=latlons,
                color="#dc2626",
                weight=4,
                tooltip=f"Impassable Road: {feat['properties'].get('name', 'Road')} ({feat['properties'].get('damaged_length_km', 0)} km damaged)",
            ).add_to(road_layer)
road_layer.add_to(m)

# Layer 3: Cut-Off Settlements
settlement_layer = folium.FeatureGroup(name="Cut-Off Settlements", show=True)
for p in priorities:
    color = "#dc2626" if p.cutoff_prob >= 0.7 else ("#f59e0b" if p.cutoff_prob >= 0.4 else "#10b981")
    folium.CircleMarker(
        location=[p.coordinates[1], p.coordinates[0]],
        radius=7 + int(p.trapped_risk * 6),
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.75,
        tooltip=f"Rank #{p.rank}: {p.settlement_name} | Cutoff: {p.cutoff_prob*100:.0f}% | Trapped Risk: {p.trapped_risk*100:.0f}%",
    ).add_to(settlement_layer)
settlement_layer.add_to(m)

# Layer 4: Helicopter Landing Zones (HLZs)
hlz_layer = folium.FeatureGroup(name="Helicopter Landing Zones (HLZ)", show=True)
for h in hlzs:
    folium.Marker(
        location=[h.coordinates[1], h.coordinates[0]],
        icon=folium.Icon(icon="plane", prefix="fa", color="green"),
        tooltip=f"{h.hlz_id} near {h.nearest_settlement} (Slope: {h.slope_deg} deg, Dist: {h.distance_to_settlement_m:.0f}m)",
    ).add_to(hlz_layer)
hlz_layer.add_to(m)

# Layer 5: Downstream Flood Flowpath
flow_layer = folium.FeatureGroup(name="Downstream Flood Path", show=True)
if results.get("downstream_trace"):
    coords = results["downstream_trace"].path_coords
    latlons = [[p[1], p[0]] for p in coords]
    folium.PolyLine(locations=latlons, color="#2563eb", weight=5, opacity=0.8, tooltip="Downstream Flood Surge Route").add_to(flow_layer)
flow_layer.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
folium_static(m, width=1250, height=520)

# Decision Support Tabs
tab_priority, tab_timeline, tab_radio, tab_downloads = st.tabs([
    "🚨 Rescue Prioritization Table",
    "⏱️ Flood Arrival Timeline",
    "📻 SMS / Radio Briefing",
    "📥 Downloads & Export Bundles",
])

with tab_priority:
    st.markdown("### Tactical Rescue Prioritization Ranks")
    p_data = []
    for item in priorities:
        p_data.append({
            "Rank": item.rank,
            "Settlement": item.settlement_name,
            "Place Type": item.place_type,
            "Cut-off Prob": f"{item.cutoff_prob*100:.1f}%",
            "Trapped Risk": f"{item.trapped_risk*100:.1f}%",
            "At-Risk Bldgs": item.buildings_at_risk,
            "Stranded (Estimate)": item.stranded_population_estimate,
            "Dry Island": "YES" if item.is_on_dry_island else "No",
            "Foot Escape Feasible": "YES" if item.escape_possible else "NO (Trapped)",
            "Priority Score": item.priority_score,
            "Recommended Action": item.recommended_action,
        })
    st.dataframe(p_data, use_container_width=True)

with tab_timeline:
    st.markdown("### Downstream Flood Arrival Sequence")
    arr_data = []
    for arr in arrivals:
        arr_data.append({
            "Arrival Sequence": arr.rank_order,
            "Settlement Name": arr.name,
            "Distance From Breach (km)": f"{arr.distance_from_source_km:.2f} km",
            "Channel Elevation": f"{arr.elevation_m:.0f} m",
            "Estimated Arrival": f"{arr.arrival_time_mins:.1f} mins ({arr.arrival_time_hours:.2f} hrs)",
            "Warning Urgency": arr.warning_lead_time_urgency,
        })
    st.dataframe(arr_data, use_container_width=True)

with tab_radio:
    st.markdown("### Plain-Text Briefing (SMS / VHF Radio Broadcast)")
    st.text_area("Radio Briefing Content", value=results["sms_summary"], height=250)

with tab_downloads:
    st.markdown("### Export Products & Decision Packs")
    d_col1, d_col2, d_col3 = st.columns(3)

    rep_paths = results.get("report_paths", {})
    if "html_en" in rep_paths and rep_paths["html_en"].exists():
        with open(rep_paths["html_en"], "r", encoding="utf-8") as f:
            d_col1.download_button("📄 Download SitRep (English HTML)", data=f.read(), file_name=f"sitrep_en_{flood_date}.html", mime="text/html")
    if "html_ne" in rep_paths and rep_paths["html_ne"].exists():
        with open(rep_paths["html_ne"], "r", encoding="utf-8") as f:
            d_col2.download_button("🇳🇵 Download SitRep (Nepali HTML)", data=f.read(), file_name=f"sitrep_ne_{flood_date}.html", mime="text/html")

    d_col3.download_button(
        "📻 Download SMS Briefing (.txt)",
        data=results["sms_summary"],
        file_name=f"triage_sitrep_{flood_date}.txt",
        mime="text/plain",
    )

# Mandatory Attributions & Disclaimer
st.markdown("""
<div class="footer-text">
  <strong>Data Attributions:</strong><br>
  - Contains modified Copernicus Sentinel data 2026.<br>
  - Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.<br>
  - © OpenStreetMap contributors.<br>
  - Citations: Bountos et al., 2024 (Kuro Siwo); Bonafilia et al., CVPR Workshops 2020 (Sen1Floods11).<br>
  <em>Disclaimer: Educational prototype, not an operational tool. No images of victims anywhere. All population counts are modelled planning estimates.</em>
</div>
""", unsafe_allow_html=True)
