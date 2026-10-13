"""UI Components and Map Builder for Triage Streamlit Dashboard.

Provides typed, modular functions for:
- Header and metadata
- Bordered KPI metrics row
- Priority villages ranking table
- Folium multi-layer tactical disaster map
- Detail tabs (damaged assets, landing zones, flood path, reports)
- Legal attributions footer
- Run folder caching and loading
"""

from pathlib import Path
import json
import zipfile
import io
from typing import Any, Dict, List, Optional, Tuple

import folium
import numpy as np
import pandas as pd
import streamlit as st


@st.cache_data
def load_run_from_folder(run_dir_str: str) -> Dict[str, Any]:
    """Read and cache pipeline outputs from a processed run directory."""
    run_dir = Path(run_dir_str)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir_str}")

    geojson_dir = run_dir / "geojson"
    reports_dir = run_dir / "reports"

    # 1. Damage summary
    summary_path = geojson_dir / "damage_summary.json"
    summary: Dict[str, Any] = {}
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

    # 2. GeoJSON layers
    damaged_bldgs = _load_geojson(geojson_dir / "damaged_buildings.geojson")
    damaged_roads = _load_geojson(geojson_dir / "damaged_roads.geojson")
    damaged_bridges = _load_geojson(geojson_dir / "damaged_bridges.geojson")
    cutoff_settlements = _load_geojson(geojson_dir / "cutoff_settlements.geojson")
    hlz_geojson = _load_geojson(geojson_dir / "landing_zones.geojson")

    # 3. Downstream trace coords
    trace_path = geojson_dir / "downstream_trace.json"
    trace_coords: List[Tuple[float, float]] = []
    if trace_path.exists():
        with open(trace_path, "r", encoding="utf-8") as f:
            trace_coords = [tuple(p) for p in json.load(f)]

    # 4. Priorities
    priorities_path = geojson_dir / "rescue_priorities.json"
    priorities: List[Dict[str, Any]] = []
    if priorities_path.exists():
        with open(priorities_path, "r", encoding="utf-8") as f:
            priorities = json.load(f)
    elif cutoff_settlements.get("features"):
        for feat in cutoff_settlements["features"]:
            props = feat.get("properties", {})
            geom = feat.get("geometry", {})
            priorities.append({
                "settlement_name": props.get("name", "Settlement"),
                "place_type": props.get("place", "village"),
                "coordinates": tuple(geom.get("coordinates", [0.0, 0.0])),
                "cutoff_prob": props.get("cutoff_prob", 0.0),
                "trapped_risk": props.get("trapped_risk", props.get("cutoff_prob", 0.5)),
                "buildings_at_risk": props.get("buildings_at_risk", 1),
                "stranded_population_estimate": props.get("population_est", 10),
                "is_on_dry_island": props.get("is_on_dry_island", False),
                "escape_possible": props.get("escape_possible", False),
                "priority_score": props.get("priority_score", 1.0),
                "recommended_action": props.get("recommended_action", "EVACUATION"),
            })

    # 5. Reports & SMS
    sms_text = ""
    sms_path = reports_dir / "sitrep_sms.txt"
    if sms_path.exists():
        with open(sms_path, "r", encoding="utf-8") as f:
            sms_text = f.read()

    # 6. Rasters & Metadata
    class_map = None
    cant_see_mask = None
    bbox = (85.20, 27.80, 85.45, 28.15)
    area_name = "Bhote Koshi - Trishuli Corridor, Nepal"
    flood_date = "2026-08-26"

    grid_path = run_dir / "hazard_grid.npz"
    if grid_path.exists():
        data = np.load(grid_path)
        class_map = data["class_map"]
        cant_see_mask = data["cant_see_mask"]
        if "bbox" in data:
            bbox = tuple(float(x) for x in data["bbox"])

    meta_path = run_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
            bbox = tuple(meta.get("bbox", bbox))
            flood_date = meta.get("flood_date", flood_date)
            area_name = meta.get("area_name", area_name)

    return {
        "bbox": bbox,
        "flood_date": flood_date,
        "area_name": area_name,
        "summary": summary,
        "damaged_buildings": damaged_bldgs,
        "damaged_roads": damaged_roads,
        "damaged_bridges": damaged_bridges,
        "cutoff_settlements": cutoff_settlements,
        "landing_zones_geojson": hlz_geojson,
        "trace_coords": trace_coords,
        "priorities": priorities,
        "class_map": class_map,
        "cant_see_mask": cant_see_mask,
        "sms_text": sms_text,
        "reports_dir": reports_dir,
        "geojson_dir": geojson_dir,
        "run_dir": run_dir,
    }


def _load_geojson(path: Path) -> Dict[str, Any]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"type": "FeatureCollection", "features": []}


def render_header(area_name: Optional[str] = None, flood_date: Optional[str] = None) -> None:
    """Render top header with clean typography and muted run context."""
    # Custom CSS to hide deploy button, minimal toolbar clutter, and clean typography
    st.markdown(
        """
        <style>
        .stDeployButton { display: none !important; }
        [data-testid="stToolbarActions"] { display: none !important; }
        #MainMenu { visibility: hidden !important; }
        footer { visibility: hidden !important; }
        
        .header-container {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            padding-bottom: 12px;
            margin-bottom: 20px;
            border-bottom: 1px solid #1E293B;
        }
        .header-title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #E6EDF7;
            margin: 0;
            line-height: 1.1;
        }
        .header-subtitle {
            font-size: 0.95rem;
            color: #94A3B8;
            margin-top: 4px;
        }
        .header-meta {
            text-align: right;
            font-size: 0.85rem;
            color: #64748B;
            line-height: 1.4;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col_title, col_meta = st.columns([3, 1])
    with col_title:
        st.markdown(
            "<div><div class='header-title'><strong>Triage</strong></div>"
            "<div class='header-subtitle'>Flood damage mapping from satellite data</div></div>",
            unsafe_allow_html=True,
        )
    with col_meta:
        if area_name and flood_date:
            st.markdown(
                f"<div class='header-meta'>Area: {area_name}<br>Flood date: {flood_date}</div>",
                unsafe_allow_html=True,
            )


def render_kpi_row(
    summary: Dict[str, Any],
    priorities: List[Any],
    dry_islands_count: int,
) -> None:
    """Render 6 KPI cards using st.metric inside bordered containers."""
    bldgs_hit = summary.get("damaged_buildings_count", 0)
    roads_km = summary.get("damaged_roads_km", 0.0)
    bridges_severed = summary.get("damaged_bridges_count", 0)

    cutoff_count = sum(
        1 for p in priorities
        if (p.get("cutoff_prob", 0.0) if isinstance(p, dict) else getattr(p, "cutoff_prob", 0.0)) >= 0.5
    )

    total_stranded = sum(
        (p.get("stranded_population_estimate", 0) if isinstance(p, dict)
         else getattr(p, "stranded_population_estimate", 0))
        for p in priorities
    )

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        with st.container(border=True):
            st.metric(label="Buildings hit", value=bldgs_hit)

    with col2:
        with st.container(border=True):
            st.metric(label="Roads impassable (km)", value=f"{roads_km:.2f}")

    with col3:
        with st.container(border=True):
            st.metric(label="Bridges severed", value=bridges_severed)

    with col4:
        with st.container(border=True):
            st.metric(label="Settlements cut off", value=cutoff_count)

    with col5:
        with st.container(border=True):
            st.metric(label="Dry islands", value=dry_islands_count)

    with col6:
        with st.container(border=True):
            st.metric(label="People stranded (estimate)", value=f"~{total_stranded}")


def render_priority_villages_table(priorities: List[Any]) -> None:
    """Render priority villages table with progress-bar style column for cut-off %."""
    st.markdown("#### Priority villages")

    rows = []
    for p in priorities:
        if isinstance(p, dict):
            name = p.get("settlement_name", p.get("name", "Unknown"))
            cutoff = float(p.get("cutoff_prob", 0.0)) * 100.0
            bldgs = int(p.get("buildings_at_risk", 0))
            stranded = int(p.get("stranded_population_estimate", p.get("population_est", 0)))
        else:
            name = getattr(p, "settlement_name", "Unknown")
            cutoff = float(getattr(p, "cutoff_prob", 0.0)) * 100.0
            bldgs = int(getattr(p, "buildings_at_risk", 0))
            stranded = int(getattr(p, "stranded_population_estimate", 0))

        rows.append({
            "Village": name,
            "Cut-off %": round(cutoff, 1),
            "Buildings at risk": bldgs,
            "Stranded est.": stranded,
        })

    df = pd.DataFrame(rows)
    if not df.empty and "Cut-off %" in df.columns:
        df = df.sort_values(by="Cut-off %", ascending=False)

    st.dataframe(
        df,
        column_config={
            "Cut-off %": st.column_config.ProgressColumn(
                "Cut-off %",
                format="%.0f%%",
                min_value=0,
                max_value=100,
            ),
            "Village": st.column_config.TextColumn("Village"),
            "Buildings at risk": st.column_config.NumberColumn("Buildings at risk"),
            "Stranded est.": st.column_config.NumberColumn("Stranded est."),
        },
        hide_index=True,
        use_container_width=True,
        height=560,
    )


def build_folium_map(
    bbox: Tuple[float, float, float, float],
    class_map: Optional[np.ndarray],
    cant_see_mask: Optional[np.ndarray],
    damaged_buildings: Dict[str, Any],
    damaged_roads: Dict[str, Any],
    damaged_bridges: Dict[str, Any],
    priorities: List[Any],
    landing_zones_geojson: Dict[str, Any],
    downstream_coords: Optional[List[Tuple[float, float]]] = None,
) -> folium.Map:
    """Build multi-layer tactical folium map with dark styling and exact layer controls."""
    min_lon, min_lat, max_lon, max_lat = bbox
    center = [(min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0]

    # Map initialization with OpenStreetMap as default, Esri Satellite as switchable base
    m = folium.Map(location=center, zoom_start=11, tiles=None)

    folium.TileLayer(
        "OpenStreetMap",
        name="Street map",
        attr="© OpenStreetMap contributors",
    ).add_to(m)

    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Satellite",
    ).add_to(m)

    # 1. Flood and debris raster layer (Blue #3B82F6 at 45%, Orange #F59E0B at 45%)
    if class_map is not None:
        h, w = class_map.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        # Water: blue #3B82F6 at 45% (alpha 115)
        rgba[class_map == 1] = [59, 130, 246, 115]
        # Debris: orange #F59E0B at 45% (alpha 115)
        rgba[class_map == 2] = [245, 158, 11, 115]

        flood_overlay = folium.raster_layers.ImageOverlay(
            image=rgba,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            name="Flood and debris",
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=1,
        )
        flood_overlay.add_to(m)
    else:
        folium.FeatureGroup(name="Flood and debris", show=True).add_to(m)

    # 2. Cannot-see areas raster layer (Grey at 30% opacity, alpha 76)
    if cant_see_mask is not None:
        h, w = cant_see_mask.shape
        cant_see_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        cant_see_rgba[cant_see_mask > 0] = [128, 128, 128, 76]

        cant_see_overlay = folium.raster_layers.ImageOverlay(
            image=cant_see_rgba,
            bounds=[[min_lat, min_lon], [max_lat, max_lon]],
            name="Cannot-see areas",
            opacity=1.0,
            interactive=False,
            cross_origin=False,
            zindex=2,
        )
        cant_see_overlay.add_to(m)
    else:
        folium.FeatureGroup(name="Cannot-see areas", show=True).add_to(m)

    # 3. Damaged buildings (amber circle markers)
    bldg_fg = folium.FeatureGroup(name="Damaged buildings", show=True)
    for feat in damaged_buildings.get("features", []):
        props = feat.get("properties", {})
        hit_prob = float(props.get("hit_prob", 0.0))
        is_dmg = props.get("is_damaged", False) or hit_prob >= 0.35
        if not is_dmg:
            continue

        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            continue

        if geom.get("type") == "Polygon":
            ring = coords[0]
            p_lon = sum(p[0] for p in ring) / len(ring)
            p_lat = sum(p[1] for p in ring) / len(ring)
        elif geom.get("type") == "Point":
            p_lon, p_lat = coords[0], coords[1]
        else:
            continue

        folium.CircleMarker(
            location=[p_lat, p_lon],
            radius=4,
            color="#D97706",
            fill=True,
            fill_color="#F59E0B",
            fill_opacity=0.85,
            weight=1,
            popup=folium.Popup(
                f"Building: {props.get('name', 'Building')}<br>Hit probability: {hit_prob:.2f}",
                max_width=250,
            ),
            tooltip=f"Building | Hit probability: {hit_prob:.2f}",
        ).add_to(bldg_fg)
    bldg_fg.add_to(m)

    # 4. Damaged roads (red lines) & Severed bridges (red X markers)
    roads_fg = folium.FeatureGroup(name="Damaged roads", show=True)
    for feat in damaged_roads.get("features", []):
        props = feat.get("properties", {})
        is_imp = props.get("is_impassable", False) or float(props.get("hit_prob", 0.0)) >= 0.35
        if not is_imp:
            continue

        coords = feat.get("geometry", {}).get("coordinates", [])
        if coords:
            latlons = [[p[1], p[0]] for p in coords]
            road_name = props.get("name", "Road")
            dmg_len = float(props.get("damaged_length_km", 0.0))
            folium.PolyLine(
                locations=latlons,
                color="#EF4444",
                weight=4,
                opacity=0.85,
                popup=folium.Popup(
                    f"Road: {road_name}<br>Impassable length: {dmg_len:.2f} km",
                    max_width=250,
                ),
                tooltip=f"Road: {road_name} | Impassable: {dmg_len:.2f} km",
            ).add_to(roads_fg)

    # Severed bridges inside roads group
    for feat in damaged_bridges.get("features", []):
        props = feat.get("properties", {})
        is_dmg = props.get("is_damaged", False) or props.get("bridge_status") == "severed" or float(props.get("hit_prob", 0.0)) >= 0.35
        if not is_dmg:
            continue

        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [])
        if not coords:
            continue

        if geom.get("type") == "LineString":
            p_lon = (coords[0][0] + coords[-1][0]) / 2.0
            p_lat = (coords[0][1] + coords[-1][1]) / 2.0
        elif geom.get("type") == "Point":
            p_lon, p_lat = coords[0], coords[1]
        else:
            continue

        b_name = props.get("name", "Bridge")
        hit_prob = float(props.get("hit_prob", 0.0))
        folium.Marker(
            location=[p_lat, p_lon],
            icon=folium.DivIcon(
                html='<div style="font-size:16px; font-weight:bold; color:#DC2626; line-height:1; text-align:center;">&#10006;</div>'
            ),
            popup=folium.Popup(
                f"Bridge: {b_name}<br>Status: Severed<br>Hit probability: {hit_prob:.2f}",
                max_width=250,
            ),
            tooltip=f"Bridge: {b_name} | Status: Severed",
        ).add_to(roads_fg)
    roads_fg.add_to(m)

    # 5. Cut-off settlements (red markers with radius scaled by priority)
    settlements_fg = folium.FeatureGroup(name="Cut-off settlements", show=True)
    for p in priorities:
        if isinstance(p, dict):
            name = p.get("settlement_name", p.get("name", "Settlement"))
            coords = p.get("coordinates", [0.0, 0.0])
            cutoff = float(p.get("cutoff_prob", 0.0))
            bldgs = int(p.get("buildings_at_risk", 0))
            stranded = int(p.get("stranded_population_estimate", p.get("population_est", 0)))
        else:
            name = getattr(p, "settlement_name", "Settlement")
            coords = getattr(p, "coordinates", (0.0, 0.0))
            cutoff = float(getattr(p, "cutoff_prob", 0.0))
            bldgs = int(getattr(p, "buildings_at_risk", 0))
            stranded = int(getattr(p, "stranded_population_estimate", 0))

        if not coords or len(coords) < 2:
            continue

        # Scale radius between 6 and 16 based on cut-off priority
        radius = max(6, min(16, int(6 + cutoff * 10)))
        lat, lon = coords[1], coords[0]

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color="#991B1B",
            fill=True,
            fill_color="#DC2626",
            fill_opacity=0.8,
            weight=2,
            popup=folium.Popup(
                f"Settlement: {name}<br>Cut-off probability: {cutoff:.2f}<br>"
                f"Buildings at risk: {bldgs}<br>Stranded population estimate: {stranded}",
                max_width=280,
            ),
            tooltip=f"{name} | Cut-off: {cutoff*100:.0f}%",
        ).add_to(settlements_fg)
    settlements_fg.add_to(m)

    # 6. Landing zones (green polygons)
    hlz_fg = folium.FeatureGroup(name="Landing zones", show=True)
    for feat in landing_zones_geojson.get("features", []):
        geom = feat.get("geometry", {})
        props = feat.get("properties", {})
        hlz_id = props.get("hlz_id", "HLZ")
        slope_deg = float(props.get("slope_deg", 0.0))
        area_m2 = float(props.get("area_m2", 0.0))
        nearest = props.get("nearest_settlement", "Community")
        dist_m = float(props.get("distance_to_settlement_m", 0.0))

        if geom.get("type") == "Polygon":
            folium.GeoJson(
                feat,
                style_function=lambda x: {
                    "fillColor": "#10B981",
                    "color": "#059669",
                    "weight": 2,
                    "fillOpacity": 0.55,
                },
                popup=folium.Popup(
                    f"Landing zone: {hlz_id}<br>Slope: {slope_deg:.1f} deg<br>"
                    f"Area: {area_m2:.0f} m2<br>Nearest: {nearest} ({dist_m:.0f} m)",
                    max_width=250,
                ),
                tooltip=f"{hlz_id} | Slope: {slope_deg:.1f} deg",
            ).add_to(hlz_fg)
        elif geom.get("type") == "Point":
            coords = geom.get("coordinates", [])
            if coords:
                lon, lat = coords[0], coords[1]
                # Construct square footprint polygon (~30m radius)
                d = 0.0003
                footprint = [
                    [lat - d, lon - d],
                    [lat - d, lon + d],
                    [lat + d, lon + d],
                    [lat + d, lon - d],
                ]
                folium.Polygon(
                    locations=footprint,
                    color="#059669",
                    fill=True,
                    fill_color="#10B981",
                    fill_opacity=0.55,
                    weight=2,
                    popup=folium.Popup(
                        f"Landing zone: {hlz_id}<br>Slope: {slope_deg:.1f} deg<br>"
                        f"Area: {area_m2:.0f} m2<br>Nearest: {nearest} ({dist_m:.0f} m)",
                        max_width=250,
                    ),
                    tooltip=f"{hlz_id} | Slope: {slope_deg:.1f} deg",
                ).add_to(hlz_fg)
    hlz_fg.add_to(m)

    # 7. Flood path (dashed light-blue line)
    path_fg = folium.FeatureGroup(name="Flood path", show=True)
    if downstream_coords:
        latlons = [[p[1], p[0]] for p in downstream_coords]
        folium.PolyLine(
            locations=latlons,
            color="#38BDF8",
            weight=4,
            dash_array="8, 8",
            opacity=0.85,
            popup=folium.Popup("Downstream flood surge path", max_width=220),
            tooltip="Flood path",
        ).add_to(path_fg)
    path_fg.add_to(m)

    # Legend box inside the map (bottom left) matching theme and colors
    legend_html = """
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 1000;
        background-color: rgba(17, 26, 46, 0.88);
        padding: 10px 14px;
        border: 1px solid #1E293B;
        border-radius: 6px;
        font-size: 11px;
        color: #E6EDF7;
        font-family: sans-serif;
        line-height: 1.5;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.5);
    ">
        <div style="font-weight: 700; margin-bottom: 5px; border-bottom: 1px solid #334155; padding-bottom: 3px;">Legend</div>
        <div><span style="display:inline-block; width:12px; height:12px; background:#3B82F6; opacity:0.65; margin-right:6px; border-radius:2px;"></span>Flood water</div>
        <div><span style="display:inline-block; width:12px; height:12px; background:#F59E0B; opacity:0.65; margin-right:6px; border-radius:2px;"></span>Debris surge</div>
        <div><span style="display:inline-block; width:12px; height:12px; background:#808080; opacity:0.5; margin-right:6px; border-radius:2px;"></span>Cannot-see area</div>
        <div><span style="display:inline-block; width:9px; height:9px; background:#F59E0B; border:1px solid #D97706; border-radius:50%; margin-right:7px;"></span>Damaged building</div>
        <div><span style="display:inline-block; width:14px; height:3px; background:#EF4444; margin-right:6px; vertical-align:middle;"></span>Damaged road</div>
        <div><span style="color:#DC2626; font-weight:bold; margin-right:6px;">&#10006;</span>Severed bridge</div>
        <div><span style="display:inline-block; width:9px; height:9px; background:#DC2626; border-radius:50%; margin-right:7px;"></span>Cut-off settlement</div>
        <div><span style="display:inline-block; width:12px; height:12px; background:#10B981; border:1px solid #059669; margin-right:6px; border-radius:2px;"></span>Landing zone</div>
        <div><span style="display:inline-block; width:14px; height:2px; border-top:2px dashed #38BDF8; margin-right:6px; vertical-align:middle;"></span>Flood path</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # Layer control with user switchability
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def render_tabs(
    summary: Dict[str, Any],
    damaged_buildings: Dict[str, Any],
    damaged_roads: Dict[str, Any],
    damaged_bridges: Dict[str, Any],
    landing_zones_geojson: Dict[str, Any],
    downstream_coords: Optional[List[Tuple[float, float]]],
    settlement_arrivals: List[Any],
    sms_text: str,
    run_dir: Optional[Path],
    flood_date: str,
) -> None:
    """Render detail tabs for damaged assets, landing zones, flood path, and reports."""
    tab_assets, tab_hlz, tab_path, tab_report = st.tabs([
        "Damaged assets",
        "Landing zones",
        "Flood path",
        "Report",
    ])

    # Tab 1: Damaged assets
    with tab_assets:
        st.markdown("#### Infrastructure damage details")
        c1, c2, c3 = st.columns(3)
        c1.metric("Damaged buildings", summary.get("damaged_buildings_count", 0))
        c2.metric("Impassable roads (km)", f"{summary.get('damaged_roads_km', 0.0):.2f}")
        c3.metric("Severed bridges", summary.get("damaged_bridges_count", 0))

        # Buildings table
        bldg_rows = []
        for feat in damaged_buildings.get("features", []):
            p = feat.get("properties", {})
            if p.get("is_damaged", False) or float(p.get("hit_prob", 0.0)) >= 0.35:
                bldg_rows.append({
                    "Building ID": feat.get("id", "Building"),
                    "Hit probability": f"{float(p.get('hit_prob', 0.0)):.2f}",
                    "Hazard type": p.get("hazard_type", "Flood"),
                })
        if bldg_rows:
            st.markdown("##### Damaged buildings sample")
            st.dataframe(pd.DataFrame(bldg_rows[:50]), hide_index=True, use_container_width=True)

        # Roads table
        road_rows = []
        for feat in damaged_roads.get("features", []):
            p = feat.get("properties", {})
            if p.get("is_impassable", False) or float(p.get("hit_prob", 0.0)) >= 0.35:
                road_rows.append({
                    "Highway": p.get("highway", "Road"),
                    "Name": p.get("name", "Unnamed"),
                    "Impassable length (km)": f"{float(p.get('damaged_length_km', 0.0)):.2f}",
                    "Hit probability": f"{float(p.get('hit_prob', 0.0)):.2f}",
                })
        if road_rows:
            st.markdown("##### Impassable road segments")
            st.dataframe(pd.DataFrame(road_rows), hide_index=True, use_container_width=True)

    # Tab 2: Landing zones
    with tab_hlz:
        st.markdown("#### Safe helicopter landing zones (HLZ)")
        hlz_rows = []
        for feat in landing_zones_geojson.get("features", []):
            p = feat.get("properties", {})
            hlz_rows.append({
                "HLZ ID": p.get("hlz_id", "HLZ"),
                "Nearest settlement": p.get("nearest_settlement", "Unknown"),
                "Distance (m)": f"{float(p.get('distance_to_settlement_m', 0.0)):.0f}",
                "Slope (deg)": f"{float(p.get('slope_deg', 0.0)):.1f}",
                "Clearance area (m2)": f"{float(p.get('area_m2', 0.0)):.0f}",
                "Footpath accessible": "Yes" if p.get("connected_footpath", True) else "No",
            })
        if hlz_rows:
            st.dataframe(pd.DataFrame(hlz_rows), hide_index=True, use_container_width=True)
        else:
            st.info("No helicopter landing zones detected in current region.")

    # Tab 3: Flood path
    with tab_path:
        st.markdown("#### Downstream flood arrival timeline")
        if settlement_arrivals:
            arr_rows = []
            for arr in settlement_arrivals:
                if isinstance(arr, dict):
                    arr_rows.append({
                        "Sequence": arr.get("rank_order", 1),
                        "Settlement": arr.get("name", "Settlement"),
                        "Distance from breach (km)": f"{float(arr.get('distance_from_source_km', 0.0)):.2f}",
                        "Elevation (m)": f"{float(arr.get('elevation_m', 0.0)):.0f}",
                        "Estimated arrival": f"{float(arr.get('arrival_time_mins', 0.0)):.1f} min ({float(arr.get('arrival_time_hours', 0.0)):.2f} hr)",
                        "Urgency": arr.get("warning_lead_time_urgency", "MEDIUM"),
                    })
                else:
                    arr_rows.append({
                        "Sequence": getattr(arr, "rank_order", 1),
                        "Settlement": getattr(arr, "name", "Settlement"),
                        "Distance from breach (km)": f"{getattr(arr, 'distance_from_source_km', 0.0):.2f}",
                        "Elevation (m)": f"{getattr(arr, 'elevation_m', 0.0):.0f}",
                        "Estimated arrival": f"{getattr(arr, 'arrival_time_mins', 0.0):.1f} min ({getattr(arr, 'arrival_time_hours', 0.0):.2f} hr)",
                        "Urgency": getattr(arr, "warning_lead_time_urgency", "MEDIUM"),
                    })
            st.dataframe(pd.DataFrame(arr_rows), hide_index=True, use_container_width=True)
        elif downstream_coords:
            st.write(f"Traced downstream flood routing corridor comprising {len(downstream_coords)} coordinate waypoints.")
        else:
            st.info("No downstream flowpath sequence available.")

    # Tab 4: Report
    with tab_report:
        st.markdown("#### Situation reports and data exports")
        c_pdf, c_geo, c_sms = st.columns(3)

        # PDF / HTML Download
        pdf_path = (run_dir / "reports" / f"sitrep_en_{flood_date}.pdf") if run_dir else None
        if not pdf_path or not pdf_path.exists():
            pdf_path = (run_dir / "reports" / "sitrep_en.pdf") if run_dir else None

        html_path = (run_dir / "reports" / "sitrep_en.html") if run_dir else None

        if pdf_path and pdf_path.exists():
            with open(pdf_path, "rb") as f:
                c_pdf.download_button(
                    label="Download PDF report",
                    data=f.read(),
                    file_name=f"sitrep_en_{flood_date}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        elif html_path and html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                c_pdf.download_button(
                    label="Download HTML report",
                    data=f.read(),
                    file_name=f"sitrep_en_{flood_date}.html",
                    mime="text/html",
                    use_container_width=True,
                )
        else:
            c_pdf.button("PDF report unavailable", disabled=True, use_container_width=True)

        # GeoJSON Bundle Download
        geo_zip_bytes = _create_geojson_zip(run_dir)
        c_geo.download_button(
            label="Download GeoJSON data",
            data=geo_zip_bytes,
            file_name=f"triage_geojson_{flood_date}.zip",
            mime="application/zip",
            use_container_width=True,
        )

        # SMS Text Download
        c_sms.download_button(
            label="Download SMS text",
            data=sms_text,
            file_name=f"sitrep_sms_{flood_date}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.markdown("##### SMS and radio broadcast text preview")
        st.code(sms_text if sms_text else "No SMS report generated.", language="text")


def _create_geojson_zip(run_dir: Optional[Path]) -> bytes:
    """Pack GeoJSON output layers into an in-memory zip file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if run_dir and (run_dir / "geojson").exists():
            for gfile in (run_dir / "geojson").glob("*.geojson"):
                zf.write(gfile, arcname=gfile.name)
            for jfile in (run_dir / "geojson").glob("*.json"):
                zf.write(jfile, arcname=jfile.name)
        else:
            zf.writestr("empty.txt", "No geojson layers available")
    buf.seek(0)
    return buf.getvalue()


def render_footer() -> None:
    """Render mandatory data and legal attributions on every page."""
    st.markdown(
        """
        <div style="font-size: 0.8rem; color: #64748B; border-top: 1px solid #1E293B; padding-top: 14px; margin-top: 40px; line-height: 1.5;">
          <div>Contains modified Copernicus Sentinel data 2026.</div>
          <div>Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.</div>
          <div>© OpenStreetMap contributors.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
