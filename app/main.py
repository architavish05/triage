"""Triage | Flood damage from satellite data.

Decision-support application for disaster response and evacuation prioritization.
"""

import sys
import os
from pathlib import Path
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import streamlit as st
from streamlit_folium import st_folium

# Prepend project root and app directory to sys.path so src and components imports succeed
app_dir = Path(__file__).resolve().parent
repo_root = app_dir.parent
for p in (str(repo_root), str(app_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.pipeline import run

try:
    from app.components import (
        build_folium_map,
        load_run_from_folder,
        render_footer,
        render_header,
        render_kpi_row,
        render_priority_villages_table,
        render_tabs,
    )
except (ImportError, ModuleNotFoundError):
    from components import (
        build_folium_map,
        load_run_from_folder,
        render_footer,
        render_header,
        render_kpi_row,
        render_priority_villages_table,
        render_tabs,
    )

# Page configuration
st.set_page_config(
    page_title="Triage",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _detect_cli_run_dir() -> Optional[str]:
    """Check CLI arguments or environment variables for a run folder."""
    env_dir = os.environ.get("TRIAGE_RUN_DIR")
    if env_dir and Path(env_dir).exists():
        return env_dir

    for idx, arg in enumerate(sys.argv):
        if arg in ("--run-dir", "-r") and idx + 1 < len(sys.argv):
            candidate = sys.argv[idx + 1]
            if Path(candidate).exists():
                return candidate
        if arg.startswith("--run-dir="):
            candidate = arg.split("=", 1)[1]
            if Path(candidate).exists():
                return candidate
        # Positional folder argument (e.g. `streamlit run app/main.py data/mock_run/`)
        clean_arg = arg.rstrip("/\\")
        if (
            arg not in ("streamlit", "run")
            and not arg.endswith(".py")
            and Path(clean_arg).exists()
            and Path(clean_arg).is_dir()
        ):
            return clean_arg
    return None


def _normalize_data(raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize outputs whether from pipeline in-memory execution or cached run folder."""
    if "damage" in raw_data:
        # From in-memory run(...)
        damage = raw_data["damage"]
        summary = damage.summary
        bldgs = damage.damaged_buildings
        roads = damage.damaged_roads
        bridges = damage.damaged_bridges
        priorities = raw_data.get("rescue_priorities", [])
        hlz_geojson = raw_data.get("landing_zones_geojson", {"type": "FeatureCollection", "features": []})
        trace_coords = raw_data["downstream_trace"].path_coords if raw_data.get("downstream_trace") else []
        arrivals = raw_data.get("settlement_arrivals", [])
        dry_islands = raw_data.get("dry_islands_count", 0)
        sms_text = raw_data.get("sms_summary", "")
        run_dir = raw_data.get("output_dir", None)
        class_map = raw_data.get("class_map")
        cant_see_mask = raw_data.get("cant_see_mask")
        bbox = raw_data.get("bbox", (85.20, 27.80, 85.45, 28.15))
        flood_date = raw_data.get("flood_date", "2026-08-26")
        area_name = "Bhote Koshi - Trishuli Corridor, Nepal"
    else:
        # From load_run_from_folder
        summary = raw_data.get("summary", {})
        bldgs = raw_data.get("damaged_buildings", {"type": "FeatureCollection", "features": []})
        roads = raw_data.get("damaged_roads", {"type": "FeatureCollection", "features": []})
        bridges = raw_data.get("damaged_bridges", {"type": "FeatureCollection", "features": []})
        priorities = raw_data.get("priorities", [])
        hlz_geojson = raw_data.get("landing_zones_geojson", {"type": "FeatureCollection", "features": []})
        trace_coords = raw_data.get("trace_coords", [])
        arrivals = []
        dry_islands = summary.get("dry_islands_count", 2)
        sms_text = raw_data.get("sms_text", "")
        run_dir = raw_data.get("run_dir", None)
        class_map = raw_data.get("class_map")
        cant_see_mask = raw_data.get("cant_see_mask")
        bbox = raw_data.get("bbox", (85.20, 27.80, 85.45, 28.15))
        flood_date = raw_data.get("flood_date", "2026-08-26")
        area_name = raw_data.get("area_name", "Bhote Koshi - Trishuli Corridor, Nepal")

    return {
        "summary": summary,
        "damaged_buildings": bldgs,
        "damaged_roads": roads,
        "damaged_bridges": bridges,
        "priorities": priorities,
        "landing_zones_geojson": hlz_geojson,
        "trace_coords": trace_coords,
        "settlement_arrivals": arrivals,
        "dry_islands_count": dry_islands,
        "sms_text": sms_text,
        "run_dir": run_dir,
        "class_map": class_map,
        "cant_see_mask": cant_see_mask,
        "bbox": bbox,
        "flood_date": flood_date,
        "area_name": area_name,
    }


def main() -> None:
    # Check CLI directory on first load
    cli_dir = _detect_cli_run_dir()
    if cli_dir and "pipeline_results" not in st.session_state:
        try:
            cached_run = load_run_from_folder(cli_dir)
            st.session_state["pipeline_results"] = cached_run
            st.session_state["last_run_info"] = {
                "area_name": cached_run.get("area_name", "Local Run"),
                "flood_date": cached_run.get("flood_date", "2026-08-26"),
            }
        except Exception:
            pass

    # -------------------------------------------------------------
    # Sidebar Controls (exact requested order)
    # -------------------------------------------------------------
    # 1. Area section
    st.sidebar.markdown("### Area")
    preset_options = [
        "Bhote Koshi - Trishuli Corridor, Nepal",
        "Custom bounding box",
        "Mock run (data/mock_run)",
    ]
    preset = st.sidebar.selectbox("Scenario preset", preset_options)

    if preset == "Bhote Koshi - Trishuli Corridor, Nepal":
        default_bbox = (85.20, 27.80, 85.45, 28.15)
        default_date = "2026-08-26"
        area_label = "Bhote Koshi - Trishuli Corridor, Nepal"
    elif preset == "Mock run (data/mock_run)":
        default_bbox = (85.20, 27.80, 85.45, 28.15)
        default_date = "2026-08-26"
        area_label = "Mock Run (Trishuli Corridor)"
    else:
        default_bbox = (85.10, 27.70, 85.50, 28.20)
        default_date = "2026-08-26"
        area_label = "Custom Area"

    # Bbox in two rows: West/South, East/North
    col_w, col_s = st.sidebar.columns(2)
    min_lon = col_w.number_input("West", value=default_bbox[0], format="%.4f")
    min_lat = col_s.number_input("South", value=default_bbox[1], format="%.4f")

    col_e, col_n = st.sidebar.columns(2)
    max_lon = col_e.number_input("East", value=default_bbox[2], format="%.4f")
    max_lat = col_n.number_input("North", value=default_bbox[3], format="%.4f")

    # 2. Event section
    st.sidebar.markdown("### Event")
    flood_date = st.sidebar.text_input("Flood date", value=default_date)

    # 3. Mode section
    st.sidebar.markdown("### Mode")
    offline_mode = st.sidebar.checkbox("Offline mode (use cached data only)", value=True)

    # 4. Full-width primary button
    run_analysis = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

    # Execution trigger
    if run_analysis:
        bbox = (min_lon, min_lat, max_lon, max_lat)
        if preset == "Mock run (data/mock_run)" and Path("data/mock_run").exists():
            with st.status("Loading cached mock run...", expanded=True) as status:
                status.write("Fetching data")
                status.write("Detecting flood")
                status.write("Terrain filter")
                status.write("Damage overlay")
                status.write("Cut-off analysis")
                status.write("Rescue priority")
                status.write("Report")
                results = load_run_from_folder("data/mock_run")
                status.update(label="Analysis complete", state="complete", expanded=False)
        else:
            with st.status("Executing disaster intelligence pipeline...", expanded=True) as status:
                status.write("Fetching data")
                time.sleep(0.04)
                status.write("Detecting flood")
                time.sleep(0.04)
                status.write("Terrain filter")
                time.sleep(0.04)
                status.write("Damage overlay")
                time.sleep(0.04)
                status.write("Cut-off analysis")
                time.sleep(0.04)
                status.write("Rescue priority")
                time.sleep(0.04)
                status.write("Report")

                out_dir = Path("data/processed") / f"{flood_date}_{min_lon:.2f}_{min_lat:.2f}"
                results = run(bbox=bbox, flood_date=flood_date, offline=offline_mode, output_dir=out_dir)

                # Persist cached rasters & geojsons for folder retrieval
                try:
                    np.savez_compressed(
                        out_dir / "hazard_grid.npz",
                        class_map=results["class_map"],
                        cant_see_mask=results["cant_see_mask"],
                        bbox=np.array(results["bbox"]),
                    )
                    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
                        json.dump({
                            "bbox": list(results["bbox"]),
                            "flood_date": results["flood_date"],
                            "area_name": area_label,
                        }, f)
                    with open(out_dir / "geojson" / "landing_zones.geojson", "w", encoding="utf-8") as f:
                        json.dump(results["landing_zones_geojson"], f)
                    if results.get("downstream_trace"):
                        with open(out_dir / "geojson" / "downstream_trace.json", "w", encoding="utf-8") as f:
                            json.dump(results["downstream_trace"].path_coords, f)
                    if results.get("rescue_priorities"):
                        with open(out_dir / "geojson" / "rescue_priorities.json", "w", encoding="utf-8") as f:
                            json.dump([p.__dict__ for p in results["rescue_priorities"]], f)
                except Exception:
                    pass

                status.update(label="Analysis complete", state="complete", expanded=False)

        st.session_state["pipeline_results"] = results
        st.session_state["last_run_info"] = {
            "area_name": area_label,
            "flood_date": flood_date,
        }

    # -------------------------------------------------------------
    # Main Dashboard Body
    # -------------------------------------------------------------
    last_info = st.session_state.get("last_run_info", {})
    render_header(
        area_name=last_info.get("area_name"),
        flood_date=last_info.get("flood_date"),
    )

    if "pipeline_results" not in st.session_state:
        # Empty state before first run
        st.info("Select an area and date, then click Run analysis")
        render_footer()
        return

    # Normalized data extraction
    data = _normalize_data(st.session_state["pipeline_results"])

    # KPI Row
    render_kpi_row(
        summary=data["summary"],
        priorities=data["priorities"],
        dry_islands_count=data["dry_islands_count"],
    )

    # Main Area: Map (65%) and Priority Villages Table (35%)
    col_map, col_table = st.columns([65, 35])

    with col_map:
        m = build_folium_map(
            bbox=data["bbox"],
            class_map=data["class_map"],
            cant_see_mask=data["cant_see_mask"],
            damaged_buildings=data["damaged_buildings"],
            damaged_roads=data["damaged_roads"],
            damaged_bridges=data["damaged_bridges"],
            priorities=data["priorities"],
            landing_zones_geojson=data["landing_zones_geojson"],
            downstream_coords=data["trace_coords"],
        )
        st_folium(
            m,
            height=620,
            use_container_width=True,
            returned_objects=[],
            key="triage_tactical_map",
        )

    with col_table:
        render_priority_villages_table(priorities=data["priorities"])

    # Detail Tabs
    render_tabs(
        summary=data["summary"],
        damaged_buildings=data["damaged_buildings"],
        damaged_roads=data["damaged_roads"],
        damaged_bridges=data["damaged_bridges"],
        landing_zones_geojson=data["landing_zones_geojson"],
        downstream_coords=data["trace_coords"],
        settlement_arrivals=data["settlement_arrivals"],
        sms_text=data["sms_text"],
        run_dir=data["run_dir"],
        flood_date=data["flood_date"],
    )

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
