"""Triage end-to-end disaster mapping pipeline orchestrator.

Primary entry point:
    run(bbox, flood_date, offline=False)
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Phase 1: Fetch
from p1_data.fetch import (
    DEMData,
    OSMData,
    S1DataPair,
    S2Data,
    fetch_dem,
    fetch_osm,
    fetch_s1,
    fetch_s2,
)

# Phase 2 & 8: Detection & U-Net
from p2_ai.detect import (
    detect_debris_amplitude_fallback,
    detect_water_sar,
    fuse_detections,
    predict_unet,
)

# Phase 3: Terrain filter
from p3_terrain.terrain import (
    build_cant_see_mask,
    compute_hand,
    compute_radar_shadow_layover,
    compute_slope,
    filter_impossible_flood,
)

# Phase 4: Damage overlay
from p4_app.damage import (
    DamageOverlayResult,
    compute_damage_overlay,
    save_damage_geojson,
)

# Phase 5: Road graph cut-off
from p4_app.network import (
    build_cutoff_table,
    build_road_graph,
    export_cutoff_geojson,
    run_monte_carlo_cutoff,
)

# Phase 6: Flowpath trace
from p3_terrain.flowpath import (
    calculate_settlement_arrival_times,
    compute_d8_flow,
    fill_depressions,
    trace_downstream_path,
)

# Phase 7: Rescue intelligence
from p3_terrain.rescue import (
    compute_high_ground_mask,
    compute_rescue_priorities,
    detect_landing_zones,
    evaluate_settlement_escape_time,
    export_rescue_priority_json,
    identify_dry_islands,
)

# Phase 9: Reporting
from p4_app.report import (
    generate_situation_reports,
    generate_sms_summary,
)

logger = logging.getLogger(__name__)


def run(
    bbox: Tuple[float, float, float, float],
    flood_date: str,
    offline: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute end-to-end Triage disaster intelligence pipeline.

    Takes raw satellite data for ANY area and date and produces:
    1. Where the flood hit (water + debris map with per-pixel confidence)
    2. What was damaged (OSM buildings, roads, bridges with hit probability)
    3. Who is cut off (settlements with cut-off probability via road graph)
    4. Who needs help first (ranked trapped settlements with helicopter landing zones)
    5. A dashboard, one-page PDF situation report (English + Nepali), and offline pack.

    Args:
        bbox: Bounding box tuple (min_lon, min_lat, max_lon, max_lat) in WGS84.
        flood_date: ISO date string for flood event (e.g. '2026-08-26').
        offline: If True, operates strictly from local pre-cached assets.
        output_dir: Directory where GeoJSONs and reports are saved.

    Returns:
        Consolidated dictionary of all hazard grids, damage tallies, priorities, and reports.
    """
    logger.info(f"=== Starting Triage Pipeline execution for {flood_date} within {bbox} ===")
    out_dir = output_dir or (Path("data/processed") / f"{flood_date}_{bbox[0]:.2f}_{bbox[1]:.2f}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Fetch & Cache (Phase 1)
    # -------------------------------------------------------------
    logger.info("[1/7] Ingesting Earth observation and OSM infrastructure...")
    s1_data: S1DataPair = fetch_s1(bbox, flood_date, offline=offline, use_synthetic_fallback=True)
    s2_data: S2Data = fetch_s2(bbox, flood_date, offline=offline, use_synthetic_fallback=True)
    dem_data: DEMData = fetch_dem(bbox, flood_date, offline=offline, use_synthetic_fallback=True)
    osm_data: OSMData = fetch_osm(bbox, snapshot_date="2026-07-27", offline=offline, use_synthetic_fallback=True)

    # -------------------------------------------------------------
    # 2. Hazard Detection & Sensor Fusion (Phase 2 & Phase 8)
    # -------------------------------------------------------------
    logger.info("[2/7] Running SAR change, InSAR coherence fallback, and U-Net deep learning...")
    slope_raw = compute_slope(dem_data.elevation, res_meters=30.0)

    # Rule-based SAR water detection
    sar_water_res = detect_water_sar(
        s1_data.post_vv,
        s1_data.pre_vv,
        already_db=True,
    )
    # Debris detection via amplitude/texture roughness surge
    debris_res = detect_debris_amplitude_fallback(
        s1_data.post_vv,
        s1_data.pre_vv,
    )
    # 5-Channel U-Net Deep Learning prediction
    unet_res = predict_unet(
        s1_data.pre_vv,
        s1_data.pre_vh,
        s1_data.post_vv,
        s1_data.post_vh,
        slope_raw,
        threshold=0.50,
    )
    # 4-Way Fusion (U-Net + SAR rule + Coherence + S2 NDWI if clear)
    optical_ndwi = s2_data.ndwi if not s2_data.metadata.cloud_skipped else None
    fused_res = fuse_detections(
        water_result=sar_water_res,
        debris_result=debris_res,
        optical_ndwi=optical_ndwi,
        unet_prob=unet_res.probability_map,
    )

    # -------------------------------------------------------------
    # 3. Terrain Filtering & Occlusion Modeling (Phase 3)
    # -------------------------------------------------------------
    logger.info("[3/7] Filtering impossible terrain (slope, HAND) & modeling radar shadow...")
    hand_grid, drainage_mask = compute_hand(dem_data.elevation, slope=slope_raw, res_meters=30.0)
    shadow_mask, layover_mask = compute_radar_shadow_layover(
        dem_data.elevation,
        res_meters=30.0,
        orbit_direction=s1_data.metadata.orbit_direction,
    )
    cloud_mask = np.zeros(fused_res.shape, dtype=bool) if s2_data.metadata.cloud_skipped else None
    occlusion_res = build_cant_see_mask(
        shadow_mask=shadow_mask,
        layover_mask=layover_mask,
        cloud_mask=cloud_mask,
    )

    # Apply physical hydrological constraints
    terrain_filtered = filter_impossible_flood(
        class_map=fused_res.class_map,
        confidence=fused_res.confidence,
        slope=slope_raw,
        hand=hand_grid,
        shadow_mask=shadow_mask,
        max_flood_slope_deg=10.0,
        max_flood_hand_m=22.0,
    )

    # -------------------------------------------------------------
    # 4. Infrastructure Damage Overlay (Phase 4)
    # -------------------------------------------------------------
    logger.info("[4/7] Intersecting hazard layers with OSM buildings, roads, and bridges...")
    damage_res: DamageOverlayResult = compute_damage_overlay(
        osm_data=osm_data,
        class_map=terrain_filtered.filtered_class_map,
        confidence=terrain_filtered.filtered_confidence,
        transform=dem_data.transform,
        damage_threshold=0.35,
    )
    damage_paths = save_damage_geojson(damage_res, out_dir / "geojson")

    # -------------------------------------------------------------
    # 5. Road Graph Cut-off Analysis (Phase 5)
    # -------------------------------------------------------------
    logger.info("[5/7] Performing Monte Carlo road network cut-off simulations (N=200)...")
    road_graph = build_road_graph(damage_res.damaged_roads)
    isolation_records = run_monte_carlo_cutoff(
        road_graph=road_graph,
        settlements_geojson=osm_data.places,
        hospitals_geojson=osm_data.hospitals,
        num_simulations=200,
        random_seed=42,
    )
    ranked_cutoff_table = build_cutoff_table(isolation_records)
    export_cutoff_geojson(isolation_records, out_dir / "geojson" / "cutoff_settlements.geojson")

    # -------------------------------------------------------------
    # 6. Downstream Flood Path Tracing (Phase 6)
    # -------------------------------------------------------------
    logger.info("[6/7] Tracing downstream flood routing and settlement arrival times...")
    filled_dem = fill_depressions(dem_data.elevation)
    d8_dir, flow_accum = compute_d8_flow(filled_dem, res_meters=30.0)

    # Use peak/top-center coordinates as default initiation breach point
    start_lon = (bbox[0] + bbox[2]) / 2.0
    start_lat = bbox[3] - 0.05
    downstream_trace = trace_downstream_path(
        d8_dir=d8_dir,
        filled_dem=filled_dem,
        start_coords=(start_lon, start_lat),
        transform=dem_data.transform,
        res_meters=30.0,
    )
    settlement_arrivals = calculate_settlement_arrival_times(
        trace=downstream_trace,
        settlements_geojson=osm_data.places,
        buffer_meters=800.0,
    )

    # -------------------------------------------------------------
    # 7. Rescue Prioritization, Dry Islands & Landing Zones (Phase 7)
    # -------------------------------------------------------------
    logger.info("[7/7] Identifying dry islands, escape times, and helicopter landing zones...")
    dry_island_mask, island_props = identify_dry_islands(
        class_map=terrain_filtered.filtered_class_map,
        hand=hand_grid,
        safe_hand_threshold=25.0,
    )

    high_ground_mask = compute_high_ground_mask(
        hand=hand_grid,
        slope=slope_raw,
        class_map=terrain_filtered.filtered_class_map,
    )

    # Foot evacuation feasibility checks
    escape_checks = {}
    for arr in settlement_arrivals:
        check = evaluate_settlement_escape_time(
            settlement_lon=arr.coordinates[0],
            settlement_lat=arr.coordinates[1],
            settlement_name=arr.name,
            arrival_time_mins=arr.arrival_time_mins,
            high_ground_mask=high_ground_mask,
            transform=dem_data.transform,
        )
        escape_checks[arr.name] = check

    # Helicopter Landing Zones
    hlz_candidates, hlz_geojson = detect_landing_zones(
        slope=slope_raw,
        class_map=terrain_filtered.filtered_class_map,
        cant_see_mask=occlusion_res.cant_see_mask,
        transform=dem_data.transform,
        settlements_geojson=osm_data.places,
        max_slope_deg=5.0,
    )

    # Multi-factor rescue priority ranking
    rescue_priorities = compute_rescue_priorities(
        isolation_records=isolation_records,
        damaged_buildings_geojson=damage_res.damaged_buildings,
        dry_island_mask=dry_island_mask,
        transform=dem_data.transform,
        escape_checks=escape_checks,
        household_size=5.0,
    )

    # -------------------------------------------------------------
    # 8. Multi-Modal Reports & SMS Summary (Phase 9)
    # -------------------------------------------------------------
    report_paths = generate_situation_reports(
        flood_date=flood_date,
        bbox=bbox,
        damage_summary=damage_res.summary,
        priority_items=rescue_priorities,
        output_dir=out_dir / "reports",
        s1_metadata=s1_data.metadata,
        s2_metadata=s2_data.metadata,
        cant_see_breakdown=occlusion_res.cant_see_breakdown,
        dry_islands_count=len(island_props),
    )

    sms_summary_text = generate_sms_summary(
        flood_date=flood_date,
        bbox=bbox,
        damage_summary=damage_res.summary,
        priority_items=rescue_priorities,
        landing_zones=hlz_candidates,
        output_path=out_dir / "reports" / "sitrep_sms.txt",
    )

    # -------------------------------------------------------------
    # Persist all CONTRACT.md outputs to run directory
    # -------------------------------------------------------------
    import json
    geojson_dir = out_dir / "geojson"
    geojson_dir.mkdir(parents=True, exist_ok=True)
    with open(geojson_dir / "landing_zones.geojson", "w", encoding="utf-8") as f:
        json.dump(hlz_geojson, f, indent=2)
    with open(geojson_dir / "downstream_trace.json", "w", encoding="utf-8") as f:
        json.dump(downstream_trace.path_coords, f, indent=2)
    export_rescue_priority_json(rescue_priorities, geojson_dir / "rescue_priorities.json")
    np.savez_compressed(
        out_dir / "hazard_grid.npz",
        class_map=terrain_filtered.filtered_class_map,
        cant_see_mask=occlusion_res.cant_see_mask,
        bbox=np.array(bbox, dtype=np.float64),
    )
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({
            "bbox": list(bbox),
            "flood_date": flood_date,
            "area_name": f"Disaster Scene ({flood_date})",
        }, f, indent=2)

    logger.info("=== Triage Pipeline successfully completed! ===")

    return {
        "bbox": bbox,
        "flood_date": flood_date,
        "class_map": terrain_filtered.filtered_class_map,
        "confidence": terrain_filtered.filtered_confidence,
        "cant_see_mask": occlusion_res.cant_see_mask,
        "slope": slope_raw,
        "hand": hand_grid,
        "transform": dem_data.transform,
        "damage": damage_res,
        "isolation_records": isolation_records,
        "ranked_cutoff_table": ranked_cutoff_table,
        "downstream_trace": downstream_trace,
        "settlement_arrivals": settlement_arrivals,
        "dry_island_mask": dry_island_mask,
        "dry_islands_count": len(island_props),
        "landing_zones": hlz_candidates,
        "landing_zones_geojson": hlz_geojson,
        "rescue_priorities": rescue_priorities,
        "report_paths": report_paths,
        "sms_summary": sms_summary_text,
        "output_dir": out_dir,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Triage disaster mapping pipeline")
    parser.add_argument("--bbox", nargs=4, type=float, default=[85.20, 27.80, 85.45, 28.15], help="min_lon min_lat max_lon max_lat")
    parser.add_argument("--date", type=str, default="2026-08-26", help="Flood date YYYY-MM-DD")
    parser.add_argument("--offline", action="store_true", help="Offline mode")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()
    out = Path(args.output_dir) if args.output_dir else None
    run(bbox=tuple(args.bbox), flood_date=args.date, offline=args.offline, output_dir=out)

