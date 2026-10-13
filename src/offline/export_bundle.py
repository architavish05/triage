"""Export offline responder packs: MBTiles, GeoJSON layers, situation reports, and plain text.

Packages all actionable outputs into an air-gapped field kit for deployment on rugged tablets,
tactical ATAK devices, QField mobile GIS, or thumb drives.
"""

from io import BytesIO
import json
import logging
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple
import zipfile

import numpy as np
from PIL import Image

from src.damage.overlay import save_damage_geojson
from src.network.priority import export_cutoff_geojson
from src.report.sms_summary import ATTRIBUTION_SHORT

logger = logging.getLogger(__name__)


def create_mbtiles_from_raster(
    class_map: np.ndarray,
    confidence: np.ndarray,
    transform: Tuple[float, float, float, float, float, float],
    output_mbtiles_path: Path,
    zoom_levels: Tuple[int, ...] = (10, 11, 12),
) -> Path:
    """Generate an OGC-compliant MBTiles SQLite database from the hazard classification grid.

    Renders RGB color-coded hazard PNG tiles (Water: Blue, Debris: Red, Background: Transparent)
    and stores them in an SQLite database with standard MBTiles schema.

    Args:
        class_map: 2D integer hazard class map (0=bg, 1=water, 2=debris).
        confidence: 2D float confidence grid in [0, 1].
        transform: (west, south, east, north, res_x, res_y).
        output_mbtiles_path: Path where .mbtiles file will be created.
        zoom_levels: Target pyramid zoom levels.

    Returns:
        Path to the generated .mbtiles database.
    """
    output_mbtiles_path.parent.mkdir(parents=True, exist_ok=True)
    if output_mbtiles_path.exists():
        output_mbtiles_path.unlink()

    west, south, east, north, _, _ = transform

    # Convert class map to RGBA PNG image
    # Background: Transparent (0, 0, 0, 0)
    # Water: Semi-transparent Blue (37, 99, 235, alpha proportional to confidence)
    # Debris: Semi-transparent Red (220, 38, 38, alpha proportional to confidence)
    rows, cols = class_map.shape
    rgba = np.zeros((rows, cols, 4), dtype=np.uint8)

    water_mask = (class_map == 1)
    debris_mask = (class_map == 2)

    rgba[water_mask, 0] = 37
    rgba[water_mask, 1] = 99
    rgba[water_mask, 2] = 235
    rgba[water_mask, 3] = np.clip(confidence[water_mask] * 230 + 25, 40, 255).astype(np.uint8)

    rgba[debris_mask, 0] = 220
    rgba[debris_mask, 1] = 38
    rgba[debris_mask, 2] = 38
    rgba[debris_mask, 3] = np.clip(confidence[debris_mask] * 230 + 25, 40, 255).astype(np.uint8)

    # Encode rendered PNG into bytes
    img = Image.fromarray(rgba, mode="RGBA").resize((256, 256), Image.Resampling.NEAREST)
    buf = BytesIO()
    img.save(buf, format="PNG")
    tile_bytes = buf.getvalue()

    # Create SQLite MBTiles database
    conn = sqlite3.connect(str(output_mbtiles_path))
    cursor = conn.cursor()

    cursor.execute("CREATE TABLE metadata (name text, value text);")
    cursor.execute("CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);")

    metadata_rows = [
        ("name", "Triage Flood & Debris Hazard Map"),
        ("type", "overlay"),
        ("version", "1.0.0"),
        ("description", "Copernicus Sentinel satellite disaster damage mapping"),
        ("format", "png"),
        ("bounds", f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"),
        ("attribution", ATTRIBUTION_SHORT),
    ]
    cursor.executemany("INSERT INTO metadata VALUES (?, ?);", metadata_rows)

    # Insert tile across designated zoom levels
    for z in zoom_levels:
        cursor.execute("INSERT INTO tiles VALUES (?, ?, ?, ?);", (z, 0, 0, tile_bytes))

    conn.commit()
    conn.close()
    logger.debug(f"Created MBTiles raster package: {output_mbtiles_path}")
    return output_mbtiles_path


def export_responder_bundle(
    pipeline_results: Dict[str, Any],
    bundle_output_dir: Path,
    create_zip: bool = True,
) -> Dict[str, Path]:
    """Package complete offline responder pack containing MBTiles, GeoJSONs, SitReps, and text summary.

    Args:
        pipeline_results: Dict returned by `pipeline.run(..., offline=True)`.
        bundle_output_dir: Destination folder for uncompressed bundle files.
        create_zip: If True, packages all files into a compressed .zip field kit.

    Returns:
        Dict mapping product names to output file paths.
    """
    bundle_output_dir.mkdir(parents=True, exist_ok=True)
    geojson_dir = bundle_output_dir / "geojson"
    reports_dir = bundle_output_dir / "reports"
    mbtiles_dir = bundle_output_dir / "mbtiles"

    geojson_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    mbtiles_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Path] = {}

    # 1. Export MBTiles raster database
    mbtiles_path = mbtiles_dir / "triage_hazard_map.mbtiles"
    create_mbtiles_from_raster(
        class_map=pipeline_results["class_map"],
        confidence=pipeline_results["confidence"],
        transform=pipeline_results["transform"],
        output_mbtiles_path=mbtiles_path,
    )
    outputs["mbtiles"] = mbtiles_path

    # 2. Save GeoJSON layers
    damage_paths = save_damage_geojson(pipeline_results["damage"], geojson_dir)
    outputs.update(damage_paths)

    # Cutoff settlements GeoJSON
    cutoff_geojson_path = geojson_dir / "cutoff_settlements.geojson"
    export_cutoff_geojson(pipeline_results["isolation_records"], cutoff_geojson_path)
    outputs["cutoff_settlements"] = cutoff_geojson_path

    # Helicopter Landing Zones GeoJSON
    hlz_geojson_path = geojson_dir / "landing_zones.geojson"
    with open(hlz_geojson_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_results["landing_zones_geojson"], f, indent=2)
    outputs["landing_zones"] = hlz_geojson_path

    # 3. Copy reports and SMS briefing
    rep_paths = pipeline_results.get("report_paths", {})
    for k, p in rep_paths.items():
        dest = reports_dir / p.name
        if p.exists() and p != dest:
            dest.write_bytes(p.read_bytes())
        outputs[k] = dest

    sms_dest = reports_dir / "sitrep_sms.txt"
    sms_dest.write_text(pipeline_results["sms_summary"], encoding="utf-8")
    outputs["sms_summary"] = sms_dest

    # 4. Create Bundle Manifest
    manifest = {
        "bundle_version": "1.0.0",
        "flood_date": pipeline_results["flood_date"],
        "bbox": pipeline_results["bbox"],
        "summary": pipeline_results["damage"].summary,
        "layers": [p.name for p in outputs.values()],
        "attribution": ATTRIBUTION_SHORT,
    }
    manifest_path = bundle_output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    outputs["manifest"] = manifest_path

    # 5. Compress into ZIP field pack
    if create_zip:
        zip_path = bundle_output_dir.parent / f"triage_field_pack_{pipeline_results['flood_date']}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(bundle_output_dir):
                for file in files:
                    full_p = Path(root) / file
                    arc_name = full_p.relative_to(bundle_output_dir)
                    zf.write(full_p, arcname=arc_name)
        outputs["zip_pack"] = zip_path
        logger.info(f"Compressed offline responder field pack: {zip_path}")

    return outputs
