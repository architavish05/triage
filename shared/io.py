"""IO helpers conforming to CONTRACT.md schemas for hazard_grid.npz, metadata.json, and GeoJSON."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


def save_hazard_grid(
    path: Union[str, Path],
    class_map: np.ndarray,
    cant_see_mask: np.ndarray,
    bbox: Union[Tuple[float, float, float, float], List[float], np.ndarray],
) -> Path:
    """Save compressed hazard grid NPZ containing class_map, cant_see_mask, and bbox."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        class_map=class_map,
        cant_see_mask=cant_see_mask,
        bbox=np.array(bbox, dtype=np.float64),
    )
    return path


def load_hazard_grid(path: Union[str, Path]) -> Dict[str, Any]:
    """Load hazard grid NPZ containing class_map, cant_see_mask, and bbox."""
    path = Path(path)
    with np.load(path) as data:
        return {
            "class_map": data["class_map"],
            "cant_see_mask": data["cant_see_mask"],
            "bbox": tuple(data["bbox"].tolist()),
        }


def save_metadata(
    path: Union[str, Path],
    bbox: Tuple[float, float, float, float],
    flood_date: str,
    area_name: Optional[str] = None,
    **extra: Any,
) -> Path:
    """Save metadata.json conforming to CONTRACT.md."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bbox": list(bbox),
        "flood_date": flood_date,
        "area_name": area_name or f"Disaster Scene ({flood_date})",
        **extra,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata.json conforming to CONTRACT.md."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_geojson(path: Union[str, Path], data: Any) -> Path:
    """Save GeoJSON FeatureCollection or JSON structure to file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_geojson(path: Union[str, Path]) -> Dict[str, Any]:
    """Load GeoJSON or JSON structure from file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
