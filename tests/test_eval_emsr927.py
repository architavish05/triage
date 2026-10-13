"""Unit tests for Phase 11 EMSR927 validation script."""

from pathlib import Path
import numpy as np
import pytest

from src.eval.compare_emsr927 import (
    compare_with_emsr927,
    generate_emsr927_ground_truth,
)


def test_generate_emsr927_ground_truth():
    """Verify EMSR927 ground truth generation."""
    shape = (64, 64)
    gt = generate_emsr927_ground_truth(shape, river_corridor_center=32)

    assert gt.shape == shape
    assert gt.dtype == bool
    assert np.any(gt)
    # The flood corridor center should be True
    assert np.count_nonzero(gt) > 100


def test_compare_with_emsr927(tmp_path):
    """Verify post-hoc comparison produces valid metrics and visualization plot."""
    shape = (60, 60)
    class_map = np.zeros(shape, dtype=np.uint8)
    confidence = np.zeros(shape, dtype=np.float32)

    # Reference ground truth
    gt = generate_emsr927_ground_truth(shape, river_corridor_center=30)

    # Prediction matches 85% of ground truth with small FP and FN
    class_map[gt] = 1
    confidence[gt] = 0.85
    # Small false positive
    class_map[0:5, 0:5] = 1
    confidence[0:5, 0:5] = 0.70

    out_dir = tmp_path / "eval_out"
    eval_res = compare_with_emsr927(
        pipeline_class_map=class_map,
        pipeline_confidence=confidence,
        output_dir=out_dir,
        reference_gt_mask=gt,
        threshold=0.35,
    )

    assert 0.0 <= eval_res.iou <= 1.0
    assert eval_res.iou > 0.60
    assert eval_res.precision > 0.60
    assert eval_res.recall > 0.90
    assert eval_res.side_by_side_plot_path.exists()
    assert eval_res.metrics_json_path.exists()
