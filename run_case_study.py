"""Case study execution script: Bhote Koshi - Trishuli River Corridor, Nepal (26 August 2026).

Executes the complete Triage pipeline on the disaster scene, followed by
post-hoc validation against Copernicus EMS EMSR927.
"""

import sys
from pathlib import Path
import logging

repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.pipeline import run
from src.eval.compare_emsr927 import compare_with_emsr927

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("case_study")


def main():
    logger.info("=================================================================")
    logger.info("TRIAGE CASE STUDY: Bhote Koshi - Trishuli Corridor (26 Aug 2026)")
    logger.info("=================================================================")

    # Geographic bounding box of the river corridor in Nepal
    bbox = (85.20, 27.80, 85.45, 28.15)
    flood_date = "2026-08-26"
    output_dir = Path("data/processed/trishuli_case_study_20260826")

    # Step 1: Run end-to-end Triage pipeline
    results = run(
        bbox=bbox,
        flood_date=flood_date,
        offline=False,
        output_dir=output_dir,
    )

    logger.info("=================================================================")
    logger.info("CASE STUDY PIPELINE EXECUTION COMPLETE")
    logger.info("=================================================================")

    damage_summary = results["damage"].summary
    print("\n--- INFRASTRUCTURE DAMAGE IMPACTS ---")
    print(f"Damaged Buildings:      {damage_summary.get('damaged_buildings_count', 0)}")
    print(f"Damaged Roads:          {damage_summary.get('damaged_roads_km', 0.0)} km")
    print(f"Impassable Road Cuts:   {damage_summary.get('impassable_roads_count', 0)}")
    print(f"Damaged Bridges:        {damage_summary.get('damaged_bridges_count', 0)}")
    print(f"Dry Islands Identified: {results['dry_islands_count']}")

    print("\n--- TOP RESCUE PRIORITIES ---")
    for item in results["rescue_priorities"][:5]:
        print(
            f"Rank #{item.rank}: {item.settlement_name} ({item.place_type}) | "
            f"Cut-off Prob: {item.cutoff_prob*100:.1f}% | "
            f"Trapped Risk: {item.trapped_risk*100:.1f}% | "
            f"At-Risk Bldgs: {item.buildings_at_risk} | "
            f"Stranded (Estimate): ~{item.stranded_population_estimate} | "
            f"Action: {item.recommended_action}"
        )

    # Step 2: Post-hoc validation against Copernicus EMS EMSR927 (CHECKING ONLY)
    logger.info("=================================================================")
    logger.info("POST-HOC VALIDATION: Copernicus EMS EMSR927 (Checking Only)")
    logger.info("=================================================================")
    eval_dir = output_dir / "eval_emsr927"
    eval_res = compare_with_emsr927(
        pipeline_class_map=results["class_map"],
        pipeline_confidence=results["confidence"],
        output_dir=eval_dir,
        threshold=0.35,
    )

    print("\n--- COPERNICUS EMS (EMSR927) VALIDATION METRICS ---")
    print(f"IoU (Intersection-over-Union): {eval_res.iou:.4f}")
    print(f"Precision:                     {eval_res.precision:.4f}")
    print(f"Recall:                        {eval_res.recall:.4f}")
    print(f"F1 Score:                      {eval_res.f1_score:.4f}")
    print(f"Side-by-side plot saved to:    {eval_res.side_by_side_plot_path}")
    print(f"Metrics JSON saved to:         {eval_res.metrics_json_path}")
    print("=================================================================\n")


if __name__ == "__main__":
    main()
