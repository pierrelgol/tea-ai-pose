from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from pipeline_config import build_layout, load_pipeline_config

from .dataset_index import index_dataset
from .geometry import run_geometry_checks
from .gui import launch_gui
from .integrity import run_integrity_checks
from .overlays import export_debug_overlays
from .predictions import run_prediction_checks
from .reports import write_reports

MAX_OUTLIER_RATE = 0.02
MAX_MEAN_CORNER_ERROR_PX = 1.5


def main() -> None:
    parser = argparse.ArgumentParser(description="Check augmented dataset integrity and geometry")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    dataset_name = str(shared.dataset.get("name") or shared.run["dataset"])
    dataset_root = shared.paths["dataset_root"] / str(shared.dataset.get("augmented_subdir", "augmented")) / dataset_name

    layout = build_layout(
        artifacts_root=shared.paths["artifacts_root"],
        model_key=str(shared.run["model_key"]),
        run_id=str(shared.run["run_id"]),
    )

    reports_dir = dataset_root / "reports"
    if reports_dir.exists():
        shutil.rmtree(reports_dir)

    cc = shared.checks
    records = index_dataset(dataset_root)
    integrity_issues, integrity_summary = run_integrity_checks(records)
    geometry_metrics, geometry_summary = run_geometry_checks(records, float(cc.get("outlier_threshold_px", 2.0)))
    model_reports = run_prediction_checks(records, layout.infer_root)

    export_debug_overlays(
        records=records,
        reports_dir=reports_dir,
        n_per_split=int(cc.get("debug_overlays_per_split", 10)),
        seed=int(cc.get("seed", shared.run["seed"])),
    )

    write_reports(
        reports_dir=reports_dir,
        integrity_issues=integrity_issues,
        integrity_summary=integrity_summary,
        geometry_metrics=geometry_metrics,
        geometry_summary=geometry_summary,
        model_reports=model_reports,
    )

    print(f"checked {len(records)} samples")
    print(f"integrity issues: {integrity_summary['total_issues']}")
    print(f"geometry outliers: {geometry_summary['num_outliers']}")

    failures: list[str] = []
    if integrity_summary.get("total_issues", 0) > 0:
        failures.append(f"integrity issues present: {integrity_summary['total_issues']}")
    if float(geometry_summary.get("outlier_rate", 0.0) or 0.0) > MAX_OUTLIER_RATE:
        failures.append(
            f"outlier_rate={geometry_summary.get('outlier_rate')} exceeds max={MAX_OUTLIER_RATE}"
        )
    mean_corner_error_px = geometry_summary.get("mean_corner_error_px")
    if mean_corner_error_px is not None and float(mean_corner_error_px) > MAX_MEAN_CORNER_ERROR_PX:
        failures.append(
            f"mean_corner_error_px={mean_corner_error_px} exceeds max={MAX_MEAN_CORNER_ERROR_PX}"
        )
    if failures:
        for failure in failures:
            print(f"CHECK FAILED: {failure}")
        raise SystemExit(2)

    if bool(cc.get("gui", False)):
        launch_gui(
            records=records,
            integrity_issues=integrity_issues,
            geometry_metrics=geometry_metrics,
            model_reports=model_reports,
        )


if __name__ == "__main__":
    main()
