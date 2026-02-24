from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import build_layout, load_pipeline_config

from .data import resolve_latest_weights
from .pipeline import GradingConfig, run_grading


def _fmt_float(v: float | None, digits: int = 4) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade detector runs from strict OBB geometry quality")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    dataset_name = str(shared.dataset.get("name") or shared.run["dataset"])
    dataset_root = shared.paths["dataset_root"] / str(shared.dataset.get("augmented_subdir", "augmented")) / dataset_name

    model_key = str(shared.run["model_key"])
    run_id = str(shared.run["run_id"])
    layout = build_layout(
        artifacts_root=shared.paths["artifacts_root"],
        model_key=model_key,
        run_id=run_id,
    )

    gc = shared.grade
    run_inference = bool(gc.get("run_inference", True))
    weights = resolve_latest_weights(shared.paths["artifacts_root"]) if run_inference else None

    result = run_grading(
        GradingConfig(
            dataset_root=dataset_root,
            predictions_root=layout.infer_root,
            artifacts_root=shared.paths["artifacts_root"],
            reports_dir=layout.grade_root / "reports",
            hard_examples_dir=layout.grade_root / "hard_examples",
            model=model_key,
            weights=weights,
            run_inference=run_inference,
            splits=[str(s) for s in gc.get("splits", ["val"])],
            imgsz=int(gc.get("imgsz", 640)),
            device=str(gc.get("device", "auto")),
            conf_threshold=float(gc.get("conf_threshold", 0.25)),
            calibrate_confidence=bool(gc.get("calibrate_confidence", True)),
            infer_iou_threshold=float(gc.get("infer_iou_threshold", 0.7)),
            match_iou_threshold=float(gc.get("match_iou_threshold", 0.5)),
            weights_json=Path(str(gc["weights_json"])) if gc.get("weights_json") else None,
            strict_obb=bool(gc.get("strict_obb", True)),
            max_samples=None if gc.get("max_samples") is None else int(gc.get("max_samples")),
            seed=int(shared.run["seed"]),
            calibration_candidates=(
                None
                if gc.get("calibration_candidates") is None
                else [float(v) for v in gc.get("calibration_candidates")]
            ),
        )
    )

    aggregate = result["aggregate"]
    run_det = aggregate.get("run_detection", {})

    print("Model Source")
    print(f"- resolved_model_key: {result['model_key']}")
    print(f"- weights: {result['weights_path'] or 'N/A (using existing predictions)'}")
    print(f"- predictions_root: {layout.infer_root / result['model_key'] / 'labels'}")
    print("")
    print("Inference")
    if result["inference"] is None:
        print("- executed: no")
        print("- reason: using existing predictions")
    else:
        print("- executed: yes")
        print(f"- device: {result['inference']['resolved_device']}")
        print(f"- images_processed: {result['inference']['images_processed']}")
        print(f"- label_files_written: {result['inference']['label_files_written']}")
    print("")
    print("Grading")
    print(f"- dataset_root: {dataset_root}")
    print(f"- samples_scored: {aggregate['num_samples_scored']}")
    print(f"- run_grade_0_100: {aggregate['run_grade_0_100']:.4f}")
    print(f"- run_precision_proxy: {_fmt_float(run_det.get('precision_proxy'))}")
    print(f"- run_recall_proxy: {_fmt_float(run_det.get('recall_proxy'))}")
    print(f"- run_miss_rate_proxy: {_fmt_float(run_det.get('miss_rate_proxy'))}")
    print(f"- run_class_match_rate: {_fmt_float(run_det.get('class_match_rate'))}")
    print("")
    print("Artifacts")
    print(f"- summary_json: {result['reports']['summary_json']}")
    print(f"- sample_jsonl: {result['reports']['sample_jsonl']}")
    print(f"- summary_md: {result['reports']['summary_md']}")


if __name__ == "__main__":
    main()
