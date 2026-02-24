from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import build_layout, load_pipeline_config
from pipeline_runtime_utils import resolve_latest_weights_from_artifacts

from .config import InferConfig
from .infer import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detector inference and export YOLO predictions")
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

    model_value = str(shared.run["model"])
    model_path = Path(model_value)
    if model_path.exists() and model_path.suffix == ".pt":
        weights = model_path
    else:
        weights = resolve_latest_weights_from_artifacts(shared.paths["artifacts_root"])

    ic = shared.infer
    cfg = InferConfig(
        weights=weights,
        dataset_root=dataset_root,
        output_root=layout.infer_root,
        model_name=model_key,
        imgsz=int(ic.get("imgsz", 640)),
        device=str(ic.get("device", "auto")),
        conf_threshold=float(ic.get("conf_threshold", 0.25)),
        iou_threshold=float(ic.get("iou_threshold", 0.7)),
        seed=int(shared.run["seed"]),
        splits=[str(s) for s in ic.get("splits", ["val"])],
        save_empty=bool(ic.get("save_empty", True)),
        batch_size=int(ic.get("batch_size", 16)),
    )

    summary = run_inference(cfg)
    print(f"status: {summary['status']}")
    print(f"model: {summary['model_name']}")
    print(f"images_processed: {summary['images_processed']}")
    print(f"label_files_written: {summary['label_files_written']}")
    print(f"output_root: {summary['output_root']}")


if __name__ == "__main__":
    main()
