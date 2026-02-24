from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import build_layout, load_pipeline_config

from .app import launch_gui
from .data import index_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Visual checker for detector predictions vs ground truth")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    dataset_name = str(shared.dataset.get("name") or shared.run["dataset"])
    dataset_root = shared.paths["dataset_root"] / str(shared.dataset.get("augmented_subdir", "augmented")) / dataset_name

    model_key = str(shared.run["model_key"])
    layout = build_layout(
        artifacts_root=shared.paths["artifacts_root"],
        model_key=model_key,
        run_id=str(shared.run["run_id"]),
    )

    split = str(shared.review.get("split", "val"))
    samples = index_samples(dataset_root=dataset_root, splits=[split])
    if not samples:
        raise RuntimeError(f"no samples found in dataset: {dataset_root}")

    print(f"dataset_root: {dataset_root}")
    print(f"predictions_root: {layout.infer_root}")
    print(f"model_key: {model_key}")
    print(f"samples: {len(samples)}")

    launch_gui(
        samples=samples,
        predictions_root=layout.infer_root,
        model_name=model_key,
        conf_threshold=float(shared.review.get("conf_threshold", 0.25)),
    )


if __name__ == "__main__":
    main()
