from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline_config import build_layout, load_pipeline_config

from .config import GeneratorConfig
from .generator import generate_dataset


def load_dataset_config(configs_root: Path, dataset_name: str) -> dict:
    config_path = configs_root / f"{dataset_name}.json"
    if not config_path.exists():
        raise FileNotFoundError(f"dataset config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic augmented dataset")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    dataset_root = shared.paths["dataset_root"]
    dataset_name = str(shared.dataset.get("name") or shared.run["dataset"])

    dataset_config = load_dataset_config(shared.paths["configs_root"], dataset_name)
    splits = dataset_config.get("splits", {})
    train_rel = splits.get("train_images_rel", "images/train")
    val_rel = splits.get("val_images_rel", "images/val")

    background_splits = {
        "train": dataset_root / dataset_name / train_rel,
        "val": dataset_root / dataset_name / val_rel,
    }

    layout = build_layout(
        artifacts_root=shared.paths["artifacts_root"],
        model_key=str(shared.run["model_key"]),
        run_id=str(shared.run["run_id"]),
    )

    output_root = dataset_root / str(shared.dataset.get("augmented_subdir", "augmented")) / dataset_name

    config = GeneratorConfig(
        background_splits=background_splits,
        background_dataset_name=dataset_name,
        target_images_dir=dataset_root / "targets" / "images",
        target_labels_dir=dataset_root / "targets" / "labels",
        target_classes_file=dataset_root / "targets" / "classes.txt",
        output_root=output_root,
        seed=int(shared.generator.get("seed", shared.run["seed"])),
        hard_examples_path=layout.grade_root / "hard_examples" / "latest.jsonl",
    )

    results = generate_dataset(config)
    print(f"generated {len(results)} synthetic samples in {config.output_root}")


if __name__ == "__main__":
    main()
