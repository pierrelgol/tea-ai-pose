from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from pipeline_config import load_pipeline_config

from .app import run_app


def main() -> None:
    parser = argparse.ArgumentParser(description="GUI tool for labeling target images in YOLO format")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    targets_root = shared.paths["dataset_root"] / "targets"

    for rel in ("images", "labels"):
        p = targets_root / rel
        if p.exists():
            shutil.rmtree(p)
    run_app(
        images_dir=shared.paths["targets_source_root"],
        labels_dir=targets_root / "labels",
        classes_file=targets_root / "classes.txt",
        export_root=targets_root,
        exts=["jpg", "jpeg", "png", "bmp"],
    )


if __name__ == "__main__":
    main()
