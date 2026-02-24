from __future__ import annotations

from pathlib import Path

import yaml


def load_class_names(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"classes file not found: {classes_file}")
    names = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not names:
        raise ValueError(f"classes file is empty: {classes_file}")
    return names


def write_data_yaml(dataset_root: Path, output_path: Path) -> tuple[Path, list[str]]:
    train_dir = dataset_root / "images" / "train"
    val_dir = dataset_root / "images" / "val"
    classes_file = dataset_root / "classes.txt"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("dataset images train/val directories are required")

    names = load_class_names(classes_file)

    payload = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_path, names
