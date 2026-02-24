from __future__ import annotations

from pathlib import Path


def list_split_images(dataset_root: Path, split: str) -> list[Path]:
    split_dir = dataset_root / "images" / split
    if not split_dir.exists():
        return []

    images = [
        p
        for p in split_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    ]
    return sorted(images)
