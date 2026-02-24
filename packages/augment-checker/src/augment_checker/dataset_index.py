from __future__ import annotations

from pathlib import Path

from .types import SampleRecord


def _stem_map(dir_path: Path, exts: set[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not dir_path.exists():
        return out
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            out[p.stem] = p
    return out


def index_dataset(dataset_root: Path) -> list[SampleRecord]:
    records: list[SampleRecord] = []
    for split in ("train", "val"):
        images = _stem_map(dataset_root / "images" / split, {".jpg", ".jpeg", ".png", ".bmp"})
        labels = _stem_map(dataset_root / "labels" / split, {".txt"})
        metas = _stem_map(dataset_root / "meta" / split, {".json"})

        all_stems = sorted(set(images) | set(labels) | set(metas))
        for stem in all_stems:
            records.append(
                SampleRecord(
                    split=split,
                    stem=stem,
                    image_path=images.get(stem),
                    label_path=labels.get(stem),
                    meta_path=metas.get(stem),
                )
            )
    return records
