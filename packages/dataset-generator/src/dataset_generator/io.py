from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

@dataclass(slots=True)
class CanonicalTarget:
    image_path: Path
    label_path: Path
    class_id_local: int
    class_name: str
    canonical_corners_px: np.ndarray


def load_target_classes(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        raise FileNotFoundError(f"Target classes file not found: {classes_file}")
    classes = [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not classes:
        raise ValueError(f"No classes found in {classes_file}")
    return classes


def _parse_yolo_line(line: str) -> tuple[int, np.ndarray]:
    parts = line.split()
    if len(parts) != 9:
        raise ValueError(f"Expected 9 values in OBB YOLO line, got {len(parts)}")
    class_id = int(parts[0])
    coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(4, 2)
    return class_id, coords


def load_canonical_targets(
    target_images_dir: Path,
    target_labels_dir: Path,
    target_classes_file: Path,
) -> list[CanonicalTarget]:
    classes = load_target_classes(target_classes_file)
    image_paths = sorted(
        [
            p
            for p in target_images_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
    )
    if not image_paths:
        raise ValueError(f"No target images found in {target_images_dir}")

    targets: list[CanonicalTarget] = []
    for image_path in image_paths:
        label_path = target_labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        h, w = image.shape[:2]
        line = label_path.read_text(encoding="utf-8").strip().splitlines()
        if not line:
            continue

        class_id, corners_norm = _parse_yolo_line(line[0])
        if class_id < 0 or class_id >= len(classes):
            raise ValueError(f"Class id {class_id} in {label_path} outside classes range")

        corners = corners_norm.astype(np.float64).copy()
        corners[:, 0] *= w
        corners[:, 1] *= h
        targets.append(
            CanonicalTarget(
                image_path=image_path,
                label_path=label_path,
                class_id_local=class_id,
                class_name=classes[class_id],
                canonical_corners_px=corners.astype(np.float32),
            )
        )

    if not targets:
        raise ValueError("No canonical targets could be loaded")
    return targets


def load_backgrounds_by_split(split_dirs: dict[str, Path]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        split_dir = split_dirs[split]
        if not split_dir.exists():
            out[split] = []
            continue
        out[split] = sorted(
            [
                p
                for p in split_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
    return out


def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def audit_background_split_overlap(backgrounds_by_split: dict[str, list[Path]]) -> dict:
    train_hashes = {_sha1_file(p): p for p in backgrounds_by_split.get("train", [])}
    val_hashes = {_sha1_file(p): p for p in backgrounds_by_split.get("val", [])}
    overlap_hashes = sorted(set(train_hashes) & set(val_hashes))
    overlaps = [
        {
            "sha1": h,
            "train_path": str(train_hashes[h]),
            "val_path": str(val_hashes[h]),
        }
        for h in overlap_hashes
    ]
    return {
        "train_count": len(backgrounds_by_split.get("train", [])),
        "val_count": len(backgrounds_by_split.get("val", [])),
        "overlap_count": len(overlaps),
        "overlaps": overlaps,
    }


def enforce_disjoint_background_splits(backgrounds_by_split: dict[str, list[Path]]) -> tuple[dict[str, list[Path]], dict]:
    train_hashes = {_sha1_file(p): p for p in backgrounds_by_split.get("train", [])}
    val_hashes = {_sha1_file(p): p for p in backgrounds_by_split.get("val", [])}
    overlap_hashes = sorted(set(train_hashes) & set(val_hashes))

    reassigned_train: list[str] = []
    reassigned_val: list[str] = []
    out_train: list[Path] = [p for h, p in train_hashes.items() if h not in overlap_hashes]
    out_val: list[Path] = [p for h, p in val_hashes.items() if h not in overlap_hashes]

    for h in overlap_hashes:
        # Deterministic 80/20 split assignment for duplicate content.
        assign_to_val = (int(h[:8], 16) % 5) == 0
        chosen = val_hashes[h] if assign_to_val else train_hashes[h]
        if assign_to_val:
            out_val.append(chosen)
            reassigned_val.append(str(chosen))
        else:
            out_train.append(chosen)
            reassigned_train.append(str(chosen))

    out_train = sorted(out_train)
    out_val = sorted(out_val)
    audit = {
        "original_train_count": len(backgrounds_by_split.get("train", [])),
        "original_val_count": len(backgrounds_by_split.get("val", [])),
        "original_overlap_count": len(overlap_hashes),
        "reassigned_overlap_to_train_count": len(reassigned_train),
        "reassigned_overlap_to_val_count": len(reassigned_val),
        "final_train_count": len(out_train),
        "final_val_count": len(out_val),
        "policy": "deterministic_hash_repartition_80_20",
        "overlap_count": 0,
    }
    return {"train": out_train, "val": out_val}, audit


def _format_yolo_obb_line(class_id: int, obb_norm: np.ndarray) -> str:
    flat = obb_norm.reshape(-1)
    coords = " ".join(f"{float(v):.10f}" for v in flat)
    return f"{class_id} {coords}"


def write_yolo_obb_labels(path: Path, labels: list[tuple[int, np.ndarray]]) -> None:
    """Write one or more YOLO OBB lines: class x1 y1 x2 y2 x3 y3 x4 y4."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [_format_yolo_obb_line(class_id, obb_norm) for class_id, obb_norm in labels]
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")


def write_metadata(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_augmented_classes(output_root: Path, target_classes: list[str], class_offset_base: int) -> None:
    classes_path = output_root / "classes.txt"
    classes_path.parent.mkdir(parents=True, exist_ok=True)

    reserved = [f"__coco_reserved_{i}__" for i in range(class_offset_base)]
    merged = reserved + target_classes
    classes_path.write_text("\n".join(merged) + "\n", encoding="utf-8")

    mapping_payload = {
        "class_offset_base": class_offset_base,
        "target_classes": [
            {
                "name": name,
                "local_id": i,
                "exported_id": class_offset_base + i,
            }
            for i, name in enumerate(target_classes)
        ],
    }
    write_metadata(output_root / "classes_map.json", mapping_payload)
