from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class YoloBox:
    x_center: float
    y_center: float
    width: float
    height: float


def discover_images(images_dir: Path, exts: list[str]) -> list[Path]:
    if not images_dir.exists() or not images_dir.is_dir():
        return []

    normalized = {e.lower().lstrip(".") for e in exts}
    image_paths = [
        p
        for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower().lstrip(".") in normalized
    ]
    return image_paths


def load_classes(classes_file: Path) -> list[str]:
    if not classes_file.exists():
        return []
    return [line.strip() for line in classes_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def ensure_class_id(classes_file: Path, class_name: str) -> int:
    classes = load_classes(classes_file)
    if class_name in classes:
        return classes.index(class_name)

    classes_file.parent.mkdir(parents=True, exist_ok=True)
    with classes_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{class_name}\n")
    return len(classes)


def load_yolo_label(label_file: Path) -> tuple[int, YoloBox] | None:
    if not label_file.exists():
        return None

    line = label_file.read_text(encoding="utf-8").strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) != 9:
        return None

    class_id = int(parts[0])
    coords = [float(x) for x in parts[1:]]
    xs = coords[0::2]
    ys = coords[1::2]
    x1 = min(xs)
    x2 = max(xs)
    y1 = min(ys)
    y2 = max(ys)
    box = YoloBox(
        x_center=(x1 + x2) * 0.5,
        y_center=(y1 + y2) * 0.5,
        width=max(0.0, x2 - x1),
        height=max(0.0, y2 - y1),
    )
    return class_id, box


def save_yolo_label(label_file: Path, class_id: int, box: YoloBox) -> None:
    label_file.parent.mkdir(parents=True, exist_ok=True)
    x1 = box.x_center - box.width * 0.5
    y1 = box.y_center - box.height * 0.5
    x2 = box.x_center + box.width * 0.5
    y2 = box.y_center + box.height * 0.5
    line = f"{class_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}\n"
    label_file.write_text(line, encoding="utf-8")
