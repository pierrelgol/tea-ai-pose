from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .types import SampleRecord
from .yolo import label_to_pixel_corners, load_yolo_labels


def _draw_polygon(image: np.ndarray, corners: np.ndarray, color: tuple[int, int, int], text: str) -> None:
    poly = corners.reshape((-1, 1, 2)).astype(np.int32)
    cv2.polylines(image, [poly], isClosed=True, color=color, thickness=2)
    x0, y0 = int(poly[0][0][0]), int(poly[0][0][1])
    cv2.putText(image, text, (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def export_debug_overlays(records: list[SampleRecord], reports_dir: Path, n_per_split: int, seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    written: list[Path] = []

    for split in ("train", "val"):
        split_records = [r for r in records if r.split == split and r.image_path and r.label_path and r.meta_path]
        if not split_records:
            continue
        count = min(n_per_split, len(split_records))
        idxs = rng.choice(len(split_records), size=count, replace=False)

        out_dir = reports_dir / "overlays" / split
        out_dir.mkdir(parents=True, exist_ok=True)

        for idx in idxs:
            rec = split_records[int(idx)]
            img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            if img is None:
                continue

            gt_labels = load_yolo_labels(rec.label_path)
            h, w = img.shape[:2]
            for i, gt in enumerate(gt_labels):
                gt_corners = label_to_pixel_corners(gt, w, h)
                _draw_polygon(img, gt_corners, (0, 255, 0), f"GT{i+1}")

            meta = json.loads(rec.meta_path.read_text(encoding="utf-8"))
            targets = meta.get("targets")
            if not isinstance(targets, list) or not targets:
                continue
            for i, t in enumerate(targets):
                corners_payload = t.get("projected_corners_px_rect_obb", t.get("projected_corners_px"))
                if corners_payload is None:
                    continue
                corners = np.array(corners_payload, dtype=np.float32)
                _draw_polygon(img, corners, (0, 0, 255), f"M{i+1}")

            out_path = out_dir / f"{rec.stem}.jpg"
            cv2.imwrite(str(out_path), img)
            written.append(out_path)

    return written
