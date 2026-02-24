from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from pipeline_runtime_utils import corners_norm_to_px, load_obb_labels, parse_obb_line


@dataclass(slots=True)
class YoloLabel:
    class_id: int
    corners_norm: np.ndarray  # (4, 2) normalized points
    format_name: str  # "obb"


def parse_yolo_line(line: str, *, is_prediction: bool = False) -> YoloLabel:
    parsed = parse_obb_line(line, is_prediction=is_prediction)
    return YoloLabel(class_id=parsed.class_id, corners_norm=parsed.corners_norm, format_name="obb")


def load_yolo_labels(path, *, is_prediction: bool = False, conf_threshold: float = 0.0) -> list[YoloLabel]:
    if path.exists() and not path.read_text(encoding="utf-8").strip():
        if is_prediction:
            raise ValueError(f"Empty label file: {path}")
        return []
    if not is_prediction:
        parsed = load_obb_labels(
            path,
            is_prediction=False,
            conf_threshold=conf_threshold,
            require_nonempty_predictions=False,
        )
        return [YoloLabel(class_id=p.class_id, corners_norm=p.corners_norm, format_name="obb") for p in parsed]

    if not path.exists():
        return []
    parsed = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 10:
            continue
        label = parse_obb_line(line, is_prediction=True)
        if float(label.confidence) < float(conf_threshold):
            continue
        parsed.append(label)
    if not parsed:
        raise ValueError("No prediction above confidence threshold")
    return [YoloLabel(class_id=p.class_id, corners_norm=p.corners_norm, format_name="obb") for p in parsed]


def load_yolo_label(path, *, is_prediction: bool = False, conf_threshold: float = 0.0) -> YoloLabel:
    labels = load_yolo_labels(path, is_prediction=is_prediction, conf_threshold=conf_threshold)
    return labels[0]


def _polygon_area_norm(corners: np.ndarray) -> float:
    x = corners[:, 0]
    y = corners[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def validate_yolo_label(label: YoloLabel, eps: float = 1e-6) -> list[str]:
    issues: list[str] = []
    for idx, (x, y) in enumerate(label.corners_norm):
        if x < 0.0 or x > 1.0:
            issues.append(f"x{idx + 1} outside [0,1]: {x}")
        if y < 0.0 or y > 1.0:
            issues.append(f"y{idx + 1} outside [0,1]: {y}")

    width = float(np.max(label.corners_norm[:, 0]) - np.min(label.corners_norm[:, 0]))
    height = float(np.max(label.corners_norm[:, 1]) - np.min(label.corners_norm[:, 1]))
    if width <= eps:
        issues.append(f"width degenerate: {width}")
    if height <= eps:
        issues.append(f"height degenerate: {height}")

    area = _polygon_area_norm(label.corners_norm)
    if area <= eps:
        issues.append(f"polygon area degenerate: {area}")

    return issues


def label_to_pixel_corners(label: YoloLabel, image_w: int, image_h: int) -> np.ndarray:
    return corners_norm_to_px(label.corners_norm, image_w, image_h)


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = a.astype(np.float32).reshape(-1, 1, 2)
    pb = b.astype(np.float32).reshape(-1, 1, 2)
    area_a = _polygon_area_norm(a)
    area_b = _polygon_area_norm(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    inter = float(max(0.0, inter_area))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def polygon_centroid_px(poly: np.ndarray) -> tuple[float, float]:
    return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))


def center_drift_px(a: np.ndarray, b: np.ndarray) -> float:
    acx, acy = polygon_centroid_px(a)
    bcx, bcy = polygon_centroid_px(b)
    dx = acx - bcx
    dy = acy - bcy
    return float((dx * dx + dy * dy) ** 0.5)
