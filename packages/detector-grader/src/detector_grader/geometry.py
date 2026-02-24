from __future__ import annotations

import cv2
import numpy as np


def polygon_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = a.astype(np.float32).reshape(-1, 1, 2)
    pb = b.astype(np.float32).reshape(-1, 1, 2)
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    inter = float(max(0.0, inter_area))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def polygon_centroid(poly: np.ndarray) -> tuple[float, float]:
    return float(np.mean(poly[:, 0])), float(np.mean(poly[:, 1]))


def polygon_center_drift_px(a: np.ndarray, b: np.ndarray) -> float:
    acx, acy = polygon_centroid(a)
    bcx, bcy = polygon_centroid(b)
    dx = acx - bcx
    dy = acy - bcy
    return float((dx * dx + dy * dy) ** 0.5)


def edge_lengths(poly: np.ndarray) -> np.ndarray:
    d = []
    for i in range(4):
        p0 = poly[i]
        p1 = poly[(i + 1) % 4]
        d.append(float(np.linalg.norm(p1 - p0)))
    return np.array(d, dtype=np.float32)


def principal_angle_deg(poly: np.ndarray) -> float:
    edges = []
    for i in range(4):
        v = poly[(i + 1) % 4] - poly[i]
        edges.append((float(np.linalg.norm(v)), v))
    edges.sort(key=lambda x: x[0], reverse=True)
    v = edges[0][1]
    ang = np.degrees(np.arctan2(v[1], v[0]))
    ang = float(ang % 180.0)
    return ang


def angle_diff_deg(a: float, b: float) -> float:
    d = abs(a - b) % 180.0
    if d > 90.0:
        d = 180.0 - d
    return d


def reorder_corners_to_best(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    best = pred
    best_err = float("inf")
    candidates = [pred, pred[::-1]]
    for cand in candidates:
        for k in range(4):
            r = np.roll(cand, shift=k, axis=0)
            err = float(np.mean(np.linalg.norm(gt - r, axis=1)))
            if err < best_err:
                best_err = err
                best = r
    return best
