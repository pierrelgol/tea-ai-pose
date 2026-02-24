from __future__ import annotations

import numpy as np


def apply_homography_to_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float64)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([pts, ones], axis=1)
    projected = (H @ pts_h.T).T
    projected_xy = projected[:, :2] / projected[:, 2:3]
    return projected_xy.astype(np.float32)


def is_convex_quad(quad: np.ndarray) -> bool:
    if quad.shape != (4, 2):
        return False

    cross_signs: list[float] = []
    for i in range(4):
        p0 = quad[i]
        p1 = quad[(i + 1) % 4]
        p2 = quad[(i + 2) % 4]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = float(v1[0] * v2[1] - v1[1] * v2[0])
        if abs(cross) > 1e-7:
            cross_signs.append(cross)

    if not cross_signs:
        return False

    first_positive = cross_signs[0] > 0
    return all((c > 0) == first_positive for c in cross_signs)


def polygon_area(quad: np.ndarray) -> float:
    x = quad[:, 0]
    y = quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) * 0.5)


def quad_inside_bounds(quad: np.ndarray, image_w: int, image_h: int) -> bool:
    return (
        np.all(quad[:, 0] >= 0)
        and np.all(quad[:, 0] < image_w)
        and np.all(quad[:, 1] >= 0)
        and np.all(quad[:, 1] < image_h)
    )


def corners_px_to_yolo_obb(quad: np.ndarray, image_w: int, image_h: int) -> np.ndarray:
    out = quad.astype(np.float64).copy()
    out[:, 0] = out[:, 0] / image_w
    out[:, 1] = out[:, 1] / image_h
    out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)
    return out.astype(np.float32)
