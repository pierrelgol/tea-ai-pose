from __future__ import annotations

import cv2
import numpy as np


def warp_target_and_mask(
    *,
    target: np.ndarray,
    canonical_corners_px: np.ndarray,
    H: np.ndarray,
    out_w: int,
    out_h: int,
) -> tuple[np.ndarray, np.ndarray]:
    src_mask = np.zeros(target.shape[:2], dtype=np.uint8)
    polygon = canonical_corners_px.reshape((-1, 1, 2)).astype(np.int32)
    cv2.fillConvexPoly(src_mask, polygon, 255)

    warped_target = cv2.warpPerspective(target, H, (out_w, out_h), flags=cv2.INTER_LINEAR)
    warped_mask = cv2.warpPerspective(src_mask, H, (out_w, out_h), flags=cv2.INTER_NEAREST)
    return warped_target, warped_mask


def blend_layer(
    *,
    background: np.ndarray,
    warped_target: np.ndarray,
    warped_mask: np.ndarray,
    feather_px: int = 5,
) -> np.ndarray:
    if feather_px <= 0:
        feather_px = 1
    mask_f = warped_mask.astype(np.float32) / 255.0
    if np.max(mask_f) <= 0:
        return background

    mask_soft = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=float(feather_px), sigmaY=float(feather_px))
    mask_soft = np.clip(mask_soft * 1.6, 0.0, 1.0)
    alpha = mask_soft[:, :, None]
    out = background.astype(np.float32) * (1.0 - alpha) + warped_target.astype(np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def visible_ratio(
    *,
    warped_mask: np.ndarray,
    occupancy_mask: np.ndarray,
) -> float:
    placed = warped_mask > 0
    total = int(np.count_nonzero(placed))
    if total == 0:
        return 0.0
    visible = int(np.count_nonzero(placed & (~occupancy_mask)))
    return float(visible / total)

