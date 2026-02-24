from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import is_convex_quad, polygon_area, quad_inside_bounds


@dataclass(slots=True)
class HomographySample:
    H: np.ndarray
    quad_px: np.ndarray


@dataclass(slots=True)
class HomographyParams:
    scale_min: float
    scale_max: float
    translate_frac: float
    perspective_jitter: float
    min_quad_area_frac: float
    max_attempts: int
    edge_bias_prob: float
    edge_band_frac: float


def sample_valid_homography(
    canonical_corners_px: np.ndarray,
    background_w: int,
    background_h: int,
    rng: np.random.Generator,
    params: HomographyParams,
) -> HomographySample:
    canonical = canonical_corners_px.astype(np.float32)
    src_w = float(np.linalg.norm(canonical[1] - canonical[0]))
    src_h = float(np.linalg.norm(canonical[3] - canonical[0]))

    if src_w <= 1.0 or src_h <= 1.0:
        raise ValueError("Canonical target bbox too small")

    bg_area = background_w * background_h
    min_area = params.min_quad_area_frac * bg_area
    jitter_factor = 1.0 + 2.0 * float(max(params.perspective_jitter, 0.0))
    max_fit_scale = min(background_w / (src_w * jitter_factor), background_h / (src_h * jitter_factor))
    scale_upper = min(params.scale_max, max_fit_scale * 0.95)
    if scale_upper <= 0:
        raise RuntimeError("No feasible scale for current target/background geometry")
    scale_lower = min(params.scale_min, scale_upper)

    def _sample_center(
        *,
        margin_x: float,
        margin_y: float,
    ) -> tuple[float, float]:
        def _u(low: float, high: float) -> float:
            if high <= low:
                return float(low)
            return float(rng.uniform(low, high))

        cx0 = background_w * 0.5 + float(rng.uniform(-params.translate_frac, params.translate_frac)) * background_w
        cy0 = background_h * 0.5 + float(rng.uniform(-params.translate_frac, params.translate_frac)) * background_h
        if rng.random() >= params.edge_bias_prob:
            return (
                float(np.clip(cx0, margin_x, background_w - margin_x)),
                float(np.clip(cy0, margin_y, background_h - margin_y)),
            )

        band_x = max(margin_x, params.edge_band_frac * background_w)
        band_y = max(margin_y, params.edge_band_frac * background_h)
        edge = int(rng.integers(0, 4))
        if edge == 0:
            cx = _u(margin_x, min(background_w - margin_x, band_x))
            cy = _u(margin_y, background_h - margin_y)
        elif edge == 1:
            cx = _u(max(margin_x, background_w - band_x), background_w - margin_x)
            cy = _u(margin_y, background_h - margin_y)
        elif edge == 2:
            cx = _u(margin_x, background_w - margin_x)
            cy = _u(margin_y, min(background_h - margin_y, band_y))
        else:
            cx = _u(margin_x, background_w - margin_x)
            cy = _u(max(margin_y, background_h - band_y), background_h - margin_y)
        return (
            float(np.clip(cx, margin_x, background_w - margin_x)),
            float(np.clip(cy, margin_y, background_h - margin_y)),
        )

    for _ in range(params.max_attempts):
        scale = float(rng.uniform(scale_lower, scale_upper))

        rect_w = src_w * scale
        rect_h = src_h * scale

        jitter_mag = params.perspective_jitter * max(rect_w, rect_h)
        margin_x = rect_w / 2.0 + jitter_mag
        margin_y = rect_h / 2.0 + jitter_mag
        if margin_x >= background_w or margin_y >= background_h:
            continue

        cx, cy = _sample_center(margin_x=margin_x, margin_y=margin_y)

        base_rect = np.array(
            [
                [cx - rect_w / 2.0, cy - rect_h / 2.0],
                [cx + rect_w / 2.0, cy - rect_h / 2.0],
                [cx + rect_w / 2.0, cy + rect_h / 2.0],
                [cx - rect_w / 2.0, cy + rect_h / 2.0],
            ],
            dtype=np.float32,
        )

        jitter = rng.uniform(-jitter_mag, jitter_mag, size=(4, 2)).astype(np.float32)
        dst_quad = base_rect + jitter

        if not is_convex_quad(dst_quad):
            continue
        if not quad_inside_bounds(dst_quad, background_w, background_h):
            continue
        if polygon_area(dst_quad) < min_area:
            continue

        H = cv2.getPerspectiveTransform(canonical, dst_quad)
        return HomographySample(H=H.astype(np.float64), quad_px=dst_quad.astype(np.float32))

    raise RuntimeError("Could not sample a valid homography within max attempts")
