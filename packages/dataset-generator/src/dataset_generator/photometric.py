from __future__ import annotations

import cv2
import numpy as np

from .config import GeneratorConfig


def _odd_kernel(v: int) -> int:
    return v if v % 2 == 1 else v + 1


def _apply_color_jitter(img: np.ndarray, rng: np.random.Generator, config: GeneratorConfig) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_shift = float(rng.uniform(-config.color_hue_shift_max_deg, config.color_hue_shift_max_deg))
    s_gain = float(rng.uniform(config.color_sat_gain_min, config.color_sat_gain_max))
    v_gain = float(rng.uniform(config.color_val_gain_min, config.color_val_gain_max))
    hsv[:, :, 0] = (hsv[:, :, 0] + h_shift) % 180.0
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_gain, 0.0, 255.0)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_gain, 0.0, 255.0)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_motion_blur(img: np.ndarray, rng: np.random.Generator, config: GeneratorConfig) -> np.ndarray:
    k = _odd_kernel(int(rng.integers(config.motion_blur_kernel_min, config.motion_blur_kernel_max + 1)))
    kernel = np.zeros((k, k), dtype=np.float32)
    kernel[k // 2, :] = 1.0
    angle = float(rng.uniform(-config.motion_blur_angle_max_deg, config.motion_blur_angle_max_deg))
    rot = cv2.getRotationMatrix2D((k / 2, k / 2), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    kernel /= max(float(np.sum(kernel)), 1e-6)
    return cv2.filter2D(img, -1, kernel)


def _apply_noise(img: np.ndarray, rng: np.random.Generator, config: GeneratorConfig) -> np.ndarray:
    sigma = float(rng.uniform(config.noise_sigma_min, config.noise_sigma_max))
    noise = rng.normal(0.0, sigma, size=img.shape).astype(np.float32)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_jpeg_artifact(img: np.ndarray, rng: np.random.Generator, config: GeneratorConfig) -> np.ndarray:
    quality = int(rng.integers(config.jpeg_quality_min, config.jpeg_quality_max + 1))
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else img


def apply_photometric_stack(
    img: np.ndarray,
    *,
    rng: np.random.Generator,
    config: GeneratorConfig,
) -> tuple[np.ndarray, dict]:
    out = img
    applied: dict[str, bool] = {
        "color_jitter": False,
        "blur": False,
        "motion_blur": False,
        "noise": False,
        "jpeg_artifact": False,
    }

    if rng.random() < config.color_jitter_prob:
        out = _apply_color_jitter(out, rng, config)
        applied["color_jitter"] = True

    if rng.random() < config.blur_prob:
        k = _odd_kernel(int(rng.integers(config.gaussian_blur_kernel_min, config.gaussian_blur_kernel_max + 1)))
        out = cv2.GaussianBlur(out, (k, k), sigmaX=0.0)
        applied["blur"] = True

    if rng.random() < config.motion_blur_prob:
        out = _apply_motion_blur(out, rng, config)
        applied["motion_blur"] = True

    if rng.random() < config.noise_prob:
        out = _apply_noise(out, rng, config)
        applied["noise"] = True

    if rng.random() < config.jpeg_artifact_prob:
        out = _apply_jpeg_artifact(out, rng, config)
        applied["jpeg_artifact"] = True

    return out, applied
