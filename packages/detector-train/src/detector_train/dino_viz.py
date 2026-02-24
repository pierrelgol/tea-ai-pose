from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def normalize_map(values: np.ndarray, *, low_pct: float = 5.0, high_pct: float = 99.0, gamma: float = 0.85) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"expected rank-2 map, got shape={arr.shape}")
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros_like(arr, dtype=np.float32)
    work = arr.copy()
    work[~finite] = 0.0
    vals = work[finite]
    lo = float(np.percentile(vals, low_pct))
    hi = float(np.percentile(vals, high_pct))
    if hi <= lo:
        return np.zeros_like(work, dtype=np.float32)
    out = (work - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    if gamma > 0 and gamma != 1.0:
        out = np.power(out, gamma, dtype=np.float32)
    return out.astype(np.float32)


def _to_color(map_2d: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    norm = normalize_map(map_2d)
    u8 = (norm * 255.0).round().astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
    h, w = size_hw
    if color.shape[:2] != (h, w):
        color = cv2.resize(color, (w, h), interpolation=cv2.INTER_LINEAR)
    return color


def _overlay_heatmap(image_bgr: np.ndarray, heatmap_bgr: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    a = float(max(0.0, min(1.0, alpha)))
    return cv2.addWeighted(image_bgr, 1.0 - a, heatmap_bgr, a, 0.0)


def _teacher_attention_map(teacher: np.ndarray, obj_mask: np.ndarray | None = None) -> np.ndarray:
    t = np.asarray(teacher, dtype=np.float32)
    if t.ndim == 3:
        # Use feature energy to avoid cancellation from channel mean.
        attn = np.linalg.norm(t, axis=0).astype(np.float32)
    elif t.ndim == 2:
        attn = t
    else:
        raise ValueError(f"teacher map must be CHW or HW, got {t.shape}")
    norm = normalize_map(attn)
    if obj_mask is not None:
        m = np.asarray(obj_mask, dtype=np.float32)
        if m.ndim == 2:
            m = np.clip(m, 0.0, 1.0)
            if m.shape != norm.shape:
                m = cv2.resize(m, (norm.shape[1], norm.shape[0]), interpolation=cv2.INTER_NEAREST)
            # Mild foreground boost; still keeps raw DINO structure visible.
            norm = np.clip(norm * (1.0 + 0.35 * m), 0.0, 1.0)
    return norm


def _draw_mask_contours(image_bgr: np.ndarray, obj_mask: np.ndarray | None) -> np.ndarray:
    if obj_mask is None:
        return image_bgr
    m = np.asarray(obj_mask, dtype=np.float32)
    if m.ndim != 2:
        return image_bgr
    if m.shape != image_bgr.shape[:2]:
        m = cv2.resize(m, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    mask_u8 = (m > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = image_bgr.copy()
    if contours:
        cv2.drawContours(out, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _to_rgb_image(image_chw: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(image_chw, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim != 3:
        raise ValueError(f"expected rank-3 image tensor, got shape={arr.shape}")
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.shape[2] != 3:
        raise ValueError(f"expected 3 channels, got shape={arr.shape}")
    finite = np.isfinite(arr)
    if not finite.any():
        arr = np.zeros_like(arr, dtype=np.float32)
    else:
        arr = arr.copy()
        arr[~finite] = 0.0
    maxv = float(np.max(arr)) if arr.size > 0 else 0.0
    if maxv > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    rgb = (arr * 255.0).round().astype(np.uint8)
    h, w = size_hw
    if rgb.shape[:2] != (h, w):
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _label_tile(tile: np.ndarray, label: str) -> np.ndarray:
    out = tile.copy()
    cv2.putText(
        out,
        label,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return out


def save_dino_visualizations(
    *,
    snapshot: dict[str, Any],
    output_dir: Path,
    max_samples: int,
) -> dict[str, Any]:
    if max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    images = np.asarray(snapshot.get("images"))
    teacher = np.asarray(snapshot.get("teacher"))
    signal_raw = snapshot.get("signal_map")
    signal_map = np.asarray(signal_raw) if signal_raw is not None else None
    obj_mask_raw = snapshot.get("obj_mask")
    obj_mask = np.asarray(obj_mask_raw) if obj_mask_raw is not None else None
    if images.ndim != 4:
        raise ValueError(f"snapshot.images must be BCHW, got {images.shape}")
    if teacher.ndim not in (3, 4):
        raise ValueError(f"snapshot.teacher must be BHW or BCHW, got {teacher.shape}")
    if signal_map is not None and signal_map.ndim != 3:
        raise ValueError(f"snapshot.signal_map must be BHW, got {signal_map.shape}")

    count = int(min(images.shape[0], teacher.shape[0], max_samples))
    if signal_map is not None and signal_map.ndim >= 1:
        count = int(min(count, signal_map.shape[0]))
    if obj_mask is not None and obj_mask.ndim >= 1:
        count = int(min(count, obj_mask.shape[0]))
    if count <= 0:
        return {"files": [], "num_samples": 0, "num_layers": 0}

    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[str] = []

    for i in range(count):
        base_h = int(images[i].shape[1]) if images[i].shape[0] in (1, 3) else int(images[i].shape[0])
        base_w = int(images[i].shape[2]) if images[i].shape[0] in (1, 3) else int(images[i].shape[1])
        size_hw = (max(1, base_h), max(1, base_w))
        input_tile = _to_rgb_image(images[i], size_hw)
        sample_mask = None if obj_mask is None else obj_mask[i]
        teacher_attn = _teacher_attention_map(teacher[i], sample_mask)
        teacher_tile = _to_color(teacher_attn, size_hw)
        teacher_overlay = _overlay_heatmap(input_tile, teacher_tile, alpha=0.5)
        teacher_overlay = _draw_mask_contours(teacher_overlay, sample_mask)
        teacher_overlay = _label_tile(teacher_overlay, "teacher_attention_overlay")

        teacher_path = output_dir / f"sample_{i:02d}_teacher_overlay.png"
        if not cv2.imwrite(str(teacher_path), teacher_overlay):
            raise RuntimeError(f"failed to write dino visualization: {teacher_path}")
        files.append(str(teacher_path))

        if signal_map is not None:
            signal_tile = _to_color(signal_map[i], size_hw)
            signal_overlay = _overlay_heatmap(input_tile, signal_tile, alpha=0.55)
            signal_overlay = _draw_mask_contours(signal_overlay, sample_mask)
            signal_overlay = _label_tile(signal_overlay, "distill_signal_overlay")
            signal_path = output_dir / f"sample_{i:02d}_distill_signal_overlay.png"
            if not cv2.imwrite(str(signal_path), signal_overlay):
                raise RuntimeError(f"failed to write dino visualization: {signal_path}")
            files.append(str(signal_path))

    return {"files": files, "num_samples": count, "num_layers": 0}
