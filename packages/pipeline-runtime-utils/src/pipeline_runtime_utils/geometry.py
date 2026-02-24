from __future__ import annotations

import numpy as np


def corners_norm_to_px(corners_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    out = corners_norm.astype(np.float64).copy()
    out[:, 0] *= width
    out[:, 1] *= height
    return out.astype(np.float32)
