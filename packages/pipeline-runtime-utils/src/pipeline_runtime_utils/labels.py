from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class OBBLabel:
    class_id: int
    corners_norm: np.ndarray
    confidence: float


def parse_obb_line(line: str, *, is_prediction: bool) -> OBBLabel:
    parts = line.strip().split()
    if is_prediction:
        if len(parts) not in (9, 10):
            raise ValueError("prediction label must have 9 (no conf) or 10 (with conf) OBB fields")
        confidence = float(parts[9]) if len(parts) == 10 else 1.0
    else:
        if len(parts) != 9:
            raise ValueError("GT label must have 9 OBB fields")
        confidence = 1.0

    class_id = int(parts[0])
    corners = np.array([float(x) for x in parts[1:9]], dtype=np.float32).reshape(4, 2)
    return OBBLabel(class_id=class_id, corners_norm=corners, confidence=confidence)


def load_obb_labels(
    path: Path,
    *,
    is_prediction: bool,
    conf_threshold: float = 0.0,
    require_nonempty_predictions: bool = False,
) -> list[OBBLabel]:
    if not path.exists():
        return []
    out: list[OBBLabel] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            label = parse_obb_line(line, is_prediction=is_prediction)
        except Exception as exc:
            raise ValueError(f"invalid OBB label at {path}:{line_no}: {exc}") from exc
        if is_prediction and float(label.confidence) < float(conf_threshold):
            continue
        out.append(label)
    if is_prediction and require_nonempty_predictions and not out:
        raise ValueError(f"No prediction above confidence threshold: {path}")
    return out

