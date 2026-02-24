from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(slots=True)
class PoseLabel:
    class_id: int
    bbox_xywh_norm: np.ndarray
    keypoints_norm: np.ndarray
    confidence: float


def parse_pose_line(line: str, *, is_prediction: bool) -> PoseLabel:
    parts = line.strip().split()
    if is_prediction:
        if len(parts) < 6:
            raise ValueError("prediction pose label must have at least class bbox and keypoint fields")
        # prediction format: cls cx cy w h (kpt_x kpt_y kpt_vis/conf)* [conf]
        maybe_conf = None
        if (len(parts) - 6) % 3 == 0:
            maybe_conf = float(parts[-1])
            payload = parts[1:-1]
        else:
            payload = parts[1:]
    else:
        if len(parts) < 8:
            raise ValueError("GT pose label must have class bbox and at least one keypoint triplet")
        maybe_conf = 1.0
        payload = parts[1:]

    if len(payload) < 7:
        raise ValueError("pose payload must include bbox and keypoint triplets")
    if (len(payload) - 4) % 3 != 0:
        raise ValueError("pose keypoints payload must be divisible by 3")

    class_id = int(parts[0])
    bbox_xywh = np.array([float(v) for v in payload[:4]], dtype=np.float32)
    kpt_flat = np.array([float(v) for v in payload[4:]], dtype=np.float32)
    keypoints = kpt_flat.reshape(-1, 3)
    confidence = float(maybe_conf) if maybe_conf is not None else 1.0
    return PoseLabel(
        class_id=class_id,
        bbox_xywh_norm=bbox_xywh,
        keypoints_norm=keypoints,
        confidence=confidence,
    )


def load_pose_labels(
    path: Path,
    *,
    is_prediction: bool,
    conf_threshold: float = 0.0,
    require_nonempty_predictions: bool = False,
) -> list[PoseLabel]:
    if not path.exists():
        return []
    out: list[PoseLabel] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            label = parse_pose_line(line, is_prediction=is_prediction)
        except Exception as exc:
            raise ValueError(f"invalid pose label at {path}:{line_no}: {exc}") from exc
        if is_prediction and float(label.confidence) < float(conf_threshold):
            continue
        out.append(label)
    if is_prediction and require_nonempty_predictions and not out:
        raise ValueError(f"No prediction above confidence threshold: {path}")
    return out
