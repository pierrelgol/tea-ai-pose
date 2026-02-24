from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import Label


@dataclass(slots=True)
class ScoreWeights:
    oks: float = 0.75
    box_iou: float = 0.25
    fn_penalty: float = 0.30
    fp_penalty: float = 0.18

    def normalized(self) -> "ScoreWeights":
        total = self.oks + self.box_iou
        if total <= 0:
            raise ValueError("invalid score weights")
        return ScoreWeights(
            oks=self.oks / total,
            box_iou=self.box_iou / total,
            fn_penalty=self.fn_penalty,
            fp_penalty=self.fp_penalty,
        )


def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    cx, cy, w, h = [float(v) for v in box]
    return np.array([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dtype=np.float32)


def _bbox_iou(a_xywh: np.ndarray, b_xywh: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a_xywh)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b_xywh)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    union = a_area + b_area - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _pose_oks(gt: Label, pred: Label, image_w: int, image_h: int) -> tuple[float, float]:
    gt_k = gt.keypoints_norm
    pr_k = pred.keypoints_norm
    n = min(int(gt_k.shape[0]), int(pr_k.shape[0]))
    if n <= 0:
        return 0.0, 0.0

    area_px = max(
        1.0,
        float(gt.bbox_xywh_norm[2] * image_w) * float(gt.bbox_xywh_norm[3] * image_h),
    )
    sigma_px = max(1.0, 0.1 * (area_px ** 0.5))

    oks_vals: list[float] = []
    err_px_vals: list[float] = []
    for i in range(n):
        gv = float(gt_k[i, 2]) if gt_k.shape[1] > 2 else 1.0
        if gv <= 0.0:
            continue
        gx = float(gt_k[i, 0]) * image_w
        gy = float(gt_k[i, 1]) * image_h
        px = float(pr_k[i, 0]) * image_w
        py = float(pr_k[i, 1]) * image_h
        d2 = (gx - px) ** 2 + (gy - py) ** 2
        oks_vals.append(float(np.exp(-d2 / (2.0 * sigma_px * sigma_px))))
        err_px_vals.append(float(np.sqrt(d2)))

    if not oks_vals:
        return 0.0, 0.0
    return float(np.mean(oks_vals)), float(np.mean(err_px_vals))


def score_sample(
    *,
    split: str,
    stem: str,
    gt_labels: list[Label],
    pred_labels: list[Label],
    w: int,
    h: int,
    oks_threshold: float,
    weights: ScoreWeights,
) -> dict[str, Any]:
    wp = weights.normalized()

    candidates: list[tuple[float, int, int, float, float]] = []
    for gi, gt in enumerate(gt_labels):
        for pi, pred in enumerate(pred_labels):
            iou = _bbox_iou(gt.bbox_xywh_norm, pred.bbox_xywh_norm)
            oks, err = _pose_oks(gt, pred, w, h)
            quality = wp.oks * oks + wp.box_iou * iou
            if oks >= float(oks_threshold):
                candidates.append((quality, gi, pi, oks, err))

    candidates.sort(key=lambda x: x[0], reverse=True)
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[tuple[float, float]] = []
    keypoint_errors: list[float] = []
    for _q, gi, pi, oks, err in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((oks, _bbox_iou(gt_labels[gi].bbox_xywh_norm, pred_labels[pi].bbox_xywh_norm)))
        keypoint_errors.append(err)

    num_gt = len(gt_labels)
    num_pred = len(pred_labels)
    num_matches = len(matches)
    num_fn = max(0, num_gt - num_matches)
    num_fp = max(0, num_pred - num_matches)

    base = float(np.mean([0.75 * oks + 0.25 * iou for oks, iou in matches])) if matches else 0.0
    denom = max(1, num_gt)
    penalty_fn = (num_fn / denom) * wp.fn_penalty
    penalty_fp = (num_fp / denom) * wp.fp_penalty
    final = float(np.clip(base - penalty_fn - penalty_fp, 0.0, 1.0))

    oks_vals = [v[0] for v in matches]
    iou_vals = [v[1] for v in matches]
    return {
        "split": split,
        "stem": stem,
        "final_score_0_100": final * 100.0,
        "num_gt": num_gt,
        "num_pred": num_pred,
        "num_matches": num_matches,
        "num_fn": num_fn,
        "num_fp": num_fp,
        "penalty_fn": penalty_fn,
        "penalty_fp": penalty_fp,
        "penalty_containment": 0.0,
        "match_oks_mean": float(np.mean(oks_vals)) if oks_vals else None,
        "diagnostics": {
            "oks_mean": float(np.mean(oks_vals)) if oks_vals else None,
            "bbox_iou_mean": float(np.mean(iou_vals)) if iou_vals else None,
            "keypoint_error_px_mean": float(np.mean(keypoint_errors)) if keypoint_errors else None,
        },
    }
