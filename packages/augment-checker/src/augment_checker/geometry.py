from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .types import GeometryMetrics, SampleRecord
from .yolo import label_to_pixel_corners, load_yolo_labels, polygon_iou


def _apply_h(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float64)
    pts_h = np.concatenate([points.astype(np.float64), ones], axis=1)
    proj = (H @ pts_h.T).T
    out = proj[:, :2] / proj[:, 2:3]
    return out.astype(np.float32)


def run_geometry_checks(records: list[SampleRecord], outlier_threshold_px: float) -> tuple[list[GeometryMetrics], dict]:
    metrics: list[GeometryMetrics] = []

    for rec in records:
        if rec.meta_path is None or rec.image_path is None or rec.label_path is None:
            metrics.append(
                GeometryMetrics(
                    split=rec.split,
                    stem=rec.stem,
                    evaluable=False,
                    mean_corner_err_px=None,
                    max_corner_err_px=None,
                    obb_iou_meta_vs_label=None,
                    is_outlier=False,
                    message="missing image/label/meta",
                )
            )
            continue

        try:
            meta = json.loads(rec.meta_path.read_text(encoding="utf-8"))
            img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("failed to read image")
            h, w = img.shape[:2]

            labels = load_yolo_labels(rec.label_path, is_prediction=False)
            targets = meta.get("targets")
            if not isinstance(targets, list) or not targets:
                raise ValueError("metadata missing targets list")

            projected_list: list[np.ndarray] = []
            projected_raw_list: list[np.ndarray] = []
            canonical_list: list[np.ndarray] = []
            for t in targets:
                raw = t.get("projected_corners_px_raw", t.get("projected_corners_px"))
                rect = t.get("projected_corners_px_rect_obb", t.get("projected_corners_px"))
                if raw is None or rect is None:
                    raise ValueError("metadata missing projected corners (raw/rect)")
                projected_raw_list.append(np.array(raw, dtype=np.float32))
                projected_list.append(np.array(rect, dtype=np.float32))
                canonical_list.append(np.array(t["canonical_corners_px"], dtype=np.float32))

            if len(projected_list) != len(labels):
                raise ValueError(f"label/meta count mismatch: labels={len(labels)} meta_targets={len(projected_list)}")

            mean_errs: list[float] = []
            max_errs: list[float] = []
            ious: list[float] = []
            for idx, projected_stored in enumerate(projected_list):
                canonical = canonical_list[idx]
                H_obj = np.array(targets[idx]["H"], dtype=np.float64)
                projected_est = _apply_h(H_obj, canonical)
                corner_err = np.linalg.norm(projected_est - projected_raw_list[idx], axis=1)
                mean_errs.append(float(np.mean(corner_err)))
                max_errs.append(float(np.max(corner_err)))
                label_poly = label_to_pixel_corners(labels[idx], w, h)
                ious.append(polygon_iou(projected_stored, label_poly))

            metrics.append(
                GeometryMetrics(
                    split=rec.split,
                    stem=rec.stem,
                    evaluable=True,
                    mean_corner_err_px=float(np.mean(mean_errs)),
                    max_corner_err_px=float(np.max(max_errs)),
                    obb_iou_meta_vs_label=float(np.mean(ious)),
                    is_outlier=float(np.mean(mean_errs)) > outlier_threshold_px,
                )
            )
        except Exception as exc:
            metrics.append(
                GeometryMetrics(
                    split=rec.split,
                    stem=rec.stem,
                    evaluable=False,
                    mean_corner_err_px=None,
                    max_corner_err_px=None,
                    obb_iou_meta_vs_label=None,
                    is_outlier=False,
                    message=str(exc),
                )
            )

    eval_metrics = [m for m in metrics if m.evaluable and m.mean_corner_err_px is not None]
    errs = [m.mean_corner_err_px for m in eval_metrics if m.mean_corner_err_px is not None]
    ious = [m.obb_iou_meta_vs_label for m in eval_metrics if m.obb_iou_meta_vs_label is not None]

    summary = {
        "num_samples": len(metrics),
        "num_evaluable": len(eval_metrics),
        "num_outliers": sum(1 for m in eval_metrics if m.is_outlier),
        "outlier_rate": (sum(1 for m in eval_metrics if m.is_outlier) / len(eval_metrics)) if eval_metrics else 0.0,
        "mean_corner_error_px": float(np.mean(errs)) if errs else None,
        "p95_corner_error_px": float(np.percentile(np.array(errs), 95)) if errs else None,
        "max_corner_error_px": float(np.max(errs)) if errs else None,
        "mean_obb_iou_meta_vs_label": float(np.mean(ious)) if ious else None,
    }

    return metrics, summary
