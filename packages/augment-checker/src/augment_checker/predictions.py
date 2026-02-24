from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .types import ModelMetrics, ModelSampleMetric, SampleRecord
from .yolo import center_drift_px, label_to_pixel_corners, load_yolo_labels, polygon_iou


def _prediction_models(predictions_root: Path) -> list[Path]:
    if not predictions_root.exists():
        return []
    return sorted([p for p in predictions_root.iterdir() if p.is_dir()])


def _greedy_iou_match(gt_polys: list[np.ndarray], pr_polys: list[np.ndarray]) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for gi, g in enumerate(gt_polys):
        for pi, p in enumerate(pr_polys):
            candidates.append((polygon_iou(g, p), gi, pi))
    candidates.sort(key=lambda x: (-x[0], x[1], x[2]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    out: list[tuple[int, int]] = []
    for iou, gi, pi in candidates:
        if iou <= 0:
            continue
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi))
    return out


def run_prediction_checks(records: list[SampleRecord], predictions_root: Path | None) -> list[ModelMetrics]:
    if predictions_root is None:
        return []

    models = _prediction_models(predictions_root)
    reports: list[ModelMetrics] = []

    for model_dir in models:
        model_name = model_dir.name
        sample_metrics: list[ModelSampleMetric] = []
        ious: list[float] = []
        drifts: list[float] = []
        misses = 0

        for rec in records:
            if rec.image_path is None or rec.label_path is None:
                sample_metrics.append(ModelSampleMetric(rec.split, rec.stem, None, None, True))
                misses += 1
                continue

            pred_label = model_dir / "labels" / rec.split / f"{rec.stem}.txt"
            if not pred_label.exists():
                sample_metrics.append(ModelSampleMetric(rec.split, rec.stem, None, None, True))
                misses += 1
                continue

            img = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR)
            if img is None:
                sample_metrics.append(ModelSampleMetric(rec.split, rec.stem, None, None, True))
                misses += 1
                continue
            h, w = img.shape[:2]

            gt_labels = load_yolo_labels(rec.label_path, is_prediction=False)
            pred_labels = load_yolo_labels(pred_label, is_prediction=True, conf_threshold=0.0)
            gt_polys = [label_to_pixel_corners(g, w, h) for g in gt_labels]
            pred_polys = [label_to_pixel_corners(p, w, h) for p in pred_labels]

            matches = _greedy_iou_match(gt_polys, pred_polys)
            if not matches:
                sample_metrics.append(ModelSampleMetric(rec.split, rec.stem, None, None, True))
                misses += 1
                continue

            row_ious: list[float] = []
            row_drifts: list[float] = []
            for gi, pi in matches:
                iou = polygon_iou(gt_polys[gi], pred_polys[pi])
                drift = center_drift_px(gt_polys[gi], pred_polys[pi])
                row_ious.append(iou)
                row_drifts.append(drift)
                ious.append(iou)
                drifts.append(drift)
            sample_metrics.append(
                ModelSampleMetric(
                    rec.split,
                    rec.stem,
                    float(np.mean(row_ious)),
                    float(np.mean(row_drifts)),
                    False,
                )
            )

        total = len(records) if records else 1
        report = ModelMetrics(
            model_name=model_name,
            num_scored=len(ious),
            mean_iou=float(np.mean(ious)) if ious else None,
            median_iou=float(np.median(ious)) if ious else None,
            mean_center_drift_px=float(np.mean(drifts)) if drifts else None,
            median_center_drift_px=float(np.median(drifts)) if drifts else None,
            miss_rate=misses / total,
            samples=sample_metrics,
        )
        reports.append(report)

    return reports
