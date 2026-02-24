from __future__ import annotations

import json
from pathlib import Path

from .types import GeometryMetrics, IntegrityIssue, ModelMetrics


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_reports(
    reports_dir: Path,
    integrity_issues: list[IntegrityIssue],
    integrity_summary: dict,
    geometry_metrics: list[GeometryMetrics],
    geometry_summary: dict,
    model_reports: list[ModelMetrics],
) -> None:
    integrity_payload = {
        "summary": integrity_summary,
        "issues": [
            {"split": i.split, "stem": i.stem, "code": i.code, "message": i.message}
            for i in integrity_issues
        ],
    }
    _write_json(reports_dir / "integrity_report.json", integrity_payload)

    geometry_payload = {
        "summary": geometry_summary,
        "samples": [
            {
                "split": m.split,
                "stem": m.stem,
                "evaluable": m.evaluable,
                "mean_corner_err_px": m.mean_corner_err_px,
                "max_corner_err_px": m.max_corner_err_px,
                "obb_iou_meta_vs_label": m.obb_iou_meta_vs_label,
                "is_outlier": m.is_outlier,
                "message": m.message,
            }
            for m in geometry_metrics
        ],
    }
    _write_json(reports_dir / "geometry_report.json", geometry_payload)

    for rep in model_reports:
        payload = {
            "model_name": rep.model_name,
            "num_scored": rep.num_scored,
            "mean_iou": rep.mean_iou,
            "median_iou": rep.median_iou,
            "mean_center_drift_px": rep.mean_center_drift_px,
            "median_center_drift_px": rep.median_center_drift_px,
            "miss_rate": rep.miss_rate,
            "samples": [
                {
                    "split": s.split,
                    "stem": s.stem,
                    "iou": s.iou,
                    "center_drift_px": s.center_drift_px,
                    "missed": s.missed,
                }
                for s in rep.samples
            ],
        }
        _write_json(reports_dir / f"model_report_{rep.model_name}.json", payload)

    summary_lines = [
        "# Augment Checker Summary",
        "",
        f"- Total samples: {integrity_summary.get('total_samples', 0)}",
        f"- Integrity issues: {integrity_summary.get('total_issues', 0)}",
        f"- Geometry evaluable: {geometry_summary.get('num_evaluable', 0)}",
        f"- Geometry outliers: {geometry_summary.get('num_outliers', 0)}",
        f"- Mean corner error (px): {geometry_summary.get('mean_corner_error_px')}",
        "",
    ]

    if model_reports:
        summary_lines.append("## Model Comparison")
        for rep in model_reports:
            summary_lines.append(
                f"- {rep.model_name}: mean IoU={rep.mean_iou}, mean drift px={rep.mean_center_drift_px}, miss rate={rep.miss_rate}"
            )
    else:
        summary_lines.append("No prediction reports supplied.")

    (reports_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
