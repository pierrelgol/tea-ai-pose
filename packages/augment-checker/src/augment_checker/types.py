from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SampleRecord:
    split: str
    stem: str
    image_path: Path | None
    label_path: Path | None
    meta_path: Path | None


@dataclass(slots=True)
class IntegrityIssue:
    split: str
    stem: str
    code: str
    message: str


@dataclass(slots=True)
class GeometryMetrics:
    split: str
    stem: str
    evaluable: bool
    mean_corner_err_px: float | None
    max_corner_err_px: float | None
    obb_iou_meta_vs_label: float | None
    is_outlier: bool
    message: str | None = None


@dataclass(slots=True)
class ModelSampleMetric:
    split: str
    stem: str
    iou: float | None
    center_drift_px: float | None
    missed: bool


@dataclass(slots=True)
class ModelMetrics:
    model_name: str
    num_scored: int
    mean_iou: float | None
    median_iou: float | None
    mean_center_drift_px: float | None
    median_center_drift_px: float | None
    miss_rate: float
    samples: list[ModelSampleMetric]
