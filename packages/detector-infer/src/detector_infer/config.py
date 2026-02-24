from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class InferConfig:
    weights: Path
    dataset_root: Path
    output_root: Path
    model_name: str
    imgsz: int
    device: str
    conf_threshold: float
    iou_threshold: float
    seed: int
    splits: list[str]
    save_empty: bool
    batch_size: int = 16

    def validate(self) -> None:
        if not self.weights.exists():
            raise FileNotFoundError(f"weights not found: {self.weights}")
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        if self.conf_threshold < 0 or self.conf_threshold > 1:
            raise ValueError("conf_threshold must be in [0,1]")
        if self.iou_threshold < 0 or self.iou_threshold > 1:
            raise ValueError("iou_threshold must be in [0,1]")
        if not self.splits:
            raise ValueError("splits must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
