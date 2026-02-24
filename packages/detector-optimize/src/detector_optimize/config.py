from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class OptimizeConfig:
    weights: Path
    output_root: Path
    run_name: str
    model_key: str

    imgsz: int = 640
    batch: int = 1
    precision: str = "fp16"  # fp16|fp32|int8
    workspace_gb: float = 4.0
    dynamic: bool = False
    simplify: bool = True
    opset: int = 17
    device: str = "0"
    trtexec_bin: str = "trtexec"
    prefer_trtexec: bool = True
    save_onnx: bool = False

    def validate(self) -> None:
        if not self.weights.exists() or not self.weights.is_file():
            raise FileNotFoundError(f"weights file not found: {self.weights}")
        if self.weights.suffix.lower() != ".pt":
            raise ValueError(f"weights must be a .pt file, got: {self.weights}")
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        if self.precision not in {"fp16", "fp32", "int8"}:
            raise ValueError("precision must be one of: fp16, fp32, int8")
        if self.workspace_gb <= 0:
            raise ValueError("workspace_gb must be > 0")
        if self.opset < 11:
            raise ValueError("opset must be >= 11")
        if not str(self.run_name).strip():
            raise ValueError("run_name must not be empty")
        if not str(self.model_key).strip():
            raise ValueError("model_key must not be empty")
