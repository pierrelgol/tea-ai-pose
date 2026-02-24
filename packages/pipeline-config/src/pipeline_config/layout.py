from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PipelineLayout:
    model_root: Path
    run_root: Path
    train_root: Path
    train_ultralytics_root: Path
    optimize_root: Path
    infer_root: Path
    grade_root: Path
    eval_root: Path
    meta_root: Path
    latest_run_json: Path

    def eval_epoch_root(self, epoch: int) -> Path:
        return self.eval_root / f"epoch_{int(epoch):03d}"



def build_layout(*, artifacts_root: Path, model_key: str, run_id: str) -> PipelineLayout:
    model_root = artifacts_root / model_key
    run_root = model_root / "runs" / run_id
    return PipelineLayout(
        model_root=model_root,
        run_root=run_root,
        train_root=run_root / "train",
        train_ultralytics_root=run_root / "train" / "ultralytics",
        optimize_root=run_root / "optimize",
        infer_root=run_root / "infer",
        grade_root=run_root / "grade",
        eval_root=run_root / "eval",
        meta_root=run_root / "meta",
        latest_run_json=artifacts_root / "latest_run.json",
    )
