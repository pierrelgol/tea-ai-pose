from __future__ import annotations

from pathlib import Path
import re

from pipeline_runtime_utils import (
    OBBLabel as Label,
    SampleRecord,
    image_shape_fast,
    index_ground_truth,
    load_obb_labels,
    load_prediction_labels as _load_prediction_labels,
    resolve_latest_weights_from_artifacts,
)


def sanitize_model_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name.strip())
    cleaned = cleaned.strip("._-")
    return cleaned or "model"


def load_labels(path: Path, *, is_prediction: bool, conf_threshold: float) -> list[Label]:
    return load_obb_labels(path, is_prediction=is_prediction, conf_threshold=conf_threshold)


def load_prediction_labels(
    predictions_root: Path,
    model_name: str,
    split: str,
    stem: str,
    conf_threshold: float,
) -> list[Label]:
    return _load_prediction_labels(
        predictions_root=predictions_root,
        model_name=model_name,
        split=split,
        stem=stem,
        conf_threshold=conf_threshold,
    )


def image_shape(path: Path | None) -> tuple[int, int] | None:
    return image_shape_fast(path)


def resolve_latest_weights(artifacts_root: Path) -> Path:
    return resolve_latest_weights_from_artifacts(artifacts_root)


def infer_model_name_from_weights(weights: Path) -> str:
    if weights.parent.name == "weights":
        parents = [p.name for p in weights.parents]
        if "runs" in parents:
            run_idx = parents.index("runs")
            if run_idx >= 2:
                run_name = parents[run_idx - 1]
                model_key = parents[run_idx - 2]
                return sanitize_model_name(f"{model_key}_{run_name}_{weights.stem}")
        if weights.parent.parent.name:
            run_name = weights.parent.parent.name
            return sanitize_model_name(f"{run_name}_{weights.stem}")
    return sanitize_model_name(weights.stem)
