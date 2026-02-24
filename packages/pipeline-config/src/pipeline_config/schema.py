from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any


_ALLOWED_TOP = {
    "paths",
    "run",
    "dataset",
    "generator",
    "tuner",
    "train",
    "infer",
    "grade",
    "review",
    "checks",
    "profile",
}


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    config_path: Path
    config_root: Path
    paths: dict[str, Any]
    run: dict[str, Any]
    dataset: dict[str, Any]
    generator: dict[str, Any]
    tuner: dict[str, Any]
    train: dict[str, Any]
    infer: dict[str, Any]
    grade: dict[str, Any]
    review: dict[str, Any]
    checks: dict[str, Any]
    profile: dict[str, Any]



def _expect_dict(payload: Any, where: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"{where} must be an object")
    return payload



def _expect_keys(obj: dict[str, Any], allowed: set[str], where: str, required: set[str] | None = None) -> None:
    extra = sorted(set(obj) - allowed)
    if extra:
        raise ValueError(f"unknown keys in {where}: {extra}")
    req = required if required is not None else set()
    missing = sorted(req - set(obj))
    if missing:
        raise ValueError(f"missing required keys in {where}: {missing}")



def _resolve_path(config_root: Path, raw: str | Path) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (config_root / p).resolve()
    else:
        p = p.resolve()
    return p



def _normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    seed = int(run.get("seed", 42))
    run_id = str(run.get("run_id") or datetime.utcnow().strftime("run-%Y%m%d-%H%M%S"))
    model_key = str(run.get("model_key") or "default")
    model = str(run.get("model") or "hf-openvision-yolo26-n-obb")
    dataset_name = str(run.get("dataset") or "coco1024")
    return {
        "seed": seed,
        "run_id": run_id,
        "model_key": model_key,
        "model": model,
        "dataset": dataset_name,
    }



def load_pipeline_config(path: Path | str = "config.json") -> PipelineConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    payload = _expect_dict(payload, "config")
    _expect_keys(payload, _ALLOWED_TOP, "config", required=set(_ALLOWED_TOP))

    config_root = config_path.parent

    paths = _expect_dict(payload.get("paths", {}), "paths")
    _expect_keys(
        paths,
        {
            "dataset_root",
            "artifacts_root",
            "configs_root",
            "targets_source_root",
        },
        "paths",
        required={"dataset_root", "artifacts_root"},
    )

    dataset_root = _resolve_path(config_root, str(paths["dataset_root"]))
    artifacts_root = _resolve_path(config_root, str(paths["artifacts_root"]))
    configs_root = _resolve_path(config_root, str(paths.get("configs_root", "configs/datasets")))
    targets_source_root = _resolve_path(config_root, str(paths.get("targets_source_root", "targets")))

    run = _expect_dict(payload.get("run", {}), "run")
    _expect_keys(run, {"seed", "run_id", "model_key", "model", "dataset"}, "run")
    run_norm = _normalize_run(run)

    dataset = _expect_dict(payload.get("dataset", {}), "dataset")
    _expect_keys(
        dataset,
        {
            "name",
            "augmented_subdir",
            "splits",
        },
        "dataset",
        required={"name", "augmented_subdir", "splits"},
    )

    generator = _expect_dict(payload.get("generator", {}), "generator")
    _expect_keys(generator, {"seed"}, "generator", required={"seed"})

    tuner = _expect_dict(payload.get("tuner", {}), "tuner")
    _expect_keys(
        tuner,
        {
            "enabled",
            "dataset",
            "coarse_epochs",
            "confirm_epochs",
            "vram_target_utilization",
            "batch_min",
            "batch_max_cap",
            "imgsz_candidates",
            "workers_candidates",
            "cache_candidates",
            "amp_candidates",
            "tf32_candidates",
            "cudnn_benchmark_candidates",
            "max_trials",
            "artifacts_subdir",
        },
        "tuner",
        required={
            "enabled",
            "dataset",
            "coarse_epochs",
            "confirm_epochs",
            "vram_target_utilization",
            "batch_min",
            "batch_max_cap",
            "imgsz_candidates",
            "workers_candidates",
            "cache_candidates",
            "amp_candidates",
            "tf32_candidates",
            "cudnn_benchmark_candidates",
            "max_trials",
            "artifacts_subdir",
        },
    )

    train = _expect_dict(payload.get("train", {}), "train")
    _expect_keys(train, {
        "epochs", "imgsz", "batch", "batch_mode", "batch_max", "batch_utilization_target", "oom_backoff_factor",
        "workers", "workers_auto", "workers_max", "patience", "cache", "throughput_mode", "device",
        "optimizer", "lr0", "lrf", "weight_decay", "warmup_epochs", "cos_lr",
        "close_mosaic", "mosaic", "mixup", "degrees", "translate", "scale", "shear", "perspective",
        "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "copy_paste", "multi_scale", "freeze",
        "amp", "plots", "tf32", "cudnn_benchmark",
        "dino_root", "dino_distill_warmup_epochs", "dino_distill_layers", "dino_distill_channels",
        "dino_distill_object_weight", "dino_distill_background_weight",
        "stage_a_ratio", "stage_a_freeze", "stage_a_distill_weight", "stage_b_distill_weight",
        "dino_viz_enabled", "dino_viz_mode", "dino_viz_every_n_epochs", "dino_viz_max_samples",
        "wandb_enabled", "wandb_project", "wandb_entity", "wandb_run_name", "wandb_tags", "wandb_notes",
        "wandb_mode", "wandb_log_system_metrics", "wandb_log_every_epoch",
        "eval_enabled", "periodic_eval_mode", "periodic_eval_sparse_epochs", "eval_interval_epochs", "eval_iou_threshold", "eval_conf_threshold",
        "eval_viz_samples", "eval_viz_split",
        "tuned_gpu_signature", "tuned_at_utc", "tuned_by", "tuned_profile_path",
    }, "train", required={
        "epochs", "imgsz", "batch", "batch_mode", "batch_max", "batch_utilization_target", "oom_backoff_factor",
        "workers", "workers_auto", "workers_max", "patience", "cache", "throughput_mode", "device",
        "optimizer", "lr0", "lrf", "weight_decay", "warmup_epochs", "cos_lr",
        "close_mosaic", "mosaic", "mixup", "degrees", "translate", "scale", "shear", "perspective",
        "hsv_h", "hsv_s", "hsv_v", "fliplr", "flipud", "copy_paste", "multi_scale", "freeze",
        "amp", "plots", "tf32", "cudnn_benchmark",
        "dino_root", "dino_distill_warmup_epochs", "dino_distill_layers", "dino_distill_channels",
        "dino_distill_object_weight", "dino_distill_background_weight",
        "stage_a_ratio", "stage_a_freeze", "stage_a_distill_weight", "stage_b_distill_weight",
        "dino_viz_enabled", "dino_viz_mode", "dino_viz_every_n_epochs", "dino_viz_max_samples",
        "wandb_enabled", "wandb_project", "wandb_entity", "wandb_run_name", "wandb_tags", "wandb_notes",
        "wandb_mode", "wandb_log_system_metrics", "wandb_log_every_epoch",
        "eval_enabled", "periodic_eval_mode", "periodic_eval_sparse_epochs", "eval_interval_epochs", "eval_iou_threshold", "eval_conf_threshold",
        "eval_viz_samples", "eval_viz_split",
        "tuned_gpu_signature", "tuned_at_utc", "tuned_by", "tuned_profile_path",
    })

    infer = _expect_dict(payload.get("infer", {}), "infer")
    _expect_keys(
        infer,
        {"imgsz", "device", "conf_threshold", "iou_threshold", "splits", "save_empty", "batch_size"},
        "infer",
        required={"imgsz", "device", "conf_threshold", "iou_threshold", "splits", "save_empty", "batch_size"},
    )

    grade = _expect_dict(payload.get("grade", {}), "grade")
    _expect_keys(
        grade,
        {
            "splits", "imgsz", "device", "conf_threshold", "infer_iou_threshold", "match_iou_threshold",
            "strict_obb", "max_samples", "calibrate_confidence", "calibration_candidates",
            "weights_json", "run_inference",
        },
        "grade",
        required={
            "splits", "imgsz", "device", "conf_threshold", "infer_iou_threshold", "match_iou_threshold",
            "strict_obb", "max_samples", "calibrate_confidence", "calibration_candidates",
            "weights_json", "run_inference",
        },
    )

    review = _expect_dict(payload.get("review", {}), "review")
    _expect_keys(review, {"split", "conf_threshold"}, "review", required={"split", "conf_threshold"})

    checks = _expect_dict(payload.get("checks", {}), "checks")
    _expect_keys(
        checks,
        {"outlier_threshold_px", "debug_overlays_per_split", "gui", "seed"},
        "checks",
        required={"outlier_threshold_px", "debug_overlays_per_split", "gui", "seed"},
    )

    profile = _expect_dict(payload.get("profile", {}), "profile")
    _expect_keys(
        profile,
        {"dataset", "train_epochs", "enable_gpu_sampling", "baseline_run", "regression_gate"},
        "profile",
        required={"dataset", "train_epochs", "enable_gpu_sampling"},
    )
    if "regression_gate" in profile:
        gate = _expect_dict(profile.get("regression_gate", {}), "profile.regression_gate")
        _expect_keys(
            gate,
            {
                "min_run_grade_delta",
                "max_precision_drop",
                "max_recall_drop",
                "max_miss_rate_increase",
                "max_total_duration_increase_pct",
            },
            "profile.regression_gate",
        )

    paths_norm = {
        "dataset_root": dataset_root,
        "artifacts_root": artifacts_root,
        "configs_root": configs_root,
        "targets_source_root": targets_source_root,
    }

    return PipelineConfig(
        config_path=config_path,
        config_root=config_root,
        paths=paths_norm,
        run=run_norm,
        dataset=dataset,
        generator=generator,
        tuner=tuner,
        train=train,
        infer=infer,
        grade=grade,
        review=review,
        checks=checks,
        profile=profile,
    )
