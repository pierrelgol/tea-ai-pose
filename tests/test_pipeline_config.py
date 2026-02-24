from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline_config import build_layout, load_pipeline_config


def _minimal_config_with_optional_sections() -> dict:
    return {
        "paths": {
            "dataset_root": "dataset",
            "artifacts_root": "artifacts/models",
        },
        "run": {"dataset": "coco1024", "model": "m", "model_key": "k", "run_id": "r", "seed": 42, "task": "pose"},
        "dataset": {"name": "coco1024", "augmented_subdir": "augmented", "splits": ["train", "val"]},
        "tuner": {
            "enabled": True,
            "dataset": "coco128",
            "coarse_epochs": 10,
            "confirm_epochs": 20,
            "vram_target_utilization": 0.92,
            "batch_min": 1,
            "batch_max_cap": 64,
            "imgsz_candidates": [64, 96],
            "workers_candidates": [0, 1],
            "cache_candidates": ["disk"],
            "amp_candidates": [False],
            "tf32_candidates": [False],
            "cudnn_benchmark_candidates": [False],
            "max_trials": 5,
            "artifacts_subdir": "tuner",
        },
        "train": {
            "epochs": 1, "imgsz": 64, "batch": 1, "batch_mode": "fixed", "batch_max": 2, "batch_utilization_target": 0.9,
            "oom_backoff_factor": 0.85, "workers": 0, "workers_auto": False, "workers_max": 4, "patience": 1,
            "cache": "auto", "throughput_mode": "balanced", "device": "cpu",
            "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0, "warmup_epochs": 0.0, "cos_lr": False,
            "close_mosaic": 0, "mosaic": 0.0, "mixup": 0.0, "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0,
            "perspective": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "fliplr": 0.0, "flipud": 0.0, "copy_paste": 0.0,
            "multi_scale": False, "freeze": None, "amp": False, "plots": False, "tf32": False, "cudnn_benchmark": False,
            "dino_root": "dinov3", "dino_distill_warmup_epochs": 0, "dino_distill_layers": [1], "dino_distill_channels": 8,
            "dino_distill_object_weight": 0.0, "dino_distill_background_weight": 0.0, "stage_a_ratio": 0.5, "stage_a_freeze": 0,
            "stage_a_distill_weight": 0.0, "stage_b_distill_weight": 0.0, "dino_viz_enabled": False, "dino_viz_mode": "off",
            "dino_viz_every_n_epochs": 1, "dino_viz_max_samples": 1, "wandb_enabled": False, "wandb_project": "p",
            "wandb_entity": None, "wandb_run_name": None, "wandb_tags": [], "wandb_notes": None, "wandb_mode": "offline",
            "wandb_log_system_metrics": False, "wandb_log_every_epoch": False, "eval_enabled": False,
            "periodic_eval_mode": "off", "periodic_eval_sparse_epochs": 10, "eval_interval_epochs": 1,
            "eval_iou_threshold": 0.5, "eval_conf_threshold": 0.5, "eval_viz_samples": 0, "eval_viz_split": "val",
            "tuned_gpu_signature": None, "tuned_at_utc": None, "tuned_by": None, "tuned_profile_path": None,
        },
        "infer": {
            "imgsz": 64,
            "device": "cpu",
            "conf_threshold": 0.25,
            "iou_threshold": 0.7,
            "splits": ["val"],
            "save_empty": True,
            "batch_size": 4,
        },
        "grade": {
            "splits": ["val"], "imgsz": 64, "device": "cpu", "conf_threshold": 0.25, "infer_iou_threshold": 0.7,
            "match_oks_threshold": 0.5, "max_samples": None, "calibrate_confidence": False,
            "fpr_negative_set_enabled": True, "fpr_threshold": 0.05,
            "calibration_candidates": None, "weights_json": None, "run_inference": False
        },
    }


def test_load_pipeline_config_rejects_unknown_top_key(tmp_path) -> None:
    payload = _minimal_config_with_optional_sections()
    payload["extra"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown keys"):
        load_pipeline_config(path)


def test_build_layout_paths() -> None:
    layout = build_layout(artifacts_root=Path("artifacts/models"), model_key="mk", run_id="rid")
    assert str(layout.run_root).endswith("artifacts/models/mk/runs/rid")
    assert str(layout.eval_epoch_root(3)).endswith("artifacts/models/mk/runs/rid/eval/epoch_003")
