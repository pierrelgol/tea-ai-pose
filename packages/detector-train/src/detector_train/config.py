from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass(slots=True)
class TrainConfig:
    dataset_root: Path
    artifacts_root: Path
    model: str
    name: str
    seed: int

    device: str = "auto"
    project: Path = Path("artifacts/models/default/runs/current/train/ultralytics")

    epochs: int = 128
    imgsz: int = 512
    batch: int = 16
    batch_mode: str = "auto_max"  # fixed|auto_max
    batch_max: int = 64
    batch_utilization_target: float = 0.92
    oom_backoff_factor: float = 0.85
    workers: int = 16
    workers_auto: bool = True
    workers_max: int = 16
    patience: int = 30
    cache: str = "auto"
    throughput_mode: str = "balanced"  # balanced|max_gpu
    amp: bool = True
    plots: bool = True
    tf32: bool = True
    cudnn_benchmark: bool = True

    optimizer: str = "AdamW"
    lr0: float = 0.0012
    lrf: float = 0.01
    weight_decay: float = 0.0006
    warmup_epochs: float = 6.0
    cos_lr: bool = True

    close_mosaic: int = 12
    mosaic: float = 0.5
    mixup: float = 0.03
    degrees: float = 1.0
    translate: float = 0.035
    scale: float = 0.35
    shear: float = 0.0
    perspective: float = 0.0
    hsv_h: float = 0.010
    hsv_s: float = 0.30
    hsv_v: float = 0.22
    fliplr: float = 0.5
    flipud: float = 0.0
    copy_paste: float = 0.0
    multi_scale: bool = False
    freeze: int | None = None
    dino_root: Path = Path("dinov3")
    dino_distill_warmup_epochs: int = 5
    dino_distill_layers: tuple[int, ...] = (19,)
    dino_distill_channels: int = 32
    dino_distill_object_weight: float = 1.15
    dino_distill_background_weight: float = 0.15
    stage_a_ratio: float = 0.30
    stage_a_freeze: int = 10
    stage_a_distill_weight: float = 0.25
    stage_b_distill_weight: float = 0.08
    dino_viz_enabled: bool = True
    dino_viz_mode: str = "final_only"  # off|interval|final_only
    dino_viz_every_n_epochs: int = 5
    dino_viz_max_samples: int = 1

    wandb_enabled: bool = True
    wandb_project: str = "tea-ai-detector"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: list[str] = field(default_factory=list)
    wandb_notes: str | None = None
    wandb_mode: str = "auto"  # online|offline|auto
    wandb_log_system_metrics: bool = False
    wandb_log_every_epoch: bool = True

    eval_enabled: bool = True
    periodic_eval_mode: str = "interval"  # off|interval|sparse
    periodic_eval_sparse_epochs: int = 10
    eval_interval_epochs: int = 2
    eval_iou_threshold: float = 0.75
    eval_conf_threshold: float = 0.9
    eval_viz_samples: int = 0
    eval_viz_split: str = "val"

    save_json: bool = True

    def validate(self) -> None:
        project_resolved = self.project if self.project.is_absolute() else (Path.cwd() / self.project)
        project_resolved = project_resolved.resolve()
        cwd_resolved = Path.cwd().resolve()
        if not project_resolved.is_relative_to(cwd_resolved):
            raise ValueError("project path must be inside the repository working directory")

        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.imgsz < 32:
            raise ValueError("imgsz must be >= 32")
        if self.batch < 1:
            raise ValueError("batch must be >= 1")
        if self.batch_mode not in {"fixed", "auto_max"}:
            raise ValueError("batch_mode must be one of: fixed, auto_max")
        if self.batch_max < 1:
            raise ValueError("batch_max must be >= 1")
        if self.batch_utilization_target <= 0 or self.batch_utilization_target > 1:
            raise ValueError("batch_utilization_target must be in (0,1]")
        if self.oom_backoff_factor <= 0 or self.oom_backoff_factor >= 1:
            raise ValueError("oom_backoff_factor must be in (0,1)")
        if self.workers < 0:
            raise ValueError("workers must be >= 0")
        if self.workers_max < 1:
            raise ValueError("workers_max must be >= 1")
        if self.patience < 0:
            raise ValueError("patience must be >= 0")
        if self.cache not in {"auto", "false", "ram", "disk"}:
            raise ValueError("cache must be one of: auto, false, ram, disk")
        if self.throughput_mode not in {"balanced", "max_gpu"}:
            raise ValueError("throughput_mode must be one of: balanced, max_gpu")
        if self.optimizer not in {"SGD", "AdamW", "auto"}:
            raise ValueError("optimizer must be one of: SGD, AdamW, auto")
        if self.lr0 <= 0:
            raise ValueError("lr0 must be > 0")
        if self.lrf <= 0:
            raise ValueError("lrf must be > 0")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.warmup_epochs < 0:
            raise ValueError("warmup_epochs must be >= 0")
        if self.close_mosaic < 0:
            raise ValueError("close_mosaic must be >= 0")
        if self.freeze is not None and self.freeze < 0:
            raise ValueError("freeze must be >= 0")
        if self.dino_distill_warmup_epochs < 0:
            raise ValueError("dino_distill_warmup_epochs must be >= 0")
        if not self.dino_distill_layers:
            raise ValueError("dino_distill_layers must contain at least one layer index")
        if self.dino_distill_channels < 8:
            raise ValueError("dino_distill_channels must be >= 8")
        if self.dino_distill_object_weight < 0:
            raise ValueError("dino_distill_object_weight must be >= 0")
        if self.dino_distill_background_weight < 0:
            raise ValueError("dino_distill_background_weight must be >= 0")
        if self.stage_a_ratio <= 0 or self.stage_a_ratio >= 1:
            raise ValueError("stage_a_ratio must be in (0,1)")
        if self.stage_a_freeze < 0:
            raise ValueError("stage_a_freeze must be >= 0")
        if self.stage_a_distill_weight < 0:
            raise ValueError("stage_a_distill_weight must be >= 0")
        if self.stage_b_distill_weight < 0:
            raise ValueError("stage_b_distill_weight must be >= 0")
        if self.dino_viz_mode not in {"off", "interval", "final_only"}:
            raise ValueError("dino_viz_mode must be one of: off, interval, final_only")
        if self.dino_viz_every_n_epochs < 1:
            raise ValueError("dino_viz_every_n_epochs must be >= 1")
        if self.dino_viz_max_samples < 1:
            raise ValueError("dino_viz_max_samples must be >= 1")
        if self.wandb_mode not in {"online", "offline", "auto"}:
            raise ValueError("wandb_mode must be one of: online, offline, auto")
        if self.periodic_eval_mode not in {"off", "interval", "sparse"}:
            raise ValueError("periodic_eval_mode must be one of: off, interval, sparse")
        if self.periodic_eval_sparse_epochs < 1:
            raise ValueError("periodic_eval_sparse_epochs must be >= 1")
        if self.eval_interval_epochs < 1:
            raise ValueError("eval_interval_epochs must be >= 1")
        if self.eval_iou_threshold < 0 or self.eval_iou_threshold > 1:
            raise ValueError("eval_iou_threshold must be in [0,1]")
        if self.eval_conf_threshold < 0 or self.eval_conf_threshold > 1:
            raise ValueError("eval_conf_threshold must be in [0,1]")
        if self.eval_viz_samples < 0:
            raise ValueError("eval_viz_samples must be >= 0")
        if self.eval_viz_split not in {"train", "val"}:
            raise ValueError("eval_viz_split must be one of: train, val")
