# pipeline-config

Centralized configuration schema, validation, and artifact layout management. Provides the single source of truth for all pipeline stages.

## Design Philosophy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DESIGN PRINCIPLES                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. SINGLE SOURCE OF TRUTH                                                   │
│     One config.json drives all stages. No per-stage configs.                 │
│                                                                              │
│  2. FAIL-FAST VALIDATION                                                     │
│     Schema errors caught at load time, not runtime.                          │
│                                                                              │
│  3. TYPE SAFETY                                                              │
│     Frozen dataclasses with slots. No accidental mutation.                   │
│                                                                              │
│  4. DETERMINISTIC PATHS                                                      │
│     Absolute path resolution relative to config location.                    │
│                                                                              │
│  5. EXTENSIBILITY                                                            │
│     Dict-based sections allow new parameters without schema changes.         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Architecture

```
config.json
     │
     ▼
load_pipeline_config() ──▶ PipelineConfig
     │                         │
     │                         ├── paths: dict
     │                         ├── run: dict
     │                         ├── dataset: dict
     │                         ├── generator: dict
     │                         ├── train: dict
     │                         ├── infer: dict
     │                         ├── grade: dict
     │                         ├── review: dict
     │                         ├── checks: dict
     │                         └── profile: dict
     │
     ▼
build_layout() ──▶ PipelineLayout
     │
     ├── layout.model_root      artifacts/{model_key}/
     ├── layout.run_root        artifacts/{model_key}/runs/{run_id}/
     ├── layout.train_root      .../train/
     ├── layout.infer_root      .../infer/
     ├── layout.grade_root      .../grade/
     ├── layout.eval_root       .../eval/
     └── layout.review_root     .../review/
```

## Schema Reference

### paths (required)

Directory roots for the pipeline.

```json
{
  "paths": {
    "dataset_root": "./dataset",
    "artifacts_root": "./artifacts/models",
    "configs_root": "configs/datasets",
    "targets_source_root": "targets"
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `dataset_root` | string | Location of datasets (fetched and generated) |
| `artifacts_root` | string | Training outputs, checkpoints, reports |
| `configs_root` | string | Dataset profile definitions |
| `targets_source_root` | string | Raw target images for labeling |

**Path Resolution:** Relative paths resolve against the config file's directory.

```python
# With config at /project/config.json:
# "dataset_root": "./dataset" → /project/dataset
```

### run

Pipeline execution metadata.

```json
{
  "run": {
    "dataset": "coco128",
    "model": "hf-openvision-yolo26-n-obb",
    "model_key": "yolo26n_obb",
    "run_id": "current",
    "seed": 42
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset` | string | — | Dataset profile name |
| `model` | string | — | Base model identifier |
| `model_key` | string | "default" | Short key for artifacts |
| `run_id` | string | timestamp | Unique run identifier |
| `seed` | int | 42 | Global RNG seed |

### dataset

Dataset organization.

```json
{
  "dataset": {
    "name": "coco128",
    "augmented_subdir": "augmented",
    "splits": ["train", "val"]
  }
}
```

### generator

Synthetic data generation parameters.

```json
{
  "generator": {
    "seed": 42
  }
}
```

### tuner

One-time offline GPU tuning parameters (used by `gpu-auto-tuner` before training).

```json
{
  "tuner": {
    "enabled": true,
    "dataset": "coco128",
    "coarse_epochs": 10,
    "confirm_epochs": 20,
    "vram_target_utilization": 0.92,
    "batch_min": 1,
    "batch_max_cap": 64,
    "imgsz_candidates": [512, 448, 384],
    "workers_candidates": [4, 8, 12, 16],
    "cache_candidates": ["ram", "disk"],
    "amp_candidates": [true, false],
    "tf32_candidates": [true, false],
    "cudnn_benchmark_candidates": [true, false],
    "max_trials": 30,
    "artifacts_subdir": "tuner"
  }
}
```

### train

Training hyperparameters including DINOv3 distillation.

```json
{
  "train": {
    "epochs": 128,
    "imgsz": 512,
    "batch": 16,
    "workers": 8,
    "patience": 30,
    "cache": "auto",
    "device": "auto",
    "optimizer": "AdamW",
    "lr0": 0.0012,
    "lrf": 0.01,
    "weight_decay": 0.0006,
    "warmup_epochs": 6.0,
    "cos_lr": true,
    "close_mosaic": 12,
    "mosaic": 0.5,
    "mixup": 0.03,
    "degrees": 1.0,
    "translate": 0.035,
    "scale": 0.35,
    "shear": 0.0,
    "perspective": 0.0,
    "hsv_h": 0.01,
    "hsv_s": 0.3,
    "hsv_v": 0.22,
    "fliplr": 0.5,
    "flipud": 0.0,
    "copy_paste": 0.0,
    "multi_scale": false,
    "freeze": null,
    "amp": true,
    "plots": true,
    "tf32": true,
    "cudnn_benchmark": true,
    "dino_root": "dinov3",
    "dino_distill_warmup_epochs": 5,
    "dino_distill_layers": [19],
    "dino_distill_channels": 32,
    "dino_distill_object_weight": 1.15,
    "dino_distill_background_weight": 0.15,
    "stage_a_ratio": 0.3,
    "stage_a_freeze": 10,
    "stage_a_distill_weight": 0.25,
    "stage_b_distill_weight": 0.08,
    "dino_viz_enabled": true,
    "dino_viz_every_n_epochs": 5,
    "dino_viz_max_samples": 4,
    "wandb_enabled": true,
    "wandb_project": "tea-ai-detector",
    "wandb_entity": null,
    "wandb_run_name": null,
    "wandb_tags": [],
    "wandb_notes": null,
    "wandb_mode": "auto",
    "wandb_log_system_metrics": false,
    "wandb_log_every_epoch": true,
    "eval_enabled": true,
    "eval_interval_epochs": 2,
    "eval_iou_threshold": 0.75,
    "eval_conf_threshold": 0.9,
    "eval_viz_samples": 8,
    "eval_viz_split": "val",
    "tuned_gpu_signature": null,
    "tuned_at_utc": null,
    "tuned_by": null,
    "tuned_profile_path": null
  }
}
```

#### Distillation Parameters

| Parameter | Description |
|-----------|-------------|
| `dino_root` | Path to cached DINOv3 model |
| `dino_distill_layers` | Student layer indices to distill |
| `dino_distill_channels` | Projection channels for feature alignment |
| `dino_distill_object_weight` | Loss weight for OBB regions |
| `dino_distill_background_weight` | Loss weight for background regions |
| `stage_a_ratio` | Fraction of epochs for stage A |
| `stage_a_freeze` | Number of backbone layers frozen in stage A |
| `stage_a_distill_weight` | Distillation weight during stage A |
| `stage_b_distill_weight` | Distillation weight during stage B |

### infer

Inference configuration.

```json
{
  "infer": {
    "imgsz": 640,
    "device": "auto",
    "conf_threshold": 0.25,
    "iou_threshold": 0.7,
    "splits": ["val"],
    "save_empty": true,
    "batch_size": 16
  }
}
```

### grade

Evaluation and grading configuration.

```json
{
  "grade": {
    "splits": ["val"],
    "imgsz": 640,
    "device": "auto",
    "conf_threshold": 0.25,
    "infer_iou_threshold": 0.7,
    "match_iou_threshold": 0.5,
    "strict_obb": true,
    "max_samples": null,
    "calibrate_confidence": true,
    "calibration_candidates": null,
    "weights_json": null,
    "run_inference": true
  }
}
```

### review

Visual review GUI settings.

```json
{
  "review": {
    "split": "val",
    "conf_threshold": 0.25
  }
}
```

### checks

Dataset validation thresholds.

```json
{
  "checks": {
    "outlier_threshold_px": 2.0,
    "debug_overlays_per_split": 10,
    "gui": false,
    "seed": 42
  }
}
```

### profile

Pipeline profiling parameters.

```json
{
  "profile": {
    "dataset": "coco128",
    "train_epochs": 50,
    "enable_gpu_sampling": true
  }
}
```

## Validation Rules

### Strict Key Checking

Unknown keys raise `ValueError`:

```python
# Raises: ValueError: unknown keys in config: ['invalid_section']
{
  "invalid_section": {}
}
```

### Required Fields

Missing required fields raise `ValueError`:

```python
# Raises: ValueError: missing required keys in paths: ['artifacts_root']
{
  "paths": {
    "dataset_root": "./dataset"
  }
}
```

### Type Coercion

Numeric values are coerced; invalid types raise `TypeError`.

## PipelineLayout

Deterministic directory structure for all artifacts:

```
artifacts_root/
└── {model_key}/
    └── runs/{run_id}/
        ├── train/
        │   ├── weights/
        │   │   ├── best.pt              # YOLO best (mAP-based)
        │   │   ├── best_geo.pt          # Best by geometric grade
        │   │   └── last.pt              # Final checkpoint
        │   └── results.csv              # Training metrics
        ├── train/ultralytics/           # Framework cache
        ├── infer/
        │   └── labels/
        │       ├── train/
        │       └── val/
        ├── grade/
        │   └── reports/
        │       ├── grade_report_*.json
        │       └── hard_examples.jsonl
        ├── eval/
        │   └── epoch_{nnn}/
        │       ├── predictions/
        │       └── reports/
        ├── review/
        └── meta/
```

### Layout Accessors

```python
from pipeline_config import load_pipeline_config, build_layout

config = load_pipeline_config("config.json")

layout = build_layout(
    artifacts_root=config.paths["artifacts_root"],
    model_key=config.run["model_key"],
    run_id=config.run["run_id"]
)

# Directory accessors
layout.model_root              # artifacts/{model_key}/
layout.run_root                # artifacts/{model_key}/runs/{run_id}/
layout.train_root              # .../train/
layout.train_weights_root      # .../train/weights/
layout.infer_root              # .../infer/
layout.grade_root              # .../grade/
layout.eval_root               # .../eval/
layout.review_root             # .../review/

# Dynamic paths
layout.eval_epoch_root(5)      # .../eval/epoch_005/
```

## Usage

```python
from pipeline_config import load_pipeline_config, build_layout
from pathlib import Path

# Load and validate
config = load_pipeline_config("config.json")

# Access configuration
seed = config.run["seed"]
dataset_name = config.dataset["name"]
epochs = config.train["epochs"]

# Build layout
layout = build_layout(
    artifacts_root=config.paths["artifacts_root"],
    model_key=config.run["model_key"],
    run_id=config.run["run_id"]
)

# Use paths
weights_dir = layout.train_weights_root
```

## API Reference

### `load_pipeline_config(path: str | Path) -> PipelineConfig`

Load and validate configuration from JSON file.

**Raises:**
- `FileNotFoundError`: Config file not found
- `ValueError`: Schema validation failed
- `TypeError`: Type coercion failed

### `build_layout(artifacts_root, model_key, run_id) -> PipelineLayout`

Create layout helper for artifact paths.

### `PipelineConfig`

Frozen dataclass with sections as dicts:

```python
@dataclass(frozen=True, slots=True)
class PipelineConfig:
    config_path: Path
    config_root: Path
    paths: dict
    run: dict
    dataset: dict
    generator: dict
    train: dict
    infer: dict
    grade: dict
    review: dict
    checks: dict
    profile: dict
```

### `PipelineLayout`

Path generator for artifact organization:

```python
@dataclass(frozen=True, slots=True)
class PipelineLayout:
    model_root: Path
    run_root: Path
    train_root: Path
    train_weights_root: Path
    infer_root: Path
    grade_root: Path
    eval_root: Path
    review_root: Path
    
    def eval_epoch_root(self, epoch: int) -> Path: ...
```

## Integration

All packages depend on `pipeline-config`:

```python
# In any package
from pipeline_config import load_pipeline_config

def main():
    config = load_pipeline_config(args.config)
    # Use config.train["epochs"], etc.
```

This ensures consistent configuration interpretation across the entire pipeline.
