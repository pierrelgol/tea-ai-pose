# TEA-AI: Targeted Extraction & Augmentation for Oriented Bounding Box Inference

Production-grade pipeline for fine-tuning YOLO OBB (Oriented Bounding Box) detectors with curriculum-based synthetic data generation and DINOv3 feature distillation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              TEA-AI PIPELINE                                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│   INPUTS                                                                                │
│   ├── Background Dataset (COCO, etc.)                                                   │
│   └── Target Objects (canonical images with OBB labels)                                 │
│          │                                                                              │
│          ▼                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         DATA PREPARATION STAGES                                 │   │
│   │                                                                                 │   │
│   │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐                  │   │
│   │  │ dataset-fetcher │  │ dinov3-fetcher  │  │ target-labeller │                  │   │
│   │  │ ─────────────── │  │ ─────────────── │  │ ─────────────── │                  │   │
│   │  │ • Download      │  │ • HF Hub        │  │ • Qt GUI        │                  │   │
│   │  │ • Validate      │  │ • Cache teacher │  │ • OBB annotate  │                  │   │
│   │  │ • Subset        │  │ • Local storage │  │ • Export        │                  │   │
│   │  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘                  │   │
│   │           │                    │                    │                           │   │
│   │           ▼                    ▼                    ▼                           │   │
│   │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│   │  │                    dataset-generator                                    │    │   │
│   │  │ ─────────────────────────────────────────────────────────────────────── │    │   │
│   │  │  Homography + Photometric Synthesis → Synthetic OBB Dataset             │    │   │
│   │  │  • Curriculum-adaptive difficulty                                       │    │   │
│   │  │  • Multi-target placement with occlusion handling                       │    │   │
│   │  │  • Class-balancing with hard-example boosting                           │    │   │
│   │  └────────┬────────────────────────────────────────────────────────────────┘    │   │
│   │           │                                                                     │   │
│   │           ▼                                                                     │   │
│   │  ┌─────────────────┐                                                            │   │
│   │  │ augment-checker │  Integrity validation before training                      │   │
│   │  │ ─────────────── │  • H-matrix verification                                   │   │
│   │  │ • Geometry QA   │  • Corner outlier detection                                │   │
│   │  │ • Format check  │  • Label consistency                                       │   │
│   │  └────────┬────────┘                                                            │   │
│   └───────────┼─────────────────────────────────────────────────────────────────────┘   │
│               │                                                                         │
│               ▼                                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                           TRAINING STAGE                                        │   │
│   │                                                                                 │   │
│   │  ┌─────────────────────────────────────────────────────────────────────────┐    │   │
│   │  │                        detector-train                                   │    │   │
│   │  │ ─────────────────────────────────────────────────────────────────────── │    │   │
│   │  │                                                                         │    │   │
│   │  │   YOLO OBB Student  ◄──────────────────  DINOv3 Teacher (frozen)        │    │   │
│   │  │          │                                    │                         │    │   │
│   │  │          │  ┌────────────────────────────────┐│                         │    │   │
│   │  │          │  │      FEATURE DISTILLATION      ││                         │    │   │
│   │  │          │  │ • Layer-wise feature alignment ││                         │    │   │
│   │  │          │  │ • OBB-masked cosine distance   ││                         │    │   │
│   │  │          │  │ • Two-stage curriculum         ││                         │    │   │
│   │  │          │  │   - Stage A: Frozen backbone   ││                         │    │   │
│   │  │          │  │   - Stage B: Full finetune     ││                         │    │   │
│   │  │          │  └────────────────────────────────┘│                         │    │   │
│   │  │          ▼                                    │                         │    │   │
│   │  │   Checkpoints (best.pt, best_geo.pt)          │                         │    │   │
│   │  │          │                                    │                         │    │   │
│   │  │          └───▶  Periodic Eval (detector-grader)                         │    │   │
│   │  └──────────────────────┬──────────────────────────────────────────────────┘    │   │
│   └─────────────────────────┼───────────────────────────────────────────────────────┘   │
│                             │                                                           │
│                             ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         EVALUATION STAGES                                       │   │
│   │                                                                                 │   │
│   │  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────────┐             │   │
│   │  │ detector-infer  │───▶│ detector-grader │───▶│ detector-reviewer│             │   │
│   │  │ ─────────────── │    │ ─────────────── │    │ ──────────────── │             │   │
│   │  │ • Batch predict │    │ • 5-axis scoring│    │ • Qt GUI         │             │   │
│   │  │ • YOLO OBB fmt  │    │ • IoU/Corner/   │    │ • GT vs Pred     │             │   │
│   │  │ • Empty handling│    │   Angle/Center/ │    │   overlay        │             │   │
│   │  │                 │    │   Shape         │    │ • Visual QA      │             │   │
│   │  │                 │    │ • Hard example  │    │                  │             │   │
│   │  │                 │    │   extraction    │    │                  │             │   │
│   │  │                 │    │ • Grade reports │    │                  │             │   │
│   │  │                 │    │   → curriculum  │    │                  │             │   │
│   │  └─────────────────┘    └─────────────────┘    └──────────────────┘             │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                         │
│   INFRASTRUCTURE                                                                        │
│   ┌─────────────────┐  ┌─────────────────────────┐  ┌─────────────────────────┐         │
│   │ pipeline-config │  │ pipeline-runtime-utils  │  │   pipeline-profile      │         │
│   │ ─────────────── │  │ ─────────────────────── │  │ ─────────────────────── │         │
│   │ • Schema        │  │ • Device resolution     │  │ • End-to-end timing     │         │
│   │ • Path layout   │  │ • Seeding               │  │ • Resource monitoring   │         │
│   │ • Validation    │  │ • Geometry utilities    │  │ • Bottleneck analysis   │         │
│   └─────────────────┘  └─────────────────────────┘  └─────────────────────────┘         │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Setup environment
uv sync --all-packages

# 2. Run full pipeline
just fetch-dataset      # Download background dataset
just fetch-dinov3       # Download DINOv3 teacher model
just label-targets      # Annotate target objects (GUI)
just generate-dataset   # Synthesize training data
just check-dataset      # Validate augmented data
just tune-gpu           # One-time GPU tuning for safe max utilization
just train              # Train with distillation
just optimize           # Export + compile TensorRT engine
just eval               # Run inference + grading
just review             # Visual inspection (GUI)

# 3. Profile pipeline performance
just profile-pipeline
```

## Configuration

Single source of truth in `config.json`:

```json
{
  "paths": {
    "dataset_root": "dataset",
    "artifacts_root": "artifacts/models",
    "targets_source_root": "targets"
  },
  "run": {
    "dataset": "coco128",
    "model_key": "yolo26n_obb",
    "seed": 42
  },
  "train": {
    "epochs": 128,
    "dino_distill_layers": [19],
    "stage_a_ratio": 0.3,
    "stage_a_freeze": 10
  }
}
```

All stages read from the same config. See `pipeline-config` for schema details.

## Package Reference

| Package | Purpose | Stage |
|---------|---------|-------|
| `pipeline-config` | Schema validation & path layout | Foundation |
| `pipeline-runtime-utils` | Device, seeding, geometry | Shared |
| `dataset-fetcher` | Dataset acquisition | 1 |
| `dinov3-fetcher` | Teacher model download | 2 |
| `target-labeller` | OBB annotation GUI | 3 |
| `dataset-generator` | Synthetic data synthesis | 4 |
| `augment-checker` | Data validation | 5 |
| `gpu-auto-tuner` | Offline GPU tuning and config writeback | 6 |
| `detector-train` | YOLO + DINOv3 training | 7 |
| `detector-optimize` | ONNX export + TensorRT compile | 8 |
| `detector-infer` | Batch inference | 9 |
| `detector-grader` | Geometric evaluation | 10 |
| `detector-reviewer` | Visual review GUI | 11 |
| `pipeline-profile` | Performance profiling | - |
| `dinov3` | Teacher model bridge | Component |

## Data Flow

```
Raw Targets ──▶ target-labeller ──▶ Canonical Targets (targets/)
                                                        │
COCO Dataset ──▶ dataset-fetcher ──▶ Backgrounds ──────┤
                                                        ▼
DINOv3 Model ──▶ dinov3-fetcher ──▶ Teacher ───────▶ dataset-generator
                                                        │
                                                        ▼
                                              Synthetic Dataset (dataset/augmented/)
                                                        │
                                                        ▼
                                              detector-train ──▶ Checkpoints
                                                        │
                                                        ▼
                                              detector-infer ──▶ Predictions
                                                        │
                                                        ▼
                                              detector-grader ──▶ Grades
                                                         │
                                                         ▼
                                              ┌────────────────────┐
                                              │  Feedback Loops    │
                                              │ • Curriculum update│
                                              │ • Hard examples    │
                                              └────────────────────┘
```

## Key Technical Concepts

### OBB (Oriented Bounding Box)
Unlike axis-aligned HBB, OBB uses 4 corner points `(x1,y1,x2,y2,x3,y3,x4,y4)` allowing arbitrary rotation. Essential for tightly fitting rotated objects.

### Homographic Projection
The generator uses 3×3 homography matrices to project flat target images into perspective views:
```
[x']   [h11 h12 h13] [x]
[y'] = [h21 h22 h23] [y]
[w']   [h31 h32 h33] [1]
```

### DINOv3 Feature Distillation
Teacher (DINOv3 ViT) provides high-quality visual features. Student (YOLO) learns to match these features via masked cosine distance loss, improving generalization especially for orientation estimation.

### Curriculum Learning
Generation difficulty adapts based on grader reports:
- **Mild**: Low perspective, minimal occlusion
- **Medium**: Moderate variation
- **Hard**: Extreme angles, heavy occlusion

## Directory Structure

```
.
├── config.json              # Master configuration
├── Justfile                 # Pipeline commands
├── dataset/                 # Generated datasets
│   ├── coco128/            # Background dataset
│   └── augmented/          # Synthetic training data
│       ├── images/
│       ├── labels/
│       └── meta/
├── targets/                 # Canonical target objects
│   ├── images/
│   ├── labels/
│   └── classes.txt
├── artifacts/models/        # Training outputs
│   └── {model_key}/
│       └── runs/{run_id}/
│           ├── train/weights/
│           │   ├── best.pt
│           │   └── best_geo.pt
│           ├── infer/labels/
│           ├── grade/reports/
│           └── eval/
├── dinov3/                  # Cached teacher model
│   ├── model.safetensors
│   └── config.json
└── packages/                # Source packages
    └── */
        └── src/
```

## Development

```bash
# Build all packages
just build

# Clean artifacts
just clean        # Keep venv
just fclean       # Full clean

# Run individual stages
uv run dataset-fetcher --config config.json
uv run detector-train --config config.json
```

## Requirements

- Python 3.12+
- CUDA 11.8+ (or MPS on Apple Silicon)
- 8GB+ VRAM for training (4GB for inference only)
- See `pyproject.toml` for full dependencies
