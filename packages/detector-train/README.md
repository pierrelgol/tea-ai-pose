# detector-train

YOLO OBB training with DINOv3 feature distillation and two-stage curriculum.

## Purpose

Trains oriented bounding box detectors using a hybrid approach: standard YOLO detection loss combined with knowledge distillation from a frozen DINOv3 teacher model.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING ARCHITECTURE                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   DINOv3 Teacher (frozen)                    YOLO OBB Student                            │
│   facebook/dinov3-vitl16                     hf-openvision-yolo26-n-obb                 │
│          │                                              │                                │
│          │  ┌────────────────────────────────────┐     │                                │
│          └──┤    FEATURE DISTILLATION           │◄────┤                                │
│             │ ───────────────────────────────────│     │                                │
│             │                                    │     │                                │
│             │  Teacher Output                    │     │ Student Output                 │
│             │  [B, C_t, H_t, W_t]                │     │ [B, C_s, H_s, W_s]             │
│             │         │                          │     │         │                      │
│             │         │ Interpolate to H_t×W_t   │     │         │ Hook layers          │
│             │         ▼                          │     │         ▼                      │
│             │  [B, C_t, H_t, W_t]                │     │ [B, C_s, H_s, W_s]             │
│             │         │                          │     │         │                      │
│             │         │ Project channels         │     │         │ Project channels     │
│             │         ▼                          │     │         ▼                      │
│             │  [B, D, H_t, W_t]                  │     │ [B, D, H_t, W_t]               │
│             │         │                          │     │         │                      │
│             │         └──────────┬───────────────┘     │         │                      │
│             │                    │                     │         │                      │
│             │                    ▼                     │         │                      │
│             │           Masked Cosine Distance         │         │                      │
│             │           (OBB regions weighted)         │         │                      │
│             │                    │                     │         │                      │
│             │                    ▼                     │         │                      │
│             │            L_distill                     │         │                      │
│             └────────────────────────────────────┘     │                                │
│                                                        │                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                              TOTAL LOSS                                          │   │
│   │                                                                                  │   │
│   │   L_total = L_yolo + w_distill × L_distill                                     │   │
│   │                                                                                  │   │
│   │   L_yolo = L_box + L_cls + L_dfl      (standard YOLO)                          │   │
│   │   L_distill = w_obj × L_obj + w_bg × L_bg                                      │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Two-Stage Training

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING CURRICULUM                                         │
├──────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                       │
│   Stage A (0-30% of epochs)              Stage B (30%-100% of epochs)                │
│   ─────────────────────────              ─────────────────────────────                │
│                                                                                       │
│   ┌─────────────────────┐                ┌─────────────────────┐                      │
│   │ Backbone: FROZEN    │                │ Backbone: UNFROZEN  │                      │
│   │ Neck: TRAIN         │───────────────▶│ Neck: TRAIN         │                      │
│   │ Head: TRAIN         │                │ Head: TRAIN         │                      │
│   └─────────────────────┘                └─────────────────────┘                      │
│                                                                                       │
│   High distillation weight             Lower distillation weight                      │
│   w_distill = 0.25                     w_distill = 0.08                               │
│                                                                                       │
│   Purpose: Learn detection task        Purpose: Adapt backbone to domain              │
│   without destroying pretrained        while preserving DINO features                 │
│   ImageNet features                                                                   │
│                                                                                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

### Stage A (Freeze Backbone)

```python
stage_a_epochs = int(epochs * stage_a_ratio)  # ~30% of training
freeze = stage_a_freeze  # Number of backbone layers frozen
distill_weight = stage_a_distill_weight  # Typically 0.25
```

### Stage B (Full Finetuning)

```python
# Unfreeze all layers
distill_weight = stage_b_distill_weight  # Typically 0.08
```

## Feature Distillation Mechanism

### Teacher Feature Extraction

```python
from dinov3_bridge.model import DinoV3Teacher

# Load frozen teacher
teacher = DinoV3Teacher(cfg, device)
teacher.eval()

# Extract spatial features
with torch.no_grad():
    teacher_features = teacher.extract_feature_map(images)
    # Shape: [B, C_t, H_t, W_t]
```

### Student Feature Hooking

```python
# Hook specific layers during forward pass
captured_features = []

for layer_idx in distill_layers:
    handle = model.model[layer_idx].register_forward_hook(
        lambda m, inp, out: captured_features.append(out)
    )
    handles.append(handle)

# Forward pass
results = model(images)

# Collect features
student_features = captured_features[0]  # [B, C_s, H_s, W_s]
```

### Feature Alignment

```python
import torch.nn.functional as F

# Spatial alignment via interpolation
student_aligned = F.interpolate(
    student_features,
    size=teacher_features.shape[2:],  # Match teacher spatial
    mode='bilinear',
    align_corners=False
)

# Channel projection (if needed)
if student_aligned.shape[1] != teacher_features.shape[1]:
    projection = nn.Conv2d(
        student_aligned.shape[1],
        teacher_features.shape[1],
        kernel_size=1
    )
    student_aligned = projection(student_aligned)
```

### Masked Distillation Loss

```python
# Generate OBB mask (1 inside OBB, 0 outside)
mask = generate_obb_mask(gt_boxes, size=teacher_features.shape[2:])
# Shape: [B, 1, H_t, W_t]

# Cosine distance
similarity = F.cosine_similarity(
    teacher_features,
    student_aligned,
    dim=1
)  # [B, H_t, W_t]

# Masked loss (weighted by object/background)
loss_obj = (1 - similarity[mask]).mean() * object_weight
loss_bg = (1 - similarity[~mask]).mean() * background_weight
loss_distill = loss_obj + loss_bg
```

## Configuration

```json
{
  "train": {
    "epochs": 128,
    "imgsz": 512,
    "batch": 16,
    "workers": 8,
    "device": "auto",
    "optimizer": "AdamW",
    "lr0": 0.0012,
    "lrf": 0.01,
    "weight_decay": 0.0006,
    
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
    
    "wandb_enabled": true,
    "wandb_project": "tea-ai-detector",
    
    "eval_enabled": true,
    "eval_interval_epochs": 2,
    "eval_iou_threshold": 0.75,
    "eval_conf_threshold": 0.9
  }
}
```

## Periodic Evaluation

Runs grading during training to track geometric performance:

```python
if epoch % eval_interval_epochs == 0:
    # 1. Save checkpoint
    # 2. Run inference on val split
    # 3. Grade predictions with detector-grader
    grade = grader.run()
    
    # 4. Log to W&B
    wandb.log({"eval/grade": grade})
    
    # 5. Track best by grade (separate from YOLO's mAP-based best)
    if grade > best_grade:
        best_grade = grade
        save_checkpoint("best_geo.pt")
```

## Visualization

### DINO Feature Maps

Periodically saves teacher/student feature comparisons:

```
train/dino_viz/epoch_005/
├── sample_001_teacher.png    # Teacher feature map
├── sample_001_student.png    # Student feature map
├── sample_001_mask.png       # OBB mask
└── sample_001_diff.png       # Difference visualization
```

### Eval Overlays

```
eval/epoch_010/viz/val/
├── sample_001_gt.jpg       # Ground truth only (green)
├── sample_001_pred.jpg     # Predictions only (red)
├── sample_001_panel.jpg    # Composite overlay
└── index.json              # Metadata
```

## API Reference

### `DinoOBBTrainer`

Main trainer class.

```python
from detector_train.dino_trainer import DinoOBBTrainer

trainer = DinoOBBTrainer(
    config=train_config,
    device=device
)

trainer.train(
    data_yaml_path=data_yaml,
    epochs=128,
    callbacks=[eval_callback, wandb_callback]
)
```

### `TrainConfig`

Training configuration dataclass.

```python
@dataclass
class TrainConfig:
    # Standard YOLO params
    epochs: int = 128
    imgsz: int = 512
    batch: int = 16
    
    # Distillation params
    dino_root: Path = Path("dinov3")
    dino_distill_layers: tuple[int, ...] = (19,)
    dino_distill_channels: int = 32
    dino_distill_object_weight: float = 1.15
    dino_distill_background_weight: float = 0.15
    
    # Stage params
    stage_a_ratio: float = 0.30
    stage_a_freeze: int = 10
    stage_a_distill_weight: float = 0.25
    stage_b_distill_weight: float = 0.08
    
    def validate(self) -> None: ...
```

## Usage

### CLI

```bash
# Standard training
uv run detector-train --config config.json
```

### Python

```python
from detector_train.config import TrainConfig
from detector_train.dino_trainer import DinoOBBTrainer

config = TrainConfig(
    dataset_root=Path("./dataset"),
    artifacts_root=Path("./artifacts/models"),
    model="hf-openvision-yolo26-n-obb",
    name="yolo26n_obb",
    seed=42,
    epochs=128
)

trainer = DinoOBBTrainer(config, device="cuda:0")
trainer.train()
```

## Output Structure

```
artifacts/{model_key}/runs/{run_id}/train/
├── weights/
│   ├── best.pt              # YOLO best (mAP-based)
│   ├── best_geo.pt          # Best by geometric grade
│   └── last.pt              # Final checkpoint
├── results.csv              # Training metrics
├── args.yaml                # Training arguments
├── dino_viz/                # Feature visualizations
│   └── epoch_{nnn}/
└── ultralytics/             # Framework cache
```

## Monitoring

### W&B Logging

Automatic tracking of:
- Training losses (box, cls, dfl, distill)
- Learning rate
- Validation mAP
- Periodic eval grades
- System metrics (if enabled)

### Metrics Extraction

Parses YOLO's `results.csv`:

```python
metrics = {
    "train/loss_box": row["train/box_loss"],
    "train/loss_cls": row["train/cls_loss"],
    "train/loss_dfl": row["train/dfl_loss"],
    "val/precision": row["metrics/precision"],
    "val/recall": row["metrics/recall"],
    "val/map50": row["metrics/mAP50"],
    "val/map50_95": row["metrics/mAP50-95"]
}
```

## Performance

| Component | VRAM | Time/Epoch | Notes |
|-----------|------|------------|-------|
| YOLO only | ~6GB | ~2 min | Baseline |
| + DINO distill | ~10GB | ~3 min | + feature extraction |
| + Periodic eval | ~10GB | ~5 min | + inference + grading |

**Bottlenecks:**
1. DINO feature extraction (forward pass)
2. Feature map interpolation
3. Periodic evaluation

## Integration

Called by `just train`:

```bash
# Justfile
train: check-dataset
    uv run detector-train --config config.json
```

Consumes:
- `dataset/augmented/`: Training data
- `dinov3/`: Teacher model

Produces:
- Checkpoints for `detector-infer`
