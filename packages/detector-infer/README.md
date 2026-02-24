# detector-infer

High-throughput batch inference for YOLO OBB models.

## Purpose

Runs trained models across train/val splits to generate prediction files for downstream evaluation and review.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            INFERENCE PIPELINE                                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   INPUT                                                                                  │
│   ├── Weights: artifacts/{model}/runs/{run}/train/weights/best.pt                      │
│   └── Dataset: dataset/augmented/images/{split}/                                        │
│          │                                                                               │
│          ▼                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         INFERENCE ENGINE                                         │   │
│   │                                                                                  │   │
│   │  1. MODEL LOADER                                                                 │   │
│   │     └── YOLO(weights) → Validate OBB task type                                 │   │
│   │                                                                                  │   │
│   │  2. BATCH PROCESSOR                                                              │   │
│   │     └── model.predict(source=images, batch_size=N, device=GPU)                 │   │
│   │                                                                                  │   │
│   │  3. COORDINATE NORMALIZER                                                        │   │
│   │     └── Pixel coords → Normalized [0, 1] (YOLO OBB format)                     │   │
│   │                                                                                  │   │
│   │  4. OUTPUT WRITER                                                                │   │
│   │     └── YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 conf                 │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│   OUTPUT                                                                                 │
│   artifacts/{model}/runs/{run}/infer/labels/{split}/                                    │
│   ├── {stem}.txt        # Predictions (YOLO OBB format)                                │
│   └── ...                                                                               │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Architecture

### Model Loading

Loads YOLO OBB checkpoint with strict validation:

```python
from ultralytics import YOLO

model = YOLO(weights_path)

# Validate OBB task type
task = model.task.lower()
if task != "obb":
    raise RuntimeError(
        f"Model task is '{task}', expected 'obb'. "
        "Use an OBB-compatible checkpoint."
    )
```

### Batch Processing

Processes images with hardware acceleration:

```python
results = model.predict(
    source=image_paths,
    imgsz=config.imgsz,
    device=device,
    conf=config.conf_threshold,
    iou=config.iou_threshold,
    batch=config.batch_size,
    verbose=False
)
```

### Coordinate Normalization

Maps predictions back to normalized [0, 1] space:

```python
# YOLO outputs boxes in xyxyxyxy format (pixels)
for result in results:
    for box in result.boxes:
        corners_px = box.xyxyxyxy.cpu().numpy()  # [1, 8]
        
        # Reshape to [4, 2]
        corners = corners_px.reshape(4, 2)
        
        # Normalize by image dimensions
        corners_norm = corners / [image_w, image_h]
        
        # Ensure [0, 1] bounds
        corners_norm = np.clip(corners_norm, 0.0, 1.0)
```

### Output Format

YOLO OBB text format:

```
{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4} {confidence}
```

Example:
```
0 0.45 0.32 0.67 0.28 0.71 0.45 0.49 0.49 0.87
1 0.12 0.56 0.34 0.51 0.38 0.68 0.16 0.73 0.92
```

All coordinates normalized [0, 1], clockwise order.

## Configuration

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `imgsz` | int | 640 | Input image size (multiple of 32) |
| `device` | string | "auto" | "auto", "cpu", "mps", or GPU ID |
| `conf_threshold` | float | 0.25 | Minimum confidence to keep detection |
| `iou_threshold` | float | 0.7 | NMS IoU threshold |
| `splits` | string[] | ["val"] | Dataset splits to run inference on |
| `save_empty` | bool | true | Create empty files for images with no detections |
| `batch_size` | int | 16 | Inference batch size |

## Empty Sample Handling

When `save_empty: true`, creates empty label files for images with no detections:

```python
if len(boxes) == 0:
    if config.save_empty:
        label_path.write_text("")  # Empty file
    else:
        skip_write  # No file created
```

Empty files distinguish "no detections" from "not processed."

## Output Structure

```
artifacts/{model_key}/runs/{run_id}/infer/
└── labels/
    ├── train/
    │   ├── sample_001.txt    # Predictions for each image
    │   ├── sample_002.txt
    │   └── ...
    └── val/
        ├── sample_001.txt
        ├── sample_002.txt
        └── ...
```

## API Reference

### `InferConfig`

Inference configuration.

```python
@dataclass
class InferConfig:
    weights: Path
    dataset_root: Path
    output_root: Path
    
    imgsz: int = 640
    device: str = "auto"
    conf_threshold: float = 0.25
    iou_threshold: float = 0.7
    splits: tuple[str, ...] = ("val",)
    save_empty: bool = True
    batch_size: int = 16
```

### `run_inference(config) -> InferenceSummary`

Run batch inference.

```python
from detector_infer.infer import run_inference
from detector_infer.config import InferConfig

config = InferConfig(
    weights=Path("./best.pt"),
    dataset_root=Path("./dataset"),
    output_root=Path("./output")
)

summary = run_inference(config)
print(f"Processed {summary.total_images} images")
print(f"Detected {summary.total_detections} objects")
```

## Usage

### CLI

```bash
# Run inference on latest model
uv run detector-infer --config config.json

# Results in:
# artifacts/{model_key}/runs/{run_id}/infer/labels/
# ├── train/
# │   ├── sample_001.txt
# │   └── sample_002.txt
# └── val/
#     ├── sample_003.txt
#     └── sample_004.txt
```

### Python

```python
from detector_infer.infer import run_inference
from detector_infer.config import InferConfig
from pipeline_config import load_pipeline_config

config = load_pipeline_config("config.json")

infer_config = InferConfig(
    weights=config.paths["artifacts_root"] / "best.pt",
    dataset_root=config.paths["dataset_root"],
    imgsz=config.infer["imgsz"],
    device=config.infer["device"]
)

run_inference(infer_config)
```

## Performance

| Batch Size | Throughput | VRAM | Notes |
|------------|-----------|------|-------|
| 1 | ~20 img/s | ~4GB | Minimal latency |
| 8 | ~60 img/s | ~8GB | Balanced |
| 16 | ~80 img/s | ~12GB | Maximum throughput |

**Tuning:**
- Increase `batch_size` until VRAM limit
- Use `device: "auto"` for GPU detection
- Lower `imgsz` for faster inference (less accurate)

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `FileNotFoundError` | Weights file missing | Check `artifacts_root` structure |
| `RuntimeError: task=...` | Loaded HBB model | Use OBB checkpoint |
| `CUDA out of memory` | Batch too large | Reduce `batch_size` |
| `No images found` | Wrong dataset path | Check `dataset_root` config |

## Integration

Called by `just eval`:

```bash
# Justfile
eval:
    uv run detector-infer --config config.json
    uv run detector-grader --config config.json
```

Also called by `detector-grader` when predictions are missing:

```python
# In grader pipeline
if not predictions_exist:
    infer_config = InferConfig(
        weights=weights_path,
        dataset_root=dataset_root,
        output_root=predictions_root,
        ...
    )
    inference_summary = run_inference(infer_config)
```

Output consumed by:
- `detector-grader`: Evaluation input
- `detector-reviewer`: Visualization input
