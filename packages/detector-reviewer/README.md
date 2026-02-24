# detector-reviewer

Visual inspection tool for comparing model predictions against ground truth.

## Purpose

Interactive GUI for qualitative evaluation of detector performance. Overlays ground truth (green) and predictions (red) to quickly identify failure modes.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           REVIEW INTERFACE                                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│  │ FILTER PANEL                        SAMPLE NAVIGATOR                             │   │
│  │ ───────────                         ──────────────                               │   │
│  │                                     ┌─────────────────────────────────────────┐  │   │
│  │  Split: [val ▼]                     │                                         │  │   │
│  │                                     │         IMAGE CANVAS                     │  │   │
│  │  Confidence: [0.25 ▼]               │                                         │  │   │
│  │                                     │    ┌─────────────────────────────┐      │  │   │
│  │  [ ] Show GT only                   │    │                             │      │  │   │
│  │  [ ] Show Pred only                 │    │   ┌─────────────────────┐   │      │  │   │
│  │  [✓] Show Overlay                   │    │   │                     │   │      │  │   │
│  │                                     │    │   │   GREEN = GT        │   │      │  │   │
│  │  ─────────────────────────          │    │   │   RED   = Pred      │   │      │  │   │
│  │                                     │    │   │                     │   │      │  │   │
│  │  Sample: 42/500                     │    │   └─────────────────────┘   │      │  │   │
│  │                                     │    │                             │      │  │   │
│  │  [Prev] [Next]                      │    └─────────────────────────────┘      │  │   │
│  │  [Jump to #]                        │                                         │  │   │
│  │                                     │                                         │  │   │
│  │                                     └─────────────────────────────────────────┘  │   │
│  │                                                                                  │   │
│  │  METADATA PANEL                                                                  │   │
│  │  ─────────────                                                                   │   │
│  │  Image: val_042.jpg                                                              │   │
│  │  GT boxes: 3                                                                     │   │
│  │  Pred boxes: 2                                                                   │   │
│  │  Confidences: [0.92, 0.78]                                                       │   │
│  │                                                                                  │   │
│  └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Features

### Data Loading

Automatically resolves the latest run:

```
artifacts/latest_run.json ──▶ Find run_root
     │
     ├── dataset_root ──▶ Load images
     ├── infer/labels/{split}/ ──▶ Load predictions
     └── dataset/augmented/labels/{split}/ ──▶ Load ground truth
```

### Visualization Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Overlay** | GT (green) + Pred (red) together | Compare accuracy |
| **GT Only** | Ground truth only | Verify labels |
| **Pred Only** | Predictions only | Check confidence distribution |

### Navigation

| Key | Action |
|-----|--------|
| `Left` / `A` | Previous sample |
| `Right` / `D` | Next sample |
| `Up/Down` | Adjust confidence threshold |
| `F` | Toggle full screen |
| `G` | Toggle GT visibility |
| `P` | Toggle Pred visibility |
| `Q` / `Esc` | Quit |

## Data Flow

```
config.json ──▶ load_pipeline_config()
                     │
                     ▼
artifacts/latest_run.json ──▶ resolve run_root
                     │
                     ├──▶ infer/labels/{split}/ ──▶ load predictions
                     │
                     ├──▶ dataset/images/{split}/ ──▶ load images
                     │
                     └──▶ dataset/labels/{split}/ ──▶ load ground truth
                                          │
                                          ▼
                                   GUI overlay
```

## Configuration

```json
{
  "review": {
    "split": "val",
    "conf_threshold": 0.25
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `split` | string | "val" | Which split to review |
| `conf_threshold` | float | 0.25 | Minimum confidence to display |

## API Reference

### `ReviewerApp`

Main application class.

```python
from detector_reviewer.app import ReviewerApp
from PySide6.QtWidgets import QApplication

app = QApplication([])
window = ReviewerApp(
    dataset_root=Path("./dataset"),
    predictions_root=Path("./artifacts/infer/labels"),
    split="val",
    conf_threshold=0.25
)
window.show()
app.exec()
```

### `load_predictions(predictions_root, split)`

Load prediction files.

```python
from detector_reviewer.data import load_predictions

preds = load_predictions(
    predictions_root=Path("./infer/labels"),
    split="val"
)
# Returns: dict[stem, list[PredictionBox]]
```

## Usage

### CLI

```bash
# Review latest run predictions
uv run detector-reviewer --config config.json

# Override split
uv run detector-reviewer --config config.json --split train
```

### Python

```python
from detector_reviewer.app import main
from pipeline_config import load_pipeline_config

config = load_pipeline_config("config.json")
main(config)
```

## Use Cases

### 1. Failure Analysis

Find systematic errors:
- **Orientation confusion**: Model predicts wrong angle
- **Scale errors**: Consistently too large/small
- **Class confusion**: Mixing similar classes
- **Edge cases**: Extreme lighting, occlusion

### 2. Threshold Tuning

Visualize precision/recall tradeoffs:

```
Confidence 0.9: Fewer detections, higher precision
Confidence 0.5: More detections, more false positives
```

### 3. Dataset Validation

Catch mislabeled ground truth:
- GT box in wrong location
- Wrong class assignment
- Missing annotations

### 4. Model Comparison

Compare predictions across checkpoints:
```bash
# Review run A
uv run detector-reviewer --config config.json --run-id run-a

# Review run B
uv run detector-reviewer --config config.json --run-id run-b
```

## Implementation

- **Backend**: PySide6 (Qt6 bindings)
- **Canvas**: Custom `ImageCanvas` widget
- **Rendering**: OpenCV for overlays, PySide6 for display
- **Navigation**: Keyboard shortcuts with focus management

## Integration

Called by `just review`:

```bash
# Justfile
review:
    uv run detector-reviewer --config config.json
```

Input sources:
- `detector-infer`: Predictions
- `dataset-generator`: Ground truth images and labels
