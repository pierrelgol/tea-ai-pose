# target-labeller

Qt-based GUI tool for annotating target objects with Oriented Bounding Boxes (OBB).

## Purpose

Interactive annotation interface for creating the canonical target dataset used by the generator. Each target image receives exactly one OBB annotation, exported in YOLO OBB format.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ANNOTATION WORKFLOW                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT                                PROCESS                        OUTPUT     │
│  ─────                                ───────                        ──────     │
│                                                                                  │
│  targets_source_root/                 ┌─────────────────────────┐                │
│  ├── object1.png                      │    target-labeller      │                │
│  ├── object2.jpg     ────────────────▶│        (GUI)            │───────────────▶│
│  └── object3.png                      │  ┌───────────────────┐  │                │
│                                       │  │  Image Canvas     │  │                │
│                                       │  │ • Click-drag OBB  │  │                │
│                                       │  │ • Real-time box   │  │                │
│                                       │  └───────────────────┘  │                │
│                                       │  ┌───────────────────┐  │                │
│                                       │  │  Control Panel    │  │                │
│                                       │  │ • Class picker    │  │                │
│                                       │  │ • Nav controls    │  │                │
│                                       │  └───────────────────┘  │                │
│                                       └─────────────────────────┘                │
│                                                                                  │
│                                                                             targets/
│                                                                             ├── images/
│                                                                             │   ├── object1.png
│                                                                             │   └── ...
│                                                                             ├── labels/
│                                                                             │   ├── object1.txt
│                                                                             │   └── ...
│                                                                             └── classes.txt
│
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Workflow

### 1. Input Discovery

- Loads images from `targets_source_root` (configurable via `config.json`)
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`
- Maintains existing annotations if present (resumable)

### 2. Annotation

- **Draw**: Click and drag on image to define oriented bounding box
  - Start drag at one corner
  - Drag to opposite corner
  - Box rotates to match drag angle
- **Adjust**: Re-drag to replace existing box
- **Clear**: Press `Delete` to remove box

### 3. Class Assignment

- Type class name in input field
- Or select from dropdown of existing classes
- Auto-populates `classes.txt` with new classes

### 4. Export

On `Finish`:
- Copies images to `targets/images/`
- Writes YOLO OBB labels to `targets/labels/`
- Appends to `targets/classes.txt`

## Interface

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  target-labeller                              [Class: __________] [v]      │
│                                                                             │
│  ┌─────────────────────────────────────────┐   ┌───────────────────────┐   │
│  │                                         │   │  Controls             │   │
│  │                                         │   │  ─────────            │   │
│  │         Image Canvas                    │   │                       │   │
│  │                                         │   │  [←] [→] Navigate     │   │
│  │    ┌─────────────────────┐              │   │  [Ctrl+S] Save        │   │
│  │    │                     │              │   │  [Del] Clear          │   │
│  │    │   OBB Overlay       │              │   │  [Finish] Export All  │   │
│  │    │   (rotated box)     │              │   │                       │   │
│  │    │                     │              │   │  Progress: 3/12       │   │
│  │    └─────────────────────┘              │   │                       │   │
│  │                                         │   └───────────────────────┘   │
│  │         Click + Drag to draw            │                               │
│  └─────────────────────────────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Controls

| Key/Action | Function |
|------------|----------|
| `Click + Drag` | Draw/replace bounding box |
| `Ctrl+S` | Save current annotation |
| `Left` / `A` | Previous image |
| `Right` / `D` | Next image |
| `Delete` | Clear current box |
| `Finish` | Export all and exit |

## Output Format

### YOLO OBB Labels

```
{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}
```

Example:
```
0 0.45 0.32 0.67 0.28 0.71 0.45 0.49 0.49
```

- Coordinates normalized [0, 1]
- Order: clockwise from top-left
- One annotation per target file

### Directory Structure

```
targets/
├── images/                    # Copied from source
│   ├── object1.png
│   ├── object2.jpg
│   └── object3.png
├── labels/                    # YOLO OBB format
│   ├── object1.txt
│   ├── object2.txt
│   └── object3.txt
└── classes.txt                # Class name mapping
    ├── bottle
    ├── can
    └── cup
```

## Configuration

```json
{
  "paths": {
    "targets_source_root": "targets"
  }
}
```

- `targets_source_root`: Directory containing raw target images to annotate

## API Reference

### `TargetLabellerApp`

Main application class.

```python
from target_labeller.app import TargetLabellerApp
from PySide6.QtWidgets import QApplication

app = QApplication([])
window = TargetLabellerApp(source_dir, output_dir)
window.show()
app.exec()
```

### `ImageCanvas`

Custom widget for OBB drawing.

- Handles mouse events for drag-to-draw
- Renders OBB overlay
- Supports real-time preview

## Usage

### CLI

```bash
# Launch GUI
uv run target-labeller --config config.json
```

Reads `config.paths.targets_source_root` for input directory.

### Python

```python
from target_labeller.app import main
from pipeline_config import load_pipeline_config

config = load_pipeline_config("config.json")
main(config)
```

## Annotation Guidelines

For best generator results:

1. **Tight fit**: Box should touch object edges
2. **Consistent orientation**: Use same corner order convention
3. **Single object**: One target object per image
4. **Clean background**: Transparent or uniform background preferred

## Implementation

- **Backend**: PySide6 (Qt6 Python bindings)
- **Canvas**: Custom `ImageCanvas` widget
- **Persistence**: Auto-saves annotations in memory; explicit export on Finish
- **Validation**: Requires class name and valid box before save

## Integration

Called by `just label-targets`:

```bash
# Justfile
label-targets:
    uv run target-labeller --config config.json
```

Output consumed by `dataset-generator`:

```python
# In dataset-generator
target_images_dir = dataset_root / "targets" / "images"
target_labels_dir = dataset_root / "targets" / "labels"
```
