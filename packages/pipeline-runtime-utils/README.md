# pipeline-runtime-utils

Cross-cutting runtime utilities for device resolution, deterministic seeding, and geometric coordinate transformations. Eliminates code duplication across all pipeline stages.

## Purpose

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SHARED UTILITIES                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌───────────────┐    ┌───────────────┐    ┌─────────────────────────────────┐  │
│  │    runtime    │    │    geometry   │    │              paths              │  │
│  │ ───────────── │    │ ───────────── │    │ ─────────────────────────────── │  │
│  │               │    │               │    │                                 │  │
│  │ • resolve_    │    │ • corners_    │    │ • resolve_latest_weights        │  │
│  │   device      │───▶│   norm_to_px  │    │ • artifact helpers              │  │
│  │               │    │               │    │                                 │  │
│  │ • set_seed    │    │ • corners_px  │    │ • checkpoint discovery          │  │
│  │               │───▶│   _to_yolo    │    │                                 │  │
│  │               │    │               │    │                                 │  │
│  │               │    │ • polygon_    │    │                                 │  │
│  │               │    │   operations  │    │                                 │  │
│  │               │    │               │    │                                 │  │
│  │               │    │ • principal_  │    │                                 │  │
│  │               │    │   angle_deg   │    │                                 │  │
│  └───────────────┘    └───────────────┘    └─────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### runtime.py

#### Device Resolution

Auto-detects best available compute device with fallback chain:

```python
from pipeline_runtime_utils import resolve_device

device = resolve_device("auto")
# Returns: "0" (CUDA) > "mps" (Apple Silicon) > "cpu"

# Explicit selection
resolve_device("0")      # CUDA GPU 0
resolve_device("cpu")    # Force CPU
resolve_device("mps")    # Apple Silicon
```

**Detection Logic:**
1. Check `torch.cuda.is_available()` → return `"0"`
2. Check `torch.backends.mps.is_available()` → return `"mps"`
3. Fallback → return `"cpu"`

#### Seed Management

Synchronizes RNGs across libraries for reproducibility:

```python
from pipeline_runtime_utils import set_seed

set_seed(42)
# Sets: random.seed(42)
#       np.random.seed(42)
#       torch.manual_seed(42)
#       torch.cuda.manual_seed_all(42)  # if CUDA
```

**Used for deterministic:**
- Dataset sampling and splitting
- Augmentation sequences
- Model initialization
- Dropout patterns

### geometry.py

YOLO OBB uses normalized coordinates [0, 1]. These functions convert between coordinate spaces:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     COORDINATE TRANSFORMATIONS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Normalized [0,1]              Pixels              YOLO OBB Format         │
│   ────────────────              ──────              ───────────────         │
│                                                                              │
│   (0,0)──────────(1,0)         (0,0)──────────(W,0)                          │
│      │              │             │              │                           │
│      │  (0.5,0.5)   │             │   (cx,cy)    │                           │
│      │              │             │              │                           │
│   (0,1)──────────(1,1)         (0,H)──────────(W,H)                          │
│                                                                              │
│   corners_norm_to_px()      corners_px_to_yolo_obb()                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Coordinate Transformations

```python
from pipeline_runtime_utils import corners_norm_to_px, corners_px_to_yolo_obb
import numpy as np

# Normalized → Pixel (for drawing)
# Input: [4, 2] float32, normalized [0, 1]
corners_norm = np.array([
    [0.4, 0.3],  # top-left
    [0.6, 0.3],  # top-right
    [0.6, 0.5],  # bottom-right
    [0.4, 0.5],  # bottom-left
], dtype=np.float32)

px_corners = corners_norm_to_px(corners_norm, image_w=1024, image_h=768)
# Output: [[409.6, 230.4], [614.4, 230.4], [614.4, 384.0], [409.6, 384.0]]

# Pixel → Normalized (for export)
norm_corners = corners_px_to_yolo_obb(px_corners, image_w=1024, image_h=768)
# Output: [[0.4, 0.3], [0.6, 0.3], [0.6, 0.5], [0.4, 0.5]]
```

#### Polygon Operations

```python
from pipeline_runtime_utils import (
    polygon_area,
    is_convex_quad,
    quad_inside_bounds,
    principal_angle_deg,
    angle_bin,
    polygon_iou
)

# Area calculation (shoelace formula)
area = polygon_area(corners_px)  # float

# Convexity check
is_convex = is_convex_quad(corners_px)  # bool

# Bounds validation
inside = quad_inside_bounds(corners_px, image_w=1024, image_h=768)  # bool

# Principal angle (dominant edge orientation, 0-180°)
angle = principal_angle_deg(corners_px)  # float degrees

# Angle binning (for balanced sampling)
bin_idx = angle_bin(angle_deg=45.0, n_bins=12)  # int 0-11

# Polygon IoU (using cv2.intersectConvexConvex)
iou = polygon_iou(poly_a, poly_b)  # float [0, 1]
```

### paths.py

#### Artifact Resolution

```python
from pipeline_runtime_utils import resolve_latest_weights_from_artifacts
from pathlib import Path

# Find latest training weights
weights_path = resolve_latest_weights_from_artifacts(
    artifacts_root=Path("./artifacts/models")
)
# Returns: artifacts/{model_key}/runs/{latest_run}/train/weights/best.pt
#          or None if not found
```

## API Reference

### `resolve_device(requested: str) -> str`

Resolve compute device string to valid torch device.

| Requested | Returns | Condition |
|-----------|---------|-----------|
| `"auto"` | `"0"` | CUDA available |
| `"auto"` | `"mps"` | MPS available (macOS) |
| `"auto"` | `"cpu"` | Fallback |
| `"0"`, `"1"`, etc. | `"0"`, `"1"` | As requested |
| `"cpu"` | `"cpu"` | As requested |
| `"mps"` | `"mps"` | As requested |

### `set_seed(seed: int) -> None`

Set random seeds across all libraries.

### `corners_norm_to_px(corners_norm, image_w, image_h) -> np.ndarray`

Convert normalized corners to pixel coordinates.

**Parameters:**
- `corners_norm`: `[4, 2]` float32 array, values in [0, 1]
- `image_w`, `image_h`: Image dimensions in pixels

**Returns:** `[4, 2]` float32 array, pixel coordinates

### `corners_px_to_yolo_obb(corners_px, image_w, image_h) -> np.ndarray`

Convert pixel corners to YOLO OBB normalized format.

**Parameters:**
- `corners_px`: `[4, 2]` float32 array, pixel coordinates
- `image_w`, `image_h`: Image dimensions in pixels

**Returns:** `[4, 2]` float32 array, normalized [0, 1], clipped

### `polygon_area(corners_px) -> float`

Calculate polygon area using shoelace formula.

### `is_convex_quad(corners_px) -> bool`

Check if quadrilateral is convex.

### `quad_inside_bounds(corners_px, image_w, image_h) -> bool`

Check if all corners are inside image bounds.

### `principal_angle_deg(corners_px) -> float`

Calculate principal orientation angle (0-180°).

Uses the dominant edge direction of the quadrilateral.

### `angle_bin(angle_deg, n_bins) -> int`

Bin angle into discrete sectors.

```python
angle_bin(0, 12)    # 0
angle_bin(90, 12)   # 6
angle_bin(180, 12)  # 0 (wraps around)
```

### `polygon_iou(poly_a, poly_b) -> float`

Calculate IoU between two convex polygons.

Uses `cv2.intersectConvexConvex` for accurate intersection area.

### `resolve_latest_weights_from_artifacts(artifacts_root) -> Path | None`

Resolve the most recent trained weights file.

Searches `artifacts_root/*/runs/*/train/weights/best.pt` and returns the newest by mtime.

## Integration Points

| Package | Functions Used |
|---------|----------------|
| `detector-train` | `resolve_device`, `set_seed` |
| `detector-infer` | `resolve_device` |
| `detector-grader` | `corners_norm_to_px`, `polygon_iou` |
| `dataset-generator` | `corners_px_to_yolo_obb`, `principal_angle_deg`, `angle_bin` |
| `augment-checker` | `corners_norm_to_px` |

## Performance

| Function | Complexity | Notes |
|----------|-----------|-------|
| `resolve_device` | O(1) | Cached by PyTorch |
| `set_seed` | O(1) | Synchronous |
| `corners_norm_to_px` | O(1) | Vectorized NumPy |
| `polygon_area` | O(1) | 4-point quads only |
| `polygon_iou` | O(1) | cv2 convex intersection |
| `principal_angle_deg` | O(1) | Sorts 4 edges |

## Dependencies

- `numpy`: Array operations
- `torch` (optional): Device detection falls back gracefully
- `cv2` (optional): Polygon intersection used in grader
