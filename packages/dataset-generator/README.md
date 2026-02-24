# dataset-generator

Procedural synthesis of OBB training data via homographic projection and photometric augmentation.

## Purpose

Generates synthetic training datasets by projecting annotated target objects onto background images. Eliminates manual labeling while providing controlled variation in pose, scale, lighting, and occlusion.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              GENERATION PIPELINE                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   INPUTS                                                                                 │
│   ├── Canonical Targets (targets/images/, targets/labels/)                              │
│   ├── Backgrounds (dataset/coco128/images/)                                             │
│   └── Grade Reports (artifacts/*/runs/*/grade/reports/) ──┐                             │
│                                                           │                             │
│                                                           ▼                             │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │  1. CURRICULUM RESOLVER                                                          │   │
│   │     └── Parse grade reports → Adjust difficulty parameters                       │   │
│   │                                                                                  │   │
│   │  2. HOMOGRAPHY SAMPLER                                                           │   │
│   │     └── Sample 3×3 H matrix → Apply to canonical corners                         │   │
│   │                                                                                  │   │
│   │  3. MULTI-TARGET PLACEMENT                                                       │   │
│   │     └── Class-balanced + occlusion-aware + angle-balanced placement              │   │
│   │                                                                                  │   │
│   │  4. SYNTHESIS ENGINE                                                             │   │
│   │     └── Warp target with H → Alpha blend onto background                         │   │
│   │                                                                                  │   │
│   │  5. PHOTOMETRIC STACK                                                            │   │
│   │     └── HSV jitter → Blur → Noise injection                                      │   │
│   │                                                                                  │   │
│   │  6. EXPORT                                                                       │   │
│   │     └── Image + YOLO OBB labels + Metadata JSON                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│   OUTPUT: Synthetic OBB Dataset                                                          │
│   dataset/augmented/                                                                     │
│   ├── images/{split}/     # Augmented images                                             │
│   ├── labels/{split}/     # YOLO OBB format                                            │
│   └── meta/{split}/       # H matrices, provenance                                     │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Pipeline Stages

### 1. Curriculum Resolver

Adapts generation difficulty based on model performance from grade reports:

```python
if orientation_accuracy > 0.9:
    stage = "hard"        # More perspective, more occlusion
elif orientation_accuracy > 0.7:
    stage = "medium"
else:
    stage = "mild"        # Less variation, easier samples
```

### 2. Homography Sampler

Projects 2D targets into 3D perspective using homography matrices.

#### Homography Matrix H

3×3 matrix representing planar perspective transformation:

```
[x']   [h11 h12 h13] [x]
[y'] = [h21 h22 h23] [y]
[w']   [h31 h32 h33] [1]

x_proj = x' / w'
y_proj = y' / w'
```

#### Sampling Parameters

```python
@dataclass
class HomographyParams:
    scale_min: float = 0.1           # Target size range
    scale_max: float = 0.5
    translate_frac: float = 0.3      # Position jitter
    perspective_jitter: float = 0.4  # 3D perspective strength
    min_quad_area_frac: float = 0.01 # Minimum valid area
    max_attempts: int = 100          # Retry limit
```

#### Validation Checks

Every projected target must pass:

1. **Convexity**: Quad must remain convex
2. **Bounds**: All corners inside image frame
3. **Minimum Area**: `area_px > config.min_target_area_px`
4. **Edge Length**: All edges > `config.min_edge_length_px`
5. **Aspect Ratio**: `max_edge / min_edge < config.max_edge_aspect_ratio`
6. **Corner Angles**: All angles in valid range
7. **Rectangular Fit**: `IoU(raw_quad, min_area_rect) > 0.72`

### 3. Multi-Target Placement

#### Class Balancing

Frequency-inverse weighting with hard example boosting:

```python
# Inverse frequency weight
freq = class_counts / total
balance_weights = (1 / freq) ** class_balance_strength

# Hard example boosting (from grade reports)
hard_boost = load_hard_class_boosts(hard_examples_path)
class_weights = balance_weights * (1 + hard_example_boost * hard_boost)
```

#### Occlusion Handling

Tracks placed targets with occupancy mask:

```python
occupancy_mask = np.zeros((h, w), dtype=bool)

for target in placements:
    warped_mask = warp_mask(target, H)
    visible_ratio = visible_ratio(warped_mask, occupancy_mask)
    
    if visible_ratio > (1 - max_occlusion_ratio):
        occupancy_mask |= warped_mask  # Accept
    else:
        continue  # Reject (too occluded)
```

#### Angle Balancing

Tracks angle distribution for orientation diversity:

```python
angle_bin_counts = np.zeros(12, dtype=int)

for placement:
    angle = principal_angle_deg(projected_corners)
    bin_idx = angle_bin(angle, n_bins=12)
    
    # Prefer rare angles
    rarity = (max_count + 1) / (angle_bin_counts[bin_idx] + 1)
    score = 0.75 * rarity + 0.25 * size_penalty
```

### 4. Synthesis Engine

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Target     │────▶│  Warp with   │────▶│ Alpha Blend  │────▶ Output
│   Image      │     │  H matrix    │     │ onto BG      │     Image
│   + Alpha    │     │ (cv2.warp    │     │              │
│              │     │  Perspective)│     │              │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 5. Photometric Augmentation Stack

Applied post-synthesis to bridge sim-to-real gap:

#### HSV Jitter

```python
hue_shift = rng.uniform(-hsv_h, hsv_h)      # degrees
sat_scale = rng.uniform(1 - hsv_s, 1 + hsv_s)
val_scale = rng.uniform(1 - hsv_v, 1 + hsv_v)
```

#### Blur Suite

- **Gaussian**: `kernel_size` odd in [3, 7], `sigma` in [0.1, 2.0]
- **Motion**: Directional blur simulating camera movement

#### Noise Models

- **Gaussian**: Additive noise, `sigma` in [0, 25]
- **JPEG**: Compression artifacts at quality [70, 95]

## Output Format

### Images

```
{output_root}/images/{split}/{split}_{background_stem}_s{sample_idx:03d}.jpg
```

### Labels (YOLO OBB)

```
{class_id} {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}
```

Normalized coordinates [0, 1], clockwise order.

### Metadata JSON

```json
{
  "seed": 42,
  "background_image": "backgrounds/train/scene_001.jpg",
  "num_targets": 3,
  "curriculum": {
    "stage": "medium",
    "orientation_within_10deg_rate": 0.85
  },
  "targets": [
    {
      "target_image": "targets/images/bottle_001.png",
      "target_class_name": "bottle",
      "target_class_id_exported": 0,
      "H": [[...], [...], [...]],
      "projected_corners_px": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "projected_corners_yolo_obb": [[xn1, yn1], ...],
      "visible_ratio": 0.95,
      "occlusion_ratio": 0.05,
      "principal_angle_deg": 45.2
    }
  ]
}
```

## API Reference

### `GeneratorConfig`

Configuration for dataset generation.

```python
@dataclass
class GeneratorConfig:
    output_root: Path
    target_images_dir: Path
    target_labels_dir: Path
    target_classes_file: Path
    background_splits: dict[str, Path]
    
    # Generation parameters
    samples_per_background: int = 10
    targets_per_image_min: int = 1
    targets_per_image_max: int = 5
    
    # Geometry
    scale_min: float = 0.1
    scale_max: float = 0.5
    
    # Curriculum
    curriculum_enabled: bool = True
    grade_reports_dir: Path | None = None
    
    seed: int = 42
```

### `generate_dataset(config) -> list[SampleResult]`

Generate synthetic dataset.

**Returns:** List of `SampleResult` with paths to generated files.

## Usage

### Python API

```python
from pathlib import Path
from dataset_generator.config import GeneratorConfig
from dataset_generator.generator import generate_dataset

config = GeneratorConfig(
    output_root=Path("./dataset/augmented"),
    target_images_dir=Path("./dataset/targets/images"),
    target_labels_dir=Path("./dataset/targets/labels"),
    target_classes_file=Path("./dataset/targets/classes.txt"),
    background_splits={
        "train": Path("./datasets/coco128/images/train"),
        "val": Path("./datasets/coco128/images/val")
    },
    samples_per_background=10,
    targets_per_image_min=1,
    targets_per_image_max=5,
    curriculum_enabled=True,
    seed=42
)

results = generate_dataset(config)
```

### CLI

```bash
# Generate with config
uv run dataset-generator --config config.json
```

## Performance

| Parameter | Impact | Typical |
|-----------|--------|---------|
| `samples_per_background` | Linear time | 10-50 |
| `targets_per_image_max` | Quadratic occlusion checks | 3-10 |
| `max_attempts` | Placement retry budget | 100-500 |
| Image resolution | Memory + warp time | 1024×1024 |

**Bottlenecks:**
1. Homography validation (geometric checks)
2. Mask warping (`cv2.warpPerspective`)
3. Alpha blending (per-pixel compositing)

## Integration

Called by `just generate-dataset`:

```bash
# Justfile
generate-dataset:
    uv run dataset-generator --config config.json
```

Reads:
- `config.paths.dataset_root` for output location
- `config.paths.targets_source_root` for targets
- `config.dataset.name` for background dataset
- `config.run.seed` for reproducibility

Output consumed by:
- `augment-checker`: Validate generated data
- `detector-train`: Training input
