# augment-checker

QA and integrity verification suite for augmented datasets. Catches data corruption, geometric anomalies, and label mismatches before training.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              VALIDATION PIPELINE                                         │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   INPUT: Generated Dataset (dataset/augmented/)                                         │
│   ├── images/train/, images/val/                                                        │
│   ├── labels/train/, labels/val/                                                        │
│   └── meta/train/, meta/val/                                                            │
│          │                                                                               │
│          ▼                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         DATA INDEXER                                             │   │
│   │   • Enumerate all samples                                                        │   │
│   │   • Build manifest: image ↔ label ↔ metadata mappings                          │   │
│   │   • Verify file existence                                                        │   │
│   └─────────────────────────┬───────────────────────────────────────────────────────┘   │
│                             │                                                            │
│                             ▼                                                            │
│   ┌─────────────────────────┴─────────────────────────┐                                 │
│   │                    PARALLEL CHECKS                 │                                 │
│   │                                                    │                                 │
│   │  ┌─────────────────┐    ┌─────────────────┐       │                                 │
│   │  │ INTEGRITY CHECKS│    │ GEOMETRY CHECKS │       │                                 │
│   │  │ ─────────────── │    │ ─────────────── │       │                                 │
│   │  │                 │    │                 │       │                                 │
│   │  │ • File sync     │    │ • H-matrix      │       │                                 │
│   │  │ • YOLO format   │    │   integrity     │       │                                 │
│   │  │ • Class mapping │    │ • Corner        │       │                                 │
│   │  │ • Missing files │    │   outliers      │       │                                 │
│   │  │                 │    │ • Projection    │       │                                 │
│   │  └────────┬────────┘    └────────┬────────┘       │                                 │
│   │           │                      │                │                                 │
│   └───────────┼──────────────────────┼────────────────┘                                 │
│               │                      │                                                  │
│               ▼                      ▼                                                  │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                      REPORT GENERATOR                                            │   │
│   │   • JSON summaries                                                               │   │
│   │   • Debug overlays (if issues found)                                             │   │
│   │   • Quality thresholds                                                           │   │
│   └─────────────────────────┬───────────────────────────────────────────────────────┘   │
│                             │                                                            │
│                    ┌────────┴────────┐                                                   │
│                    ▼                 ▼                                                   │
│              Pass (exit 0)    Fail (exit 2)                                              │
│              ┌───────────┐    ┌───────────┐                                              │
│              │ Continue  │    │ Block     │                                              │
│              │ training  │    │ training  │                                              │
│              └───────────┘    └───────────┘                                              │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Integrity Checks

### Filesystem Synchronization

Verifies 1:1 mapping between directories:

```python
# Every image has label and metadata
images/  ──┐
labels/  ──┼── All three must exist for valid sample
meta/    ──┘

# No orphans allowed
if image_exists and not label_exists:
    issue = "missing_label"
```

### YOLO Format Validation

Parses label files for correctness:

```python
# Expected: class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 values)
parts = line.strip().split()
if len(parts) != 9:
    issue = "invalid_format"

# Coordinates must be normalized [0, 1]
for coord in parts[1:]:
    if not (0.0 <= float(coord) <= 1.0):
        issue = "out_of_bounds"
```

### Class Registry Validation

Ensures all class IDs exist in `classes.txt`:

```python
max_class_id = len(classes) - 1
for label in labels:
    if label.class_id > max_class_id:
        issue = "unknown_class"
```

## Geometry Checks

### H-Matrix Validation

Reconstructs projected corners from metadata and validates:

```python
# Load H matrix from metadata
H = np.array(metadata["H"])

# Check for singularity
if abs(np.linalg.det(H)) < 1e-6:
    issue = "singular_homography"

# Reproject and compare
corners_reconstructed = apply_homography(H, canonical_corners)
corners_stored = metadata["projected_corners_px"]
error = np.linalg.norm(corners_reconstructed - corners_stored)
```

### Corner Outlier Detection

Flags samples with excessive geometric error:

```python
# Compute corner error distribution
errors = [sample.corner_error for sample in samples]
mean_error = np.mean(errors)
std_error = np.std(errors)

# Flag outliers (3 sigma)
threshold = mean_error + 3 * std_error
for sample in samples:
    if sample.corner_error > threshold:
        issue = "geometry_outlier"
```

### Projection Sanity

Revalidates geometric constraints:

```python
# Check convexity
if not is_convex_quad(corners):
    issue = "non_convex"

# Check bounds
if not quad_inside_bounds(corners, w, h):
    issue = "out_of_bounds"

# Check minimum area
if polygon_area(corners) < min_area:
    issue = "too_small"
```

## Prediction Checks (Optional)

If inference has been run, validates predictions:

```python
# Load predictions for this sample
preds = load_predictions(infer_root, sample.stem)

# Check format consistency
for pred in preds:
    if len(pred.corners) != 4:
        issue = "malformed_prediction"
```

## Report Generation

### Console Summary

```
checked 1,247 samples
integrity issues: 3
  - missing_label: 2
  - invalid_format: 1
geometry outliers: 5
  - mean_corner_error_px: 0.8
  - max_corner_error_px: 12.3
```

### JSON Reports

Written to `dataset/reports/`:

```json
{
  "total_samples": 1247,
  "integrity": {
    "total_issues": 3,
    "by_type": {
      "missing_label": 2,
      "invalid_format": 1
    }
  },
  "geometry": {
    "num_outliers": 5,
    "outlier_rate": 0.004,
    "mean_corner_error_px": 0.8,
    "max_corner_error_px": 12.3
  }
}
```

### Debug Overlays

Generates visualization images for failed samples:

```
reports/overlays/
├── train/
│   ├── sample_001_overlay.jpg   # Green: GT, Red: Reconstructed
│   └── sample_042_overlay.jpg
└── val/
    └── sample_003_overlay.jpg
```

Overlays show:
- **Green**: Ground truth corners
- **Red**: Reconstructed corners (from H matrix)
- **Yellow**: Outliers flagged

## Configuration

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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `outlier_threshold_px` | float | 2.0 | Threshold for geometry outlier detection |
| `debug_overlays_per_split` | int | 10 | Max overlay images to generate per split |
| `gui` | bool | false | Launch GUI browser after checks |
| `seed` | int | 42 | RNG seed for deterministic sampling |

## Quality Thresholds

Hardcoded gates that must pass:

```python
MAX_OUTLIER_RATE = 0.02          # 2% outliers max
MAX_MEAN_CORNER_ERROR_PX = 1.5   # Mean error must be < 1.5px
```

Exceeding thresholds causes exit code 2, preventing training on bad data.

## API Reference

### `check_dataset(dataset_root, checks_config) -> CheckResult`

Run all validation checks.

**Parameters:**
- `dataset_root`: Path to augmented dataset
- `checks_config`: Configuration dict from pipeline config

**Returns:** `CheckResult` with issue counts and details

### `CheckResult`

```python
@dataclass
class CheckResult:
    total_samples: int
    integrity_issues: dict[str, int]  # type -> count
    geometry_outliers: list[SampleIssue]
    passed: bool
```

## Usage

### CLI

```bash
# Run checks (headless)
uv run augment-checker --config config.json

# Run with GUI browser
uv run augment-checker --config config.json --gui
```

### Python

```python
from augment_checker.integrity import check_dataset
from pipeline_config import load_pipeline_config

config = load_pipeline_config("config.json")
result = check_dataset(
    dataset_root=config.paths["dataset_root"],
    checks_config=config.checks
)

if not result.passed:
    print("Checks failed - blocking training")
    exit(2)
```

## GUI Browser

Interactive PySide6 application for deep inspection:

### Features

- **Sample grid**: Thumbnail view of all samples
- **Filter panel**: Show only samples with issues
- **Detail view**: Full image with overlay
- **Metadata panel**: Raw JSON metadata
- **Navigation**: Arrow keys, click to select

### Controls

| Key | Action |
|-----|--------|
| `←/→` | Previous/Next sample |
| `F` | Toggle filter (all/issues only) |
| `O` | Open overlay in external viewer |
| `Q` | Quit |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All checks passed |
| 1 | Command error |
| 2 | Checks failed (integrity issues or outliers exceed threshold) |

## Integration

Called by `just check-dataset`:

```bash
# Justfile
check-dataset:
    uv run augment-checker --config config.json
```

Blocks training if checks fail:

```bash
# Justfile - training depends on checks
train: check-dataset
    uv run detector-train --config config.json
```
