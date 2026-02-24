# detector-grader

Canonical evaluation and grading tool for detector runs. Provides high-precision geometric assessment of OBB detections.

## Purpose

Moves beyond simple mAP metrics to evaluate how well the model masters exact pose, scale, and shape of target objects. Produces diagnostic reports used for curriculum generation.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           GRADING PIPELINE                                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   INPUTS                                                                                 │
│   ├── Ground Truth: dataset/augmented/labels/{split}/                                   │
│   └── Predictions:  artifacts/{model}/runs/{run}/infer/labels/{split}/                  │
│          │                                                                               │
│          ▼                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                         GRADING ENGINE                                           │   │
│   │                                                                                  │   │
│   │  1. GREEDY MATCHER                                                               │   │
│   │     └── Quality-weighted bipartite matching (GT ↔ Pred)                        │   │
│   │                                                                                  │   │
│   │  2. GEOMETRIC SCORER                                                             │   │
│   │     └── Five-axis OBB alignment check                                          │   │
│   │                                                                                  │   │
│   │  3. PENALTY ENGINE                                                               │   │
│   │     └── FN, FP, and Containment analysis                                       │   │
│   │                                                                                  │   │
│   │  4. REPORT GENERATOR                                                             │   │
│   │     └── JSON/CSV artifacts + Hard example extraction                           │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                          │                                               │
│                                          ▼                                               │
│   OUTPUTS                                                                                │
│   ├── grade/reports/grade_report_{split}_{timestamp}.json                               │
│   ├── grade/reports/hard_examples.jsonl                                                 │
│   └── grade/reports/summary.txt                                                         │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Five-Axis Geometric Scoring

Each matched detection receives a score from 0.0 to 1.0 based on weighted metrics:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         QUALITY METRIC COMPONENTS                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   Quality = w_iou × IoU_score + w_corner × Corner_score +                              │
│             w_angle × Angle_score + w_center × Center_score +                            │
│             w_shape × Shape_score                                                        │
│                                                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │ IoU Score          │ Intersection over Union                                      │   │
│   │                    │ Gamma emphasis rewards high-precision overlaps              │   │
│   ├────────────────────┼──────────────────────────────────────────────────────────────┤   │
│   │ Corner Score       │ Mean pixel distance between reordered corners               │   │
│   │                    │ Exponentially decaying score: exp(-distance/σ)              │   │
│   ├────────────────────┼──────────────────────────────────────────────────────────────┤   │
│   │ Angle Score        │ Difference in principal axes                                 │   │
│   │                    │ Weighted by eccentricity (orientation reliability)          │   │
│   ├────────────────────┼──────────────────────────────────────────────────────────────┤   │
│   │ Center Score       │ Normalized Euclidean distance between centroids             │   │
│   │                    │ 1.0 at same position, 0.0 at opposite corners               │   │
│   ├────────────────────┼──────────────────────────────────────────────────────────────┤   │
│   │ Shape Score        │ Consistency of edge length ratios                          │   │
│   │                    │ Area ratio between GT and predicted boxes                   │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Penalty Logic

Final "Grade" for a sample is the mean matched score minus weighted penalties:

```python
# Base score from matches
base_score = mean(matched_qualities)

# Penalties
fn_penalty = fn_count * FN_WEIGHT        # False Negatives
fp_penalty = fp_count * FP_WEIGHT        # False Positives  
containment_penalty = containment_errors * CONTAINMENT_WEIGHT

# Final grade
grade = max(0, base_score - fn_penalty - fp_penalty - containment_penalty)
```

### Penalty Types

| Penalty | Description | Weight |
|---------|-------------|--------|
| **False Negative (FN)** | Missed targets (ground truth not detected) | High |
| **False Positive (FP)** | Hallucinations or redundant detections | Medium |
| **Containment** | Prediction inside GT but wrong size, or vice versa | Low |

## Diagnostic Metrics

The grader exports 30+ diagnostic fields:

| Metric | Description | Use Case |
|--------|-------------|----------|
| `angle_le_5_rate` | Detections with <5° orientation error | Orientation quality |
| `angle_le_15_rate` | Detections with <15° orientation error | Coarse orientation |
| `corner_mean_error_px` | Mean pixel error for corner matching | Localization precision |
| `gt_area_missed_ratio` | Missed area fraction | Coverage analysis |
| `hard_class_ids` | Classes failing geometric checks | Curriculum targeting |
| `containment_misses` | Count of containment errors | Scale calibration |

## Report Format

### Grade Report JSON

```json
{
  "timestamp": "2024-01-15T12:00:00Z",
  "split": "val",
  "model_key": "yolo26n_obb",
  "run_id": "run-001",
  "summary": {
    "total_samples": 128,
    "total_gt_boxes": 342,
    "total_pred_boxes": 338,
    "matched_pairs": 325,
    "false_negatives": 17,
    "false_positives": 13,
    "overall_grade": 0.847
  },
  "metrics": {
    "iou": {"mean": 0.823, "median": 0.851},
    "corner_error_px": {"mean": 2.4, "p95": 8.1},
    "angle_error_deg": {"mean": 4.2, "p95": 15.3},
    "angle_le_5_rate": 0.78,
    "angle_le_15_rate": 0.94
  },
  "per_class": {
    "0": {"grade": 0.891, "count": 156},
    "1": {"grade": 0.742, "count": 98}
  },
  "hard_examples": [
    {"sample_id": "val_042", "grade": 0.23, "issues": ["orientation", "scale"]}
  ]
}
```

### Hard Examples

JSONL file for curriculum learning:

```json
{"sample_id": "val_042", "class_id": 1, "angle_error": 45.2, "grade": 0.23}
{"sample_id": "val_087", "class_id": 0, "corner_error": 23.4, "grade": 0.31}
```

Consumed by `dataset-generator` to boost hard classes.

## Configuration

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
    "run_inference": true
  }
}
```

| Parameter | Description |
|-----------|-------------|
| `splits` | Which splits to grade |
| `match_iou_threshold` | Minimum IoU for GT-Pred matching |
| `strict_obb` | Enforce OBB format validation |
| `calibrate_confidence` | Apply confidence calibration |
| `run_inference` | Auto-run inference if predictions missing |

## API Reference

### `grade_predictions(grade_config) -> GradeReport`

Run grading on predictions.

```python
from detector_grader.pipeline import grade_predictions
from detector_grader.data import GradeConfig

config = GradeConfig(
    predictions_root=Path("./infer/labels"),
    ground_truth_root=Path("./dataset/labels"),
    output_root=Path("./grade"),
    match_iou_threshold=0.5
)

report = grade_predictions(config)
print(f"Overall grade: {report.overall_grade}")
print(f"Angle <5° rate: {report.metrics.angle_le_5_rate}")
```

### `compute_quality(gt_box, pred_box) -> QualityScores`

Compute five-axis quality scores for a matched pair.

```python
from detector_grader.scoring import compute_quality

quality = compute_quality(gt_box, pred_box)
print(f"IoU: {quality.iou}")
print(f"Angle error: {quality.angle_error_deg}")
print(f"Corner error: {quality.corner_error_px}")
```

## Usage

### CLI

```bash
# Grade latest predictions
uv run detector-grader --config config.json

# The grader will:
# 1. Resolve latest weights and run inference if needed
# 2. Normalize all OBB quads to pixel space
# 3. Perform bipartite matching using quality-weighted greedy algorithm
# 4. Write comprehensive JSON reports to grade/reports/
```

### Python

```python
from detector_grader.pipeline import grade_predictions
from detector_grader.data import GradeConfig
from pipeline_config import load_pipeline_config

config = load_pipeline_config("config.json")

grade_config = GradeConfig.from_pipeline_config(config)
report = grade_predictions(grade_config)
```

## Integration

Called by `just eval`:

```bash
# Justfile
eval:
    uv run detector-infer --config config.json
    uv run detector-grader --config config.json
```

Output consumed by:
- `dataset-generator`: Curriculum adaptation via hard_examples.jsonl
- `pipeline-profile`: Quality metrics
- Manual analysis: grade_report_*.json

## Greedy Matching Algorithm

```python
def greedy_match(gt_boxes, pred_boxes, quality_fn):
    """Quality-weighted bipartite matching."""
    matches = []
    unmatched_gt = set(range(len(gt_boxes)))
    unmatched_pred = set(range(len(pred_boxes)))
    
    # Compute all pairwise qualities
    qualities = {}
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            qualities[(gi, pi)] = quality_fn(gt, pred)
    
    # Sort by quality descending
    sorted_pairs = sorted(qualities.items(), key=lambda x: x[1].overall, reverse=True)
    
    # Greedily assign
    for (gi, pi), quality in sorted_pairs:
        if gi in unmatched_gt and pi in unmatched_pred:
            if quality.iou >= match_iou_threshold:
                matches.append((gi, pi, quality))
                unmatched_gt.remove(gi)
                unmatched_pred.remove(pi)
    
    return matches, unmatched_gt, unmatched_pred
```

This ensures the best possible matches are made first, maximizing overall quality.
