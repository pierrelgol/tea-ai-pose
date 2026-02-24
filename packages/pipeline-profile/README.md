# pipeline-profile

End-to-end pipeline profiler for performance analysis and bottleneck detection.

## Purpose

Executes a complete pipeline run on a downscaled dataset while collecting timing, CPU, memory, and GPU metrics per stage. Identifies hotspots and validates pipeline health.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         PROFILING WORKFLOW                                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐       │
│   │ fetch-dataset│────▶│fetch-dinov3  │────▶│generate-ds   │────▶│check-dataset │       │
│   └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘       │
│          │                    │                    │                    │                │
│          │  [Metrics]         │  [Metrics]         │  [Metrics]         │  [Metrics]     │
│          │  • Wall time       │  • Wall time       │  • Wall time       │  • Wall time   │
│          │  • CPU %           │  • CPU %           │  • CPU %           │  • CPU %       │
│          │  • RAM             │  • RAM             │  • RAM             │  • RAM         │
│          │  • GPU util        │  • GPU util        │  • GPU util        │  • GPU util    │
│          │  • GPU mem         │  • GPU mem         │  • GPU mem         │  • GPU mem     │
│          │                    │                    │                    │                │
│          └────────────────────┴────────────────────┴────────────────────┘                │
│                                          │                                               │
│                                          ▼                                               │
│                               ┌────────────────────┐                                    │
│                               │     train          │                                    │
│                               │     (50 epochs)    │                                    │
│                               └─────────┬──────────┘                                    │
│                                         │                                                │
│                                         ▼                                                │
│                               ┌────────────────────┐                                    │
│                               │     infer          │                                    │
│                               └─────────┬──────────┘                                    │
│                                         │                                                │
│                                         ▼                                                │
│                               ┌────────────────────┐                                    │
│                               │     eval           │                                    │
│                               └─────────┬──────────┘                                    │
│                                         │                                                │
│                                         ▼                                                │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                      PROFILE REPORT GENERATOR                                    │   │
│   │                                                                                  │   │
│   │  • JSON report (machine-readable)                                                │   │
│   │  • Markdown summary (human-readable)                                             │   │
│   │  • Per-stage timing breakdown                                                    │   │
│   │  • Resource utilization charts                                                   │   │
│   │  • Quality metrics extraction                                                    │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Execution Plan

Stages run sequentially with dependency chain:

| # | Stage | Command | Status |
|---|-------|---------|--------|
| 1 | fetch-dataset | `dataset-fetcher` | Auto |
| 2 | fetch-dinov3 | `dinov3-fetcher` | Auto |
| 3 | label-targets | — | **Skipped** (requires GUI) |
| 4 | generate-dataset | `dataset-generator` | Auto |
| 5 | check-dataset | `augment-checker` | Auto |
| 6 | train | `detector-train` | Auto |
| 7 | infer | `detector-infer` | Auto |
| 8 | eval | `detector-grader` | Auto |
| 9 | review | — | **Skipped** (requires GUI) |

If any stage fails, subsequent stages are marked "skipped" and profiling continues.

## Metrics Collected

### Per-Stage Timing

- **Wall-clock duration** (seconds)
- Sequential execution (no parallelization)

### Resource Sampling

Sampled every 500ms during stage execution:

| Metric | Source | Description |
|--------|--------|-------------|
| CPU % | `psutil.Process.cpu_percent()` | Average and peak utilization |
| Memory | `psutil.Process.memory_info().rss` | Peak RSS in MB |
| GPU Util % | `nvidia-smi` | GPU utilization percentage |
| GPU Memory | `nvidia-smi` | Peak VRAM in MB |

### Quality Metrics

Extracted from grade reports after eval:

| Metric | Description |
|--------|-------------|
| `run_grade_0_100` | Overall pipeline score (0-100) |
| `precision_proxy` | Detection precision |
| `recall_proxy` | Detection recall |
| `miss_rate_proxy` | False negative rate |

## Configuration

```json
{
  "profile": {
    "dataset": "coco128",
    "train_epochs": 50,
    "enable_gpu_sampling": true
  }
}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | string | "coco128" | Dataset profile for profiling |
| `train_epochs` | int | 50 | Reduced epochs for quick iteration |
| `enable_gpu_sampling` | bool | true | Monitor GPU via nvidia-smi |

### Dataset Selection

The `profile.dataset` should be small enough for quick iteration:

| Dataset | Images | Duration | Use Case |
|---------|--------|----------|----------|
| `coco128` | 128 | ~6-10 min | Default profiling |
| `coco32` | 32 | ~2-3 min | Smoke test |

## Output Format

### JSON Report

Machine-readable for CI integration:

```json
{
  "timestamp_utc": "2024-01-15T12:00:00Z",
  "dataset": "coco128",
  "train_epochs": 50,
  "total_duration_s": 420.5,
  "stages": [
    {
      "stage": "train",
      "status": "ok",
      "duration_s": 300.2,
      "cpu_percent_avg": 45.2,
      "cpu_percent_max": 98.1,
      "rss_mb_max": 8192,
      "gpu_util_percent_avg": 85.4,
      "gpu_mem_mb_max": 6144
    }
  ],
  "quality_metrics": {
    "run_grade_0_100": 72.5,
    "precision_proxy": 0.85,
    "recall_proxy": 0.78
  }
}
```

### Markdown Summary

Human-readable report with:
- Execution summary table
- Resource utilization per stage
- Quality metrics
- Failure logs (if any)

Example:
```markdown
# Pipeline Profile Report

## Summary
| Stage | Duration | CPU Avg | RAM Max | GPU Util | Status |
|-------|----------|---------|---------|----------|--------|
| fetch-dataset | 8.2s | 12% | 256MB | 0% | ok |
| train | 302.1s | 78% | 8.2GB | 92% | ok |

## Quality
- Grade: 72.5/100
- Precision: 0.85
- Recall: 0.78
```

## Usage

### CLI

```bash
# Profile with defaults from config.json
uv run pipeline-profile --config config.json

# Override dataset and epochs
uv run pipeline-profile --config config.json --dataset coco32 --train-epochs 10

# Disable GPU monitoring
uv run pipeline-profile --config config.json --no-gpu-sampling

# Custom report directory
uv run pipeline-profile --config config.json --report-dir ./profiles
```

### Python

```python
from pipeline_profile.cli import main
from pipeline_config import load_pipeline_config

config = load_pipeline_config("config.json")
main(config)
```

## Implementation Details

### Process Monitoring

```python
import subprocess
import threading
import psutil

# Spawn stage in subprocess
proc = subprocess.Popen(cmd, ...)
psutil_proc = psutil.Process(proc.pid)

# Monitor in background thread
def monitor():
    while not stop_event.is_set():
        metrics = {
            'cpu': psutil_proc.cpu_percent(),
            'rss': psutil_proc.memory_info().rss / 1024 / 1024,
            'gpu_util': sample_nvidia_smi_util(),
            'gpu_mem': sample_nvidia_smi_memory()
        }
        history.append(metrics)
        time.sleep(0.5)

monitor_thread = threading.Thread(target=monitor)
monitor_thread.start()
proc.wait()
stop_event.set()
```

### Temp Config Generation

Creates isolated config to avoid polluting main workspace:

```python
def create_temp_config(base_config, temp_dir):
    temp_config = {
        **base_config,
        'paths': {
            k: str(Path(v).resolve())  # Convert to absolute
            for k, v in base_config['paths'].items()
        },
        'run': {
            **base_config['run'],
            'dataset': profile_dataset,  # Override
        },
        'train': {
            **base_config['train'],
            'epochs': profile_epochs,    # Override
            'wandb_enabled': False       # Disable
        }
    }
    return temp_config
```

## Performance Expectations

| Stage | coco128 (50 epochs) | Dominant Resource |
|-------|--------------------|--------------------|
| fetch-dataset | 5-10s | Network/Disk |
| fetch-dinov3 | 0s (cached) | Network/Disk |
| generate-dataset | 30-60s | CPU |
| check-dataset | 10-20s | CPU/Disk |
| train | 5-8 min | GPU |
| infer | 30-60s | GPU |
| eval | 20-40s | CPU |
| **Total** | **6-10 min** | — |

## Integration

Called by `just profile-pipeline`:

```bash
# Justfile
profile-pipeline:
    uv run pipeline-profile --config config.json
```

Use cases:
- **CI/CD**: Validate pipeline health before merge
- **Benchmarking**: Compare performance across commits
- **Debugging**: Identify slow stages
- **Resource planning**: Determine hardware requirements
