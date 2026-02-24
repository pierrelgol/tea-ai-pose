# dataset-fetcher

Declarative dataset acquisition with support for remote URLs, local paths, and deterministic subsetting.

## Purpose

Handles ingestion of background datasets into the pipeline. Uses JSON profiles to define dataset sources, decoupling code from dataset-specific URLs or directory layouts.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        DATASET ACQUISITION FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐ │
│   │   Profile    │───▶│   resolve_   │───▶│   fetch_     │───▶│   validate   │ │
│   │   JSON       │    │   profile()  │    │   dataset()  │    │   dataset()  │ │
│   └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘ │
│          │                                                             │       │
│          │ Source definition                                    Integrity │       │
│          │ (URLs, local path)                                    checks    │       │
│          │                                                             │       │
│          ▼                                                             ▼       │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                    DATASET ROOT (configured)                             │  │
│   │                                                                          │  │
│   │   dataset_root/                                                          │  │
│   │   └── {dataset_dir_name}/                                                │  │
│   │       ├── images/                                                        │  │
│   │       │   ├── train/                                                     │  │
│   │       │   └── val/                                                       │  │
│   │       └── labels/                                                        │  │
│   │           ├── train/                                                     │  │
│   │           └── val/                                                       │  │
│   │                                                                          │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Source Types

### remote_zip / ultralytics_zip

Downloads and extracts zip archives from URLs with multi-URL fallback.

```json
{
  "name": "coco128",
  "source_type": "ultralytics_zip",
  "urls": [
    "https://ultralytics.com/assets/coco128.zip"
  ],
  "dataset_dir_name": "coco128",
  "required_paths_rel": [
    "images/train",
    "labels/train"
  ]
}
```

**Features:**
- Multi-URL fallback (tries next URL on failure)
- Resume-capable downloads
- Extracts to `{dataset_root}/{dataset_dir_name}/`

### local_dir

Symlinks an existing local directory (fast, non-destructive):

```json
{
  "name": "mydataset",
  "source_type": "local_dir",
  "local_path": "/data/mydataset",
  "dataset_dir_name": "mydataset"
}
```

Creates symlink: `{dataset_root}/{dataset_dir_name} -> /data/mydataset`

### coco_subset_local

Deterministically samples from a local COCO-format dataset:

```json
{
  "name": "coco_subset",
  "source_type": "coco_subset_local",
  "local_path": "/data/coco",
  "train_images_rel": "images/train2017",
  "val_images_rel": "images/val2017",
  "subset_train_max_images": 1000,
  "subset_val_max_images": 100,
  "subset_seed": 42
}
```

Useful for rapid prototyping on smaller scales without duplicating large datasets.

### coco_ids_local

Constructs splits from explicit image ID lists:

```json
{
  "name": "coco_ids",
  "source_type": "coco_ids_local",
  "local_path": "/data/coco",
  "ids_images_rel": "images",
  "ids_train_ids_rel": "train_ids.txt",
  "ids_val_ids_rel": "val_ids.txt",
  "subset_train_max_images": 500,
  "subset_val_max_images": 50
}
```

## Deterministic Sampling

When subsetting, uses seeded cryptographic hashing to ensure consistent samples across runs:

```python
def _rank_key(pair):
    stem = pair[0].stem
    digest = hashlib.sha1(f"{seed}:{split}:{stem}".encode()).hexdigest()
    return digest

# Sort by hash, take first N
ranked = sorted(pairs, key=_rank_key)
selected = ranked[:limit]
```

**Guarantees:**
- Same images selected on different machines
- Reproducible train/val splits
- No random state dependencies

## Profile Schema

```json
{
  "name": "coco128",
  "source_type": "remote_zip",
  "urls": ["https://ultralytics.com/assets/coco128.zip"],
  "dataset_dir_name": "coco128",
  "required_paths_rel": [
    "images/train",
    "images/val",
    "labels/train",
    "labels/val"
  ],
  "subset_seed": 42,
  "subset_train_max_images": null,
  "subset_val_max_images": null
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Profile identifier |
| `source_type` | string | One of: `remote_zip`, `ultralytics_zip`, `local_dir`, `coco_subset_local`, `coco_ids_local` |
| `dataset_dir_name` | string | Target directory name in `dataset_root` |
| `required_paths_rel` | string[] | Paths that must exist after fetch (validation) |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `urls` | string[] | List of download URLs (for remote types) |
| `local_path` | string | Source directory path (for local types) |
| `subset_seed` | int | RNG seed for deterministic sampling |
| `subset_train_max_images` | int | Train split size limit |
| `subset_val_max_images` | int | Val split size limit |

## API Reference

### `resolve_profile(name, profile_path) -> (DatasetProfile, Path)`

Load dataset profile from configuration.

**Parameters:**
- `name`: Profile name (without .json extension)
- `profile_path`: Directory containing profile JSONs (uses `configs_root` from main config if None)

**Returns:**
- `DatasetProfile`: Parsed profile object
- `Path`: Path to profile file

### `fetch_dataset(profile, dataset_root) -> Path`

Fetch dataset according to profile.

**Parameters:**
- `profile`: `DatasetProfile` from `resolve_profile()`
- `dataset_root`: Root directory for datasets

**Returns:** Path to fetched dataset directory

**Raises:**
- `RuntimeError`: Download failed (all URLs exhausted)
- `FileNotFoundError`: Local path does not exist
- `ValidationError`: Post-fetch validation failed

### `validate_dataset(profile, dataset_dir) -> None`

Post-fetch validation.

**Checks:**
- All `required_paths_rel` exist
- Images/labels directories non-empty
- Consistent train/val splits

Raises `RuntimeError` on failure with descriptive message.

## Usage

### Python API

```python
from pathlib import Path
from dataset_fetcher.profiles import resolve_profile
from dataset_fetcher.fetch import fetch_dataset

# Load profile from configs/datasets/
profile, _ = resolve_profile("coco128", profile_path=None)

# Fetch and validate
dataset_path = fetch_dataset(
    profile=profile,
    dataset_root=Path("./datasets"),
)

print(f"Dataset ready at: {dataset_path}")
```

### CLI

```bash
# Fetch from configured source
uv run dataset-fetcher --config config.json
```

Reads `config.run.dataset` to determine which profile to fetch.

## Directory Structure

After fetch:

```
dataset_root/
└── {dataset_dir_name}/
    ├── images/
    │   ├── train/
    │   │   ├── 0001.jpg
    │   │   └── ...
    │   └── val/
    │       ├── 0001.jpg
    │       └── ...
    └── labels/
        ├── train/
        │   ├── 0001.txt
        │   └── ...
        └── val/
            ├── 0001.txt
            └── ...
```

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `FileNotFoundError` | `local_path` does not exist | Verify path in profile |
| `RuntimeError` | Download failed (all URLs exhausted) | Check network, try alternative URLs |
| `ValidationError` | Post-fetch validation failed | Check profile `required_paths_rel` |
| `zipfile.BadZipFile` | Corrupted archive | Clear cache, retry download |

## Integration

Called by `just fetch-dataset`:

```bash
# Justfile
fetch-dataset:
    uv run dataset-fetcher --config config.json
```

The fetcher reads the pipeline config to determine:
- Which dataset profile to fetch (`config.run.dataset`)
- Where to put it (`config.paths.dataset_root`)
- Profile search path (`config.paths.configs_root`)
