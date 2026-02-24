# dinov3-fetcher

Retrieves DINOv3 Vision Transformer teacher models from Hugging Face Hub for knowledge distillation.

## Purpose

Downloads DINOv3 checkpoints to a local cache directory used by the training pipeline. Handles authentication, resume-capable downloads, and directory structure setup.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DINOv3 TEACHER ACQUISITION                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   Hugging Face Hub                                                               │
│   │                                                                              │
│   ├── facebook/dinov3-vitl16-pretrain-lvd1689m  (ViT-L/16, 300M params)        │
│   ├── facebook/dinov3-vitb16-pretrain-lvd1689m  (ViT-B/16, 86M params)         │
│   └── ...                                                                        │
│   │                                                                              │
│   │  huggingface-hub library                                                      │
│   │  • Authentication via HF_TOKEN                                               │
│   │  • Resume-capable downloads                                                  │
│   ▼                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         Local Cache (dino_root)                          │   │
│   │                                                                          │   │
│   │   dinov3/                                                                │   │
│   │   ├── config.json                    # Model architecture                │   │
│   │   ├── preprocessor_config.json       # Image preprocessing               │   │
│   │   ├── model.safetensors             # Model weights (~1.2GB)            │   │
│   │   └── README.md                      # Model card                        │   │
│   │                                                                          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Configuration

Controlled via `config.json`:

```json
{
  "paths": {
    "dataset_root": "dataset"
  },
  "train": {
    "dino_root": "dinov3"
  }
}
```

- `dino_root`: Relative path (resolved against config location) or absolute path
- Default model: `facebook/dinov3-vitl16-pretrain-lvd1689m`

## Fetch Process

1. **Resolve path**: Convert relative `dino_root` to absolute path
2. **Check cache**: Skip download if valid checkpoint exists
3. **HF Hub download**:
   - Authenticate (uses `HF_TOKEN` env var if available)
   - Download `model.safetensors`, `config.json`, `preprocessor_config.json`
   - Resume interrupted downloads automatically
4. **Validate**: Ensure required files present

## Model Variants

| Model | Params | Embedding Dim | Heads | Description |
|-------|--------|---------------|-------|-------------|
| `dinov3-vits16` | 21M | 384 | 6 | Small, fast inference |
| `dinov3-vitb16` | 86M | 768 | 12 | Base model |
| `dinov3-vitl16` | 300M | 1024 | 16 | Large, recommended |
| `dinov3-vith16` | 840M | 1280 | 20 | Huge, best quality |
| `dinov3-vit7b` | 6.7B | 4096 | 32 | 7B parameter model |

**Recommended for TEA-AI:** `dinov3-vitl16` (balance of quality and resource usage)

## API Reference

### `fetch_dinov3(dino_root, model_name) -> Path`

Fetch DINOv3 model from Hugging Face Hub.

**Parameters:**
- `dino_root`: Local directory to cache model
- `model_name`: Hugging Face model ID (e.g., `facebook/dinov3-vitl16-pretrain-lvd1689m`)

**Returns:** Path to model directory

**Raises:**
- `AuthenticationError`: Invalid/missing HF token
- `RepositoryNotFoundError`: Model not found
- `LocalEntryNotFoundError`: Partial/corrupted download

### `validate_dinov3(dino_root) -> bool`

Check if valid DINOv3 model exists locally.

**Checks:**
- `config.json` exists
- `preprocessor_config.json` exists
- `model.safetensors` exists and non-empty

## Usage

### Python API

```python
from pathlib import Path
from dinov3_fetcher.fetch import fetch_dinov3

# Fetch model
model_path = fetch_dinov3(
    dino_root=Path("./dinov3"),
    model_name="facebook/dinov3-vitl16-pretrain-lvd1689m"
)

print(f"Model ready at: {model_path}")
```

### CLI

```bash
# Fetch from configured source
uv run dinov3-fetcher --config config.json

# Results in:
# dinov3/
# ├── config.json
# ├── preprocessor_config.json
# └── model.safetensors
```

## Authentication

For gated models or to avoid rate limits:

```bash
export HF_TOKEN="your_huggingface_token"
uv run dinov3-fetcher --config config.json
```

Get token from: https://huggingface.co/settings/tokens

## Cache Behavior

- **Check first**: Validates existing files before downloading
- **Resume support**: Interrupted downloads continue where left off
- **No versioning**: Overwrites existing model if re-fetched

## Error Handling

| Error | Cause | Resolution |
|-------|-------|------------|
| `AuthenticationError` | Invalid/missing HF token | Set `HF_TOKEN` env var |
| `RepositoryNotFoundError` | Wrong model name | Check model ID on HF Hub |
| `LocalEntryNotFoundError` | Partial download | Automatic resume or clear cache |
| `OSError` | Disk full | Free up space (~1.5GB needed) |

## Dependencies

- `huggingface-hub>=0.20.0`: Hub client with resume support
- `pipeline-config`: Path resolution from shared config

## Integration

Called by `just fetch-dinov3`:

```bash
# Justfile
fetch-dinov3:
    uv run dinov3-fetcher --config config.json
```

Used by `detector-train` during `DinoOBBTrainer` initialization:

```python
# In detector-train
from dinov3_bridge.model import DinoV3Teacher
from dinov3_bridge.config import DinoV3Config

cfg = DinoV3Config(root=config.train["dino_root"])
teacher = DinoV3Teacher(cfg, device)
```

## File Format

The cached model follows HuggingFace Transformers format:

```
dino_root/
├── config.json
│   {
│     "architectures": ["Dinov3Model"],
│     "hidden_size": 1024,
│     "num_hidden_layers": 24,
│     "num_attention_heads": 16,
│     "patch_size": 16,
│     ...
│   }
├── preprocessor_config.json
│   {
│     "image_mean": [0.485, 0.456, 0.406],
│     "image_std": [0.229, 0.224, 0.225],
│     "size": {"height": 518, "width": 518}
│   }
└── model.safetensors  # Weights in safetensors format
```

The `preprocessor_config.json` is essential for correct image preprocessing during distillation.
