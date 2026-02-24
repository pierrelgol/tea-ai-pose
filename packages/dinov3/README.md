# dinov3

Local DINOv3 teacher model bridge for knowledge distillation in detector training.

## Purpose

Wraps a local DINOv3 Vision Transformer (ViT) checkpoint to provide feature extraction services for the `detector-train` distillation pipeline. Operates in eval mode with frozen weights.

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                         DINOv3 TEACHER BRIDGE                                            │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   INPUT: Images [B, 3, H, W]                                                            │
│          │                                                                               │
│          ▼                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                      PREPROCESSING                                               │   │
│   │   1. Resize: Bilinear to 518×518 (from preprocessor_config.json)               │   │
│   │   2. Normalize: Per-channel mean/std subtraction                               │   │
│   │   3. To Tensor: [B, 3, 518, 518]                                               │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                               │
│          ▼                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
│   │                      DINOv3 ViT (frozen)                                         │   │
│   │                                                                                  │   │
│   │   Input: [B, 3, 518, 518]                                                       │   │
│   │   Patchify: 16×16 patches → 196 patches + CLS token + 4 registers              │   │
│   │   Transformer: 24 layers (ViT-L)                                               │   │
│   │                                                                                  │   │
│   │   Outputs:                                                                       │   │
│   │   ├── CLS Token: [B, 1024]      (global image representation)                  │   │
│   │   └── Patch Tokens: [B, 196, 1024] → [B, 1024, 14, 14] (spatial features)      │   │
│   │                                                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────────────┘   │
│          │                                                                               │
│          ├──▶ extract_features() ──▶ [B, 1024] (CLS)                                    │
│          │                                                                               │
│          └──▶ extract_feature_map() ──▶ [B, 1024, 14, 14] (spatial)                    │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Architecture

The bridge exposes two primary extraction modes used during distillation:

### 1. CLS Token Extraction

Returns the [CLS] token from the final transformer layer (global image representation):

```python
from dinov3_bridge.model import DinoV3Teacher
from dinov3_bridge.config import DinoV3Config

cfg = DinoV3Config(root=Path("dinov3"))
teacher = DinoV3Teacher(cfg, device=torch.device("cuda:0"))

features = teacher.extract_features(images)
# Shape: [B, hidden_size] = [B, 1024] for ViT-L
```

### 2. Patch Token Map Extraction

Returns spatial feature maps from patch tokens for pixel-aligned distillation:

```python
feature_map = teacher.extract_feature_map(images)
# Shape: [B, C, H, W] = [B, 1024, 14, 14] for ViT-L @ 518px
```

## Technical Details

### Preprocessing Pipeline

1. **Resize**: Bilinear interpolation to model input size (518×518 for ViT-L)
2. **Normalize**: Per-channel mean/std subtraction using preprocessor config
3. **Device transfer**: Move to target device (CUDA/MPS/CPU)

### Configuration Loading

Reads `preprocessor_config.json` from the model root:

```json
{
  "image_mean": [0.485, 0.456, 0.406],
  "image_std": [0.229, 0.224, 0.225],
  "size": {"height": 518, "width": 518},
  "patch_size": 16,
  "num_register_tokens": 4
}
```

### Frozen Inference

```python
@torch.no_grad()
def extract_features(self, images: Tensor) -> Tensor:
    # Model remains in eval mode, no gradient computation
    outputs = self.model(images)
    return outputs.last_hidden_state[:, 0]  # CLS token
```

## Model Variants

| Model | Params | Hidden Dim | Layers | Heads | Output Shape @ 518px |
|-------|--------|------------|--------|-------|---------------------|
| ViT-S/16 | 21M | 384 | 12 | 6 | [B, 384, 32, 32] |
| ViT-B/16 | 86M | 768 | 12 | 12 | [B, 768, 32, 32] |
| ViT-L/16 | 300M | 1024 | 24 | 16 | [B, 1024, 32, 32] |
| ViT-H/16 | 840M | 1280 | 32 | 20 | [B, 1280, 32, 32] |

**Recommended for TEA-AI:** ViT-L/16 (balance of quality and efficiency)

## API Reference

### `DinoV3Config`

Configuration for the teacher model.

```python
@dataclass
class DinoV3Config:
    root: Path  # Path to model directory
```

### `DinoV3Teacher`

Teacher model wrapper.

```python
class DinoV3Teacher:
    def __init__(self, config: DinoV3Config, device: torch.device):
        """Load model and move to device."""
        
    def extract_features(self, images: Tensor) -> Tensor:
        """Extract CLS token features."""
        # Returns: [B, hidden_size]
        
    def extract_feature_map(self, images: Tensor) -> Tensor:
        """Extract spatial feature map from patch tokens."""
        # Returns: [B, C, H, W]
```

## Usage

### Python API

```python
from pathlib import Path
import torch
from dinov3_bridge.config import DinoV3Config
from dinov3_bridge.model import DinoV3Teacher

# Configuration
cfg = DinoV3Config(root=Path("dinov3"))

# Load teacher
device = torch.device("cuda:0")
teacher = DinoV3Teacher(cfg, device)

# Extract features
images = torch.randn(4, 3, 512, 512).to(device)

with torch.no_grad():
    # Global features
    cls_features = teacher.extract_features(images)
    # Shape: [4, 1024]
    
    # Spatial features
    feature_map = teacher.extract_feature_map(images)
    # Shape: [4, 1024, 14, 14]
```

### Integration in Training

```python
# In DinoOBBTrainer
from dinov3_bridge.model import DinoV3Teacher

class DinoOBBTrainer:
    def __init__(self, config, device):
        self.teacher = DinoV3Teacher(
            DinoV3Config(root=config.dino_root),
            device
        )
        self.teacher.eval()  # Frozen
        
    def compute_distillation_loss(self, images, student_features):
        with torch.no_grad():
            teacher_features = self.teacher.extract_feature_map(images)
        
        # Align and compute loss
        loss = F.cosine_embedding_loss(
            student_features.flatten(2).transpose(1, 2),
            teacher_features.flatten(2).transpose(1, 2),
            target=torch.ones(...)
        )
        return loss
```

## File Structure

The bridge expects a HuggingFace-compatible model directory:

```
dino_root/
├── config.json                    # Model architecture
│   {
│     "architectures": ["Dinov3Model"],
│     "hidden_size": 1024,
│     "num_hidden_layers": 24,
│     "num_attention_heads": 16,
│     "patch_size": 16,
│     "num_registers": 4
│   }
├── preprocessor_config.json       # Image preprocessing
├── model.safetensors             # Model weights
└── README.md                      # Model card
```

## Performance

| Model | VRAM | Throughput | Notes |
|-------|------|------------|-------|
| ViT-S | ~1GB | ~200 img/s | Fastest |
| ViT-B | ~2GB | ~100 img/s | Balanced |
| ViT-L | ~4GB | ~50 img/s | Recommended |
| ViT-H | ~8GB | ~25 img/s | Best quality |

**Throughput measured on A100 @ 518×518**

## Dependencies

- `torch`: Core tensor operations
- `transformers`: HuggingFace model loading
- `safetensors`: Efficient weight storage

## Integration

Used exclusively by `detector-train`:

```python
# In detector-train's DinoOBBTrainer
teacher_features = teacher.extract_feature_map(images)
student_features = student.backneck(...)  # Hooked layers

# Compute distillation loss
loss_distill = cosine_distance(
    teacher_features[mask],
    student_features[mask]
)
```

The DINOv3 teacher provides high-quality visual features that guide the YOLO student to learn better orientation estimation and object boundaries.
