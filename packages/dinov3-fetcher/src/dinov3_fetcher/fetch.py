from __future__ import annotations

from pathlib import Path
import shutil

from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DEFAULT_PATTERNS = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
    "README.md",
    "LICENSE*",
]


def fetch_dinov3(
    *,
    output_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str | None = None,
    force: bool = False,
) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if force and output_dir.exists():
        shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(output_dir),
        local_dir_use_symlinks=False,
        allow_patterns=DEFAULT_PATTERNS,
    )
    return output_dir
