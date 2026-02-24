from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class DinoV3Config:
    root: Path
    model_file: Path
    config_file: Path
    preprocessor_file: Path



def resolve_local_dinov3_root(root: Path | None = None) -> DinoV3Config:
    base = (root or Path("dinov3")).resolve()
    model_file = base / "model.safetensors"
    config_file = base / "config.json"
    preprocessor_file = base / "preprocessor_config.json"

    missing: list[Path] = [
        p
        for p in (model_file, config_file, preprocessor_file)
        if not p.exists()
    ]
    if missing:
        missing_list = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            "DINOv3 local files are required but missing: "
            f"{missing_list}. Expected local model under {base}."
        )

    return DinoV3Config(
        root=base,
        model_file=model_file,
        config_file=config_file,
        preprocessor_file=preprocessor_file,
    )
