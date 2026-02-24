from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import load_pipeline_config

from .fetch import DEFAULT_REPO_ID, fetch_dinov3


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch DINOv3 files from Hugging Face")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    dino_root = Path(str(shared.train.get("dino_root", "dinov3")))
    if not dino_root.is_absolute():
        dino_root = (shared.config_root / dino_root).resolve()

    out = fetch_dinov3(
        output_dir=dino_root,
        repo_id=str(args.repo_id),
        revision=(str(args.revision) if args.revision else None),
        force=bool(args.force),
    )

    print(f"status: ok")
    print(f"repo_id: {args.repo_id}")
    print(f"output_dir: {out}")


if __name__ == "__main__":
    main()
