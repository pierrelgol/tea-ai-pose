from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import load_pipeline_config

from .fetch import fetch_dataset
from .profiles import resolve_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a selected dataset into top-level dataset directory")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    dataset_name = str(shared.dataset.get("name") or shared.run["dataset"])

    profile, profile_path = resolve_profile(dataset_name=dataset_name, profile_path=None)
    dataset_path = fetch_dataset(
        profile=profile,
        dataset_root=shared.paths["dataset_root"],
        source_url_override=None,
        dataset_dir_name_override=None,
    )
    print(f"{profile.name} ready at: {dataset_path}")
    print(f"profile: {profile_path}")
    print(f"train_images: {dataset_path / profile.train_images_rel}")
    print(f"val_images: {dataset_path / profile.val_images_rel}")


if __name__ == "__main__":
    main()
