from __future__ import annotations

from pathlib import Path

from .profiles import DatasetProfile


class DatasetValidationError(RuntimeError):
    pass


def validate_dataset(profile: DatasetProfile, dataset_dir: Path) -> None:
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise DatasetValidationError(f"dataset directory does not exist: {dataset_dir}")

    missing: list[Path] = []
    for rel in profile.required_paths_rel:
        p = dataset_dir / rel
        if not p.exists():
            missing.append(p)

    if missing:
        miss = "\n".join(str(p) for p in missing)
        raise DatasetValidationError(
            f"dataset '{profile.name}' failed strict validation. Missing required paths:\n{miss}"
        )
