from __future__ import annotations

import json
from pathlib import Path


def resolve_latest_weights_from_artifacts(artifacts_root: Path) -> Path:
    latest_file = artifacts_root / "latest_run.json"
    if latest_file.exists():
        payload = json.loads(latest_file.read_text(encoding="utf-8"))
        best = payload.get("weights_best")
        if isinstance(best, str) and best:
            p = Path(best)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.exists():
                return p

    candidates = sorted(
        artifacts_root.glob("*/runs/*/train/ultralytics/*/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    legacy_candidates = sorted(
        (artifacts_root / "runs").glob("**/weights/best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if legacy_candidates:
        return legacy_candidates[0]

    raise FileNotFoundError(f"could not resolve latest best weights from {artifacts_root}")
