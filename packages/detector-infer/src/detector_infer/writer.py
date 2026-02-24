from __future__ import annotations

from pathlib import Path

import numpy as np


def _format_obb_line(class_id: int, corners_norm: np.ndarray, conf: float) -> str:
    flat = corners_norm.reshape(-1)
    coords = " ".join(f"{float(v):.10f}" for v in flat)
    return f"{class_id} {coords} {float(conf):.10f}"


def write_prediction_file(path: Path, lines: list[str], save_empty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not lines and not save_empty:
        if path.exists():
            path.unlink()
        return
    text = "\n".join(lines)
    if text:
        text += "\n"
    path.write_text(text, encoding="utf-8")
