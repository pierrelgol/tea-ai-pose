from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from .config import InferConfig
from .dataset import list_split_images
from .writer import _format_obb_line, write_prediction_file
from pipeline_runtime_utils import resolve_device, set_seed


import numpy as np


def _result_to_lines(res: Any) -> list[str]:
    lines: list[str] = []

    if getattr(res, "obb", None) is not None:
        obb = res.obb
        if hasattr(obb, "xyxyxyxyn") and obb.xyxyxyxyn is not None:
            coords = obb.xyxyxyxyn.cpu().numpy()
        elif hasattr(obb, "xyxyxyxy") and obb.xyxyxyxy is not None:
            px = obb.xyxyxyxy.cpu().numpy()
            h, w = res.orig_shape[:2]
            coords = px.astype(np.float64)
            coords[:, :, 0] /= w
            coords[:, :, 1] /= h
        else:
            coords = np.zeros((0, 4, 2), dtype=np.float32)

        confs = obb.conf.cpu().numpy() if hasattr(obb, "conf") else np.ones((coords.shape[0],), dtype=np.float32)
        classes = obb.cls.cpu().numpy().astype(int) if hasattr(obb, "cls") else np.zeros((coords.shape[0],), dtype=int)

        coords = np.clip(coords, 0.0, 1.0)
        for i in range(coords.shape[0]):
            lines.append(_format_obb_line(int(classes[i]), coords[i], float(confs[i])))
        return lines

    raise RuntimeError("Model prediction does not expose OBB output; OBB model/weights are required")


def run_inference(config: InferConfig) -> dict:
    config.validate()
    set_seed(config.seed)
    device = resolve_device(config.device)
    model_labels_root = config.output_root / config.model_name / "labels"
    if model_labels_root.exists():
        shutil.rmtree(model_labels_root)

    from ultralytics import YOLO

    model = YOLO(str(config.weights))
    written = 0
    total_images = 0

    for split in config.splits:
        images = list_split_images(config.dataset_root, split)
        for start in range(0, len(images), config.batch_size):
            batch_paths = images[start : start + config.batch_size]
            if not batch_paths:
                continue
            total_images += len(batch_paths)
            results = model.predict(
                source=[str(p) for p in batch_paths],
                conf=config.conf_threshold,
                iou=config.iou_threshold,
                imgsz=config.imgsz,
                device=device,
                verbose=False,
            )
            for image_path, res in zip(batch_paths, results):
                lines = _result_to_lines(res)
                out_path = config.output_root / config.model_name / "labels" / split / f"{image_path.stem}.txt"
                write_prediction_file(out_path, lines, save_empty=config.save_empty)
                written += 1

    return {
        "status": "ok",
        "weights": str(config.weights),
        "model_name": config.model_name,
        "output_root": str(config.output_root),
        "resolved_device": device,
        "images_processed": total_images,
        "label_files_written": written,
    }
