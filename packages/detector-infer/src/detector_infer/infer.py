from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from .config import InferConfig
from .dataset import list_split_images
from .writer import _format_pose_line, write_prediction_file
from pipeline_runtime_utils import resolve_device, set_seed


import numpy as np


def _result_to_lines(res: Any) -> list[str]:
    lines: list[str] = []

    boxes = getattr(res, "boxes", None)
    keypoints = getattr(res, "keypoints", None)
    if boxes is None or keypoints is None:
        raise RuntimeError("model prediction does not expose pose output; pose model/weights are required")

    if not hasattr(boxes, "xywhn") or boxes.xywhn is None:
        raise RuntimeError("pose prediction missing normalized bbox output")
    bbox = boxes.xywhn.cpu().numpy()
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") and boxes.conf is not None else np.ones((bbox.shape[0],), dtype=np.float32)
    classes = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") and boxes.cls is not None else np.zeros((bbox.shape[0],), dtype=int)

    if hasattr(keypoints, "xyn") and keypoints.xyn is not None:
        kxy = keypoints.xyn.cpu().numpy()
    else:
        raise RuntimeError("pose prediction missing normalized keypoint coordinates")
    if hasattr(keypoints, "conf") and keypoints.conf is not None:
        kconf = keypoints.conf.cpu().numpy()
    else:
        kconf = np.ones((kxy.shape[0], kxy.shape[1]), dtype=np.float32)

    bbox = np.clip(bbox, 0.0, 1.0)
    kxy = np.clip(kxy, 0.0, 1.0)
    for i in range(bbox.shape[0]):
        kpt = np.concatenate([kxy[i], np.clip(kconf[i][:, None], 0.0, 1.0)], axis=1)
        lines.append(_format_pose_line(int(classes[i]), bbox[i], kpt, float(confs[i])))
    return lines


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
