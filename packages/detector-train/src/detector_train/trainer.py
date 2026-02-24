from __future__ import annotations

from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

import csv
import json
import os
import re
import shutil
import cv2
import numpy as np

from .config import TrainConfig
from .data_yaml import write_data_yaml
from .dino_viz import save_dino_visualizations
from .dino_trainer import DinoDistillConfig, DinoOBBTrainer
from .wandb_logger import finish_wandb, init_wandb, log_wandb
from pipeline_runtime_utils import (
    corners_norm_to_px,
    index_ground_truth,
    load_obb_labels,
    load_prediction_labels,
    resolve_device,
)


def _configure_torch_runtime(config: TrainConfig) -> None:
    try:
        import torch

        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "matmul"):
                torch.backends.cuda.matmul.allow_tf32 = bool(config.tf32)
            if hasattr(torch.backends, "cudnn"):
                torch.backends.cudnn.allow_tf32 = bool(config.tf32)
                torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
    except Exception:
        pass


def _dataset_images_size_bytes(dataset_root: Path) -> int:
    total = 0
    for split in ("train", "val"):
        images_dir = dataset_root / "images" / split
        if not images_dir.exists():
            continue
        for p in images_dir.iterdir():
            if not p.is_file():
                continue
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return total


def _resolve_cache_mode(requested: str, dataset_root: Path) -> str | bool:
    if requested == "false":
        return False
    if requested in {"ram", "disk"}:
        return requested
    # auto: prefer RAM cache only when dataset comfortably fits available memory.
    dataset_bytes = _dataset_images_size_bytes(dataset_root)
    if dataset_bytes <= 0:
        return "disk"
    available = 0
    try:
        import psutil  # type: ignore

        available = int(psutil.virtual_memory().available)
    except Exception:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            available = int(page_size * avail_pages)
        except Exception:
            available = 0
    if available > 0 and dataset_bytes * 3 <= available:
        return "ram"
    return "disk"


def _resolve_workers(config: TrainConfig) -> int:
    base = max(0, int(config.workers))
    if not bool(config.workers_auto):
        return base
    cpu = os.cpu_count() or 4
    reserve = 2 if cpu > 4 else 1
    auto = max(1, cpu - reserve)
    auto = min(auto, int(config.workers_max))
    if str(config.throughput_mode) == "max_gpu":
        return auto
    return min(auto, max(1, base)) if base > 0 else auto


def _initial_batch_for_mode(config: TrainConfig) -> int:
    base = max(1, int(config.batch))
    if str(config.batch_mode) != "auto_max":
        return base
    cap = max(1, int(config.batch_max))
    target = float(np.clip(float(config.batch_utilization_target), 0.5, 1.0))
    if str(config.throughput_mode) == "max_gpu":
        return max(base, min(cap, int(round(cap * target))))
    growth = 1.0 + 0.5 * (target - 0.5)
    return min(cap, max(base, int(round(base * growth))))


def _is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def _json_safe(obj):
    return json.loads(json.dumps(obj, default=str))


def _to_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    if out in (float("inf"), float("-inf")):
        return None
    return out


def _normalize_key(key: str) -> str:
    key = key.strip().lower()
    key = key.replace("(", "").replace(")", "")
    key = re.sub(r"\s+", "", key)
    return key


def _canonical_key(raw_key: str) -> str | None:
    k = _normalize_key(raw_key)

    if "metrics/precision" in k:
        return "val/precision"
    if "metrics/recall" in k:
        return "val/recall"
    if "metrics/map50-95" in k or "metrics/map50_95" in k:
        return "val/map50_95"
    if "metrics/map50" in k:
        return "val/map50"

    if "train/box_loss" in k or k.endswith("box_loss"):
        return "train/loss_box"
    if "train/cls_loss" in k or k.endswith("cls_loss"):
        return "train/loss_cls"
    if "train/dfl_loss" in k or k.endswith("dfl_loss"):
        return "train/loss_dfl"

    if k in {"lr", "train/lr"} or "lr/pg0" in k:
        return "train/lr"

    if "instances" in k:
        return "train/num_instances"

    return None


def _extract_epoch_metrics(trainer) -> dict[str, float]:
    out: dict[str, float] = {}

    metrics = getattr(trainer, "metrics", None)
    if isinstance(metrics, dict):
        speed_vals: list[float] = []
        for key, value in metrics.items():
            fv = _to_float(value)
            if fv is None:
                continue
            canonical = _canonical_key(key)
            if canonical is not None:
                out[canonical] = fv
            nk = _normalize_key(str(key))
            if nk.startswith("speed/"):
                speed_vals.append(fv)
        if speed_vals:
            out["train/speed_ms_per_img"] = float(sum(speed_vals))

    tloss = getattr(trainer, "tloss", None)
    if tloss is not None:
        try:
            arr = tloss.detach().cpu().numpy().reshape(-1)
            if arr.size > 0:
                box = _to_float(arr[0])
                if box is not None:
                    out["train/loss_box"] = box
            if arr.size > 1:
                cls = _to_float(arr[1])
                if cls is not None:
                    out["train/loss_cls"] = cls
            if arr.size > 2:
                dfl = _to_float(arr[2])
                if dfl is not None:
                    out["train/loss_dfl"] = dfl
        except Exception:
            pass

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is not None:
        try:
            lrs = [float(pg.get("lr", 0.0)) for pg in optimizer.param_groups]
            if lrs:
                out["train/lr"] = lrs[0]
        except Exception:
            pass

    loss_keys = ("train/loss_box", "train/loss_cls", "train/loss_dfl")
    losses = [out[k] for k in loss_keys if k in out]
    if losses:
        out["train/loss_total"] = float(sum(losses))
    dino_epoch_loss = _to_float(getattr(trainer, "_dino_epoch_loss", None))
    if dino_epoch_loss is not None:
        out["train/loss_distill"] = dino_epoch_loss
    dino_weight = _to_float(getattr(trainer, "_dino_last_weight", None))
    if dino_weight is not None:
        out["train/distill_weight"] = dino_weight
    dino_obj = _to_float(getattr(trainer, "_dino_epoch_obj_loss", None))
    if dino_obj is not None:
        out["train/loss_distill_obj"] = dino_obj
    dino_bg = _to_float(getattr(trainer, "_dino_epoch_bg_loss", None))
    if dino_bg is not None:
        out["train/loss_distill_bg"] = dino_bg

    return out


def _extract_last_metrics(results_csv: Path) -> dict[str, float]:
    if not results_csv.exists():
        return {}

    with results_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        raw_row: dict[str, str] | None = None
        for row in reader:
            raw_row = row
    if not raw_row:
        return {}
    out: dict[str, float] = {}
    speed_vals: list[float] = []

    for key, value in raw_row.items():
        if value is None:
            continue
        fv = _to_float(str(value).strip())
        if fv is None:
            continue

        canonical = _canonical_key(key)
        if canonical is not None:
            out[canonical] = fv

        nk = _normalize_key(key)
        if nk.startswith("speed/"):
            speed_vals.append(fv)

    if speed_vals:
        out["train/speed_ms_per_img"] = float(sum(speed_vals))

    loss_keys = ("train/loss_box", "train/loss_cls", "train/loss_dfl")
    losses = [out[k] for k in loss_keys if k in out]
    if losses:
        out["train/loss_total"] = float(sum(losses))

    return out


def _run_periodic_eval(
    *,
    config: TrainConfig,
    save_dir: Path,
    epoch: int,
    device: str,
    run_name: str,
    wandb_run,
) -> dict[str, Any]:
    from detector_grader.pipeline import GradingConfig, run_grading

    weights_last = save_dir / "weights" / "last.pt"
    if not weights_last.exists():
        log_wandb(wandb_run, {"eval/status": 0.0}, step=epoch)
        return {"status": "skipped", "reason": f"missing checkpoint: {weights_last}"}

    epoch_root = config.artifacts_root / "eval" / "latest"
    pred_root = epoch_root / "predictions"
    reports_dir = epoch_root / "reports"
    result = run_grading(
        GradingConfig(
            dataset_root=config.dataset_root,
            predictions_root=pred_root,
            artifacts_root=config.artifacts_root,
            reports_dir=reports_dir,
            model=run_name,
            weights=weights_last,
            run_inference=True,
            splits=["val"],
            imgsz=config.imgsz,
            device=device,
            conf_threshold=config.eval_conf_threshold,
            infer_iou_threshold=config.eval_iou_threshold,
            match_iou_threshold=config.eval_iou_threshold,
            strict_obb=True,
            seed=config.seed,
        )
    )

    row = result.get("eval_like")
    if row:
        log_wandb(
            wandb_run,
            {
                "eval/status": 1.0,
                "eval/precision": row.get("precision"),
                "eval/recall": row.get("recall"),
                "eval/miss_rate": row.get("miss_rate"),
                "eval/mean_iou": row.get("mean_iou"),
                "eval/mean_center_drift_px": row.get("mean_center_drift_px"),
                "eval/run_grade_0_100": result["aggregate"].get("run_grade_0_100"),
            },
            step=epoch,
        )
    else:
        log_wandb(wandb_run, {"eval/status": 0.0}, step=epoch)

    return {
        "status": "ok",
        "epoch": epoch,
        "weights_last": str(weights_last),
        "epoch_root": str(epoch_root),
        "predictions_root": str(pred_root),
        "grading": result,
    }


def _draw_label_set(
    image_bgr: np.ndarray,
    labels,
    *,
    color: tuple[int, int, int],
    tag: str,
) -> np.ndarray:
    out = image_bgr.copy()
    h, w = out.shape[:2]
    for idx, label in enumerate(labels):
        px = corners_norm_to_px(label.corners_norm, w, h).astype(np.int32)
        cv2.polylines(out, [px.reshape((-1, 1, 2))], True, color, 2)
        x = int(np.min(px[:, 0]))
        y = int(np.min(px[:, 1])) - 6
        cv2.putText(
            out,
            f"{tag} c{int(label.class_id)} #{idx}",
            (max(2, x), max(16, y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def _write_eval_visual_artifacts(
    *,
    config: TrainConfig,
    eval_result: dict[str, Any],
    epoch: int,
    selected_records,
    gt_cache: dict[Path, list[Any]],
    image_cache: dict[str, np.ndarray],
    gt_overlay_cache: dict[str, np.ndarray],
) -> dict[str, Any]:
    if config.eval_viz_samples <= 0:
        return {"enabled": False, "reason": "eval_viz_samples=0"}

    grading = eval_result.get("grading", {})
    model_key = grading.get("model_key")
    predictions_root_raw = eval_result.get("predictions_root")
    if not model_key or not predictions_root_raw:
        return {"enabled": False, "reason": "missing model key or predictions root"}
    predictions_root = Path(str(predictions_root_raw))

    split = str(config.eval_viz_split)
    if not selected_records:
        return {"enabled": False, "reason": f"no {split} samples found"}

    rec = selected_records[0]
    if rec.image_path is None:
        return {"enabled": False, "reason": "selected sample has no image path"}

    image = image_cache.get(rec.stem)
    if image is None:
        return {"enabled": False, "reason": f"image missing for sample: {rec.stem}"}

    gt = gt_cache.get(rec.gt_label_path, []) if rec.gt_label_path else []
    preds = load_prediction_labels(
        predictions_root=predictions_root,
        model_name=str(model_key),
        split=rec.split,
        stem=rec.stem,
        conf_threshold=float(config.eval_conf_threshold),
    )

    gt_only = gt_overlay_cache.get(rec.stem)
    if gt_only is None:
        gt_only = _draw_label_set(image, gt, color=(0, 255, 0), tag="GT")
        gt_overlay_cache[rec.stem] = gt_only
    panel = _draw_label_set(gt_only, preds, color=(0, 0, 255), tag="PR")
    cv2.putText(
        panel,
        f"epoch {epoch:03d}",
        (10, max(18, panel.shape[0] - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    artifact_dir = config.artifacts_root / "train_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    panel_path = artifact_dir / "prediction_vs_gt.png"
    if not cv2.imwrite(str(panel_path), panel):
        return {"enabled": False, "reason": f"failed to write visualization: {panel_path}"}

    meta = {
        "epoch": int(epoch),
        "split": str(rec.split),
        "stem": str(rec.stem),
        "gt_count": int(len(gt)),
        "pred_count": int(len(preds)),
        "panel_overlay": str(panel_path),
    }
    meta_path = artifact_dir / "prediction_vs_gt.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {
        "enabled": True,
        "artifact_dir": str(artifact_dir),
        "panel_overlay": str(panel_path),
        "meta_json": str(meta_path),
        "sample": meta,
    }


def train_detector(config: TrainConfig) -> dict[str, Any]:
    config.validate()
    _configure_torch_runtime(config)
    device = resolve_device(config.device)
    cache_mode = _resolve_cache_mode(config.cache, config.dataset_root)
    workers_effective = _resolve_workers(config)
    batch_requested = _initial_batch_for_mode(config)
    project_dir = config.project if config.project.is_absolute() else (Path.cwd() / config.project)
    project_dir = project_dir.resolve()

    if config.artifacts_root.exists():
        shutil.rmtree(config.artifacts_root)
    config.artifacts_root.mkdir(parents=True, exist_ok=True)
    save_dir_base = project_dir / config.name
    if save_dir_base.exists():
        shutil.rmtree(save_dir_base)

    data_yaml_path, names = write_data_yaml(
        dataset_root=config.dataset_root,
        output_path=config.artifacts_root / "data.yaml",
    )

    wandb_cfg = {
        "dataset_root": str(config.dataset_root),
        "data_yaml": str(data_yaml_path),
        "model": config.model,
        "epochs": config.epochs,
        "imgsz": config.imgsz,
        "batch": batch_requested,
        "device": device,
        "seed": config.seed,
        "workers": workers_effective,
        "cache": cache_mode,
        "throughput_mode": config.throughput_mode,
        "batch_mode": config.batch_mode,
        "batch_max": config.batch_max,
        "batch_utilization_target": config.batch_utilization_target,
        "oom_backoff_factor": config.oom_backoff_factor,
        "workers_auto": config.workers_auto,
        "workers_max": config.workers_max,
        "cache_requested": config.cache,
        "patience": config.patience,
        "amp": config.amp,
        "plots": config.plots,
        "tf32": config.tf32,
        "cudnn_benchmark": config.cudnn_benchmark,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "cos_lr": config.cos_lr,
        "close_mosaic": config.close_mosaic,
        "mosaic": config.mosaic,
        "mixup": config.mixup,
        "degrees": config.degrees,
        "translate": config.translate,
        "scale": config.scale,
        "shear": config.shear,
        "perspective": config.perspective,
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
        "fliplr": config.fliplr,
        "flipud": config.flipud,
        "copy_paste": config.copy_paste,
        "multi_scale": config.multi_scale,
        "freeze": config.freeze,
        "dino_root": str(config.dino_root),
        "dino_distill_warmup_epochs": config.dino_distill_warmup_epochs,
        "dino_distill_layers": list(config.dino_distill_layers),
        "dino_distill_channels": config.dino_distill_channels,
        "dino_distill_object_weight": config.dino_distill_object_weight,
        "dino_distill_background_weight": config.dino_distill_background_weight,
        "stage_a_ratio": config.stage_a_ratio,
        "stage_a_freeze": config.stage_a_freeze,
        "stage_a_distill_weight": config.stage_a_distill_weight,
        "stage_b_distill_weight": config.stage_b_distill_weight,
        "dino_viz_enabled": config.dino_viz_enabled,
        "dino_viz_mode": config.dino_viz_mode,
        "dino_viz_every_n_epochs": config.dino_viz_every_n_epochs,
        "dino_viz_max_samples": config.dino_viz_max_samples,
        "classes_count": len(names),
        "wandb_log_every_epoch": config.wandb_log_every_epoch,
        "wandb_log_system_metrics": config.wandb_log_system_metrics,
        "periodic_eval_mode": config.periodic_eval_mode,
        "periodic_eval_sparse_epochs": config.periodic_eval_sparse_epochs,
    }
    eval_viz_selected_records: list[Any] = []
    eval_viz_gt_cache: dict[Path, list[Any]] = {}
    eval_viz_image_cache: dict[str, np.ndarray] = {}
    eval_viz_gt_overlay_cache: dict[str, np.ndarray] = {}
    if config.eval_viz_samples > 0:
        split = str(config.eval_viz_split)
        records = [r for r in index_ground_truth(config.dataset_root) if r.split == split and r.image_path is not None]
        eval_viz_selected_records = sorted(records, key=lambda r: r.stem)[: int(config.eval_viz_samples)]
        for rec in eval_viz_selected_records:
            image = cv2.imread(str(rec.image_path), cv2.IMREAD_COLOR) if rec.image_path is not None else None
            if image is None:
                continue
            eval_viz_image_cache[rec.stem] = image
            if rec.gt_label_path is None:
                continue
            if rec.gt_label_path not in eval_viz_gt_cache:
                eval_viz_gt_cache[rec.gt_label_path] = load_obb_labels(
                    rec.gt_label_path, is_prediction=False, conf_threshold=0.0
                )
            eval_viz_gt_overlay_cache[rec.stem] = _draw_label_set(
                image,
                eval_viz_gt_cache.get(rec.gt_label_path, []),
                color=(0, 255, 0),
                tag="GT",
            )

    run_name = config.wandb_run_name or config.name
    wandb_run, wandb_state = init_wandb(
        enabled=config.wandb_enabled,
        mode=config.wandb_mode,
        project=config.wandb_project,
        entity=config.wandb_entity,
        run_name=run_name,
        tags=config.wandb_tags,
        notes=config.wandb_notes,
        config=wandb_cfg,
        log_system_metrics=config.wandb_log_system_metrics,
    )

    try:
        from ultralytics import YOLO

        stage_a_epochs = 0
        if config.epochs >= 2:
            stage_a_epochs = int(round(config.epochs * config.stage_a_ratio))
            stage_a_epochs = max(1, min(config.epochs - 1, stage_a_epochs))

        safe_warmup_epochs = float(config.warmup_epochs) if config.warmup_epochs is not None else 0.0
        safe_fliplr = float(config.fliplr) if config.fliplr is not None else 0.0
        safe_flipud = float(config.flipud) if config.flipud is not None else 0.0
        safe_copy_paste = float(config.copy_paste) if config.copy_paste is not None else 0.0
        periodic_eval: list[dict[str, Any]] = []
        best_geo_score = -1.0
        best_geo_weights: Path | None = None
        last_eval_epoch = -1
        logged_keys_by_step: dict[int, set[str]] = {}
        dino_viz_records: list[dict[str, Any]] = []

        def _log_step_payload(step: int, payload: dict[str, float]) -> None:
            if step <= 0 or not payload:
                return
            sent = logged_keys_by_step.setdefault(step, set())
            unique_payload = {k: v for k, v in payload.items() if k not in sent}
            if not unique_payload:
                return
            log_wandb(wandb_run, unique_payload, step=step)
            sent.update(unique_payload.keys())

        def _on_fit_epoch_end(trainer) -> None:
            nonlocal last_eval_epoch, best_geo_score, best_geo_weights

            current_epoch = int(getattr(trainer, "epoch", -1)) + 1
            if current_epoch <= 0:
                return

            if config.wandb_log_every_epoch:
                _log_step_payload(current_epoch, _extract_epoch_metrics(trainer))

            if config.dino_viz_enabled and config.dino_viz_mode != "off":
                if config.dino_viz_mode == "final_only":
                    should_save_viz = current_epoch == config.epochs
                else:
                    should_save_viz = (
                        current_epoch == config.epochs
                        or current_epoch % config.dino_viz_every_n_epochs == 0
                    )
                if should_save_viz:
                    snapshot = getattr(trainer, "_dino_viz_snapshot", None)
                    if snapshot is not None:
                        artifact_dir = config.artifacts_root / "train_artifacts"
                        tmp_dir = artifact_dir / "_dino_tmp"
                        dino_path = artifact_dir / "dino_distill.png"
                        try:
                            if tmp_dir.exists():
                                shutil.rmtree(tmp_dir, ignore_errors=True)
                            artifact_dir.mkdir(parents=True, exist_ok=True)
                            viz_result = save_dino_visualizations(
                                snapshot=snapshot,
                                output_dir=tmp_dir,
                                max_samples=1,
                            )
                            files = [Path(str(p)) for p in viz_result.get("files", [])]
                            preferred = next((p for p in files if "distill_signal_overlay" in p.name), None)
                            if preferred is None:
                                preferred = next((p for p in files if "teacher_overlay" in p.name), None)
                            if preferred is not None and preferred.exists():
                                shutil.copy2(preferred, dino_path)
                            dino_viz_records.append(
                                {
                                    "epoch": current_epoch,
                                    "artifact": str(dino_path),
                                    "source": str(preferred) if preferred is not None else None,
                                }
                            )
                        except Exception as exc:
                            dino_viz_records.append(
                                {
                                    "epoch": current_epoch,
                                    "artifact": str(dino_path),
                                    "error": str(exc),
                                }
                            )
                        finally:
                            if tmp_dir.exists():
                                shutil.rmtree(tmp_dir, ignore_errors=True)

            if not config.eval_enabled or config.periodic_eval_mode == "off":
                return
            if current_epoch == last_eval_epoch:
                return

            warmup_done = current_epoch >= int(max(1, round(config.warmup_epochs)))
            if config.periodic_eval_mode == "sparse":
                should_eval = ((warmup_done and current_epoch % config.periodic_eval_sparse_epochs == 0) or (current_epoch == config.epochs))
            else:
                should_eval = ((warmup_done and current_epoch % config.eval_interval_epochs == 0) or (current_epoch == config.epochs))
            if not should_eval:
                return

            save_dir_cb = Path(getattr(trainer, "save_dir", project_dir / config.name))
            try:
                eval_result = _run_periodic_eval(
                    config=config,
                    save_dir=save_dir_cb,
                    epoch=current_epoch,
                    device=device,
                    run_name=config.name,
                    wandb_run=wandb_run,
                )
                eval_result["visualization"] = _write_eval_visual_artifacts(
                    config=config,
                    eval_result=eval_result,
                    epoch=current_epoch,
                    selected_records=eval_viz_selected_records,
                    gt_cache=eval_viz_gt_cache,
                    image_cache=eval_viz_image_cache,
                    gt_overlay_cache=eval_viz_gt_overlay_cache,
                )
                periodic_eval.append(eval_result)
                grade = eval_result.get("grading", {}).get("aggregate", {}).get("run_grade_0_100")
                weights_last_path_raw = eval_result.get("weights_last")
                if grade is not None and weights_last_path_raw is not None:
                    weights_last_path = Path(weights_last_path_raw)
                    if weights_last_path.exists() and float(grade) >= best_geo_score:
                        best_geo_score = float(grade)
                        best_geo_weights = save_dir_cb / "weights" / "best_geo.pt"
                        best_geo_weights.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(weights_last_path, best_geo_weights)
                        _log_step_payload(current_epoch, {"eval/best_geo_score": best_geo_score})
            except Exception as exc:
                periodic_eval.append(
                    {
                        "status": "error",
                        "epoch": current_epoch,
                        "error": str(exc),
                    }
                )
                _log_step_payload(current_epoch, {"eval/status": 0.0, "eval/error_flag": 1.0})

            last_eval_epoch = current_epoch

        train_kwargs: dict[str, Any] = {
            "data": str(data_yaml_path),
            "imgsz": config.imgsz,
            "batch": batch_requested,
            "device": device,
            "project": str(project_dir),
            "seed": config.seed,
            "workers": workers_effective,
            "patience": config.patience,
            "cache": cache_mode,
            "amp": config.amp,
            "plots": config.plots,
            "exist_ok": True,
            "optimizer": config.optimizer,
            "lr0": config.lr0,
            "lrf": config.lrf,
            "weight_decay": config.weight_decay,
            "warmup_epochs": safe_warmup_epochs,
            "cos_lr": config.cos_lr,
            "close_mosaic": config.close_mosaic,
            "mosaic": config.mosaic,
            "mixup": config.mixup,
            "degrees": config.degrees,
            "translate": config.translate,
            "scale": config.scale,
            "shear": config.shear,
            "perspective": config.perspective,
            "hsv_h": config.hsv_h,
            "hsv_s": config.hsv_s,
            "hsv_v": config.hsv_v,
            "fliplr": safe_fliplr,
            "flipud": safe_flipud,
            "copy_paste": safe_copy_paste,
            "multi_scale": config.multi_scale,
            "epochs": config.epochs,
            "name": config.name,
        }

        def _make_trainer_factory():
            return partial(
                DinoOBBTrainer,
                dino_cfg=DinoDistillConfig(
                    dino_root=config.dino_root,
                    stage_a_epochs=stage_a_epochs,
                    stage_a_freeze=config.stage_a_freeze,
                    stage_a_weight=config.stage_a_distill_weight,
                    stage_b_weight=config.stage_b_distill_weight,
                    warmup_epochs=config.dino_distill_warmup_epochs,
                    student_layers=tuple(int(v) for v in config.dino_distill_layers),
                    channels=int(config.dino_distill_channels),
                    object_weight=config.dino_distill_object_weight,
                    background_weight=config.dino_distill_background_weight,
                    viz_enabled=config.dino_viz_enabled and config.dino_viz_mode != "off",
                    viz_mode=config.dino_viz_mode,
                    viz_every_n_epochs=config.dino_viz_every_n_epochs,
                    total_epochs=config.epochs,
                    viz_max_samples=config.dino_viz_max_samples,
                ),
            )
        model = YOLO(config.model)
        model_task = str(getattr(model, "task", "")).strip().lower()
        if model_task != "obb":
            raise RuntimeError(
                f"model is not an OBB model (task={model_task!r}): {config.model}. "
                "Use an OBB checkpoint."
            )
        model.add_callback("on_fit_epoch_end", _on_fit_epoch_end)
        resolved_batch = int(train_kwargs["batch"])
        batch_attempts: list[int] = [resolved_batch]
        while True:
            train_kwargs["batch"] = int(resolved_batch)
            try:
                train_result = model.train(
                    trainer=_make_trainer_factory(),
                    **train_kwargs,
                )
                break
            except RuntimeError as exc:
                if str(config.batch_mode) != "auto_max" or not _is_cuda_oom(exc) or resolved_batch <= 1:
                    raise
                next_batch = max(1, int(resolved_batch * float(config.oom_backoff_factor)))
                if next_batch >= resolved_batch:
                    next_batch = max(1, resolved_batch - 1)
                if next_batch == resolved_batch:
                    raise
                resolved_batch = next_batch
                batch_attempts.append(resolved_batch)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if save_dir_base.exists():
                    shutil.rmtree(save_dir_base, ignore_errors=True)

        save_dir = Path(getattr(train_result, "save_dir", project_dir / config.name))
        weights_dir = save_dir / "weights"
        best_weights = weights_dir / "best.pt"
        last_weights = weights_dir / "last.pt"
        if best_geo_weights is None:
            fallback_best_geo = weights_dir / "best_geo.pt"
            if best_weights.exists():
                shutil.copy2(best_weights, fallback_best_geo)
                best_geo_weights = fallback_best_geo
        selected_best = best_geo_weights if best_geo_weights is not None else best_weights
        results_csv = save_dir / "results.csv"

        metrics = _extract_last_metrics(results_csv)
        if metrics:
            final_payload = {f"final/{k.replace('/', '_')}": v for k, v in metrics.items()}
            _log_step_payload(config.epochs, final_payload)

        summary = {
            "status": "ok",
            "config": _json_safe(asdict(config)),
            "wandb": asdict(wandb_state),
            "artifacts": {
                "data_yaml": str(data_yaml_path),
                "save_dir": str(save_dir),
                "weights_best": str(selected_best),
                "weights_best_ultralytics": str(best_weights),
                "weights_best_geo": str(best_geo_weights) if best_geo_weights is not None else None,
                "weights_last": str(last_weights),
                "results_csv": str(results_csv),
            },
            "metrics": metrics,
            "resolved_device": device,
            "effective_batch": int(train_kwargs["batch"]),
            "workers_effective": int(workers_effective),
            "auto_batch_attempts": batch_attempts,
            "periodic_eval": periodic_eval,
            "dino_visualization": {
                "enabled": config.dino_viz_enabled,
                "mode": config.dino_viz_mode,
                "every_n_epochs": config.dino_viz_every_n_epochs,
                "max_samples": config.dino_viz_max_samples,
                "records": dino_viz_records,
            },
        }
    except Exception:
        raise
    finally:
        finish_wandb(wandb_run)

    if config.save_json:
        save_dir = Path(summary["artifacts"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)
        run_summary_path = save_dir / "train_summary.json"
        run_summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        artifacts_base = config.artifacts_root
        if len(config.artifacts_root.parents) >= 3:
            artifacts_base = config.artifacts_root.parents[2]
        latest_path = artifacts_base / "latest_run.json"
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        latest_payload = {
            "run_name": config.name,
            "run_root": str(config.artifacts_root),
            "save_dir": summary["artifacts"]["save_dir"],
            "weights_best": summary["artifacts"]["weights_best"],
            "weights_last": summary["artifacts"]["weights_last"],
            "data_yaml": summary["artifacts"]["data_yaml"],
            "wandb": summary["wandb"],
        }
        latest_path.write_text(json.dumps(latest_payload, indent=2), encoding="utf-8")

    return summary
