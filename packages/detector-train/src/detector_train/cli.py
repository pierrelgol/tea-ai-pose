from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import re
import subprocess

from pipeline_config import build_layout, load_pipeline_config
from pipeline_runtime_utils import resolve_device

from .config import TrainConfig
from .trainer import train_detector

HF_DEFAULT_ALIAS = "hf-openvision-yolo26-n-obb"


def _resolve_model_arg(model_arg: str) -> str:
    if model_arg != HF_DEFAULT_ALIAS:
        return model_arg
    try:
        from huggingface_hub import hf_hub_download

        return str(
            hf_hub_download(
                repo_id="openvision/yolo26-n-obb",
                filename="model.pt",
            )
        )
    except Exception as exc:
        raise RuntimeError(
            "failed to resolve default HF OBB model; set run.model to a local OBB .pt path"
        ) from exc


def _current_gpu_signature(resolved_device: str) -> str | None:
    if resolved_device in {"cpu", "mps"}:
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        idx = 0
        rd = resolved_device.strip().lower()
        if rd.isdigit():
            idx = int(rd)
        elif rd.startswith("cuda:"):
            idx = int(rd.split(":", 1)[1])
        props = torch.cuda.get_device_properties(idx)
        driver = "unknown"
        try:
            proc = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                    "--id",
                    str(idx),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            lines = [x.strip() for x in proc.stdout.splitlines() if x.strip()]
            if lines:
                driver = lines[0]
        except Exception:
            pass
        raw = f"{props.name}|{int(props.total_memory // (1024 * 1024))}|{props.major}.{props.minor}|{driver}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    except Exception:
        return None


def _slug_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", str(value).strip().lower()).strip("-")
    return token or "na"


def _resolve_wandb_run_name(*, shared, run_id: str, model_key: str, dataset_name: str) -> str:
    tc = shared.train if isinstance(shared.train, dict) else {}
    configured = tc.get("wandb_run_name")
    if isinstance(configured, str) and configured.strip() and configured.strip().lower() != "auto":
        return configured.strip()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    seed = int(shared.run.get("seed", 0)) if isinstance(shared.run, dict) else 0
    epochs = int(tc.get("epochs", 0))
    name = (
        f"tea-{_slug_token(dataset_name)}-{_slug_token(model_key)}-"
        f"{_slug_token(run_id)}-e{epochs}-s{seed}-{ts}"
    )
    return name[:128]


def _enforce_tuner_lock(shared) -> None:
    tuner_cfg = shared.tuner if isinstance(shared.tuner, dict) else {}
    if not bool(tuner_cfg.get("enabled", True)):
        return
    resolved = resolve_device(str(shared.train.get("device", "auto")))
    if resolved in {"cpu", "mps"}:
        return
    tuned_sig = shared.train.get("tuned_gpu_signature")
    if not isinstance(tuned_sig, str) or not tuned_sig.strip():
        raise RuntimeError(
            "missing train.tuned_gpu_signature for current configuration; run `just tune-gpu` before training"
        )
    current_sig = _current_gpu_signature(resolved)
    if current_sig is None:
        raise RuntimeError("failed to detect current GPU signature; run `just tune-gpu` after fixing CUDA visibility")
    if current_sig != tuned_sig:
        raise RuntimeError(
            f"tuned GPU signature mismatch (config={tuned_sig}, current={current_sig}); run `just tune-gpu` before training"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train detector model")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)
    _enforce_tuner_lock(shared)

    dataset_name = str(shared.dataset.get("name") or shared.run["dataset"])
    dataset_root = shared.paths["dataset_root"] / str(shared.dataset.get("augmented_subdir", "augmented")) / dataset_name

    run_id = str(shared.run["run_id"])
    model_key = str(shared.run["model_key"])
    layout = build_layout(
        artifacts_root=shared.paths["artifacts_root"],
        model_key=model_key,
        run_id=run_id,
    )
    model_path = _resolve_model_arg(str(shared.run["model"]))

    tc = shared.train
    config = TrainConfig(
        dataset_root=dataset_root,
        artifacts_root=layout.run_root,
        project=layout.train_root / "ultralytics",
        model=model_path,
        name=run_id,
        seed=int(shared.run["seed"]),
        device=str(tc.get("device", "auto")),
        epochs=int(tc.get("epochs", 128)),
        imgsz=int(tc.get("imgsz", 512)),
        batch=int(tc.get("batch", 16)),
        batch_mode=str(tc.get("batch_mode", "auto_max")),
        batch_max=int(tc.get("batch_max", 64)),
        batch_utilization_target=float(tc.get("batch_utilization_target", 0.92)),
        oom_backoff_factor=float(tc.get("oom_backoff_factor", 0.85)),
        workers=int(tc.get("workers", 16)),
        workers_auto=bool(tc.get("workers_auto", True)),
        workers_max=int(tc.get("workers_max", 16)),
        patience=int(tc.get("patience", 30)),
        cache=str(tc.get("cache", "auto")),
        throughput_mode=str(tc.get("throughput_mode", "balanced")),
        amp=bool(tc.get("amp", True)),
        plots=bool(tc.get("plots", True)),
        tf32=bool(tc.get("tf32", True)),
        cudnn_benchmark=bool(tc.get("cudnn_benchmark", True)),
        optimizer=str(tc.get("optimizer", "AdamW")),
        lr0=float(tc.get("lr0", 0.0012)),
        lrf=float(tc.get("lrf", 0.01)),
        weight_decay=float(tc.get("weight_decay", 0.0006)),
        warmup_epochs=float(tc.get("warmup_epochs", 6.0)),
        cos_lr=bool(tc.get("cos_lr", True)),
        close_mosaic=int(tc.get("close_mosaic", 12)),
        mosaic=float(tc.get("mosaic", 0.5)),
        mixup=float(tc.get("mixup", 0.03)),
        degrees=float(tc.get("degrees", 1.0)),
        translate=float(tc.get("translate", 0.035)),
        scale=float(tc.get("scale", 0.35)),
        shear=float(tc.get("shear", 0.0)),
        perspective=float(tc.get("perspective", 0.0)),
        hsv_h=float(tc.get("hsv_h", 0.01)),
        hsv_s=float(tc.get("hsv_s", 0.30)),
        hsv_v=float(tc.get("hsv_v", 0.22)),
        fliplr=float(tc.get("fliplr", 0.5)),
        flipud=float(tc.get("flipud", 0.0)),
        copy_paste=float(tc.get("copy_paste", 0.0)),
        multi_scale=bool(tc.get("multi_scale", False)),
        freeze=None if tc.get("freeze") is None else int(tc.get("freeze")),
        dino_root=Path(str(tc.get("dino_root", "dinov3"))),
        dino_distill_warmup_epochs=int(tc.get("dino_distill_warmup_epochs", 5)),
        dino_distill_layers=tuple(int(v) for v in tc.get("dino_distill_layers", [19])),
        dino_distill_channels=int(tc.get("dino_distill_channels", 32)),
        dino_distill_object_weight=float(tc.get("dino_distill_object_weight", 1.15)),
        dino_distill_background_weight=float(tc.get("dino_distill_background_weight", 0.15)),
        stage_a_ratio=float(tc.get("stage_a_ratio", 0.30)),
        stage_a_freeze=int(tc.get("stage_a_freeze", 10)),
        stage_a_distill_weight=float(tc.get("stage_a_distill_weight", 0.25)),
        stage_b_distill_weight=float(tc.get("stage_b_distill_weight", 0.08)),
        dino_viz_enabled=bool(tc.get("dino_viz_enabled", True)),
        dino_viz_mode=str(tc.get("dino_viz_mode", "final_only")),
        dino_viz_every_n_epochs=int(tc.get("dino_viz_every_n_epochs", 5)),
        dino_viz_max_samples=int(tc.get("dino_viz_max_samples", 4)),
        wandb_enabled=bool(tc.get("wandb_enabled", True)),
        wandb_project=str(tc.get("wandb_project", "tea-ai-detector")),
        wandb_entity=tc.get("wandb_entity"),
        wandb_run_name=_resolve_wandb_run_name(
            shared=shared,
            run_id=run_id,
            model_key=model_key,
            dataset_name=dataset_name,
        ),
        wandb_tags=list(tc.get("wandb_tags", [])),
        wandb_notes=tc.get("wandb_notes"),
        wandb_mode=str(tc.get("wandb_mode", "auto")),
        wandb_log_system_metrics=bool(tc.get("wandb_log_system_metrics", False)),
        wandb_log_every_epoch=bool(tc.get("wandb_log_every_epoch", True)),
        eval_enabled=bool(tc.get("eval_enabled", True)),
        periodic_eval_mode=str(tc.get("periodic_eval_mode", "interval")),
        periodic_eval_sparse_epochs=int(tc.get("periodic_eval_sparse_epochs", 10)),
        eval_interval_epochs=int(tc.get("eval_interval_epochs", 2)),
        eval_iou_threshold=float(tc.get("eval_iou_threshold", 0.75)),
        eval_conf_threshold=float(tc.get("eval_conf_threshold", 0.90)),
        eval_viz_samples=int(tc.get("eval_viz_samples", 8)),
        eval_viz_split=str(tc.get("eval_viz_split", "val")),
    )

    summary = train_detector(config)
    print(f"status: {summary['status']}")
    if summary["status"] == "ok":
        print(f"run_root: {layout.run_root}")
        print(f"run_dir: {summary['artifacts']['save_dir']}")
        print(f"best_weights: {summary['artifacts']['weights_best']}")
        print(f"wandb_mode: {summary['wandb']['mode_used']}")
        print(f"wandb_run_name: {config.wandb_run_name}")
        if summary["wandb"].get("error"):
            print(f"wandb_note: {summary['wandb']['error']}")


if __name__ == "__main__":
    main()
