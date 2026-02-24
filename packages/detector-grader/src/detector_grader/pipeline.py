from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import shutil
from typing import Any

from detector_infer.config import InferConfig
from detector_infer.infer import run_inference

from .data import (
    image_shape,
    index_ground_truth,
    infer_model_name_from_weights,
    load_labels,
    load_prediction_labels,
    resolve_latest_weights,
    sanitize_model_name,
)
from .reporting import aggregate_scores, write_reports
from .scoring import ScoreWeights, score_sample


@dataclass(slots=True)
class GradingConfig:
    dataset_root: Path
    predictions_root: Path = Path("artifacts/models/default/runs/current/infer")
    artifacts_root: Path = Path("artifacts/models")
    reports_dir: Path | None = None
    hard_examples_dir: Path | None = None
    model: str = "latest"
    weights: Path | None = None
    run_inference: bool = True
    splits: list[str] | None = None
    imgsz: int = 640
    device: str = "auto"
    conf_threshold: float = 0.25
    infer_iou_threshold: float = 0.7
    match_oks_threshold: float = 0.5
    weights_json: Path | None = None
    max_samples: int | None = None
    seed: int = 42
    calibrate_confidence: bool = True
    calibration_candidates: list[float] | None = None
    fpr_negative_set_enabled: bool = True
    fpr_threshold: float = 0.05


def load_weights_profile(path: Path | None) -> ScoreWeights:
    defaults = ScoreWeights()
    if path is None:
        return defaults.normalized()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScoreWeights(
        oks=float(payload.get("oks", defaults.oks)),
        box_iou=float(payload.get("box_iou", defaults.box_iou)),
        fn_penalty=float(payload.get("fn_penalty", defaults.fn_penalty)),
        fp_penalty=float(payload.get("fp_penalty", defaults.fp_penalty)),
    ).normalized()


def resolve_model_source(
    *,
    model_arg: str,
    weights_arg: Path | None,
    artifacts_root: Path,
    predictions_root: Path,
) -> tuple[Path | None, str, bool]:
    if weights_arg is not None:
        if not weights_arg.exists():
            raise FileNotFoundError(f"weights not found: {weights_arg}")
        key = infer_model_name_from_weights(weights_arg)
        existing = (predictions_root / key / "labels").exists()
        return weights_arg, key, existing

    normalized = model_arg.strip()
    latest_aliases = {".", "latest", "latest-best", "best"}
    if normalized in latest_aliases:
        w = resolve_latest_weights(artifacts_root)
        key = infer_model_name_from_weights(w)
        existing = (predictions_root / key / "labels").exists()
        return w, key, existing

    key = sanitize_model_name(normalized)
    existing = (predictions_root / key / "labels").exists()
    return None, key, existing


def _split_geometry_summary(aggregate: dict[str, Any]) -> dict[str, float | None]:
    splits = aggregate.get("splits", [])
    total = 0
    oks_acc = 0.0
    err_acc = 0.0
    oks_seen = 0
    err_seen = 0
    for split in splits:
        n = int(split.get("num_samples", 0) or 0)
        if n <= 0:
            continue
        total += n
        geom = split.get("geometry", {})
        oks = geom.get("oks_mean")
        err = geom.get("keypoint_error_px_mean")
        if oks is not None:
            oks_acc += float(oks) * n
            oks_seen += n
        if err is not None:
            err_acc += float(err) * n
            err_seen += n
    return {
        "mean_oks": (oks_acc / oks_seen) if oks_seen > 0 else None,
        "mean_keypoint_error_px": (err_acc / err_seen) if err_seen > 0 else None,
    }


def run_grading(config: GradingConfig) -> dict[str, Any]:
    splits = config.splits if config.splits is not None else ["train", "val"]
    reports_dir = config.reports_dir if config.reports_dir is not None else (config.predictions_root.parent / "reports")
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
    weights_profile = load_weights_profile(config.weights_json)

    weights_path, model_key, has_existing = resolve_model_source(
        model_arg=config.model,
        weights_arg=config.weights,
        artifacts_root=config.artifacts_root,
        predictions_root=config.predictions_root,
    )

    inference_summary: dict[str, Any] | None = None
    if config.run_inference:
        if weights_path is None:
            raise RuntimeError(
                "run-inference requested but no weights resolved. "
                "Pass weights, use model latest/. , or provide a run/weights path."
            )
        infer_cfg = InferConfig(
            weights=weights_path,
            dataset_root=config.dataset_root,
            output_root=config.predictions_root,
            model_name=model_key,
            imgsz=config.imgsz,
            device=config.device,
            conf_threshold=config.conf_threshold,
            iou_threshold=config.infer_iou_threshold,
            seed=config.seed,
            splits=splits,
            save_empty=True,
        )
        inference_summary = run_inference(infer_cfg)
    else:
        if not has_existing:
            raise RuntimeError(
                f"prediction set not found for model key '{model_key}' under {config.predictions_root}. "
                "Enable run-inference or provide a valid predictions model key."
            )

    records = index_ground_truth(config.dataset_root)
    sample_rows: list[dict[str, Any]] = []

    processed = 0
    invalid_count = 0
    indexed_count = 0
    missing_image_count = 0
    missing_gt_count = 0
    by_split_counts = {s: 0 for s in splits}

    neg_total = 0
    neg_fp_total = 0

    for rec in records:
        if rec.split not in splits:
            continue
        indexed_count += 1
        if config.max_samples is not None and processed >= int(config.max_samples):
            break
        if rec.image_path is None:
            missing_image_count += 1
            continue

        shape = image_shape(rec.image_path)
        if shape is None:
            invalid_count += 1
            continue
        h, w = shape

        gt_labels = []
        if rec.gt_label_path is not None:
            gt_labels = load_labels(rec.gt_label_path, is_prediction=False, conf_threshold=0.0)
        else:
            missing_gt_count += 1

        pred_labels = load_prediction_labels(
            predictions_root=config.predictions_root,
            model_name=model_key,
            split=rec.split,
            stem=rec.stem,
            conf_threshold=config.conf_threshold,
        )

        if config.fpr_negative_set_enabled and not gt_labels:
            neg_total += 1
            if pred_labels:
                neg_fp_total += 1

        row = score_sample(
            split=rec.split,
            stem=rec.stem,
            gt_labels=gt_labels,
            pred_labels=pred_labels,
            w=w,
            h=h,
            oks_threshold=float(config.match_oks_threshold),
            weights=weights_profile,
        )
        sample_rows.append(row)
        by_split_counts[rec.split] = int(by_split_counts.get(rec.split, 0)) + 1
        processed += 1

    aggregate = aggregate_scores(sample_rows)
    reports = write_reports(
        reports_dir=reports_dir,
        hard_examples_dir=config.hard_examples_dir,
        model_name=model_key,
        config={
            "dataset_root": str(config.dataset_root),
            "splits": splits,
            "imgsz": config.imgsz,
            "device": config.device,
            "conf_threshold": config.conf_threshold,
            "infer_iou_threshold": config.infer_iou_threshold,
            "match_oks_threshold": config.match_oks_threshold,
            "max_samples": config.max_samples,
            "calibrate_confidence": config.calibrate_confidence,
            "fpr_negative_set_enabled": config.fpr_negative_set_enabled,
            "fpr_threshold": config.fpr_threshold,
        },
        sample_rows=sample_rows,
        aggregate=aggregate,
    )

    detection = aggregate.get("run_detection", {})
    geo = _split_geometry_summary(aggregate)
    fpr_negative = (float(neg_fp_total) / float(max(neg_total, 1))) if neg_total > 0 else None
    eval_like = {
        "precision": detection.get("precision_proxy"),
        "recall": detection.get("recall_proxy"),
        "miss_rate": detection.get("miss_rate_proxy"),
        "mean_oks": geo.get("mean_oks"),
        "mean_keypoint_error_px": geo.get("mean_keypoint_error_px"),
        "fpr_negative": fpr_negative,
    }

    return {
        "status": "ok",
        "model_key": model_key,
        "weights_path": str(weights_path) if weights_path is not None else None,
        "inference": inference_summary,
        "aggregate": aggregate,
        "reports": reports,
        "sample_rows": sample_rows,
        "eval_like": eval_like,
        "indexed_count": indexed_count,
        "processed_count": processed,
        "invalid_count": invalid_count,
        "missing_image_count": missing_image_count,
        "missing_gt_count": missing_gt_count,
        "split_counts": by_split_counts,
        "hallucination": {
            "negative_samples": neg_total,
            "negative_fp_samples": neg_fp_total,
            "fpr_negative": fpr_negative,
            "fpr_threshold": float(config.fpr_threshold),
            "fpr_gate_pass": None if fpr_negative is None else (fpr_negative <= float(config.fpr_threshold)),
        },
    }
