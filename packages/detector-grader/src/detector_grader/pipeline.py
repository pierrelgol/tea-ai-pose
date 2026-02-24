from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
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
    match_iou_threshold: float = 0.5
    weights_json: Path | None = None
    strict_obb: bool = True
    max_samples: int | None = None
    seed: int = 42
    calibrate_confidence: bool = True
    calibration_candidates: list[float] | None = None


@dataclass(slots=True)
class CachedSample:
    split: str
    stem: str
    w: int
    h: int
    gt_labels: list[Any]
    pred_labels_all: list[Any]


CALIBRATION_HOLDOUT_RATIO = 0.4
MIN_CALIBRATION_SAMPLES = 40
MIN_EVALUATION_SAMPLES = 40


def _split_for_calibration(
    samples: list[CachedSample],
    *,
    seed: int,
    holdout_ratio: float = CALIBRATION_HOLDOUT_RATIO,
) -> tuple[list[CachedSample], list[CachedSample]]:
    holdout_ratio = float(max(0.0, min(1.0, holdout_ratio)))
    calibration: list[CachedSample] = []
    evaluation: list[CachedSample] = []
    for s in samples:
        token = f"{seed}:{s.split}:{s.stem}".encode("utf-8")
        h = int.from_bytes(hashlib.sha1(token).digest()[:8], "big")
        u = (h % 10_000) / 10_000.0
        if u < holdout_ratio:
            calibration.append(s)
        else:
            evaluation.append(s)
    return calibration, evaluation


def load_weights_profile(path: Path | None) -> ScoreWeights:
    defaults = ScoreWeights()
    if path is None:
        return defaults.normalized()
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ScoreWeights(
        iou=float(payload.get("iou", defaults.iou)),
        corner=float(payload.get("corner", defaults.corner)),
        angle=float(payload.get("angle", defaults.angle)),
        center=float(payload.get("center", defaults.center)),
        shape=float(payload.get("shape", defaults.shape)),
        fn_penalty=float(payload.get("fn_penalty", defaults.fn_penalty)),
        fp_penalty=float(payload.get("fp_penalty", defaults.fp_penalty)),
        containment_miss_penalty=float(payload.get("containment_miss_penalty", defaults.containment_miss_penalty)),
        containment_outside_penalty=float(payload.get("containment_outside_penalty", defaults.containment_outside_penalty)),
        tau_corner_px=float(payload.get("tau_corner_px", defaults.tau_corner_px)),
        tau_center_px=float(payload.get("tau_center_px", defaults.tau_center_px)),
        iou_gamma=float(payload.get("iou_gamma", defaults.iou_gamma)),
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
    iou_acc = 0.0
    drift_acc = 0.0
    iou_seen = 0
    drift_seen = 0
    for split in splits:
        n = int(split.get("num_samples", 0) or 0)
        if n <= 0:
            continue
        total += n
        geom = split.get("geometry", {})
        iou = geom.get("iou_mean")
        drift = geom.get("center_error_px_mean")
        if iou is not None:
            iou_acc += float(iou) * n
            iou_seen += n
        if drift is not None:
            drift_acc += float(drift) * n
            drift_seen += n
    return {
        "mean_iou": (iou_acc / iou_seen) if iou_seen > 0 else None,
        "mean_center_drift_px": (drift_acc / drift_seen) if drift_seen > 0 else None,
    }


def _filter_predictions(preds: list[Any], per_class_thresholds: dict[int, float], default_threshold: float) -> list[Any]:
    out: list[Any] = []
    for p in preds:
        thr = float(per_class_thresholds.get(int(p.class_id), default_threshold))
        if float(p.confidence) >= thr:
            out.append(p)
    return out


def _score_cached_samples(
    samples: list[CachedSample],
    *,
    per_class_thresholds: dict[int, float],
    default_threshold: float,
    match_iou_threshold: float,
    weights_profile: ScoreWeights,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        preds = _filter_predictions(sample.pred_labels_all, per_class_thresholds, default_threshold)
        rows.append(
            score_sample(
                split=sample.split,
                stem=sample.stem,
                gt_labels=sample.gt_labels,
                pred_labels=preds,
                w=sample.w,
                h=sample.h,
                iou_threshold=match_iou_threshold,
                weights=weights_profile,
            )
        )
    return rows


def _calibrate_per_class_thresholds(
    samples: list[CachedSample],
    *,
    base_threshold: float,
    candidates: list[float],
    match_iou_threshold: float,
    weights_profile: ScoreWeights,
) -> dict[int, float]:
    class_ids: set[int] = set()
    for s in samples:
        class_ids.update(int(g.class_id) for g in s.gt_labels)
        class_ids.update(int(p.class_id) for p in s.pred_labels_all)
    if not class_ids:
        return {}

    thresholds: dict[int, float] = {}
    for class_id in sorted(class_ids):
        # Only samples containing predictions for this class are affected by threshold tweaks.
        affected_samples: list[CachedSample] = []
        unaffected_samples: list[CachedSample] = []
        for s in samples:
            if any(int(p.class_id) == class_id for p in s.pred_labels_all):
                affected_samples.append(s)
            else:
                unaffected_samples.append(s)

        base_trial = dict(thresholds)
        unaffected_rows = _score_cached_samples(
            unaffected_samples,
            per_class_thresholds=base_trial,
            default_threshold=base_threshold,
            match_iou_threshold=match_iou_threshold,
            weights_profile=weights_profile,
        )

        affected_static_preds: list[tuple[CachedSample, list[Any], list[Any]]] = []
        for s in affected_samples:
            static_preds: list[Any] = []
            class_preds: list[Any] = []
            for p in s.pred_labels_all:
                pid = int(p.class_id)
                if pid == class_id:
                    class_preds.append(p)
                    continue
                thr = float(base_trial.get(pid, base_threshold))
                if float(p.confidence) >= thr:
                    static_preds.append(p)
            affected_static_preds.append((s, static_preds, class_preds))

        best_t = float(base_threshold)
        best_grade = -1.0
        for t in candidates:
            t_val = float(t)
            affected_rows: list[dict[str, Any]] = []
            for s, static_preds, class_preds in affected_static_preds:
                filtered = list(static_preds)
                filtered.extend([p for p in class_preds if float(p.confidence) >= t_val])
                affected_rows.append(
                    score_sample(
                        split=s.split,
                        stem=s.stem,
                        gt_labels=s.gt_labels,
                        pred_labels=filtered,
                        w=s.w,
                        h=s.h,
                        iou_threshold=match_iou_threshold,
                        weights=weights_profile,
                    )
                )
            grade = float(aggregate_scores(unaffected_rows + affected_rows).get("run_grade_0_100", 0.0))
            if grade > best_grade:
                best_grade = grade
                best_t = t_val
        thresholds[class_id] = best_t
    return thresholds


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
    cached_samples: list[CachedSample] = []
    invalid_count = 0
    indexed_count = 0
    missing_image_count = 0
    missing_gt_count = 0
    missing_pred_file_count = 0
    by_split_indexed: dict[str, int] = {s: 0 for s in splits}
    by_split_invalid: dict[str, int] = {s: 0 for s in splits}
    by_split_missing_image: dict[str, int] = {s: 0 for s in splits}
    by_split_missing_gt: dict[str, int] = {s: 0 for s in splits}
    by_split_missing_pred_file: dict[str, int] = {s: 0 for s in splits}

    split_set = set(splits)
    for rec in records:
        if rec.split not in split_set:
            continue
        if config.max_samples is not None and len(sample_rows) >= config.max_samples:
            break
        if rec.image_path is None:
            missing_image_count += 1
            by_split_missing_image[rec.split] = by_split_missing_image.get(rec.split, 0) + 1
            continue
        if rec.gt_label_path is None:
            missing_gt_count += 1
            by_split_missing_gt[rec.split] = by_split_missing_gt.get(rec.split, 0) + 1
            continue
        shape = image_shape(rec.image_path)
        if shape is None:
            missing_image_count += 1
            by_split_missing_image[rec.split] = by_split_missing_image.get(rec.split, 0) + 1
            continue
        h, w = shape
        indexed_count += 1
        by_split_indexed[rec.split] = by_split_indexed.get(rec.split, 0) + 1

        pred_file = config.predictions_root / model_key / "labels" / rec.split / f"{rec.stem}.txt"
        if not pred_file.exists():
            missing_pred_file_count += 1
            by_split_missing_pred_file[rec.split] = by_split_missing_pred_file.get(rec.split, 0) + 1

        try:
            gt_labels = load_labels(rec.gt_label_path, is_prediction=False, conf_threshold=0.0)
            pred_labels_all = load_prediction_labels(
                predictions_root=config.predictions_root,
                model_name=model_key,
                split=rec.split,
                stem=rec.stem,
                conf_threshold=0.0,
            )
        except Exception:
            if config.strict_obb:
                raise
            invalid_count += 1
            by_split_invalid[rec.split] = by_split_invalid.get(rec.split, 0) + 1
            continue
        cached_samples.append(
            CachedSample(
                split=rec.split,
                stem=rec.stem,
                w=w,
                h=h,
                gt_labels=gt_labels,
                pred_labels_all=pred_labels_all,
            )
        )

    calibration_candidates = (
        [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]
        if config.calibration_candidates is None
        else [float(x) for x in config.calibration_candidates]
    )
    calibration_samples: list[CachedSample] = []
    evaluation_samples: list[CachedSample] = list(cached_samples)
    per_class_thresholds: dict[int, float] = {}
    calibration_applied = False
    if config.calibrate_confidence and cached_samples:
        calibration_samples, evaluation_samples = _split_for_calibration(
            cached_samples,
            seed=config.seed,
        )
        if (
            len(calibration_samples) >= MIN_CALIBRATION_SAMPLES
            and len(evaluation_samples) >= MIN_EVALUATION_SAMPLES
        ):
            per_class_thresholds = _calibrate_per_class_thresholds(
                calibration_samples,
                base_threshold=float(config.conf_threshold),
                candidates=calibration_candidates,
                match_iou_threshold=config.match_iou_threshold,
                weights_profile=weights_profile,
            )
            calibration_applied = True
        else:
            calibration_samples = []
            evaluation_samples = list(cached_samples)

    sample_rows = _score_cached_samples(
        evaluation_samples,
        per_class_thresholds=per_class_thresholds,
        default_threshold=float(config.conf_threshold),
        match_iou_threshold=config.match_iou_threshold,
        weights_profile=weights_profile,
    )
    aggregate = aggregate_scores(sample_rows)
    aggregate["invalid_samples_skipped"] = invalid_count
    aggregate["calibration"] = {
        "enabled": bool(config.calibrate_confidence),
        "applied": calibration_applied,
        "holdout_ratio": CALIBRATION_HOLDOUT_RATIO,
        "calibration_samples": len(calibration_samples),
        "evaluation_samples": len(evaluation_samples),
    }
    aggregate["data_quality"] = {
        "indexed_samples": indexed_count,
        "missing_image_samples": missing_image_count,
        "missing_gt_label_samples": missing_gt_count,
        "missing_prediction_file_samples": missing_pred_file_count,
        "invalid_samples_skipped": invalid_count,
        "by_split": {
            split: {
                "indexed_samples": by_split_indexed.get(split, 0),
                "missing_image_samples": by_split_missing_image.get(split, 0),
                "missing_gt_label_samples": by_split_missing_gt.get(split, 0),
                "missing_prediction_file_samples": by_split_missing_pred_file.get(split, 0),
                "invalid_samples_skipped": by_split_invalid.get(split, 0),
            }
            for split in splits
        },
    }

    run_detection = aggregate.get("run_detection", {})
    eval_like = {
        "precision": run_detection.get("precision_proxy"),
        "recall": run_detection.get("recall_proxy"),
        "miss_rate": run_detection.get("miss_rate_proxy"),
        **_split_geometry_summary(aggregate),
    }

    report_config = {
        "dataset_root": str(config.dataset_root),
        "predictions_root": str(config.predictions_root),
        "model_arg": config.model,
        "resolved_model_key": model_key,
        "weights_path": str(weights_path) if weights_path is not None else None,
        "run_inference": config.run_inference,
        "splits": splits,
        "conf_threshold": config.conf_threshold,
        "calibrate_confidence": config.calibrate_confidence,
        "calibration_applied": calibration_applied,
        "calibration_holdout_ratio": CALIBRATION_HOLDOUT_RATIO,
        "calibration_samples": len(calibration_samples),
        "evaluation_samples": len(evaluation_samples),
        "calibration_candidates": calibration_candidates,
        "calibrated_per_class_thresholds": {str(k): float(v) for k, v in per_class_thresholds.items()},
        "infer_iou_threshold": config.infer_iou_threshold,
        "match_iou_threshold": config.match_iou_threshold,
        "weights_profile": {
            "iou": weights_profile.iou,
            "corner": weights_profile.corner,
            "angle": weights_profile.angle,
            "center": weights_profile.center,
            "shape": weights_profile.shape,
            "fn_penalty": weights_profile.fn_penalty,
            "fp_penalty": weights_profile.fp_penalty,
            "containment_miss_penalty": weights_profile.containment_miss_penalty,
            "containment_outside_penalty": weights_profile.containment_outside_penalty,
            "tau_corner_px": weights_profile.tau_corner_px,
            "tau_center_px": weights_profile.tau_center_px,
            "iou_gamma": weights_profile.iou_gamma,
        },
    }

    out = write_reports(
        reports_dir=reports_dir,
        hard_examples_dir=config.hard_examples_dir,
        model_name=model_key,
        config=report_config,
        sample_rows=sample_rows,
        aggregate=aggregate,
    )

    return {
        "dataset_root": str(config.dataset_root),
        "predictions_root": str(config.predictions_root),
        "reports_dir": str(reports_dir),
        "model_key": model_key,
        "weights_path": str(weights_path) if weights_path is not None else None,
        "inference": inference_summary,
        "aggregate": aggregate,
        "eval_like": eval_like,
        "reports": out,
    }
