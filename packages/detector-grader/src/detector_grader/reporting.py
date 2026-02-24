from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import sanitize_model_name


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _safe_div(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return float(num / den)


def _percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    vals_sorted = sorted(vals)
    if len(vals_sorted) == 1:
        return float(vals_sorted[0])
    idx = (len(vals_sorted) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(vals_sorted) - 1)
    frac = idx - lo
    return float(vals_sorted[lo] * (1.0 - frac) + vals_sorted[hi] * frac)


def _collect(rows: list[dict[str, Any]], key: str) -> list[float]:
    out: list[float] = []
    for row in rows:
        cur: Any = row
        for part in key.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if cur is not None:
            out.append(float(cur))
    return out


def _split_summary(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(r["final_score_0_100"]) for r in rows]
    weights = [max(1, int(r["num_gt"])) for r in rows]
    wsum = float(sum(weights))
    wavg = float(sum(s * w for s, w in zip(scores, weights)) / max(wsum, 1.0))

    gt_total = int(sum(int(r["num_gt"]) for r in rows))
    pred_total = int(sum(int(r["num_pred"]) for r in rows))
    matched_total = int(sum(int(r["num_matches"]) for r in rows))
    fn_total = int(sum(int(r["num_fn"]) for r in rows))
    fp_total = int(sum(int(r["num_fp"]) for r in rows))
    class_mismatch_total = int(sum(int(r.get("num_class_mismatch", 0)) for r in rows))

    no_gt_samples = int(sum(1 for r in rows if int(r["num_gt"]) == 0))
    no_pred_samples = int(sum(1 for r in rows if int(r["num_pred"]) == 0))
    no_match_samples = int(sum(1 for r in rows if int(r["num_matches"]) == 0))

    detection = {
        "precision_proxy": _safe_div(matched_total, pred_total),
        "recall_proxy": _safe_div(matched_total, gt_total),
        "miss_rate_proxy": None,
        "false_discovery_rate_proxy": _safe_div(fp_total, pred_total),
        "fn_per_image": _safe_div(fn_total, len(rows)),
        "fp_per_image": _safe_div(fp_total, len(rows)),
        "gt_match_coverage": _safe_div(matched_total, gt_total),
        "pred_match_coverage": _safe_div(matched_total, pred_total),
        "class_match_rate": _safe_div(matched_total - class_mismatch_total, matched_total),
    }
    if detection["recall_proxy"] is not None:
        detection["miss_rate_proxy"] = float(1.0 - detection["recall_proxy"])

    score_distribution = {
        "mean": _mean(scores),
        "median": _percentile(scores, 0.50),
        "p10": _percentile(scores, 0.10),
        "p25": _percentile(scores, 0.25),
        "p75": _percentile(scores, 0.75),
        "p90": _percentile(scores, 0.90),
        "p95": _percentile(scores, 0.95),
        "min": min(scores) if scores else None,
        "max": max(scores) if scores else None,
    }

    ious = _collect(rows, "diagnostics.iou_mean")
    corner_err = _collect(rows, "diagnostics.corner_error_px_mean")
    angle_err = _collect(rows, "diagnostics.angle_error_deg_mean")
    center_err = _collect(rows, "diagnostics.center_error_px_mean")
    center_norm_err = _collect(rows, "diagnostics.center_error_norm_mean")
    area_ratio = _collect(rows, "diagnostics.area_ratio_mean")
    abs_log_area_ratio = _collect(rows, "diagnostics.abs_log_area_ratio_mean")
    edge_rel_error = _collect(rows, "diagnostics.edge_rel_error_mean")
    gt_area_missed_ratio = _collect(rows, "diagnostics.gt_area_missed_ratio_mean")
    pred_outside_ratio = _collect(rows, "diagnostics.pred_outside_ratio_mean")
    conf = _collect(rows, "diagnostics.confidence_mean")
    angle_le5 = _collect(rows, "diagnostics.angle_le_5_rate")
    angle_le10 = _collect(rows, "diagnostics.angle_le_10_rate")
    iou50 = _collect(rows, "diagnostics.iou_ge_50_rate")
    iou75 = _collect(rows, "diagnostics.iou_ge_75_rate")

    geometry = {
        "iou_mean": _mean(ious),
        "iou_median": _percentile(ious, 0.50),
        "iou_p90": _percentile(ious, 0.90),
        "iou_p95": _percentile(ious, 0.95),
        "corner_error_px_mean": _mean(corner_err),
        "corner_error_px_median": _percentile(corner_err, 0.50),
        "corner_error_px_p90": _percentile(corner_err, 0.90),
        "angle_error_deg_mean": _mean(angle_err),
        "angle_error_deg_median": _percentile(angle_err, 0.50),
        "angle_error_deg_p90": _percentile(angle_err, 0.90),
        "center_error_px_mean": _mean(center_err),
        "center_error_px_median": _percentile(center_err, 0.50),
        "center_error_px_p90": _percentile(center_err, 0.90),
        "center_error_norm_mean": _mean(center_norm_err),
        "area_ratio_mean": _mean(area_ratio),
        "area_ratio_median": _percentile(area_ratio, 0.50),
        "abs_log_area_ratio_mean": _mean(abs_log_area_ratio),
        "edge_rel_error_mean": _mean(edge_rel_error),
        "gt_area_missed_ratio_mean": _mean(gt_area_missed_ratio),
        "gt_area_missed_ratio_p90": _percentile(gt_area_missed_ratio, 0.90),
        "pred_outside_ratio_mean": _mean(pred_outside_ratio),
        "orientation_within_5deg_rate": _mean(angle_le5),
        "orientation_within_10deg_rate": _mean(angle_le10),
        "iou_ge_50_rate": _mean(iou50),
        "iou_ge_75_rate": _mean(iou75),
        "pred_confidence_mean": _mean(conf),
        "pred_confidence_median": _percentile(conf, 0.50),
    }

    return {
        "split": split,
        "num_samples": len(rows),
        "grade_0_100": wavg,
        "score_distribution_0_100": score_distribution,
        "counts": {
            "gt_total": gt_total,
            "pred_total": pred_total,
            "matched_total": matched_total,
            "fn_total": fn_total,
            "fp_total": fp_total,
            "class_mismatch_total": class_mismatch_total,
            "no_gt_samples": no_gt_samples,
            "no_pred_samples": no_pred_samples,
            "no_match_samples": no_match_samples,
        },
        "detection": detection,
        "geometry": geometry,
        "mean_num_fn": _mean([float(r["num_fn"]) for r in rows]),
        "mean_num_fp": _mean([float(r["num_fp"]) for r in rows]),
        "mean_penalty_fn": _mean([float(r["penalty_fn"]) for r in rows]),
        "mean_penalty_fp": _mean([float(r["penalty_fp"]) for r in rows]),
        "mean_penalty_containment": _mean([float(r["penalty_containment"]) for r in rows]),
        "mean_match_iou": _mean([float(r["match_iou_mean"]) for r in rows if r["match_iou_mean"] is not None]),
        "components_mean": {
            "iou": _mean(
                [float(r["components_mean"]["iou_score"]) for r in rows if r["components_mean"]["iou_score"] is not None]
            ),
            "corner": _mean(
                [float(r["components_mean"]["corner_score"]) for r in rows if r["components_mean"]["corner_score"] is not None]
            ),
            "angle": _mean(
                [float(r["components_mean"]["angle_score"]) for r in rows if r["components_mean"]["angle_score"] is not None]
            ),
            "center": _mean(
                [float(r["components_mean"]["center_score"]) for r in rows if r["components_mean"]["center_score"] is not None]
            ),
            "shape": _mean(
                [float(r["components_mean"]["shape_score"]) for r in rows if r["components_mean"]["shape_score"] is not None]
            ),
        },
    }


def aggregate_scores(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        by_split.setdefault(row["split"], []).append(row)

    split_rows: list[dict[str, Any]] = []
    run_vals: list[float] = []
    run_weight = 0.0
    for split, rows in sorted(by_split.items()):
        split_summary = _split_summary(split, rows)
        split_rows.append(split_summary)

        weights = [max(1, int(r["num_gt"])) for r in rows]
        wsum = float(sum(weights))
        wavg = float(split_summary["grade_0_100"])
        run_vals.append(wavg * wsum)
        run_weight += wsum

    run_grade = float(sum(run_vals) / max(run_weight, 1.0)) if split_rows else 0.0
    total_samples = len(samples)
    total_gt = int(sum(s["counts"]["gt_total"] for s in split_rows))
    total_pred = int(sum(s["counts"]["pred_total"] for s in split_rows))
    total_matched = int(sum(s["counts"]["matched_total"] for s in split_rows))
    total_fn = int(sum(s["counts"]["fn_total"] for s in split_rows))
    total_fp = int(sum(s["counts"]["fp_total"] for s in split_rows))
    total_mismatch = int(sum(s["counts"]["class_mismatch_total"] for s in split_rows))

    return {
        "run_grade_0_100": run_grade,
        "splits": split_rows,
        "num_samples_scored": total_samples,
        "run_detection": {
            "gt_total": total_gt,
            "pred_total": total_pred,
            "matched_total": total_matched,
            "fn_total": total_fn,
            "fp_total": total_fp,
            "class_mismatch_total": total_mismatch,
            "precision_proxy": _safe_div(total_matched, total_pred),
            "recall_proxy": _safe_div(total_matched, total_gt),
            "miss_rate_proxy": None if total_gt <= 0 else float(1.0 - (total_matched / max(total_gt, 1))),
            "false_discovery_rate_proxy": _safe_div(total_fp, total_pred),
            "class_match_rate": _safe_div(total_matched - total_mismatch, total_matched),
        },
    }


def write_reports(
    reports_dir: Path,
    hard_examples_dir: Path | None,
    model_name: str,
    config: dict[str, Any],
    sample_rows: list[dict[str, Any]],
    aggregate: dict[str, Any],
) -> dict[str, str]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_key = sanitize_model_name(model_name)
    summary_json = reports_dir / f"grade_report_{model_key}.json"
    sample_jsonl = reports_dir / f"grade_samples_{model_key}.jsonl"
    summary_md = reports_dir / f"grade_summary_{model_key}.md"

    payload = {
        "model_name": model_name,
        "model_key": model_key,
        "config": config,
        "aggregate": aggregate,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with sample_jsonl.open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    hard_rows = sorted(
        sample_rows,
        key=lambda r: (
            float(r.get("final_score_0_100", 0.0)),
            -float((r.get("diagnostics", {}) or {}).get("angle_error_deg_mean") or 0.0),
            float((r.get("diagnostics", {}) or {}).get("iou_mean") or 1.0),
            -int(r.get("num_class_mismatch", 0)),
        ),
    )
    hard_top = hard_rows[:200]
    hard_payload = [
        {
            "split": str(r.get("split")),
            "stem": str(r.get("stem")),
            "final_score_0_100": float(r.get("final_score_0_100", 0.0)),
            "angle_error_deg_mean": float((r.get("diagnostics", {}) or {}).get("angle_error_deg_mean") or 0.0),
            "iou_mean": float((r.get("diagnostics", {}) or {}).get("iou_mean") or 0.0),
            "num_class_mismatch": int(r.get("num_class_mismatch", 0)),
            "gt_class_ids": [int(x) for x in (r.get("gt_class_ids") or [])],
            "pred_class_ids": [int(x) for x in (r.get("pred_class_ids") or [])],
            "hard_class_ids": [int(x) for x in (r.get("hard_class_ids") or [])],
        }
        for r in hard_top
    ]
    hard_examples_dir = hard_examples_dir if hard_examples_dir is not None else (reports_dir / "hard_examples")
    hard_examples_dir.mkdir(parents=True, exist_ok=True)
    hard_examples_model = hard_examples_dir / f"hard_examples_{model_key}.jsonl"
    hard_examples_latest = hard_examples_dir / "latest.jsonl"
    with hard_examples_model.open("w", encoding="utf-8") as f:
        for row in hard_payload:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    hard_examples_latest.write_text(hard_examples_model.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [
        "# Detector Grade Summary",
        "",
        f"- model: `{model_name}`",
        f"- run grade (0-100): **{aggregate['run_grade_0_100']:.4f}**",
        f"- samples scored: {aggregate['num_samples_scored']}",
        f"- matched objects: {aggregate['run_detection']['matched_total']}",
        f"- gt objects: {aggregate['run_detection']['gt_total']}",
        f"- pred objects: {aggregate['run_detection']['pred_total']}",
        "",
        "## Split Grades",
    ]
    for s in aggregate["splits"]:
        lines.append(
            f"- {s['split']}: {s['grade_0_100']:.4f} ({s['num_samples']} samples), "
            f"iou_mean={s['geometry']['iou_mean']}, "
            f"angle_err_mean_deg={s['geometry']['angle_error_deg_mean']}, "
            f"center_err_mean_px={s['geometry']['center_error_px_mean']}, "
            f"gt_area_missed_mean={s['geometry']['gt_area_missed_ratio_mean']}, "
            f"containment_penalty_mean={s['mean_penalty_containment']}"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary_json": str(summary_json),
        "sample_jsonl": str(sample_jsonl),
        "summary_md": str(summary_md),
        "hard_examples_jsonl": str(hard_examples_model),
        "hard_examples_latest_jsonl": str(hard_examples_latest),
    }
