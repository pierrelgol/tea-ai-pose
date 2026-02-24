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


def aggregate_scores(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for row in samples:
        by_split.setdefault(str(row["split"]), []).append(row)

    split_rows: list[dict[str, Any]] = []
    total_weight = 0.0
    weighted_sum = 0.0
    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_fn = 0
    total_fp = 0

    for split, rows in sorted(by_split.items()):
        grades = [float(r["final_score_0_100"]) for r in rows]
        gts = [max(1, int(r["num_gt"])) for r in rows]
        wavg = float(sum(g * w for g, w in zip(grades, gts)) / max(sum(gts), 1))

        split_gt = int(sum(int(r["num_gt"]) for r in rows))
        split_pred = int(sum(int(r["num_pred"]) for r in rows))
        split_matched = int(sum(int(r["num_matches"]) for r in rows))
        split_fn = int(sum(int(r["num_fn"]) for r in rows))
        split_fp = int(sum(int(r["num_fp"]) for r in rows))
        oks_vals = [float(r["diagnostics"]["oks_mean"]) for r in rows if r["diagnostics"].get("oks_mean") is not None]
        kerr_vals = [float(r["diagnostics"]["keypoint_error_px_mean"]) for r in rows if r["diagnostics"].get("keypoint_error_px_mean") is not None]

        split_rows.append(
            {
                "split": split,
                "num_samples": len(rows),
                "grade_0_100": wavg,
                "counts": {
                    "gt_total": split_gt,
                    "pred_total": split_pred,
                    "matched_total": split_matched,
                    "fn_total": split_fn,
                    "fp_total": split_fp,
                },
                "detection": {
                    "precision_proxy": _safe_div(split_matched, split_pred),
                    "recall_proxy": _safe_div(split_matched, split_gt),
                    "miss_rate_proxy": None if split_gt <= 0 else float(1.0 - (split_matched / max(split_gt, 1))),
                },
                "geometry": {
                    "oks_mean": _mean(oks_vals),
                    "keypoint_error_px_mean": _mean(kerr_vals),
                },
            }
        )

        total_weight += float(sum(gts))
        weighted_sum += float(sum(g * w for g, w in zip(grades, gts)))
        total_gt += split_gt
        total_pred += split_pred
        total_matched += split_matched
        total_fn += split_fn
        total_fp += split_fp

    return {
        "run_grade_0_100": float(weighted_sum / max(total_weight, 1.0)) if split_rows else 0.0,
        "splits": split_rows,
        "num_samples_scored": len(samples),
        "run_detection": {
            "gt_total": total_gt,
            "pred_total": total_pred,
            "matched_total": total_matched,
            "fn_total": total_fn,
            "fp_total": total_fp,
            "precision_proxy": _safe_div(total_matched, total_pred),
            "recall_proxy": _safe_div(total_matched, total_gt),
            "miss_rate_proxy": None if total_gt <= 0 else float(1.0 - (total_matched / max(total_gt, 1))),
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

    hard_rows = sorted(sample_rows, key=lambda r: float(r.get("final_score_0_100", 0.0)))[:200]
    hard_examples_dir = hard_examples_dir if hard_examples_dir is not None else (reports_dir / "hard_examples")
    hard_examples_dir.mkdir(parents=True, exist_ok=True)
    hard_examples_model = hard_examples_dir / f"hard_examples_{model_key}.jsonl"
    hard_examples_latest = hard_examples_dir / "latest.jsonl"
    with hard_examples_model.open("w", encoding="utf-8") as f:
        for row in hard_rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    hard_examples_latest.write_text(hard_examples_model.read_text(encoding="utf-8"), encoding="utf-8")

    lines = [
        "# Detector Grade Summary",
        "",
        f"- model: `{model_name}`",
        f"- run grade (0-100): **{aggregate['run_grade_0_100']:.4f}**",
        f"- samples scored: {aggregate['num_samples_scored']}",
        f"- precision: {aggregate['run_detection']['precision_proxy']}",
        f"- recall: {aggregate['run_detection']['recall_proxy']}",
        "",
        "## Split Grades",
    ]
    for s in aggregate["splits"]:
        lines.append(
            f"- {s['split']}: {s['grade_0_100']:.4f}, "
            f"oks_mean={s['geometry']['oks_mean']}, "
            f"keypoint_error_px_mean={s['geometry']['keypoint_error_px_mean']}"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "summary_json": str(summary_json),
        "sample_jsonl": str(sample_jsonl),
        "summary_md": str(summary_md),
        "hard_examples_jsonl": str(hard_examples_model),
        "hard_examples_latest_jsonl": str(hard_examples_latest),
    }
