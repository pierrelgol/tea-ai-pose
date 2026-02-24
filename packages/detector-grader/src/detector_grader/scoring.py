from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import Label
from .geometry import (
    angle_diff_deg,
    edge_lengths,
    polygon_area,
    polygon_center_drift_px,
    polygon_iou,
    principal_angle_deg,
    reorder_corners_to_best,
)


@dataclass(slots=True)
class ScoreWeights:
    iou: float = 0.32
    corner: float = 0.20
    angle: float = 0.24
    center: float = 0.14
    shape: float = 0.10

    fn_penalty: float = 0.30
    fp_penalty: float = 0.18
    containment_miss_penalty: float = 0.22
    containment_outside_penalty: float = 0.08
    tau_corner_px: float = 20.0
    tau_center_px: float = 20.0
    iou_gamma: float = 1.2

    def normalized(self) -> "ScoreWeights":
        s = self.iou + self.corner + self.angle + self.center + self.shape
        if s <= 0:
            raise ValueError("invalid weights sum <= 0")
        return ScoreWeights(
            iou=self.iou / s,
            corner=self.corner / s,
            angle=self.angle / s,
            center=self.center / s,
            shape=self.shape / s,
            fn_penalty=self.fn_penalty,
            fp_penalty=self.fp_penalty,
            containment_miss_penalty=self.containment_miss_penalty,
            containment_outside_penalty=self.containment_outside_penalty,
            tau_corner_px=self.tau_corner_px,
            tau_center_px=self.tau_center_px,
            iou_gamma=self.iou_gamma,
        )


@dataclass(slots=True)
class Match:
    gt_idx: int
    pred_idx: int
    iou: float
    quality: float


@dataclass(slots=True)
class MatchComponents:
    iou_score: float
    corner_score: float
    angle_score: float
    center_score: float
    shape_score: float
    weighted_score: float


@dataclass(slots=True)
class MatchDiagnostics:
    iou: float
    gt_area_missed_ratio: float
    pred_outside_ratio: float
    corner_error_px: float
    angle_error_deg: float
    center_error_px: float
    center_error_norm: float
    area_ratio: float
    abs_log_area_ratio: float
    edge_rel_error: float
    confidence: float
    class_match: int


def _poly_px(label: Label, w: int, h: int) -> np.ndarray:
    return (label.corners_norm * np.array([[w, h]], dtype=np.float32)).astype(np.float32)


def _greedy_match(
    gt_polys: list[np.ndarray],
    pred_polys: list[np.ndarray],
    gt_class_ids: list[int],
    pred_class_ids: list[int],
    iou_threshold: float,
) -> tuple[list[Match], list[int], list[int]]:
    candidates: list[tuple[float, float, int, int]] = []
    for gi, g in enumerate(gt_polys):
        g_center = np.mean(g, axis=0)
        g_diag = float(np.linalg.norm(np.max(g, axis=0) - np.min(g, axis=0)))
        g_angle = principal_angle_deg(g)
        for pi, p in enumerate(pred_polys):
            iou = polygon_iou(g, p)
            if iou >= iou_threshold:
                p_reordered = reorder_corners_to_best(g, p)
                p_center = np.mean(p_reordered, axis=0)
                center_err = float(np.linalg.norm(g_center - p_center)) / max(g_diag, 1e-6)
                center_score = float(np.exp(-center_err / 0.20))
                angle_err = angle_diff_deg(g_angle, principal_angle_deg(p_reordered))
                angle_score = float(np.clip(1.0 - (angle_err / 90.0), 0.0, 1.0))
                class_score = 1.0 if gt_class_ids[gi] == pred_class_ids[pi] else 0.7
                quality = float((iou ** 1.5) * center_score * (0.7 + 0.3 * angle_score) * class_score)
                candidates.append((quality, iou, gi, pi))
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    used_g: set[int] = set()
    used_p: set[int] = set()
    matches: list[Match] = []
    for quality, iou, gi, pi in candidates:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append(Match(gt_idx=gi, pred_idx=pi, iou=iou, quality=quality))
    unmatched_g = [i for i in range(len(gt_polys)) if i not in used_g]
    unmatched_p = [i for i in range(len(pred_polys)) if i not in used_p]
    return matches, unmatched_g, unmatched_p


def _safe_exp_score(err: float, tau: float) -> float:
    if tau <= 1e-9:
        return 0.0
    return float(np.exp(-max(0.0, err) / tau))


def _iou_emphasis_score(iou: float, gamma: float) -> float:
    if gamma <= 1e-9:
        return float(np.clip(iou, 0.0, 1.0))
    return float(np.clip(iou, 0.0, 1.0) ** gamma)


def _shape_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_l = edge_lengths(gt)
    pr_l = edge_lengths(pred)
    gt_l = gt_l / max(float(np.mean(gt_l)), 1e-9)
    pr_l = pr_l / max(float(np.mean(pr_l)), 1e-9)
    edge_consistency = float(np.mean(np.clip(1.0 - np.abs(gt_l - pr_l), 0.0, 1.0)))

    ga = polygon_area(gt)
    pa = polygon_area(pred)
    area_ratio = min(ga, pa) / max(ga, pa, 1e-9)
    return float(np.clip(0.5 * edge_consistency + 0.5 * area_ratio, 0.0, 1.0))


def _edge_rel_error(gt: np.ndarray, pred: np.ndarray) -> float:
    gt_l = edge_lengths(gt)
    pr_l = edge_lengths(pred)
    gt_l = gt_l / max(float(np.mean(gt_l)), 1e-9)
    pr_l = pr_l / max(float(np.mean(pr_l)), 1e-9)
    return float(np.mean(np.abs(gt_l - pr_l)))


def _orientation_reliability(poly: np.ndarray) -> float:
    lengths = edge_lengths(poly)
    longest = float(np.max(lengths))
    shortest = float(max(np.min(lengths), 1e-9))
    ratio = longest / shortest
    return float(np.clip((ratio - 1.0) / 0.35, 0.0, 1.0))


def _stat_or_none(vals: list[float], fn: str) -> float | None:
    if not vals:
        return None
    arr = np.array(vals, dtype=np.float32)
    if fn == "mean":
        return float(np.mean(arr))
    if fn == "median":
        return float(np.median(arr))
    if fn == "p90":
        return float(np.percentile(arr, 90))
    if fn == "p95":
        return float(np.percentile(arr, 95))
    raise ValueError(f"unsupported stat fn: {fn}")


def compute_match_components(gt: np.ndarray, pred: np.ndarray, weights: ScoreWeights) -> MatchComponents:
    pred_reordered = reorder_corners_to_best(gt, pred)

    iou_raw = float(np.clip(polygon_iou(gt, pred_reordered), 0.0, 1.0))
    iou_score = _iou_emphasis_score(iou_raw, weights.iou_gamma)

    corner_err = float(np.mean(np.linalg.norm(gt - pred_reordered, axis=1)))
    corner_score = _safe_exp_score(corner_err, weights.tau_corner_px)

    angle_gt = principal_angle_deg(gt)
    angle_pr = principal_angle_deg(pred_reordered)
    angle_err = angle_diff_deg(angle_gt, angle_pr)
    angle_score_raw = float(np.clip(1.0 - (angle_err / 90.0), 0.0, 1.0))
    orientation_weight = min(_orientation_reliability(gt), _orientation_reliability(pred_reordered))
    angle_score = float(orientation_weight * angle_score_raw + (1.0 - orientation_weight) * 1.0)

    center_err = polygon_center_drift_px(gt, pred_reordered)
    center_score = _safe_exp_score(center_err, weights.tau_center_px)

    shape_score = _shape_score(gt, pred_reordered)

    weighted = (
        weights.iou * iou_score
        + weights.corner * corner_score
        + weights.angle * angle_score
        + weights.center * center_score
        + weights.shape * shape_score
    )
    weighted = float(np.clip(weighted, 0.0, 1.0))

    return MatchComponents(
        iou_score=iou_score,
        corner_score=corner_score,
        angle_score=angle_score,
        center_score=center_score,
        shape_score=shape_score,
        weighted_score=weighted,
    )


def compute_match_diagnostics(
    gt: np.ndarray,
    pred: np.ndarray,
    pred_conf: float,
    class_match: bool,
    image_diag_px: float,
) -> MatchDiagnostics:
    pred_reordered = reorder_corners_to_best(gt, pred)

    iou = float(np.clip(polygon_iou(gt, pred_reordered), 0.0, 1.0))
    inter = iou * (polygon_area(gt) + polygon_area(pred_reordered)) / max(1e-9, (1.0 + iou))
    gt_area = max(polygon_area(gt), 1e-9)
    pred_area = max(polygon_area(pred_reordered), 1e-9)
    gt_area_missed_ratio = float(np.clip(1.0 - (inter / gt_area), 0.0, 1.0))
    pred_outside_ratio = float(np.clip(1.0 - (inter / pred_area), 0.0, 1.0))
    corner_error_px = float(np.mean(np.linalg.norm(gt - pred_reordered, axis=1)))

    angle_gt = principal_angle_deg(gt)
    angle_pr = principal_angle_deg(pred_reordered)
    angle_error_deg = float(angle_diff_deg(angle_gt, angle_pr))

    center_error_px = float(polygon_center_drift_px(gt, pred_reordered))
    center_error_norm = center_error_px / max(image_diag_px, 1e-9)

    ga = polygon_area(gt)
    pa = polygon_area(pred_reordered)
    area_ratio = pa / max(ga, 1e-9)
    abs_log_area_ratio = float(abs(np.log(max(area_ratio, 1e-9))))
    edge_rel_error = _edge_rel_error(gt, pred_reordered)

    return MatchDiagnostics(
        iou=iou,
        gt_area_missed_ratio=gt_area_missed_ratio,
        pred_outside_ratio=pred_outside_ratio,
        corner_error_px=corner_error_px,
        angle_error_deg=angle_error_deg,
        center_error_px=center_error_px,
        center_error_norm=float(center_error_norm),
        area_ratio=float(area_ratio),
        abs_log_area_ratio=abs_log_area_ratio,
        edge_rel_error=float(edge_rel_error),
        confidence=float(pred_conf),
        class_match=1 if class_match else 0,
    )


def score_sample(
    *,
    split: str,
    stem: str,
    gt_labels: list[Label],
    pred_labels: list[Label],
    w: int,
    h: int,
    iou_threshold: float,
    weights: ScoreWeights,
) -> dict[str, Any]:
    wp = weights.normalized()
    image_diag_px = float((w * w + h * h) ** 0.5)
    gt_polys = [_poly_px(g, w, h) for g in gt_labels]
    pred_polys = [_poly_px(p, w, h) for p in pred_labels]

    matches, unmatched_gt, unmatched_pred = _greedy_match(
        gt_polys,
        pred_polys,
        [g.class_id for g in gt_labels],
        [p.class_id for p in pred_labels],
        iou_threshold,
    )

    comps: list[MatchComponents] = []
    diags: list[MatchDiagnostics] = []
    for m in matches:
        c = compute_match_components(gt_polys[m.gt_idx], pred_polys[m.pred_idx], wp)
        comps.append(c)
        d = compute_match_diagnostics(
            gt=gt_polys[m.gt_idx],
            pred=pred_polys[m.pred_idx],
            pred_conf=pred_labels[m.pred_idx].confidence,
            class_match=gt_labels[m.gt_idx].class_id == pred_labels[m.pred_idx].class_id,
            image_diag_px=image_diag_px,
        )
        diags.append(d)

    if comps:
        base = float(np.mean([c.weighted_score for c in comps]))
    else:
        base = 0.0

    denom = max(1, len(gt_labels))
    fn_pen = (len(unmatched_gt) / denom) * wp.fn_penalty
    fp_pen = (len(unmatched_pred) / denom) * wp.fp_penalty
    containment_miss_mean = float(np.mean([d.gt_area_missed_ratio for d in diags])) if diags else 0.0
    containment_outside_mean = float(np.mean([d.pred_outside_ratio for d in diags])) if diags else 0.0
    containment_pen = (
        containment_miss_mean * wp.containment_miss_penalty
        + containment_outside_mean * wp.containment_outside_penalty
    )

    final = float(np.clip(base - fn_pen - fp_pen - containment_pen, 0.0, 1.0))

    comp_mean = {
        "iou_score": float(np.mean([c.iou_score for c in comps])) if comps else None,
        "corner_score": float(np.mean([c.corner_score for c in comps])) if comps else None,
        "angle_score": float(np.mean([c.angle_score for c in comps])) if comps else None,
        "center_score": float(np.mean([c.center_score for c in comps])) if comps else None,
        "shape_score": float(np.mean([c.shape_score for c in comps])) if comps else None,
    }

    corner_errors = [d.corner_error_px for d in diags]
    angle_errors = [d.angle_error_deg for d in diags]
    center_errors = [d.center_error_px for d in diags]
    center_norm_errors = [d.center_error_norm for d in diags]
    area_ratios = [d.area_ratio for d in diags]
    abs_log_area_ratios = [d.abs_log_area_ratio for d in diags]
    edge_rel_errors = [d.edge_rel_error for d in diags]
    confidences = [d.confidence for d in diags]
    class_match_values = [d.class_match for d in diags]
    gt_area_missed = [d.gt_area_missed_ratio for d in diags]
    pred_outside = [d.pred_outside_ratio for d in diags]

    diagnostics = {
        "iou_mean": _stat_or_none([d.iou for d in diags], "mean"),
        "iou_median": _stat_or_none([d.iou for d in diags], "median"),
        "iou_p90": _stat_or_none([d.iou for d in diags], "p90"),
        "iou_p95": _stat_or_none([d.iou for d in diags], "p95"),
        "corner_error_px_mean": _stat_or_none(corner_errors, "mean"),
        "corner_error_px_median": _stat_or_none(corner_errors, "median"),
        "corner_error_px_p90": _stat_or_none(corner_errors, "p90"),
        "corner_error_px_p95": _stat_or_none(corner_errors, "p95"),
        "angle_error_deg_mean": _stat_or_none(angle_errors, "mean"),
        "angle_error_deg_median": _stat_or_none(angle_errors, "median"),
        "angle_error_deg_p90": _stat_or_none(angle_errors, "p90"),
        "angle_error_deg_p95": _stat_or_none(angle_errors, "p95"),
        "center_error_px_mean": _stat_or_none(center_errors, "mean"),
        "center_error_px_median": _stat_or_none(center_errors, "median"),
        "center_error_px_p90": _stat_or_none(center_errors, "p90"),
        "center_error_px_p95": _stat_or_none(center_errors, "p95"),
        "center_error_norm_mean": _stat_or_none(center_norm_errors, "mean"),
        "area_ratio_mean": _stat_or_none(area_ratios, "mean"),
        "abs_log_area_ratio_mean": _stat_or_none(abs_log_area_ratios, "mean"),
        "edge_rel_error_mean": _stat_or_none(edge_rel_errors, "mean"),
        "gt_area_missed_ratio_mean": _stat_or_none(gt_area_missed, "mean"),
        "gt_area_missed_ratio_p90": _stat_or_none(gt_area_missed, "p90"),
        "pred_outside_ratio_mean": _stat_or_none(pred_outside, "mean"),
        "confidence_mean": _stat_or_none(confidences, "mean"),
        "confidence_median": _stat_or_none(confidences, "median"),
        "class_match_rate": _stat_or_none(class_match_values, "mean"),
        "angle_le_5_rate": _stat_or_none([1.0 if v <= 5.0 else 0.0 for v in angle_errors], "mean"),
        "angle_le_10_rate": _stat_or_none([1.0 if v <= 10.0 else 0.0 for v in angle_errors], "mean"),
        "iou_ge_50_rate": _stat_or_none([1.0 if v >= 0.50 else 0.0 for v in [d.iou for d in diags]], "mean"),
        "iou_ge_75_rate": _stat_or_none([1.0 if v >= 0.75 else 0.0 for v in [d.iou for d in diags]], "mean"),
    }

    return {
        "split": split,
        "stem": stem,
        "image_width_px": int(w),
        "image_height_px": int(h),
        "image_diagonal_px": image_diag_px,
        "num_gt": len(gt_labels),
        "num_pred": len(pred_labels),
        "num_matches": len(matches),
        "num_fn": len(unmatched_gt),
        "num_fp": len(unmatched_pred),
        "match_iou_mean": float(np.mean([m.iou for m in matches])) if matches else None,
        "match_quality_mean": float(np.mean([m.quality for m in matches])) if matches else None,
        "components_mean": comp_mean,
        "diagnostics": diagnostics,
        "num_class_mismatch": int(sum(1 for d in diags if d.class_match == 0)),
        "gt_class_ids": [int(g.class_id) for g in gt_labels],
        "pred_class_ids": [int(p.class_id) for p in pred_labels],
        "hard_class_ids": sorted(
            {
                int(gt_labels[m.gt_idx].class_id)
                for m, d in zip(matches, diags)
                if int(d.class_match) == 0
            }
        ),
        "match_coverage_gt": float(len(matches) / max(1, len(gt_labels))),
        "match_coverage_pred": float(len(matches) / max(1, len(pred_labels))),
        "has_gt": bool(len(gt_labels) > 0),
        "has_pred": bool(len(pred_labels) > 0),
        "base_score_0_1": base,
        "penalty_fn": fn_pen,
        "penalty_fp": fp_pen,
        "penalty_containment": containment_pen,
        "final_score_0_1": final,
        "final_score_0_100": 100.0 * final,
    }
