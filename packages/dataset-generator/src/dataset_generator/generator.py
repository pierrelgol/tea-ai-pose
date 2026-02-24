from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from .config import GeneratorConfig
from .geometry import (
    apply_homography_to_points,
    corners_px_to_yolo_obb,
    is_convex_quad,
    polygon_area,
    quad_inside_bounds,
)
from .homography import HomographyParams, sample_valid_homography
from .io import (
    CanonicalTarget,
    enforce_disjoint_background_splits,
    load_backgrounds_by_split,
    load_canonical_targets,
    load_target_classes,
    write_augmented_classes,
    write_metadata,
    write_yolo_obb_labels,
)
from .photometric import apply_photometric_stack
from .synthesis import blend_layer, visible_ratio, warp_target_and_mask

MIN_RAW_RECT_IOU = 0.72


@dataclass(slots=True)
class SampleResult:
    image_out_path: Path
    label_out_path: Path
    metadata_out_path: Path


@dataclass(slots=True)
class PlacedTarget:
    target: CanonicalTarget
    projected_corners_px: np.ndarray
    projected_corners_px_raw: np.ndarray
    projected_corners_norm: np.ndarray
    warped_target: np.ndarray
    warped_mask: np.ndarray
    class_id_exported: int
    placement: dict[str, Any]


def _output_stem(split: str, background_stem: str, sample_idx: int) -> str:
    return f"{split}_{background_stem}_s{sample_idx:03d}"


def _ensure_output_layout(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "meta" / split).mkdir(parents=True, exist_ok=True)


def _scaled_homography_params_for_target_count(
    *,
    base: HomographyParams,
    n_targets: int,
    config: GeneratorConfig,
) -> HomographyParams:
    """Shrink target scale bounds when placing more targets on the same image."""
    min_targets = config.targets_per_image_min
    max_targets = config.targets_per_image_max
    if max_targets <= min_targets:
        return base

    crowd_ratio = (n_targets - min_targets) / float(max_targets - min_targets)
    crowd_ratio = float(np.clip(crowd_ratio, 0.0, 1.0))

    # At max crowd, targets are scaled down to configured floor.
    min_scale_factor = float(config.crowd_scale_floor)
    scale_factor = 1.0 - (1.0 - min_scale_factor) * crowd_ratio

    scaled_min = max(0.05, base.scale_min * scale_factor)
    scaled_max = max(scaled_min, base.scale_max * scale_factor)
    return replace(base, scale_min=scaled_min, scale_max=scaled_max)


def _is_valid_projected_obb(projected_corners: np.ndarray, image_w: int, image_h: int, config: GeneratorConfig) -> bool:
    if projected_corners.shape != (4, 2):
        return False
    if not is_convex_quad(projected_corners):
        return False
    if not quad_inside_bounds(projected_corners, image_w, image_h):
        return False
    if polygon_area(projected_corners) < config.min_target_area_px:
        return False
    edge_lengths: list[float] = []
    for i in range(4):
        p0 = projected_corners[i]
        p1 = projected_corners[(i + 1) % 4]
        edge_len = float(np.linalg.norm(p1 - p0))
        edge_lengths.append(edge_len)
        if edge_len < config.min_edge_length_px:
            return False
    longest = max(edge_lengths)
    shortest = max(1e-9, min(edge_lengths))
    if (longest / shortest) > config.max_edge_aspect_ratio:
        return False
    for i in range(4):
        v_prev = projected_corners[(i - 1) % 4] - projected_corners[i]
        v_next = projected_corners[(i + 1) % 4] - projected_corners[i]
        n_prev = float(np.linalg.norm(v_prev))
        n_next = float(np.linalg.norm(v_next))
        if n_prev <= 1e-9 or n_next <= 1e-9:
            return False
        c = float(np.clip(np.dot(v_prev, v_next) / (n_prev * n_next), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(c)))
        if angle < config.min_corner_angle_deg or angle > config.max_corner_angle_deg:
            return False
    return True


def _principal_angle_deg(quad: np.ndarray) -> float:
    edges: list[tuple[float, np.ndarray]] = []
    for i in range(4):
        v = quad[(i + 1) % 4] - quad[i]
        edges.append((float(np.linalg.norm(v)), v))
    edges.sort(key=lambda x: x[0], reverse=True)
    vec = edges[0][1]
    return float(np.degrees(np.arctan2(float(vec[1]), float(vec[0]))) % 180.0)


def _angle_bin(angle_deg: float, n_bins: int = 12) -> int:
    step = 180.0 / float(n_bins)
    idx = int(np.floor(angle_deg / step))
    return max(0, min(n_bins - 1, idx))


def _canonicalize_quad_cw_start_tl(quad: np.ndarray) -> np.ndarray:
    center = np.mean(quad, axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    order = np.argsort(angles)
    ccw = quad[order]
    cw = ccw[::-1].copy()
    start = int(np.argmin(cw[:, 1] * 100000.0 + cw[:, 0]))
    return np.roll(cw, -start, axis=0).astype(np.float32)


def _polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    pa = a.astype(np.float32).reshape(-1, 1, 2)
    pb = b.astype(np.float32).reshape(-1, 1, 2)
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0
    inter_area, _ = cv2.intersectConvexConvex(pa, pb)
    inter = float(max(0.0, inter_area))
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _fit_rectangular_obb(projected_quad: np.ndarray) -> np.ndarray:
    rect = cv2.minAreaRect(projected_quad.astype(np.float32))
    corners = cv2.boxPoints(rect).astype(np.float32)
    return _canonicalize_quad_cw_start_tl(corners)


def _load_hard_class_boosts(path: Path | None) -> dict[int, float]:
    if path is None or not path.exists():
        return {}
    out: dict[int, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        ids = list(row.get("hard_class_ids") or [])
        if not ids:
            ids.extend(row.get("gt_class_ids") or [])
        for raw in ids:
            try:
                cid = int(raw)
            except Exception:
                continue
            out[cid] = out.get(cid, 0.0) + 1.0
    return out


def _resolve_curriculum_context(config: GeneratorConfig) -> dict[str, Any]:
    if not config.curriculum_enabled:
        return {
            "stage": "off",
            "orientation_within_10deg_rate": None,
            "report_path": None,
            "perspective_mult": 1.0,
            "occlusion_mult": 1.0,
        }

    stage = "mild"
    orientation_rate = None
    report_path = None
    perspective_mult = 0.7
    occlusion_mult = 0.75
    reports_dir = config.output_root / "grade_reports"
    candidates = sorted(reports_dir.glob("grade_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        report_path = candidates[0]
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
            vals: list[float] = []
            for split in payload.get("aggregate", {}).get("splits", []):
                value = split.get("geometry", {}).get("orientation_within_10deg_rate")
                if value is not None:
                    vals.append(float(value))
            if vals:
                orientation_rate = float(sum(vals) / len(vals))
        except Exception:
            orientation_rate = None

    if orientation_rate is not None:
        if orientation_rate >= config.curriculum_orientation_metric_threshold_hard:
            stage = "hard"
            perspective_mult = 1.2
            occlusion_mult = 1.15
        elif orientation_rate >= config.curriculum_orientation_metric_threshold_medium:
            stage = "medium"
            perspective_mult = 1.0
            occlusion_mult = 1.0

    return {
        "stage": stage,
        "orientation_within_10deg_rate": orientation_rate,
        "report_path": str(report_path) if report_path is not None else None,
        "perspective_mult": perspective_mult,
        "occlusion_mult": occlusion_mult,
    }


def _try_place_target(
    *,
    background: np.ndarray,
    target: CanonicalTarget,
    target_image: np.ndarray,
    occupancy_mask: np.ndarray,
    homography_params: HomographyParams,
    max_occlusion_ratio: float,
    rng: np.random.Generator,
    config: GeneratorConfig,
) -> PlacedTarget | None:
    bg_h, bg_w = background.shape[:2]
    hs = sample_valid_homography(
        canonical_corners_px=target.canonical_corners_px,
        background_w=bg_w,
        background_h=bg_h,
        rng=rng,
        params=homography_params,
    )
    projected_corners_raw = apply_homography_to_points(hs.H, target.canonical_corners_px)
    if not _is_valid_projected_obb(projected_corners_raw, bg_w, bg_h, config):
        return None
    projected_corners_rect = _fit_rectangular_obb(projected_corners_raw)
    if not _is_valid_projected_obb(projected_corners_rect, bg_w, bg_h, config):
        return None
    raw_rect_iou = _polygon_iou(projected_corners_raw, projected_corners_rect)
    if raw_rect_iou < MIN_RAW_RECT_IOU:
        return None
    projected_corners_norm = corners_px_to_yolo_obb(projected_corners_rect, bg_w, bg_h)

    warped_target, warped_mask = warp_target_and_mask(
        target=target_image,
        canonical_corners_px=target.canonical_corners_px,
        H=hs.H,
        out_w=bg_w,
        out_h=bg_h,
    )
    ratio_visible = visible_ratio(warped_mask=warped_mask, occupancy_mask=occupancy_mask)
    occlusion_ratio = 1.0 - ratio_visible
    if (not config.allow_partial_visibility and ratio_visible < 0.999) or occlusion_ratio > max_occlusion_ratio:
        return None

    class_id_exported = config.class_offset_base + target.class_id_local
    placement = {
        "target_image": str(target.image_path),
        "target_class_name": target.class_name,
        "target_class_id_local": target.class_id_local,
        "target_class_id_exported": class_id_exported,
        "H": hs.H.tolist(),
        "canonical_corners_px": target.canonical_corners_px.tolist(),
        "projected_corners_px_raw": projected_corners_raw.tolist(),
        "projected_corners_px_rect_obb": projected_corners_rect.tolist(),
        "projected_corners_px": projected_corners_rect.tolist(),
        "projected_corners_yolo_obb": projected_corners_norm.tolist(),
        "rect_fit_iou_raw_vs_rect": raw_rect_iou,
        "visible_ratio": ratio_visible,
        "occlusion_ratio": occlusion_ratio,
    }
    return PlacedTarget(
        target=target,
        projected_corners_px=projected_corners_rect,
        projected_corners_px_raw=projected_corners_raw,
        projected_corners_norm=projected_corners_norm,
        warped_target=warped_target,
        warped_mask=warped_mask,
        class_id_exported=class_id_exported,
        placement=placement,
    )


def generate_dataset(config: GeneratorConfig) -> list[SampleResult]:
    config.validate()
    if config.output_root.exists():
        shutil.rmtree(config.output_root)

    targets = load_canonical_targets(
        target_images_dir=config.target_images_dir,
        target_labels_dir=config.target_labels_dir,
        target_classes_file=config.target_classes_file,
    )
    target_classes = load_target_classes(config.target_classes_file)
    curriculum = _resolve_curriculum_context(config)
    target_indices_by_class: dict[int, list[int]] = {}
    for idx, target in enumerate(targets):
        target_indices_by_class.setdefault(int(target.class_id_local), []).append(idx)
    hard_boost_raw = _load_hard_class_boosts(config.hard_examples_path)
    class_ids = sorted(target_indices_by_class.keys())
    class_ids_np = np.array(class_ids, dtype=np.int32) if class_ids else np.zeros((0,), dtype=np.int32)
    class_weights = np.ones((len(class_ids),), dtype=np.float64)
    if class_ids:
        frequencies = np.array([len(target_indices_by_class[cid]) for cid in class_ids], dtype=np.float64)
        if float(np.sum(frequencies)) > 0:
            balance_weights = (1.0 / np.maximum(frequencies, 1.0)) ** float(config.class_balance_strength)
            balance_weights = balance_weights / max(float(np.sum(balance_weights)), 1e-9)
        else:
            balance_weights = np.ones_like(frequencies)
        max_boost = max([hard_boost_raw.get(cid, 0.0) for cid in class_ids], default=0.0)
        for i, cid in enumerate(class_ids):
            rel = 0.0 if max_boost <= 0 else (hard_boost_raw.get(cid, 0.0) / max_boost)
            hard_weight = 1.0 + float(config.hard_example_boost) * rel
            class_weights[i] = float(balance_weights[i]) * hard_weight
        class_weights = class_weights / max(float(np.sum(class_weights)), 1e-9)

    backgrounds_by_split = load_backgrounds_by_split(config.background_splits)
    backgrounds_by_split, enforced_audit = enforce_disjoint_background_splits(backgrounds_by_split)
    split_audit = {
        "enforced": int(enforced_audit.get("original_overlap_count", 0)) > 0,
        "post_enforcement": enforced_audit,
        "overlap_count": int(enforced_audit.get("overlap_count", 0)),
    }
    write_metadata(config.output_root / "split_audit.json", split_audit)
    target_images_cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
    max_target_cache_items = 256

    _ensure_output_layout(config.output_root)
    write_augmented_classes(config.output_root, target_classes, config.class_offset_base)

    homography_params = HomographyParams(
        scale_min=config.scale_min,
        scale_max=config.scale_max,
        translate_frac=config.translate_frac,
        perspective_jitter=config.perspective_jitter * float(curriculum["perspective_mult"]),
        min_quad_area_frac=config.min_quad_area_frac,
        max_attempts=config.max_attempts,
        edge_bias_prob=config.edge_bias_prob,
        edge_band_frac=config.edge_band_frac,
    )
    effective_max_occlusion = float(np.clip(config.max_occlusion_ratio * float(curriculum["occlusion_mult"]), 0.0, 0.95))
    rng = np.random.default_rng(config.seed)
    angle_bin_counts = np.zeros(12, dtype=np.int32)
    results: list[SampleResult] = []

    for split in ("train", "val"):
        backgrounds = backgrounds_by_split.get(split, [])
        for bg_path in backgrounds:
            background = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
            if background is None:
                continue
            bg_h, bg_w = background.shape[:2]

            for sample_idx in range(config.samples_per_background):
                planned_empty = bool(rng.random() < config.empty_sample_prob)
                if planned_empty:
                    n_targets = 0
                else:
                    n_targets = int(rng.integers(config.targets_per_image_min, config.targets_per_image_max + 1))
                sample_homography_params = _scaled_homography_params_for_target_count(
                    base=homography_params,
                    n_targets=n_targets,
                    config=config,
                )
                composited = background.copy()
                occupancy_mask = np.zeros((bg_h, bg_w), dtype=bool)
                placed: list[PlacedTarget] = []

                for _ in range(n_targets):
                    placed_target: PlacedTarget | None = None
                    placed_score = -1.0
                    for _attempt in range(config.max_attempts):
                        if class_ids:
                            picked_class = int(rng.choice(class_ids_np, p=class_weights))
                            candidates = target_indices_by_class.get(picked_class, [])
                            if candidates:
                                target = targets[int(candidates[int(rng.integers(0, len(candidates)))])]
                            else:
                                target = targets[int(rng.integers(0, len(targets)))]
                        else:
                            target = targets[int(rng.integers(0, len(targets)))]
                        key = str(target.image_path)
                        if key not in target_images_cache:
                            image = cv2.imread(str(target.image_path), cv2.IMREAD_COLOR)
                            if image is None:
                                break
                            target_images_cache[key] = image
                            if len(target_images_cache) > max_target_cache_items:
                                target_images_cache.popitem(last=False)
                        else:
                            target_images_cache.move_to_end(key, last=True)
                        target_image = target_images_cache[key]
                        candidate = _try_place_target(
                            background=composited,
                            target=target,
                            target_image=target_image,
                            occupancy_mask=occupancy_mask,
                            homography_params=sample_homography_params,
                            max_occlusion_ratio=effective_max_occlusion,
                            rng=rng,
                            config=config,
                        )
                        if candidate is not None:
                            angle_deg = _principal_angle_deg(candidate.projected_corners_px)
                            angle_idx = _angle_bin(angle_deg)
                            max_count = int(np.max(angle_bin_counts)) if int(np.sum(angle_bin_counts)) > 0 else 0
                            rarity_ratio = float((max_count + 1) / float(angle_bin_counts[angle_idx] + 1))
                            rarity_bonus = rarity_ratio ** float(config.angle_balance_strength)
                            area_px = polygon_area(candidate.projected_corners_px)
                            area_ratio = float(np.clip(area_px / max(1.0, float(bg_w * bg_h)), 0.0, 1.0))
                            size_penalty = 1.0 - area_ratio
                            score = 0.75 * rarity_bonus + 0.25 * size_penalty
                            if score > placed_score:
                                placed_score = score
                                candidate.placement["principal_angle_deg"] = angle_deg
                                placed_target = candidate
                    if placed_target is None:
                        continue

                    composited = blend_layer(
                        background=composited,
                        warped_target=placed_target.warped_target,
                        warped_mask=placed_target.warped_mask,
                        feather_px=5,
                    )
                    occupancy_mask = occupancy_mask | (placed_target.warped_mask > 0)
                    placed.append(placed_target)
                    angle_deg = float(
                        placed_target.placement.get(
                            "principal_angle_deg",
                            _principal_angle_deg(placed_target.projected_corners_px),
                        )
                    )
                    angle_bin_counts[_angle_bin(angle_deg)] += 1

                if not placed and n_targets > 0:
                    continue

                composited, photometric_applied = apply_photometric_stack(composited, rng=rng, config=config)
                labels_out = [(p.class_id_exported, p.projected_corners_norm) for p in placed]

                stem = _output_stem(split, bg_path.stem, sample_idx)
                image_out_path = config.output_root / "images" / split / f"{stem}{bg_path.suffix}"
                label_out_path = config.output_root / "labels" / split / f"{stem}.txt"
                meta_out_path = config.output_root / "meta" / split / f"{stem}.json"

                image_out_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(image_out_path), composited)
                write_yolo_obb_labels(label_out_path, labels_out)

                metadata = {
                    "seed": config.seed,
                    "generator_version": config.generator_version,
                    "background_dataset_name": config.background_dataset_name,
                    "curriculum": curriculum,
                    "background_image": str(bg_path),
                    "num_targets": len(placed),
                    "planned_empty": planned_empty,
                    "photometric_applied": photometric_applied,
                    "targets": [p.placement for p in placed],
                }
                write_metadata(meta_out_path, metadata)

                results.append(
                    SampleResult(
                        image_out_path=image_out_path,
                        label_out_path=label_out_path,
                        metadata_out_path=meta_out_path,
                    )
                )

    angle_counts = [int(v) for v in angle_bin_counts.tolist()]
    total_angles = max(1, int(sum(angle_counts)))
    angle_distribution = [float(v / total_angles) for v in angle_counts]
    write_metadata(
        config.output_root / "generation_summary.json",
        {
            "generator_version": config.generator_version,
            "curriculum": curriculum,
            "angle_bin_counts": angle_counts,
            "angle_bin_distribution": angle_distribution,
            "num_samples": len(results),
        },
    )

    return results
