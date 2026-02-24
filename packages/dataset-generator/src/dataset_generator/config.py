from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from pathlib import Path


@dataclass(slots=True)
class GeneratorConfig:
    background_splits: dict[str, Path] = field(
        default_factory=lambda: {
            "train": Path("backgrounds/train"),
            "val": Path("backgrounds/val"),
        }
    )
    background_dataset_name: str = "default"
    target_images_dir: Path = Path("targets/images")
    target_labels_dir: Path = Path("targets/labels")
    target_classes_file: Path = Path("targets/classes.txt")
    output_root: Path = Path("augmented/default")
    hard_examples_path: Path | None = Path("artifacts/models/default/runs/current/grade/hard_examples/latest.jsonl")
    hard_example_boost: float = 1.5
    class_balance_strength: float = 1.0
    curriculum_enabled: bool = True
    curriculum_orientation_metric_threshold_medium: float = 0.70
    curriculum_orientation_metric_threshold_hard: float = 0.85

    samples_per_background: int = 1
    seed: int | None = None
    generator_version: str = "obb_robust_v2"

    targets_per_image_min: int = 1
    targets_per_image_max: int = 3
    empty_sample_prob: float = 0.15
    max_occlusion_ratio: float = 0.60
    allow_partial_visibility: bool = True

    scale_min: float = 0.12
    scale_max: float = 0.65
    crowd_scale_floor: float = 0.30
    translate_frac: float = 0.35
    perspective_jitter: float = 0.12
    min_quad_area_frac: float = 0.0015
    min_target_area_px: float = 24.0
    min_edge_length_px: float = 4.0
    min_corner_angle_deg: float = 16.0
    max_corner_angle_deg: float = 164.0
    max_edge_aspect_ratio: float = 8.0
    angle_balance_strength: float = 1.0
    max_attempts: int = 50
    edge_bias_prob: float = 0.40
    edge_band_frac: float = 0.22

    class_offset_base: int = 80
    blur_prob: float = 0.55
    motion_blur_prob: float = 0.35
    noise_prob: float = 0.45
    jpeg_artifact_prob: float = 0.35
    color_jitter_prob: float = 0.75
    color_hue_shift_max_deg: float = 14.0
    color_sat_gain_min: float = 0.60
    color_sat_gain_max: float = 1.45
    color_val_gain_min: float = 0.60
    color_val_gain_max: float = 1.40
    gaussian_blur_kernel_min: int = 5
    gaussian_blur_kernel_max: int = 11
    motion_blur_kernel_min: int = 7
    motion_blur_kernel_max: int = 19
    motion_blur_angle_max_deg: float = 45.0
    noise_sigma_min: float = 5.0
    noise_sigma_max: float = 20.0
    jpeg_quality_min: int = 25
    jpeg_quality_max: int = 75

    def validate(self) -> None:
        if self.samples_per_background < 1:
            raise ValueError("samples_per_background must be >= 1")
        if self.hard_example_boost < 0:
            raise ValueError("hard_example_boost must be >= 0")
        if self.class_balance_strength < 0:
            raise ValueError("class_balance_strength must be >= 0")
        if self.curriculum_orientation_metric_threshold_medium < 0 or self.curriculum_orientation_metric_threshold_medium > 1:
            raise ValueError("curriculum medium threshold must be in [0,1]")
        if self.curriculum_orientation_metric_threshold_hard < 0 or self.curriculum_orientation_metric_threshold_hard > 1:
            raise ValueError("curriculum hard threshold must be in [0,1]")
        if self.curriculum_orientation_metric_threshold_hard < self.curriculum_orientation_metric_threshold_medium:
            raise ValueError("curriculum hard threshold must be >= medium threshold")
        if self.scale_min <= 0 or self.scale_max <= 0:
            raise ValueError("scale bounds must be > 0")
        if self.scale_min > self.scale_max:
            raise ValueError("scale_min must be <= scale_max")
        if self.translate_frac < 0:
            raise ValueError("translate_frac must be >= 0")
        if self.perspective_jitter < 0:
            raise ValueError("perspective_jitter must be >= 0")
        if self.min_quad_area_frac <= 0:
            raise ValueError("min_quad_area_frac must be > 0")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.edge_bias_prob < 0 or self.edge_bias_prob > 1:
            raise ValueError("edge_bias_prob must be in [0,1]")
        if self.edge_band_frac <= 0 or self.edge_band_frac >= 0.5:
            raise ValueError("edge_band_frac must be in (0,0.5)")
        if self.class_offset_base < 0:
            raise ValueError("class_offset_base must be >= 0")
        if self.targets_per_image_min < 1:
            raise ValueError("targets_per_image_min must be >= 1")
        if self.targets_per_image_max < self.targets_per_image_min:
            raise ValueError("targets_per_image_max must be >= targets_per_image_min")
        if self.empty_sample_prob < 0 or self.empty_sample_prob > 1:
            raise ValueError("empty_sample_prob must be in [0,1]")
        if self.max_occlusion_ratio < 0 or self.max_occlusion_ratio >= 1:
            raise ValueError("max_occlusion_ratio must be in [0,1)")
        if self.crowd_scale_floor <= 0 or self.crowd_scale_floor > 1:
            raise ValueError("crowd_scale_floor must be in (0,1]")
        if self.min_target_area_px <= 0:
            raise ValueError("min_target_area_px must be > 0")
        if self.min_edge_length_px <= 0:
            raise ValueError("min_edge_length_px must be > 0")
        if self.min_corner_angle_deg <= 0 or self.min_corner_angle_deg >= 90:
            raise ValueError("min_corner_angle_deg must be in (0,90)")
        if self.max_corner_angle_deg <= 90 or self.max_corner_angle_deg >= 180:
            raise ValueError("max_corner_angle_deg must be in (90,180)")
        if self.min_corner_angle_deg >= self.max_corner_angle_deg:
            raise ValueError("min_corner_angle_deg must be < max_corner_angle_deg")
        if self.max_edge_aspect_ratio < 1.0:
            raise ValueError("max_edge_aspect_ratio must be >= 1")
        if self.angle_balance_strength < 0:
            raise ValueError("angle_balance_strength must be >= 0")
        for name, value in (
            ("blur_prob", self.blur_prob),
            ("motion_blur_prob", self.motion_blur_prob),
            ("noise_prob", self.noise_prob),
            ("jpeg_artifact_prob", self.jpeg_artifact_prob),
            ("color_jitter_prob", self.color_jitter_prob),
        ):
            if value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.color_hue_shift_max_deg < 0:
            raise ValueError("color_hue_shift_max_deg must be >= 0")
        if self.color_sat_gain_min <= 0 or self.color_sat_gain_max <= 0:
            raise ValueError("color_sat_gain bounds must be > 0")
        if self.color_sat_gain_min > self.color_sat_gain_max:
            raise ValueError("color_sat_gain_min must be <= color_sat_gain_max")
        if self.color_val_gain_min <= 0 or self.color_val_gain_max <= 0:
            raise ValueError("color_val_gain bounds must be > 0")
        if self.color_val_gain_min > self.color_val_gain_max:
            raise ValueError("color_val_gain_min must be <= color_val_gain_max")
        if self.gaussian_blur_kernel_min < 1 or self.gaussian_blur_kernel_max < 1:
            raise ValueError("gaussian blur kernel bounds must be >= 1")
        if self.gaussian_blur_kernel_min > self.gaussian_blur_kernel_max:
            raise ValueError("gaussian_blur_kernel_min must be <= gaussian_blur_kernel_max")
        if self.motion_blur_kernel_min < 1 or self.motion_blur_kernel_max < 1:
            raise ValueError("motion blur kernel bounds must be >= 1")
        if self.motion_blur_kernel_min > self.motion_blur_kernel_max:
            raise ValueError("motion_blur_kernel_min must be <= motion_blur_kernel_max")
        if self.motion_blur_angle_max_deg < 0:
            raise ValueError("motion_blur_angle_max_deg must be >= 0")
        if self.noise_sigma_min < 0 or self.noise_sigma_max < 0:
            raise ValueError("noise sigma bounds must be >= 0")
        if self.noise_sigma_min > self.noise_sigma_max:
            raise ValueError("noise_sigma_min must be <= noise_sigma_max")
        if self.jpeg_quality_min < 1 or self.jpeg_quality_max > 100:
            raise ValueError("jpeg quality bounds must be in [1,100]")
        if self.jpeg_quality_min > self.jpeg_quality_max:
            raise ValueError("jpeg_quality_min must be <= jpeg_quality_max")
        if "train" not in self.background_splits or "val" not in self.background_splits:
            raise ValueError("background_splits must define train and val paths")
        for split, path in self.background_splits.items():
            if not path.exists():
                raise FileNotFoundError(f"background split path does not exist ({split}): {path}")
