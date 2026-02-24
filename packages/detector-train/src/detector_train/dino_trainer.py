from __future__ import annotations

import types
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.utils.torch_utils import unwrap_model

from dinov3_bridge import DinoV3Teacher, resolve_local_dinov3_root


@dataclass(frozen=True, slots=True)
class DinoDistillConfig:
    dino_root: Path
    stage_a_epochs: int
    stage_a_freeze: int
    stage_a_weight: float
    stage_b_weight: float
    warmup_epochs: int
    student_layers: tuple[int, ...]
    channels: int
    object_weight: float
    background_weight: float
    viz_enabled: bool
    viz_mode: str
    viz_every_n_epochs: int
    total_epochs: int
    viz_max_samples: int


class DinoPoseTrainer(PoseTrainer):
    """Pose trainer with mandatory spatial DINOv3 feature distillation."""

    def __init__(self, cfg=None, overrides=None, _callbacks=None, *, dino_cfg: DinoDistillConfig | None = None):
        if cfg is None:
            super().__init__(overrides=overrides, _callbacks=_callbacks)
        else:
            super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        if dino_cfg is None:
            raise ValueError("dino_cfg is required for DinoPoseTrainer")
        self._dino_cfg = dino_cfg

    def _setup_train(self) -> None:
        super()._setup_train()

        model = unwrap_model(self.model)
        self._dino_teacher = DinoV3Teacher(
            resolve_local_dinov3_root(self._dino_cfg.dino_root),
            device=self.device,
        )
        self._dino_stage_a_epochs = max(0, int(self._dino_cfg.stage_a_epochs))
        self._dino_stage_a_freeze = max(0, int(self._dino_cfg.stage_a_freeze))
        self._dino_stage_a_weight = max(0.0, float(self._dino_cfg.stage_a_weight))
        self._dino_stage_b_weight = max(0.0, float(self._dino_cfg.stage_b_weight))
        self._dino_warmup_epochs = max(0, int(self._dino_cfg.warmup_epochs))
        self._dino_viz_enabled = bool(self._dino_cfg.viz_enabled)
        self._dino_viz_mode = str(self._dino_cfg.viz_mode)
        self._dino_viz_every_n_epochs = max(1, int(self._dino_cfg.viz_every_n_epochs))
        self._dino_total_epochs = max(1, int(self._dino_cfg.total_epochs))
        self._dino_viz_capture_epoch = False
        self._dino_viz_max_samples = max(1, int(self._dino_cfg.viz_max_samples))
        self._dino_active_weight = 0.0
        self._dino_last_weight = 0.0
        self._dino_loss_sum = 0.0
        self._dino_loss_count = 0
        self._dino_epoch_loss = 0.0
        self._dino_epoch_obj_loss = 0.0
        self._dino_epoch_bg_loss = 0.0
        self._dino_viz_snapshot: dict[str, object] | None = None
        self._dino_stage_a_active = self._dino_stage_a_epochs > 0 and self._dino_stage_a_freeze > 0
        self._dino_stage_a_prefixes = tuple(f"model.{i}." for i in range(self._dino_stage_a_freeze))

        layer_count = len(getattr(model, "model", []))
        if layer_count <= 0:
            raise RuntimeError("unable to resolve YOLO model layers for DINOv3 distillation")
        resolved_layers: list[int] = []
        for idx in self._dino_cfg.student_layers:
            ridx = layer_count + idx if idx < 0 else idx
            if ridx < 0 or ridx >= layer_count:
                raise ValueError(f"invalid dino student layer index {idx} for layer_count={layer_count}")
            resolved_layers.append(int(ridx))
        if not resolved_layers:
            raise ValueError("at least one dino student layer is required")
        self._dino_layers = tuple(dict.fromkeys(resolved_layers))

        self._dino_feature_cache: dict[int, torch.Tensor] = {}
        self._dino_hooks = []

        def _capture_layer(layer_idx: int):
            def _hook(_module, _inputs, output):
                if isinstance(output, torch.Tensor):
                    self._dino_feature_cache[layer_idx] = output
                elif isinstance(output, (list, tuple)) and output and isinstance(output[0], torch.Tensor):
                    self._dino_feature_cache[layer_idx] = output[0]
            return _hook

        for layer_idx in self._dino_layers:
            handle = model.model[layer_idx].register_forward_hook(_capture_layer(layer_idx))
            self._dino_hooks.append(handle)

        original_loss = model.loss
        trainer = self

        def _build_object_mask(batch: dict[str, torch.Tensor], out_hw: tuple[int, int], device: torch.device) -> torch.Tensor:
            h, w = out_hw
            mask = torch.zeros((int(batch["img"].shape[0]), 1, h, w), device=device)
            batch_idx = batch["batch_idx"].view(-1).to(device=device, dtype=torch.long)
            boxes = batch["bboxes"].to(device=device, dtype=torch.float32)
            if boxes.numel() == 0:
                return mask
            boxes_px = boxes.clone()
            boxes_px[:, 0] = boxes_px[:, 0].clamp(0.0, 1.0) * w
            boxes_px[:, 1] = boxes_px[:, 1].clamp(0.0, 1.0) * h
            boxes_px[:, 2] = boxes_px[:, 2].clamp(0.0, 1.0) * w
            boxes_px[:, 3] = boxes_px[:, 3].clamp(0.0, 1.0) * h
            for bi in range(mask.shape[0]):
                idxs = (batch_idx == bi).nonzero(as_tuple=False).view(-1).detach().cpu().numpy()
                if idxs.size == 0:
                    continue
                canvas = np.zeros((h, w), dtype=np.uint8)
                for i in idxs.tolist():
                    cx, cy, bw, bh = [float(v) for v in boxes_px[i].tolist()]
                    x1 = max(0, min(w - 1, int(round(cx - 0.5 * bw))))
                    y1 = max(0, min(h - 1, int(round(cy - 0.5 * bh))))
                    x2 = max(0, min(w - 1, int(round(cx + 0.5 * bw))))
                    y2 = max(0, min(h - 1, int(round(cy + 0.5 * bh))))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color=1, thickness=-1)
                mask[bi, 0] = torch.from_numpy(canvas).to(device=device, dtype=mask.dtype)
            return mask

        def _compute_distill_loss(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            if trainer._dino_active_weight <= 0.0:
                z = batch["img"].sum() * 0.0
                return z, z, z

            teacher_map = trainer._dino_teacher.extract_feature_map(batch["img"])
            obj_weight = max(0.0, float(trainer._dino_cfg.object_weight))
            bg_weight = max(0.0, float(trainer._dino_cfg.background_weight))

            layer_losses: list[torch.Tensor] = []
            layer_obj_losses: list[torch.Tensor] = []
            layer_bg_losses: list[torch.Tensor] = []
            signal_maps: list[torch.Tensor] = []
            mask_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

            def _get_masks(hw: tuple[int, int], dev: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
                cached = mask_cache.get(hw)
                if cached is not None:
                    return cached
                obj = _build_object_mask(batch, hw, dev)
                bg = 1.0 - obj
                mask_cache[hw] = (obj, bg)
                return obj, bg

            viz_obj_mask: torch.Tensor | None = None
            sample_count = 0
            if trainer._dino_viz_enabled and trainer._dino_viz_capture_epoch:
                sample_count = min(int(batch["img"].shape[0]), trainer._dino_viz_max_samples)
                if sample_count > 0:
                    teacher_hw = (int(teacher_map.shape[-2]), int(teacher_map.shape[-1]))
                    viz_obj_mask, _ = _get_masks(teacher_hw, teacher_map.device)

            for layer_idx in trainer._dino_layers:
                if layer_idx not in trainer._dino_feature_cache:
                    raise RuntimeError(f"missing captured student feature for layer {layer_idx}")
                student_map = trainer._dino_feature_cache[layer_idx]
                if student_map.ndim != 4:
                    raise RuntimeError(f"student feature at layer {layer_idx} must be BCHW, got {student_map.shape}")

                target_c = min(
                    max(8, int(trainer._dino_cfg.channels)),
                    int(student_map.shape[1]),
                    int(teacher_map.shape[1]),
                )
                smap = student_map[:, :target_c, :, :]
                tbase = teacher_map[:, :target_c, :, :].to(device=smap.device, dtype=smap.dtype)
                tmap = F.interpolate(
                    tbase,
                    size=smap.shape[-2:],
                    mode="nearest",
                )

                cos_dist = 1.0 - F.cosine_similarity(smap, tmap, dim=1).unsqueeze(1)  # Bx1xHxW
                obj_mask, bg_mask = _get_masks((int(smap.shape[-2]), int(smap.shape[-1])), smap.device)

                obj_den = obj_mask.sum().clamp_min(1.0)
                bg_den = bg_mask.sum().clamp_min(1.0)
                obj_loss = (cos_dist * obj_mask).sum() / obj_den
                bg_loss = (cos_dist * bg_mask).sum() / bg_den
                weighted = obj_weight * obj_loss + bg_weight * bg_loss
                signal_map = (
                    cos_dist[:, 0]
                    * (obj_weight * obj_mask[:, 0] + bg_weight * bg_mask[:, 0])
                )
                signal_maps.append(signal_map)

                layer_losses.append(weighted)
                layer_obj_losses.append(obj_loss)
                layer_bg_losses.append(bg_loss)

            if trainer._dino_viz_enabled and sample_count > 0 and viz_obj_mask is not None:
                mean_signal = torch.stack(signal_maps, dim=0).mean(dim=0)
                trainer._dino_viz_snapshot = {
                    "images": batch["img"][:sample_count].detach().float().cpu().numpy(),
                    "teacher": teacher_map[:sample_count].detach().float().cpu().numpy(),
                    "obj_mask": viz_obj_mask[:sample_count, 0].detach().float().cpu().numpy(),
                    "signal_map": mean_signal[:sample_count].detach().float().cpu().numpy(),
                }

            mean_loss = torch.stack(layer_losses).mean() * trainer._dino_active_weight
            mean_obj = torch.stack(layer_obj_losses).mean() * trainer._dino_active_weight
            mean_bg = torch.stack(layer_bg_losses).mean() * trainer._dino_active_weight
            return mean_loss, mean_obj, mean_bg

        def _loss_with_distill(model_self, batch, preds=None):
            trainer._dino_feature_cache.clear()
            if preds is None:
                preds = model_self.predict(batch["img"])
            det_out = original_loss(batch, preds)
            if not isinstance(det_out, (tuple, list)) or len(det_out) < 2:
                return det_out

            det_loss, loss_items = det_out[0], det_out[1]
            distill_loss, obj_loss, bg_loss = _compute_distill_loss(batch)

            trainer._dino_last_weight = float(trainer._dino_active_weight)
            trainer._dino_loss_sum += float(distill_loss.detach().item())
            trainer._dino_loss_count += 1
            trainer._dino_epoch_loss = trainer._dino_loss_sum / max(1, trainer._dino_loss_count)
            trainer._dino_epoch_obj_loss = float(obj_loss.detach().item())
            trainer._dino_epoch_bg_loss = float(bg_loss.detach().item())

            return det_loss + distill_loss, loss_items

        model.loss = types.MethodType(_loss_with_distill, model)

        def _on_train_epoch_start(cb_trainer) -> None:
            cb_trainer._dino_loss_sum = 0.0
            cb_trainer._dino_loss_count = 0
            cb_trainer._dino_epoch_loss = 0.0
            cb_trainer._dino_epoch_obj_loss = 0.0
            cb_trainer._dino_epoch_bg_loss = 0.0
            cb_trainer._dino_viz_snapshot = None

            epoch_idx = int(getattr(cb_trainer, "epoch", 0))
            in_stage_a = epoch_idx < cb_trainer._dino_stage_a_epochs
            cb_trainer._dino_stage_a_active = (
                in_stage_a and cb_trainer._dino_stage_a_epochs > 0 and cb_trainer._dino_stage_a_freeze > 0
            )
            warmup = cb_trainer._dino_warmup_epochs
            if warmup <= 0:
                weight_scale = 1.0
            else:
                weight_scale = min(1.0, float(epoch_idx + 1) / float(warmup))
            base_weight = cb_trainer._dino_stage_a_weight if in_stage_a else cb_trainer._dino_stage_b_weight
            cb_trainer._dino_active_weight = base_weight * weight_scale
            if cb_trainer._dino_viz_enabled:
                if cb_trainer._dino_viz_mode == "off":
                    cb_trainer._dino_viz_capture_epoch = False
                elif cb_trainer._dino_viz_mode == "final_only":
                    cb_trainer._dino_viz_capture_epoch = (epoch_idx + 1) == cb_trainer._dino_total_epochs
                else:
                    cb_trainer._dino_viz_capture_epoch = (
                        (epoch_idx + 1) == cb_trainer._dino_total_epochs
                        or ((epoch_idx + 1) % cb_trainer._dino_viz_every_n_epochs == 0)
                    )
            else:
                cb_trainer._dino_viz_capture_epoch = False

            if cb_trainer._dino_stage_a_active and cb_trainer._dino_stage_a_prefixes:
                base_model = unwrap_model(cb_trainer.model)
                for name, module in base_model.named_modules():
                    if any(name.startswith(pref) for pref in cb_trainer._dino_stage_a_prefixes) and isinstance(
                        module, nn.BatchNorm2d
                    ):
                        module.eval()

        self.add_callback("on_train_epoch_start", _on_train_epoch_start)

    def optimizer_step(self):
        if getattr(self, "_dino_stage_a_active", False):
            base_model = unwrap_model(self.model)
            freeze_prefixes = tuple(getattr(self, "_dino_stage_a_prefixes", ()))
            for name, param in base_model.named_parameters():
                if any(name.startswith(pref) for pref in freeze_prefixes):
                    param.grad = None
        return super().optimizer_step()

    def __del__(self):
        hooks = getattr(self, "_dino_hooks", None)
        if not hooks:
            return
        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

