from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from ultralytics import YOLO

from .config import OptimizeConfig


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _find_first(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def _assert_pose_model(model: YOLO, weights: Path) -> None:
    task = str(getattr(model, "task", "")).strip().lower()
    if task != "pose":
        raise RuntimeError(f"weights are not a pose model (task={task!r}): {weights}")


def _export_onnx(cfg: OptimizeConfig, *, device: str) -> Path:
    model = YOLO(str(cfg.weights))
    _assert_pose_model(model, cfg.weights)
    parent_before = {p.resolve() for p in cfg.weights.parent.glob("*.onnx")}
    result = model.export(
        format="onnx",
        imgsz=int(cfg.imgsz),
        batch=int(cfg.batch),
        dynamic=bool(cfg.dynamic),
        simplify=bool(cfg.simplify),
        opset=int(cfg.opset),
        half=bool(cfg.precision == "fp16"),
        device=str(device),
    )

    candidates: list[Path] = []
    if isinstance(result, (str, Path)):
        candidates.append(Path(str(result)))

    parent_after = {p.resolve() for p in cfg.weights.parent.glob("*.onnx")}
    new_paths = sorted(parent_after - parent_before, key=lambda p: p.stat().st_mtime, reverse=True)
    candidates.extend(new_paths)
    candidates.extend(sorted(parent_after, key=lambda p: p.stat().st_mtime, reverse=True))

    selected = _find_first(candidates)
    if selected is None:
        raise RuntimeError("failed to locate exported ONNX model")
    return selected.resolve()


def _build_engine_with_trtexec(*, onnx_path: Path, engine_path: Path, cfg: OptimizeConfig, log_path: Path) -> dict[str, Any]:
    cmd = [
        str(cfg.trtexec_bin),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={int(max(1, round(cfg.workspace_gb * 1024)))}",
        "--skipInference",
        "--builderOptimizationLevel=5",
        "--best",
    ]
    if cfg.precision == "fp16":
        cmd.append("--fp16")
    elif cfg.precision == "int8":
        cmd.append("--int8")

    proc = subprocess.run(cmd, capture_output=True, text=True)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    log_path.write_text(combined, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(
            "trtexec failed. Check log for details: "
            f"{log_path}\nexit_code={proc.returncode}"
        )
    if not engine_path.exists():
        raise RuntimeError(f"trtexec completed but engine was not created: {engine_path}")

    return {
        "backend": "trtexec",
        "command": cmd,
        "log_path": str(log_path),
        "exit_code": int(proc.returncode),
    }


def _device_supports_tensorrt(device: str) -> bool:
    d = str(device).strip().lower()
    if d in {"cpu", "mps"}:
        return False
    if d == "auto":
        return False
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


def _build_engine_with_ultralytics(*, cfg: OptimizeConfig) -> tuple[Path, dict[str, Any]]:
    model = YOLO(str(cfg.weights))
    _assert_pose_model(model, cfg.weights)
    parent_before = {p.resolve() for p in cfg.weights.parent.glob("*.engine")}
    result = model.export(
        format="engine",
        imgsz=int(cfg.imgsz),
        batch=int(cfg.batch),
        dynamic=bool(cfg.dynamic),
        half=bool(cfg.precision == "fp16"),
        int8=bool(cfg.precision == "int8"),
        workspace=float(cfg.workspace_gb),
        device=str(cfg.device),
    )

    candidates: list[Path] = []
    if isinstance(result, (str, Path)):
        candidates.append(Path(str(result)))

    parent_after = {p.resolve() for p in cfg.weights.parent.glob("*.engine")}
    new_paths = sorted(parent_after - parent_before, key=lambda p: p.stat().st_mtime, reverse=True)
    candidates.extend(new_paths)
    candidates.extend(sorted(parent_after, key=lambda p: p.stat().st_mtime, reverse=True))

    selected = _find_first(candidates)
    if selected is None:
        raise RuntimeError("failed to locate exported TensorRT engine")

    return selected.resolve(), {"backend": "ultralytics-export", "result": str(result)}


def optimize_detector(cfg: OptimizeConfig) -> dict[str, Any]:
    cfg.validate()

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    output_onnx = cfg.output_root / f"{cfg.model_key}_{cfg.run_name}.onnx"
    output_engine = cfg.output_root / f"{cfg.model_key}_{cfg.run_name}_{cfg.precision}.engine"
    log_path = cfg.output_root / "trtexec.log"

    if output_engine.exists():
        output_engine.unlink()

    backend: dict[str, Any] | None = None
    onnx_tmp: Path | None = None

    try:
        trt_supported = _device_supports_tensorrt(cfg.device)
        trtexec_available = shutil.which(cfg.trtexec_bin) is not None
        use_trtexec = bool(trt_supported and cfg.prefer_trtexec and trtexec_available)

        if not trt_supported:
            onnx_tmp = _export_onnx(cfg, device="cpu")
            shutil.copy2(onnx_tmp, output_onnx)
            raise RuntimeError(
                "TensorRT engine compilation requires a CUDA-visible GPU. "
                f"Resolved device='{cfg.device}', CUDA is unavailable, so ONNX was exported on CPU. "
                "Re-run on a CUDA host to build .engine."
            )

        if use_trtexec:
            onnx_tmp = _export_onnx(cfg, device=str(cfg.device))
            shutil.copy2(onnx_tmp, output_onnx)
            backend = _build_engine_with_trtexec(
                onnx_path=output_onnx,
                engine_path=output_engine,
                cfg=cfg,
                log_path=log_path,
            )
        else:
            engine_tmp, backend = _build_engine_with_ultralytics(cfg=cfg)
            shutil.copy2(engine_tmp, output_engine)
            onnx_sidecar = engine_tmp.with_suffix(".onnx")
            if cfg.save_onnx and onnx_sidecar.exists():
                shutil.copy2(onnx_sidecar, output_onnx)
            if onnx_sidecar.exists() and (not cfg.save_onnx or onnx_sidecar.resolve() != output_onnx.resolve()):
                try:
                    onnx_sidecar.unlink()
                except OSError:
                    pass
            if engine_tmp.exists() and engine_tmp.resolve() != output_engine.resolve():
                try:
                    engine_tmp.unlink()
                except OSError:
                    pass

        if not cfg.save_onnx and output_onnx.exists():
            output_onnx.unlink()

        engine_size_mb = float(output_engine.stat().st_size / (1024 * 1024)) if output_engine.exists() else None
        pt_size_mb = float(cfg.weights.stat().st_size / (1024 * 1024))
        shrink_ratio = (engine_size_mb / pt_size_mb) if (engine_size_mb is not None and pt_size_mb > 0) else None

        summary = {
            "status": "ok",
            "timestamp_utc": _utc_now_iso(),
            "config": asdict(cfg),
            "backend": backend,
            "artifacts": {
                "weights_pt": str(cfg.weights),
                "onnx": str(output_onnx) if output_onnx.exists() else None,
                "engine": str(output_engine),
                "trtexec_log": str(log_path) if log_path.exists() else None,
            },
            "sizes_mb": {
                "weights_pt": pt_size_mb,
                "engine": engine_size_mb,
                "engine_to_pt_ratio": shrink_ratio,
            },
        }
    except Exception as exc:
        summary = {
            "status": "error",
            "timestamp_utc": _utc_now_iso(),
            "config": asdict(cfg),
            "error": str(exc),
            "backend": backend,
            "artifacts": {
                "weights_pt": str(cfg.weights),
                "onnx": str(output_onnx) if output_onnx.exists() else None,
                "engine": str(output_engine) if output_engine.exists() else None,
                "trtexec_log": str(log_path) if log_path.exists() else None,
            },
        }

    summary_path = cfg.output_root / "optimize_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
