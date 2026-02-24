from __future__ import annotations

import argparse
from pathlib import Path

from pipeline_config import build_layout, load_pipeline_config
from pipeline_runtime_utils import resolve_device, resolve_latest_weights_from_artifacts

from .config import OptimizeConfig
from .optimizer import optimize_detector


def _resolve_weights(*, shared, layout, explicit_weights: Path | None) -> Path:
    if explicit_weights is not None:
        return explicit_weights.resolve()

    run_weights = [
        layout.train_ultralytics_root / str(shared.run["run_id"]) / "weights" / "best_geo.pt",
        layout.train_ultralytics_root / str(shared.run["run_id"]) / "weights" / "best.pt",
    ]
    for p in run_weights:
        if p.exists():
            return p.resolve()

    return resolve_latest_weights_from_artifacts(shared.paths["artifacts_root"]).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize trained detector weights and compile TensorRT engine")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--precision", choices=["fp16", "fp32", "int8"], default="fp16")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--keep-onnx", action="store_true")
    parser.add_argument("--force-ultralytics", action="store_true")
    parser.add_argument("--trtexec-bin", type=str, default="trtexec")
    args = parser.parse_args()

    shared = load_pipeline_config(args.config)

    model_key = str(shared.run["model_key"])
    run_id = str(shared.run["run_id"])
    layout = build_layout(
        artifacts_root=shared.paths["artifacts_root"],
        model_key=model_key,
        run_id=run_id,
    )

    weights = _resolve_weights(shared=shared, layout=layout, explicit_weights=args.weights)
    requested_device = str(args.device or shared.infer.get("device", shared.train.get("device", "0")))
    device = str(resolve_device(requested_device))
    imgsz = int(args.imgsz or shared.infer.get("imgsz", shared.train.get("imgsz", 640)))

    cfg = OptimizeConfig(
        weights=weights,
        output_root=layout.optimize_root,
        run_name=run_id,
        model_key=model_key,
        imgsz=imgsz,
        batch=int(args.batch),
        precision=str(args.precision),
        workspace_gb=float(args.workspace_gb),
        dynamic=bool(args.dynamic),
        device=device,
        trtexec_bin=str(args.trtexec_bin),
        prefer_trtexec=not bool(args.force_ultralytics),
        save_onnx=bool(args.keep_onnx),
    )

    summary = optimize_detector(cfg)

    print(f"status: {summary['status']}")
    print(f"weights: {cfg.weights}")
    print(f"device: {cfg.device}")
    print(f"output_root: {cfg.output_root}")
    print(f"summary: {summary['summary_path']}")
    if summary["status"] == "ok":
        print(f"engine: {summary['artifacts']['engine']}")
        print(f"backend: {summary['backend']['backend']}")
        sizes = summary.get("sizes_mb", {})
        print(f"weights_pt_mb: {sizes.get('weights_pt')}")
        print(f"engine_mb: {sizes.get('engine')}")
    else:
        print(f"error: {summary.get('error')}")


if __name__ == "__main__":
    main()
