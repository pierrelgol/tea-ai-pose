from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
from typing import Any

from pipeline_config import build_layout, load_pipeline_config
from pipeline_runtime_utils import resolve_device

from .runner import TrialResult, run_trial_via_detector_train
from .search import binary_search_max_feasible
from .system import build_gpu_signature


@dataclass(slots=True)
class CandidateResult:
    params: dict[str, Any]
    batch: int
    coarse_trial: TrialResult
    throughput_proxy: float


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(config_root: Path, raw: str) -> Path:
    p = Path(raw)
    if not p.is_absolute():
        p = (config_root / p).resolve()
    return p


def _count_train_images(payload: dict[str, Any], config_path: Path) -> int:
    config_root = config_path.parent
    dataset_root = _resolve_path(config_root, str(payload["paths"]["dataset_root"]))
    dataset_name = str(payload["dataset"]["name"])
    augmented = str(payload["dataset"].get("augmented_subdir", "augmented"))
    train_dir = dataset_root / augmented / dataset_name / "images" / "train"
    if not train_dir.exists():
        return 0
    count = 0
    for p in train_dir.iterdir():
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            count += 1
    return count


def _resolve_augmented_dataset_root(payload: dict[str, Any], config_path: Path) -> Path:
    config_root = config_path.parent
    dataset_root = _resolve_path(config_root, str(payload["paths"]["dataset_root"]))
    dataset_name = str(payload["dataset"]["name"])
    augmented = str(payload["dataset"].get("augmented_subdir", "augmented"))
    return dataset_root / augmented / dataset_name


def _assert_dataset_ready(payload: dict[str, Any], config_path: Path) -> None:
    ds_root = _resolve_augmented_dataset_root(payload, config_path)
    train_dir = ds_root / "images" / "train"
    val_dir = ds_root / "images" / "val"
    if train_dir.exists() and val_dir.exists():
        return
    raise RuntimeError(
        "tuner dataset is not ready for detector-train. "
        f"expected directories: {train_dir} and {val_dir}. "
        "run `just generate-dataset` (and dataset fetch steps if needed), "
        "or point `tuner.dataset` to an existing augmented dataset."
    )


def _safe_list(raw: Any, fallback: list[Any]) -> list[Any]:
    if isinstance(raw, list) and raw:
        return raw
    return fallback


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _trial_feasible(*, trial: TrialResult, peak_limit_mb: float) -> bool:
    if not trial.success or trial.oom:
        return False
    if trial.peak_vram_mb is None:
        return False
    return float(trial.peak_vram_mb) <= float(peak_limit_mb)


def _throughput_proxy(*, batch: int, imgsz: int, epochs: int, elapsed_s: float, train_images: int) -> float:
    if elapsed_s <= 0:
        return 0.0
    samples = max(1, train_images) * max(1, epochs)
    pixels = float(max(1, imgsz) * max(1, imgsz))
    return float((samples * max(1, batch) * pixels) / elapsed_s)


def _build_search_space(tuner: dict[str, Any], train: dict[str, Any]) -> list[dict[str, Any]]:
    imgsz_values = [int(v) for v in _safe_list(tuner.get("imgsz_candidates"), [int(train.get("imgsz", 512))])]
    workers_values = [int(v) for v in _safe_list(tuner.get("workers_candidates"), [int(train.get("workers", 8))])]
    cache_values = [str(v) for v in _safe_list(tuner.get("cache_candidates"), [str(train.get("cache", "disk"))])]
    amp_values = [bool(v) for v in _safe_list(tuner.get("amp_candidates"), [bool(train.get("amp", True))])]
    tf32_values = [bool(v) for v in _safe_list(tuner.get("tf32_candidates"), [bool(train.get("tf32", True))])]
    cudnn_values = [bool(v) for v in _safe_list(tuner.get("cudnn_benchmark_candidates"), [bool(train.get("cudnn_benchmark", True))])]

    out: list[dict[str, Any]] = []
    for imgsz, workers, cache, amp, tf32, cudnn in itertools.product(
        imgsz_values,
        workers_values,
        cache_values,
        amp_values,
        tf32_values,
        cudnn_values,
    ):
        out.append(
            {
                "imgsz": int(imgsz),
                "workers": int(workers),
                "cache": str(cache),
                "amp": bool(amp),
                "tf32": bool(tf32),
                "cudnn_benchmark": bool(cudnn),
            }
        )
    return out


def _apply_tuned_values(payload: dict[str, Any], *, tuned_params: dict[str, Any], gpu_signature: str, tuned_by: str, report_path: Path) -> None:
    t = payload["train"]
    t.update(
        {
            "batch": int(tuned_params["batch"]),
            "batch_mode": "fixed",
            "imgsz": int(tuned_params["imgsz"]),
            "workers": int(tuned_params["workers"]),
            "cache": str(tuned_params["cache"]),
            "amp": bool(tuned_params["amp"]),
            "tf32": bool(tuned_params["tf32"]),
            "cudnn_benchmark": bool(tuned_params["cudnn_benchmark"]),
            "tuned_gpu_signature": gpu_signature,
            "tuned_at_utc": _utc_now_iso(),
            "tuned_by": tuned_by,
            "tuned_profile_path": str(report_path),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline GPU tuner that writes train settings back into config.json")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg_path = args.config.resolve()
    shared = load_pipeline_config(cfg_path)
    payload = _load_payload(cfg_path)

    resolved_device = resolve_device(str(shared.train.get("device", "auto")))
    if resolved_device in {"cpu", "mps"}:
        raise RuntimeError("gpu-auto-tuner requires CUDA device resolution")

    probe = build_gpu_signature(resolved_device)
    existing_sig = str(shared.train.get("tuned_gpu_signature") or "").strip()
    if existing_sig and existing_sig == probe.signature and not args.force:
        print(f"status: skipped")
        print(f"reason: config already tuned for gpu signature {probe.signature}")
        return

    tuner = shared.tuner
    train = shared.train

    coarse_epochs = int(tuner.get("coarse_epochs", 10))
    confirm_epochs = int(tuner.get("confirm_epochs", 20))
    vram_target = float(tuner.get("vram_target_utilization", 0.92))
    batch_min = int(tuner.get("batch_min", 1))
    batch_cap = int(min(int(tuner.get("batch_max_cap", train.get("batch_max", 64))), int(train.get("batch_max", 64))))
    max_trials = int(tuner.get("max_trials", 30))

    if batch_min < 1:
        batch_min = 1
    if batch_cap < batch_min:
        raise RuntimeError("invalid tuner batch range: batch_max_cap < batch_min")

    # Use small profile dataset for tuning runs.
    tune_dataset = str(tuner.get("dataset") or shared.profile.get("dataset") or shared.run["dataset"])
    payload["run"]["dataset"] = tune_dataset
    payload["dataset"]["name"] = tune_dataset

    train_image_count = _count_train_images(payload, cfg_path)
    vram_limit_mb = float(probe.total_vram_mb) * float(vram_target)

    search_space = _build_search_space(tuner, train)
    trial_records: list[dict[str, Any]] = []
    trial_counter = 0
    failed_oom = 0
    failed_non_oom = 0
    first_non_oom_tail: str | None = None

    best: CandidateResult | None = None

    _assert_dataset_ready(payload, cfg_path)

    for combo in search_space:
        if trial_counter >= max_trials:
            break

        attempts_to_trial: dict[int, TrialResult] = {}

        def _is_feasible(batch: int) -> bool:
            nonlocal trial_counter
            if trial_counter >= max_trials:
                return False
            trial_counter += 1
            overrides = {
                "epochs": coarse_epochs,
                "batch": int(batch),
                "batch_mode": "fixed",
                "workers_auto": False,
                "throughput_mode": "balanced",
                **combo,
            }
            trial = run_trial_via_detector_train(
                base_payload=payload,
                gpu_index=probe.index,
                config_path_hint=cfg_path,
                overrides=overrides,
            )
            attempts_to_trial[int(batch)] = trial
            trial_records.append(
                {
                    "phase": "coarse",
                    "params": dict(combo),
                    "batch": int(batch),
                    "result": asdict(trial),
                }
            )
            nonlocal failed_oom, failed_non_oom, first_non_oom_tail
            if trial.oom:
                failed_oom += 1
            elif not trial.success:
                failed_non_oom += 1
                if first_non_oom_tail is None:
                    first_non_oom_tail = (trial.stdout_tail or "").strip()[:800]
            return _trial_feasible(trial=trial, peak_limit_mb=vram_limit_mb)

        bs = binary_search_max_feasible(low=batch_min, high=batch_cap, is_feasible=_is_feasible)
        if bs.best_value is None:
            continue

        chosen_batch = int(bs.best_value)
        chosen_trial = attempts_to_trial.get(chosen_batch)
        if chosen_trial is None:
            continue

        score = _throughput_proxy(
            batch=chosen_batch,
            imgsz=int(combo["imgsz"]),
            epochs=coarse_epochs,
            elapsed_s=float(chosen_trial.elapsed_s),
            train_images=train_image_count,
        )

        candidate = CandidateResult(
            params=dict(combo),
            batch=chosen_batch,
            coarse_trial=chosen_trial,
            throughput_proxy=score,
        )

        if best is None:
            best = candidate
        else:
            if candidate.throughput_proxy > best.throughput_proxy:
                best = candidate
            elif candidate.throughput_proxy == best.throughput_proxy:
                c_util = candidate.coarse_trial.avg_gpu_util_percent or 0.0
                b_util = best.coarse_trial.avg_gpu_util_percent or 0.0
                if c_util > b_util:
                    best = candidate

    if best is None:
        details = (
            f"no feasible config found; tried {trial_counter} trials "
            f"(oom_failures={failed_oom}, non_oom_failures={failed_non_oom})."
        )
        if first_non_oom_tail:
            details += f" first_non_oom_error_tail: {first_non_oom_tail}"
        raise RuntimeError(details)

    # Confirm pass at longer horizon.
    confirm_overrides = {
        "epochs": confirm_epochs,
        "batch": int(best.batch),
        "batch_mode": "fixed",
        "workers_auto": False,
        "throughput_mode": "balanced",
        **best.params,
    }
    confirm_trial = run_trial_via_detector_train(
        base_payload=payload,
        gpu_index=probe.index,
        config_path_hint=cfg_path,
        overrides=confirm_overrides,
    )
    trial_records.append(
        {
            "phase": "confirm",
            "params": dict(best.params),
            "batch": int(best.batch),
            "result": asdict(confirm_trial),
        }
    )

    if not _trial_feasible(trial=confirm_trial, peak_limit_mb=vram_limit_mb):
        # Conservative fallback: lower batch by one and reconfirm once.
        fallback_batch = max(batch_min, best.batch - 1)
        fallback_overrides = {
            **confirm_overrides,
            "batch": int(fallback_batch),
        }
        fallback_trial = run_trial_via_detector_train(
            base_payload=payload,
            gpu_index=probe.index,
            config_path_hint=cfg_path,
            overrides=fallback_overrides,
        )
        trial_records.append(
            {
                "phase": "confirm_fallback",
                "params": dict(best.params),
                "batch": int(fallback_batch),
                "result": asdict(fallback_trial),
            }
        )
        if not _trial_feasible(trial=fallback_trial, peak_limit_mb=vram_limit_mb):
            raise RuntimeError("confirm phase failed to validate safe no-OOM settings")
        best = CandidateResult(
            params=dict(best.params),
            batch=int(fallback_batch),
            coarse_trial=fallback_trial,
            throughput_proxy=best.throughput_proxy,
        )

    layout = build_layout(
        artifacts_root=shared.paths["artifacts_root"],
        model_key=str(shared.run["model_key"]),
        run_id=str(shared.run["run_id"]),
    )
    artifacts_subdir = str(tuner.get("artifacts_subdir", "tuner"))
    report_dir = layout.run_root / artifacts_subdir / probe.signature
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "tuner_report.json"

    report = {
        "status": "ok",
        "timestamp_utc": _utc_now_iso(),
        "gpu": asdict(probe),
        "limits": {
            "vram_target_utilization": vram_target,
            "vram_limit_mb": vram_limit_mb,
            "max_trials": max_trials,
            "coarse_epochs": coarse_epochs,
            "confirm_epochs": confirm_epochs,
        },
        "selected": {
            "batch": int(best.batch),
            **best.params,
        },
        "trials": trial_records,
    }
    _atomic_write_json(report_path, report)

    backup = cfg_path.with_name(f"{cfg_path.name}.bak.{datetime.now().strftime('%Y%m%d%H%M%S')}")
    shutil.copy2(cfg_path, backup)

    tuned_params = {
        "batch": int(best.batch),
        **best.params,
    }
    _apply_tuned_values(
        payload,
        tuned_params=tuned_params,
        gpu_signature=probe.signature,
        tuned_by="gpu-auto-tuner@0.1.0",
        report_path=report_path,
    )
    _atomic_write_json(cfg_path, payload)

    print("status: ok")
    print(f"gpu_signature: {probe.signature}")
    print(f"config_backup: {backup}")
    print(f"tuned_batch: {best.batch}")
    print(f"tuned_imgsz: {best.params['imgsz']}")
    print(f"tuned_workers: {best.params['workers']}")
    print(f"tuned_cache: {best.params['cache']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
