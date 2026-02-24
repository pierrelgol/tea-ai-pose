from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
import time

from .system import GpuMonitor


@dataclass(slots=True)
class TrialResult:
    success: bool
    oom: bool
    return_code: int
    elapsed_s: float
    peak_vram_mb: float | None
    avg_gpu_util_percent: float | None
    stdout_tail: str
    trial_config_path: Path


def _deep_copy_json(payload: dict) -> dict:
    return json.loads(json.dumps(payload))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_trial_via_detector_train(
    *,
    base_payload: dict,
    gpu_index: int,
    config_path_hint: Path,
    overrides: dict,
) -> TrialResult:
    trial_payload = _deep_copy_json(base_payload)

    # Tuner controls train knobs directly and disables lock enforcement during probe runs.
    trial_payload["tuner"]["enabled"] = False
    trial_payload["train"].update(overrides)

    # Offline probe run settings.
    trial_payload["train"]["wandb_enabled"] = False
    trial_payload["train"]["plots"] = False
    trial_payload["train"]["dino_viz_enabled"] = False
    trial_payload["train"]["eval_enabled"] = False
    trial_payload["train"]["periodic_eval_mode"] = "off"
    trial_payload["train"]["wandb_mode"] = "offline"
    # Keep path resolution stable even when trial config lives in a temp directory.
    root = config_path_hint.parent.resolve()
    for key in ("dataset_root", "artifacts_root", "configs_root", "targets_source_root"):
        raw = trial_payload["paths"].get(key)
        if raw is None:
            continue
        p = Path(str(raw))
        if not p.is_absolute():
            p = (root / p).resolve()
        trial_payload["paths"][key] = str(p)

    # Keep each trial isolated.
    trial_payload["run"]["run_id"] = f"tune-{int(time.time() * 1000)}-{overrides.get('batch', 'x')}"

    tmp_dir = Path(tempfile.mkdtemp(prefix="tea-ai-tuner-", dir=str(config_path_hint.parent)))
    trial_cfg_path = tmp_dir / "trial_config.json"
    _write_json(trial_cfg_path, trial_payload)

    cmd = ["uv", "run", "detector-train", "--config", str(trial_cfg_path)]
    monitor = GpuMonitor(index=gpu_index, interval_s=0.5)
    monitor.start()
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(config_path_hint.parent), capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    sample = monitor.stop()

    tail = "\n".join((proc.stdout or "").splitlines()[-60:])
    err = (proc.stderr or "")
    msg = f"{tail}\n{err}".lower()
    oom = "out of memory" in msg and "cuda" in msg

    return TrialResult(
        success=proc.returncode == 0,
        oom=oom,
        return_code=int(proc.returncode),
        elapsed_s=float(elapsed),
        peak_vram_mb=sample.peak_vram_mb,
        avg_gpu_util_percent=sample.avg_util_percent,
        stdout_tail=tail,
        trial_config_path=trial_cfg_path,
    )
