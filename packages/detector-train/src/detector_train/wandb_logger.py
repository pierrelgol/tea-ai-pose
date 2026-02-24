from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Number
from typing import Any


@dataclass(slots=True)
class WandbState:
    enabled: bool
    mode_used: str | None
    run_id: str | None
    error: str | None


def _try_init(mode: str, init_kwargs: dict[str, Any]):
    import wandb

    return wandb.init(mode=mode, **init_kwargs)


def init_wandb(
    enabled: bool,
    mode: str,
    project: str,
    entity: str | None,
    run_name: str,
    tags: list[str],
    notes: str | None,
    config: dict[str, Any],
    log_system_metrics: bool = False,
):
    if not enabled:
        return None, WandbState(enabled=False, mode_used=None, run_id=None, error=None)

    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "entity": entity,
        "tags": tags,
        "notes": notes,
        "config": config,
        "reinit": "finish_previous",
    }
    if not log_system_metrics:
        try:
            import wandb

            init_kwargs["settings"] = wandb.Settings(_disable_stats=True)
        except Exception:
            pass

    try:
        if mode == "auto":
            try:
                run = _try_init("online", init_kwargs)
                return run, WandbState(enabled=True, mode_used="online", run_id=getattr(run, "id", None), error=None)
            except Exception as exc:
                run = _try_init("offline", init_kwargs)
                return run, WandbState(
                    enabled=True,
                    mode_used="offline",
                    run_id=getattr(run, "id", None),
                    error=f"online init failed, fell back to offline: {exc}",
                )

        run = _try_init(mode, init_kwargs)
        return run, WandbState(enabled=True, mode_used=mode, run_id=getattr(run, "id", None), error=None)
    except Exception as exc:
        return None, WandbState(enabled=False, mode_used=None, run_id=None, error=str(exc))


def log_wandb(run, payload: dict, step: int | None = None) -> None:
    if run is None:
        return
    clean = sanitize_payload(payload)
    if not clean:
        return
    try:
        if step is None:
            run.log(clean)
        else:
            run.log(clean, step=step)
    except Exception:
        pass


def sanitize_payload(payload: dict[str, Any]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or key == "":
            continue
        if isinstance(value, bool):
            clean[key] = float(int(value))
            continue
        if isinstance(value, Number):
            v = float(value)
            if math.isfinite(v):
                clean[key] = v
    return clean


def finish_wandb(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
