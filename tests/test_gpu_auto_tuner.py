from __future__ import annotations

from gpu_auto_tuner.search import binary_search_max_feasible
from gpu_auto_tuner.system import signature_from_parts


def test_signature_from_parts_is_stable() -> None:
    a = signature_from_parts(
        name="NVIDIA RTX 4090",
        total_vram_mb=24564,
        compute_capability="8.9",
        driver_version="550.54.14",
    )
    b = signature_from_parts(
        name="NVIDIA RTX 4090",
        total_vram_mb=24564,
        compute_capability="8.9",
        driver_version="550.54.14",
    )
    assert a == b
    assert len(a) == 16


def test_binary_search_max_feasible_finds_boundary() -> None:
    res = binary_search_max_feasible(low=1, high=32, is_feasible=lambda x: x <= 19)
    assert res.best_value == 19
    assert res.attempts


def test_binary_search_returns_none_when_no_value_feasible() -> None:
    res = binary_search_max_feasible(low=3, high=7, is_feasible=lambda _: False)
    assert res.best_value is None
