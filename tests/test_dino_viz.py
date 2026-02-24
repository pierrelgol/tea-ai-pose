from __future__ import annotations

import cv2
import numpy as np

from detector_train.dino_viz import normalize_map, save_dino_visualizations


def test_normalize_map_handles_non_finite_and_constant_values() -> None:
    const = np.full((4, 4), 7.0, dtype=np.float32)
    out_const = normalize_map(const)
    assert out_const.shape == (4, 4)
    assert np.allclose(out_const, 0.0)

    bad = np.array([[np.nan, np.inf], [-np.inf, 5.0]], dtype=np.float32)
    out_bad = normalize_map(bad)
    assert out_bad.shape == (2, 2)
    assert np.isfinite(out_bad).all()
    assert float(out_bad.min()) >= 0.0
    assert float(out_bad.max()) <= 1.0


def test_save_dino_visualizations_writes_overlay_pngs(tmp_path) -> None:
    rng = np.random.default_rng(123)
    images = rng.random((2, 3, 32, 32), dtype=np.float32)
    teacher = rng.random((2, 16, 8, 8), dtype=np.float32)
    signal_map = rng.random((2, 8, 8), dtype=np.float32)
    obj_mask = np.zeros((2, 8, 8), dtype=np.float32)
    obj_mask[:, 2:6, 2:6] = 1.0
    snapshot = {"images": images, "teacher": teacher, "signal_map": signal_map, "obj_mask": obj_mask}

    out = save_dino_visualizations(
        snapshot=snapshot,
        output_dir=tmp_path / "viz",
        max_samples=2,
    )

    files = out["files"]
    assert len(files) == 4
    for f in files:
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape[0] == 32
        assert img.shape[1] == 32
