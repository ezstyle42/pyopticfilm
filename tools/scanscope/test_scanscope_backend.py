# SPDX-License-Identifier: GPL-3.0-or-later
"""ScanScope backend: mode mapping and scan kwargs (no Qt / no display)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from pyopticfilm.device.model_8200i_se import MODEL_8200I_SE
from tools.scanscope.backend import (
    MODE_NAMES,
    ScanRequest,
    mode_kwargs,
    output_paths,
    scan_kwargs,
)


def test_mode_kwargs_covers_readme_combinations():
    assert mode_kwargs("Single-Pass") == {"multi_exposure": False, "n_passes": 1}
    assert mode_kwargs("Multi-Pass", n_passes=4) == {
        "multi_exposure": False,
        "n_passes": 4,
    }
    assert mode_kwargs("Adaptive Multi-Exposure") == {
        "multi_exposure": True,
        "n_passes": 1,
    }
    assert mode_kwargs("Adaptive Multi-Pass", n_passes=4) == {
        "multi_exposure": True,
        "n_passes": 4,
    }


def test_mode_kwargs_rejects_unknown_mode():
    with pytest.raises(ValueError):
        mode_kwargs("Turbo")


def test_scan_kwargs_full_frame_8200i_se():
    request = ScanRequest(
        dpi=1800,
        mode_name="Adaptive Multi-Pass",
        infrared=True,
        crop_norm=None,
    )
    kwargs = scan_kwargs(MODEL_8200I_SE, request)
    assert kwargs["resolution"] == 1800
    assert kwargs["mode"] == "color"
    assert kwargs["multi_exposure"] is True
    assert kwargs["infrared"] is True
    assert kwargs["n_passes"] == 2
    assert kwargs["geometry"] is not None


def test_scan_kwargs_crop_uses_crop_geometry():
    request = ScanRequest(
        dpi=2400,
        mode_name="Single-Pass",
        infrared=False,
        crop_norm=(0.1, 0.1, 0.9, 0.9),
    )
    kwargs = scan_kwargs(MODEL_8200I_SE, request)
    assert kwargs["n_passes"] == 1
    assert kwargs["geometry"] is not None


def test_scan_kwargs_without_ir_omits_flag():
    request = ScanRequest(dpi=1800, mode_name="Single-Pass", infrared=False, crop_norm=None)
    assert "infrared" not in scan_kwargs(MODEL_8200I_SE, request)


def test_output_paths_pair_colour_and_ir(tmp_path):
    colour, ir = output_paths(tmp_path, 3)
    assert colour == tmp_path / "frame_0003.tif"
    assert ir == tmp_path / "frame_0003_ir.tif"


def test_session_save_writes_tiffs(tmp_path, monkeypatch):
    import numpy as np

    import tools.scanscope.backend as backend_mod
    from tools.scanscope.backend import ScannerSession

    saved = {}
    monkeypatch.setattr(
        backend_mod,
        "save_rgb16_tiff",
        lambda rgb, path, *, dpi: saved.update(rgb=rgb, path=path, dpi=dpi) or path,
    )
    session = ScannerSession.__new__(ScannerSession)
    image = MagicMock()
    image.ir = None
    image.dpi = 1800
    image.rgb = np.full((4, 4, 3), 30000, dtype=np.uint16)
    result = session.save(image, tmp_path / "out", 1, auto_level=False)
    assert result.tiff_path == tmp_path / "out" / "frame_0001.tif"
    assert saved["path"] == result.tiff_path
    assert saved["dpi"] == 1800
    assert np.array_equal(saved["rgb"], image.rgb)
    assert result.ir_path is None


def test_session_save_auto_levels_default(monkeypatch, tmp_path):
    import numpy as np

    import tools.scanscope.backend as backend_mod
    from tools.scanscope.backend import ScannerSession

    calls = {}
    real = backend_mod.auto_level_u16
    monkeypatch.setattr(
        backend_mod,
        "save_rgb16_tiff",
        lambda rgb, path, *, dpi: calls.update(rgb=rgb) or path,
    )
    monkeypatch.setattr(backend_mod, "auto_level_u16", real)
    session = ScannerSession.__new__(ScannerSession)
    image = MagicMock()
    image.ir = None
    image.dpi = 1800
    image.rgb = np.full((4, 4, 3), 30000, dtype=np.uint16)
    session.save(image, tmp_path / "out", 2)
    assert not np.array_equal(calls["rgb"], image.rgb)


def test_mode_names_match_readme():
    assert MODE_NAMES == (
        "Single-Pass",
        "Multi-Pass",
        "Adaptive Multi-Exposure",
        "Adaptive Multi-Pass",
    )


def test_auto_level_u16_removes_colour_cast_and_clips():
    import numpy as np

    from tools.scanscope.backend import auto_level_u16

    rng = np.random.default_rng(7)
    rgb = rng.integers(10000, 40000, size=(64, 48, 3), dtype=np.uint16)
    rgb[:, :, 0] += 12000
    rgb[:, :, 2] = rgb[:, :, 2] // 2
    out = auto_level_u16(rgb)
    assert out.dtype == np.uint16 and out.shape == rgb.shape
    flat = out.reshape(-1, 3).astype(np.float32)
    assert flat[:, 0].min() <= 655.0
    assert flat[:, 0].max() >= 64800.0
    means = flat.mean(axis=0)
    assert abs(means[0] - means[1]) < 8000.0
    assert abs(means[1] - means[2]) < 8000.0


def test_auto_level_u16_flat_channel_is_zero():
    import numpy as np

    from tools.scanscope.backend import auto_level_u16

    rgb = np.full((16, 16, 3), 12345, dtype=np.uint16)
    out = auto_level_u16(rgb)
    assert out.max() == 0
