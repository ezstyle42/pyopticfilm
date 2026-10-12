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
    result = session.save(image, tmp_path / "out", 1, negative=False, auto_level=False)
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


def test_invert_u16_flips_range():
    import numpy as np

    from tools.scanscope.backend import invert_u16

    rgb = np.array([[[0, 32768, 65535]]], dtype=np.uint16)
    out = invert_u16(rgb)
    assert out.dtype == np.uint16
    assert out[0, 0].tolist() == [65535, 32767, 0]


def test_session_save_negative_inverts_before_level(monkeypatch, tmp_path):
    import numpy as np

    import tools.scanscope.backend as backend_mod
    from tools.scanscope.backend import ScannerSession

    saved = {}
    monkeypatch.setattr(
        backend_mod,
        "save_rgb16_tiff",
        lambda rgb, path, *, dpi: saved.update(rgb=rgb) or path,
    )
    session = ScannerSession.__new__(ScannerSession)
    image = MagicMock()
    image.ir = None
    image.dpi = 1800
    image.rgb = np.full((4, 4, 3), 30000, dtype=np.uint16)
    session.save(image, tmp_path / "out", 1, negative=True, auto_level=False)
    assert np.array_equal(saved["rgb"], 65535 - image.rgb)

    session.save(image, tmp_path / "out", 2, negative=False, auto_level=False)
    assert np.array_equal(saved["rgb"], image.rgb)


def test_main_window_constructs_with_all_widgets(monkeypatch):
    try:
        from PyQt6.QtWidgets import QApplication
    except (ImportError, OSError) as exc:
        pytest.skip(f"Qt native libs unavailable: {exc}")

    import tools.scanscope.app as app_mod

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(app_mod, "list_connected_devices", list)
    window = app_mod.MainWindow()
    for attr in (
        "device_combo",
        "refresh_btn",
        "warmup_btn",
        "lamp_btn",
        "ppi_combo",
        "mode_combo",
        "npasses_spin",
        "ir_check",
        "negative_check",
        "autolevel_check",
        "out_label",
        "out_btn",
        "preview",
        "prescan_btn",
        "scan_btn",
        "cancel_btn",
        "clear_crop_btn",
        "reset_btn",
        "status_label",
    ):
        assert hasattr(window, attr), f"MainWindow missing widget: {attr}"
    assert window.negative_check.isChecked()
    assert window.autolevel_check.isChecked()
    assert not window.ir_check.isChecked()
    window.on_reset_settings()
    assert str(window.out_dir) == "scan_out"


def test_unique_path_appends_counter(tmp_path):
    from tools.scanscope.backend import unique_path

    first = tmp_path / "frame_0001.tif"
    assert unique_path(first) == first
    first.touch()
    second = unique_path(first)
    assert second == tmp_path / "frame_0001_1.tif"
    second.touch()
    assert unique_path(first) == tmp_path / "frame_0001_2.tif"


def test_save_jpeg_writes_8bit_copy(tmp_path):
    np = pytest.importorskip("numpy")
    from tools.scanscope.backend import save_jpeg

    rgb = np.full((4, 6, 3), 65535, dtype=np.uint16)
    rgb[0, :, :] = 0
    out = save_jpeg(rgb, tmp_path / "frame_0001.tif")
    assert out == tmp_path / "frame_0001.jpg"
    assert out.exists()
    from PIL import Image

    with Image.open(out) as im:
        assert im.format == "JPEG"
        assert im.mode == "RGB"
        assert im.size == (6, 4)
        assert abs(im.getpixel((0, 0))[0] - 0) <= 2
        assert abs(im.getpixel((5, 3))[0] - 255) <= 2
