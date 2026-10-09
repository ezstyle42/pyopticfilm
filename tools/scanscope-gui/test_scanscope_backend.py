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
    request = ScanRequest(
        dpi=1800, mode_name="Single-Pass", infrared=False, crop_norm=None
    )
    assert "infrared" not in scan_kwargs(MODEL_8200I_SE, request)


def test_output_paths_pair_colour_and_ir(tmp_path):
    colour, ir = output_paths(tmp_path, 3)
    assert colour == tmp_path / "frame_0003.tif"
    assert ir == tmp_path / "frame_0003_ir.tif"


def test_session_save_writes_tiffs(tmp_path, monkeypatch):
    from tools.scanscope.backend import ScannerSession

    session = ScannerSession.__new__(ScannerSession)
    image = MagicMock()
    image.ir = None
    result = session.save(image, tmp_path / "out", 1)
    assert result.tiff_path == tmp_path / "out" / "frame_0001.tif"
    image.save_tiff.assert_called_once_with(result.tiff_path)
    assert result.ir_path is None


def test_mode_names_match_readme():
    assert MODE_NAMES == (
        "Single-Pass",
        "Multi-Pass",
        "Adaptive Multi-Exposure",
        "Adaptive Multi-Pass",
    )
