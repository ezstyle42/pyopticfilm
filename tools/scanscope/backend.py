# SPDX-License-Identifier: GPL-3.0-or-later
"""Scanner-facing control surface for the ScanScope GUI.

Numpy/Qt-free: device discovery, warmup/scan orchestration, and 16-bit TIFF
export. The Qt layer stays a thin shell over this module, mirroring how
``tools/scanlab`` splits ``backend``/``worker`` from ``app``.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pyopticfilm import ScanImage, Scanner
from pyopticfilm.device.select import model_for_device, model_is_scan_ready
from pyopticfilm.exceptions import PlustekError
from pyopticfilm.image import save_gray16_tiff, save_rgb16_tiff
from pyopticfilm.scan.bringup import (
    bringup_scan_geometry,
    crop_scan_geometry,
    image_crop_to_scan_area,
)
from pyopticfilm.usb.device import find_devices

Progress = Callable[[float], None]
CropNorm = tuple[float, float, float, float]


def invert_u16(rgb: np.ndarray) -> np.ndarray:
    """Invert a film negative to a positive (16-bit RGB)."""
    arr = np.asarray(rgb)
    if arr.dtype != np.uint16:
        raise ValueError(f"invert_u16 expects uint16, got {arr.dtype}")
    return (65535 - arr).astype(np.uint16)


def auto_level_u16(rgb: np.ndarray) -> np.ndarray:
    """Per-channel 1-99% percentile stretch, matching the preview's look.

    Same percentile policy as ``tools.scanlab.preview.auto_level_u8`` (probe
    on a subsampled copy, full-res stretch), but in 16-bit so the saved TIFF
    has the film-base colour removed like the GUI preview shows it.
    """
    arr = np.asarray(rgb)
    probe = arr[::8, ::8] if min(arr.shape[:2]) >= 32 else arr
    out = np.empty_like(arr)
    for c in range(arr.shape[2]):
        lo, hi = np.percentile(probe[:, :, c].astype(np.float32), (1.0, 99.0))
        plane = arr[:, :, c].astype(np.float32)
        if hi <= lo:
            out[:, :, c] = 0
        else:
            scaled = (plane - lo) * (65535.0 / (hi - lo))
            out[:, :, c] = np.clip(scaled, 0, 65535).astype(np.uint16)
    return out


#: Scan-mode combinations a consumer UI needs (README "Common scan-mode
#: combinations"). Manual exposure overrides stay in the Scan Lab.
MODE_NAMES = (
    "Single-Pass",
    "Multi-Pass",
    "Adaptive Multi-Exposure",
    "Adaptive Multi-Pass",
)


@dataclass(frozen=True)
class DeviceEntry:
    """One connected Plustek scanner offered in the device combo box."""

    device_id: str
    label: str
    scan_ready: bool
    model_name: str = ""


@dataclass
class ScanRequest:
    """One queued scan: PPI, mode, crop, IR flag."""

    dpi: int
    mode_name: str
    infrared: bool
    crop_norm: CropNorm | None = None


@dataclass
class ScanResult:
    """Deliverables of one finished scan request."""

    image: ScanImage
    tiff_path: Path | None = None
    ir_path: Path | None = None
    jpeg_path: Path | None = None
    error: str | None = None


def list_connected_devices() -> list[DeviceEntry]:
    """Enumerate plugged-in Plustek scanners, scan-ready ones first."""
    entries: list[DeviceEntry] = []
    for info in find_devices():
        try:
            model = model_for_device(info.product_id, info.bcd_device)
        except Exception:  # noqa: BLE001 - unknown revision still enumerable
            model = None
        scan_ready = bool(model and model_is_scan_ready(model))
        name = (model.model if model else None) or info.product or info.device_id
        tag = "scan-ready" if scan_ready else "probe-only"
        entries.append(
            DeviceEntry(
                device_id=info.device_id,
                label=f"{name} ({info.device_id}) — {tag}",
                scan_ready=scan_ready,
                model_name=model.model if model else "",
            )
        )
    entries.sort(key=lambda e: (not e.scan_ready, e.device_id))
    return entries


def mode_kwargs(mode_name: str, n_passes: int = 1) -> dict:
    """Map a mode name from ``MODE_NAMES`` to ``Scanner.scan`` kwargs."""
    if mode_name not in MODE_NAMES:
        raise ValueError(f"Unknown scan mode: {mode_name!r}")
    multi_exposure = mode_name.startswith("Adaptive")
    if multi_exposure:
        passes = 1 if mode_name == "Adaptive Multi-Exposure" else max(2, n_passes)
    else:
        passes = 1 if mode_name == "Single-Pass" else max(2, n_passes)
    return {"multi_exposure": multi_exposure, "n_passes": passes}


def scan_kwargs(model, request: ScanRequest) -> dict:
    """``Scanner.scan`` kwargs for a scan-ready model and crop, motor-safe."""
    kwargs: dict = {"resolution": request.dpi, "mode": "color"}
    kwargs.update(mode_kwargs(request.mode_name))
    if request.infrared:
        kwargs["infrared"] = True
    if request.crop_norm is not None:
        area = image_crop_to_scan_area(model, request.crop_norm)
        geometry, _meta = crop_scan_geometry(model, request.dpi, area)
        kwargs["geometry"] = geometry
    else:
        geometry, _meta = bringup_scan_geometry(model, request.dpi, profile="preview_safe")
        kwargs["geometry"] = geometry
    return kwargs


def default_output_dir() -> Path:
    return Path("scan_out")


def output_paths(out_dir: Path, index: int) -> tuple[Path, Path]:
    """Colour and IR TIFF paths for the n-th scan in a session."""
    stem = f"frame_{index:04d}"
    return out_dir / f"{stem}.tif", out_dir / f"{stem}_ir.tif"


def unique_path(path: Path) -> Path:
    """Return ``path`` unchanged, or with an appended counter if it exists."""
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for n in range(1, 10000):
        candidate = path.with_name(f"{stem}_{n}{suffix}")
        if not candidate.exists():
            return candidate
    raise PlustekError(f"Could not find a free filename for {path}")


def save_jpeg(rgb: np.ndarray, path: Path) -> Path:
    """Write an 8-bit JPEG copy of the processed 16-bit RGB image."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise PlustekError("JPEG export requires Pillow") from exc
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise PlustekError(f"rgb must be HxWx3, got shape {arr.shape}")
    if arr.dtype == np.uint16:
        u8 = (arr >> 8).astype(np.uint8)
    else:
        u8 = np.clip(arr, 0, 255).astype(np.uint8)
    out = path.with_suffix(".jpg")
    Image.fromarray(u8, mode="RGB").save(out, "JPEG", quality=95)
    return out


class ScannerSession:
    """Owns one open scanner for the GUI's lifetime.

    USB I/O must stay off the GUI thread: the worker drives ``warmup`` /
    ``prescan`` / ``scan`` and only the finished results cross back to Qt.
    """

    def __init__(self, device_id: str) -> None:
        self._scanner = Scanner.open(device_id)
        self._lock = threading.Lock()

    @property
    def model(self):
        return self._scanner.model

    @property
    def scanner(self) -> Scanner:
        return self._scanner

    def warmup(self, on_status: Callable[[str], None] | None = None) -> None:
        with self._lock:
            if on_status is not None:
                on_status("Homing carriage, warming lamp…")
            self._scanner.warmup(home=True, lamp=True)
            if on_status is not None:
                on_status("Ready.")

    def prescan(self, progress: Progress | None = None) -> ScanImage:
        from tools.scanlab.backend import prescan_resolution

        with self._lock:
            geometry, _meta = bringup_scan_geometry(
                self._scanner.model, prescan_resolution(self.model), profile="preview_safe"
            )
            return self._scanner.scan(
                resolution=prescan_resolution(self.model),
                mode="color",
                geometry=geometry,
                progress=progress,
            )

    def scan(
        self,
        request: ScanRequest,
        *,
        progress: Progress | None = None,
        cancel: threading.Event | None = None,
    ) -> ScanImage:
        kwargs = scan_kwargs(self.model, request)
        with self._lock:
            return self._scanner.scan(progress=progress, cancel=cancel, **kwargs)

    def save(
        self,
        image: ScanImage,
        out_dir: Path,
        index: int,
        *,
        negative: bool = True,
        auto_level: bool = True,
    ) -> ScanResult:
        """Write the deliverable TIFFs.

        ``negative`` inverts film to a positive first; ``auto_level`` then
        applies the preview's percentile stretch on the (inverted) data.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        colour_path, ir_path = output_paths(out_dir, index)
        rgb = image.rgb
        if negative:
            rgb = invert_u16(rgb)
        if auto_level:
            rgb = auto_level_u16(rgb)
        colour_path = unique_path(colour_path)
        save_rgb16_tiff(rgb, colour_path, dpi=image.dpi)
        jpeg_path = save_jpeg(rgb, colour_path)
        if image.ir is not None:
            ir_path = unique_path(ir_path)
            save_gray16_tiff(image.ir, ir_path, dpi=image.dpi)
        return ScanResult(
            image=image,
            tiff_path=colour_path,
            ir_path=ir_path if image.ir is not None else None,
            jpeg_path=jpeg_path,
        )

    def lamp(self, on: bool, on_status: Callable[[str], None] | None = None) -> None:
        with self._lock:
            (self._scanner.lamp_on if on else self._scanner.lamp_off)()
        if on_status is not None:
            on_status(f"Lamp {'on' if on else 'off'}.")

    def close(self) -> None:
        try:
            self._scanner.close()
        except Exception:  # noqa: BLE001, S110 - closing must not raise into Qt
            pass
