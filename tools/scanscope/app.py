# SPDX-License-Identifier: GPL-3.0-or-later
"""ScanScope main window: device picker, prescan crop, scan, save."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QRect, QSettings, Qt
from PyQt6.QtGui import QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from tools.scanlab.preview import auto_level_u8, downsample_for_display
from tools.scanscope.backend import (
    MODE_NAMES,
    DeviceEntry,
    ScannerSession,
    ScanRequest,
    default_output_dir,
    list_connected_devices,
)
from tools.scanscope.worker import ScannerWorker

SETTINGS_ORG = "pyopticfilm"
SETTINGS_APP = "scanscope"
DEFAULT_SETTINGS = {
    "out_dir": "scan_out",
    "ppi": 1800,
    "mode": MODE_NAMES[0],
    "n_passes": 2,
    "infrared": False,
    "negative": True,
    "auto_level": True,
    "frame_index": 1,
}


def rgb_to_qimage(rgb: np.ndarray) -> QImage:
    """Display-ready QImage from a uint16 HxWx3 array (auto-levelled)."""
    small = downsample_for_display(rgb)
    u8 = auto_level_u8(small)
    height, width = u8.shape[:2]
    return QImage(u8.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


def gray_to_qimage(plane: np.ndarray) -> QImage:
    """Grayscale QImage from a uint16 HxW plane (auto-levelled)."""
    small = downsample_for_display(plane)
    if small.ndim == 3:
        small = small[:, :, 0]
    lo, hi = np.percentile(small.astype(np.float32), (1.0, 99.0))
    if hi <= lo:
        lo, hi = 0.0, 1.0
    u8 = np.clip((small.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255)
    u8 = np.ascontiguousarray(u8.astype(np.uint8))
    height, width = u8.shape[:2]
    return QImage(u8.data, width, height, width, QImage.Format.Format_Grayscale8).copy()


class PreviewLabel(QLabel):
    """Prescan image with a draggable crop rectangle in normalised coords.

    The image is letterboxed inside the widget (KeepAspectRatio), so widget
    coordinates must be mapped through the displayed-image rectangle. The
    stored crop is normalised against the image itself, making it stable
    across widget resizes and exact at the moment the drag ends.
    """

    def __init__(self) -> None:
        super().__init__()
        self._pixmap = None
        self._image_rect: QRect | None = None
        self._crop: tuple[float, float, float, float] | None = None
        self._drag_start: tuple[float, float] | None = None
        self._drag_cur: tuple[float, float] | None = None
        self.setMinimumSize(400, 300)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_image(self, image: QImage) -> None:
        self._pixmap = image
        self._image_rect = None
        self._repaint()

    def _displayed_rect(self) -> QRect | None:
        """The letterboxed rect the scaled image occupies in the widget."""
        if self._pixmap is None:
            return None
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        return QRect(
            (self.width() - scaled.width()) // 2,
            (self.height() - scaled.height()) // 2,
            scaled.width(),
            scaled.height(),
        )

    def crop(self) -> tuple[float, float, float, float] | None:
        return self._crop

    def clear_crop(self) -> None:
        self._crop = None
        self._drag_start = None
        self._drag_cur = None
        self._repaint()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._image_rect = None
            self._repaint()

    def _widget_to_image(self, x: float, y: float) -> tuple[float, float]:
        """Clamp a widget point into the image, as image-pixel coordinates."""
        rect = self._displayed_rect()
        if rect is None:
            return 0.0, 0.0
        x = max(rect.x(), min(float(rect.x() + rect.width()), x))
        y = max(rect.y(), min(float(rect.y() + rect.height()), y))
        return x - rect.x(), y - rect.y()

    def _image_to_widget(self, x: float, y: float) -> tuple[float, float]:
        """Image-pixel coordinates back to widget coordinates (for drawing)."""
        rect = self._displayed_rect()
        if rect is None:
            return 0.0, 0.0
        return rect.x() + x, rect.y() + y

    def mousePressEvent(self, event) -> None:
        self._drag_start = (*self._widget_to_image(event.position().x(), event.position().y()),)
        self._drag_cur = self._drag_start
        self._repaint()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is not None:
            self._drag_cur = self._widget_to_image(event.position().x(), event.position().y())
            self._repaint()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_start is None or self._drag_cur is None:
            return
        iw, ih = float(self._pixmap.width()), float(self._pixmap.height())
        x1, y1 = self._drag_start
        x2, y2 = self._drag_cur
        if abs(x2 - x1) * abs(y2 - y1) < 100:
            self.clear_crop()
            return
        self._crop = (
            max(0.0, min(x1, x2) / iw),
            max(0.0, min(y1, y2) / ih),
            min(1.0, max(x1, x2) / iw),
            min(1.0, max(y1, y2) / ih),
        )
        self._drag_start = None
        self._drag_cur = None
        self._repaint()

    def _repaint(self) -> None:
        if self._pixmap is None:
            return
        canvas = QPixmap.fromImage(self._pixmap).scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        rect = self._displayed_rect()
        if rect is None:
            return
        painter = QPainter(canvas)
        painter.setPen(QPen(Qt.GlobalColor.red, 2))
        if self._drag_start is not None and self._drag_cur is not None:
            x1, y1 = self._drag_start
            x2, y2 = self._drag_cur
            painter.drawRect(QRect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        elif self._crop is not None:
            iw, ih = float(self._pixmap.width()), float(self._pixmap.height())
            cx1, cy1, cx2, cy2 = self._crop
            painter.drawRect(
                QRect(
                    int(cx1 * iw),
                    int(cy1 * ih),
                    int((cx2 - cx1) * iw),
                    int((cy2 - cy1) * ih),
                )
            )
        painter.end()
        self.setPixmap(canvas)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ScanScope — Plustek OpticFilm")
        self.settings = QSettings(SETTINGS_ORG, SETTINGS_APP)
        self.devices: list[DeviceEntry] = []
        self.session: ScannerSession | None = None
        self.worker: ScannerWorker | None = None
        self.frame_index = 1
        self.out_dir = default_output_dir()

        central = QWidget()
        layout = QVBoxLayout(central)

        device_row = QGridLayout()
        self.device_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.refresh_devices)
        self.warmup_btn = QPushButton("Warmup")
        self.warmup_btn.clicked.connect(self.on_warmup)
        self.lamp_btn = QPushButton("Lamp on/off")
        self.lamp_btn.clicked.connect(self.on_lamp)
        device_row.addWidget(QLabel("Device"), 0, 0)
        device_row.addWidget(self.device_combo, 0, 1)
        device_row.addWidget(self.refresh_btn, 0, 2)
        device_row.addWidget(self.warmup_btn, 0, 3)
        device_row.addWidget(self.lamp_btn, 0, 4)
        layout.addLayout(device_row)

        params = QGridLayout()
        self.ppi_combo = QComboBox()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(MODE_NAMES)
        self.npasses_spin = QSpinBox()
        self.npasses_spin.setRange(2, 9)
        self.npasses_spin.setValue(2)
        self.ir_check = QCheckBox("Infrared dust plane")
        self.negative_check = QCheckBox("Negative film (invert to positive)")
        self.negative_check.setChecked(True)
        self.autolevel_check = QCheckBox("Auto-level saved TIFF (remove film colour)")
        self.autolevel_check.setChecked(True)
        self.out_label = QLabel(str(self.out_dir))
        self.out_btn = QPushButton("Choose folder…")
        self.out_btn.clicked.connect(self.on_choose_out)
        params.addWidget(QLabel("PPI"), 0, 0)
        params.addWidget(self.ppi_combo, 0, 1)
        params.addWidget(QLabel("Mode"), 1, 0)
        params.addWidget(self.mode_combo, 1, 1)
        params.addWidget(QLabel("Passes"), 1, 2)
        params.addWidget(self.npasses_spin, 1, 3)
        params.addWidget(self.ir_check, 1, 4)
        params.addWidget(self.negative_check, 2, 0, 1, 2)
        params.addWidget(self.autolevel_check, 2, 2, 1, 3)
        params.addWidget(self.out_btn, 0, 2)
        params.addWidget(self.out_label, 0, 3)
        layout.addLayout(params)

        self.preview = PreviewLabel()
        self.preview.setStyleSheet("border: 1px solid #888")
        layout.addWidget(self.preview, stretch=1)

        actions = QGridLayout()
        self.prescan_btn = QPushButton("Prescan")
        self.prescan_btn.clicked.connect(self.on_prescan)
        self.scan_btn = QPushButton("Scan")
        self.scan_btn.clicked.connect(self.on_scan)
        self.scan_btn.setStyleSheet("font-weight: bold")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.on_cancel)
        self.clear_crop_btn = QPushButton("Clear crop")
        self.clear_crop_btn.clicked.connect(self.preview.clear_crop)
        actions.addWidget(self.prescan_btn, 0, 0)
        actions.addWidget(self.clear_crop_btn, 0, 1)
        actions.addWidget(self.scan_btn, 0, 2)
        actions.addWidget(self.cancel_btn, 0, 3)
        layout.addLayout(actions)

        self.reset_btn = QPushButton("Reset settings")
        self.reset_btn.clicked.connect(self.on_reset_settings)
        actions.addWidget(self.reset_btn, 0, 4)

        self.status_label = QLabel("Connect a scanner and press Refresh.")
        layout.addWidget(self.status_label)

        self.setCentralWidget(central)

        self._load_settings()
        self._connect_settings_changed()

        self.refresh_devices()

    # --- Persistent settings -------------------------------------------------

    def _load_settings(self) -> None:
        """Apply persisted settings to the widgets (safe against bad values)."""
        out_dir = str(self.settings.value("out_dir", DEFAULT_SETTINGS["out_dir"]))
        self.out_dir = Path(out_dir)
        self.out_label.setText(out_dir)

        mode = str(self.settings.value("mode", DEFAULT_SETTINGS["mode"]))
        pos = self.mode_combo.findText(mode)
        self.mode_combo.setCurrentIndex(max(pos, 0))

        self.npasses_spin.setValue(int(self.settings.value("n_passes", DEFAULT_SETTINGS["n_passes"])))
        self.ir_check.setChecked(self.settings.value("infrared", DEFAULT_SETTINGS["infrared"], type=bool))
        self.negative_check.setChecked(self.settings.value("negative", DEFAULT_SETTINGS["negative"], type=bool))
        self.autolevel_check.setChecked(self.settings.value("auto_level", DEFAULT_SETTINGS["auto_level"], type=bool))
        self.frame_index = int(self.settings.value("frame_index", DEFAULT_SETTINGS["frame_index"]))

    def _connect_settings_changed(self) -> None:
        self.mode_combo.currentTextChanged.connect(lambda mode: self.settings.setValue("mode", mode))
        self.npasses_spin.valueChanged.connect(lambda n: self.settings.setValue("n_passes", n))
        self.ir_check.toggled.connect(lambda on: self.settings.setValue("infrared", on))
        self.negative_check.toggled.connect(lambda on: self.settings.setValue("negative", on))
        self.autolevel_check.toggled.connect(lambda on: self.settings.setValue("auto_level", on))
        self.ppi_combo.currentIndexChanged.connect(lambda: self.settings.setValue("ppi", self.ppi_combo.currentData()))

    def _save_frame_index(self) -> None:
        self.settings.setValue("frame_index", self.frame_index)

    def on_reset_settings(self) -> None:
        """Restore every option to its default."""
        self.settings.clear()
        self.out_dir = default_output_dir()
        self.out_label.setText(str(self.out_dir))
        self.mode_combo.setCurrentIndex(0)
        self.npasses_spin.setValue(DEFAULT_SETTINGS["n_passes"])
        self.ir_check.setChecked(DEFAULT_SETTINGS["infrared"])
        self.negative_check.setChecked(DEFAULT_SETTINGS["negative"])
        self.autolevel_check.setChecked(DEFAULT_SETTINGS["auto_level"])
        self.frame_index = DEFAULT_SETTINGS["frame_index"]
        pos = self.ppi_combo.findData(DEFAULT_SETTINGS["ppi"])
        if pos >= 0:
            self.ppi_combo.setCurrentIndex(pos)
        self.status_label.setText("Settings reset to defaults.")

    # --- Device management ---------------------------------------------------

    def refresh_devices(self) -> None:
        self.devices = list_connected_devices()
        self.device_combo.clear()
        for entry in self.devices:
            self.device_combo.addItem(entry.label)
        if not self.devices:
            self.status_label.setText("No Plustek OpticFilm scanners found.")
            self.ppi_combo.clear()
            return
        self.device_combo.setCurrentIndex(0)
        self._open_selected()

    def _open_selected(self) -> None:
        index = self.device_combo.currentIndex()
        if index < 0 or index >= len(self.devices):
            return
        entry = self.devices[index]
        if not entry.scan_ready:
            self.status_label.setText(
                f"{entry.label} is not scan-ready in this release; "
                "only the OpticFilm 8200i SE and OpticFilm 8100 (V2) can scan."
            )
            self.ppi_combo.clear()
            return
        session = ScannerSession(entry.device_id)
        self.session = session
        self.worker = ScannerWorker(session)
        self._connect_worker(self.worker)
        self.ppi_combo.clear()
        for dpi in sorted(session.model.resolutions_dpi):
            self.ppi_combo.addItem(str(dpi), dpi)
        saved_ppi = int(self.settings.value("ppi", DEFAULT_SETTINGS["ppi"]))
        pos = self.ppi_combo.findData(saved_ppi)
        if pos < 0:
            pos = max(0, self.ppi_combo.findData(DEFAULT_SETTINGS["ppi"]))
        self.ppi_combo.setCurrentIndex(pos)
        self.ir_check.setEnabled(bool(getattr(session.model, "supports_infrared", False)))
        self.status_label.setText(f"{session.model.model} open. Press Warmup.")

    def _connect_worker(self, worker: ScannerWorker) -> None:
        worker.status_changed.connect(self.status_label.setText)
        worker.progress_changed.connect(self.on_progress)
        worker.prescan_ready.connect(self.on_prescan_ready)
        worker.scan_ready.connect(self.on_scan_ready)
        worker.failed.connect(self.on_failed)
        worker.busy_changed.connect(self.set_busy)

    def on_warmup(self) -> None:
        if self.worker:
            self.worker.warmup()

    def on_lamp(self) -> None:
        if self.session:
            self.session.lamp(not self._lamp_on)

    _lamp_on = False

    def on_choose_out(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Output folder", str(self.out_dir))
        if chosen:
            self.out_dir = Path(chosen)
            self.out_label.setText(chosen)
            self.settings.setValue("out_dir", chosen)

    # --- Prescan / scan --------------------------------------------------------

    def on_prescan(self) -> None:
        if self.worker:
            self.worker.prescan()

    def on_scan(self) -> None:
        if not (self.worker and self.session):
            return
        dpi = self.ppi_combo.currentData()
        if dpi is None:
            self.status_label.setText("Pick a supported PPI first.")
            return
        request = ScanRequest(
            dpi=dpi,
            mode_name=self.mode_combo.currentText(),
            infrared=self.ir_check.isChecked(),
            crop_norm=self.preview.crop(),
        )
        self.worker.set_output(
            self.out_dir,
            self.frame_index,
            negative=self.negative_check.isChecked(),
            auto_level=self.autolevel_check.isChecked(),
        )
        self.worker.scan(request)

    def on_cancel(self) -> None:
        if self.worker:
            self.worker.cancel()

    # --- Worker results ---------------------------------------------------------

    def on_progress(self, fraction: float) -> None:
        self.status_label.setText(f"Scanning… {int(fraction * 100)}%")

    def on_prescan_ready(self, image) -> None:
        from tools.scanscope.backend import invert_u16

        rgb = invert_u16(image.rgb) if self.negative_check.isChecked() else image.rgb
        self.preview.set_image(rgb_to_qimage(rgb))
        self.status_label.setText("Prescan done. Drag a crop or press Scan.")

    def on_scan_ready(self, result) -> None:
        self.frame_index += 1
        self._save_frame_index()
        if result.ir_path is not None:
            msg = f"Saved {result.tiff_path} (+ IR {result.ir_path.name})"
        else:
            msg = f"Saved {result.tiff_path}"
        self.status_label.setText(msg)

    def on_failed(self, message: str) -> None:
        self.status_label.setText(f"Error: {message}")
        QMessageBox.warning(self, "ScanScope", message)

    def set_busy(self, busy: bool) -> None:
        for btn in (self.prescan_btn, self.scan_btn, self.warmup_btn, self.refresh_btn):
            btn.setEnabled(not busy)
        self.cancel_btn.setEnabled(busy)

    def closeEvent(self, event) -> None:
        if self.session:
            self.session.close()
        super().closeEvent(event)


def main(app_cls) -> int:
    app = app_cls(sys.argv)
    window = MainWindow()
    window.resize(900, 700)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(QApplication))
