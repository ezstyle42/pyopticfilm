# SPDX-License-Identifier: GPL-3.0-or-later
"""Background worker: runs scanner I/O off the GUI thread and emits results.

Mirrors ``tools/scanlab/worker.py``: a queued request model plus signals, so
the GUI thread never blocks on USB.
"""

from __future__ import annotations

import threading
import traceback

from PyQt6.QtCore import QObject, pyqtSignal

from pyopticfilm.exceptions import ScanCancelled
from tools.scanscope.backend import ScannerSession, ScanRequest


class ScannerWorker(QObject):
    """Bridge between the GUI thread and a :class:`ScannerSession`."""

    status_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(float)
    prescan_ready = pyqtSignal(object)
    scan_ready = pyqtSignal(object)
    failed = pyqtSignal(str)
    busy_changed = pyqtSignal(bool)

    def __init__(self, session: ScannerSession) -> None:
        super().__init__()
        self._session = session
        self._cancel = threading.Event()
        self._busy = False

    # --- GUI-thread API ----------------------------------------------------

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.busy_changed.emit(busy)

    @property
    def busy(self) -> bool:
        return self._busy

    def cancel(self) -> None:
        self._cancel.set()
        self.status_changed.emit("Cancelling…")

    def warmup(self) -> None:
        self._run(self._do_warmup)

    def prescan(self) -> None:
        self._run(self._do_prescan)

    def scan(self, request: ScanRequest) -> None:
        self._run(lambda: self._do_scan(request))

    # --- Worker-thread implementation --------------------------------------

    def _run(self, job) -> None:
        if self._busy:
            self.failed.emit("Scanner is busy.")
            return
        self.set_busy(True)
        self._cancel.clear()
        thread = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        thread.start()

    def _run_job(self, job) -> None:
        try:
            job()
        except ScanCancelled:
            self.status_changed.emit("Cancelled.")
        except Exception as exc:  # noqa: BLE001 - surfaced to the GUI
            traceback.print_exc()
            self.failed.emit(str(exc) or exc.__class__.__name__)
        finally:
            self.set_busy(False)

    def _do_warmup(self) -> None:
        self._session.warmup(on_status=self.status_changed.emit)
        self.status_changed.emit("Scanner warmed up.")

    def _do_prescan(self) -> None:
        self.status_changed.emit("Prescanning…")
        image = self._session.prescan(progress=self._emit_progress)
        self.prescan_ready.emit(image)

    def _do_scan(self, request: ScanRequest) -> None:
        self.status_changed.emit("Scanning…")
        image, full = self._session.scan(request, progress=self._emit_progress, cancel=self._cancel)
        self.status_changed.emit("Saving…")
        result = self._session.save(
            image,
            self._out_dir,
            self._index,
            negative=self._negative,
            auto_level=self._auto_level,
            full_frame=full,
        )
        self.scan_ready.emit(result)

    # --- Set per-request context from the GUI thread -----------------------

    def set_output(self, out_dir, index: int, *, negative: bool = True, auto_level: bool = True) -> None:
        self._out_dir = out_dir
        self._index = index
        self._negative = negative
        self._auto_level = auto_level

    def _emit_progress(self, fraction: float) -> None:
        self.progress_changed.emit(max(0.0, min(1.0, fraction)))
