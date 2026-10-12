# SPDX-License-Identifier: GPL-3.0-or-later
"""End-user OpticFilm scan application (Qt6).

A focused scan app for a plugged-in scan-ready scanner (OpticFilm 8200i SE,
OpticFilm 8100 V2): warmup, prescan, crop, scan, save 16-bit TIFFs. For the
bring-up lab surface (mock USB, HW-gate override, USB forensics) use
``tools.scanlab`` instead.

Run from the repository root::

    uv run --group lab python -m tools.scanscope
"""

from __future__ import annotations

from PyQt6.QtWidgets import QApplication

from tools.scanscope.app import main

if __name__ == "__main__":
    raise SystemExit(main(QApplication))
