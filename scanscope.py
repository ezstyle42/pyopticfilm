#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""ScanScope launcher: self-installing frontend for Plustek OpticFilm scanners.

Run this single script on Linux Mint (or any Linux):

    python3 scanscope.py

On first run it creates a private virtualenv at ``.venv`` (Python on Mint is
externally managed, so nothing is installed system-wide), installs the
pyopticfilm driver from this repository plus PyQt6/tifffile, checks that
libusb can see your scanner, and then opens the ScanScope GUI. Subsequent
runs are fast: existing environments are reused and only missing or
outdated requirements are installed.

Optional:

    python3 scanscope.py --recreate   throw the venv away and rebuild it
    python3 scanscope.py --no-gui     set up only, do not launch the GUI
    python3 scanscope.py --udev       install the udev access rule (sudo)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
REQUIREMENTS = ("PyQt6>=6.6", "tifffile>=2024.2")
PLUSTEK_VID = 0x07B3
UDEV_RULE = (
    "# Plustek OpticFilm scanners - allow local users libusb access\n"
    'SUBSYSTEM=="usb", ATTR{idVendor}=="07b3", MODE="0666"\n'
)
UDEV_PATH = Path("/etc/udev/rules.d/99-plustek-opticfilm.rules")


def log(message: str) -> None:
    print(f"[scanscope] {message}", flush=True)


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    log(" ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=False, **kwargs)


def create_venv() -> Path:
    if VENV_DIR.exists():
        return VENV_DIR
    log(f"Creating virtual environment at {VENV_DIR}")
    venv.create(VENV_DIR, with_pip=True)
    return VENV_DIR


def venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def pip_install(*args: str) -> None:
    result = run(
        [venv_python(), "-m", "pip", "install", "--disable-pip-version-check", *args],
        cwd=ROOT,
    )
    if result.returncode != 0:
        sys.exit("pip install failed; check your internet connection and retry.")


def ensure_environment() -> None:
    create_venv()
    pip_install("--upgrade", "pip")
    pip_install("-e", str(ROOT))
    pip_install(*REQUIREMENTS)


def check_libusb_backend() -> bool:
    probe = "import usb.core, usb.backend.libusb1 as l1;assert l1.get_backend() is not None"
    result = subprocess.run(
        [str(venv_python()), "-c", probe],
        capture_output=True,
        check=False,
        text=True,
        cwd=ROOT,
    )
    if result.returncode == 0:
        return True
    log("libusb 1.0 backend not found.")
    print("Install it first, then rerun this script:\n    sudo apt install libusb-1.0-0")
    return False


def check_scanner_visible() -> bool:
    probe = f"import usb.core;devs = usb.core.find(find_all=True, idVendor={PLUSTEK_VID});print(sum(1 for _ in devs))"
    result = subprocess.run(
        [str(venv_python()), "-c", probe],
        capture_output=True,
        check=False,
        text=True,
        cwd=ROOT,
    )
    if result.returncode != 0:
        return False
    return result.stdout.strip() not in ("", "0")


def install_udev_rule() -> None:
    if not shutil.which("sudo"):
        sys.exit("sudo not found; install the rule manually (see README).")
    if UDEV_PATH.exists() and UDEV_PATH.read_text() == UDEV_RULE:
        log(f"udev rule already present at {UDEV_PATH}")
        return
    log(f"Installing udev rule to {UDEV_PATH} (needs your sudo password)")
    result = subprocess.run(
        ["sudo", "tee", str(UDEV_PATH)],
        input=UDEV_RULE,
        check=False,
        text=True,
    )
    if result.returncode != 0:
        sys.exit("Could not write the udev rule.")
    subprocess.run(["sudo", "udevadm", "control", "--reload-rules"], check=False)
    subprocess.run(["sudo", "udevadm", "trigger", "--subsystem-match=usb"], check=False)
    log("udev rule installed. Unplug and replug the scanner if it stays hidden.")


def check_gui_imports() -> bool:
    probe = "import PyQt6.QtWidgets, numpy; import tools.scanscope.app"
    result = subprocess.run(
        [str(venv_python()), "-c", probe],
        capture_output=True,
        check=False,
        text=True,
        cwd=ROOT,
    )
    if result.returncode == 0:
        return True
    print(result.stderr)
    print(
        "The GUI could not load. If the error above mentions libGL or other\n"
        "missing system libraries, install them and rerun:\n"
        "    sudo apt install libgl1 libegl1 libxkbcommon0 libdbus-1-3 libfontconfig1"
    )
    return False


def launch_gui() -> int:
    if not check_gui_imports():
        return 1
    log("Starting ScanScope GUI")
    result = subprocess.run(
        [str(venv_python()), "-m", "tools.scanscope"],
        check=False,
        cwd=ROOT,
    )
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recreate", action="store_true", help="rebuild the venv")
    parser.add_argument("--no-gui", action="store_true", help="set up only, no GUI")
    parser.add_argument("--udev", action="store_true", help="install the udev rule and exit")
    args = parser.parse_args()

    if args.recreate and VENV_DIR.exists():
        log(f"Removing {VENV_DIR}")
        shutil.rmtree(VENV_DIR)

    if args.udev:
        install_udev_rule()
        return 0

    ensure_environment()

    if not check_libusb_backend():
        return 1

    if not check_scanner_visible():
        print(
            "\nNo Plustek scanner is visible over USB right now.\n"
            "Check the cable, then either run:\n"
            "    python3 scanscope.py --udev\n"
            "or install the rule yourself (README 'USB access'):\n"
            f"    echo '{UDEV_RULE.strip()}' | sudo tee {UDEV_PATH}\n"
            "then replug the scanner. Continuing anyway — you can press\n"
            "Refresh inside the GUI once the scanner shows up.\n"
        )

    if args.no_gui:
        log("Setup complete (--no-gui).")
        return 0

    return launch_gui()


if __name__ == "__main__":
    raise SystemExit(main())
