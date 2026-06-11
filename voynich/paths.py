"""
paths.py — where data and results live.

Running from a source checkout keeps the historical behaviour (./data,
./results, relative to the working directory) so tests and CLI workflows
are unchanged.  A frozen desktop build (PyInstaller) must not write next
to its executable, so it uses the platform's per-user application data
directory instead.  VOYNICH_DATA_DIR / VOYNICH_RESULTS_DIR override
everything.
"""

from __future__ import annotations

import os
import pathlib
import sys

APP_NAME = "VoynichWorkbench"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def user_data_root() -> pathlib.Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(pathlib.Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = str(pathlib.Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(
            pathlib.Path.home() / ".local" / "share"
        )
    return pathlib.Path(base) / APP_NAME


def default_data_dir() -> pathlib.Path:
    env = os.environ.get("VOYNICH_DATA_DIR")
    if env:
        return pathlib.Path(env)
    if is_frozen():
        return user_data_root() / "data"
    return pathlib.Path("data")


def default_results_dir() -> pathlib.Path:
    env = os.environ.get("VOYNICH_RESULTS_DIR")
    if env:
        return pathlib.Path(env)
    if is_frozen():
        return user_data_root() / "results"
    return pathlib.Path("results")
