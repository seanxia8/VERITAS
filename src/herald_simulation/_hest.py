# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""Import shim for the pinned HeST clone (``fetch_hest.sh``) — never patched, only imported."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

EXTERNAL = Path(__file__).resolve().parent / "external" / "HeST"
COMMIT_FILE = Path(__file__).resolve().parent / "external" / "hest_commit.txt"


class HeSTUnavailable(ImportError):
    pass


def _load() -> tuple[ModuleType, ModuleType, ModuleType]:
    candidates = [EXTERNAL, Path(os.environ.get("HEST_PATH", ""))]
    for root in candidates:
        if root and (root / "HeST" / "core").is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        core = importlib.import_module("HeST.core.HeST_Core")
        det = importlib.import_module("HeST.core.Detection")
        geo = importlib.import_module("HeST.core.Geometry")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise HeSTUnavailable(
            "HeST is not importable. Run src/herald_simulation/fetch_hest.sh and "
            "`pip install qetpy numba` (HeST_Core imports qetpy.utils at module scope)."
        ) from exc
    return core, det, geo


def hest_commit() -> str | None:
    if COMMIT_FILE.exists():
        return COMMIT_FILE.read_text().strip()
    return None


def available() -> bool:
    try:
        _load()
        return True
    except HeSTUnavailable:
        return False
