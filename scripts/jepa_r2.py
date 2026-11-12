#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Round-2 CLI (runbook docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md §4.1).

Modes: dry-run, preflight --detector, generate --detector, pack --detector,
smoke --detector --arm, train --detector --arm --seed, evaluate --detector, report.
Thin wrapper over :mod:`jepa_noise.ext_r2`.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jepa_noise.ext_r2 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
