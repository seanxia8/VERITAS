#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Round-2 frozen-checkpoint repair probe CLI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jepa_noise.ext_r2_repair_probes import main  # noqa: E402


if __name__ == "__main__":
    main()
