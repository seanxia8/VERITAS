# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Deprecated shim: the LUCiD front-end noise now lives in ``src/noise_module_lucid/``.

Kept only so older scripts that do ``from pmt_frontend_v2 import ...`` keep
working (e.g. ``docs/reviews/plot_pmt_frontend_v2_final.py``). New code should
``import noise_module_lucid`` and read ``docs/noise_module_lucid.md``. The
implementation is not duplicated here.
"""
from noise_module_lucid import *  # noqa: F401,F403
from noise_module_lucid import __all__ as _all

__all__ = list(_all)
