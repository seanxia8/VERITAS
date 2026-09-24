# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
from __future__ import annotations

import pytest

import herald_simulation as hs

requires_hest = pytest.mark.skipif(not hs.available(), reason="HeST not importable: run fetch_hest.sh and pip install qetpy numba")
