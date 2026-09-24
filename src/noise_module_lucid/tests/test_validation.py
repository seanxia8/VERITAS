# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Physics-validation helpers (P0.3, P1.1, P2.3)."""
from __future__ import annotations

import numpy as np
import pytest

from noise_module_lucid import (
    bandwidth_report,
    grouping_report,
    grouping_sensitivity,
    realized_csd_check,
)


def test_bandwidth_report_keys():
    rep = bandwidth_report()
    assert rep["signal_f3db_hz"] > 0.0
    assert rep["bandwidth_ratio"] == pytest.approx(2.5)
    assert 0.0 <= rep["noise_above_3x_signal_fraction"] < 0.2


def test_realized_csd_matches_implied():
    # P2.3: the realized Welch coherence must follow the implied rho_ij(f)
    rep = realized_csd_check(N=1 << 16, C=2, seed=7)
    assert rep["peak_frequency_hz"] == pytest.approx(6.25e7, rel=0.05)
    assert rep["rho_implied_peak"] == pytest.approx(rep["rho_realized_peak"], abs=0.1)
    assert rep["median_abs_diff"] < 0.1


def test_grouping_sensitivity_reports_compactness():
    rep = grouping_sensitivity(C=64, group_size=16)
    assert set(rep) == {"contiguous", "z_plane", "proximity"}
    for method, info in rep.items():
        assert info["n_groups"] == 4
        assert np.isfinite(info["compactness"]) or info["compactness"] == float("inf")


def test_grouping_report_compact_groups():
    pos = np.array([[0.0, 0, 0], [0.1, 0, 0], [10, 0, 0], [10.1, 0, 0]])
    groups = [np.array([0, 1]), np.array([2, 3])]
    rep = grouping_report(pos, groups)
    assert rep["mean_intra_distance_m"] == pytest.approx(0.1, abs=1e-6)
    assert rep["mean_centroid_distance_m"] == pytest.approx(10.0, abs=1e-6)
