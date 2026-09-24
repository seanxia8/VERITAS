# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Units bridge, preset invariants and the P0.3 co-calibration."""
from __future__ import annotations

import numpy as np
import pytest

from noise_module import NoiseGenerator
from noise_module_lucid import (
    FRONT_END_BANDWIDTH_RATIO,
    FRONT_END_CORNER_HZ,
    FS_L,
    PMT_FRONTEND_V2,
    PMT_PRIVATE,
    PMT_SHARED,
    RMS_MV,
    SPE_BANDWIDTH_HZ,
    bandwidth_report,
    charge_to_mv,
    spe_bandwidth_hz,
    spe_template,
    to_mv,
)


def test_spe_template_peak_and_shape():
    spe = spe_template()
    assert spe.max() == pytest.approx(4.0)
    assert spe[0] == 0.0
    assert np.all(spe >= 0.0)
    assert len(spe) == 60


def test_spe_bandwidth_and_derived_corner():
    bw = spe_bandwidth_hz()
    assert bw == pytest.approx(SPE_BANDWIDTH_HZ)
    assert 45e6 < bw < 60e6                      # WCTE-like 3" PMT, ~51 MHz
    assert FRONT_END_CORNER_HZ == pytest.approx(FRONT_END_BANDWIDTH_RATIO * bw)


def test_to_mv_preserves_length_and_convolves():
    wf = np.zeros((3, 512))
    wf[0, 10] = 1.0
    out = to_mv(wf)
    assert out.shape == (3, 512)
    assert out[0, 10:70].max() == pytest.approx(4.0)
    assert np.allclose(out[1], 0.0)


def test_charge_to_mv_is_to_mv():
    wf = np.zeros((2, 128))
    assert np.allclose(charge_to_mv(wf), to_mv(wf))


def test_preset_total_power_and_rolloff():
    N = 512
    f, S = NoiseGenerator(PMT_FRONTEND_V2, seed=0).build_psd_density(N)
    df = FS_L / N
    assert np.sum(S[1:]) * df == pytest.approx(RMS_MV**2, rel=1e-6)
    # the 4th-order floor rolls off hard; the ringing bump is above the corner
    assert S[-1] / S[1] < 0.02


def test_bandwidth_consistency_is_bounded():
    rep = bandwidth_report()
    assert rep["bandwidth_ratio"] == pytest.approx(FRONT_END_BANDWIDTH_RATIO)
    assert rep["noise_above_3x_signal_fraction"] < 0.2


def test_preset_components_are_private_or_shared():
    names = {c["name"] for c in PMT_PRIVATE} | {c["name"] for c in PMT_SHARED}
    assert "amplifier_floor" in names and "clock_62.5MHz" in names
