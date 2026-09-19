# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Tests for the LUCiD front-end noise package (no LUCiD clone required)."""
from __future__ import annotations

import numpy as np
import pytest

from noise_module import NoiseGenerator

from noise_module_lucid import (
    GROUP,
    PMT_CRATE_V2,
    PMT_FRONTEND_V2,
    add_readout_noise,
    charge_to_mv,
    clock_scale,
    crate_noise,
    crate_preset,
    decimate_alias_fold,
    groups_from_positions,
    groups_from_string_id,
    kappa,
    spe_template,
)
from noise_module_lucid.grouping import contiguous_groups
from noise_module_lucid.presets import PMT_FRONTEND_LONG, preset_for

FS = 1.0e9


def test_spe_template_peak_is_mv_per_pe():
    spe = spe_template(mv_per_pe=4.0, length_ns=80.0)
    assert spe.max() == pytest.approx(4.0, rel=1e-9)
    assert spe[0] == pytest.approx(0.0)
    assert np.all(spe >= 0)


def test_charge_to_mv_unit_impulse():
    spe = spe_template()
    wf = np.zeros((1, 512)); wf[0, 0] = 1.0
    out = charge_to_mv(wf, spe)
    assert out.shape == wf.shape
    assert out.max() == pytest.approx(spe.max(), rel=1e-9)


def test_preset_psd_integrates_to_power():
    n = 512
    f, s, _ = NoiseGenerator(PMT_FRONTEND_V2, seed=0).build_psd_density(n, return_metadata=True)
    integral = float(np.sum(s[1:]) * (FS / n))
    assert integral == pytest.approx(PMT_FRONTEND_V2["noise_power"], rel=1e-6)


def test_long_preset_exists_and_integrates():
    n = 32768
    f, s, _ = NoiseGenerator(PMT_FRONTEND_LONG, seed=0).build_psd_density(n, return_metadata=True)
    integral = float(np.sum(s[1:]) * (FS / n))
    assert integral == pytest.approx(PMT_FRONTEND_LONG["noise_power"], rel=1e-6)


def test_preset_for_window():
    assert preset_for(512)[0] is PMT_FRONTEND_V2
    assert preset_for(32768)[0] is PMT_FRONTEND_LONG


def test_clock_scale_multiplies_line_scales():
    scaled = clock_scale(5.0)
    base_lines = [c for c in PMT_CRATE_V2["shared_components"] if c["type"] == "line"]
    new_lines = [c for c in scaled if c["type"] == "line"]
    assert len(base_lines) == len(new_lines)
    for b, n in zip(base_lines, new_lines):
        assert n["scale"] == pytest.approx(5.0 * b["scale"])


def test_groups_contiguous_and_string_id():
    g = contiguous_groups(10, 4)
    assert [len(x) for x in g] == [4, 4, 2]
    s = groups_from_string_id(np.array([0, 0, 1, 1, 2]))
    assert [list(x) for x in s] == [[0, 1], [2, 3], [4]]


def test_groups_from_positions_partition_all_channels():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(50, 3))
    groups = groups_from_positions(pos, n_sectors=4, n_bands=3)
    flat = np.sort(np.concatenate(groups))
    assert np.array_equal(flat, np.arange(50))


def test_crate_noise_metadata_and_kappa():
    base, crate = crate_preset()
    noise, meta = crate_noise(base, crate, 512, seed=0, n_channels=8)
    assert noise.shape == (8, 512)
    assert meta["implied_covariance"].shape == (8, 8)
    assert np.isfinite(kappa(meta))


def test_add_readout_noise_shapes_metadata_and_determinism():
    wf = np.zeros((GROUP, 512))
    trace, meta = add_readout_noise(wf, seed=3, window_ns=512)
    assert trace.shape == (GROUP, 512)
    assert len(meta["groups"]) == 1
    assert len(meta["kappa"]) == 1
    trace2, _ = add_readout_noise(wf, seed=3, window_ns=512)
    assert np.allclose(trace, trace2)


def test_add_readout_noise_in_mv_skips_convolution():
    wf = np.zeros((4, 512))
    trace, meta = add_readout_noise(wf, in_units="mv", group_size=4, seed=1)
    assert trace.shape == (4, 512)
    assert meta["spe_mv_per_pe"] is None


def test_alias_fold_decimation_shape():
    wf = np.zeros((2, 512))
    trace, _ = add_readout_noise(wf, in_units="mv", group_size=2, seed=0)
    dec, fs_new = decimate_alias_fold(trace, factor=4)
    assert dec.shape == (2, 128)
    assert fs_new == FS / 4
