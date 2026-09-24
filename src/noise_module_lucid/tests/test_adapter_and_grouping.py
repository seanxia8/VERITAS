# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Adapter, kappa floor, channel gains and channel grouping."""
from __future__ import annotations

import numpy as np
import pytest

from noise_module import MultiChannelNoiseGenerator
from noise_module_lucid import (
    GROUP,
    add_pmt_noise,
    apply_channel_gains,
    channel_groups,
    crate_preset,
    kappa,
    long_window_preset,
    matched_cell_kappa_floor,
)


def test_add_pmt_noise_shape_and_metadata():
    sig = np.zeros((GROUP, 512))
    out, groups = add_pmt_noise(sig, seed=0)
    assert out.shape == (GROUP, 512)
    assert len(groups) == 1
    assert groups[0]["implied_covariance"].shape == (GROUP, GROUP)
    assert groups[0]["signal_gain_applied"] is False


def test_channel_gains_are_applied_and_pinned():
    sig = np.ones((GROUP, 512))
    gains = np.linspace(0.8, 1.2, GROUP)
    out, groups = add_pmt_noise(sig, seed=0, channel_gains=gains)
    assert groups[0]["signal_gain_applied"] is True
    assert np.allclose(groups[0]["gains"], gains)      # pinned into the noise too
    assert np.allclose(apply_channel_gains(sig, gains), gains[:, None])
    # the signal part is scaled by the gains: the mean of the channel means
    # tracks mean(gains) plus the (small, zero-mean) noise contribution
    assert out.mean(axis=1).mean() == pytest.approx(gains.mean(), abs=0.2)


def test_matched_cell_kappa_floor_nc_rule():
    assert matched_cell_kappa_floor(512, C=64) > 4.0
    assert matched_cell_kappa_floor(32768, C=64) < 1.6


def test_long_window_preset_builds():
    base, crate = long_window_preset(32000.0)
    names = [c["name"] for c in base["components"]]
    assert any("dc_dc" in n for n in names)
    gen = MultiChannelNoiseGenerator(base, {**crate, "n_channels": 8}, seed=1)
    _, m = gen.generate(32000, return_metadata=True)
    assert kappa(m) > 1.0


def test_channel_groups_contiguous():
    groups = channel_groups(70, 64)
    assert [len(g) for g in groups] == [64, 6]


def test_channel_groups_z_plane_orders_by_z():
    pos = np.column_stack([np.zeros(4), np.zeros(4), [3.0, 1.0, 4.0, 2.0]])
    groups = channel_groups(4, 2, positions=pos, method="z_plane")
    assert np.array_equal(groups[0], np.array([1, 3]))
    assert np.array_equal(groups[1], np.array([0, 2]))


def test_channel_groups_board_map():
    board_ids = np.array([0, 0, 1, 1, 2, 2])
    groups = channel_groups(6, 8, method="board_map", board_ids=board_ids)
    assert len(groups) == 3
    assert np.array_equal(groups[0], np.array([0, 1]))


def test_channel_groups_proximity_is_spatially_compact():
    pos = np.array([[0.0, 0, 0], [0.1, 0, 0], [10, 0, 0], [10.1, 0, 0]])
    groups = channel_groups(4, 2, positions=pos, method="proximity")
    assert set(groups[0]) == {0, 1}
    assert set(groups[1]) == {2, 3}


def test_channel_groups_errors():
    with pytest.raises(ValueError):
        channel_groups(0, 8)
    with pytest.raises(ValueError):
        channel_groups(8, 4, method="z_plane")
    with pytest.raises(ValueError):
        channel_groups(8, 4, method="board_map")
    with pytest.raises(ValueError):
        channel_groups(8, 4, method="nope")
