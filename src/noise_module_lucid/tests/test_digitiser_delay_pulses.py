# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""New front-end modules: digitiser, cable delay, pulses, clock, interventions."""
from __future__ import annotations

import numpy as np
import pytest

from noise_module_lucid import (
    LUCID_DARK_RATE_HZ,
    add_afterpulses,
    add_dark_pulses,
    add_deterministic_clock,
    aperture_jitter_noise,
    cable_delay,
    group_delay_s,
    quantise,
    sampling_jitter,
)
from noise_module_lucid import pulses
from noise_module_lucid.interventions import INTERVENTIONS, describe

FS = 1.0e9


# --- digitiser ---------------------------------------------------------------
def test_quantise_variance_is_lsb2_over_12():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 50.0, size=200_000)
    q, meta = quantise(x, lsb=1.0, return_metadata=True)
    assert (q - x).var() == pytest.approx(1.0 / 12.0, rel=0.05)
    assert meta["noise_variance"] == pytest.approx(1.0 / 12.0)


def test_aperture_jitter_is_slew_rate_noise():
    rng = np.random.default_rng(0)
    flat = np.ones((1, 500))
    assert np.allclose(aperture_jitter_noise(flat, 1e-11, FS, rng), flat)
    edge = np.zeros((1, 500)); edge[0, 100:200] = np.hanning(100) * 4
    _, meta = aperture_jitter_noise(edge, 50e-12, FS, rng, return_metadata=True)
    assert meta["model"] == "first_order_derivative"
    assert meta["added_rms"] > 0.0


def test_sampling_jitter_strict_raises_on_white():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        sampling_jitter(rng.normal(size=(1, 1024)), 50e-12, FS, rng)


# --- cable -------------------------------------------------------------------
def test_cable_delay_integer_shift():
    x = np.zeros((1, 256)); x[0, 100] = 1.0
    out = cable_delay(x, 10.0 / FS, FS)
    assert int(np.argmax(out[0])) == 110


def test_cable_delay_zero_is_identity():
    x = np.random.default_rng(2).normal(size=(2, 128))
    assert np.allclose(cable_delay(x, 0.0, FS), x)


def test_cable_attenuation_reduces_power():
    x = np.random.default_rng(3).normal(size=(1, 1024))
    out = cable_delay(x, 5.0 / FS, FS, attenuation_per_hz=2e-9)
    assert np.sum(out**2) < np.sum(x**2)


def test_group_delay_is_frequency_dependent():
    f = np.array([0.0, 1e8])
    g = group_delay_s(f, 10e-9, dispersion_s2=1e-17)
    assert g[1] > g[0]


# --- pulses ------------------------------------------------------------------
def test_place_pulses_adds_spe_shape():
    out = pulses.place_pulses(np.zeros((1, 512)), [np.array([100.0])])
    assert out[0, 100:160].max() == pytest.approx(4.0)


def test_dark_noise_contract_subtracts_baseline():
    _, meta = add_dark_pulses(np.zeros((2, 10_000)), FS, 5_000.0,
                              rng=np.random.default_rng(4), return_metadata=True)
    assert meta["target_rate_hz"] == 5_000.0
    assert meta["baseline_rate_hz"] == LUCID_DARK_RATE_HZ
    assert meta["injected_rate_hz"] == pytest.approx(5_000.0 - LUCID_DARK_RATE_HZ)
    assert meta["assumes_lucid_dark"] is True
    # supplying the full rate
    _, meta2 = add_dark_pulses(np.zeros((2, 10_000)), FS, 5_000.0, rng=np.random.default_rng(4),
                               baseline_rate_hz=0.0, return_metadata=True)
    assert meta2["injected_rate_hz"] == 5_000.0
    assert meta2["assumes_lucid_dark"] is False


def test_afterpulse_is_charge_dependent():
    prim = [np.array([100.0, 200.0])]
    _, meta = add_afterpulses(np.zeros((1, 4096)), FS, prim, probability=0.0, tau_s=100e-9,
                              rng=np.random.default_rng(5), primary_charge=[np.array([0.0, 10.0])],
                              probability_per_pe=0.5, return_metadata=True)
    assert meta["counts_per_channel"] == [1]        # only the 10-pe primary fires


# --- clock -------------------------------------------------------------------
def test_deterministic_clock_power_matches_lines():
    from noise_module_lucid import PMT_SHARED, line_power
    N = 1 << 16
    out, meta = add_deterministic_clock(np.zeros((1, N)), FS, return_metadata=True)
    assert meta["total_power"] > 0.0
    assert out.var() == pytest.approx(meta["total_power"], rel=0.3)


# --- interventions -----------------------------------------------------------
def test_interventions_registry():
    kinds = describe()
    assert kinds["clock_x5"] == "preset"
    assert kinds["clock_deterministic"] == "preset"
    assert kinds["aperture_jitter_50ps"] == "trace"
    x = np.random.default_rng(7).normal(size=(2, 512))
    for name, (kind, fn) in INTERVENTIONS.items():
        if kind == "trace":
            assert fn(x, FS).shape[1] in (512, 128)
