# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Detector-response N families (12 Oct 2026 re-scope, R-2/R-3)."""
from __future__ import annotations

import numpy as np
import pytest

from noise_module_lucid import channel_groups, to_mv
from noise_module_lucid.presets import SPE_MV_PER_PE, SPE_TAU_FALL_NS, SPE_TAU_RISE_NS
from noise_module_lucid.response import (
    EXCLUDED,
    FROZEN,
    RESPONSE_FAMILIES,
    dead_channels_from_groups,
    describe_response,
    efficiency_loss_counts,
    efficiency_loss_hits,
    gain_drift_hits,
    gain_drift_pe,
    group_values,
    leading_edge_time,
    leading_edge_times,
    module_loss_hits,
    module_loss_trace,
    scaled_qe,
    threshold_walk_hits,
    timing_drift_hits,
    shifted_spe_template,
    timing_drift_to_mv,
    timing_drift_trace,
    walk_readout_times,
)

FS = 1.0e9


# --- efficiency loss ----------------------------------------------------------
def test_counts_thinning_is_binomial():
    rng = np.random.default_rng(1)
    counts = np.full((4, 20_000), 5, dtype=np.int64)
    s = np.array([1.0, 0.9, 0.5, 0.0])
    out = efficiency_loss_counts(counts, s, rng)
    assert out.dtype == counts.dtype
    assert np.all(out <= counts) and np.all(out >= 0)
    for c in range(4):
        mean, var = 5 * s[c], 5 * s[c] * (1 - s[c])
        se = np.sqrt(max(var, 1e-12) / counts.shape[1])
        assert abs(out[c].mean() - mean) < 5 * se + 1e-12
        if 0 < s[c] < 1:
            assert out[c].var() == pytest.approx(var, rel=0.05)


def test_counts_thinning_refuses_charge_smeared_waveform():
    with pytest.raises(ValueError, match="integer pe-count"):
        efficiency_loss_counts(np.full((2, 8), 1.3), 0.9, 0)
    with pytest.raises(ValueError, match="integer pe-count"):
        efficiency_loss_counts(-np.ones((2, 8)), 0.9, 0)


def test_hits_thinning_per_pmt_and_scalar():
    rng = np.random.default_rng(2)
    ids = np.repeat(np.arange(3), 50_000)
    keep = efficiency_loss_hits(ids, np.array([1.0, 0.7, 0.0]), rng)
    assert keep[ids == 0].all() and not keep[ids == 2].any()
    assert keep[ids == 1].mean() == pytest.approx(0.7, abs=0.01)
    assert efficiency_loss_hits(ids, 0.5, 3).mean() == pytest.approx(0.5, abs=0.01)


def test_survival_validation_and_scaled_qe():
    with pytest.raises(ValueError):
        efficiency_loss_hits(np.array([0, 1]), np.array([0.5, 1.2]), 0)
    with pytest.raises(ValueError):
        efficiency_loss_hits(np.array([0, 5]), np.array([0.5, 0.5]), 0)   # id out of range
    qe = np.array([0.25, 0.30, 0.20])
    np.testing.assert_allclose(scaled_qe(qe, 0.8), qe * 0.8)
    np.testing.assert_allclose(scaled_qe(qe, [1.0, 0.5, 0.0]), [0.25, 0.15, 0.0])


# --- gain drift ---------------------------------------------------------------
def test_gain_changes_charge_not_count():
    q = np.array([1.0, 2.0, 0.5, 3.0])
    ids = np.array([0, 1, 1, 2])
    out = gain_drift_hits(q, ids, np.array([1.0, 0.5, 2.0]))
    np.testing.assert_allclose(out, [1.0, 1.0, 0.25, 6.0])
    assert out.shape == q.shape
    wf = np.ones((3, 10))
    np.testing.assert_allclose(gain_drift_pe(wf, [1.0, 1.1, 0.9])[:, 0], [1.0, 1.1, 0.9])
    with pytest.raises(ValueError):
        gain_drift_pe(wf, [-1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        gain_drift_hits(q, ids, -1.0)


# --- timing drift -------------------------------------------------------------
def test_timing_drift_hits_and_integer_trace_shift():
    np.testing.assert_allclose(timing_drift_hits([10.0, 20.0], [0, 1], np.array([1.5, -2.0])), [11.5, 18.0])
    x = np.zeros((2, 256))
    x[:, 50] = 1.0
    y = timing_drift_trace(x, np.array([3e-9, -2e-9]), FS)
    assert np.argmax(y[0]) == 53 and np.argmax(y[1]) == 48
    assert y.sum() == pytest.approx(2.0) and y.min() == 0.0         # exact, non-circular, no ringing


def test_timing_drift_trace_refuses_sub_sample_offsets():
    # the FFT phase ramp would put Gibbs precursors ahead of a sharp SPE pulse
    with pytest.raises(ValueError, match="integer-sample"):
        timing_drift_trace(np.zeros((1, 64)), 0.5e-9, FS)


def test_sub_sample_drift_rendered_from_pe_is_the_shifted_pulse():
    pe = np.zeros((1, 512))
    pe[0, 100] = 30.0
    np.testing.assert_allclose(timing_drift_to_mv(pe, 0.0), to_mv(pe))          # zero offset = to_mv
    np.testing.assert_allclose(shifted_spe_template(0.0), to_mv(np.eye(1, 60))[0], atol=1e-12)
    r, f = SPE_TAU_RISE_NS, SPE_TAU_FALL_NS
    t = np.arange(60.0)
    p = lambda u: np.where(u > 0, (1 - np.exp(-u / r)) * np.exp(-u / f), 0.0)
    norm = p(t).max()
    j = np.arange(512.0)
    for off in (0.25, 0.5, 2.0, 5.5, -1.5):
        y = timing_drift_to_mv(pe, off)[0]
        # exact: the continuous pulse started at 100 + off, sampled on the 1 ns grid
        expected = 30.0 * SPE_MV_PER_PE * p(j - 100.0 - off) / norm
        expected[j - 100.0 - np.floor(off) >= 60] = 0.0                         # template length 60 ns
        np.testing.assert_allclose(y, expected, atol=1e-9)
        assert np.all(y[: int(np.floor(100 + off)) + 1] == 0.0)                 # causal


# --- threshold walk -----------------------------------------------------------
def test_walk_curve_monotone_and_lost_below_threshold():
    thr = 0.25 * SPE_MV_PER_PE
    q = np.array([0.1, 0.3, 0.5, 1.0, 2.0, 10.0])
    t = leading_edge_time(q, thr)
    assert np.isnan(t[0])                                # 0.1 pe never reaches 0.25 pe
    assert np.all(np.diff(t[1:]) < 0)                    # bigger pulse crosses earlier
    t_peak = SPE_TAU_RISE_NS * np.log1p(SPE_TAU_FALL_NS / SPE_TAU_RISE_NS)
    assert np.all(t[1:] <= t_peak + 1e-9)
    # higher threshold -> later
    t_hi = leading_edge_time(np.array([1.0, 2.0]), 0.5 * SPE_MV_PER_PE)
    t_lo = leading_edge_time(np.array([1.0, 2.0]), 0.25 * SPE_MV_PER_PE)
    assert np.all(t_hi > t_lo)


def test_walk_curve_matches_closed_form_for_large_charge():
    # for q >> thr/A the crossing is on the initial linear rise: s(t) ~ t / (r * p*)
    r, f = SPE_TAU_RISE_NS, SPE_TAU_FALL_NS
    t_peak = r * np.log1p(f / r)
    p_peak = (1 - np.exp(-t_peak / r)) * np.exp(-t_peak / f)
    q, thr = 1000.0, 0.25 * SPE_MV_PER_PE
    level = thr / (q * SPE_MV_PER_PE)
    expected = level * r * p_peak
    assert leading_edge_time(np.array([q]), thr, template_fs=None)[0] == pytest.approx(expected, rel=0.02)


def test_threshold_walk_shift_and_loss():
    ref, new = 0.25 * SPE_MV_PER_PE, 0.75 * SPE_MV_PER_PE
    times = np.array([100.0, 100.0, 100.0])
    q = np.array([0.5, 1.0, 5.0])
    out, keep = threshold_walk_hits(times, q, ref, new)
    assert keep.tolist() == [False, True, True]          # 0.5 pe no longer crosses 0.75 pe
    assert np.isnan(out[0])
    shift = out[keep] - 100.0
    assert np.all(shift > 0) and shift[0] > shift[1]     # small charge walks more
    same, k2 = threshold_walk_hits(times, q, ref, ref)
    np.testing.assert_allclose(same, times)
    assert k2.all()


def test_threshold_walk_refuses_subthreshold_reference():
    ref = 0.25 * SPE_MV_PER_PE
    with pytest.raises(ValueError, match="below the reference threshold"):
        threshold_walk_hits(np.array([1.0, 2.0]), np.array([0.1, 1.0]), ref, ref)
    out, keep = threshold_walk_hits(np.array([1.0, 2.0]), np.array([0.1, 1.0]), ref, ref,
                                    drop_subthreshold_reference=True)
    assert keep.tolist() == [False, True]


def test_hit_walk_agrees_with_a_to_mv_trace():
    # the default normalisation follows units.spe_template, so the 1 GHz trace built by
    # to_mv crosses where the hit-level walk says (up to linear interpolation on 1 ns bins)
    thr = 0.3 * SPE_MV_PER_PE
    for q in (1.0, 3.0, 20.0):
        pe = np.zeros((1, 256))
        pe[0, 50] = q
        measured = leading_edge_times(to_mv(pe), thr, FS)[0]
        predicted = walk_readout_times(np.array([50.0]), np.array([q]), thr)[0]
        assert measured == pytest.approx(predicted, abs=0.15)


def test_analytic_walk_and_fine_trace_leading_edge_agree():
    # a 2 pe SPE pulse on a fine grid: the trace crossing equals the analytic walk
    fs = 100e9                                          # 10 ps samples
    t_ns = np.arange(4000) / fs * 1e9
    r, f = SPE_TAU_RISE_NS, SPE_TAU_FALL_NS
    t_peak = r * np.log1p(f / r)
    p_peak = (1 - np.exp(-t_peak / r)) * np.exp(-t_peak / f)
    q, start = 2.0, 5.0
    tt = np.clip(t_ns - start, 0, None)
    pulse = q * SPE_MV_PER_PE * (1 - np.exp(-tt / r)) * np.exp(-tt / f) / p_peak
    thr = 0.3 * SPE_MV_PER_PE
    measured = leading_edge_times(pulse[None, :], thr, fs)[0]
    predicted = walk_readout_times(np.array([start]), np.array([q]), thr, template_fs=None)[0]
    assert measured == pytest.approx(predicted, abs=0.02)
    assert np.isnan(leading_edge_times(np.zeros((1, 10)), 1.0, FS)[0])


# --- module loss --------------------------------------------------------------
def test_module_loss_from_groups():
    groups = channel_groups(256, 64)
    dead = dead_channels_from_groups(groups, [1, 3])
    np.testing.assert_array_equal(dead, np.r_[64:128, 192:256])
    x = np.ones((256, 8))
    y = module_loss_trace(x, dead)
    assert y[dead].sum() == 0 and y[np.setdiff1d(np.arange(256), dead)].min() == 1.0
    assert x.min() == 1.0                                # input untouched
    keep = module_loss_hits(np.array([0, 70, 130, 200]), dead)
    assert keep.tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        dead_channels_from_groups(groups, [9])


def test_group_values_per_crate_drift():
    groups = channel_groups(128, 32)
    g = group_values(groups, 128, 0.8, [2], 1.0)
    assert np.all(g[64:96] == 0.8) and g[:64].min() == 1.0 and g[96:].min() == 1.0


# --- registry -----------------------------------------------------------------
def test_registry_taxonomy():
    assert set(describe_response()) == {"efficiency_loss", "gain_drift", "timing_drift",
                                        "threshold_walk", "module_loss"}
    assert set(describe_response().values()) == {"N_struct"}
    # "N_struct" is the origin label of latent_monitor.protocol.labels.ORIGIN (not imported:
    # this package does not depend on latent_monitor)
    assert EXCLUDED["angular_acceptance_model"]["category"] == "M"
    assert EXCLUDED["static_qe_spread"]["category"] == "reference"
    assert {v["category"] for v in EXCLUDED.values()} == {"M", "reference"}
    assert "front_end_electronics_noise" in FROZEN and "front_end_electronics_noise" not in EXCLUDED
    assert not set(EXCLUDED) & set(RESPONSE_FAMILIES)
