# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The §1 table on the linear subject: every predicted signature and attribution."""

from __future__ import annotations

import numpy as np
import pytest

from latent_monitor import DesignedCell, DesignedFamily, attribute, cell_statistics
from latent_monitor.tier1 import event_cells, geometry_cells, sigma_covariance_cells, sigma_structural_cells

from .conftest import EVAL_IDS, NOISE_IDS


def _stats(reference, subject, ref_cell, cell):
    return cell_statistics(reference, subject, ref_cell, cell, EVAL_IDS, NOISE_IDS)


def test_reference_projectors_are_sane(reference, subject):
    k = subject.latent_dim
    assert reference.fisher_rank == k, "tied whitened decoder: every latent direction is excited"
    assert reference.k_out == 3
    np.testing.assert_allclose(reference.P_out + reference.P_null, np.eye(k), atol=1e-12)
    np.testing.assert_allclose(reference.P_exc + reference.P_unexc, np.eye(k), atol=1e-12)
    np.testing.assert_allclose(reference.P_out @ reference.P_out, reference.P_out, atol=1e-10)


def test_reference_cell_against_itself_is_undeclared(reference, subject, ref_cell, thresholds):
    s = _stats(reference, subject, ref_cell, ref_cell)
    a = attribute(s, thresholds)
    assert a.label == "undeclared"
    assert s.mean_shift_norm < thresholds.mean_shift_norm
    assert abs(np.mean(s.z_var_ratio) - 1.0) < 0.15


@pytest.mark.parametrize("i", range(4))
def test_sigma_covariance_cells_show_variance_without_mean_shift(reference, subject, ref_cell, thresholds, i):
    cell = sigma_covariance_cells(ref_cell)[i]
    s = _stats(reference, subject, ref_cell, cell)
    a = attribute(s, thresholds)
    assert a.label == "sigma_cov", (cell.label, a.reason)
    assert s.mean_shift_norm < thresholds.mean_shift_norm, "covariance-type Σ: no consistent mean shift"
    assert a.evidence["var_off"], "noise-only statistics must move"
    assert "re-estimate Σ" in a.adjustment


def test_line_pickup_is_a_narrow_band_signature(reference, subject, ref_cell):
    line = [c for c in sigma_covariance_cells(ref_cell) if "line" in c.label][0]
    s = _stats(reference, subject, ref_cell, line)
    assert s.psd_dev_line > 3.0 and s.residual_psd_ratio_participation < 0.5


@pytest.mark.parametrize("name", ["gain_drift", "channel_loss"])
def test_structural_cells_shift_the_mean_and_the_noise(reference, subject, ref_cell, thresholds, name):
    cell = [c for c in sigma_structural_cells(ref_cell) if name in c.label][0]
    s = _stats(reference, subject, ref_cell, cell)
    a = attribute(s, thresholds)
    assert a.label == "sigma_struct", (cell.label, a.reason)
    assert s.mean_shift_norm > thresholds.mean_shift_norm
    assert s.layer_peak in ("whitened", "channel"), "structural N peaks at the per-channel stage"
    assert a.evidence["var_off"]


def test_timing_jitter_decorrelates_a_shared_component(reference, subject, ref_cell, thresholds):
    """Documented outcome: jitter on a trace with a shared noise component is a covariance change."""
    cell = [c for c in sigma_structural_cells(ref_cell) if "jitter" in c.label][0]
    a = attribute(_stats(reference, subject, ref_cell, cell), thresholds)
    assert a.label in ("sigma_cov", "sigma_struct")


@pytest.mark.parametrize("i", range(2))
def test_geometry_cells_peak_at_the_token_stage(reference, subject, ref_cell, thresholds, i):
    cell = geometry_cells(ref_cell)[i]
    s = _stats(reference, subject, ref_cell, cell)
    a = attribute(s, thresholds)
    assert a.label == "geometry", (cell.label, a.reason)
    assert s.layer_peak == "token"
    assert s.layer_profile["z"]["total"] < 0.5 * s.layer_profile["token"]["total"]
    assert abs(np.mean(s.z_var_ratio) - 1.0) < 0.2, "the subject's own pooling scale must explain the noise variance"
    assert s.out_of_span_fraction < 0.5


def test_out_of_span_event_abstains_and_in_span_events_are_supported_but_rare(reference, subject, ref_cell, thresholds):
    cells = {c.label: c for c in event_cells(ref_cell)}
    g = _stats(reference, subject, ref_cell, cells["event:glitch"])
    ag = attribute(g, thresholds)
    assert ag.label == "event" and g.out_of_span_fraction > 0.5 and not ag.evidence["var_off"]
    assert "abstain" in ag.adjustment and "no weight adjustment" in ag.adjustment
    for name in ("event_in_span:oscillation", "event_in_span:double_pulse"):
        s = _stats(reference, subject, ref_cell, cells[name])
        a = attribute(s, thresholds)
        assert a.label == "event_in_span", (name, a.reason)
        assert s.out_of_span_fraction < 0.5 and not a.evidence["var_off"]
        assert np.max(s.consequence_ratio) > 1.1, "supported-but-rare physics still raises the consequence"


def test_noise_only_statistics_never_move_for_a_physics_change(reference, subject, ref_cell):
    """The N-vs-S discriminator: random-trigger records cannot see an event-type change."""
    for cell in event_cells(ref_cell):
        s = _stats(reference, subject, ref_cell, cell)
        np.testing.assert_allclose(s.z_var_ratio, 1.0, atol=1e-9)
        assert s.psd_dev_line < 1e-9 and s.residual_chan_corr_shift < 1e-9


def test_designed_dissociation_has_the_predicted_sign(reference, subject, ref_cell, thresholds):
    null = DesignedCell(ref_cell, subject, reference, DesignedFamily("output_null", 3.0))
    aligned = DesignedCell(ref_cell, subject, reference, DesignedFamily("output_aligned", 3.0))
    sn, sa = _stats(reference, subject, ref_cell, null), _stats(reference, subject, ref_cell, aligned)
    an, aa = attribute(sn, thresholds), attribute(sa, thresholds)
    assert an.label == "output_null" and aa.label == "output_aligned"
    # same alarm magnitude by construction (norm-matched), opposite consequence
    assert sn.dz_norm_mean > thresholds.dz_norm and sa.dz_norm_mean > thresholds.dz_norm
    assert np.max(np.abs(sn.consequence_ratio - 1.0)) < 0.05
    assert np.max(sa.consequence_ratio) > 1.2
    assert sn.dz_energy_event["null"] > 0.95 and sa.dz_energy_event["out"] > 0.95
    # norm-matched in Euclidean z by construction: a Euclidean alarm cannot tell them apart; the projector split can
    assert sn.dz_euclid_mean == pytest.approx(sa.dz_euclid_mean, rel=0.05)
