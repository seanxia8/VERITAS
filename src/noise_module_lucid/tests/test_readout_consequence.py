# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Readout chain, consequence variable K, toy source and the response-cell driver (12 Oct 2026)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from noise_module_lucid import consequence as K
from noise_module_lucid import toy
from noise_module_lucid.dataset import response_cells, run_response_matrix
from noise_module_lucid.readout import ReadoutParams, readout_chain, with_change
from noise_module_lucid.response import RESPONSE_FAMILIES


@pytest.fixture(scope="module")
def det():
    return toy.cylinder_detector()


@pytest.fixture(scope="module")
def events(det):
    return [toy.point_source_event(det, e, n_photons=800) for e in range(12)]


# --- readout chain -------------------------------------------------------------
def test_reference_chain_digits_are_first_arrivals(det, events):
    ev = events[0]
    d = readout_chain(ev.photon_sensor, ev.photon_time, det.n_channels, seed=0)
    assert d.counts["photons_in"] == d.counts["photons_detected"] == ev.photon_sensor.size
    for s, t, n in zip(d.sensor[:20], d.time[:20], d.n_pe_true[:20]):
        mine = ev.photon_time[ev.photon_sensor == s]
        assert n == mine.size and t == pytest.approx(mine.min())   # zero residual walk at the reference


def test_cells_are_paired_and_move_one_stage(det, events):
    ev = events[1]
    ref = ReadoutParams()
    a = readout_chain(ev.photon_sensor, ev.photon_time, det.n_channels, ref, seed=3)
    b = readout_chain(ev.photon_sensor, ev.photon_time, det.n_channels, with_change(ref, timing_offsets_ns=2.0), seed=3)
    np.testing.assert_array_equal(a.sensor, b.sensor)
    np.testing.assert_allclose(b.time - a.time, 2.0)
    np.testing.assert_allclose(a.charge, b.charge)
    with pytest.raises(ValueError, match="moves one parameter"):
        with_change(ref, gain=0.9, threshold_pe=0.5)


def test_gain_drift_at_fixed_mv_threshold_loses_small_hits(det):
    sensor = np.arange(200)
    t = np.zeros(200)
    ref = ReadoutParams(threshold_pe=0.5, walk_calibration_pe=0.5, spe_sigma=0.0)
    a = readout_chain(sensor, t, det.n_channels, ref, seed=0)               # 1 pe each: all fire
    b = readout_chain(sensor, t, det.n_channels, with_change(ref, gain=0.4), seed=0)
    assert len(a) == 200 and len(b) == 0                                  # 0.4 pe < 0.5 pe threshold
    c = readout_chain(sensor, t, det.n_channels, with_change(ref, threshold_pe=0.75), seed=0)
    assert np.all(c.time > 0)                                             # residual walk: later than calibrated


def test_efficiency_and_dead_channels(det, events):
    ev = events[2]
    ref = ReadoutParams()
    e = readout_chain(ev.photon_sensor, ev.photon_time, det.n_channels, with_change(ref, survival=0.5), seed=1)
    assert e.counts["photons_detected"] == pytest.approx(0.5 * ev.photon_sensor.size, rel=0.1)
    dead = tuple(range(0, 100))
    m = readout_chain(ev.photon_sensor, ev.photon_time, det.n_channels, with_change(ref, dead_channels=dead), seed=1)
    assert not np.isin(m.sensor, dead).any() and m.counts["lost_dead"] > 0


# --- consequence ---------------------------------------------------------------
def test_vertex_fit_recovers_the_source(det, events):
    errs = []
    for ev in events:
        d = readout_chain(ev.photon_sensor, ev.photon_time, det.n_channels, seed=ev.event_id)
        errs.append(K.vertex_error(K.fit_vertex(det.positions, d.sensor, d.time, d.charge), ev.vertex))
    assert np.median(errs) < 0.3                                          # metres, on a 4 m detector
    assert not K.fit_vertex(det.positions, [0, 1], [0.0, 1.0], [1.0, 1.0]).ok   # too few digits


def test_cell_consequence_ratio_and_bootstrap():
    r = np.ones(50)
    out = K.cell_consequence(2 * r, r, n_boot=200)
    assert out["K"] == pytest.approx(2.0) and out["K_lo"] == pytest.approx(2.0) and out["n_paired"] == 50
    c = np.r_[np.full(49, 2.0), np.nan]
    assert K.cell_consequence(c, r)["n_failed_cell"] == 1
    assert np.isnan(K.cell_consequence(r, np.zeros(50))["K"])             # zero baseline: undefined
    with pytest.raises(ValueError):
        K.cell_consequence(r, np.ones(3))


def test_energy_calibration_and_error():
    c = K.energy_calibration([100.0, 200.0], [50.0, 100.0])
    assert c == pytest.approx(0.5)
    assert K.energy_error(220.0, 100.0, c) == pytest.approx(0.1)


# --- response-cell driver --------------------------------------------------------
def test_response_cells_cover_every_family(det):
    cells, groups = response_cells(det.n_channels)
    fams = {c.family for c in cells}
    assert fams == {"reference", *RESPONSE_FAMILIES}
    assert cells[0].origin == "clean"
    for c in cells[1:]:
        assert c.origin == "N_struct" and len(c.params.moved_from(cells[0].params)) == 1
    assert len(cells) == 1 + sum(len(f["severity"]["grid"]) for f in RESPONSE_FAMILIES.values())


def test_response_matrix_directions_and_outputs(det, events, tmp_path):
    cells, _ = response_cells(det.n_channels, families=["timing_drift", "gain_drift", "module_loss"])
    pick = [cells[0]] + [c for c in cells if c.label in
                         ("N_struct:timing_drift@5", "N_struct:gain_drift@0.8", "N_struct:module_loss@4")]
    s = run_response_matrix(pick, events, det.positions, tmp_path, n_boot=100)
    by = {c["label"]: c for c in s["cells"]}
    assert by["reference"]["K"] == pytest.approx(1.0)
    assert by["N_struct:timing_drift@5"]["K_vertex"]["K"] > 1.3            # timing harms the vertex
    assert by["N_struct:timing_drift@5"]["K_energy"]["K"] == pytest.approx(1.0)
    assert by["N_struct:gain_drift@0.8"]["K_energy"]["K"] > 1.5             # gain harms the energy scale
    assert by["N_struct:module_loss@4"]["mean_digits"] < by["reference"]["mean_digits"]
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["non_citable"] is True and summary["kappa_m"]["status"] == "provisional_dev"
    z = np.load(tmp_path / "reference" / "digits.npz")
    assert z["offsets"][-1] == z["sensor"].size and (tmp_path / "reference" / "provenance.json").exists()
    with pytest.raises(ValueError, match="first cell"):
        run_response_matrix(pick[1:], events, det.positions)


def test_threshold_below_calibration_is_declared(det):
    sensor = np.arange(50)
    ref = ReadoutParams(threshold_pe=0.25, walk_calibration_pe=0.25, spe_sigma=0.0)
    low = readout_chain(sensor, np.zeros(50), det.n_channels,
                        with_change(ref, gain=0.2), seed=0)                    # 0.2 pe: below threshold, lost
    assert len(low) == 0
    d = readout_chain(sensor, np.zeros(50), det.n_channels,
                      ReadoutParams(threshold_pe=0.1, walk_calibration_pe=0.25, gain=0.2, spe_sigma=0.0), seed=0)
    assert d.counts["uncalibrated_walk"] == 50 and np.all(d.time > 0)          # fired, walk left uncorrected


def test_dead_map_correction_removes_the_known_charge_deficit(det, events):
    cells, _ = response_cells(det.n_channels, families=["module_loss"])
    pick = [cells[0], next(c for c in cells if c.label == "N_struct:module_loss@2")]
    s = run_response_matrix(pick, events, det.positions, n_boot=0)
    c = s["cells"][1]
    assert c["K_energy_deadcorr"]["K"] < c["K_energy"]["K"]
    assert c["K_deadcorr"] <= c["K"]
