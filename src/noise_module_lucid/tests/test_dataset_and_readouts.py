# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Dataset driver, provenance, readout registry and the covariance-window gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import noise_module_lucid as nml
from noise_module_lucid import (
    EventSpec,
    PMT_1GHZ,
    covariance_cells,
    custom_readout,
    get_readout,
    intervention_matrix,
    kappa_floor_sweep,
    reference_cell,
    run_cell,
)


def _pe(C=64, N=512, seed=0):
    rng = np.random.default_rng(seed)
    pe = np.zeros((C, N))
    for c in rng.choice(C, 10, replace=False):
        pe[c, rng.integers(0, N - 1)] += rng.integers(1, 5)
    return pe


# --- readouts ----------------------------------------------------------------
def test_default_readout_is_declared_placeholder():
    assert PMT_1GHZ.calibrated is False
    assert get_readout(PMT_1GHZ.name) is PMT_1GHZ
    base, crate = PMT_1GHZ.base_crate()
    assert base["sampling_frequency"] == PMT_1GHZ.sampling_frequency_hz
    assert base["noise_power"] == pytest.approx(PMT_1GHZ.rms_mv**2)


def test_custom_readout_derives_corner_and_calibration_flag():
    ro = custom_readout("dom_300msps", sampling_frequency_hz=3.0e8, spe_tau_rise_ns=2.0,
                        spe_tau_fall_ns=8.0, mv_per_pe=2.0, rms_mv=0.5, reference="ICE-DOM paper")
    assert ro.calibrated is True
    assert ro.frontend_corner_hz == pytest.approx(2.5 * ro.spe_bandwidth_hz())
    with pytest.raises(KeyError):
        get_readout("nope")


# --- dataset -----------------------------------------------------------------
def test_run_cell_writes_truth_traces_provenance(tmp_path):
    ev = EventSpec(7, "isotropic", {"position": [0.0, 0.0, 0.0], "intensity": 50_000})
    pos = np.random.default_rng(0).normal(size=(64, 3))
    r = run_cell(reference_cell("WCTE_like", ev), [(ev, _pe())], tmp_path, positions=pos)
    assert r["shape"] == (1, 64, 512)
    d = tmp_path / "reference"
    assert (d / "traces.npy").exists() and (d / "provenance.json").exists()
    assert (d / "truth.parquet").exists() or (d / "truth.csv").exists()
    prov = json.loads((d / "provenance.json").read_text())
    for key in ("geometry", "window_ns", "readout", "preset", "grouping", "implied_covariance",
                "kappa_floor_mean", "release", "lucid_commit"):
        assert key in prov
    assert prov["release"]["releasable"] is False          # gate A0 open + placeholder readout
    assert prov["readout"]["calibrated"] is False


def test_run_cell_trace_intervention_changes_shape(tmp_path):
    ev = EventSpec(1, "isotropic", {})
    cells = {c.label: c for c in intervention_matrix("WCTE_like", ev)}
    r = run_cell(cells["sigma_struct:alias_fold_4"], [(ev, _pe())], tmp_path)
    assert r["shape"] == (1, 64, 128)                       # decimated by 4


def test_intervention_matrix_labels():
    labels = [c.label for c in intervention_matrix("WCTE_like", EventSpec(1))]
    assert labels[0] == "reference"
    assert "sigma_cov:clock_deterministic" in labels
    assert "sigma_struct:aperture_jitter_50ps" in labels


def test_covariance_cells_require_long_window():
    with pytest.raises(ValueError):
        covariance_cells("WCTE_like", EventSpec(1), window_ns=512.0)
    cells = covariance_cells("WCTE_like", EventSpec(1), window_ns=32_000.0)
    assert cells[0].label == "cov_reference"
    assert cells[0].preset_override is not None
    ev = EventSpec(1, "isotropic", {})
    r = run_cell(cells[0], [(ev, _pe(N=32_000))])
    assert r["shape"] == (1, 64, 32_000)


def test_kappa_floor_sweep_recommends_window():
    sweep = kappa_floor_sweep(Ns=(512, 32768), C=64, target=1.5)
    assert sweep["rows"][0]["kappa_floor"] > sweep["rows"][1]["kappa_floor"]
    assert sweep["recommended"]["N"] == 32768


def test_lucid_commit_is_readable_if_clone_present():
    # tolerant: the clone may be absent in a fresh checkout
    commit = nml.lucid_commit()
    assert commit is None or (len(commit) == 40 and all(ch in "0123456789abcdef" for ch in commit))
