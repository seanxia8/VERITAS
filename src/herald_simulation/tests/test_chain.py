# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""arrival times → clean traces → TES noise → export: shapes, units, provenance."""

from __future__ import annotations

import json

import numpy as np
import pytest

import herald_simulation as hs
from herald_simulation.export import write_cell
from herald_simulation.interventions import Structural
from herald_simulation.noise import sigma_cells

from .conftest import requires_hest


def _fake_event(C: int = 3) -> hs.HeraldEvent:
    rng = np.random.default_rng(0)
    return hs.HeraldEvent(event_id=1, geometry_name="fake", geometry_hash="0", energy_ev=500.0, interaction="ER",
                          vertex_cm=(0, 0, 2), n_qp=100, seed=1,
                          arrival_times_us=[np.sort(rng.uniform(0, 3000, size=40)) for _ in range(C)],
                          energies_ev=[np.ones(40) for _ in range(C)])


def test_clean_traces_units_and_amplitude_scaling():
    ev = _fake_event()
    cfg = hs.TraceConfig(amplitude_scale=2.0)
    X, meta = hs.clean_traces(ev, cfg)
    assert X.shape == (3, 16384)
    assert all(n == 40 for n in meta["n_inside"]), "µs → ns conversion keeps every arrival inside the 65.5 ms record"
    X1, _ = hs.clean_traces(ev, hs.TraceConfig(amplitude_scale=1.0))
    np.testing.assert_allclose(X, 2.0 * X1)
    assert np.argmax(X[0]) > int(6000e-6 * 2.5e5), "the event sits after the configured offset"


def test_add_noise_reports_both_covariances_and_kappa_floor():
    X = np.zeros((24, 16384))
    Xn, meta = hs.add_noise(X, hs.NoiseSpec(), 2.5e5, event_seed=5)
    assert Xn.shape == X.shape and meta["channel_structure_frozen"]
    assert meta["implied_covariance"].shape == (24, 24)
    assert 1.0 <= meta["kappa"] < 1.6, "matched cell at N/C = 683: the estimator floor, not a mismatch"
    Xn2, meta2 = hs.add_noise(X, hs.NoiseSpec(), 2.5e5, event_seed=6)
    np.testing.assert_array_equal(meta["implied_covariance"], meta2["implied_covariance"]), "one Σ̂ spans the cell"
    assert not np.array_equal(Xn, Xn2)


def test_sigma_cells_change_the_implied_covariance_or_spectrum():
    ref = hs.NoiseSpec()
    X = np.zeros((8, 16384))
    _, m0 = hs.add_noise(X, ref, 2.5e5, 1)
    with pytest.raises(ValueError, match="drift"):
        hs.add_noise(np.zeros((8, 4096)), ref, 2.5e5, 1)   # df = 61 Hz: the 50 Hz line is not a line on that grid
    for spec in sigma_cells(ref):
        _, m = hs.add_noise(X, spec, 2.5e5, 1)
        changed_cov = not np.allclose(m["implied_covariance"], m0["implied_covariance"])
        changed_budget = spec.budget.to_dict() != ref.budget.to_dict() or spec.mode != ref.mode
        assert changed_cov or changed_budget, spec.label
        assert spec.budget.provenance_record()["tfn_psd"] == "placeholder"


def test_structural_interventions_are_seeded_and_shape_preserving():
    X = np.ones((6, 100))
    for st in (Structural("sensor_loss"), Structural("gain_drift"), Structural("timing_jitter")):
        a, b = st.apply(X, 3), st.apply(X, 3)
        np.testing.assert_array_equal(a, b)
        assert a.shape == X.shape
    assert (Structural("sensor_loss").apply(X, 3) == 0).all(axis=1).sum() == 1
    with pytest.raises(ValueError):
        Structural("bogus").apply(X, 0)


def test_export_round_trip(tmp_path):
    ev = _fake_event()
    X, _ = hs.clean_traces(ev)
    rows = [{"event_id": 1, "cell": "t", **ev.truth(), "amplitude_adc": [1.0, 2.0, 3.0]}]
    paths = write_cell(tmp_path, "cell_t", rows, X[None], {"geometry": {"positions_cm": np.zeros((3, 3))}})
    import pyarrow.parquet as pq
    t = pq.read_table(paths["truth"]).to_pylist()
    assert t[0]["event_id"] == 1 and json.loads(t[0]["amplitude_adc"]) == [1.0, 2.0, 3.0]
    assert np.load(paths["traces"]).shape == (1, 3, 16384)
    prov = json.loads(paths["provenance"].read_text())
    assert prov["cell"] == "cell_t" and "herald_simulation_version" in prov


@requires_hest
def test_pilot_two_geometries_end_to_end(tmp_path):
    from herald_simulation.simulate import build_event_set
    s = build_event_set(tmp_path, n_events=2, qp_fraction=0.005, only=["geometry:HeRALD_v1_monolithic"])
    labels = [c["label"] for c in s["cells"]]
    assert labels == ["reference", "geometry:HeRALD_v1_monolithic"]
    assert tuple(s["cells"][0]["shape"]) == (2, 24, 16384) and tuple(s["cells"][1]["shape"]) == (2, 1, 16384)
