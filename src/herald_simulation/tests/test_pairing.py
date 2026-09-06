# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The pairing contract: one event_id → one initial QP population in every geometry."""

from __future__ import annotations

import numpy as np
import pytest

import herald_simulation as hs

from .conftest import requires_hest


@requires_hest
def test_initial_population_is_identical_across_geometries():
    g24, g1 = hs.granularity_pair()
    e24 = hs.evaporate(g24, 3, 500.0, "NR", qp_fraction=0.01)
    e1 = hs.evaporate(g1, 3, 500.0, "NR", qp_fraction=0.01)
    assert e24.n_qp == e1.n_qp and e24.seed == e1.seed
    p24 = hs.initial_population(3, e24.meta["n_qp_simulated"])
    p1 = hs.initial_population(3, e1.meta["n_qp_simulated"])
    np.testing.assert_array_equal(p24, p1)
    assert e24.n_sensors == 24 and e1.n_sensors == 1
    assert e24.n_detected > 0 and e1.n_detected > 0


@requires_hest
def test_evaporation_is_deterministic_and_seeded_by_event_id():
    g = hs.shipped("HeRALD_v1_monolithic")
    a = hs.evaporate(g, 11, 300.0, "ER", qp_fraction=0.01)
    b = hs.evaporate(g, 11, 300.0, "ER", qp_fraction=0.01)
    c = hs.evaporate(g, 12, 300.0, "ER", qp_fraction=0.01)
    for x, y in zip(a.arrival_times_us, b.arrival_times_us):
        np.testing.assert_array_equal(x, y)
    assert a.seed != c.seed


@requires_hest
def test_yields_switch_with_interaction():
    er, nr = hs.quanta(1000.0, "ER"), hs.quanta(1000.0, "NR")
    assert nr["quasiparticles"] > 1.5 * er["quasiparticles"], "NR partitions far more of the deposit into quasiparticles"
    with pytest.raises(ValueError):
        hs.quanta(1000.0, "alpha")


@requires_hest
def test_make_cell_matches_the_shipped_herald_v1_layout():
    custom = hs.make_cell()
    assert custom.n_sensors == 24 and int(hs.HERALD_V1_MAP.sum()) == 24
    shipped = hs.shipped("HeRALD_v1")
    np.testing.assert_allclose(custom.positions_cm, shipped.positions_cm)
    small = hs.make_cell(array_map=np.ones((2, 2)))
    assert small.n_sensors == 4 and small.geometry_hash != custom.geometry_hash
    assert small.detector().get_nsensors() == 4
