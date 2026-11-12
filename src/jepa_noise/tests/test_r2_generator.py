# SPDX-License-Identifier: MIT
"""Round-2 generator: test 2 (WCTE regeneration) and synthetic cylinder/sphere runs."""
from __future__ import annotations

import numpy as np
import pytest

from jepa_noise import lucid_bombs as lb
from jepa_noise import r2_generator as gen
from jepa_noise import r2_geometry as rg
from jepa_noise.tests.r2_fixtures import ROUND1_DEV_TRAIN, lucid_available, small_cylinder, small_sphere


# ---------------------------------------------------------------- test 2: WCTE regeneration
@pytest.mark.skipif(not (lucid_available() and ROUND1_DEV_TRAIN.is_dir()),
                    reason="needs LUCiD at the pin and the stored round-1 dev_train cohort")
def test_wcte_regenerates_stored_round1_run_bit_for_bit():
    geometry = rg.load_r2_geometry("wcte")
    cfg = gen.load_r2_config()
    run = gen.cohort_runs("dev_train")[0]
    run_dir = ROUND1_DEV_TRAIN / "runs" / run.run_id.replace("/", "__")
    manifest = __import__("json").loads((run_dir / "manifest.json").read_text())
    assert manifest["geometry_sha256"] == geometry.digest
    assert manifest["generator_config_sha256"] == cfg.digest
    assert manifest["runs"][0]["state_sha256"] == run.state_sha256(geometry)
    rule = gen.time_rule_for(geometry)
    assert rule == gen.ROUND1_TIME_RULE
    generated = gen.generate_run(geometry, cfg, run, rule)
    got = lb.sparse_from_readouts(generated.event_ids, generated.hits)
    ref = lb.read_sparse_hits(run_dir / "hits.npz")
    assert ref.event_ids.size == 1000
    for name in ("event_ids", "offsets", "pmt_id", "charge_pe", "time_ns"):
        a, b = getattr(got, name), getattr(ref, name)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name
    # Truth: the stored round-1 file differs from the unchanged round-1 ``draw_event`` on this
    # host by <= 1 ulp in a few photon yields (libm exp/log of the generating host); the hits,
    # which are what test 2 requires, are bit for bit.  Truth is compared to 1e-12 relative.
    with np.load(run_dir / "events.npz") as data:
        truth = gen.truth_payload(generated)
        for name in data.files:
            want, got = data[name], truth[name].astype(data[name].dtype)
            if want.dtype.kind == "f":
                assert np.allclose(got, want, rtol=1e-12, atol=0, equal_nan=True), name
            else:
                assert got.tobytes() == want.tobytes(), name
    assert generated.readout_record.n_prompt_dropped == manifest["n_prompt_dropped"]
    assert generated.readout_record.n_dark_moved_trigger == manifest["n_dark_moved_trigger"]


# ---------------------------------------------------------------- synthetic detectors
@pytest.mark.parametrize("make", [small_cylinder, small_sphere])
def test_generate_run_on_r2_geometry(make, tmp_path):
    geometry = make()
    cfg = gen.load_r2_config(2000.0, 20000.0)
    rule = gen.TimeRule(4, 20.0, (-20.0, 60.0), 100.0)
    run = lb.RunSpec("lucid/dev_probe_train/0000", "lucid/dev_probe_train", "F1", 0.05,
                     1_200_000, 12, lb.DEFAULT_SEED)
    r = gen.generate_run(geometry, cfg, run, rule, with_twin=True, store_expected_pmt=True)
    assert r.n_events == 12 and r.twin_hits is not None
    assert r.expected_pe_pmt.shape == (12, geometry.n_pmt)
    assert np.allclose(r.expected_pe_pmt.sum(1), r.expected_pe, rtol=1e-4)
    assert sorted(set(r.domains.tolist())) == ["electron", "muon", "photon"]
    for i, t in enumerate(r.particle_truth):
        if t is not None:
            assert rg.inside_fiducial(geometry, t.vertex_m[None])[0]
            end = t.vertex_m + t.track_length_m * t.direction
            assert rg.inside_fiducial(geometry, end[None], fraction=1.0 + 1e-9)[0]
    manifest = gen.write_generated_run(tmp_path / "run", r, geometry, cfg, rule)
    assert manifest["time_rule"]["n"] == 4
    lb.verify_sha256sums(tmp_path / "run")
    with np.load(tmp_path / "run" / "events.npz") as data:
        assert "expected_pe_pmt" in data.files


def test_explicit_trigger_and_prompt_window_are_used():
    geometry = small_cylinder()
    cfg = gen.load_r2_config(2000.0, 20000.0)
    draws = gen.draw_event(geometry, cfg, "t", 5)
    state = lb.nominal_state(geometry, "t")
    narrow = gen.readout(state, draws, geometry, cfg, gen.TimeRule(4, 20.0, (-1.0, 1.0), 100.0))
    wide = gen.readout(state, draws, geometry, cfg, gen.TimeRule(4, 20.0, (-20.0, 60.0), 100.0))
    assert len(narrow) < len(wide)
    assert np.all((narrow.time_ns >= -1.0) & (narrow.time_ns <= 1.0))
    never = gen.readout(state, draws, geometry, cfg, gen.TimeRule(10_000, 20.0, (-20.0, 60.0), 1.0))
    assert never.trigger_ns is None


def test_fixed_capacity_draws_share_random_numbers():
    from jepa_noise import ext_r2 as x
    g = small_cylinder()
    cfg = gen.load_r2_config(2000.0, 20000.0)
    a = x._fixed_yield_draws(g, cfg, 5000.0, 3, 1, "t", capacity_y=80000.0)
    b = x._fixed_yield_draws(g, cfg, 9000.0, 3, 1, "t", capacity_y=80000.0)
    assert np.array_equal(a.poisson_u, b.poisson_u) and a.yield_photons == 5000.0
