# SPDX-License-Identifier: MIT
"""Round-2 geometry: tests 1 (WCTE identity), 3 (pseudo-modules), 4 (sphere), 5 (rules)."""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from jepa_noise import lucid_bombs as lb
from jepa_noise import lucid_views as lv
from jepa_noise import r2_geometry as rg
from jepa_noise.tests.r2_fixtures import lucid_available, small_cylinder, small_sphere


# ---------------------------------------------------------------- test 1: WCTE identity
@pytest.mark.skipif(not lucid_available(), reason="LUCiD config at the pin not available")
def test_wcte_identity_geometry_and_rules():
    g1 = lb.load_wcte_geometry(rg.lucid_config_dir() / "WCTE_geom_config.json")
    g2 = rg.load_r2_geometry("wcte")
    for name in ("positions_m", "directions", "surfaces", "pmt_id", "mpmt_id", "pmt_id_in_mpmt",
                 "module_index", "module_ids", "module_centres_m"):
        a, b = getattr(g1, name), getattr(g2, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    for name in ("radius_m", "height_m", "half_height_m", "sensor_radius_m", "n_pmt",
                 "n_modules", "pmts_per_module"):
        assert getattr(g1, name) == getattr(g2, name), name
    assert g1.digest == g2.digest
    assert np.array_equal(lv.token_hilbert_order(g1), lv.token_hilbert_order(g2))
    assert np.array_equal(lv.module_hilbert_order(g1), lv.module_hilbert_order(g2))
    from jepa_noise.point_encoder import FeatureScaling, geometry_features
    assert FeatureScaling.from_geometry(g1) == FeatureScaling.from_geometry(g2)
    assert np.array_equal(geometry_features(g1, FeatureScaling.from_geometry(g1)),
                          geometry_features(g2, FeatureScaling.from_geometry(g2)))
    v = rg.derived_values(g2)
    assert v["prompt_window_ns"] == tuple(lb.TIME_RULE["prompt_window_ns"])
    assert v["nhits_n"] == lb.TIME_RULE["n"]
    assert v["nhits_window_ns"] == lb.TIME_RULE["trigger_window_ns"]
    assert v["time_scale_ns"] == FeatureScaling.from_geometry(g1).time_scale_ns == 100.0
    assert (v["w_rf"], v["shift_rf"]) == (64, 32)
    assert v["k"] == 19 and v["n_modules"] == 105 and not v["pseudo_modules"]
    assert rg.compare_with_design(v, "wcte") == []
    assert rg.readout_window_check(v)["ok"]
    # fiducial sampling and boundary are the round-1 functions on a cylinder
    for i in range(20):
        r1, r2 = lb.keyed_rng(1, "t", i), lb.keyed_rng(1, "t", i)
        assert np.array_equal(lb.fiducial_vertex(g1, r1), rg.fiducial_vertex(g2, r2))
    p, d = np.array([0.3, -0.2, 0.4]), np.array([0.6, 0.0, 0.8])
    assert lb._distance_to_boundary(g1, p, d) == rg.distance_to_boundary(g2, p, d)


@pytest.mark.skipif(not lucid_available(), reason="LUCiD config at the pin not available")
@pytest.mark.parametrize("detector", ["sk", "hk", "juno"])
def test_design_table_reproduced(detector):
    v = rg.derived_values(rg.load_r2_geometry(detector))
    assert rg.compare_with_design(v, detector) == []
    assert rg.readout_window_check(v)["ok"]


# ---------------------------------------------------------------- test 3: pseudo-modules
def test_pseudo_module_size_rule():
    assert rg.pseudo_module_size(1995) == 19
    assert rg.pseudo_module_size(11096) == 19
    assert rg.pseudo_module_size(19746) == 18
    assert rg.pseudo_module_size(10000) == 20
    assert rg.pseudo_module_size(360) == 20  # 18 and 20 tie -> larger


@pytest.mark.parametrize("make", [small_cylinder, small_sphere])
def test_pseudo_modules_equal_contiguous_deterministic(make):
    g = make()
    k = g.pmts_per_module
    assert k == rg.pseudo_module_size(g.n_pmt)
    assert np.all(np.bincount(g.module_index) == k)
    order = rg.token_order_for_box(g.positions_m, g.radius_m, g.half_height_m)
    assert np.array_equal(g.module_index[order], np.repeat(np.arange(g.n_modules), k))
    assert np.array_equal(g.pmt_id_in_mpmt[order], np.tile(np.arange(1, k + 1), g.n_modules))
    assert np.array_equal(lv.token_hilbert_order(g), order)
    g2 = make()
    assert g.digest == g2.digest and np.array_equal(g.module_index, g2.module_index)
    # the within (parity) split halves each module: odd slots -> target, even -> context
    mask = lv.within_mask(g, 0, np.zeros(0, dtype=np.int64), n_min=0)
    for m in range(g.n_modules):
        members = np.flatnonzero(g.module_index == m)
        t = np.intersect1d(members, mask.target).size
        c = np.intersect1d(members, mask.context).size
        assert t == math.ceil(k / 2) and c == k // 2


# ---------------------------------------------------------------- test 4: sphere
def test_sphere_fiducial_uniform_in_volume():
    from scipy.stats import kstest
    g = small_sphere(200, 17.5)
    pts = np.array([rg.fiducial_vertex(g, lb.keyed_rng(7, "sphere", i)) for i in range(5000)])
    r = np.linalg.norm(pts, axis=1)
    assert r.max() <= 0.8 * 17.5
    assert kstest((r / (0.8 * 17.5)) ** 3, "uniform").pvalue > 0.01
    u = pts / r[:, None]
    assert kstest((u[:, 2] + 1) / 2, "uniform").pvalue > 0.01
    assert np.all(rg.inside_fiducial(g, pts))


def test_sphere_distance_to_boundary_matches_ray_march():
    g = small_sphere(200, 17.5)
    rng = np.random.default_rng(3)
    for _ in range(25):
        p = rg.fiducial_vertex(g, rng)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        t = rg.distance_to_boundary(g, p, d)
        step = 1e-3
        s = np.arange(0.0, 40.0, step)
        inside = np.linalg.norm(p[None, :] + s[:, None] * d[None, :], axis=1) <= 17.5
        march = s[np.argmin(inside)]  # first outside sample
        assert abs(march - t) <= step + 1e-9
    assert g.surfaces.tolist() == ["sphere"] * g.n_pmt
    assert g.half_height_m == g.radius_m
    assert np.allclose(np.linalg.norm(g.directions, axis=1), 1.0)
    assert np.allclose(g.half_extent_m, [17.5, 17.5, 17.5])


# ---------------------------------------------------------------- test 5: derived values
@pytest.mark.parametrize("det,shape,r,h,n", [
    ("wcte", "cylinder", 1.5771308488246254, 2.77697099384, 1995),
    ("sk", "cylinder", 16.96228024518598, 36.37842222222, 11096),
    ("hk", "cylinder", 32.42959111111606, 65.92942222222001, 19746),
    ("juno", "sphere", 17.5, 35.0, 10000)])
def test_derived_values_against_hand_values(det, shape, r, h, n):
    geo = SimpleNamespace(shape=shape, radius_m=r, height_m=h)
    hand = {"wcte": (4.20, 18.6, 60.0, 100.0, 4, 64), "sk": (49.74, 220.7, 270.0, 450.0, 6, 384),
            "hk": (92.48, 410.3, 460.0, 766.7, 8, 640), "juno": (35.00, 155.3, 200.0, 333.3, 6, 320)}
    L, tc, hi, ts, nn, w = hand[det]
    length = rg.longest_path_m(geo)
    assert round(length, 2) == L
    t_cross = rg.crossing_time_ns(length)
    assert round(t_cross, 1) == tc
    prompt = rg.prompt_window_rule(t_cross)
    assert prompt == (-20.0, hi)
    assert round(rg.time_scale_rule(prompt), 1) == ts
    assert rg.nhits_n_rule(n) == nn
    assert rg.rf_window_rule(n) == (w, w // 2)
    # by hand: lambda_20 = 1 kHz x N x 20 ns; N-1 must fail the 1e-5 rule (unless N = 4)
    lam = 1e3 * n * 20e-9
    from scipy.stats import poisson
    assert 50 * poisson.sf(nn - 1, lam) <= 1e-5
    if nn > 4:
        assert 50 * poisson.sf(nn - 2, lam) > 1e-5
