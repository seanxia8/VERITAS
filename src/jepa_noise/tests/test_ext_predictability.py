"""Focused engineering checks for the predictability extension."""
from dataclasses import fields

import numpy as np
import pytest

from jepa_noise import lucid_bombs as lb
from jepa_noise.ext_predictability import resample_noise
from jepa_noise.tests.test_lucid_bombs import CONFIG


def test_resample_preserves_physics_and_redraws_noise():
    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(CONFIG)
    original = lb.draw_event(geometry, cfg, "fixture/run", 5, seed=11)
    same = resample_noise(original, geometry, cfg, 0, seed=11)
    for field in fields(original):
        np.testing.assert_equal(getattr(same, field.name), getattr(original, field.name))
    fresh = resample_noise(original, geometry, cfg, 1, seed=11)
    for name in ("vertex_m", "yield_photons", "qe_eff", "emission_offset_ns", "capacity"):
        np.testing.assert_equal(getattr(fresh, name), getattr(original, name))
    for name in ("poisson_u", "charge_z", "transit_z"):
        assert not np.array_equal(getattr(fresh, name), getattr(original, name))
    for name in ("dark_counts", "dark_times"):
        assert any(not np.array_equal(getattr(resample_noise(original, geometry, cfg, r, seed=11), name),
                                      getattr(original, name)) for r in range(1, 9))
    again = resample_noise(original, geometry, cfg, 1, seed=11)
    for name in ("poisson_u", "charge_z", "transit_z", "dark_counts", "dark_times"):
        np.testing.assert_equal(getattr(fresh, name), getattr(again, name))
    assert fresh.dark_times.size == fresh.dark_counts.sum()


def test_predictability_deterministic_and_noise():
    from jepa_noise.ext_predictability import predictability

    rng = np.random.default_rng(512)
    deterministic = np.repeat(rng.normal(size=(2000, 1)), 16, axis=1)
    assert abs(predictability(deterministic) - 1.0) < 1e-9
    assert abs(predictability(rng.normal(size=(2000, 16)))) < 0.02


def test_scheme_b_state_follows_cell_law():
    from jepa_noise.ext_predictability import scheme_b_state

    geometry = lb.from_synthetic()
    gains = np.concatenate([scheme_b_state(geometry, 0.05, "fixture/run", i, 1,
                                          seed=11).module_gain for i in range(2000)])
    assert abs(np.std(np.log(gains)) / 0.05 - 1) < 0.05


def test_ridge_recovers_predictable_component():
    from scripts.jepa_pregrid_checks import ridge_r2

    rng = np.random.default_rng(42)
    values = rng.normal(size=200)
    result = ridge_r2(values[:100, None], values[:100], values[100:, None], values[100:])
    assert result["r2"] > 1 - 1e-9


def test_particle_noise_resampling_preserves_domain_and_truth():
    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(CONFIG)
    for domain in ("muon", "electron"):
        original = lb.draw_event(geometry, cfg, "fixture/run", 7, seed=11, domain=domain)
        fresh = resample_noise(original, geometry, cfg, 1, seed=11)
        assert fresh.domain == original.domain == domain
        assert fresh.truth is original.truth
        for name in ("vertex_m", "yield_photons", "qe_eff", "emission_offset_ns", "capacity"):
            np.testing.assert_equal(getattr(fresh, name), getattr(original, name))
        for name in ("poisson_u", "charge_z", "transit_z"):
            assert not np.array_equal(getattr(fresh, name), getattr(original, name))


def test_tokenisation_reuses_readout_and_cleans_temporary_files(tmp_path, monkeypatch):
    from jepa_noise.ext_predictability import tokenise_realisations
    from jepa_noise.tests.test_lucid_bombs import install_time_rule

    install_time_rule(monkeypatch)
    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(CONFIG)
    draws = lb.draw_event(geometry, cfg, "fixture/run", 5, seed=11)
    state = lb.state_for_run("F1", .05, draws.run_id, geometry, seed=11)
    tokens, target = tokenise_realisations(draws, geometry, cfg, state, "A", tmp_path,
                                           spread=.05, seed=11, replicates=2)
    assert tokens.shape == (2, geometry.n_pmt, 9)
    np.testing.assert_array_equal(target, np.flatnonzero(geometry.pmt_id_in_mpmt % 2 == 1))
    assert not list(tmp_path.iterdir())
    from jepa_noise import point_encoder as pe
    from jepa_noise.lucid_views import ViewMasker
    original_hit = lb.make_readout(geometry, cfg)(state, draws)
    original_file = tmp_path / 'original.npz'
    lb.write_sparse_hits(original_file, lb.sparse_from_readouts([5], [original_hit]))
    original_batch = pe.tokenise(original_file, geometry, split='within',
                                 masker=ViewMasker(geometry, 'within', n_min=0))
    np.testing.assert_array_equal(tokens[0], original_batch.features.numpy()[0])


def test_event_moments_match_selected_realisations():
    from jepa_noise.ext_predictability import event_moments

    rng = np.random.default_rng(14)
    features = np.zeros((16, 5, 9), dtype=np.float32)
    hits = rng.integers(0, 2, (16, 5)).astype(bool)
    features[:, :, 6:8] = rng.normal(size=(16, 5, 2)) * hits[:, :, None]
    features[:, :, 8] = hits
    target = np.array([1, 3])
    result = event_moments(features, target)
    x = features[:, target, 6:9]
    np.testing.assert_allclose(result["mean"], x.mean(axis=0), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(result["variance"], x.var(axis=0, ddof=1), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(result["hit_count"], hits[:, target].sum(axis=0))
    np.testing.assert_allclose(result["hit_sum"], x.sum(axis=0), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result["hit_sum_squares"], (x * x).sum(axis=0), rtol=1e-6, atol=1e-6)


def test_moment_predictability_and_hit_only_constant():
    from jepa_noise.ext_predictability import (hit_moments_predictability,
                                                moments_predictability, predictability)
    rng = np.random.default_rng(37)
    raw = rng.normal(size=(2000, 16))
    assert moments_predictability(raw.mean(axis=1), raw.var(axis=1, ddof=1)) == \
        pytest.approx(predictability(raw), abs=1e-12)
    count = np.array([3, 7])
    result = hit_moments_predictability(count, count, count)
    assert result == {"rho": None, "pairs": 2, "observations": 10}


def test_summary_domain_alignment_and_hit_flag_undefined():
    from scripts.jepa_ext_predictability_summary import summarise
    rng = np.random.default_rng(123)
    x = rng.normal(size=(12, 2, 16, 3))
    x[..., 2] = 1
    sample = {'event_ids': np.arange(12), 'run_ids': np.repeat(np.arange(3), 4),
              'domains': np.tile(['photon', 'muon', 'electron'], 4),
              'yields_photons': np.arange(1, 13), 'target_index': np.arange(2),
              'original_target_hits': np.full(12, 21),
              'mean': x.mean(axis=2), 'variance': x.var(axis=2, ddof=1),
              'hit_count': np.full((12, 2), 16),
              'hit_sum': x.sum(axis=2), 'hit_sum_squares': (x * x).sum(axis=2)}
    result = summarise(sample, sample)
    assert result['rows']['photon']['all']['n_events'] == 4
    assert result['rows']['all']['all']['n_events'] == 12
    hit = result['rows']['photon']['all']['features']['hit']['hit_only']
    assert hit['rho_total'] is None and hit['rho_phys'] is None
    assert hit['scheme_a_counts']['observations'] == 4 * 2 * 16


def test_linear_module_pool_and_grouped_probe():
    from scripts.jepa_ext_predictability_linear import grouped_scores, pool, quadratic
    rng = np.random.default_rng(10)
    features = rng.normal(size=(50, 6, 9))
    pooled = pool(features, [np.array([0, 1]), np.array([2, 3, 4, 5])])
    np.testing.assert_allclose(pooled[:, :3], features[:, :2, 6:9].mean(axis=1))
    np.testing.assert_allclose(pooled[:, 3:], features[:, 2:, 6:9].mean(axis=1))
    group = np.repeat(np.arange(10), 5)
    x = rng.normal(size=(50, 4))
    assert grouped_scores(x, x[:, 0], group) > 1 - 1e-8
    assert quadratic(x[:, :3], np.exp(x[:, 3])).shape == (50, 14)


def test_reused_scratch_is_confined_and_not_accumulated(tmp_path, monkeypatch):
    from jepa_noise.ext_predictability import tokenise_realisations
    from jepa_noise.tests.test_lucid_bombs import install_time_rule

    install_time_rule(monkeypatch)
    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(CONFIG)
    draws = lb.draw_event(geometry, cfg, 'fixture/run', 5, seed=11)
    state = lb.state_for_run('F1', .05, draws.run_id, geometry, seed=11)
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    for scheme in ('A', 'B'):
        features, _ = tokenise_realisations(draws, geometry, cfg, state, scheme, tmp_path,
                                             spread=.05, seed=11, replicates=2, scratch_dir=scratch)
        assert features.shape[0] == 2
        assert [p.name for p in scratch.iterdir()] == ['hits.npz']
    with pytest.raises(ValueError, match='within the temporary root'):
        tokenise_realisations(draws, geometry, cfg, state, 'A', scratch,
                              spread=.05, seed=11, replicates=2, scratch_dir=tmp_path)
