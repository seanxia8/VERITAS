"""Focused engineering checks for the predictability extension."""
from dataclasses import fields

import numpy as np

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
