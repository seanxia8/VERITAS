"""Cached-truth preservation for mixed-domain resampling."""
from dataclasses import replace

import numpy as np
import pytest

from jepa_noise import lucid_bombs as lb
from jepa_noise.ext_predictability import restore_cached_truth
from jepa_noise.tests.test_lucid_bombs import CONFIG


def test_cached_particle_truth_is_restored_exactly_and_large_changes_rejected():
    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(CONFIG)
    draw = lb.draw_event(geometry, cfg, 'fixture/run', 5, seed=11, domain='electron')
    saved_deposit = np.nextafter(draw.truth.deposited_MeV, 0.0)
    saved_yield = np.nextafter(draw.yield_photons, 0.0)
    cached = {'domain': draw.domain, 'vertex_m': draw.vertex_m.copy(),
              'yield_photons': saved_yield, 'emission_offset_ns': draw.emission_offset_ns,
              'direction_m': draw.truth.direction.copy(), 'energy_MeV': draw.truth.energy_MeV,
              'deposited_MeV': saved_deposit, 'track_length_m': draw.truth.track_length_m,
              'contained': draw.truth.contained}
    restored = restore_cached_truth(draw, cached)
    assert restored.truth.deposited_MeV == saved_deposit
    assert restored.yield_photons == saved_yield
    assert restored.qe_eff is draw.qe_eff and restored.poisson_u is draw.poisson_u
    assert restored.capacity == draw.capacity
    with pytest.raises(ValueError, match='yield_photons'):
        restore_cached_truth(draw, dict(cached, yield_photons=saved_yield * 1.01))


def test_cached_photon_truth_and_noise_are_preserved():
    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(CONFIG)
    draw = lb.draw_event(geometry, cfg, 'fixture/run', 5, seed=11)
    cached = {'domain': 'photon', 'vertex_m': draw.vertex_m.copy(),
              'yield_photons': np.nextafter(draw.yield_photons, 0.0),
              'emission_offset_ns': draw.emission_offset_ns}
    restored = restore_cached_truth(draw, cached)
    assert restored.truth is None and restored.domain == 'photon'
    assert restored.yield_photons == cached['yield_photons']
    assert restored.qe_eff is draw.qe_eff and restored.charge_z is draw.charge_z
    with pytest.raises(ValueError, match='domain'):
        restore_cached_truth(draw, dict(cached, domain='muon'))
