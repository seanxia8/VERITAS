"""Bounded A1 implementation fixtures, not calibration or campaign outcomes."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from jepa_noise.views import Tier1Views, temporal_covariance, leading_temporal_basis


@pytest.fixture(scope='module')
def model():
    return Tier1Views(line_multiplier=.75, common_strength=.45, seed=27182,
                      cell='unit_fixture_only')


def test_source_absolute_covariance_and_white_scale(model):
    from noise_module import NoiseGenerator
    source = model.reference_cell.noise.generator(8, 11)
    source.generate(8)
    _, metadata = source.generate(256, return_metadata=True)
    assert_allclose(model.channel_covariance*np.mean(np.diag(model.temporal_covariance)),
                    metadata['implied_covariance'], atol=1e-12)
    assert not np.isclose(np.trace(model.channel_covariance)/8, 1., atol=1e-4)
    _, _, psd = NoiseGenerator(model.reference_cell.noise.base).build_psd_density(256, return_metadata=True)
    white = psd['component_contributions'][0]['integrated_power_after_global_scaling']
    assert_allclose(model.v_white, white*np.trace(model.channel_covariance)/8, atol=1e-12)
    assert_allclose(model.epsilon_dc, .01*np.trace(model.raw_noise_covariance)/2048, atol=1e-12)
    assert_allclose(model.temporal_covariance.sum(axis=0), 0., atol=1e-10)
    assert model.epsilon_dc > 0
    # Sum of PSD covariances plus physical white is SPD without dense eigensolve.
    assert np.linalg.eigvalsh(model.channel_covariance).min() > 0
    assert model.temporal_eigenvalues.min() >= 0


def test_degenerate_basis_projection_and_normalization(model):
    v = model.common_basis
    assert_allclose(v@v.T/256, np.eye(2), atol=1e-12)
    assert_allclose(model.temporal_covariance@v.T,
                    v.T*model.temporal_spectrum[:2], atol=1e-10)
    for x in v:
        assert x[np.argmax(np.abs(x))] > 0
    repeated, _, _ = leading_temporal_basis(model.temporal_eigenvalues)
    assert_allclose(v, repeated, atol=0, rtol=0)
    # Explicit all-equal spectrum gives projected coordinate axes.
    basis, spectrum, multiplicities = leading_temporal_basis(np.ones(5))
    assert_allclose(basis/np.sqrt(8), np.eye(8)[:2], atol=1e-12)
    assert multiplicities == [8]


def test_additive_covariance_and_namespaced_draws(model):
    x, targets, parts = model.noise_record(8, return_components=True)
    assert_allclose(x, sum(parts.values()), atol=0, rtol=0)
    assert_allclose(x, model.noise_record(8)[0], atol=0, rtol=0)
    assert not np.array_equal(x, model.noise_record(8, role='noise_overlay')[0])
    assert not np.array_equal(x, model.noise_record(9)[0])
    assert_allclose(model.noise_covariance-model.raw_noise_covariance-model.line_covariance-model.common_covariance,
                    model.epsilon_dc*np.eye(2048), atol=1e-12)
    assert_allclose(parts['line'][0], targets['line_I']*model.line_basis[0]+targets['line_Q']*model.line_basis[1])
    batch = model.sample([2, 7], return_components=True)
    reverse = model.sample([7, 2])
    assert_allclose(batch['raw'], reverse['raw'][::-1], atol=0, rtol=0)
    for i in range(2):
        signal = batch['raw'][i]-sum(batch['components'][i].values())
        assert_allclose(signal.ravel(), model.j0@batch['theta'][i], atol=1e-12)
    assert_allclose(batch['context'], batch['raw'].reshape(2,-1)[:,model.split.context])
    assert_allclose(batch['target'], batch['raw'].reshape(2,-1)[:,model.split.target])


def test_actual_generator_covariance_bounded_fixture(model):
    # 18k implementation records, three declared coordinates, <300 MB retained.
    # No fitted/calibration settings, no scientific prediction or sample grid.
    indices = np.array([0, 260, 1100])
    draws = np.empty((18000,3))
    total = np.zeros(2048)
    squares = np.zeros(2048)
    component_squares = {}
    endpoint_power = np.zeros(2)
    for i in range(18000):
        noise, _, parts = model.noise_record(i, role='unit_covariance', return_components=True)
        flat = noise.ravel()
        draws[i] = flat[indices]
        total += flat
        squares += flat**2
        for name, part in parts.items():
            component_squares[name] = component_squares.get(name,0.)+np.mean(part**2)
        # Source base has exactly absent DC, physical epsilon restores DC.
        endpoint_power += (np.abs(np.fft.rfft(parts['epsilon_private'], axis=-1)[:,[0,-1]])**2).mean(axis=0)
        assert np.max(np.abs(parts['base_white_shared'].sum(axis=-1))) < 1e-10
    diagonal = squares/18000-(total/18000)**2
    relative = diagonal/np.diag(model.noise_covariance)-1
    assert np.sqrt(np.mean(relative**2)) < .02
    assert np.max(np.abs(relative)) < .05
    assert_allclose(endpoint_power/18000, 256*model.epsilon_dc, rtol=.02)
    expected = {
        'base_white_shared':np.mean(model.shared_weight**2)*model.white_temporal_variance,
        'base_white_private':np.mean(model.private_weight**2)*model.white_temporal_variance,
        'base_colored_shared':np.mean(model.shared_weight**2)*(np.trace(model.temporal_covariance)/256-model.white_temporal_variance),
        'base_colored_private':np.mean(model.private_weight**2)*(np.trace(model.temporal_covariance)/256-model.white_temporal_variance),
        'epsilon_private':model.epsilon_dc,
        'line':model.line_multiplier*model.v_white,
        'common':model.common_strength*model.v_white}
    for name in expected:
        assert_allclose(component_squares[name]/18000, expected[name], rtol=.03)
    print('waveform_fixture_diagonal_rms_relative', np.sqrt(np.mean(relative**2)),
          'diagonal_max_relative', np.max(np.abs(relative)))
    empirical = np.cov(draws, rowvar=False, ddof=0)
    target = model.noise_covariance[np.ix_(indices,indices)]
    error = np.linalg.norm(empirical-target)/np.linalg.norm(target)
    assert error < .02, error
    print('waveform_fixture_three_coordinate_covariance_relative_frobenius', error)


def test_jacobian_stability_and_full_fisher(model):
    # Full 8x256 support; source template and finite-difference step cross-check.
    half = model.jacobian(model.theta0, half=True)
    assert np.linalg.norm(model.j0-half)/np.linalg.norm(half) < 1e-5
    # Dense SPD solve is an implementation fixture, no model/fit selection.
    from scipy.linalg import cho_factor, cho_solve
    factor = cho_factor(model.noise_covariance)
    info = model.j0.T@cho_solve(factor, model.j0)
    assert np.linalg.matrix_rank(info) == 3
    bound = np.linalg.inv(info)
    assert_allclose(info@bound, np.eye(3), atol=1e-8)
    amplitude_template = model.waveform(model.theta0).ravel()/model.theta0[0]
    amp_var = 1/(amplitude_template@cho_solve(factor, amplitude_template))
    assert abs(amp_var/(1/info[0,0])-1) < .01


def test_zero_nuisance_targets_and_nonlinear_signal():
    m = Tier1Views(signal_mode='nonlinear', split='channel', common_strength=0,
                   amplitude_scale=.25, seed=901, cell='unit_nonlinear')
    batch = m.sample([1000000,1000001], return_components=True)
    assert batch['nuisance_valid'] == [False]*4
    assert_allclose(batch['nuisance'], 0.)
    assert np.all((batch['theta'][:,0]>=.75)&(batch['theta'][:,0]<=1.5))
    for i in range(2):
        signal = batch['raw'][i]-sum(batch['components'][i].values())
        assert_allclose(signal, m.waveform(batch['theta'][i]), atol=1e-12)
    assert m.epsilon_dc > 0
