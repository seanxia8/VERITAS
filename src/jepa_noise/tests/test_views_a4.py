"""WP2-v4 A4 implementation fixtures: bounded population checks, not campaign outcomes.

Values are the runbook v4 Appendix A design check (channel split); tolerances are
the registered 0.5° for D and 0.005 for gamma_n.  No sampling or tuning here.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from jepa_noise.views import (
    A4_BASE_RHO, A4_CELL_GRID, A4_COMMON_STRENGTHS, A4_LAW, A4_LINE_MULTIPLIERS,
    A4_THETA_SD, COMMON_STRENGTHS, LAW, PRIMARY_SPLIT, SPLIT_ROLES,
    SUPPORTED_LAWS, Tier1Views, V3_BASE_RHO, V3_LAW, V3_THETA_SD, displacement,
    validate_a4_cells)

ROOT = Path(__file__).resolve().parents[3]

# runbook v4 Appendix A, A4 design, channel split: g -> (angle k=3 deg, gamma_n)
APPENDIX_A_A4 = {
    0.: (0.0, 0.000), .005: (11.3, 0.357), .01: (22.3, 0.526), .02: (48.0, 0.690),
    .05: (77.0, 0.847), .1: (84.1, 0.917), .225: (87.5, 0.962), .45: (88.8, 0.980),
    .9: (89.4, 0.990)}
# runbook v3 Appendix A design, channel split: g -> angle k=3 deg
APPENDIX_A_V3 = {0.: 18.4, .005: 74.0, .05: 84.6, .45: 89.2}


@lru_cache(maxsize=None)
def views(common_strength, line_multiplier=0., split=PRIMARY_SPLIT, law='v4'):
    """Cached Tier1Views instance; both registered laws."""
    if law == 'v4':
        return Tier1Views(split=split, line_multiplier=line_multiplier,
                          common_strength=common_strength, base_rho=A4_BASE_RHO,
                          theta_sd=A4_THETA_SD, seed=0, cell='unit_fixture_wp2v4')
    if law == 'v3':
        return Tier1Views(split=split, line_multiplier=line_multiplier,
                          common_strength=common_strength, base_rho=V3_BASE_RHO,
                          theta_sd=V3_THETA_SD, seed=0, cell='unit_fixture_wp2v4')
    raise ValueError('unknown fixture law')


@lru_cache(maxsize=None)
def population(common_strength, line_multiplier=0., split=PRIMARY_SPLIT, law='v4'):
    return views(common_strength, line_multiplier, split, law).population()


@lru_cache(maxsize=None)
def displacement_of(common_strength, line_multiplier=0., split=PRIMARY_SPLIT, law='v4'):
    return displacement(population(common_strength, line_multiplier, split, law), k=3)


@lru_cache(maxsize=None)
def default_views():
    """Bare Tier1Views(): every constructor default at once."""
    return Tier1Views()


def test_a4_grid_is_the_registered_thirteen_cells():
    assert tuple(A4_COMMON_STRENGTHS) == (0., .005, .01, .02, .05, .1, .225, .45, .9)
    assert tuple(A4_LINE_MULTIPLIERS) == (.75, 1.5, 2.25, 3.)
    # v3 values stay selectable (constructor accepts them).
    assert set(COMMON_STRENGTHS) >= {0., .225, .45, .675, .9}
    grid = validate_a4_cells(A4_CELL_GRID)
    assert len(grid) == 13 and len(set(grid)) == 13
    assert grid[0] == (0., 0.)                              # clean reference cell
    assert set(grid) == ({(g, 0.) for g in A4_COMMON_STRENGTHS}
                         | {(0., line) for line in A4_LINE_MULTIPLIERS})
    with pytest.raises(ValueError):
        validate_a4_cells(grid + ((.675, 0.),))              # 14th cell not registered
    with pytest.raises(ValueError):
        validate_a4_cells(grid[:-1])
    with pytest.raises(ValueError):
        validate_a4_cells([(g, line) for g in A4_COMMON_STRENGTHS[1:]
                           for line in A4_LINE_MULTIPLIERS])  # Cartesian product rejected
    with pytest.raises(ValueError):
        Tier1Views(common_strength=.3)
    with pytest.raises(ValueError):
        Tier1Views(base_rho=.2)


def test_config_mirrors_module_grid():
    cfg = json.loads((ROOT/'src/jepa_noise/configs/waveform_v2_a4.json').read_text())
    assert cfg['law'] == A4_LAW
    assert cfg['previous_law'] == V3_LAW
    assert cfg['base_rho'] == A4_BASE_RHO
    assert cfg['v3_base_rho'] == V3_BASE_RHO
    assert tuple(tuple(c) for c in cfg['cells']) == A4_CELL_GRID
    assert tuple(cfg['common_strengths']) == A4_COMMON_STRENGTHS
    assert tuple(cfg['line_multipliers_at_g0']) == A4_LINE_MULTIPLIERS
    assert tuple(cfg['v3_common_strengths']) == (0., .225, .45, .675, .9)
    assert_allclose(cfg['theta_sd'], A4_THETA_SD, rtol=0, atol=0)
    assert_allclose(cfg['v3_theta_sd'], V3_THETA_SD, rtol=0, atol=0)
    assert cfg['primary_split'] == PRIMARY_SPLIT
    assert cfg['split_roles'] == SPLIT_ROLES
    assert cfg['split_candidates'] == ['channel', 'interleaved', 'time', 'block']


def test_defaults_are_the_a4_law_and_channel_split():
    m = default_views()
    assert m.split_name == 'channel' == PRIMARY_SPLIT
    assert m.split_role == 'primary'
    assert m.base_rho == A4_BASE_RHO
    assert m.law == A4_LAW
    assert m.theta_fixture == 'v4-a4-e1b'
    assert_allclose(m.theta_sd, A4_THETA_SD, rtol=0, atol=0)
    assert m.manifest()['law'] == A4_LAW
    assert m.manifest()['split_role'] == 'primary'
    assert m.manifest()['primary_split'] == PRIMARY_SPLIT
    assert m.manifest()['base_rho'] == A4_BASE_RHO
    assert SPLIT_ROLES == {'channel': 'primary', 'interleaved': 'alternative',
                           'time': 'diagnostic', 'block': 'diagnostic'}


def test_theta_sd_default_and_v3_selectable():
    m = default_views()
    assert_allclose(m.theta_sd, np.array([3., .1, .04])/np.sqrt(12.), rtol=0, atol=0)
    v3 = views(0., 0., PRIMARY_SPLIT, 'v3')
    assert v3.law == V3_LAW
    assert v3.base_rho == V3_BASE_RHO
    assert np.array_equal(v3.theta_sd, V3_THETA_SD)
    assert v3.theta_fixture == 'v3-a1'
    assert v3.manifest()['law'] == V3_LAW
    assert np.linalg.norm(V3_THETA_SD-A4_THETA_SD) > .1
    with pytest.raises(ValueError):
        Tier1Views(theta_sd=[.2, .005, .002])


def test_base_rho_zero_is_channel_private():
    m = views(0., 0.)
    assert m.base_rho == 0.
    assert m.law == A4_LAW
    # A4 base channel factor is diagonal (channel-private).
    assert_allclose(m.base_channel_covariance,
                    np.diag(np.diag(m.base_channel_covariance)), rtol=0, atol=1e-14)
    assert_allclose(m.shared_weight, 0., rtol=0, atol=0)
    # Every off-diagonal cross-channel 256x256 block is zero (epsilon/white are
    # channel-diagonal); the covariance equals the declared additive law.
    for i in range(8):
        for j in range(8):
            if i != j:
                assert_allclose(m.noise_covariance[i*256:(i+1)*256, j*256:(j+1)*256],
                                0., rtol=0, atol=1e-13)
    assert_allclose(m.noise_covariance,
                    np.kron(m.base_channel_covariance, m.temporal_covariance)
                    + m.epsilon_dc*np.eye(2048), rtol=0, atol=1e-13)
    # §4.1 ε is recomputed from the A4 base; the A1 normalisation is unchanged.
    assert_allclose(m.epsilon_dc, .01*np.trace(m.raw_noise_covariance)/2048,
                    rtol=0, atol=1e-15)
    assert_allclose(m.v_white,
                    np.trace(m.channel_covariance)/8*m.white_temporal_variance,
                    rtol=0, atol=1e-15)
    assert not np.allclose(m.channel_covariance, m.base_channel_covariance)
    # The v3 law keeps the shared reference-cell base: cross-channel blocks are not zero.
    old = views(0., 0., PRIMARY_SPLIT, 'v3')
    assert old.base_rho == V3_BASE_RHO
    cross = old.noise_covariance[:256, 256:512]
    assert np.max(np.abs(cross)) > 1e-3
    assert_allclose(old.base_channel_covariance, old.channel_covariance, rtol=0, atol=1e-14)


def test_displacement_zero_monotone_and_appendix_a():
    grid = [g for g in A4_COMMON_STRENGTHS]
    angles = [displacement_of(g, 0., PRIMARY_SPLIT, 'v4')['angle_deg'] for g in grid]
    gammas = [population(g, 0., PRIMARY_SPLIT, 'v4')['gamma_n'] for g in grid]
    # Clean reference cell: exactly no displacement.
    assert angles[0] <= .01
    # Monotone nondecreasing in g (P1's displacement reading).
    assert np.min(np.diff(angles)) >= -.05
    assert np.all(np.diff(angles) > -.05)
    for g, angle, gamma in zip(grid, angles, gammas):
        expected_angle, expected_gamma = APPENDIX_A_A4[g]
        assert abs(angle-expected_angle) <= .5, (g, angle, expected_angle)
        assert abs(gamma-expected_gamma) <= .005, (g, gamma, expected_gamma)
        # gamma_n stays reported and is a canonical correlation.
        assert 0. <= gamma <= 1.
    # The k=1 / k=2 diagnostics reproduce Appendix A's (≤).
    assert abs(displacement_of(0., 0.)['angles_deg'][1]-2.1) <= .5
    assert displacement_of(0., 0.)['angles_deg'][2] <= .5
    assert abs(displacement_of(.9, 0.)['angles_deg'][1]-2.6) <= .5
    assert abs(displacement_of(.9, 0.)['angles_deg'][2]-1.8) <= .5


def test_interleaved_alternative_agrees_with_channel():
    for g in (.005, .9):
        channel = displacement_of(g, 0., 'channel', 'v4')['angle_deg']
        interleaved = displacement_of(g, 0., 'interleaved', 'v4')['angle_deg']
        assert abs(channel-interleaved) <= .5, (g, channel, interleaved)


def test_v3_law_reproduces_v3_appendix_a():
    for g, expected in APPENDIX_A_V3.items():
        angle = displacement_of(g, 0., PRIMARY_SPLIT, 'v3')['angle_deg']
        assert abs(angle-expected) <= .5, (g, angle, expected)
    # v3 base alone selects the v3 law and its θ fixture.
    m = Tier1Views(base_rho=V3_BASE_RHO)
    assert m.law == V3_LAW and m.theta_fixture == 'v3-a1'
    assert_allclose(m.theta_sd, V3_THETA_SD, rtol=0, atol=0)


def test_named_law_api_and_supported_laws():
    # E1b/E2 callers select the registered law by name (the E1b path in run.py).
    assert LAW == A4_LAW and SUPPORTED_LAWS == (A4_LAW, V3_LAW)
    m = Tier1Views(split='channel', common_strength=0., law=A4_LAW)
    assert m.law == A4_LAW and m.base_rho == A4_BASE_RHO
    assert m.manifest()['law'] == A4_LAW
    assert_allclose(m.theta_sd, A4_THETA_SD, rtol=0, atol=0)
    old = Tier1Views(split='channel', common_strength=0., law=V3_LAW)
    assert old.law == V3_LAW and old.base_rho == V3_BASE_RHO
    assert np.array_equal(old.theta_sd, V3_THETA_SD)
    with pytest.raises(ValueError):
        Tier1Views(law='v2-unknown-law')
    with pytest.raises(ValueError):
        Tier1Views(base_rho=V3_BASE_RHO, law=A4_LAW)
    # The 7-decimal registered θ SDs from the E1b config are accepted and keep
    # the A4 fixture label.
    rounded = Tier1Views(theta_sd=[.8660254, .0288675, .0115470])
    assert rounded.theta_fixture == 'v4-a4-e1b'
    assert rounded.law == A4_LAW


def test_displacement_input_validation():
    m = Tier1Views(signal_mode='nonlinear', split='channel', common_strength=0.,
                   seed=902, cell='unit_nonlinear')
    with pytest.raises(ValueError):
        m.displacement()
    with pytest.raises(ValueError):
        displacement(population(0., 0.), k=4)
    assert displacement(population(0., 0.), k=1)['k'] == 1
