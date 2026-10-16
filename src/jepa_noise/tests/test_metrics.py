"""Independent edge cases for nuisance metrics and registered R5 eligibility."""
import numpy as np
import pytest
from jepa_noise.metrics import nuisance_r2, learning_time, learning_order


def test_constant_nonbinary_float_is_undefined():
    r = nuisance_r2(np.full((100, 1), .1), np.zeros((100, 1)))
    assert not r['valid'][0]
    assert np.isnan(r['per_target'][0]) and np.isnan(r['mean'])


def test_tiny_actual_variation_is_not_hidden():
    y = np.full((100, 1), .1)
    y[0, 0] = np.nextafter(.1, 1.)
    r = nuisance_r2(y, y)
    assert r['valid'][0] and r['per_target'][0] == 1.


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('side', ['truth', 'prediction'])
def test_nonfinite_nuisance_rejected(bad, side):
    truth = np.arange(10., dtype=float).reshape(-1, 1)
    pred = truth.copy()
    (truth if side == 'truth' else pred)[0, 0] = bad
    with pytest.raises(ValueError, match='nonfinite'):
        nuisance_r2(truth, pred)


def plateau(cross=100):
    q = np.zeros(100001)
    q[cross:] = 1.
    return q


@pytest.mark.parametrize('q,reason', [
    (np.zeros(100001), 'below_absolute_floor'),
    (np.full(100001, .1), 'insufficient_absolute_growth'),
    (np.linspace(1., .01, 100001), 'insufficient_absolute_growth'),
])
def test_nonlearning_controls(q, reason):
    r = learning_time(q)
    assert not r['eligible'] and r['censored']
    assert r['learning_time'] is None and r['reason'] == reason


def test_rising_plateau_crossing_and_initialization_baseline():
    q = np.full(100001, .1)
    q[500:] = 1.
    r = learning_time(q)
    assert r['eligible'] and r['learning_time'] == 500
    assert r['threshold'] == .55


def test_sustained_crossing_exact_100_and_no_lookahead_past_horizon():
    q = plateau(500)
    q[100:199] = 1.  # 99 steps cannot count.
    assert learning_time(q)['learning_time'] == 500
    q[100:200] = 1.
    assert learning_time(q)['learning_time'] == 100


def test_no_sustained_crossing_censored_despite_terminal_mean():
    q = np.tile([0., 2.], 50001)[:100001]
    r = learning_time(q)
    assert r['qf'] == 1. and r['terminal_difference'] == 0.
    assert r['reason'] == 'no_sustained_crossing'
    assert r['censored'] and r['learning_time'] is None


def test_terminal_step_indexing_and_stability_boundary():
    q = plateau()
    q[98000] = 10000.  # Outside specified previous block.
    assert learning_time(q)['eligible']
    q[98001:99001] = .9
    assert learning_time(q)['reason'] == 'terminal_not_stabilized'
    q[98001:99001] = .96
    assert learning_time(q)['eligible']


def test_incomplete_nonfinite_and_invalid_absolute_curves():
    assert learning_time(np.ones(100000))['reason'] == 'incomplete_or_invalid_horizon'
    q = plateau(); q[10] = np.nan
    assert learning_time(q)['reason'] == 'nonfinite_contribution'
    q[10] = -1.
    assert learning_time(q)['reason'] == 'negative_absolute_contribution'


def test_growth_and_floor_boundaries():
    q = plateau(); q[0] = .5
    assert learning_time(q)['eligible']  # qf == 2*q0 is permitted.
    q[0] = np.nextafter(.5, 1.)
    assert learning_time(q)['reason'] == 'insufficient_absolute_growth'
    q = plateau()*1e-6
    assert learning_time(q)['reason'] == 'below_absolute_floor'


def test_all_modes_required_before_delta():
    r = learning_order(np.stack([plateau(100), plateau(300)]), plateau(500))
    assert r['status'] == 'eligible' and r['delta'] == 400
    r = learning_order(np.stack([plateau(100), np.zeros(100001)]), plateau(500))
    assert r['status'] == 'not_evaluable' and r['delta'] is None
    r = learning_order(np.stack([plateau(100)]), np.zeros(100001))
    assert r['status'] == 'not_evaluable' and r['delta'] is None
