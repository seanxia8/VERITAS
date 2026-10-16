"""Subspace and nuisance diagnostics with explicit undefined target handling."""
import math
import numpy as np
from scipy.linalg import subspace_angles


def principal_angles(a,b,degrees=False):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if np.linalg.matrix_rank(a)!=len(a) or np.linalg.matrix_rank(b)!=len(b):
        raise ValueError('rank deficient row functionals')
    angles=subspace_angles(a.T,b.T)
    return np.degrees(angles) if degrees else angles


def nuisance_r2(truth, prediction):
    truth, prediction = np.asarray(truth, float), np.asarray(prediction, float)
    if truth.shape != prediction.shape or truth.ndim != 2 or len(truth) < 2:
        raise ValueError('paired target matrices with at least two rows required')
    if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError('nonfinite nuisance truth or prediction')
    # Decide constantness from observations, not rounded centering arithmetic.
    # Exact comparison preserves even representable, extremely small variation.
    valid = np.any(truth != truth[0], axis=0)
    scores = np.full(truth.shape[1], np.nan)
    if valid.any():
        # Extended precision avoids avoidable centering/overflow error; an
        # unrepresentable metric raises rather than masquerading as a constant.
        values = truth[:, valid].astype(np.longdouble)
        fitted = prediction[:, valid].astype(np.longdouble)
        denom = np.sum((values - values.mean(0)) ** 2, axis=0)
        if np.any(denom <= 0) or not np.isfinite(denom).all():
            raise ValueError('nonconstant nuisance variance numerically invalid')
        result = 1 - np.sum((values - fitted) ** 2, axis=0) / denom
        with np.errstate(over='ignore', invalid='ignore'):
            scores[valid] = result
        if not np.isfinite(scores[valid]).all():
            raise ValueError('nuisance R2 numerically invalid')
    return {'per_target': scores, 'valid': valid,
            'mean': float(np.mean(scores[valid])) if valid.any() else np.nan}


def learning_time(contribution):
    """Registered R5 absolute mode contribution, indexed by step 0..100000.

    A complete curve contains initialization plus every scheduled GD step.
    No horizon shortening, normalization or replacement of censored times.
    """
    q = np.asarray(contribution, dtype=float)
    if q.ndim != 1:
        raise ValueError('one absolute-contribution curve required')
    result = {'status': 'not_evaluable', 'eligible': False, 'censored': True,
              'learning_time': None, 'reason': None}
    if len(q) != 100001:
        return {**result, 'reason': 'incomplete_or_invalid_horizon'}
    if not np.isfinite(q).all():
        return {**result, 'reason': 'nonfinite_contribution'}
    if np.any(q < 0):
        return {**result, 'reason': 'negative_absolute_contribution'}
    q0 = float(q[0])
    qf = math.fsum(q[99001:100001]) / 1000
    previous = math.fsum(q[98001:99001]) / 1000
    result.update(q0=q0, qf=qf, terminal_previous_mean=previous,
                  terminal_difference=abs(previous-qf))
    if qf <= 1e-6:
        return {**result, 'reason': 'below_absolute_floor'}
    if qf < 2*q0:
        return {**result, 'reason': 'insufficient_absolute_growth'}
    if abs(previous-qf) > .05*qf:
        return {**result, 'reason': 'terminal_not_stabilized'}
    threshold = q0 + .5*(qf-q0)
    result['threshold'] = threshold
    # Cumulative counts identify the first complete run of >=100 steps.
    above = q >= threshold
    counts = np.r_[0, np.cumsum(above, dtype=np.int64)]
    crossings = np.flatnonzero(counts[100:] - counts[:-100] == 100)
    if not len(crossings):
        return {**result, 'reason': 'no_sustained_crossing'}
    return {**result, 'status': 'eligible', 'eligible': True, 'censored': False,
            'learning_time': int(crossings[0]), 'reason': None}


def learning_order(physics_curves, nuisance_curve):
    """R5 delta only after *every* physics and nuisance mode is eligible."""
    physics = np.asarray(physics_curves, dtype=float)
    if physics.ndim != 2 or len(physics) == 0:
        raise ValueError('physics curves require shape (modes, steps), modes > 0')
    results = [learning_time(q) for q in physics]
    nuisance = learning_time(nuisance_curve)
    if not all(r['eligible'] for r in [*results, nuisance]):
        return {'status': 'not_evaluable', 'reason': 'mode_learning_ineligible',
                'delta': None, 'physics': results, 'nuisance': nuisance}
    return {'status': 'eligible', 'reason': None,
            'delta': nuisance['learning_time'] - min(r['learning_time'] for r in results),
            'physics': results, 'nuisance': nuisance}
