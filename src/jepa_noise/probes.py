"""Grouped ridge probes; caller supplies cohorts, groups and intercept policy.

No cohort size, grouping rule or intercept convention is selected here.
The fixed alpha grid and training-only standardization follow runbook §1.3.
"""
from dataclasses import dataclass
import numpy as np

ALPHAS = (1e-6, 1e-4, 1e-2, 1., 100.)


def _matrix(value, name):
    value = np.asarray(value, dtype=float)
    if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError(f'{name} must be a nonempty finite matrix')
    return value


@dataclass
class RidgeFit:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float

    def predict(self, features):
        features = _matrix(features, 'features')
        if features.shape[1] != len(self.mean):
            raise ValueError('feature dimension mismatch')
        return ((features-self.mean)/self.scale) @ self.coefficients + self.intercept


def fit_ridge(features, target, *, alpha, fit_intercept):
    """Fit on this supplied training partition only, never a combined matrix."""
    x = _matrix(features, 'features')
    y = np.asarray(target, dtype=float)
    if y.shape != (len(x),) or not np.isfinite(y).all():
        raise ValueError('finite paired target vector required')
    if alpha not in ALPHAS or type(fit_intercept) is not bool:
        raise ValueError('registered alpha and explicit boolean intercept required')
    mean = x.mean(0)
    scale = x.std(0, ddof=0)
    constant = np.all(x == x[0], axis=0)
    mean[constant] = x[0, constant]
    scale[constant] = 1.
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('feature scaling numerically invalid')
    z = (x-mean)/scale
    intercept = float(y.mean()) if fit_intercept else 0.
    coefficients = np.linalg.solve(z.T@z + alpha*np.eye(z.shape[1]), z.T@(y-intercept))
    return RidgeFit(mean, scale, coefficients, intercept, float(alpha))


def select_alpha(scores):
    """Exact MSE ties prefer larger alpha; no unspecified near-tie tolerance."""
    if set(scores) != set(ALPHAS) or not all(np.isfinite(v) for v in scores.values()):
        raise ValueError('finite MSE for every registered alpha required')
    return min(ALPHAS, key=lambda alpha: (scores[alpha], -alpha))


def grouped_ridge(features, targets, groups, *, fit_intercept):
    """Five caller-supplied groups define five whole-group held-out folds.

    Returns per-output fits/status plus fold fit evidence. Validation losses
    are pooled over held-out events, so unequal group sizes do not silently
    reweight observations. No external probe-test features or labels accepted.
    """
    x, y = _matrix(features, 'features'), _matrix(targets, 'targets')
    group = np.asarray(groups)
    if len(y) != len(x) or group.shape != (len(x),):
        raise ValueError('paired features/targets and explicit groups required')
    if type(fit_intercept) is not bool:
        raise ValueError('explicit boolean intercept required')
    # Object/string labels are permitted, but None/nonfinite numeric groups are not.
    if any(v is None for v in group.tolist()):
        raise ValueError('missing group label')
    if group.dtype.kind in 'fc' and not np.isfinite(group).all():
        raise ValueError('nonfinite group label')
    unique = np.unique(group)
    if len(unique) != 5:
        raise ValueError('exactly five explicit groups required; no inferred folds')
    results = []
    for output in range(y.shape[1]):
        target = y[:, output]
        if np.all(target == target[0]):
            results.append({'status':'not_evaluable', 'reason':'constant_probe_train_target',
                            'alpha':None, 'fit':None, 'validation_mse':None, 'folds':[]})
            continue
        squared_errors = {alpha: 0. for alpha in ALPHAS}
        folds = []
        for held_group in unique:
            held = group == held_group
            train = ~held
            evidence = {'held_group':held_group.item() if isinstance(held_group,np.generic) else held_group,
                        'train_indices':np.flatnonzero(train), 'held_indices':np.flatnonzero(held), 'fits':{}}
            for alpha in ALPHAS:
                fitted = fit_ridge(x[train], target[train], alpha=alpha, fit_intercept=fit_intercept)
                errors = fitted.predict(x[held])-target[held]
                squared_errors[alpha] += float(errors@errors)
                evidence['fits'][alpha] = fitted
            folds.append(evidence)
        scores = {alpha: total/len(x) for alpha,total in squared_errors.items()}
        chosen = select_alpha(scores)
        fitted = fit_ridge(x, target, alpha=chosen, fit_intercept=fit_intercept)
        results.append({'status':'fit', 'reason':None, 'alpha':chosen, 'fit':fitted,
                        'validation_mse':scores, 'folds':folds})
    return results


def evaluate(probes, features, targets):
    """Pure evaluation: no changes to fitted probes, scalers or alpha choices."""
    x, y = _matrix(features, 'features'), _matrix(targets, 'targets')
    if len(x) != len(y) or y.shape[1] != len(probes):
        raise ValueError('paired evaluation features/targets required')
    rows = []
    for j, probe in enumerate(probes):
        if probe['fit'] is None:
            rows.append({'status':'not_evaluable','reason':probe['reason'],'mse':None,'r2':None})
            continue
        if np.all(y[:,j] == y[0,j]):
            rows.append({'status':'not_evaluable','reason':'constant_probe_test_target','mse':None,'r2':None})
            continue
        prediction = probe['fit'].predict(x)
        error = prediction-y[:,j]
        from .metrics import nuisance_r2
        r2 = nuisance_r2(y[:,j:j+1], prediction[:,None])['per_target'][0]
        rows.append({'status':'evaluated','reason':None,'mse':float(np.mean(error**2)), 'r2':float(r2)})
    return rows
