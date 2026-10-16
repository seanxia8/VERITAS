"""A2 exact empirical-moment tied stop-gradient deep-linear dynamics.

This engine does not authorize execution. Campaign callers must pass reviewed
config/data manifests and a remaining resource allocation before run().
"""
from dataclasses import dataclass
import time
import numpy as np
from .masks import namespace_rng
from .metrics import learning_order


@dataclass(frozen=True)
class Moments:
    cc: np.ndarray
    tt: np.ndarray
    ct: np.ndarray  # E[x_c x_t.T]
    n: int | None
    source: str
    context_mean: np.ndarray | None = None
    target_mean: np.ndarray | None = None


def empirical_moments(context, target):
    c, t = np.asarray(context, dtype=float), np.asarray(target, dtype=float)
    if c.ndim != 2 or c.shape != t.shape or len(c) < 2:
        raise ValueError('paired square-view sample matrices required')
    if not np.isfinite(c).all() or not np.isfinite(t).all():
        raise ValueError('nonfinite training samples')
    cm, tm = c.mean(axis=0), t.mean(axis=0)
    c, t = c-cm, t-tm
    n = len(c)
    return Moments(c.T@c/n, t.T@t/n, c.T@t/n, n,
                   'fixed_centered_empirical_divisor_n', cm, tm)


def diagonal_fixture():
    """Population diagonal construction for formulas, never campaign data.

    Positive diagonal marginals/cross-moments and diagonal initialization permit
    coordinatewise decoupled dynamics. No random-loading theorem is asserted.
    """
    cc = np.diag([2., 1.5, 1., .7])
    ct = np.diag([1.4, .9, .3, .1])
    return Moments(cc, cc.copy(), ct, None, 'population_diagonal_engineering_only')


def initialize(seed, depth, d=40):
    if depth not in (1, 2, 3) or d < 1:
        raise ValueError('registered depth is 1, 2 or 3; positive d required')
    layers = [namespace_rng(seed, f'E1c/init/layer/{i}').normal(0., .001, size=(d,d))
              for i in range(depth)]
    predictor = namespace_rng(seed, 'E1c/init/predictor').normal(0., .001, size=(d,d))
    return layers, predictor


def encoder(layers):
    if not layers:
        raise ValueError('at least one square encoder layer required')
    d = np.asarray(layers[0]).shape[0]
    w = np.eye(d)
    for layer in layers:
        if np.asarray(layer).shape != (d,d):
            raise ValueError('square equal-width layers required')
        w = layer@w
    return w


def loss_and_gradients(layers, predictor, moments, fixed_target=None):
    """Half sum-coordinate MSE, differentiating ONLY prediction branch.

    Target defaults to a value-copy of current encoder, then is held fixed for
    differentiation. Explicit fixed_target enables honest finite differences.
    """
    w = encoder(layers)
    p = np.asarray(predictor, dtype=float)
    target = w.copy() if fixed_target is None else np.asarray(fixed_target, dtype=float)
    if p.shape != w.shape or target.shape != w.shape:
        raise ValueError('predictor/target must be square encoder dimension')
    cc, tt, ct = moments.cc, moments.tt, moments.ct
    if any(np.asarray(s).shape != w.shape for s in (cc,tt,ct)):
        raise ValueError('moment dimensions differ from encoder')
    a = p@w
    loss = .5*np.trace(a@cc@a.T)-np.trace(a@ct@target.T)+.5*np.trace(target@tt@target.T)
    ga = a@cc-target@ct.T
    gp = ga@w.T
    gw = p.T@ga
    prefix = [np.eye(len(w))]
    for layer in layers:
        prefix.append(layer@prefix[-1])
    suffix = np.eye(len(w))
    gradients = [None]*len(layers)
    for i in range(len(layers)-1,-1,-1):
        gradients[i] = suffix.T@gw@prefix[i].T
        suffix = suffix@layers[i]
    return float(loss), gradients, gp


def mode_contributions(layers, ac, bc, theta_sd, eta_sd):
    w = encoder(layers)
    ac, bc = np.asarray(ac), np.asarray(bc)
    if bc.ndim != 2 or bc.shape[1] != 1:
        raise ValueError('E1c has one separately declared nuisance loading')
    physics = np.linalg.norm(w@ac, axis=0)*np.asarray(theta_sd)
    nuisance = float(np.linalg.norm(w@bc)*eta_sd)
    norm = float(np.linalg.norm(w))
    absolute = np.r_[physics,nuisance]
    normalized = absolute/norm if norm > 0 else np.full_like(absolute,np.nan)
    return absolute, normalized, norm


def prepare_campaign_moments(seed, gamma_load):
    """Outcome-producing training-data construction; never called by unit tests.

    Loading seed preserves the ExactModel seed population. Only its sample RNG
    is replaced by the independent E1c namespace; all depths share returned data.
    """
    from .views import ExactModel
    if seed not in range(5) or gamma_load not in (.5, 1.):
        raise ValueError('unregistered E1c seed/gamma')
    model = ExactModel(seed=seed, theta_scale=.4, eta_scale=3., share='target')
    model.rng = namespace_rng(seed, 'E1c/data', cell=f'gamma_load={gamma_load:g}')
    initial_state = model.rng.bit_generator.state
    sample = model.sample(100000, gamma_load=gamma_load)
    moments = empirical_moments(sample['context'], sample['target'])
    pop = sample['population']
    provenance = {'seed':seed,'gamma_load':gamma_load,'n':100000,
                  'loading_seed':seed,'role':'E1c/data','cell':f'gamma_load={gamma_load:g}',
                  'rng_initial_state':initial_state,'rng_final_state':sample['rng_state'],
                  'source':'fixed finite batch; not population covariance',
                  'auxiliary_calibration_draws':'ExactModel.sample draws these; unused in E1c and charged'}
    return moments, pop['ac'], pop['bc'], np.full(3,.4), 3., provenance


def run(moments, layers, predictor, ac, bc, theta_sd, eta_sd,
        *, budget_seconds, engineering_steps=None):
    """Simultaneous full-batch GD at lr=.001, preserving every step and STOP.

    Production horizon is always 100000. A shorter engineering invocation is
    explicitly incomplete and cannot yield an R5 outcome. Caller owns ledger.
    """
    if not np.isfinite(budget_seconds) or budget_seconds <= 0:
        raise ValueError('positive remaining CPU budget required')
    if engineering_steps is not None and (not isinstance(engineering_steps,int) or not 0<=engineering_steps<100000):
        raise ValueError('engineering horizon must be integer 0..99999')
    steps = 100000 if engineering_steps is None else engineering_steps
    if engineering_steps is None and (len(layers) not in (1,2,3) or np.shape(predictor)!=(40,40)
                                      or moments.n!=100000 or moments.source!='fixed_centered_empirical_divisor_n'):
        raise ValueError('campaign requires A2 dimensions and fixed n100000 empirical moments')
    layers = [np.array(x,dtype=float,copy=True) for x in layers]
    predictor = np.array(predictor,dtype=float,copy=True)
    start = time.monotonic()
    absolute, normalized, norms, losses = [], [], [], []
    status, reason = 'complete', None
    completed = 0
    with np.errstate(over='ignore', invalid='ignore'):
        for step in range(steps+1):
            q, relative, norm = mode_contributions(layers, ac, bc, theta_sd, eta_sd)
            absolute.append(q); normalized.append(relative); norms.append(norm)
            loss, gl, gp = loss_and_gradients(layers,predictor,moments)
            losses.append(loss)
            if not (np.isfinite(q).all() and np.isfinite(loss) and
                    np.isfinite(predictor).all() and all(np.isfinite(g).all() for g in gl) and np.isfinite(gp).all()):
                status, reason = 'failed','numerical_instability'
                break
            if step == steps:
                break
            if time.monotonic()-start >= budget_seconds:
                status, reason = 'stopped','resource_cap'
                break
            # All derivatives are at the same pre-update iterate and frozen target.
            layers = [x-.001*g for x,g in zip(layers,gl)]
            predictor = predictor-.001*gp
            completed += 1
    if status=='complete' and engineering_steps is not None:
        status,reason='engineering_incomplete','short_engineering_horizon'
    curves=np.array(absolute)
    outcome=learning_order(curves[:,:-1].T,curves[:,-1])
    return {'status':status,'reason':reason,'completed_steps':completed,
            'layers':layers,'predictor':predictor,'absolute':curves,
            'normalized_diagnostic':np.array(normalized),'encoder_norm':np.array(norms),
            'loss':np.array(losses),'R5':outcome,'elapsed_seconds':time.monotonic()-start,
            'learning_rate':.001,'loss_reduction':'half mean events sum coordinates',
            'engineering_only':engineering_steps is not None}
