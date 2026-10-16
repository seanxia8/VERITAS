"""Float64 covariance solvers returning context row functionals.

The finite-beta solver solves the reduced *global* objective with both
encoders hard-isotropic and an optimized unrestricted predictor. It is not GD.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import eigh

TOL = 1e-8


def symmetric(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError('finite square matrix required')
    if not np.allclose(a, a.T, atol=1e-12, rtol=1e-12):
        raise ValueError('matrix is not symmetric')
    return (a + a.T) / 2


def invsqrt(a):
    a = symmetric(a)
    w, v = eigh(a, driver='evr')
    if w[0] <= 0:
        raise ValueError('positive definite covariance required; no jitter permitted')
    return (v / np.sqrt(w)) @ v.T


def _sign(v):
    return v * (1 if v[np.argmax(np.abs(v))] >= 0 else -1)


def _basis(projector, count):
    """Deterministic basis from a projector, in ascending coordinate order."""
    basis = []
    for vector in projector.T:
        v = vector.copy()
        for _ in range(2):
            for b in basis:
                v -= b * (b @ v)
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            basis.append(_sign(v / norm))
        if len(basis) == count:
            break
    if len(basis) != count:
        raise ValueError('failed deterministic subspace completion')
    return np.array(basis)


def _eig(a):
    a = symmetric(a)
    w, v = eigh(a, driver='evr')
    w, v = w[::-1], v[:, ::-1]
    # Rotate only exactly equal eigenvalues. Near ties remain measurable gaps.
    first = 0
    while first < len(w):
        last = first + 1
        while last < len(w) and w[last] == w[first]:
            last += 1
        block = v[:, first:last]
        v[:, first:last] = _basis(block @ block.T, last-first).T
        first = last
    residual = np.linalg.norm(a @ v - v * w) / max(1., np.linalg.norm(a))
    if residual > TOL:
        raise ArithmeticError('eigen residual exceeds fixed tolerance')
    return w, v, residual


@dataclass
class Fit:
    context: np.ndarray
    target: np.ndarray | None = None
    predictor: np.ndarray | None = None
    spectrum: np.ndarray = field(default_factory=lambda: np.array([]))
    diagnostics: dict = field(default_factory=dict)


def _validate_k(k, dc, dt=None):
    if not isinstance(k, int) or k < 1 or k > min(dc, dt if dt is not None else dc):
        raise ValueError('invalid embedding rank')


def _complete_target(q, k):
    _, s, vt = np.linalg.svd(q, full_matrices=True)
    rank = int(np.sum(s > (s[0]*1e-12 if len(s) and s[0] > 0 else 0)))
    row = _basis(vt[:rank].T @ vt[:rank], rank) if rank else np.empty((0,q.shape[1]))
    if rank < k:
        complement = np.eye(q.shape[1]) - row.T @ row
        row = np.vstack([row, _basis(complement, k-rank)])
    return row, rank


def finite_beta(scc, stt, sct, ncc, k, beta):
    if not np.isfinite(beta) or beta < 0:
        raise ValueError('beta must be finite and nonnegative')
    hc, ht = invsqrt(scc), invsqrt(stt)
    _validate_k(k, len(hc), len(ht))
    t = hc @ np.asarray(sct, dtype=float) @ ht
    nw = hc @ symmetric(ncc) @ hc
    invsqrt(ncc)  # validate noise SPD, without changing covariance
    a = t @ t.T - beta * nw
    scores, v, residual = _eig(a)
    uc = v[:, :k].T
    ut, rank = _complete_target(uc @ t, k)
    wc, wt = uc @ hc, ut @ ht
    predictor = wt @ np.asarray(sct).T @ wc.T
    iso = max(np.linalg.norm(wc @ scc @ wc.T-np.eye(k)), np.linalg.norm(wt @ stt @ wt.T-np.eye(k)))
    objective = k - np.linalg.norm(predictor)**2 + beta*np.trace(wc@ncc@wc.T)
    reduced = k - np.sum(scores[:k])
    if iso > TOL or abs(objective-reduced) > TOL*max(1.,abs(reduced)):
        raise ArithmeticError('isotropy or reduced-objective residual exceeded')
    return Fit(wc, wt, predictor, np.linalg.svd(t,compute_uv=False),
               {'objective':float(objective), 'reduced_objective':float(reduced),
                'eigen_residual':float(residual), 'isotropy_residual':float(iso),
                'target_cross_rank':rank, 'score_eigenvalues':scores.tolist(),
                'boundary_gap':float(scores[k-1]-scores[k]) if k<len(scores) else None,
                'solver':'exact_eigen', 'status':'converged'})


def cca(scc, stt, sct, k):
    # beta=0 uses the same exact hard-isotropy solution, including null completion.
    return finite_beta(scc, stt, sct, scc, k, 0.)


def gevp(scc, ncc, k):
    h = invsqrt(scc); invsqrt(ncc); _validate_k(k,len(h))
    scores, v, residual = _eig(-h@ncc@h)
    return Fit(v[:,:k].T@h, spectrum=-scores,
               diagnostics={'eigen_residual':residual,'status':'converged'})


def pca(scc, k, ncc=None):
    h = np.eye(len(scc)) if ncc is None else invsqrt(ncc)
    _validate_k(k,len(h)); scores,v,residual = _eig(h@symmetric(scc)@h)
    return Fit(v[:,:k].T@h, spectrum=scores,
               diagnostics={'eigen_residual':residual,'status':'converged',
                            'control':'context_autoencoding'})


def sample_covariances(xc, xt, ridge_relative=0.):
    xc,xt=np.asarray(xc,float),np.asarray(xt,float)
    if xc.ndim!=2 or xt.ndim!=2 or len(xc)!=len(xt) or len(xc)<2:
        raise ValueError('paired sample matrices required')
    if ridge_relative not in (0.,1e-6):
        raise ValueError('only registered ridge rules permitted')
    xc,xt=xc-xc.mean(0),xt-xt.mean(0)
    cc,tt,ct=xc.T@xc/(len(xc)-1),xt.T@xt/(len(xc)-1),xc.T@xt/(len(xc)-1)
    for s in [cc,tt]: s += np.eye(len(s))*ridge_relative*np.trace(s)/len(s)
    return cc,tt,ct


def fit(method, scc, stt=None, sct=None, *, k, ncc=None, beta=0.):
    if method in ('cca','hard_isotropy'): return cca(scc,stt,sct,k)
    if method=='finite_beta': return finite_beta(scc,stt,sct,ncc,k,beta)
    if method=='gevp': return gevp(scc,ncc,k)
    if method=='pca': return pca(scc,k)
    if method=='whitened_pca': return pca(scc,k,ncc)
    raise ValueError(f'unknown solver {method}')


def oracle(scc, stt, sct, ncc, ntt, nct, ac, at, k):
    """Retained coordinates only; returns ineligibility without forced recovery."""
    n=np.block([[ncc,nct],[nct.T,ntt]])
    invsqrt(n)  # reject singular joint covariance before Cholesky
    dc=len(ncc)
    transform=np.linalg.solve(np.linalg.cholesky(n),np.eye(len(n)))
    s=np.block([[scc,sct],[sct.T,stt]])
    sw=transform@s@transform.T
    result=cca(sw[:dc,:dc],sw[dc:,dc:],sw[:dc,dc:],k)
    residual=at-nct.T@np.linalg.solve(ncc,ac)
    def ratio(a):
        singular=np.linalg.svd(a,compute_uv=False)
        return float(singular[k-1]/singular[0]) if len(singular)>=k and singular[0]>0 else 0.
    rc,rt=ratio(ac),ratio(residual)
    kth=float(result.spectrum[k-1]); gap=kth-(float(result.spectrum[k]) if k<len(result.spectrum) else 0.)
    rank_ok=min(rc,rt)>1e-8
    eligible=rank_ok and kth>1e-8 and gap>1e-8
    return {'status':'eligible' if eligible else ('oracle_rank_ineligible' if not rank_ok else 'oracle_canonical_ineligible'),
            'context_ratio':rc,'residual_target_ratio':rt,'kth_correlation':kth,'correlation_gap':gap,
            'transform':transform,'residual_target':residual,
            'context':result.context@transform[:dc,:dc] if eligible else None,
            'transformed_fit':result}
