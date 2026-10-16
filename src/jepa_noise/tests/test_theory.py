"""Independent population checks of runbook v3; no sample-grid outcomes."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import eigh, subspace_angles

ATOL = 1e-8


def invsqrt(a):
    vals, vecs = eigh(a)
    if vals[0] <= 0:
        raise ValueError('covariance_not_spd')
    return (vecs / np.sqrt(vals)) @ vecs.T


def cca(sc, st, cross, k):
    qc, qt = invsqrt(sc), invsqrt(st)
    u, rho, _ = np.linalg.svd(qc @ cross @ qt)
    return qc @ u[:, :k], rho


def oracle(nc, nt, nct, ac, at, theta):
    n = np.block([[nc, nct], [nct.T, nt]])
    if np.linalg.eigvalsh(n)[0] <= 0:
        raise ValueError('joint_noise_not_spd')
    d = at - nct.T @ np.linalg.solve(nc, ac)
    ratios = []
    for a in (ac, d):
        s = np.linalg.svd(a, compute_uv=False)
        ratios.append(0.0 if s[0] == 0 else s[-1]/s[0])
    l = np.linalg.cholesky(n)
    t = np.linalg.solve(l, np.eye(len(n)))
    a = t @ np.vstack((ac, at))
    s = a @ theta @ a.T + np.eye(len(n))
    dc, k = ac.shape
    w, rho = cca(s[:dc, :dc], s[dc:, dc:], s[:dc, dc:], k)
    gap = rho[k-1] - (rho[k] if len(rho) > k else 0)
    eligible = min(ratios) > 1e-8 and rho[k-1] > 1e-8 and gap > 1e-8
    return eligible, t[:dc, :dc].T @ w, t, ratios, rho[k-1], gap


def fixture():
    ac = np.array([[1., .2], [.3, 1.], [.2, -.4], [.1, .3]])
    at = np.array([[.8, .1], [.2, 1.2], [.4, -.3], [.2, .2]])
    nc = np.diag([1., 1.2, 1.4, .8])
    nt = np.diag([1.1, .9, 1.3, 1.2])
    theta = np.diag([.7, 1.3])
    return ac, at, nc, nt, theta


def test_p1_factor_assumptions_and_displacement():
    a = np.array([[1.], [0.]])
    b = np.array([[0.], [1.]])
    load = np.column_stack((a, b))
    independent = np.diag([.01, .99])
    assert_allclose(load @ independent @ load.T, .01*a@a.T + .99*b@b.T, atol=ATOL)
    correlated = independent + np.array([[0, .03], [.03, 0]])
    assert np.linalg.eigvalsh(correlated).min() > 0
    assert not np.allclose(load @ correlated @ load.T, .01*a@a.T + .99*b@b.T)
    w, _ = cca(np.eye(2), np.eye(2), independent, 1)
    assert subspace_angles(w, b).max() < 1e-6


def test_p2_invertible_map_invariance_and_pullback():
    ac, at, nc, nt, theta = fixture()
    sc, st, cross = ac@theta@ac.T+nc, at@theta@at.T+nt, ac@theta@at.T
    tc = np.array([[2., .2, 0, 0], [0, .7, .1, 0], [0, 0, 1.3, .3], [.1, 0, 0, .9]])
    tt = tc.T + .2*np.eye(4)
    w, rho = cca(sc, st, cross, 2)
    wt, rhot = cca(tc@sc@tc.T, tt@st@tt.T, tc@cross@tt.T, 2)
    assert_allclose(rho, rhot, atol=ATOL, rtol=0)
    assert subspace_angles(w, tc.T@wt).max() < 1e-6


def test_p3_woodbury_and_p4_generalized_eigenspace():
    ac, at, nc, nt, theta = fixture()
    sc, st = ac@theta@ac.T+nc, at@theta@at.T+nt
    ref = np.linalg.solve(nc, ac)
    rhs = ref @ np.linalg.inv(np.eye(2)+theta@ac.T@ref)
    assert_allclose(np.linalg.solve(sc, ac), rhs, atol=ATOL, rtol=0)
    w, rho = cca(sc, st, ac@theta@at.T, 2)
    assert rho[1] > 1e-8 and rho[1]-rho[2] > 1e-8
    assert subspace_angles(w, ref).max() < 1e-6
    vals, vectors = eigh(nc, sc)
    assert vals[1] < 1-1e-8
    assert_allclose(vals[2:], 1., atol=ATOL, rtol=0)
    assert subspace_angles(vectors[:, :2], ref).max() < 1e-6
    assert_allclose(vectors.T@sc@vectors, np.eye(4), atol=ATOL, rtol=0)


def test_p4_overlay_identity_by_exact_symmetric_distribution():
    w = np.array([[1., 2.], [-.3, .8]])
    n = np.array([[2., .4], [.4, 1.]])
    l = np.linalg.cholesky(n)
    draws = np.sqrt(2)*np.vstack((l.T, -l.T))
    assert_allclose(draws.mean(axis=0), 0, atol=ATOL)
    assert_allclose(draws.T@draws/4, n, atol=ATOL)
    assert_allclose(np.mean(np.sum((draws@w.T)**2, axis=1)), np.trace(w@n@w.T), atol=ATOL)


@pytest.mark.parametrize('beta', [.1, 1., 10.])
def test_finite_beta_global_nuisance_optimum_with_both_isotropies(beta):
    cross = np.diag([.01, .99])
    noise = np.diag([.99, 1.])
    nct = np.diag([0., .99])
    assert np.linalg.eigvalsh(np.block([[noise, nct], [nct.T, noise]])).min() > 0
    scores, vecs = eigh(cross@cross.T-beta*noise)
    wc = vecs[:, -1:].T
    wt = wc@cross
    wt /= np.linalg.norm(wt)
    predictor = wc@cross@wt.T
    assert_allclose(wc@wc.T, np.eye(1), atol=ATOL)
    assert_allclose(wt@wt.T, np.eye(1), atol=ATOL)
    objective = 1-float((predictor**2).sum())+beta*float((wc@noise@wc.T).item())
    assert_allclose(objective, 1-scores[-1], atol=ATOL)
    assert subspace_angles(wc.T, np.array([[0.], [1.]])).max() < 1e-6
    assert 1-.99**2+beta < 1-.01**2+.99*beta


def test_p5_eligible_mapping_covariance_and_context_visibility():
    ac, at, nc, nt, theta = fixture()
    nct = .15*np.eye(4)
    eligible, w, t, ratios, rho, gap = oracle(nc, nt, nct, ac, at, theta)
    assert eligible and min(ratios) > 1e-8 and min(rho, gap) > 1e-8
    n = np.block([[nc, nct], [nct.T, nt]])
    assert_allclose(t@n@t.T, np.eye(8), atol=ATOL, rtol=0)
    assert_allclose(t[:4, 4:], 0., atol=ATOL, rtol=0)
    assert subspace_angles(w, np.linalg.solve(nc, ac)).max() < 1e-6


def test_p5_registered_annihilation_is_ineligible():
    nc, nt, nct = np.diag([2., 1.]), np.diag([1.25, 1.]), np.diag([.5, 0.])
    ac, at = np.array([[1.], [0.]]), np.array([[.25], [0.]])
    eligible, _, t, ratios, rho, gap = oracle(nc, nt, nct, ac, at, np.eye(1))
    assert not eligible and ratios[1] == 0
    assert_allclose((t@np.vstack((ac, at)))[2:], 0., atol=ATOL)
    assert_allclose([rho, gap], 0., atol=ATOL)


def test_p5_singular_joint_rejected_despite_spd_marginals():
    n = np.eye(2)
    a = np.array([[1.], [0.]])
    with pytest.raises(ValueError, match='joint_noise_not_spd'):
        oracle(n, n, n, a, a, np.eye(1))


def test_general_finite_beta_elimination_including_target_rank_deficiency():
    # k=2 but rank(T)=1: target completion must retain hard isotropy.
    t = np.array([[.6, 0., 0.], [0., 0., 0.], [.2, 0., 0.]])
    nw = np.diag([.5, .7, .9])
    beta, k = 1., 2
    values, vectors = eigh(t@t.T-beta*nw)
    uc = vectors[:, -k:].T
    _, _, vh = np.linalg.svd(uc@t, full_matrices=True)
    ut = vh[:k]
    p = ut@t.T@uc.T
    mse = k-2*np.trace(p@uc@t@ut.T)+np.trace(p@p.T)
    penalty = beta*np.trace(uc@nw@uc.T)
    assert_allclose(uc@uc.T, np.eye(k), atol=ATOL)
    assert_allclose(ut@ut.T, np.eye(k), atol=ATOL)
    assert_allclose(mse+penalty, k-values[-k:].sum(), atol=ATOL)
    assert np.linalg.matrix_rank(p) == 1
