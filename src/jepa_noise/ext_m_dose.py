"""Unregistered M-dose extension: covariance and canonical-pair diagnostics."""
from dataclasses import replace

import numpy as np

from jepa_noise import lucid_bombs as lb
from jepa_noise.views import invsqrt as _invsqrt
from jepa_noise.run import _seed_entropy


def m_terms(scc, stt, sct, ncc, ntt, nct, k):
    # Copied verbatim from run_m_diagnostic, apart from indentation.
    def eps(matrix):
        return 1e-3 * np.trace(matrix) / len(matrix) * np.eye(len(matrix))
    h_signal = (_invsqrt(scc + eps(scc))
                @ (sct - nct)
                @ _invsqrt(stt + eps(stt)))
    h_noise = (_invsqrt(ncc + eps(ncc))
               @ nct
               @ _invsqrt(ntt + eps(ntt)))
    signal_singular = np.linalg.svd(h_signal, compute_uv=False)
    noise_singular = np.linalg.svd(h_noise, compute_uv=False)
    score = float(signal_singular[k - 1] - noise_singular[0])
    return score, signal_singular, noise_singular


def regularise(a):
    return a + 1e-3 * np.trace(a) / len(a) * np.eye(len(a))


def project(blocks, label):
    scc, stt, sct, ncc, ntt, nct = blocks
    dc, dt = len(scc), len(stt)
    d = min(8, dc, dt)
    rng = np.random.default_rng(np.random.SeedSequence(_seed_entropy(2718, label)))
    qc, _ = np.linalg.qr(rng.normal(size=(dc, d)))
    qt, _ = np.linalg.qr(rng.normal(size=(dt, d)))
    return (qc.T @ scc @ qc, qt.T @ stt @ qt, qc.T @ sct @ qt,
            qc.T @ ncc @ qc, qt.T @ ntt @ qt, qc.T @ nct @ qt)


def pair_diagnostics(scc, stt, sct, ncc, ntt, nct, k, *other_cross):
    wc, wt = _invsqrt(regularise(scc)), _invsqrt(regularise(stt))
    hs = wc @ sct @ wt
    a, sig, bt = np.linalg.svd(hs, full_matrices=False)
    if not np.isclose(np.trace(a.T @ hs @ bt.T), sig.sum(), atol=1e-10, rtol=0):
        raise ArithmeticError("canonical pairing identity failed")
    def share(cross):
        numerator = np.einsum("ij,ij->j", a[:, :k], (wc @ cross @ wt) @ bt[:k].T)
        return numerator / sig[:k]
    sigma_p = np.linalg.svd(wc @ (sct - nct) @ wt, compute_uv=False)
    sigma_n_s = np.linalg.svd(wc @ nct @ wt, compute_uv=False)
    score, ps, ns = m_terms(scc, stt, sct, ncc, ntt, nct, k)
    primary = share(nct)
    return dict(M_full=score, sigma_phys=float(ps[k-1]), sigma_noise=float(ns[0]),
                ratio=float(ns[0]/ps[k-1]), ratio_s=float(sigma_n_s[0]/sigma_p[k-1]),
                canonical=sig[:k], context=wc @ a[:, :k], target=wt @ bt[:k].T,
                shares=[primary] + [share(cross) for cross in other_cross],
                aggregate=float(primary @ sig[:k] / sig[:k].sum()))


def f1x_state(geometry, spread, run_id, *, seed=lb.DEFAULT_SEED):
    state = lb.state_for_run("F1", 0.0, run_id, geometry, seed=seed)
    u = lb.keyed_rng(seed, lb.NS_F1, run_id).standard_normal(geometry.n_modules)
    gain = np.exp(float(spread) * u)
    return replace(state, law="F1x", setting=float(spread), module_gain=gain,
                   gains=gain[geometry.module_of_pmt] * state.pmt_gain)
