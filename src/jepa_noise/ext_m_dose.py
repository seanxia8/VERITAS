"""Unregistered M-dose extension: covariance and canonical-pair diagnostics.

Runbook: ``docs/jepa/RUNBOOK_2026-10-30_M_DOSE.md`` (v2, 30 Oct 2026).
Everything here is an extension (``registered: false``); no registered code is
edited.  ``m_terms`` is the verbatim copy of the inline Criterion-M computation
in ``jepa_noise.run.run_m_diagnostic`` (lines 1399-1411); test 5 pins it to the
registered E1b development score.
"""
from dataclasses import replace

import numpy as np

from jepa_noise import lucid_bombs as lb
from jepa_noise.run import _seed_entropy
from jepa_noise.views import invsqrt as _invsqrt


def m_terms(scc, stt, sct, ncc, ntt, nct, k):
    """M = sigma_k^phys - sigma_1^noise, copied from ``run.run_m_diagnostic``.

    Verbatim copy of run.py lines 1399-1411 apart from indentation: do not
    "improve" it, test 5 checks the registered +0.013519688568 value.
    """
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


def regularise(matrix):
    """The runbook epsilon: 1e-3 * trace / d * I."""
    return matrix + 1e-3 * np.trace(matrix) / len(matrix) * np.eye(len(matrix))


def project(blocks, label):
    """Registered M_reg protocol: random QR to d = min(8, d_c, d_t) per view.

    Same generator, seed and labels as ``run.run_m_diagnostic``:
    ``run._seed_entropy(2718, label)`` with label in {channel, within,
    across_fixed}.  Applied to the covariances as Q^T S Q.
    """
    scc, stt, sct, ncc, ntt, nct = blocks
    dc, dt = len(scc), len(stt)
    d = min(8, dc, dt)
    rng = np.random.default_rng(np.random.SeedSequence(_seed_entropy(2718, label)))
    qc, _ = np.linalg.qr(rng.normal(size=(dc, d)))
    qt, _ = np.linalg.qr(rng.normal(size=(dt, d)))
    return (qc.T @ scc @ qc, qt.T @ stt @ qt, qc.T @ sct @ qt,
            qc.T @ ncc @ qc, qt.T @ ntt @ qt, qc.T @ nct @ qt)


def _ratio(numerator, denominator):
    denominator = float(denominator)
    if denominator == 0.0:
        return None
    return float(numerator / denominator)


def pair_diagnostics(scc, stt, sct, ncc, ntt, nct, k, *other_cross):
    """Canonical pairs of the context/target pair and their nuisance shares.

    The pairs are built directly from the SVD of the S-whitened cross block; the
    registered ``linear.cca`` is deliberately not used (its target rows are an
    ordered basis of the span, not the canonical partners).  ``nct`` is the
    primary (N_B) noise cross block; every extra positional argument is another
    cross block (N_A, N_gain, shuffled null, ...) whose shares are reported in
    the same canonical basis, in order.

    Returns (all values unclipped):
      M_full     sigma_k^phys - sigma_1^noise
      sigma_phys k-th singular value of the S-whitened physics block
      sigma_noise top singular value of the N-whitened noise block
      ratio      r_k = sigma_1^noise / sigma_k^phys
      ratio_s    r_k^S = sigma_1(H_N^S) / sigma_k(H_P^S), both S-whitened
      canonical  top-k canonical correlations sigma_j of H_S
      context    u_j = (S_cc+eps)^-1/2 a_j, columns (d_c, k)
      target     v_j = (S_tt+eps)^-1/2 b_j, columns (d_t, k)
      shares     [nu(nct), nu(other_cross)...], each (k,)
      aggregate  nu_k = sum_j nu_j sigma_j / sum_j sigma_j (primary block)
      identity_error  max |sum_j a_j^T H_S b_j - sum_j sigma_j|
    """
    wc, wt = _invsqrt(regularise(scc)), _invsqrt(regularise(stt))
    h_signal = wc @ sct @ wt
    a, sigma, bt = np.linalg.svd(h_signal, full_matrices=False)
    if not np.isclose(np.trace(a.T @ h_signal @ bt.T), sigma.sum(), atol=1e-10, rtol=0):
        raise ArithmeticError("canonical pairing identity failed")
    identity_error = float(abs(np.trace(a.T @ h_signal @ bt.T) - sigma.sum()))
    context, target = wc @ a[:, :k], wt @ bt[:k].T

    def share(cross):
        whitened = wc @ cross @ wt
        numerator = np.einsum("ij,ij->j", a[:, :k], whitened @ bt[:k].T)
        return numerator / sigma[:k]

    sigma_phys_spectrum = np.linalg.svd(wc @ (sct - nct) @ wt, compute_uv=False)
    sigma_noise_s_spectrum = np.linalg.svd(wc @ nct @ wt, compute_uv=False)
    score, physics, noise = m_terms(scc, stt, sct, ncc, ntt, nct, k)
    primary = share(nct)
    shares = [primary] + [share(cross) for cross in other_cross]
    return {
        "M_full": score,
        "sigma_phys": float(physics[k - 1]),
        "sigma_noise": float(noise[0]),
        "ratio": _ratio(noise[0], physics[k - 1]),
        "ratio_s": _ratio(sigma_noise_s_spectrum[0], sigma_phys_spectrum[k - 1]),
        "canonical": sigma[:k],
        "context": context,
        "target": target,
        "shares": shares,
        "shares_in_unit": [bool(np.all((value >= 0) & (value <= 1)))
                           for value in shares],
        "aggregate": float(primary @ sigma[:k] / sigma[:k].sum()),
        "identity_error": identity_error,
    }


def f1x_state(geometry, spread, run_id, *, seed=lb.DEFAULT_SEED):
    """Extension-only F1x state: lognormal module gain, no clipping.

    Built without editing ``lucid_bombs``: start from the F1 state at s = 0
    (F3 at its registered s_p, module gain 1), reuse the same F1 normal draw,
    and replace the gains by exp(s * u) after the Poisson draw is made.
    """
    spread = float(spread)
    if not np.isfinite(spread) or spread < 0:
        raise ValueError("F1x spread must be finite and non-negative")
    state = lb.state_for_run("F1", 0.0, run_id, geometry, seed=seed)
    u = lb.keyed_rng(seed, lb.NS_F1, run_id).standard_normal(geometry.n_modules)
    gain = np.exp(spread * u)
    return replace(state, law="F1x", setting=spread, module_gain=gain,
                   gains=gain[geometry.module_of_pmt] * state.pmt_gain)
