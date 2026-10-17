"""Shared fixtures and estimators for the 2026-10-17 E1 finite-sample diagnostics.

Read-only use of the registered code: no file under ``src/jepa_noise`` is edited.
The E1a path reproduces ``run.run_e1a_row``'s draw law and the E1b path
reproduces ``views.Tier1Views``' draw law (see the notes in each function).

Writes: nothing (import-only helper).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg.blas import dgemm, dsyrk

from jepa_noise import linear
from jepa_noise.metrics import principal_angles

HERE = Path(__file__).resolve().parent
OUT = HERE


def gram(X):
    """Exact X^T X: BLAS syrk fills one triangle only, so mirror explicitly."""
    S = dsyrk(1.0, np.asarray(X, dtype=float), trans=1, lower=1)
    return np.tril(S) + np.tril(S, -1).T


def ridge(covariance, relative: float = 1e-6):
    """Registered CCA marginal ridge (``run.ridge_covariance``)."""
    covariance = np.asarray(covariance, dtype=float)
    return covariance + np.eye(len(covariance)) * relative * np.trace(covariance) / len(covariance)


def cca_angle(scc, stt, sct, reference, k: int = 3):
    """Registered r1_cca row: ridge on marginals, unridged cross term.

    Returns (max principal angle in degrees vs ``reference``, Fit).
    """
    fit = linear.cca(ridge(scc), ridge(stt), np.asarray(sct, dtype=float), k)
    angle = float(np.max(principal_angles(fit.context, reference, degrees=True)))
    return angle, fit


def population_cca_angle(population, reference, k: int = 3):
    fit = linear.cca(population["scc"], population["stt"], population["sct"], k)
    return float(np.max(principal_angles(fit.context, reference, degrees=True))), fit


def subspace_floor(a, b):
    """Max principal angle (deg) between two estimated row spaces (k x d)."""
    return float(np.max(principal_angles(np.asarray(a), np.asarray(b), degrees=True)))


# ---------------------------------------------------------------------------
# E1a: ExactModel draw law, chunked for n = 1e6
# ---------------------------------------------------------------------------
def e1a_covariances_chunked(model, n: int, gamma_load: float, rng, chunk: int = 20000):
    """Chunked ``ExactModel.sample`` sufficient statistics.

    Draw order is theta, eta, private context, private target per chunk; the law
    is exactly ``ExactModel.sample`` (same ``population`` blocks).  The draw
    interleaving differs from the unchunked method, so this is a new
    independent realisation of the same law, not a bitwise replay.
    """
    p = model.population(gamma_load)
    d, k = model.d, model.k
    chol = model.chol
    bc, bt = p["bc"], p["bt"]
    sums = dict(c=np.zeros(d), t=np.zeros(d))
    xx_c = np.zeros((d, d))
    xx_t = np.zeros((d, d))
    ct = np.zeros((d, d))
    done = 0
    while done < n:
        m = min(chunk, n - done)
        theta = rng.normal(size=(m, k)) * model.theta_scale
        eta = rng.normal(size=(m, 1)) * model.eta_scale
        nc = rng.normal(size=(m, d)) @ chol.T
        nt = rng.normal(size=(m, d)) @ chol.T
        xc = theta @ model.ac.T + eta @ bc.T + nc
        xt = theta @ model.at.T + eta @ bt.T + nt
        sums["c"] += xc.sum(0)
        sums["t"] += xt.sum(0)
        xx_c += gram(xc)
        xx_t += gram(xt)
        ct += dgemm(1.0, xc, xt, trans_a=1)
        done += m
    scc = (xx_c - np.outer(sums["c"], sums["c"]) / n) / (n - 1)
    stt = (xx_t - np.outer(sums["t"], sums["t"]) / n) / (n - 1)
    sct = (ct - np.outer(sums["c"], sums["t"]) / n) / (n - 1)
    for a in (scc, stt):
        a[:] = (a + a.T) * 0.5
    return scc, stt, sct


# ---------------------------------------------------------------------------
# E1b: A4 channel-split clean-cell draw law, vectorised over events
# ---------------------------------------------------------------------------
def e1b_population(views):
    """A4.5/E1b population blocks with the registered theta_sd diagonal."""
    population = dict(views.population())
    population["sigma_theta"] = np.diag(np.asarray(views.theta_sd, dtype=float) ** 2)
    return population


def e1b_covariances_chunked(views, n: int, rng, chunk: int = 5000):
    """Chunked sufficient statistics for the A4 clean cell (g=0, line=0).

    Event law of ``Tier1Views.sample(signal_mode='linearized')``:

        signal = j0 @ delta,  delta ~ N(0, diag(theta_sd^2))
        noise  = sum_{white,colored} private_weight[c] * record_{c}(t) + eps

    with ``record = irfft(rfft(standard normal) * sqrt(temporal_eigenvalue))``
    and ``eps ~ N(0, epsilon_dc)``.  The per-event namespace RNG is replaced by
    one bulk generator: the resulting law (i.i.d. Gaussian records with the same
    spectral shaping) is identical, the realisation is independent.
    ``common_strength`` and ``line_multiplier`` are 0 for the clean cell, so the
    shared/line/common components are exactly zero and are not drawn.
    """
    assert views.line_multiplier == 0. and views.common_strength == 0.
    assert views.base_rho == 0.
    split = views.split
    c_idx, t_idx = split.context, split.target
    pw = np.asarray(views.private_weight, dtype=float)
    eps = float(views.epsilon_dc)
    j0 = np.asarray(views.j0, dtype=float)
    theta_sd = np.asarray(views.theta_sd, dtype=float)
    spectra = (np.asarray(views.white_eigenvalues, dtype=float),
               np.asarray(views.colored_eigenvalues, dtype=float))
    dc, dt = len(c_idx), len(t_idx)
    sum_c = np.zeros(dc)
    sum_t = np.zeros(dt)
    xx_c = np.zeros((dc, dc))
    xx_t = np.zeros((dt, dt))
    ct = np.zeros((dc, dt))
    done = 0
    while done < n:
        m = min(chunk, n - done)
        delta = rng.normal(size=(m, 3)) * theta_sd
        signal = delta @ j0.T
        noise = np.empty_like(signal)
        noise[:] = np.sqrt(eps) * rng.normal(size=(m, 2048))
        for eigen in spectra:
            z = rng.normal(size=(m, 8, 256))
            rec = np.fft.irfft(np.fft.rfft(z, axis=-1) * np.sqrt(eigen), n=256, axis=-1)
            rec *= pw[None, :, None]
            noise += rec.reshape(m, 2048)
            del z, rec
        values = signal + noise
        del signal, noise
        xc = values[:, c_idx]
        xt = values[:, t_idx]
        sum_c += xc.sum(0)
        sum_t += xt.sum(0)
        xx_c += gram(xc)
        xx_t += gram(xt)
        ct += dgemm(1.0, xc, xt, trans_a=1)
        done += m
    scc = (xx_c - np.outer(sum_c, sum_c) / n) / (n - 1)
    stt = (xx_t - np.outer(sum_t, sum_t) / n) / (n - 1)
    sct = (ct - np.outer(sum_c, sum_t) / n) / (n - 1)
    for a in (scc, stt):
        a[:] = (a + a.T) * 0.5
    return scc, stt, sct


def dump(path: Path, payload):
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")


class timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
        print(f"[time] {self.label}: {self.elapsed:.1f}s", flush=True)
