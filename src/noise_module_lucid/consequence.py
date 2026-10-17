# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The LUCiD arm's consequence variable K (12 Oct 2026; open item §II.9 / D-K).

K must be a physical endpoint computed **independently of the monitored model**
(the model whose representations Claims 1 and 2 read). For the LUCiD arm it is
the error of frozen, declared reconstructions on the per-PMT digits, with two
endpoints:

* **vertex** — a time-of-flight vertex fit: ``err_v = |v_hat - v_true|`` (m);
* **energy** — the total measured charge times a calibration constant fixed on
  the reference cell (the training-time energy scale):
  ``err_E = |c * Q_total - E_true| / E_true``;

    K_x(cell) = mean_e err_x(cell, e) / mean_e err_x(reference, e)
    K(cell)   = max(K_vertex, K_energy)          (draft primary; D-K)

over the same events, the same convention as Tier 1's amplitude K (relative to
the clean reference on the same events; ``PREREGISTRATION.md`` §5). The fitter
is ours (written here; LUCiD's own ``recon.py`` is not used, both to keep the endpoint independent, and because the monitored model may be trained on the
same observables, so an independent endpoint is wanted).

The fitter
----------
Point-source multilateration on first-arrival times. For a trial vertex ``v``
the emission-time residual of digit ``i`` is ``r_i = t_i - |P_i - v| / v_g``
with ``v_g`` the group velocity of light in water. Scattered and reflected light
arrives late, so the loss is one-sided-robust: ``t0`` is profiled as a low
quantile of ``r`` and each digit contributes ``w_i * log(1 + ((r_i - t0)/s)^2)``
(Cauchy), with ``w_i = sqrt(charge_i)``. A coarse grid over the detector volume
picks the start; Nelder–Mead polishes it. The result is a declared function of
the digits only — frozen, so K differences come from the readout cell alone.

Limits (declared): a point-source model; for long tracks it measures a
light-weighted emission point, not the track start — not a validated track-start endpoint; scoring tracks requires a separate target and reconstruction. Development values are non-citable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

#: speed of light in vacuum [m/ns] / group index of water (~1.38 near 400 nm)
V_GROUP_M_PER_NS = 0.299792458 / 1.38


@dataclass(frozen=True)
class FitConfig:
    v_group: float = V_GROUP_M_PER_NS
    scale_ns: float = 3.0          # Cauchy scale of the time residual
    t0_quantile: float = 0.2       # early-edge reference for t0
    grid: int = 9                  # coarse grid points per axis
    min_digits: int = 6
    max_iter: int = 400


@dataclass
class VertexFit:
    vertex: np.ndarray
    t0: float
    loss: float
    n_digits: int
    ok: bool


def _loss(v, P, T, w, cfg: FitConfig):
    r = T - np.linalg.norm(P - v, axis=1) / cfg.v_group
    t0 = np.quantile(r, cfg.t0_quantile)
    u = (r - t0) / cfg.scale_ns
    return float(np.sum(w * np.log1p(u * u))), float(t0)


def fit_vertex(positions, sensor, time, charge, cfg: FitConfig | None = None,
               bounds: tuple[float, float] | None = None) -> VertexFit:
    """Fit a point-source vertex and t0 to per-PMT digits.

    ``positions`` is ``(n_channels, 3)`` in metres; ``bounds`` is
    ``(radius, half_height)`` of the search cylinder (default: the PMT array).
    """
    cfg = cfg or FitConfig()
    pos = np.asarray(positions, dtype=float)
    s = np.asarray(sensor, dtype=int)
    T = np.asarray(time, dtype=float)
    q = np.asarray(charge, dtype=float)
    if s.size < cfg.min_digits:
        return VertexFit(np.full(3, np.nan), float("nan"), float("nan"), int(s.size), False)
    if T.shape != s.shape or q.shape != s.shape or not np.all(np.isfinite(T)) or not np.all(np.isfinite(q)) or np.any(q <= 0):
        return VertexFit(np.full(3, np.nan), float("nan"), float("nan"), int(s.size), False)
    P = pos[s]
    w = np.sqrt(np.clip(q, 0.0, None))
    R, H = bounds if bounds is not None else (float(np.hypot(pos[:, 0], pos[:, 1]).max()),
                                             float(np.abs(pos[:, 2]).max()))
    g = np.linspace(-1.0, 1.0, cfg.grid)
    gx, gy, gz = np.meshgrid(g * R, g * R, g * H, indexing="ij")
    cand = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], 1)
    cand = cand[np.hypot(cand[:, 0], cand[:, 1]) <= R]
    losses = [_loss(c, P, T, w, cfg)[0] for c in cand]
    start = cand[int(np.argmin(losses))]
    res = minimize(lambda v: _loss(v, P, T, w, cfg)[0], start, method="Nelder-Mead",
                   options={"maxiter": cfg.max_iter, "xatol": 1e-3, "fatol": 1e-6,
                            "initial_simplex": np.vstack([start, start + np.diag([0.05*R, 0.05*R, 0.05*H])])})
    v = np.asarray(res.x, dtype=float)
    loss, t0 = _loss(v, P, T, w, cfg)
    inside = bool(np.hypot(v[0], v[1]) <= 1.05 * R and abs(v[2]) <= 1.05 * H)
    ok = bool(np.all(np.isfinite(v)) and res.success and inside)
    return VertexFit(v, t0, loss, int(s.size), ok)


def vertex_error(fit: VertexFit, true_vertex) -> float:
    """|v_hat - v_true| in metres (NaN for a failed fit)."""
    if not fit.ok:
        return float("nan")
    return float(np.linalg.norm(fit.vertex - np.asarray(true_vertex, dtype=float)))


def energy_calibration(charges_total: Sequence[float], true_energy: Sequence[float]) -> float:
    """Training-time energy scale ``c = mean(E) / mean(Q)`` (``E_hat = c * Q_total``), reference cell only."""
    q = np.asarray(charges_total, dtype=float)
    e = np.asarray(true_energy, dtype=float)
    if q.ndim != 1 or q.shape != e.shape or not q.size or not np.all(np.isfinite(q)) or not np.all(np.isfinite(e)) or np.any(e <= 0) or np.any(q <= 0):
        raise ValueError("reference charges must be positive.")
    return float(np.mean(e) / np.mean(q))


def energy_error(charge_total: float, true_energy: float, calibration: float) -> float:
    """Relative energy error ``|c Q - E| / E``."""
    if not np.isfinite(true_energy) or true_energy <= 0 or not np.isfinite(calibration) or calibration <= 0:
        raise ValueError("positive finite energy and calibration required")
    return float(abs(calibration * charge_total - true_energy) / true_energy)


def cell_consequence(cell_errors: Sequence[float], reference_errors: Sequence[float], *,
                     n_boot: int = 0, seed: int = 0) -> dict:
    """K for a cell: mean event error over the mean reference error on the same events.

    Failed fits make K unevaluable; a separately named complete-case diagnostic is retained.
    Returns ``K`` (ratio), the two means, the paired event count and the
    failure counts; with ``n_boot`` > 0 also a paired percentile bootstrap 95 %
    interval over events (``K_lo``, ``K_hi``). ``K`` is NaN when the reference
    mean is zero (undefined, as ``PREREGISTRATION.md`` §5 requires).
    """
    c = np.asarray(cell_errors, dtype=float)
    r = np.asarray(reference_errors, dtype=float)
    if c.ndim != 1 or not c.size or c.shape != r.shape:
        raise ValueError("cell and reference must cover the same events.")
    ok = np.isfinite(c) & np.isfinite(r)
    mc, mr = (float(c[ok].mean()), float(r[ok].mean())) if ok.any() else (float("nan"), float("nan"))
    K = mc / mr if ok.any() and mr > 0 else float("nan")
    conditional = K
    failed = not bool(np.all(ok))
    if failed:
        K = float("nan")
    out = {"K": K, "K_complete_case": conditional,
           "status": "unevaluable_fit_failure" if failed else ("ok" if np.isfinite(K) else "undefined_denominator"),
           "failure_rate_cell": float(np.mean(~np.isfinite(c))) if c.size else None,
           "failure_rate_reference": float(np.mean(~np.isfinite(r))) if r.size else None, "mean_error_cell": mc, "mean_error_reference": mr, "n_paired": int(ok.sum()),
           "n_failed_cell": int((~np.isfinite(c)).sum()), "n_failed_reference": int((~np.isfinite(r)).sum())}
    if n_boot > 0 and not failed and ok.sum() > 1 and mr > 0:
        cc, rr = c[ok], r[ok]
        idx = np.random.default_rng(seed).integers(0, cc.size, size=(n_boot, cc.size))
        num, den = cc[idx].mean(axis=1), rr[idx].mean(axis=1)
        ratios = num[den > 0] / den[den > 0]
        out["K_lo"], out["K_hi"] = (float(np.quantile(ratios, 0.025)), float(np.quantile(ratios, 0.975)))
    return out


#: Development κ_m for the LUCiD arm: the Tier-1 controlled-tier convention mirrored —
#: harm when the primary K = max(K_vertex, K_energy) is >= 1.10, i.e. either endpoint's
#: mean error exceeds the reference by 10 %. Because K is relative to the reference
#: *resolution*, K_energy ~ |scale bias| / resolution: at the toy's 2.8 % energy resolution
#: a ~0.3 % scale shift already crosses it, and better resolution (more photons) makes the
#: same bias "more harmful". NOT declared: convention versus a physics requirement is D-K.
KAPPA_M_DEV = 1.10
KAPPA_M_STATUS = "provisional_dev"


def aggregate_endpoints(*ratios):
    """Draft max, unevaluable if ANY required endpoint is undefined."""
    return float(max(ratios)) if ratios and np.all(np.isfinite(ratios)) else float("nan")
