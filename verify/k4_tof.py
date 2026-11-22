"""K4: ToF-only vertex fit and timing-only CRLB with a deliberately simple model, plus the
optional charge-only CRLB (K4b).

Model: t_i = t0 + n_g |r_i - v| / c for every hit PMT (stored digit - trigger times are the
first-hit times, already inside the prompt window).  No walk, no scattering, no TTS shape.
n_g = 1.33: LUCiD materials/water.json refractive_index used by the analytic generator
(single index, no dispersion); 1.38 reported as a sensitivity.  Huber loss, scale 3 ns,
8 random starts uniform in the fiducial volume; best cost kept.

Timing CRLB: J = sum_i g_i g_i^T / sigma^2, g_i = d t_i / d(v, t0) at the true vertex over
the observed hit PMTs; sigma = 1.5 ns (reference.json timing_width at 1 PE).  Reported:
median sqrt(tr (J^-1)_pos).  Variant: per-hit sigma(q) from the reference.json table.

K4b charge-only CRLB (simplification stated): mu_i = A cos(theta_i)_+ / d_i^2 exp(-d_i / 50 m),
Poisson over all PMTs, parameters (v, ln A), A set so sum mu = observed total PE.

Usage: python k4_tof.py OUT.json [n_events]
"""
import json
import sys

import numpy as np
from scipy.optimize import least_squares

from common import DET, Cohort, Geometry

C_M_PER_NS = 0.299792
SIGMA_SPE_NS = 1.5
TW_Q, TW_S = np.array([0.25, 1.0, 5.0, 1000.0]), np.array([2.5, 1.5, 0.8, 0.5])
LAMBDA_M = 50.0
N_EV = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def inside(geo, v, frac=1.0):
    if geo.shape == "sphere":
        return np.linalg.norm(v) <= frac * geo.radius
    return np.hypot(v[0], v[1]) <= frac * geo.radius and abs(v[2]) <= frac * geo.half_h


def sample_start(geo, rng):
    while True:
        v = rng.uniform(-1, 1, 3) * geo.fid_frac * geo.scale
        if inside(geo, v, geo.fid_frac):
            return v


def fit_event(geo, r, t, ng, rng):
    k = ng / C_M_PER_NS

    def res(p):
        return t - p[3] - k * np.linalg.norm(r - p[:3], axis=1)

    def jac(p):
        d = r - p[:3]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return np.column_stack([k * d / n, -np.ones(len(t))])

    best = None
    for _ in range(8):
        v0 = sample_start(geo, rng)
        t00 = np.median(t - k * np.linalg.norm(r - v0, axis=1))
        sol = least_squares(res, np.r_[v0, t00], jac=jac, loss="huber", f_scale=3.0, max_nfev=200)
        if best is None or sol.cost < best.cost:
            best = sol
    return best


def timing_crlb(r, v, ng, sigma):
    d = r - v
    g = np.column_stack([-(ng / C_M_PER_NS) * d / np.linalg.norm(d, axis=1, keepdims=True), np.ones(len(r))])
    J = (g / sigma[:, None] ** 2).T @ g
    return float(np.sqrt(np.trace(np.linalg.inv(J)[:3, :3])))


def charge_crlb(geo, v, total_pe):
    d = geo.pos - v
    dist = np.linalg.norm(d, axis=1)
    u = -d / dist[:, None]                          # unit vector PMT -> vertex
    cos = np.einsum("ij,ij->i", geo.dirs, u)
    ok = cos > 1e-3
    shape = np.where(ok, cos / dist ** 2 * np.exp(-dist / LAMBDA_M), 0.0)
    A = total_pe / shape.sum()
    mu = A * shape
    # d ln mu / d v for visible PMTs: from cos (d cos/dv), 1/d^2 and attenuation
    dd = d[ok]; di = dist[ok]; ni = geo.dirs[ok]
    # cos = -n.d/|d| with d = r - v  =>  d cos / d v = n/|d| - (n.d) d / |d|^3
    dcos = ni / di[:, None] - np.einsum("ij,ij->i", ni, dd)[:, None] * dd / di[:, None] ** 3
    dlnmu = dcos / cos[ok][:, None] + (2.0 / di[:, None] + 1.0 / LAMBDA_M) * (dd / di[:, None])
    g = np.column_stack([mu[ok][:, None] * dlnmu, mu[ok]])   # d mu / d(v, ln A)
    J = (g / mu[ok][:, None]).T @ g
    return float(np.sqrt(np.trace(np.linalg.inv(J)[:3, :3])))


out = {"model": __doc__.split("Usage")[0].strip(), "n_events": N_EV, "detectors": {}}
for det in DET:
    geo, coh = Geometry(det), Cohort(det, "test")
    rng = np.random.default_rng(20261121)
    rows = {1.33: [], 1.38: []}
    crlb, crlb_q, crlb_c, nhit = [], [], [], []
    for k in range(N_EV):
        p, q, t = coh.hits(k)
        r, vt = geo.pos[p], coh.vertex[k]
        nhit.append(len(p))
        for ng in rows:
            sol = fit_event(geo, r, t, ng, rng)
            v = sol.x[:3]
            rows[ng].append((np.linalg.norm(v - vt), bool(sol.success and inside(geo, v))))
        crlb.append(timing_crlb(r, vt, 1.33, np.full(len(p), SIGMA_SPE_NS)))
        crlb_q.append(timing_crlb(r, vt, 1.33, np.interp(np.log(np.clip(q, .25, 1000)), np.log(TW_Q), TW_S)))
        crlb_c.append(charge_crlb(geo, vt, q.sum()))
    res = {"r_fid_m": geo.r_fid, "median_n_hit": float(np.median(nhit))}
    for ng, rr in rows.items():
        e = np.array([x[0] for x in rr]); ok = np.array([x[1] for x in rr])
        res[f"tof_ng{ng}"] = {"mean_err_m": float(e[ok].mean()), "median_err_m": float(np.median(e[ok])),
                              "mean_err_all_m": float(e.mean()), "convergence_rate": float(ok.mean())}
    res["crlb_timing_sigma1p5_median_m"] = float(np.median(crlb))
    res["crlb_timing_sigma_q_median_m"] = float(np.median(crlb_q))
    res["crlb_charge_only_median_m"] = float(np.median(crlb_c))
    out["detectors"][det] = res
    print(det, json.dumps(res), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
