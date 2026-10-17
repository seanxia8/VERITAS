"""s03b: which reviewed-solver check trips on the two R3 rows.

``linear.finite_beta`` raises when the isotropy residual or the
|objective - reduced objective| gap exceeds TOL = 1e-8.  Both are recomputed here
for the base and the R3-transformed covariances of the two failing rows, in
float64, exactly as the solver computes them (same equations, no modification).
Also reports the float64 precision scale eps * cond for the transformed
covariance.  Writes r3_residuals.json.
"""
from __future__ import annotations

import json

import numpy as np

import e1f_common as C
from jepa_noise import linear

from s03_r3_conditioning import N, TOL, rebuild

ROWS = [(0.4, 1.0, 0.0, 3), (0.4, 3.0, 1.0, 2)]


def residuals(scc, stt, sct, ncc=None, k=3, beta=0.0):
    """Reproduce finite_beta's two residual checks, returning their values."""
    if ncc is None:
        ncc = scc
    hc, ht = linear.invsqrt(scc), linear.invsqrt(stt)
    t = hc @ np.asarray(sct, float) @ ht
    nw = hc @ linear.symmetric(ncc) @ hc
    a = t @ t.T - beta * nw
    scores, v, eigen_residual = linear._eig(a)
    uc = v[:, :k].T
    ut, rank = linear._complete_target(uc @ t, k)
    wc, wt = uc @ hc, ut @ ht
    iso = max(np.linalg.norm(wc @ scc @ wc.T - np.eye(k)),
              np.linalg.norm(wt @ stt @ wt.T - np.eye(k)))
    objective = k - np.linalg.norm(wt @ np.asarray(sct).T @ wc.T) ** 2 \
        + beta * np.trace(wc @ ncc @ wc.T)
    reduced = k - np.sum(scores[:k])
    gap = abs(objective - reduced)
    return {"isotropy_residual": float(iso),
            "objective_gap": float(gap),
            "relative_objective_gap": float(gap / max(1.0, abs(reduced))),
            "eigen_residual": float(eigen_residual),
            "trips_isotropy": bool(iso > TOL),
            "trips_objective_gap": bool(gap > TOL * max(1.0, abs(reduced))),
            "cond_scc": float(np.linalg.cond(scc)),
            "cond_stt": float(np.linalg.cond(stt)),
            "eps_times_cond_scc": float(np.finfo(float).eps * np.linalg.cond(scc)),
            "eps_times_cond_stt": float(np.finfo(float).eps * np.linalg.cond(stt))}


def main():
    out = {"tolerance": TOL, "rows": {}}
    for theta, eta, gamma, seed in ROWS:
        data = rebuild(theta, eta, gamma, seed)
        name = f"th{theta:g}_eta{eta:g}_g{gamma:g}_s{seed}"
        out["rows"][name] = {
            "base": residuals(data["scc"], data["stt"], data["sct"]),
            "transformed": residuals(data["scc_t"], data["stt_t"], data["sct_t"]),
            "cond_tc": float(np.linalg.cond(data["tc"])),
            "cond_tt": float(np.linalg.cond(data["tt"]))}
        print(name, json.dumps(out["rows"][name], indent=1), flush=True)
    C.dump(C.OUT / "r3_residuals.json", out)


if __name__ == "__main__":
    main()
