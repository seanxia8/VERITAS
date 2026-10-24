# SPDX-License-Identifier: MIT
"""E1a extension (registered: false): pure-noise and pure-physics cells.

Author request, 24 Oct 2026: "线性模型组，加两组实验，一组是纯噪音没物理，一组是纯物理没噪音，
来验证猜想".  This script adds labelled, unregistered cells beside the registered
E1a grid.  It does not modify any registered file, row or verdict; it reuses the
registered row runner ``jepa_noise.run.run_e1a_row`` (same solvers, probes,
namespaces, n = 1e5, seeds 0-4) and only relaxes ExactModel's ``theta_scale > 0``
check so that a zero-physics model can be built.

Cell families
-------------
* ``noise_only``: theta_scale = 0 (no physics in either view), eta_scale in {1, 3},
  gamma_load over the registered 7 levels.  The data are shared nuisance (in the
  target with strength gamma) plus private noise.
* ``physics_only``: eta_scale = 0 (no nuisance factor anywhere), theta_scale in
  {0.1, 0.4, 1}; private noise kept, because without any noise the physics
  reference R_c = span(N_cc^-1 A_c) is undefined and every covariance is rank 3.
  gamma_load in {0, 1} only (gamma multiplies a zero nuisance, so it must not matter).
* ``physics_only_lownoise``: as physics_only at theta 0.4, with the private-noise
  covariance scaled by 0.01 (near-noiseless limit that keeps R_c defined).

Predictions, written before the first run (24 Oct 2026)
-------------------------------------------------------
noise_only
  N1  CCA nuisance R^2 ~ 0 at gamma = 0 and -> ~1 once gamma > 0 (JEPA learns the
      shared factor even when there is nothing else to learn).
  N2  No method recovers R_c: CCA, GEVP and finite-beta angles stay near 90 deg at
      every gamma, because the data contain no physics.  (R_c is still defined by
      A_c and N_cc; it is simply not present in the data.)
  N3  GEVP is degenerate: with S_cc = N_cc every generalised eigenvalue is ~1, so
      its subspace is arbitrary (angle large, varies between seeds).
physics_only
  P1  CCA angle at the sampling floor (as E1a gamma = 0) for BOTH gamma = 0 and 1.
  P2  Context PCA angle drops sharply compared with the registered cells (it was
      74-90 deg with the nuisance present); what remains comes from the
      anisotropic private noise.
  P3  Whitened PCA, GEVP and beta >= 1 at the floor, as before.
physics_only_lownoise
  L1  Every method's physics R^2 -> ~1; CCA and whitened PCA at the floor.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import jepa_noise.views as views
from jepa_noise import run as runmod

_PRIVATE_SCALE = {"value": 1.0}


class ExtExactModel(views.ExactModel):
    """ExactModel that also allows theta_scale = 0 and a scaled private noise."""

    def __init__(self, d=40, k=3, seed=0, theta_scale=.4, eta_scale=3., share='target'):
        super().__init__(d=d, k=k, seed=seed, theta_scale=max(theta_scale, 1e-12),
                         eta_scale=eta_scale, share=share)
        self.theta_scale = float(theta_scale)
        scale = float(_PRIVATE_SCALE["value"])
        if scale != 1.0:
            self.private = self.private * scale
            self.chol = np.linalg.cholesky(self.private)


def cells():
    rows = []
    gammas = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for eta in (1.0, 3.0):
        for g in gammas:
            rows.append(("noise_only", {"theta_scale": 0.0, "eta_scale": eta,
                                        "gamma_load": g, "n": 100000}, 1.0))
    for th in (0.1, 0.4, 1.0):
        for g in (0.0, 1.0):
            rows.append(("physics_only", {"theta_scale": th, "eta_scale": 0.0,
                                          "gamma_load": g, "n": 100000}, 1.0))
    for g in (0.0, 1.0):
        rows.append(("physics_only_lownoise", {"theta_scale": 0.4, "eta_scale": 0.0,
                                               "gamma_load": g, "n": 100000}, 0.01))
    return rows


def summarise(record):
    out = {}
    for mid, m in record["methods"].items():
        if mid in ("r2_hard_isotropy", "r10_of"):
            continue
        angles = (m.get("angles") or {})
        probes = m.get("probes") or {}
        phys = [p.get("r2") for p in probes.get("physics", []) or []
                if isinstance(p, dict) and p.get("r2") is not None]
        out[mid] = {
            "max_angle_deg": angles.get("max_angle_deg"),
            "nuisance_r2": probes.get("nuisance_mean_r2"),
            "physics_r2_mean": float(np.mean(phys)) if phys else None,
            "status": m.get("status"),
        }
    cca = record["methods"].get("r1_cca", {})
    spec = cca.get("spectrum")
    out["cca_spectrum_top4"] = (spec[:4] if isinstance(spec, list) else
                                (spec.get("values", [])[:4] if isinstance(spec, dict) else None))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/jepa_noise/configs/e1_grid_v4.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    config, _ = runmod.load_config(Path(args.config))
    views.ExactModel = ExtExactModel          # run_e1a_row imports it at call time
    seeds = [int(s) for s in args.seeds.split(",")]
    results = []
    t0 = time.time()
    for family, cell, private_scale in cells():
        _PRIVATE_SCALE["value"] = private_scale
        for seed in seeds:
            label = runmod.e1a_cell_label(cell)
            row = {"campaign": "E1a", "row_id": f"ext_{family}_{label}_s{seed}",
                   "cell": cell, "seed": seed}
            record = runmod.run_e1a_row(config, row, lambda m: None)
            results.append({"family": family, "private_scale": private_scale,
                            "cell": cell, "seed": seed, "row_status": record["status"],
                            "row_reason": record["reason"], "summary": summarise(record)})
            print(f"{family:22s} {label:24s} s{seed} {time.time() - t0:7.1f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"registered": False,
                                          "extension": "e1a_pure_cells_2026-10-24",
                                          "rows": results}, indent=1, default=float))


if __name__ == "__main__":
    main()
