"""s08: R4a (r5_gevp) finite-sample angles and their two-sample floor.

R4a's estimator in the E1a table is ``r5_gevp``: GEVP of the unregularised
subject covariance against an *independently drawn* calibration covariance
(``E1a_noise_cal``), with the population ``R_c`` as reference.  This script
repeats the registered sampling law for the 14 theta=0.1 cells at n=1e5 (and for
the two gamma=0 cells at n in {1e4,1e6}) with two independent subject draws, and
reports the GEVP angle and the two-sample GEVP floor.

Writes gevp_floor.json.
"""
from __future__ import annotations

import json
import time

import numpy as np

import e1f_common as C
from jepa_noise import linear
from jepa_noise.masks import namespace_rng
from jepa_noise.metrics import principal_angles
from jepa_noise.views import ExactModel

THETA = 0.1
GAMMAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
ETAS = [1.0, 3.0]
K = 3
NS = [10000, 100000, 1000000]


def calibration(model, eta, n, label, rng):
    """Registered E1a_noise_cal draw -> sample calibration covariance."""
    cal = (rng.normal(size=(n, 1)) * eta @ model.bc.T
           + rng.normal(size=(n, model.d)) @ model.chol.T)
    centred = cal - cal.mean(0)
    return centred.T @ centred / (n - 1)


def subject(model, gamma, n, label, replicate):
    role = "E1a_subject" if replicate == 0 else f"E1a_subject_floor{replicate}"
    rng = namespace_rng(0, role, cell=f"{label}/n{n}")
    if n <= 100000:
        model.rng = rng
        sample = model.sample(n, gamma)
        scc, stt, sct = linear.sample_covariances(sample["context"],
                                                  sample["target"],
                                                  ridge_relative=0.0)
        del sample
    else:
        scc, stt, sct = C.e1a_covariances_chunked(model, n, gamma, rng)
    return scc, stt, sct


def gevp_angle(scc, ncc_cal, reference):
    fit = linear.gevp(scc, ncc_cal, K)
    return float(np.max(principal_angles(fit.context, reference, degrees=True))), fit.context


def run_cell(model, eta, gamma, n):
    label = f"th{THETA:g}_eta{eta:g}_g{gamma:g}"
    population = model.population(gamma)
    reference = np.linalg.solve(population["ncc"], population["ac"]).T
    angles, contexts = [], []
    for r in range(2):
        scc, _, _ = subject(model, gamma, n, label, r)
        cal_role = "E1a_noise_cal" if r == 0 else "E1a_noise_cal_floor1"
        ncc_cal = calibration(model, eta, n, label,
                              namespace_rng(0, cal_role, cell=f"{label}/n{n}"))
        angle, context = gevp_angle(scc, ncc_cal, reference)
        angles.append(angle)
        contexts.append(context)
    floor = C.subspace_floor(contexts[0], contexts[1])
    return {"angles_deg": angles, "mean_angle_deg": float(np.mean(angles)),
            "floor_deg": floor}


def main():
    out = {"estimator": "r5_gevp (R4a): GEVP(scc_unregularized, ncc_cal), "
                        "max angle vs R_c", "theta_scale": THETA, "n": {}}
    models = {eta: ExactModel(d=40, k=K, seed=0, theta_scale=THETA,
                              eta_scale=eta, share="target") for eta in ETAS}
    t = time.time()
    for eta in ETAS:
        for gamma in GAMMAS:
            label = f"th{THETA:g}_eta{eta:g}_g{gamma:g}"
            record = {}
            for n in NS:
                if n != NS[1] and gamma != 0.0:
                    continue
                record[str(n)] = run_cell(models[eta], eta, gamma, n)
            out[label] = record
            print(f"[gevp] {label}: n=1e5 angle "
                  f"{record['100000']['mean_angle_deg']:.2f} floor "
                  f"{record['100000']['floor_deg']:.2f}", flush=True)
    out["seconds"] = time.time() - t
    C.dump(C.OUT / "gevp_floor.json", out)
    print(f"[gevp] done in {out['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
