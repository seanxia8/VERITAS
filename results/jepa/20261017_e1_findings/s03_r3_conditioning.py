"""s03: R3 (per-view invertible-transform invariance) evidence.

* max observed spectrum difference over the 208 computable n=1e5 rows, read from
  the stored row files (not from the aggregate summary);
* the two rows that fail the reviewed solver's fixed tolerance, with the
  condition numbers of the *transformed* sample covariances that are fed to
  ``linear.cca`` (the failure is the solver's isotropy/objective residual check,
  which is a float64 conditioning symptom, not a different transform);
* the same condition numbers over all 210 n=1e5 rows for context, so the two
  failures can be placed in the distribution.

Rebuild is bit-exact: the registered namespaces are reused, and s00 confirmed
the sample-covariance digest.  Writes r3_conditioning.json.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

import e1f_common as C
from jepa_noise import linear
from jepa_noise.masks import namespace_rng
from jepa_noise.views import ExactModel

ROWS = (C.HERE.parent / "20261017_wp3_e1a" / "attempt_007_20261017T122356Z_E1a"
        / "rows")
THETA = [0.1, 0.4, 1.0]
ETA = [1.0, 3.0]
GAMMA = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
SEEDS = [0, 1, 2, 3, 4]
N = 100000
TOL = 1e-8


def fmt(value: float) -> str:
    return f"{value:g}"


def label(theta, eta, gamma):
    return f"th{fmt(theta)}_eta{fmt(eta)}_g{fmt(gamma)}"


def load_r3():
    clean, failed = [], []
    for theta in THETA:
        for eta in ETA:
            for gamma in GAMMA:
                for seed in SEEDS:
                    path = ROWS / f"{label(theta, eta, gamma)}_n{N}_s{seed}.json"
                    row = json.loads(path.read_text(encoding="utf-8"))
                    r3 = row["r3_per_view_invariance"]
                    record = {"cell": label(theta, eta, gamma), "seed": seed,
                              "status": r3["status"],
                              "max_spectrum_difference": r3["max_spectrum_difference"],
                              "reason": r3.get("reason")}
                    (clean if r3["status"] == "evaluated" else failed).append(record)
    return clean, failed


def rebuild(theta, eta, gamma, seed):
    model = ExactModel(d=40, k=3, seed=seed, theta_scale=theta,
                       eta_scale=eta, share="target")
    model.rng = namespace_rng(seed, "E1a_subject",
                              cell=f"{label(theta, eta, gamma)}/n{N}")
    sample = model.sample(N, gamma)
    scc, stt, sct = linear.sample_covariances(sample["context"], sample["target"],
                                              ridge_relative=0.0)
    population = sample["population"]
    reference = np.linalg.solve(population["ncc"], population["ac"]).T
    rng = namespace_rng(seed, "E1a_R3_transform",
                        cell=label(theta, eta, gamma))
    tc = 1.0 * rng.normal(size=(40, 40)) + 5.0 * np.eye(40)
    tt = 1.0 * rng.normal(size=(40, 40)) + 5.0 * np.eye(40)
    transformed = (tc @ scc @ tc.T, tt @ stt @ tt.T, tc @ sct @ tt.T)
    return {"scc": scc, "stt": stt, "sct": sct, "tc": tc, "tt": tt,
            "scc_t": transformed[0], "stt_t": transformed[1],
            "sct_t": transformed[2], "reference": reference,
            "scc_sha256": hashlib.sha256(
                np.ascontiguousarray(scc, dtype="<f8").tobytes()).hexdigest()}


def solver_probe(scc, stt, sct):
    """Return success + the two residual checks the reviewed solver applies."""
    out = {"status": "evaluated"}
    try:
        fit = linear.cca(scc, stt, sct, 3)
        out["max_angles_deg"] = fit.diagnostics.get("objective")
        out["isotropy_residual"] = fit.diagnostics.get("isotropy_residual")
        out["objective"] = fit.diagnostics.get("objective")
        out["reduced_objective"] = fit.diagnostics.get("reduced_objective")
        out["objective_gap"] = abs(out["objective"] - out["reduced_objective"])
        out["eigen_residual"] = fit.diagnostics.get("eigen_residual")
    except Exception as error:  # noqa: BLE001 - reproduce the recorded failure
        out.update(status="fail", error=f"{type(error).__name__}: {error}")
    return out


def main():
    clean, failed = load_r3()
    diffs = np.array([r["max_spectrum_difference"] for r in clean])
    out = {
        "registered": {
            "rows_with_r3": len(clean) + len(failed),
            "evaluated_rows": len(clean),
            "failed_rows": len(failed),
            "tolerance": TOL,
            "max_observed_spectrum_difference_208": float(diffs.max()),
            "max_observed_row": clean[int(np.argmax(diffs))],
            "p99_observed_spectrum_difference": float(np.quantile(diffs, .99)),
            "median_observed_spectrum_difference": float(np.median(diffs)),
            "all_208_within_1e-8": bool(np.all(diffs <= TOL)),
            "failed_row_ids": [f"{r['cell']}_s{r['seed']}" for r in failed],
            "failed_row_reasons": sorted({r["reason"] for r in failed}),
        },
        "failed_rows": {},
        "all_rows_conditioning": {},
    }
    print(f"[R3] 208-row max spectrum difference {diffs.max():.3e}", flush=True)

    conditions = []
    rebuild_failures = []
    for theta in THETA:
        for eta in ETA:
            for gamma in GAMMA:
                for seed in SEEDS:
                    name = f"{label(theta, eta, gamma)}_s{seed}"
                    data = rebuild(theta, eta, gamma, seed)
                    cond = {
                        "scc": float(np.linalg.cond(data["scc"])),
                        "stt": float(np.linalg.cond(data["stt"])),
                        "tc": float(np.linalg.cond(data["tc"])),
                        "tt": float(np.linalg.cond(data["tt"])),
                        "scc_transformed": float(np.linalg.cond(data["scc_t"])),
                        "stt_transformed": float(np.linalg.cond(data["stt_t"])),
                    }
                    conditions.append({"row": name, **cond})
                    if any(r["cell"] == label(theta, eta, gamma)
                           and r["seed"] == seed for r in failed):
                        base = solver_probe(data["scc"], data["stt"], data["sct"])
                        trans = solver_probe(data["scc_t"], data["stt_t"],
                                             data["sct_t"])
                        rebuild_failures.append({"row": name, **cond,
                                                 "base_solver": base,
                                                 "transformed_solver": trans,
                                                 "base_spectrum": None})
                        print(f"[R3] {name}: base {base['status']}, "
                              f"transformed {trans['status']}", flush=True)
    out["failed_rows"] = {r["row"]: r for r in rebuild_failures}
    arr = {k: np.array([c[k] for c in conditions]) for k in
           ("scc", "stt", "tc", "tt", "scc_transformed", "stt_transformed")}
    out["all_rows_conditioning"] = {
        "rows": len(conditions),
        "scc_transformed": {"median": float(np.median(arr["scc_transformed"])),
                            "p95": float(np.quantile(arr["scc_transformed"], .95)),
                            "max": float(arr["scc_transformed"].max())},
        "stt_transformed": {"median": float(np.median(arr["stt_transformed"])),
                            "p95": float(np.quantile(arr["stt_transformed"], .95)),
                            "max": float(arr["stt_transformed"].max())},
        "tc": {"median": float(np.median(arr["tc"])),
               "max": float(arr["tc"].max())},
        "tt": {"median": float(np.median(arr["tt"])),
               "max": float(arr["tt"].max())},
        "rows_ranked_by_scc_transformed_cond": [
            {"row": c["row"], "cond": c["scc_transformed"]}
            for c in sorted(conditions, key=lambda x: -x["scc_transformed"])[:6]],
        "rows_ranked_by_stt_transformed_cond": [
            {"row": c["row"], "cond": c["stt_transformed"]}
            for c in sorted(conditions, key=lambda x: -x["stt_transformed"])[:6]],
    }
    C.dump(C.OUT / "r3_conditioning.json", out)
    print(json.dumps(out["registered"], indent=1))
    print(json.dumps(out["all_rows_conditioning"], indent=1))


if __name__ == "__main__":
    main()
