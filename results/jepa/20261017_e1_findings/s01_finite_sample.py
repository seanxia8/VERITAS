"""s01: finite-sample floor and 1/sqrt(n) scaling of the registered CCA subspace.

Cells:
  * E1b clean cell (A4 law, base_rho=0, g=0, line=0, linearised fixture,
    ``channel`` split, n in {1e4,1e5,1e6});
  * the 14 E1a cells with theta_scale = 0.1 (eta in {1,3} x 7 gamma_load),
    same n.

For every (cell, n) two independent samples are drawn from the same law; the
"floor" is the maximum principal angle between their top-3 context canonical
subspaces.  The reported angle is the registered r1_cca angle against the
population reference R_c = span(N_cc^-1 A_c).

Writes finite_sample.json and figures/fig_sampling_floor.png.
"""
from __future__ import annotations

import json
import time

import numpy as np

import e1f_common as C
from jepa_noise.masks import namespace_rng, raw_split
from jepa_noise.views import ExactModel, Tier1Views

NS = [10000, 100000, 1000000]
THETA = 0.1
GAMMAS = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
ETAS = [1.0, 3.0]
K = 3


def e1a_reference(model, gamma):
    population = model.population(gamma)
    reference = np.linalg.solve(population["ncc"], population["ac"]).T
    return population, reference


def e1a_one(model, gamma, n, label, replicate):
    """One independent sample of the E1a law; returns (angle_deg, context rows)."""
    role = "E1a_subject" if replicate == 0 else f"E1a_subject_floor{replicate}"
    rng = namespace_rng(0, role, cell=f"{label}/n{n}")
    if n <= 100000:
        model.rng = rng
        sample = model.sample(n, gamma)
        context, target = sample["context"], sample["target"]
        del sample
        scc, stt, sct = C.linear.sample_covariances(context, target,
                                                    ridge_relative=0.0)
        del context, target
    else:
        scc, stt, sct = C.e1a_covariances_chunked(model, n, gamma, rng)
    population, reference = e1a_reference(model, gamma)
    angle, fit = C.cca_angle(scc, stt, sct, reference)
    return angle, fit.context, population, reference, scc


def self_test():
    """The chunked Gram must equal the direct product exactly (float64)."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(997, 17))
    error = float(np.max(np.abs(C.gram(x) - x.T @ x)))
    assert error < 1e-12, f"chunked Gram mismatch {error}"
    return {"gram_abs_error": error}


def run_e1a(out):
    models = {eta: ExactModel(d=40, k=K, seed=0, theta_scale=THETA,
                              eta_scale=eta, share="target") for eta in ETAS}
    cells = {}
    for eta in ETAS:
        model = models[eta]
        for gamma in GAMMAS:
            label = f"th{THETA:g}_eta{eta:g}_g{gamma:g}"
            pop, reference = e1a_reference(model, gamma)
            pop_angle, _ = C.population_cca_angle(pop, reference)
            record = {"population_cca_deg": pop_angle,
                      "population_canonical_correlations": [
                          float(x) for x in np.linalg.svd(
                              C.linear.invsqrt(pop["scc"]) @ pop["sct"]
                              @ C.linear.invsqrt(pop["stt"]),
                              compute_uv=False)[:K]],
                      "n": {}}
            for n in NS:
                t = time.time()
                angle0, ctx0, pop, ref, scc0 = e1a_one(model, gamma, n, label, 0)
                angle1, ctx1, *_ = e1a_one(model, gamma, n, label, 1)
                record["n"][str(n)] = {
                    "angles_deg": [angle0, angle1],
                    "mean_angle_deg": float(np.mean([angle0, angle1])),
                    "dev_from_population_deg": [
                        abs(angle0 - pop_angle), abs(angle1 - pop_angle)],
                    "floor_deg": C.subspace_floor(ctx0, ctx1),
                    "seconds": time.time() - t,
                }
            cells[label] = record
            print(f"[E1a] {label}: pop {pop_angle:.3f} deg, "
                  f"n=1e5 floor {cells[label]['n']['100000']['floor_deg']:.3f}",
                  flush=True)
    out["E1a_theta0.1"] = cells


def e1b_cell(out):
    view = Tier1Views(signal_mode="linearized", split="channel",
                      line_multiplier=0., common_strength=0., seed=0,
                      cell="E1b/diagnostic_floor")
    population = C.e1b_population(view)
    reference = np.linalg.solve(population["ncc"], population["ac"]).T
    pop_angle, pop_fit = C.population_cca_angle(population, reference)
    spectrum = np.linalg.svd(C.linear.invsqrt(population["scc"])
                            @ population["sct"]
                            @ C.linear.invsqrt(population["stt"]),
                            compute_uv=False)
    record = {"split": "channel", "context_dim": int(len(view.split.context)),
              "target_dim": int(len(view.split.target)),
              "population_cca_deg": pop_angle,
              "population_cca_rad": float(np.deg2rad(pop_angle)),
              "population_canonical_correlations_top3": [
                  float(x) for x in spectrum[:K]],
              "population_canonical_kth_gap": float(spectrum[K - 1]),
              "n": {}}
    del pop_fit
    for n in NS:
        t = time.time()
        reps = []
        for r in range(2):
            rng = np.random.default_rng([20261017, 7, n, r])
            scc, stt, sct = C.e1b_covariances_chunked(view, n, rng)
            angle, fit = C.cca_angle(scc, stt, sct, reference)
            reps.append({"angle_deg": angle, "context": fit.context})
            del scc, stt, sct, fit
        record["n"][str(n)] = {
            "angles_deg": [reps[0]["angle_deg"], reps[1]["angle_deg"]],
            "mean_angle_deg": float(np.mean([r["angle_deg"] for r in reps])),
            "dev_from_population_deg": [abs(r["angle_deg"] - pop_angle)
                                        for r in reps],
            "floor_deg": C.subspace_floor(reps[0]["context"], reps[1]["context"]),
            "seconds": time.time() - t}
        print(f"[E1b] n={n}: angles "
              f"{record['n'][str(n)]['angles_deg']} floor "
              f"{record['n'][str(n)]['floor_deg']:.3f} "
              f"({record['n'][str(n)]['seconds']:.0f}s)", flush=True)
    out["E1b_clean_channel"] = record


def slope(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    ok = np.isfinite(ys) & (ys > 0)
    if ok.sum() < 2:
        return None
    return float(np.polyfit(np.log(xs[ok]), np.log(ys[ok]), 1)[0])


def scaling_summary(out):
    e1b = out["E1b_clean_channel"]["n"]
    ns = [float(n) for n in NS]
    summary = {
        "E1b_clean_channel": {
            "angle_slope_loglog": slope(ns, [e1b[str(n)]["mean_angle_deg"] for n in NS]),
            "floor_slope_loglog": slope(ns, [e1b[str(n)]["floor_deg"] for n in NS]),
            "floor_over_angle": [e1b[str(n)]["floor_deg"]
                                 / e1b[str(n)]["mean_angle_deg"] for n in NS]},
        "E1a_gamma0": {},
        "E1a_gamma_gt0_deviation": {},
    }
    for label, record in out["E1a_theta0.1"].items():
        ns_c = [float(n) for n in record["n"]]
        if record["population_cca_deg"] < 1e-6:
            summary["E1a_gamma0"][label] = {
                "angle_slope_loglog": slope(
                    ns_c, [record["n"][str(n)]["mean_angle_deg"] for n in NS]),
                "floor_slope_loglog": slope(
                    ns_c, [record["n"][str(n)]["floor_deg"] for n in NS]),
                "floor_over_angle": [record["n"][str(n)]["floor_deg"]
                                     / record["n"][str(n)]["mean_angle_deg"]
                                     for n in NS]}
        else:
            dev = [float(np.mean(record["n"][str(n)]["dev_from_population_deg"]))
                   for n in NS]
            summary["E1a_gamma_gt0_deviation"][label] = {
                "mean_dev_from_population_deg": dev,
                "dev_slope_loglog": slope(ns_c, dev)}
    out["scaling"] = summary


def r4a_population(out):
    """R4a population recovery error (GEVP path) plus a whitened-PCA contrast.

    ``report.compute_r4a`` reads ``population.gevp_max_angle_deg``
    (``r5_gevp`` is R4a's estimator), so the R4a population number is the GEVP
    angle; the whitened-PCA angle is recorded only as a contrast.
    """
    from jepa_noise.metrics import principal_angles
    gevp, pca = {}, {}
    for seed in (4, 0):
        for eta in (1.0, 3.0):
            for theta in (0.1, 0.4, 1.0):
                model = ExactModel(d=40, k=K, seed=seed, theta_scale=theta,
                                   eta_scale=eta, share="target")
                for gamma in GAMMAS:
                    pop = model.population(gamma)
                    ref = np.linalg.solve(pop["ncc"], pop["ac"]).T
                    key = f"th{theta:g}_eta{eta:g}_g{gamma:g}_s{seed}"
                    gevp[key] = float(np.max(principal_angles(
                        C.linear.gevp(pop["scc"], pop["ncc"], K).context, ref)))
                    if seed == 4:
                        pca[key] = float(np.max(principal_angles(
                            C.linear.pca(pop["scc"], K, pop["ncc"]).context, ref)))
    out["R4a_population_recovery"] = {
        "cells": 42, "estimator": "r5_gevp (R4a's registered estimator)",
        "gevp_rad": gevp, "whitened_pca_contrast_rad": pca,
        "max_gevp_rad": max(gevp.values()),
        "max_whitened_pca_rad": max(pca.values()),
        "aggregate_reported_rad": 3.024592555866384e-14}
    print(f"[R4a] GEVP population recovery max {max(gevp.values()):.3e} rad",
          flush=True)


def check_against_registered(out):
    """Replicate 0 at n in {1e4,1e5} must equal the stored E1a seed-0 row angle."""
    path = C.HERE.parent / "20261017_wp3_e1a" / "aggregate_E1a.json"
    table = json.loads(path.read_text(encoding="utf-8"))["row_table"]
    lookup = {}
    for row in table:
        cell = row["cell"]
        key = (cell["theta_scale"], cell["eta_scale"], cell["gamma_load"],
               cell["n"], row["seed"])
        lookup[key] = row["cca_max_angle_deg"]
    diffs = {}
    for label, record in out["E1a_theta0.1"].items():
        eta = 1.0 if "_eta1_" in label else 3.0
        gamma = float(label.split("_g")[-1])
        for n in (10000, 100000):
            stored = lookup.get((THETA, eta, gamma, n, 0))
            observed = record["n"][str(n)]["angles_deg"][0]
            diffs[f"{label}_n{n}"] = {
                "observed_deg": observed, "registered_deg": stored,
                "abs_diff_deg": None if stored is None else abs(observed - stored)}
    worst = max(v["abs_diff_deg"] for v in diffs.values()
                if v["abs_diff_deg"] is not None)
    out["checks"] = {"registered_row_replay": diffs, "max_abs_diff_deg": worst,
                     "status": "pass" if worst < 1e-9 else "fail"}
    print(f"[check] registered E1a seed-0 row replay max |diff| {worst:.2e} deg",
          flush=True)


def figure(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), dpi=140)
    ax = axes[0]
    ns = np.array(NS, float)
    e1b = out["E1b_clean_channel"]["n"]
    ax.plot(ns, [e1b[str(n)]["mean_angle_deg"] for n in NS], "o-",
            color="#1f4e79", label="E1b clean cell, CCA angle")
    ax.plot(ns, [e1b[str(n)]["floor_deg"] for n in NS], "o--", markerfacecolor="none",
            color="#1f4e79", label="E1b clean cell, two-sample floor")
    ax.axhline(10.0, color="0.35", lw=0.9, ls=":")
    ax.text(1.1e4, 10.8, "R1 threshold 10$^\\circ$", fontsize=7, color="0.35")
    colors = {("th0.1_eta1_g0"): "#c0504d", ("th0.1_eta3_g0"): "#9bbb59"}
    for label, color in colors.items():
        rec = out["E1a_theta0.1"][label]["n"]
        ax.plot(ns, [rec[str(n)]["mean_angle_deg"] for n in NS], "s-", color=color,
                label=f"E1a {label}, CCA angle")
        ax.plot(ns, [rec[str(n)]["floor_deg"] for n in NS], "s--",
                markerfacecolor="none", color=color,
                label=f"E1a {label}, floor")
    ref = out["E1b_clean_channel"]["n"]["100000"]["floor_deg"]
    guide = ref * np.sqrt(1e5 / ns)
    ax.plot(ns, guide, "k-.", lw=0.9, label=r"$n^{-1/2}$ through 1e5 floor")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("events n")
    ax.set_ylabel("max principal angle (deg)")
    ax.set_title("(a) population-angle-zero cells", fontsize=9)
    ax.legend(fontsize=6, loc="lower left", frameon=False)
    ax.grid(alpha=0.25, lw=0.4)

    ax = axes[1]
    for label, color in (("th0.1_eta1_g0.1", "#4f81bd"),
                         ("th0.1_eta1_g0.5", "#8064a2"),
                         ("th0.1_eta1_g1", "#4bacc6")):
        rec = out["E1a_theta0.1"][label]
        dev = [float(np.mean(rec["n"][str(n)]["dev_from_population_deg"]))
               for n in NS]
        ax.plot(ns, dev, "o-", color=color,
                label=f"E1a {label} (pop {rec['population_cca_deg']:.1f}$^\\circ$)")
        ax.plot(ns, rec["n"]["100000"]["floor_deg"] * np.sqrt(1e5 / ns), ":",
                color=color, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("events n")
    ax.set_ylabel("|sample angle - population angle| (deg)")
    ax.set_title("(b) saturated (gamma>0) E1a cells, deviation", fontsize=9)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    path = C.OUT / "figures" / "fig_sampling_floor.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)
    print(f"[fig] {path} {path.stat().st_size/1024:.0f} KB", flush=True)


if __name__ == "__main__":
    out = {"self_test": self_test(),
           "cells": {"E1a_theta_scale": THETA, "n": NS},
           "estimator": "registered r1_cca: CCA of ridge(1e-6) marginals, "
                        "unridged cross term, max principal angle vs R_c"}
    run_e1a(out)
    e1b_cell(out)
    r4a_population(out)
    check_against_registered(out)
    scaling_summary(out)
    C.dump(C.OUT / "finite_sample.json", out)
    figure(out)
    print(json.dumps(out["scaling"], indent=1))
