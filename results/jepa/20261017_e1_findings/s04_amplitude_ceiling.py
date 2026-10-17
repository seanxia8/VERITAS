"""s04: amplitude-probe information ceiling for the WP4a bridge cell.

R2_max(c) = Var(A_c) / (Var(A_c) + CRLB_c)

A_c is the probe target of the bridge cohort: the per-channel pulse amplitude of
the noiseless A4 waveform, ``max_t views.waveform(theta)[c]`` with the registered
nonlinear prior (amplitude U[3,6], t0 U[.30,.40], tau_t U[.06,.10]) and unit
channel gains.  CRLB_c is the Cramer-Rao variance of that amplitude under the
A4 bridge-cell N_actual (g=0.02, line 0, channel split), computed two ways:

  * joint:  g_c^T (J^T N^-1 J)^-1 g_c with g_c = dA_c/dtheta, i.e. all 2048
    samples (the probe sees every channel through attention);
  * local:  1 / (u_c^T N_cc,c^-1 u_c) with u_c the unit-peak channel-c shape,
    i.e. channel c's own 256 samples only (a conservative reference ceiling).

``jepa_noise.crlb.crlb`` is used for the joint Fisher (and cross-checked against
the accelerated triangular path).  Measured probe R2 comes from the per-run
``probes.json`` (tokens and post_time, 8 per-channel values).

Writes amplitude_ceiling.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

import e1f_common as C
from jepa_noise import crlb as crlb_module
from jepa_noise.train import load_reference_marginal, relative_frobenius_error
from jepa_noise.views import Tier1Views

ARMS = {"sigreg_jepa": "wp4a_grid_sigreg",
        "sigreg_jepa_anchored": "wp4a_grid_anchored",
        "mae_chi2": "wp4a_grid_mae_chi2",
        "sigreg_jepa_parallel": "wp4a_grid_parallel"}
CONSTRUCTOR = dict(
    signal_mode="nonlinear", split="channel", line_multiplier=0.0,
    common_strength=0.02, amplitude_scale=1.0, law="v4-A4-waveform-v1",
    base_rho=0.0, seed=0, cell="bridge", model_seed=0)
RANGES = np.array([3.0, 0.1, 0.04])
PRIOR = np.array([[3.0, 0.30, 0.06], [6.0, 0.40, 0.10]])
N_PRIOR = 40000
N_CRLB_PRIOR = 60


def amplitude_of(views, theta):
    """Per-channel pulse amplitude target (``max_t waveform``), shape (8,)."""
    return np.max(views.waveform(np.asarray(theta, float)), axis=1)


def gradient(views, theta, step=1e-4):
    """Central-difference dA_c/dtheta (8, 3); step is the registered 1e-4*range."""
    theta = np.asarray(theta, float)
    h = step * RANGES
    columns = []
    for i in range(3):
        e = np.zeros(3)
        e[i] = h[i]
        columns.append((amplitude_of(views, theta + e)
                        - amplitude_of(views, theta - e)) / (2 * h[i]))
    return np.stack(columns, axis=1)


def main():
    views = Tier1Views(**CONSTRUCTOR)
    noise = np.asarray(views.noise_covariance, dtype=float)
    target_marginal = noise[1024:, 1024:]
    reference = load_reference_marginal()
    reference_error = relative_frobenius_error(target_marginal, reference)
    theta0 = np.asarray(views.theta0, float)
    gains = np.asarray(views.gains, float)

    jacobian = views.jacobian(theta0)
    chol = np.linalg.cholesky(noise)
    whitened = solve_triangular(chol, jacobian, lower=True)
    information = whitened.T @ whitened
    fisher_cov = np.linalg.inv(information)
    slow = crlb_module.crlb(jacobian, noise)
    fisher_agreement = float(np.max(np.abs(fisher_cov - slow))
                             / np.max(np.abs(slow)))

    g = gradient(views, theta0)
    amplitude0 = amplitude_of(views, theta0)
    crlb_joint0 = np.einsum("ci,ij,cj->c", g, fisher_cov, g)

    # prior draws for Var(A_c), and a prior mean CRLB via the accelerated path
    rng = np.random.default_rng(20261017)
    draws = rng.uniform(PRIOR[0], PRIOR[1], size=(N_PRIOR, 3))
    amplitudes = np.array([amplitude_of(views, th) for th in draws])
    var_amplitude = amplitudes.var(axis=0, ddof=1)
    crlb_prior = []
    for th in draws[:N_CRLB_PRIOR]:
        j = views.jacobian(th)
        w = solve_triangular(chol, j, lower=True)
        info = w.T @ w
        cov = np.linalg.inv(info)
        gg = gradient(views, th)
        crlb_prior.append(np.einsum("ci,ij,cj->c", gg, cov, gg))
    crlb_prior_mean = np.mean(np.asarray(crlb_prior), axis=0)

    # channel-local bound: channel c's own 256 samples, unit-peak shape
    channel_crlb = []
    for c in range(8):
        block = noise[c * 256:(c + 1) * 256, c * 256:(c + 1) * 256]
        shape = views.waveform(theta0)[c] / amplitude0[c]
        channel_crlb.append(float(1.0 / (shape @ np.linalg.solve(block, shape))))
    channel_crlb = np.asarray(channel_crlb)

    ceiling_joint = var_amplitude / (var_amplitude + crlb_prior_mean)
    ceiling_joint0 = var_amplitude / (var_amplitude + crlb_joint0)
    ceiling_local = var_amplitude / (var_amplitude + channel_crlb)

    out = {
        "fixture": {"constructor": CONSTRUCTOR, "gains": gains.tolist(),
                    "theta0": theta0.tolist(),
                    "amplitude_prior": {"low": PRIOR[0].tolist(),
                                        "high": PRIOR[1].tolist()},
                    "target_definition": "max over 256 samples of the noiseless "
                                         "per-channel A4 waveform",
                    "reference_marginal_relative_frobenius_error": reference_error},
        "fisher": {
            "crlb_theta0": np.diag(fisher_cov).tolist(),
            "amplitude_parameter_crlb_theta0": float(fisher_cov[0, 0]),
            "amplitude_only_inverse_information": float(1.0 / information[0, 0]),
            "accelerated_vs_jepa_noise_crlb_max_rel_diff": fisher_agreement,
            "note": "(J^T N^-1 J)^-1 diagonal is the 3-parameter CRLB; 1/I_aa is "
                    "the known-other-parameters amplitude-only bound"},
        "per_channel": {
            "channels": list(range(8)),
            "var_amplitude": var_amplitude.tolist(),
            "crlb_joint_theta0": crlb_joint0.tolist(),
            "crlb_joint_prior_mean": crlb_prior_mean.tolist(),
            "crlb_channel_local": channel_crlb.tolist(),
            "r2_max_joint_prior": ceiling_joint.tolist(),
            "r2_max_joint_theta0": ceiling_joint0.tolist(),
            "r2_max_channel_local": ceiling_local.tolist(),
            "amplitude_at_theta0": amplitude0.tolist(),
            "prior_amplitude_mean": amplitudes.mean(axis=0).tolist(),
            "prior_amplitude_sd": amplitudes.std(axis=0, ddof=1).tolist()},
        "ceiling_summary": {
            "r2_max_joint_prior_mean_over_channels": float(ceiling_joint.mean()),
            "r2_max_joint_theta0_mean_over_channels": float(ceiling_joint0.mean()),
            "r2_max_channel_local_mean_over_channels": float(ceiling_local.mean()),
            "threshold_0.9_above_joint_ceiling": bool(0.9 > ceiling_joint.max()),
            "threshold_0.9_above_local_ceiling": bool(0.9 > ceiling_local.max())},
        "measured": {},
    }
    print(f"[ceiling] joint R2_max {ceiling_joint.mean():.4f}, "
          f"theta0 {ceiling_joint0.mean():.4f}, local "
          f"{ceiling_local.mean():.4f}, N-ref rel err {reference_error:.1e}",
          flush=True)

    for arm, folder in ARMS.items():
        runs = sorted((C.HERE.parent / folder / "runs").glob("*/probes.json"))
        out["measured"][arm] = {}
        for path in runs:
            payload = json.loads(path.read_text(encoding="utf-8"))
            seed = int(payload["run"]["seed"])
            entry = {"run_dir": path.parent.name}
            for block in ("tokens", "post_time"):
                amp = payload["probes"]["amplitude"][block]
                per_channel = [r["test_r2"] for r in amp["rows"]]
                entry[block] = {
                    "mean_r2": float(amp["mean_r2"]),
                    "per_channel_r2": per_channel,
                    "ratio_to_joint_ceiling": float(
                        amp["mean_r2"] / ceiling_joint.mean()),
                    "ratio_to_local_ceiling": float(
                        amp["mean_r2"] / ceiling_local.mean()),
                    "channel_ratios_to_joint_ceiling": [
                        float(r / c) for r, c in zip(per_channel, ceiling_joint)]}
            entry["random_init_floor"] = {
                block: payload["random_init_floor"]["amplitude"][block]["mean_r2"]
                for block in ("tokens", "post_time")}
            entry["random_init_floor"]["nuisance_r2"] = payload[
                "random_init_floor"]["nuisance"]["defined_mean_r2"]
            out["measured"][arm][str(seed)] = entry
        print(f"[measured] {arm}: "
              + ", ".join(f"s{s} tokens {out['measured'][arm][s]['tokens']['mean_r2']:.4f}"
                          for s in out["measured"][arm]), flush=True)

    # B1 supporting numbers, read from the combined aggregate (no recomputation)
    combined = json.loads((C.HERE.parent / "wp4a_grid_combined" / "aggregate.json")
                          .read_text(encoding="utf-8"))
    b1 = combined["verdicts"]["B1"]
    floors = [json.loads(p.read_text(encoding="utf-8"))["random_init_floor"]
              ["nuisance"]["defined_mean_r2"]
              for arm in ARMS
              for p in (C.HERE.parent / ARMS[arm] / "runs").glob("*/probes.json")]
    out["B1_support"] = {
        "plain_mean_nuisance_r2": b1["plain_mean"],
        "anchored_mean_nuisance_r2": b1["anchored_mean"],
        "paired_differences": b1["paired_differences"],
        "seed_floor": b1["seed_floor"],
        "all_seeds_plain_greater": b1["all_seeds_plain_greater"],
        "random_init_floor_nuisance_r2": {
            "values_across_runs": sorted(set(round(f, 12) for f in floors)),
            "reported_in_findings": -0.004},
        "floor_margin_plain": b1["plain_mean"] - min(floors),
        "combined_verdict_status": b1["status"]}
    C.dump(C.OUT / "amplitude_ceiling.json", out)
    print(json.dumps(out["ceiling_summary"], indent=1))


if __name__ == "__main__":
    main()
