# SPDX-License-Identifier: MIT
"""Population-level effect of channel-ordered arrival delays on linear JEPA (registered: false).

For each delay variant and split, compute at population level (no sampling, so
no 17-degree finite-sample floor):
  * D: max principal angle between the top-k population CCA context directions
    and the Paper 1 reference R_c (registered ``views.displacement``);
  * the signal / noise variance ratio of the context view;
  * per-parameter linearised vs true nonlinear signal variance (the t0 issue).
Run from the repo root:  PYTHONPATH=src python scripts/jepa_toy_delay_population.py --out <json>
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from jepa_noise.delay_views import DelayedTier1Views
from jepa_noise.views import Tier1Views, invsqrt
from jepa_noise.metrics import principal_angles
from scipy.linalg import eigh


def displacement(pop, k):
    """Same computation as jepa_noise.views.displacement, allowing k = 4."""
    hc, ht = invsqrt(pop["scc"]), invsqrt(pop["stt"])
    left, canonical, _ = np.linalg.svd(hc @ pop["sct"] @ ht, full_matrices=False)
    context = hc @ left[:, :k]
    w, vec = eigh(pop["ac"] @ pop["sigma_theta"] @ pop["ac"].T, pop["ncc"])
    ref = vec[:, np.argsort(w)[::-1][:k]]
    return float(np.max(principal_angles(context.T, ref.T, degrees=True))), canonical[:k + 1].tolist()

G = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.225, 0.45, 0.9]
VARIANTS = {
    "no_delay": None,
    "fixed_delay_0.01": dict(delay_mode="fixed", delay_step=0.01),
    "per_event_delay_0.005-0.015": dict(delay_mode="per_event", delay_range=(0.005, 0.015)),
}


def make(variant, split, g, signal_mode="linearized"):
    kw = dict(signal_mode=signal_mode, split=split, line_multiplier=0., common_strength=g, base_rho=0.)
    if VARIANTS[variant] is None:
        return Tier1Views(**kw)
    return DelayedTier1Views(**kw, **VARIANTS[variant])


def variance_check(view, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    noise = float(np.trace(view.noise_covariance))
    lo = [3., .30, .06] + ([view.delay_range[0]] if getattr(view, "n_physics", 3) == 4 else [])
    hi = [6., .40, .10] + ([view.delay_range[1]] if getattr(view, "n_physics", 3) == 4 else [])
    base = 0.5 * (np.array(lo) + np.array(hi))
    out = {}
    names = ["amplitude", "t0", "tau", "delay"][:len(lo)]
    for i, name in enumerate(names):
        lin = float((view.j0[:, i] ** 2).sum() * view.theta_sd[i] ** 2)
        th = np.tile(base, (n, 1)); th[:, i] = rng.uniform(lo[i], hi[i], n)
        w = np.array([view.waveform(t).ravel() for t in th])
        out[name] = {"linearised_over_noise": lin / noise, "true_over_noise": float(w.var(0).sum() / noise)}
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    result = {"registered": False, "extension": "toy_channel_delays_2026-10-24", "g": G, "variants": {}}
    for variant in VARIANTS:
        rec = {}
        for split in ("channel", "interleaved", "time"):
            rows = []
            for g in G:
                v = make(variant, split, g)
                pop = v.population()
                c = pop["context_indices"]
                sig = float(np.trace(pop["ac"] @ pop["sigma_theta"] @ pop["ac"].T))
                k = v.theta_sd.size
                D, cc = displacement(pop, k)
                rows.append({"g": g, "D_deg": D, "k": k, "canonical_correlations": cc,
                             "gamma_n": pop["gamma_n"],
                             "signal_over_noise_context": sig / float(np.trace(pop["ncc"]))})
            rec[split] = rows
            print(variant, split, [round(r["D_deg"], 1) for r in rows], flush=True)
        rec["variance_check"] = variance_check(make(variant, "channel", 0.02))
        result["variants"][variant] = rec
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)


if __name__ == "__main__":
    main()
