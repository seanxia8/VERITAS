"""Part-B statistics, written from PREREG_2026-11-22_R2_TIER1.md §2.2-§2.5 (r2-tier1v).

Per-event vertex errors of each cell on the sealed photon bombs (the same events for every
cell) -> e = mean error.  Per-seed contrast T = e2 / e1 - 1, T-bar = mean over seeds.
SE_boot: paired bootstrap over sealed events (2,000 resamples, one index draw shared by all
cells and seeds), SD of the resampled T-bar.  SE_seed: SD_pooled / sqrt(n_seeds), SD_pooled the
RMS over the detector's registered contrasts of the across-seed SD (ddof = 1) of T.
N_sigma = |T-bar| / max(SE_boot, SE_seed); clear = N_sigma >= 2 and the same sign in every seed.
"""
from __future__ import annotations

import numpy as np

B = 2000
BOOT_SEED = 20261122


def boot_index(n, b=B, seed=BOOT_SEED):
    return np.random.default_rng(seed).integers(0, n, size=(b, n))


def contrast(err1: dict, err2: dict, idx):
    """err1/err2: seed -> per-event error vectors (same events). Returns dict without SE_seed."""
    seeds = sorted(set(err1) & set(err2))
    T = {s: float(err2[s].mean() / err1[s].mean() - 1) for s in seeds}
    boot = np.mean([err2[s][idx].mean(1) / err1[s][idx].mean(1) - 1 for s in seeds], axis=0)
    return {"seeds": seeds, "T_per_seed": T, "T_bar": float(np.mean(list(T.values()))),
            "SE_boot": float(boot.std(ddof=1)), "e1": {s: float(err1[s].mean()) for s in seeds},
            "e2": {s: float(err2[s].mean()) for s in seeds}}


def finish(contrasts: dict, registered: list):
    """Add SE_seed, N_sigma, clear to every contrast; SD_pooled from the registered ones."""
    sds = [np.std(list(contrasts[k]["T_per_seed"].values()), ddof=1)
           for k in registered if len(contrasts[k]["T_per_seed"]) >= 2]
    sd_pooled = float(np.sqrt(np.mean(np.square(sds)))) if sds else float("nan")
    for k, c in contrasts.items():
        n = len(c["T_per_seed"])
        c["SD_pooled"] = sd_pooled
        c["SE_seed"] = sd_pooled / np.sqrt(n) if n >= 1 and np.isfinite(sd_pooled) else float("nan")
        se = np.nanmax([c["SE_boot"], c["SE_seed"]])
        c["N_sigma"] = float(abs(c["T_bar"]) / se) if se > 0 else float("inf")
        signs = {np.sign(v) for v in c["T_per_seed"].values()}
        c["same_sign_all_seeds"] = len(signs) == 1
        c["clear"] = bool(c["N_sigma"] >= 2 and c["same_sign_all_seeds"])
    return sd_pooled


def half_width_68(rel):
    """68 % half-width of (E_hat - E)/E: half the central 68 % interval (q84 - q16)/2."""
    q16, q84 = np.quantile(rel, [0.15865525, 0.84134475])
    return float((q84 - q16) / 2)


def resolution_contrast(rel1: dict, rel2: dict, idx):
    """Per-seed res2/res1 - 1 for energy resolution with a paired bootstrap over electrons."""
    seeds = sorted(set(rel1) & set(rel2))
    T = {s: half_width_68(rel2[s]) / half_width_68(rel1[s]) - 1 for s in seeds}
    q = [0.15865525, 0.84134475]

    def hw(r):
        a = np.quantile(r[idx], q, axis=1)
        return (a[1] - a[0]) / 2
    boot = np.mean([hw(rel2[s]) / hw(rel1[s]) - 1 for s in seeds], axis=0)
    return {"seeds": seeds, "T_per_seed": T, "T_bar": float(np.mean(list(T.values()))),
            "SE_boot": float(boot.std(ddof=1)),
            "res1": {s: half_width_68(rel1[s]) for s in seeds}, "res2": {s: half_width_68(rel2[s]) for s in seeds}}
