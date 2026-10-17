"""s02: R2 seed-mean angle-vs-gamma curves and the saturation diagnostic.

The registered R2 statistic (Spearman over the seven gamma points, per cell) is
recomputed from the stored seed means and *not* replaced.  The diagnostic
recomputes Spearman on the unsaturated subset only, to show where the
registered failures sit; it is diagnostic only.

Writes r2_saturation.json and figures/fig_r2_saturation.png.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import spearmanr

import e1f_common as C

AGG = C.HERE.parent / "20261017_wp3_e1a" / "aggregate_E1a.json"


def main():
    aggregate = json.loads(AGG.read_text(encoding="utf-8"))
    clauses = aggregate["predictions"]["R2"]["clauses"]["e1a"]["cells"]
    out = {"registered_R2": {"statistic": "Spearman(gamma, seed-mean max angle)",
                             "cells": {}},
           "diagnostic_only": {
               "note": "recomputed Spearman after dropping saturated gamma "
                       "points; never a replacement for the registered value",
               "saturation_angle_deg": 80.0,
               "cells": {}},
           "R4b_exact_linear": {}}
    for label, cell in clauses.items():
        gamma = np.asarray(cell["gamma_loads"], float)
        angles = np.asarray(cell["seed_mean_angles_deg"], float)
        registered = float(cell["spearman"])
        recomputed = float(spearmanr(gamma, angles).statistic)
        out["registered_R2"]["cells"][label] = {
            "gamma_loads": gamma.tolist(),
            "seed_mean_angles_deg": angles.tolist(),
            "spearman_registered": registered,
            "spearman_recomputed": recomputed,
            "spearman_pass": bool(cell["spearman_pass"]),
            "saturated_points": int(np.sum(angles >= 80.0)),
            "angle_range_deg": [float(angles.min()), float(angles.max())]}
        keep = angles < 80.0
        if keep.sum() >= 3:
            diag = float(spearmanr(gamma[keep], angles[keep]).statistic)
            diag_all_kept = keep.tolist()
        else:
            diag, diag_all_kept = None, keep.tolist()
        out["diagnostic_only"]["cells"][label] = {
            "kept_gamma_loads": gamma[keep].tolist(),
            "kept_angles_deg": angles[keep].tolist(),
            "spearman_unsaturated": diag,
            "n_points": int(keep.sum()),
            "dropped_gamma_loads": gamma[~keep].tolist()}
    r4b = aggregate["predictions"]["R4b"]["cells"]
    pass10 = sum(1 for v in r4b.values() if v["pass_beta_10"])
    out["R4b_exact_linear"] = {"cells": len(r4b), "pass_beta_10": int(pass10),
                               "fail_beta_10": int(len(r4b) - pass10)}
    C.dump(C.OUT / "r2_saturation.json", out)
    print(json.dumps(out["diagnostic_only"], indent=1))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=140)
    ax.axhspan(80.0, 92.0, color="0.88", lw=0, zorder=0)
    ax.text(0.02, 88.6, "saturated plateau (>= 80 deg)", fontsize=7, color="0.35")
    styles = {(0.1, 1.0): ("#c0504d", "-"), (0.1, 3.0): ("#c0504d", "--"),
              (0.4, 1.0): ("#4f81bd", "-"), (0.4, 3.0): ("#4f81bd", "--"),
              (1.0, 1.0): ("#9bbb59", "-"), (1.0, 3.0): ("#9bbb59", "--")}
    for label, cell in clauses.items():
        theta = float(label.split("_")[0].replace("theta", ""))
        eta = float(label.split("_")[1].replace("eta", ""))
        color, ls = styles[(theta, eta)]
        gamma = np.asarray(cell["gamma_loads"], float)
        angles = np.asarray(cell["seed_mean_angles_deg"], float)
        ax.plot(gamma, angles, ls, color=color, marker="o", ms=2.8, lw=1.1,
                label=f"theta={theta:g}, eta={eta:g} (rho={cell['spearman']:.3f}"
                      f"{'' if cell['spearman_pass'] else ', fail'})")
        saturated = angles >= 80.0
        ax.plot(gamma[saturated], angles[saturated], "o", ms=4.2,
                markerfacecolor="none", markeredgecolor=color)
    ax.set_xlabel("gamma_load")
    ax.set_ylabel("seed-mean max principal angle (deg)")
    ax.set_title("E1a angle-vs-gamma: registered seed means (filled = Spearman "
                 "input, open = saturated)", fontsize=8)
    ax.legend(fontsize=6, frameon=False, loc="lower right")
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    path = C.OUT / "figures" / "fig_r2_saturation.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path)
    print(f"[fig] {path} {path.stat().st_size/1024:.0f} KB", flush=True)


if __name__ == "__main__":
    main()
