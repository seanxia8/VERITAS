"""s07: redraw the finite-sample figure from finite_sample.json.

Two panels: (a) the population-angle-zero cells (E1b clean cell and the two
E1a theta=0.1, gamma=0 cells): CCA angle vs R_c and the two-independent-sample
floor, with 1/sqrt(n) guides through each curve's n=1e5 value; (b) the
deviation |sample angle - population angle| for saturated E1a cells with
1/sqrt(n) guides.  Reads finite_sample.json; writes figures/fig_sampling_floor.png.
"""
from __future__ import annotations

import json

import numpy as np

import e1f_common as C

NS = [10000, 100000, 1000000]


def main():
    fs = json.loads((C.OUT / "finite_sample.json").read_text(encoding="utf-8"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5), dpi=140)
    ns = np.array(NS, float)
    ax = axes[0]
    e1b = fs["E1b_clean_channel"]["n"]
    ax.plot(ns, [e1b[str(n)]["mean_angle_deg"] for n in NS], "o-",
            color="#1f4e79", label="E1b clean cell: CCA angle vs $R_c$")
    ax.plot(ns, [e1b[str(n)]["floor_deg"] for n in NS], "o--",
            markerfacecolor="none", color="#1f4e79",
            label="E1b clean cell: two-sample floor")
    ax.axhline(10.0, color="0.35", lw=0.9, ls=":")
    ax.text(1.05e4, 10.9, "R1 threshold 10$^\\circ$", fontsize=7, color="0.35")
    for label, color, marker in (("th0.1_eta1_g0", "#c0504d", "s"),
                                 ("th0.1_eta3_g0", "#9bbb59", "^")):
        rec = fs["E1a_theta0.1"][label]["n"]
        ax.plot(ns, [rec[str(n)]["mean_angle_deg"] for n in NS], marker + "-",
                color=color, label=f"E1a {label}: CCA angle")
        ax.plot(ns, [rec[str(n)]["floor_deg"] for n in NS], marker + "--",
                markerfacecolor="none", color=color,
                label=f"E1a {label}: floor")
    for values, color in (([e1b[str(n)]["mean_angle_deg"] for n in NS], "#1f4e79"),
                          ([fs["E1a_theta0.1"]["th0.1_eta1_g0"]["n"][str(n)]
                            ["mean_angle_deg"] for n in NS], "#c0504d"),
                          ([fs["E1a_theta0.1"]["th0.1_eta3_g0"]["n"][str(n)]
                            ["mean_angle_deg"] for n in NS], "#9bbb59")):
        ax.plot(ns, values[1] * np.sqrt(1e5 / ns), ":", color=color, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("events n")
    ax.set_ylabel("max principal angle (deg)")
    ax.set_title("(a) cells with zero population angle", fontsize=8)
    ax.legend(fontsize=6, loc="lower left", frameon=False)
    ax.grid(alpha=0.25, lw=0.4)

    ax = axes[1]
    for label, color in (("th0.1_eta1_g0.1", "#4f81bd"),
                         ("th0.1_eta1_g0.5", "#8064a2"),
                         ("th0.1_eta1_g1", "#4bacc6")):
        rec = fs["E1a_theta0.1"][label]
        dev = [float(np.mean(rec["n"][str(n)]["dev_from_population_deg"]))
               for n in NS]
        ax.plot(ns, dev, "o-", color=color,
                label=f"E1a {label} (population {rec['population_cca_deg']:.1f}$^\\circ$)")
        ax.plot(ns, dev[1] * np.sqrt(1e5 / ns), ":", color=color, lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("events n")
    ax.set_ylabel("|sample angle - population angle| (deg)")
    ax.set_title("(b) saturated E1a cells: distance to population", fontsize=8)
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    path = C.OUT / "figures" / "fig_sampling_floor.png"
    fig.savefig(path)
    print(f"[fig] {path} {path.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
