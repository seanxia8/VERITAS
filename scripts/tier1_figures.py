#!/usr/bin/env python3
"""r2-tier1 T8: figures for GATE_G1 (sealed_test, step 10,000, frozen spatial-moment Ridge unless stated).

unbundling.png      mask x unit, one point per seed (SK and WCTE)
dose_response.png   vertex error against time unit, patch and random, u50 marked (SK and WCTE)
energy_vs_unit.png  electron / muon energy 68 % half-width against unit (SK patch)
yardstick_ladder.png  Y1 surrogate, Y2 achieved and frozen readouts on the same events (SK, WCTE)
Run on Deepthought: tier1_figures.py --out results/round2/20261122_tier1/figures
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import tier1_gate as tg  # noqa: E402

T1 = Path("/ceph/dwong/jepa/returns/r2_tier1_2026-11-22")
SEED = {"A": 9000, "B": 9001, "C": 9002}
UNITS = {"sk": [450, 150, 50, 15, 5], "wcte": [450, 50, 15, 5, 1.5]}


def metric(rid, path):
    p = T1 / "eval" / rid / "step_10000" / "sealed.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())["metrics"]
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
        if d is None:
            return None
    return float(d)


def vertex(rid):
    return metric(rid, ("vertex", "ridge_spatial_moment_latent", "mean_m"))


def cells(det, mask, unit):
    out = {}
    for s, seed in SEED.items():
        v = vertex(tg.run_id(det, mask, unit, seed))
        if v is not None:
            out[s] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    u50 = json.loads((REPO / "results/round2/20261122_tier1/U50.json").read_text())
    U50 = {"sk": u50["sk__mae_repaired__s9000"]["u50_ns"], "wcte": u50["wcte__mae_repaired__s9000"]["u50_ns"]}
    col = {"patch": "#1f5fa8", "random": "#c8553d", "parity": "#5b8c3a"}
    mk = {"A": "o", "B": "s", "C": "^"}
    data = {}

    # 1 un-bundling
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, det, units in ((axes[0], "sk", [450, 50]), (axes[1], "wcte", [450, 50, 5, 1.5])):
        for j, mask in enumerate(("patch", "random")):
            for i, u in enumerate(units):
                c = cells(det, mask, u)
                data.setdefault(det, {}).setdefault(mask, {})[str(u)] = c
                for s, v in c.items():
                    ax.scatter(i + (j - 0.5) * 0.25, v, color=col[mask], marker=mk[s], s=36,
                               label=f"{mask}, seed {s}" if i == 0 else None)
        ax.set_xticks(range(len(units)), [f"{u:g} ns" for u in units])
        ax.set(title=f"{det.upper()}: mask × time unit (sealed)", ylabel="vertex error (m)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "unbundling.png", dpi=130)
    plt.close(fig)

    # 2 dose-response with u50
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, det in ((axes[0], "sk"), (axes[1], "wcte")):
        for mask in ("patch", "random"):
            xs, ms = [], []
            for u in UNITS[det]:
                c = cells(det, mask, u)
                data.setdefault(det, {}).setdefault(mask, {})[str(u)] = c
                for s, v in c.items():
                    ax.scatter(u, v, color=col[mask], marker=mk[s], s=24, alpha=0.7)
                if c:
                    xs.append(u)
                    ms.append(np.mean(list(c.values())))
            if xs:
                ax.plot(xs, ms, color=col[mask], label=f"{mask} (seed mean)")
        ax.axvline(U50[det], color="k", ls="--", lw=1, label=f"u₅₀ = {U50[det]:.1f} ns")
        ax.set(xscale="log", xlabel="time unit (ns)", ylabel="vertex error (m)", title=f"{det.upper()} dose-response (sealed)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "dose_response.png", dpi=130)
    plt.close(fig)

    # 3 energy against unit (SK patch)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for part, c_ in (("electron", "#7a3b9e"), ("muon", "#2a9d8f")):
        xs, ms = [], []
        for u in UNITS["sk"]:
            vals = [metric(tg.run_id("sk", "patch", u, seed), ("energy", part, "half_width_68")) for seed in SEED.values()]
            vals = [v for v in vals if v is not None]
            data.setdefault("sk_energy", {}).setdefault(part, {})[str(u)] = vals
            for v in vals:
                ax.scatter(u, v, color=c_, s=20, alpha=0.7)
            if vals:
                xs.append(u)
                ms.append(np.mean(vals))
        ax.plot(xs, ms, color=c_, label=f"{part} (seed mean)")
    ax.set(xscale="log", xlabel="time unit (ns)", ylabel="68 % half-width of (Ê−E)/E", title="SK patch: energy against unit (sealed)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "energy_vs_unit.png", dpi=130)
    plt.close(fig)

    # 4 yardstick ladder
    y = json.loads((REPO / "results/round2/20261122_tier1/YARDSTICKS.json").read_text())["detectors"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, det in ((axes[0], "sk"), (axes[1], "wcte")):
        d = y.get(det)
        if not d:
            continue
        rows = [("Y1 joint (surrogate)", d["Y1"]["joint"]["median"]),
                ("Y1 joint ×√12 counts", d["Y1"].get("honesty_sqrt12", {}).get("joint_median")),
                ("Y2 counts (truth start)", d["Y2_truth_start"]["counts"]["median_3d_m"]),
                ("Y2 time (truth start)", d["Y2_truth_start"]["time"]["median_3d_m"]),
                ("Y2 joint (truth start)", d["Y2_truth_start"]["joint"]["median_3d_m"])]
        for rid, row in d["ridge_same_events"].items():
            if row and "ridge_spatial_moment_latent" in row:
                rows.append((f"Ridge {rid.split('__')[1]}", row["ridge_spatial_moment_latent"]["median_m"]))
        rows = [(n, v) for n, v in rows if v is not None]
        ax.barh([n for n, _ in rows], [v for _, v in rows], color="#6c8ebf")
        ax.set(xscale="log", xlabel="median 3-D error / σ_vertex (m)", title=f"{det.upper()}: yardstick ladder (same events)")
        ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out / "yardstick_ladder.png", dpi=130)
    plt.close(fig)
    (out / "figures_data.json").write_text(json.dumps(data, indent=1) + "\n")
    print("wrote figures to", out)


if __name__ == "__main__":
    main()
