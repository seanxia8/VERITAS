"""Round-2 report figures (runbook §10): RF − MAE difference, time-scale diagnostics, particle tasks.

Reads only committed outputs (``results/round2/20261112_eval/readings.json`` and the round-1
``results/jepa/20261103_followup/readings.json`` A-3 particle tasks). No training, no probing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DETS = ("sk", "juno", "hk")
LABEL = {"wcte": "WCTE", "sk": "SK", "juno": "JUNO", "hk": "HK"}
ARMS = (("mae_mse", "MAE"), ("target_posfree", "position-free JEPA"), ("init", "random init"),
        ("sigreg_jepa", "fixed-label JEPA"), ("mae_mse_rf", "MAE (RF)"))
COLORS = {"mae_mse": "#1f77b4", "target_posfree": "#2ca02c", "init": "#7f7f7f",
          "sigreg_jepa": "#d62728", "mae_mse_rf": "#ff7f0e"}
TASKS = (("mue_auc", "μ/e AUC"), ("electron_direction_mlp_median_deg", "e direction (deg)"),
         ("muon_direction_mlp_median_deg", "μ direction (deg)"), ("electron_energy_r2", "e energy R²"),
         ("muon_energy_r2", "μ energy R²"))
#: round-1 A-3 item per round-2 arm (target_posfree has no committed A-3 entry)
WCTE_A3 = {"mae_mse": "mae_within_trigger9000", "init": "init9000", "sigreg_jepa": "fix9000"}


def wcte_particle(a3: dict) -> dict:
    out = {}
    for arm, key in WCTE_A3.items():
        t = a3[key]["tasks"]
        dom = t["domains"]
        v = {"mue_auc": t["mue_classification"]["model"]["auc"],
             "electron_direction_mlp_median_deg": dom["electron"]["direction"]["mlp"]["median_deg"],
             "muon_direction_mlp_median_deg": dom["muon"]["direction"]["mlp"]["median_deg"],
             "electron_energy_r2": (dom["electron"]["energy_model"].get("r2_mean")
                                    if dom["electron"]["energy_model"]["status"] == "evaluated" else None),
             "muon_energy_r2": (dom["muon"]["energy_model"].get("r2_mean")
                                if dom["muon"]["energy_model"]["status"] == "evaluated" else None)}
        out[arm] = v
    return out


def main(argv=None) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = argparse.ArgumentParser()
    p.add_argument("--readings", default="results/round2/20261112_eval/readings.json")
    p.add_argument("--wcte-a3", default="results/jepa/20261103_followup/readings.json")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    d = json.loads(Path(a.readings).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    two = {det: d["detectors"][det]["readings"]["two_seed"] for det in DETS}

    # ---- RF − MAE (R2-e)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(len(DETS))
    e = [two[det]["R2-e"] for det in DETS]
    diff = np.array([r["vertex"]["mean_difference_m"] for r in e])
    lo = np.array([r["vertex"]["ci95_m"][0] for r in e])
    hi = np.array([r["vertex"]["ci95_m"][1] for r in e])
    ax[0].errorbar(x, diff, yerr=[diff - lo, hi - diff], fmt="o", color=COLORS["mae_mse_rf"], capsize=4)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].axhspan(-0.015, 0.015, color="0.85", label="round-1 MAE seed spread (±1.5 cm)")
    ax[0].set_ylabel("vertex error, RF − MAE (m)")
    ax[0].legend(fontsize=7, loc="lower left")
    ax[1].bar(x - 0.18, [r["token_r2_difference"] for r in e], 0.36, label="token ln q R²")
    ax[1].bar(x + 0.18, [r["f_event_difference"] for r in e], 0.36, label="f_event")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_ylabel("RF − MAE")
    ax[1].legend(fontsize=7)
    w = 0.16
    for i, (k, lab) in enumerate(TASKS):
        ax[2].bar(x + (i - 2) * w, [r["particle_difference"][k] for r in e], w, label=lab)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_ylabel("RF − MAE (task units)")
    ax[2].legend(fontsize=6)
    for axis in ax:
        axis.set_xticks(x, [LABEL[t] for t in DETS])
    fig.suptitle("R2-e: receptive-field-matched MAE minus MAE (seed 9000)")
    fig.tight_layout()
    fig.savefig(out / "rf_minus_mae.png", dpi=130)
    plt.close(fig)

    # ---- time-scale diagnostics (§6)
    from .ext_r2_results import wcte_loss_shares  # round-1 WCTE MAE train log (read-only)
    w = wcte_loss_shares()
    wls = {"start": w["start"]["shares"]["time"], "end": w["end"]["shares"]["time"]} if w else None
    (out / "wcte_loss_shares_round1.json").write_text(json.dumps(w, indent=1, default=float) + "\n")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    stds = [d["wcte_time_feature"]["time_feature_std"]] + [d["detectors"][t]["time_feature"]["time_feature_std"] for t in DETS]
    ax[0].bar(range(4), stds, color=["0.5"] + ["#1f77b4"] * 3)
    ax[0].set_xticks(range(4), ["WCTE"] + [LABEL[t] for t in DETS])
    ax[0].set_ylabel("time-feature std over hit tokens")
    labels, starts, ends = [], [], []
    for det in DETS:
        for arm in ("mae_mse", "mae_mse_rf"):
            sh = d["detectors"][det]["loss_shares"].get(arm)
            if sh:
                labels.append(f"{LABEL[det]}\n{'RF' if arm.endswith('rf') else 'MAE'}")
                starts.append(sh["start"]["shares"]["time"])
                ends.append(sh["end"]["shares"]["time"])
    xx = np.arange(len(labels))
    ax[1].bar(xx - 0.2, starts, 0.4, label="first 100 steps")
    ax[1].bar(xx + 0.2, ends, 0.4, label="last 100 steps")
    if wls:
        for val, ls_ in ((wls["start"], "--"), (wls["end"], ":")):
            ax[1].axhline(val, color="k", ls=ls_, lw=0.9)
            ax[1].axhspan(val / 2, val * 2, color="0.9", zorder=0)
    ax[1].set_yscale("log")
    ax[1].set_xticks(xx, labels, fontsize=7)
    ax[1].set_ylabel("time-term share of the MAE loss")
    ax[1].legend(fontsize=7)
    ax[1].set_title("lines: WCTE round 1 (dashed start, dotted end); grey: ×/÷2", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "time_scale.png", dpi=130)
    plt.close(fig)

    # ---- particle tasks with the five-seed init floor band
    wp = wcte_particle(json.loads(Path(a.wcte_a3).read_text())["A-3"])
    cols = ("wcte",) + DETS
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.6))
    for j, (k, lab) in enumerate(TASKS):
        axis = ax[j]
        for c, det in enumerate(cols):
            if det != "wcte":
                fb = two[det]["floor_band"].get(k)
                if fb:
                    axis.add_patch(plt.Rectangle((c - 0.4, fb["min"]), 0.8, fb["max"] - fb["min"],
                                                 color="0.85", zorder=0))
            for i, (arm, _) in enumerate(ARMS):
                if det == "wcte":
                    v = wp.get(arm, {}).get(k)
                else:
                    v = two[det]["metrics"].get(f"{arm}:9000", {}).get("particle", {}).get(k)
                if v is not None:
                    axis.plot(c - 0.3 + i * 0.15, v, "o", color=COLORS[arm], ms=5)
        axis.set_xticks(range(len(cols)), [LABEL[t] for t in cols])
        axis.set_title(lab, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[arm], label=lab) for arm, lab in ARMS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color="0.85", label="init floor band (5 seeds)"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "particle_tasks.png", dpi=130)
    plt.close(fig)
    (out / "wcte_particle_round1.json").write_text(json.dumps(wp, indent=1) + "\n")
    print("ok", sorted(x.name for x in out.iterdir()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
