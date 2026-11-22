#!/usr/bin/env python3
"""r2-tier1 T3.6 / T4.3: RESCORE_LUCID.{md,json} and BUDGET_AXIS.{md,json,png} from tier1_eval outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

T1 = Path("/ceph/dwong/jepa/returns/r2_tier1_2026-11-22")
LUCID_RUNS = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/train/runs")
DETS = ("wcte", "sk", "juno", "hk")


def load(root, rid, step, cohort):
    p = Path(root) / rid / f"step_{step:05d}" / f"{cohort}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def g(d, *keys):
    for k in keys:
        if d is None or k not in d:
            return None
        d = d[k]
    return d


def f(x, nd=4):
    return "–" if x is None else f"{x:.{nd}f}"


def summary(js):
    m = js["metrics"]
    v = m["vertex"]
    row = {"ridge_spatial": g(v, "ridge_spatial_moment_latent", "mean_m"),
           "ridge_all_mean": g(v, "ridge_all_mean", "mean_m"),
           "ridge_raw": g(v, "ridge_raw_spatial_moment", "mean_m"),
           "ridge_spatial_full": g(v, "ridge_spatial_moment_full", "mean_m"),
           "mlp_full_mean3": g(v, "mlp_spatial_moment_full_mean3", "mean_m"),
           "mlp_latent_mean3": g(v, "mlp_spatial_moment_latent_mean3", "mean_m"),
           "attentive_mean3": g(v, "attentive_mean3", "mean_m"),
           "f_event": g(m, "collapse", "f_event"), "f_layout": g(m, "collapse", "f_layout")}
    for dom in ("muon", "electron"):
        row[f"dir_{dom}_ridge_median_deg"] = g(m, "direction", f"{dom}_ridge_spatial", "median_deg")
        row[f"dir_{dom}_ridge_q68_deg"] = g(m, "direction", f"{dom}_ridge_spatial", "q68_deg")
        row[f"dir_{dom}_committed_mlp_median_deg"] = g(m, "direction", f"{dom}_committed_mlp_all_mean", "median_deg")
        row[f"energy_{dom}_hw68"] = g(m, "energy", dom, "half_width_68")
        row[f"energy_{dom}_rms"] = g(m, "energy", dom, "rms")
        row[f"energy_{dom}_r2_E"] = g(m, "energy", dom, "r2_E")
        row[f"energy_{dom}_n"] = g(m, "energy", f"{dom}_n_contained")
    row["mu_misid_99e"] = g(m, "pid", "mu_misid_at_99_e_eff")
    row["e_misid_99mu"] = g(m, "pid", "e_misid_at_99_mu_eff")
    row["auc"] = g(m, "pid", "auc")
    return row


def rescore(args):
    manifest = json.loads(Path(args.manifest).read_text())
    rows = []
    for case in manifest["cases"]:
        rid, step = case["run_id"], case["step"]
        for cohort in ("validation", "sealed"):
            js = load(args.eval_root, rid, step, cohort)
            r = {"run_id": rid, "detector": case["detector"], "arm": case["arm"], "seed": case["seed"],
                 "step": step, "cohort": cohort, "available": js is not None}
            if js:
                r.update(summary(js))
            rows.append(r)
    idx = {(r["run_id"], r["step"], r["cohort"]): r for r in rows}
    for r in rows:
        if not r["available"] or r["arm"] == "init" or r["step"] != 10000:
            continue
        matched = idx.get((r["run_id"], 0, r["cohort"]))
        legacy = idx.get((f"{r['detector']}__init__s{r['seed']}", 0, r["cohort"]))
        for name, ref in (("gain_vs_matched_init_m", matched), ("gain_vs_legacy_init_m", legacy)):
            r[name] = (ref["ridge_spatial"] - r["ridge_spatial"]) if ref and ref.get("ridge_spatial") and \
                r.get("ridge_spatial") else None
    out = {"rows": rows, "manifest": args.manifest}
    Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    L = ["# RESCORE_LUCID — existing LUCiD checkpoints re-scored (r2-tier1 T3)", "",
         "Readouts frozen in `docs/round2/READOUT_SPEC_v1.md`; probes fitted on `dev_probe_train`. "
         "Validation = `dev_probe_test` (≈ 3,400 photon bombs, the split the readout was designed on); "
         "sealed = `sealed_test` (≈ 10,000 photon bombs, never used for any choice). Vertex: mean 3-D error (m). "
         "Matched init = the run's own step-0 snapshot (same encoder configuration); legacy init = "
         "`build_r2_arm('init')` (window 64). Gains are init error − arm error on the primary readout (positive = better).", ""]
    for det in DETS:
        sub = [r for r in rows if r["detector"] == det and r["available"]]
        if not sub:
            continue
        L += [f"## {det.upper()}", "",
              "| Run | Step | Cohort | Ridge spatial | Gain vs matched init | Gain vs legacy init | All-mean | Raw | MLP full | MLP latent | Attentive | f_event | f_layout |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for r in sub:
            L.append(f"| {r['run_id']} | {r['step']} | {r['cohort']} | {f(r['ridge_spatial'])} | "
                     f"{f(r.get('gain_vs_matched_init_m'))} | {f(r.get('gain_vs_legacy_init_m'))} | "
                     f"{f(r['ridge_all_mean'])} | {f(r['ridge_raw'])} | {f(r['mlp_full_mean3'])} | "
                     f"{f(r['mlp_latent_mean3'])} | {f(r['attentive_mean3'])} | {f(r['f_event'], 3)} | {f(r['f_layout'], 3)} |")
        L += ["", "| Run | Step | Cohort | μ dir. Ridge med / 68 % (°) | e dir. Ridge med / 68 % (°) | μ dir. committed MLP (°) | e dir. committed MLP (°) | e energy hw68 / RMS (n) | μ energy hw68 / RMS (n) | μ mis-ID @99 % e | e mis-ID @99 % μ | AUC |",
              "|---|---:|---|---|---|---:|---:|---|---|---:|---:|---:|"]
        for r in sub:
            L.append(f"| {r['run_id']} | {r['step']} | {r['cohort']} | "
                     f"{f(r['dir_muon_ridge_median_deg'], 2)} / {f(r['dir_muon_ridge_q68_deg'], 2)} | "
                     f"{f(r['dir_electron_ridge_median_deg'], 2)} / {f(r['dir_electron_ridge_q68_deg'], 2)} | "
                     f"{f(r['dir_muon_committed_mlp_median_deg'], 2)} | {f(r['dir_electron_committed_mlp_median_deg'], 2)} | "
                     f"{f(r['energy_electron_hw68'], 3)} / {f(r['energy_electron_rms'], 3)} ({r['energy_electron_n']}) | "
                     f"{f(r['energy_muon_hw68'], 3)} / {f(r['energy_muon_rms'], 3)} ({r['energy_muon_n']}) | "
                     f"{f(r['mu_misid_99e'], 4)} | {f(r['e_misid_99mu'], 4)} | {f(r['auc'], 4)} |")
        L.append("")
    missing = sorted({r["run_id"] + f"@{r['step']}/{r['cohort']}" for r in rows if not r["available"]})
    L += ["## Feature and probe provenance of the committed particle suite (T3.3)", "",
          "- Committed particle tasks (`ext_r2.metrics_from_streams`, `ext_r2.py:1105-1131`) use the **all-mean token "
          "feature** of each μ/e event (`la._mean_tokens(tokens, None)`, `ext_r2.py:1040`): μ/e AUC by "
          "`ext_followup._logistic_auc` (`ext_r2.py:1110`); direction by `ext_followup._direction_tasks`, reported as "
          "its MLP median angle (`ext_r2.py:1117-1123`); energy by `ext_followup._energy_task`, linear R² of "
          "`deposited_MeV` on LUCiD-contained events (`ext_r2.py:1124-1131`).",
          "- Committed vertex readouts (`ext_r2_repair_probes.evaluate`, `ext_r2_repair_probes.py:171-215`): Ridge on the "
          "`VERTEX_FORMS` (incl. `spatial_moment_latent` = Appendix A.2) and the MLP on `spatial_moment_full` "
          "with probe seed = run seed.",
          "- New here (READOUT_SPEC_v1): direction, energy (ln E, resolution metrics) and PID on the spatial-moment "
          "latent (A.2) features instead of the all-mean; the committed MLP direction probe is kept for continuity.", ""]
    if missing:
        L += ["Not available at build time: " + ", ".join(missing), ""]
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n")
    print("rescore rows", len(rows), "missing", len(missing))


def budget(args):
    runs = {r.split("=")[0]: r.split("=")[1] for r in args.runs}
    steps = (0, 500, 1000, 2000, 5000, 10000)
    data = {}
    for label, rid in runs.items():
        data[label] = {"run_id": rid, "steps": {}}
        for s in steps:
            js = load(args.eval_root, rid, s, "validation")
            if not js and args.sealed_root:  # steps scored in full (0, 10,000) live in eval/
                js = load(args.sealed_root, rid, s, "validation")
            if js:
                data[label]["steps"][s] = js["metrics"]["vertex"]["ridge_spatial_moment_latent"]["mean_m"]
    sealed = {}
    if args.sealed_root:
        for label, rid in runs.items():
            for s in (5000, 10000):
                js = load(args.sealed_root, rid, s, "sealed")
                if js:
                    sealed.setdefault(label, {})[s] = js["metrics"]["vertex"]["ridge_spatial_moment_latent"]["mean_m"]
    Path(args.out).with_suffix(".json").write_text(json.dumps({"validation": data, "sealed_5000_10000": sealed},
                                                              indent=1) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, d in data.items():
        ax = axes[0] if label.startswith("WCTE") else axes[1]
        xs = sorted(d["steps"])
        ax.plot([max(x, 50) for x in xs], [d["steps"][x] for x in xs], marker="o", label=label)
    for ax, t in zip(axes, ("WCTE (dev_probe_test)", "SK (dev_probe_test)")):
        ax.set(xscale="log", xlabel="training step (0 plotted at 50)", ylabel="spatial-moment Ridge vertex error (m)", title=t)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(args.out).with_suffix(".png"), dpi=130)
    L = ["# BUDGET_AXIS — vertex error against training step (r2-tier1 T4, exploratory)", "",
         "Spatial-moment Ridge readout (READOUT_SPEC_v1), `dev_probe_test` photon bombs, snapshots of each run. "
         "Exploratory, not registered (§2.6), except the sealed 5,000/10,000-step values that feed F5.", "",
         "| Run | " + " | ".join(str(s) for s in steps) + " |", "|---|" + "---:|" * len(steps)]
    for label, d in data.items():
        L.append(f"| {label} (`{d['run_id']}`) | " + " | ".join(f(d['steps'].get(s)) for s in steps) + " |")
    if sealed:
        L += ["", "Sealed (F5 inputs):", "", "| Run | 5,000 | 10,000 |", "|---|---:|---:|"]
        for label, d in sealed.items():
            L.append(f"| {label} | {f(d.get(5000))} | {f(d.get(10000))} |")
    L += ["", f"Figure: `{Path(args.out).with_suffix('.png').name}`.", ""]
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n")
    print("budget", {k: len(v["steps"]) for k, v in data.items()})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rescore")
    r.add_argument("--manifest", required=True)
    r.add_argument("--eval-root", default=str(T1 / "eval"))
    r.add_argument("--out", required=True)
    b = sub.add_parser("budget")
    b.add_argument("--runs", nargs="+", required=True, help="label=run_id")
    b.add_argument("--eval-root", default=str(T1 / "eval_budget"))
    b.add_argument("--sealed-root", default=str(T1 / "eval"))
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    {"rescore": rescore, "budget": budget}[a.cmd](a)


if __name__ == "__main__":
    main()
