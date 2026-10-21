"""E45 summary: NHits N = 4 + prompt window, arms (a)-(c), validation split only; pass mark (E45).

Run from the repository root: python -B results/pmt_r4c_dev_2026-10-18/e45/summarise.py
"""
from pathlib import Path
import json
import numpy as np

here = Path(__file__).parent
ARMS = {"a": "prompt window, dark on", "b": "prompt window, dark off", "c": "dark on, truth-dark removed (upper bound)"}
COUNTS = ("prompt_dropped", "prompt_dropped_source", "prompt_dropped_source_no_dark", "truth_dark_dropped")
out = {"arms": {}}
lines = ["## Arms (validation split only; seeds 20–22; model ÷ constant, pooled)", "",
         "| Arm | Setting | Vertex (20 / 21 / 22) | Mean | Worst energy cell, vertex | Direction (°) | E_dep | Fallback |", "|---|---|---|---|---|---|---|---|"]
for arm, label in ARMS.items():
    t = json.loads((here / arm / "training.json").read_text())
    g = json.loads((here / "data" / f"validation_{arm}.json").read_text())
    seeds = []
    for r in t["results"]:
        tab = r["relative_gates"]["table"]
        seeds.append(dict(seed=r["seed"], vertex=tab["pooled"]["vertex_rmse_m"]["ratio"], direction=tab["pooled"]["direction_mean_deg"]["model"],
                          energy=tab["pooled"]["energy_relative_mae"]["ratio"],
                          worst_energy_vertex=max(v["vertex_rmse_m"]["ratio"] for v in tab["subgroups"].values()), passed=r["passed"]))
    rc = g["readout_counts"]; ev = g["events"]
    val = [e for e in ev if e["role"] == "validation"]
    counts = {c: rc.get(c) for c in COUNTS} | {f"{c}_by_cell": rc.get(f"{c}_by_cell") for c in COUNTS}
    counts |= dict(placements=len(ev), digits=sum(e["digits"] for e in ev), model_digits=sum(e["model_digits"] for e in ev),
                   validation_digits=sum(e["digits"] for e in val), validation_model_digits=sum(e["model_digits"] for e in val),
                   nhits_fallback=rc["nhits_fallback"], nhits_fallback_by_cell=rc["nhits_fallback_by_cell"])
    out["arms"][arm] = dict(setting=label, seeds=seeds, counts=counts, readout_settings=g["readout_settings"])
    v = [s["vertex"] for s in seeds]
    lines.append(f"| ({arm}) | {label} | {' / '.join(f'{x:.3f}' for x in v)} | {np.mean(v):.3f} | {max(s['worst_energy_vertex'] for s in seeds):.2f} | "
                 f"{min(s['direction'] for s in seeds):.1f}–{max(s['direction'] for s in seeds):.1f} | {min(s['energy'] for s in seeds):.2f}–{max(s['energy'] for s in seeds):.2f} | "
                 f"{rc['nhits_fallback']} of {len(ev)} |")
va = [s["vertex"] for s in out["arms"]["a"]["seeds"]]; vb = [s["vertex"] for s in out["arms"]["b"]["seeds"]]
crit = dict(every_seed_le_0p88=bool(max(va) <= 0.88), mean_gap=float(np.mean(va) - np.mean(vb)), mean_gap_le_0p05=bool(abs(np.mean(va) - np.mean(vb)) <= 0.05))
crit["passed"] = crit["every_seed_le_0p88"] and crit["mean_gap_le_0p05"]; out["pass_mark"] = crit
lines += ["", f"**Pass mark (E45):** arm (a) ≤ 0.88 in every seed: {crit['every_seed_le_0p88']} (max {max(va):.3f}); "
          f"|mean(a) − mean(b)| ≤ 0.05: {crit['mean_gap_le_0p05']} ({crit['mean_gap']:+.3f}). **{'PASS' if crit['passed'] else 'FAIL'}**", "",
          "## Dropped digits (all roles; validation in brackets)", "",
          "| Arm | Digits read | Into the model | Prompt-window drops | of which source | of which source, no dark PE | Truth-dark removed |", "|---|---|---|---|---|---|---|"]
for arm, a in out["arms"].items():
    c = a["counts"]
    lines.append(f"| ({arm}) | {c['digits']} ({c['validation_digits']}) | {c['model_digits']} ({c['validation_model_digits']}) | {c['prompt_dropped']} | "
                 f"{c['prompt_dropped_source']} | {c['prompt_dropped_source_no_dark']} | {c['truth_dark_dropped']} |")
lines += ["", "Per (role, energy) cell counts: `summary.json`."]
(here / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
(here / "summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
