"""R-4b summary: 2x2 validation diagnostic, data-limitation arms, and R-4b beside R-4 (test).

Run from the repository root: python -B results/pmt_r4b_dev_2026-10-18/summarise.py
"""
from pathlib import Path
import json
import numpy as np

here = Path(__file__).parent
r4 = here.parent / "pmt_r4_dev_2026-10-18" / "training_01" / "training.json"


def load(path):
    t = json.loads(Path(path).read_text())
    seeds = []
    for r in t["results"]:
        g = r["relative_gates"]
        pooled = g["table"]["pooled"]
        sub = g["table"]["subgroups"]
        seeds.append(dict(seed=r["seed"], vertex=pooled["vertex_rmse_m"]["ratio"], direction=pooled["direction_mean_deg"]["model"],
                          energy=pooled["energy_relative_mae"]["ratio"],
                          vertex_worst=max(v["vertex_rmse_m"]["ratio"] for v in sub.values()),
                          energy_worst=max(v["energy_relative_mae"]["ratio"] for v in sub.values()),
                          passed=g["passed"], failures=g["failures"], best_epoch=r["best_epoch"]))
    return dict(split=t.get("evaluate_split", "test"), n_train=t.get("n_train_placements"), n_eval=t.get("n_eval_placements"),
                seeds=seeds, charge=t["charge_benchmark"]["relative_mae"])


def fmt(x):
    return f"{np.mean(x):.3f} [{min(x):.3f}–{max(x):.3f}]"


arms = {"20 ns, dark on": "diag_u20_dark", "20 ns, dark off": "diag_u20_nodark", "1 µs, dark on": "diag_u1000_dark", "1 µs, dark off": "diag_u1000_nodark"}
lines = ["## 2×2 diagnostic (validation split only; mean [min–max] over seeds 30–32; model ÷ constant)", "",
         "| Arm | Vertex RMSE, pooled | Vertex, worst energy | Direction (°) | E_dep MAE, pooled | Seeds passing all gates |", "|---|---|---|---|---|---|"]
out = {"diagnostic": {}, "learning_curve": {}}
for label, d in arms.items():
    s = load(here / d / "training.json"); out["diagnostic"][label] = s
    v = [x["vertex"] for x in s["seeds"]]
    lines.append(f"| {label} | {fmt(v)} | {fmt([x['vertex_worst'] for x in s['seeds']])} | {fmt([x['direction'] for x in s['seeds']])} | "
                 f"{fmt([x['energy'] for x in s['seeds']])} | {sum(x['passed'] for x in s['seeds'])}/3 |")
lines += ["", "## Data limitation (20 ns, dark on; validation split only)", "",
          "| Training events per energy | Training placements | Vertex RMSE, pooled | E_dep MAE, pooled |", "|---|---|---|---|"]
for k, d in ((4, "lc_train4"), (6, "lc_train6"), (8, "diag_u20_dark")):
    s = load(here / d / "training.json"); out["learning_curve"][k] = s
    lines.append(f"| {k} of 8 | {s['n_train']} | {fmt([x['vertex'] for x in s['seeds']])} | {fmt([x['energy'] for x in s['seeds']])} |")
lines += ["", "## R-4b beside R-4 (test split; R-4 stays as recorded)", "",
          "| Run | Seed | Vertex, pooled | Vertex, worst energy | Direction (°) | E_dep, pooled | E_dep, worst energy | Verdict |", "|---|---|---|---|---|---|---|---|"]
for name, path in (("R-4 (1 µs)", r4), ("R-4b (20 ns rule)", here / "r4b_training" / "training.json")):
    s = load(path); out[name] = s
    for x in s["seeds"]:
        lines.append(f"| {name} | {x['seed']} | {x['vertex']:.3f} | {x['vertex_worst']:.3f} | {x['direction']:.1f} | {x['energy']:.3f} | {x['energy_worst']:.3f} | "
                     f"{'PASS' if x['passed'] else 'FAIL: ' + '; '.join(x['failures'])} |")
md = "\n".join(lines) + "\n"
(here / "summary.md").write_text(md)
(here / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(md)
