"""R-4 gate table against the constant predictor, from training_01/training.json (R-3 gates).

Run from the repository root: python -B results/pmt_r4_dev_2026-10-18/gate_table.py
"""
from pathlib import Path
import json

here = Path(__file__).parent
t = json.loads((here / "training_01" / "training.json").read_text())
names = {"vertex_rmse_m": "Vertex RMSE", "direction_mean_deg": "Direction mean", "energy_relative_mae": "E_dep MAE"}
lines = ["| Seed | Subgroup | n | " + " | ".join(f"{v} model / constant (ratio)" for v in names.values()) + " | R-3 gates |",
         "|---|---|---:|" + "---|"*len(names) + "---|"]
for r in t["results"]:
    rel = r["relative_gates"]
    rows = [("pooled", rel["table"]["pooled"])] + [(f"{k} MeV", v) for k, v in rel["table"]["subgroups"].items()]
    for label, row in rows:
        cells = []
        for k in names:
            e = row[k]
            ratio = "undefined" if e["ratio"] is None else f"{e['ratio']:.3f}"
            cells.append(f"{e['model']:.4g} / {e['constant']:.4g} ({ratio})")
        key = "pooled" if label == "pooled" else f"subgroup_{label.split()[0]}"
        gates = [g for g, v in rel["gates"].items() if g.endswith(key)]
        verdict = ", ".join(f"{g.split('_')[0]} {'pass' if rel['gates'][g]['passed'] else 'FAIL'}" for g in gates)
        lines.append(f"| {r['seed']} | {label} | {row['n']} | " + " | ".join(cells) + f" | {verdict} |")
    lines.append(f"| {r['seed']} | **all gates** | | | | | **{'PASS' if rel['passed'] else 'FAIL'}** |")
cb = t["charge_benchmark"]
lines += ["", f"Charge-calibration benchmark ({cb['endpoint']}): pooled relative MAE {cb['relative_mae']:.3f}; "
          + ", ".join(f"{k} MeV {v:.3f}" for k, v in cb["by_energy"].items()) + "."]
md = "\n".join(lines) + "\n"
(here / "gate_table.md").write_text(md)
print(md)
