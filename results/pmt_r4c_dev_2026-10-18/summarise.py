"""E40 vertex diagnostic summary: arms 0-4 (validation split only), the 56179eb validation reference,
per-head losses and gradient norms, and the input comparisons.

Run from the repository root: python -B results/pmt_r4c_dev_2026-10-18/summarise.py
"""
from pathlib import Path
import json
import numpy as np

here = Path(__file__).parent
old = json.loads((here.parent / "pmt_multienergy_dev_2026-10-16" / "training_01" / "training.json").read_text())
old_hash = {r["seed"]: r["checkpoint_sha256"] for r in old["results"]}
ARMS = {"arm0": ("first digit", "initial", "arm0"), "arm1": ("trigger", "initial", "arm1"), "arm2": ("trigger", "deposited", "arm2"),
        "arm3": ("window start", "initial", "arm3"), "arm4": ("window start", "deposited", "arm4")}
HEADS = ("direction", "vertex", "energy")


def fmt(x):
    return f"{np.mean(x):.3f} [{min(x):.3f}–{max(x):.3f}]"


def share(h):
    tot = sum(h[k] for k in HEADS)
    return {k: h[k] / tot for k in HEADS}


out = {"arms": {}}
lines = ["## Arms (validation split only; seeds 20–22; mean [min–max]; model ÷ constant)", "",
         "| Arm | Time origin | Energy target | Vertex RMSE, pooled | Direction (°) | Energy MAE, pooled | Best epochs | Checkpoints = 56179eb |", "|---|---|---|---|---|---|---|---|"]
for arm, (origin, target, d) in ARMS.items():
    t = json.loads((here / d / "training.json").read_text())
    seeds = []
    for r in t["results"]:
        pooled = r["relative_gates"]["table"]["pooled"]
        seeds.append(dict(seed=r["seed"], vertex=pooled["vertex_rmse_m"]["ratio"], direction=pooled["direction_mean_deg"]["model"],
                          energy=pooled["energy_relative_mae"]["ratio"], best_epoch=r["best_epoch"], same_as_56179eb=r["checkpoint_sha256"] == old_hash.get(r["seed"]),
                          head_losses=r["head_losses"]))
    out["arms"][arm] = dict(origin=origin, target=target, seeds=seeds, target_scale=t["target_scale"])
    lines.append(f"| {arm} | {origin} | {target} | {fmt([s['vertex'] for s in seeds])} | {fmt([s['direction'] for s in seeds])} | "
                 f"{fmt([s['energy'] for s in seeds])} | {', '.join(str(s['best_epoch']) for s in seeds)} | {sum(s['same_as_56179eb'] for s in seeds)}/3 |")
ref_path = here / "reference_56179eb.json"
if ref_path.exists():
    ref = json.loads(ref_path.read_text()); out["reference_56179eb"] = ref
    v = [r["vertex_ratio"] for r in ref["results"]]; e = [r["energy_ratio"] for r in ref["results"]]
    lines.append(f"| 56179eb checkpoints (reference) | first digit | initial | {fmt(v)} | {fmt([r['direction_deg'] for r in ref['results']])} | {fmt(e)} | — | — |")
lines += ["", "## Target scale (training split; constant-predictor loss per head, vertex in 0.35 m units)", "",
          "| Arm | Direction | Vertex | Energy | SD of log-energy target | Energy share of constant loss |", "|---|---|---|---|---|---|"]
for arm, a in out["arms"].items():
    c = a["target_scale"]["constant_loss_train"]
    lines.append(f"| {arm} | {c['direction']:.3f} | {c['vertex']:.3f} | {c['energy']:.3f} | {a['target_scale']['log_energy_target_sd_train']:.3f} | {share(c)['energy']:.2f} |")
for point, label in (("initialisation", "at initialisation"), ("selected_checkpoint", "at the selected checkpoint")):
    lines += ["", f"## Per-head loss {label} (mean over seeds; validation loss ÷ constant loss, and share of the summed training loss; encoder gradient norm)", "",
              "| Arm | Vertex val ÷ const | Energy val ÷ const | Direction val | Train share v / e / d | Encoder grad norm v / e / d |", "|---|---|---|---|---|---|"]
    for arm, a in out["arms"].items():
        cv = a["target_scale"]["constant_loss_validation"]
        h = [s["head_losses"][point] for s in a["seeds"]]
        rv = np.mean([x["validation"]["vertex"] / cv["vertex"] for x in h]); re = np.mean([x["validation"]["energy"] / cv["energy"] for x in h])
        dv = np.mean([x["validation"]["direction"] for x in h])
        sh = {k: np.mean([share(x["train"])[k] for x in h]) for k in HEADS}
        g = {k: np.mean([x["encoder_grad_norm"][k] for x in h]) for k in HEADS}
        lines.append(f"| {arm} | {rv:.3f} | {re:.3f} | {dv:.3f} | {sh['vertex']:.2f} / {sh['energy']:.2f} / {sh['direction']:.2f} | {g['vertex']:.3g} / {g['energy']:.3g} / {g['direction']:.3g} |")
cmp = {}
for f in sorted(here.glob("inputs_*.json")):
    cmp[f.stem] = json.loads(f.read_text())
if cmp:
    out["input_comparisons"] = cmp
    lines += ["", "## Input comparisons", "", "| Comparison | x identical | x columns identical | y identical | y columns identical |", "|---|---|---|---|---|"]
    for k, c in cmp.items():
        lines.append(f"| {k} | {c['x']['identical']} | {c['x'].get('columns_identical', '—')} | {c['y']['identical']} | {c['y'].get('columns_identical', '—')} |")
md = "\n".join(lines) + "\n"
(here / "summary.md").write_text(md); (here / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
print(md)
