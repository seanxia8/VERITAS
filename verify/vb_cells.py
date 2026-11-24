"""VB1: VERIFY_INPUTS_B.json -> cases_b.json (va_dump format), cells_b.json (vb_eval), GPU/CPU queues.

Usage: python vb_cells.py VERIFY_INPUTS_B.json OUTDIR
One case per unique (run_id, step); the canonical checkpoint is the one outside a `.prev_*`
directory (the manifest gives byte-identical SHA256 for the `.prev_*` copies).  Mask from
recipe_variant.mask (default patch), time unit from time_loss_scale_ns (default 50 ns, as the
registered contrasts use the default run as the 50 ns cell).  charge-weight and time-reference
variants get their own mask label so they never enter a registered contrast.
"""
import json
import sys

src, outdir = sys.argv[1:3]
M = json.load(open(src))
by = {}
for c in M["cases"]:
    k = (c["run_id"], int(c["step"]))
    prev = ".prev_" in c["checkpoint"]
    if k not in by or (by[k][1] and not prev):
        by[k] = (c, prev)
cases, cells, copies = [], [], {}
for c in M["cases"]:
    copies.setdefault(f"{c['run_id']}__step{int(c['step']):05d}", []).append(c["checkpoint"])
for (rid, step), (c, _) in sorted(by.items()):
    rv = c.get("recipe_variant") or {}
    mask = rv.get("mask", "patch")
    if "charge_loss_weight" in rv:
        mask += f"_cw{rv['charge_loss_weight']:g}"
    if rv.get("time_reference", "trigger") != "trigger":
        mask += f"_tr{rv['time_reference']}"
    unit = float(rv.get("time_loss_scale_ns", 50.0))
    name = f"{rid}__step{step:05d}"
    cases.append({"det": c["detector"], "arm": c["arm"], "seed": int(c["seed"]), "run_id": rid,
                  "window": c["encoder"]["window"], "shift": c["encoder"]["shift"], "encoder": c["encoder"],
                  "scaling": c["scaling"], "readout": "raw", "case": name, "kind": f"step{step}",
                  "ckpt": c["checkpoint"], "sha256": c["checkpoint_sha256"], "ckpt_step": step,
                  "time_reference": c.get("time_reference_for_encoder_input", "trigger")})
    cells.append({"case": name, "det": c["detector"], "mask": mask, "unit": unit, "seed": int(c["seed"]),
                  "step": step, "run_id": rid})
json.dump({"cases": cases}, open(f"{outdir}/cases_b.json", "w"), indent=1)
json.dump({"cells": cells, "copies": copies, "registered": M["registered_contrasts"]}, open(f"{outdir}/cells_b.json", "w"), indent=1)


def prio(x):
    sk_f5 = x["det"] == "sk" and x["step"] == 5000 and x["mask"] == "patch" and x["unit"] in (450.0, 5.0)
    return (0 if x["step"] == 10000 else 1 if sk_f5 else 2 if x["step"] == 5000 else 3,
            0 if x["det"] == "sk" else 1, x["case"])


trm = [x["case"] for x in cells if "_tr" in x["mask"]]
order = [x["case"] for x in sorted(cells, key=prio) if x["case"] not in trm]
cpu = [x["case"] for x in sorted(cells, key=prio, reverse=True)
       if x["det"] == "wcte" and x["step"] != 10000 and x["case"] not in trm]
open(f"{outdir}/queue_gpu.txt", "w").write("\n".join(order) + "\n")
open(f"{outdir}/queue_cpu.txt", "w").write("\n".join(cpu) + "\n")
p3 = [x["case"] for x in cells if x["det"] == "sk" and x["step"] == 10000 and x["mask"] == "patch"
      and x["unit"] in (450.0, 5.0)]
open(f"{outdir}/queue_p3.txt", "w").write("\n".join(sorted(p3)) + "\n")
print(len(cases), "cases;", len(order), "gpu queue;", len(cpu), "cpu queue;", "p3", p3, "; skipped", trm)

# job lists "case group domain"; P3 electron jobs right after the step-10000 block
fin = [c for c in order if c.endswith("__step10000")]
rest = [c for c in order if not c.endswith("__step10000")]
J = lambda cs, dom="photon": [f"{c} {g} {dom}" for c in cs for g in ("dev_probe_train", "sealed_test")]
open(f"{outdir}/jobs_gpu.txt", "w").write("\n".join(J(fin) + J(sorted(p3), "electron") + J(rest)) + "\n")
open(f"{outdir}/jobs_cpu.txt", "w").write("\n".join(J(cpu)) + "\n")
