"""VA1: discover the existing LUCiD checkpoints (runbook §2 VA1.1) and build the part-A case table.

Reads only checkpoint.pt / snapshot_step_00000.pt (hash + step) and the identity, encoder and
scaling keys of recipe.json under train/runs/.  Smokes and dry runs are excluded.
Cases: every run (final checkpoint), its matched init (snapshot_step_00000.pt), and the legacy
init (default EncoderConfig, torch.manual_seed(seed)) at seeds 9000 and 9001 per detector.

Usage: python va1_cases.py OUT.json
"""
import glob
import json
import os
import sys

import torch

from lucid_common import DETS, sha256

ROOT = "/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/train/runs"


def step_of(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("step", "global_step"):
        if k in blob:
            return int(blob[k])
    return None


cases = []
for d in sorted(glob.glob(f"{ROOT}/*/recipe.json")):
    run_dir = os.path.dirname(d)
    rec = json.load(open(d))["recipe"]
    ident, enc, sc = rec["identity"], rec["encoder"], rec["scaling"]
    det = ident["run_id"].split("__")[0]
    base = {"det": det, "arm": ident["arm"], "seed": int(ident["seed"]), "run_id": ident["run_id"],
            "cell_id": ident.get("cell_id"), "steps_scheduled": ident.get("steps_scheduled"),
            "window": int(enc["window"]), "shift": int(enc["shift"]), "encoder": enc, "scaling": sc,
            "readout": "layernorm" if ident["arm"] == "jepa_repaired" else
                       ("posfree" if ident["arm"] == "sigreg_jepa_all_posfree" else "raw")}
    for kind, fn in (("final", "checkpoint.pt"), ("matched_init", "snapshot_step_00000.pt")):
        p = f"{run_dir}/{fn}"
        cases.append({**base, "case": f"{ident['run_id']}__{kind}", "kind": kind, "ckpt": p,
                      "sha256": sha256(p), "ckpt_step": step_of(p)})
        print(cases[-1]["case"], cases[-1]["window"], cases[-1]["ckpt_step"], cases[-1]["sha256"][:12], flush=True)
for det in DETS:
    for s in (9000, 9001):
        cases.append({"det": det, "arm": "legacy_init", "seed": s, "run_id": None, "window": 64, "shift": 32,
                      "readout": "raw", "case": f"{det}__legacy_init__s{s}", "kind": "legacy_init",
                      "ckpt": None, "sha256": None, "ckpt_step": 0})
json.dump({"root": ROOT, "cases": cases}, open(sys.argv[1], "w"), indent=1)
