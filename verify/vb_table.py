"""VB2: assemble table_b_blind.json from the vb_eval output.

Usage: python vb_table.py VB.json CELLS.json MANIFEST_B.json OUT.json
"""
import hashlib
import json
import sys

vb, cells, man, out = sys.argv[1:5]
V = json.load(open(vb)); C = json.load(open(cells)); raw = open(man, "rb").read()
listed = {c["case"] for c in C["cells"]}
done = {r["case"] for r in V["cells"]}
T = {"phase": "r2-tier1v", "part": "B", "blind": True,
     "note": "computed before opening GATE_TABLE.json or any other r2-tier1 part-B result; "
             "statistics PREREG_2026-11-22_R2_TIER1 §2.2-§2.5 (verify/vb_stats.py)",
     "manifest": {"path": "results/round2/20261122_tier1/verify_inputs/VERIFY_INPUTS_B.json (origin/tier1)",
                  "sha256": hashlib.sha256(raw).hexdigest(), "n_entries": len(json.loads(raw)["cases"]),
                  "n_unique_run_step": len(listed)},
     "coverage": {"evaluated": len(done), "listed": len(listed), "not_evaluated": sorted(listed - done)},
     "readout": "Ridge on (c) spatial-moment latent and (a) all-mean, fitted on dev_probe_train photon bombs, "
                "tested on sealed_test photon bombs; e = mean 3-D error (m)",
     "cells": V["cells"], "contrasts": V["contrasts"], "P3": V["P3"], "criteria": V["criteria"],
     "mapping_check": V.get("mapping_check"), "step0_alias": V.get("step0_alias"),
     "not_recomputed": ["F4", "K3", "K4(b)", "P4", "P5", "P6", "secondary outcomes other than P3 sign"]}
json.dump(T, open(out, "w"), indent=1)
print("cells", len(done), "/", len(listed), "outcome", V["criteria"].get("outcome_from_K1_K2_F1_F5"))
