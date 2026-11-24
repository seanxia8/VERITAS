"""VB2 (after seal B): compare table_b_blind.json with GATE_TABLE.json (+ BUDGET_AXIS.json cells).

Usage: python vb_compare.py TABLE_B.json GATE_TABLE.json BUDGET_AXIS.json TABLE_A2.json OUT.json
Tolerances (runbook VB2.3): cell error 5 % relative; contrast T-bar within 2 percentage points; same
clear / not-clear; same outcome from K1, K2, F1, F5.
"""
import json
import sys

tb, gt, ba, ta, out = sys.argv[1:6]
B = json.load(open(tb)); G = json.load(open(gt)); BA = json.load(open(ba)); A = json.load(open(ta))
SEED = {"A": "9000", "B": "9001", "C": "9002"}
mine_c = {}
for d in B["contrasts"].values():
    mine_c.update(d["contrasts"])
rows = []
for cid, g in G["registered"].items():
    m = mine_c.get(cid)
    if m is None:
        rows.append({"contrast": cid, "missing_in_mine": True}); continue
    dpp = 100 * (m["T_bar"] - g["T_bar"])
    seeds = {s: 100 * (m["T_per_seed"][SEED[s]] - v) for s, v in g["per_seed"].items()}
    rows.append({"contrast": cid, "T_bar_mine": m["T_bar"], "T_bar_theirs": g["T_bar"], "diff_pp": dpp,
                 "per_seed_diff_pp": seeds, "SE_boot": [m["SE_boot"], g["SE_boot"]], "SE_seed": [m["SE_seed"], g["SE_seed"]],
                 "N_sigma": [m["N_sigma"], g["N_sigma"]], "clear": [m["clear"], g["clear"]],
                 "pass_T": abs(dpp) <= 2.0, "pass_clear": m["clear"] == g["clear"]})
# cells: BUDGET_AXIS sealed cells, P2 / P4 seed-mean curves
cell_e = {(r["run_id"], r["step"]): r["c_err_m"] for r in B["cells"]}
a_rows = {r["run_id"]: r["c_err_m"] for r in A["rows"] if r["test"] == "sealed_test" and r["kind"] == "final"}
cells = []
for lab, v in BA["sealed_5000_10000"].items():
    rid = BA["validation"][lab]["run_id"]
    for st, e in v.items():
        mine = cell_e.get((rid, int(st)))
        src = "table_b"
        if mine is None and int(st) == 10000:
            mine, src = a_rows.get(rid), "table_a"
        if mine is None:
            continue
        cells.append({"cell": f"{rid}@{st}", "mine": mine, "theirs": e, "rel": mine / e - 1, "pass": abs(mine / e - 1) <= 0.05,
                      "mine_source": src})


def seedmean(det, mask_runs):
    vals = [cell_e[(r, 10000)] for r in mask_runs if (r, 10000) in cell_e]
    return sum(vals) / len(vals) if vals else None


U = {"450": "__ts450", "150": "__ts150", "50": "", "15": "__ts15", "5": "__ts5", "1.5": "__ts1p5"}
curves = []
for det, curve in (("sk", G["verdicts"]["P2"]["curve_seed_mean_m"]), ("wcte", G["verdicts"]["P4"]["per_detector"]["wcte"]["curve"])):
    seeds = ("9000", "9001") if det == "sk" else ("9000", "9001", "9002")
    for u, e in curve.items():
        runs = [f"{det}__mae_repaired__s{s}{U[u]}" for s in seeds]
        m = seedmean(det, runs)
        if m is not None:
            curves.append({"cell": f"{det} patch {u} ns seed mean", "mine": m, "theirs": e, "rel": m / e - 1,
                           "pass": abs(m / e - 1) <= 0.05})
mc = B["criteria"]; gv = G["verdicts"]
crit = {"K1": [mc["K1"], gv["K1"]], "K2": [mc["K2"], gv["K2"]], "F1": [mc["F1"], gv["F1"]],
        "F5": [mc["F5"], gv["F5"]["wording"]], "A_weak": [mc["A_weak"], gv["A_prime_weak_condition"]],
        "P2_holds": [mc["P2_holds"], gv["P2"]["holds"]], "P1": [all(mc["P1_signs"].values()), gv["P1"]["holds"]],
        "P3_electron_clear_positive": [mc["P3_confirmed"], bool(gv["P3"]["electron"]["clear"] and gv["P3"]["electron"]["T_bar"] > 0)],
        "outcome": [mc["outcome_from_K1_K2_F1_F5"], gv["outcome"]]}
p3 = B["P3"]["deposited_MeV"]
p3row = {"T_bar": [p3["T_bar"], gv["P3"]["electron"]["T_bar"]], "N_sigma": [p3["N_sigma"], gv["P3"]["electron"]["N_sigma"]],
         "per_seed_mine": p3["T_per_seed"], "per_seed_theirs": gv["P3"]["electron"]["per_seed"],
         "diff_pp": 100 * (p3["T_bar"] - gv["P3"]["electron"]["T_bar"]),
         "sign_same": (p3["T_bar"] > 0) == (gv["P3"]["electron"]["T_bar"] > 0),
         "clear_same": p3["clear"] == gv["P3"]["electron"]["clear"]}
f5 = {"mine": mc["F5_values"], "theirs": {"abs_T_5000": abs(gv["F5"]["CT_p5_5000"]), "abs_T_10000": abs(gv["F5"]["CT_p5_10000"])}}
res = {"contrasts": rows, "cells": cells, "curves": curves, "criteria": crit, "P3": p3row, "F5": f5,
       "SD_pooled": {d: [B["contrasts"][d]["SD_pooled"], G["SD_pooled"][d]] for d in G["SD_pooled"]},
       "summary": {"contrasts_T_within_2pp": sum(r.get("pass_T", False) for r in rows), "contrasts": len(rows),
                   "clear_same": sum(r.get("pass_clear", False) for r in rows),
                   "max_abs_diff_pp": max(abs(r["diff_pp"]) for r in rows if "diff_pp" in r),
                   "cells_within_5pct": sum(c["pass"] for c in cells + curves), "cells": len(cells + curves),
                   "max_abs_cell_rel": max(abs(c["rel"]) for c in cells + curves),
                   "criteria_same": all(a == b for a, b in crit.values())}}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res["summary"], indent=1)); print(json.dumps(crit)); print(json.dumps(p3row))
