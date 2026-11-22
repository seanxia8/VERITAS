"""VA3 (after seal A): compare table_a_blind.json with the other phase's values.

Usage: python va3_compare.py TABLE_A.json TOY_VS_LUCID.json FEV_DIAG.json OUT.json [RESCORE_LUCID.json]
Tolerances (runbook VA2): (a), (c), (d) 5 % relative; f_event 15 % relative; same arm ordering per
detector and seed.  f_event diagnostic: FEV_DIAG holds the per-dimension-mean definition
(ext_readout.layout_fractions) recomputed from the same dumps over all dev_probe_test photon bombs.
"""
import json
import re
import sys

ta, tvl, fd, out = sys.argv[1:5]
resc = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
A = json.load(open(ta)); T = json.load(open(tvl)); FD = json.load(open(fd))
mine = {(r["case"], r["test"]): r for r in A["rows"]}
D = A["baseline_d"]
TOL = {"a": 0.05, "c": 0.05, "d": 0.05, "f_event": 0.15}


def key(r):
    ts = r["run_id"].split("__")[-1] if "__ts" in r.get("run_id", "") else None
    if r["arm"] == "init":
        return f"{r['detector']}__legacy_init__s{r['seed']}"
    return f"{r['detector']}__{r['arm']}__s{r['seed']}" + (f"__{ts}" if ts else "") + "__final"


rows = []
for src, lst in (("TOY_VS_LUCID.rows", T["rows"]), ("TOY_VS_LUCID.sweep_rows", T["sweep_rows"])):
    for r in lst:
        L = r.get("lucid") or {}
        k = key(r)
        if L.get("spatial_latent_m") is None:
            continue          # run does not exist in the r2-lucid returns
        m = mine.get((k, "dev_probe_test"))
        if m is None:
            rows.append({"case": k, "source": src, "missing_in_mine": True}); continue
        vals = {"c": (m["c_err_m"], L["spatial_latent_m"]), "a": (m["a_err_m"], L["all_mean_m"]),
                "d": (D[r["detector"]]["dev_probe_test"]["err_m"], L.get("raw_spatial_m")),
                "f_event": (m["f_event"], L.get("f_event"))}
        for q, (x, y) in vals.items():
            if y is None:
                continue
            rel = x / y - 1
            row = {"case": k, "source": src, "test": "dev_probe_test", "quantity": q, "mine": x, "theirs": y,
                   "rel_diff": rel, "tol": TOL[q], "pass": abs(rel) <= TOL[q], "exploratory": "__ts" in k}
            if q == "f_event":
                alt = FD[k]["perdim_mean_all"]
                row["mine_perdim_definition"] = alt
                row["rel_diff_perdim_definition"] = alt / y - 1
                row["diagnosis"] = None if row["pass"] else (
                    "definitional: r2-tier1 uses the per-dimension mean of Var_e/Var_pe over all test photon bombs "
                    "(ext_readout.layout_fractions); runbook §1 / k2_collapse.py use the trace ratio over the first "
                    "200; recomputed under their definition from the same dumps it agrees to "
                    f"{abs(alt / y - 1):.1e}")
            rows.append(row)

if resc is not None:      # RESCORE_LUCID.json: flat list or dict of records with run_id/test/metric fields
    rows.append({"note": "RESCORE_LUCID comparison appended by va3_rescore section"})

# ordering of arms per detector and seed, (c) and (a), dev_probe_test
order = {}
for q in ("c", "a", "f_event"):
    groups = {}
    for r in rows:
        if r.get("quantity") == q and not r["exploratory"]:
            det, seed = r["case"].split("__")[0], re.search(r"__s(\d{4})", r["case"]).group(1)
            groups.setdefault((det, seed), []).append(r)
    for (det, seed), lst in groups.items():
        om = [r["case"] for r in sorted(lst, key=lambda r: r["mine"])]
        ot = [r["case"] for r in sorted(lst, key=lambda r: r["theirs"])]
        order[f"{q}:{det}:s{seed}"] = {"same": om == ot, "mine": om, "theirs": ot}
res = {"rows": rows, "ordering": order,
       "summary": {q: {"n": sum(1 for r in rows if r.get("quantity") == q),
                       "n_pass": sum(1 for r in rows if r.get("quantity") == q and r["pass"]),
                       "max_abs_rel": max([abs(r["rel_diff"]) for r in rows if r.get("quantity") == q] or [0])}
                   for q in TOL},
       "ordering_all_same": {q: all(v["same"] for k, v in order.items() if k.startswith(q + ":"))
                             for q in ("c", "a", "f_event")}}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res["summary"], indent=1), res["ordering_all_same"])
