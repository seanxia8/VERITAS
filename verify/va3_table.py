"""VA3: assemble table_a_blind.json (every independent part-A number) from the VA2 output.

Usage: python va3_table.py VA2.json K0.json RECONCILE.json OUT.json
"""
import json
import sys

va2, k0, rec, out = sys.argv[1:5]
r = json.load(open(va2))
rows = []
for name, v in sorted(r["cases"].items()):
    for t in v["c"]:
        rows.append({"case": name, "run_id": v["run_id"], "det": v["det"], "arm": v["arm"], "seed": v["seed"],
                     "kind": v["kind"], "test": t, "a_err_m": v["a"][t]["err_m"], "c_err_m": v["c"][t]["err_m"],
                     "shuffled_c_err_m": v["shuffled_c"][t]["err_m"],
                     "predict_mean_err_m": r["predict_mean"][v["det"]][t],
                     "a_over_c": v["a"][t]["err_m"] / v["c"][t]["err_m"],
                     "f_event": v["f_event"], "f_event_all_test": v["f_event_all_test"],
                     "exploratory": "__ts" in name})
table = {"phase": "r2-tier1v", "part": "A", "blind": True,
         "note": "computed before opening any r2-tier1 result; definitions runbook §1",
         "protocol": r["protocol"], "k0": json.load(open(k0)), "reconcile": json.load(open(rec)),
         "baseline_d": r["raw_d"], "predict_mean": r["predict_mean"], "rows": rows,
         "invariants": {k: v["pass"] for k, v in r["invariants"].items()},
         "invariant_detail": r["invariants"]}
json.dump(table, open(out, "w"), indent=1)
print(len(rows), "rows;", table["invariants"])
