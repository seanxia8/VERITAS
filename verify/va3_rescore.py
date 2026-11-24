"""VA3: compare table_a_blind.json with RESCORE_LUCID.json (opened after seal A).

Usage: python va3_rescore.py TABLE_A.json RESCORE_LUCID.json FEV_DIAG.json OUT.json [--exclude det:cohort,...]
Rows of excluded (detector, cohort) pairs are dropped unread (used to keep HK/JUNO sealed blind
until seal A2).  FEV_DIAG: per-dimension-mean f_event recomputed from my dumps (keys: case, or
case@sealed_test).
"""
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("table"); ap.add_argument("rescore"); ap.add_argument("fev"); ap.add_argument("out")
ap.add_argument("--exclude", default="")
a = ap.parse_args()
A = json.load(open(a.table)); FD = json.load(open(a.fev))
excl = {tuple(x.split(":")) for x in a.exclude.split(",") if x}
mine = {(r["case"], r["test"]): r for r in A["rows"]}
D = A["baseline_d"]
COH = {"validation": "dev_probe_test", "sealed": "sealed_test"}
TOL = {"c": 0.05, "a": 0.05, "d": 0.05, "f_event": 0.15}
rows, skipped = [], []
for r in json.load(open(a.rescore))["rows"]:
    if (r["detector"], r["cohort"]) in excl:
        skipped.append(f"{r['run_id']}@{r['step']}@{r['cohort']}"); continue
    t = COH[r["cohort"]]
    if r["arm"] == "init":
        case = f"{r['detector']}__legacy_init__s{r['seed']}"
    else:
        case = r["run_id"] + ("__final" if r["step"] == 10000 else "__matched_init")
    m = mine.get((case, t))
    if m is None:
        rows.append({"case": case, "test": t, "missing_in_mine": True}); continue
    dd = D.get(r["detector"], {}).get(t, {}).get("err_m")
    vals = {"c": (m["c_err_m"], r["ridge_spatial"]), "a": (m["a_err_m"], r["ridge_all_mean"]),
            "d": (dd, r["ridge_raw"]), "f_event": (m["f_event"], r["f_event"])}
    for q, (x, y) in vals.items():
        if x is None or y is None:
            continue
        rel = x / y - 1
        row = {"case": case, "test": t, "quantity": q, "mine": x, "theirs": y, "rel_diff": rel,
               "tol": TOL[q], "pass": abs(rel) <= TOL[q], "exploratory": False}
        if q == "f_event":
            k = case if t == "dev_probe_test" else f"{case}@{t}"
            if k in FD:
                alt = FD[k]["perdim_mean_all"]
                row["mine_perdim_definition"] = alt; row["rel_diff_perdim_definition"] = alt / y - 1
            if not row["pass"]:
                row["diagnosis"] = ("definitional (per-dimension mean over all test photon bombs vs trace ratio "
                                    "over the first 200), see va3_fevent_diagnostic.json")
        rows.append(row)
summ = {q: {"n": sum(1 for r in rows if r.get("quantity") == q),
            "n_pass": sum(1 for r in rows if r.get("quantity") == q and r["pass"]),
            "max_abs_rel": max([abs(r["rel_diff"]) for r in rows if r.get("quantity") == q] or [0])} for q in TOL}
n = sum(s["n"] for k, s in summ.items() if k != "f_event")
res = {"rows": rows, "summary": summ, "skipped_unread": skipped,
       "n": n, "n_pass": sum(s["n_pass"] for k, s in summ.items() if k != "f_event"),
       "max_abs_rel": max(s["max_abs_rel"] for k, s in summ.items() if k != "f_event"),
       "note": "(a), (c), (d) only; f_event separately (definitional)"}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(summ, indent=1), "skipped", len(skipped), "missing", [r["case"] for r in rows if r.get("missing_in_mine")])
