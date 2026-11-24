"""VB1: part-B quantities from the dumps (runbook VB1, PREREG_2026-11-22_R2_TIER1 §2).

Usage: python vb_eval.py CELLS.json DUMPDIR OUT.json
CELLS.json: {"cells": [{"case", "det", "mask", "unit", "seed", "step"}, ...]}  (built by vb_cells.py
from VERIFY_INPUTS_B.json).  Dumps: DUMPDIR/<case>__<group>__<domain>.npz from va_dump.py.

Per cell: readouts (c) and (a) (Ridge fitted on dev_probe_train photon bombs) on sealed_test photon
bombs, per-event 3-D errors kept for the paired bootstrap.  Contrasts, SE, N_sigma, clear: vb_stats.
P3 sign check: Ridge on ln E (contained electrons, E = deposited_MeV; energy_MeV secondary) from the
(c) features, 68 % half-width of (E_hat - E)/E, contrast 450 -> 5 at SK patch.
"""
import json
import os
import sys

import numpy as np

import vb_stats as S
from readout_spec import ridge_fit_predict

CELLS, DUMP, OUT = sys.argv[1:4]
CJ = json.load(open(CELLS))
cells = CJ["cells"]
REGM = CJ.get("registered")
TRAIN, TEST = "dev_probe_train", "sealed_test"


EQ0 = {}   # step-0 checkpoints with bitwise-identical encoder weights (step0_equality.json beside CELLS)
_eq = os.path.join(os.path.dirname(os.path.abspath(CELLS)), "step0_equality.json")
if os.path.exists(_eq):
    for k, v in json.load(open(_eq)).items():
        if v["equal"]:
            EQ0.setdefault(v["ref"], []).append(k)
ALIAS = {}


def ld(case, group, dom="photon"):
    p = f"{DUMP}/{case}__{group}__{dom}.npz"
    if os.path.exists(p):
        return np.load(p)
    for grp in EQ0.values():          # same encoder weights -> same features on the same events
        if case in grp:
            for alt in grp:
                q = f"{DUMP}/{alt}__{group}__{dom}.npz"
                if os.path.exists(q):
                    ALIAS[f"{case}__{group}"] = alt
                    return np.load(q)
    return None


per = {}            # key -> {"c": per-event err, "a": per-event err}
per_case = {}
ref = {}
res = {"cells": [], "missing": []}
for c in cells:
    if "_tr" in c["mask"]:      # non-trigger time reference: encoder input differs, not recomputed blind
        res["missing"].append(c["case"]); continue
    tr, te = ld(c["case"], TRAIN), ld(c["case"], TEST)
    if tr is None or te is None:
        res["missing"].append(c["case"]); continue
    if c["det"] in ref:
        assert np.array_equal(ref[c["det"]], te["event_ids"]), c["case"]
    else:
        ref[c["det"]] = te["event_ids"]
    row = dict(c)
    errs = {}
    for form in ("c", "a"):
        (p,), al = ridge_fit_predict(tr[form].astype(np.float64), tr["vertex_m"], [te[form].astype(np.float64)],
                                     tr["run"])
        errs[form] = np.linalg.norm(p - te["vertex_m"], axis=1)
        row[f"{form}_err_m"] = float(errs[form].mean()); row[f"{form}_alpha"] = al
    row["n_test"] = int(len(errs["c"]))
    per[(c["det"], c["mask"], float(c["unit"]), int(c["seed"]), int(c["step"]))] = errs
    per_case[c["case"]] = errs
    res["cells"].append(row)
    print(c["case"], round(row["c_err_m"], 5), round(row["a_err_m"], 5), flush=True)


def E(det, mask, unit, step=10000):
    out = {s: v["c"] for (d, m, u, s, st), v in per.items()
           if d == det and m == mask and abs(u - unit) < 1e-9 and st == step}
    return out


def need(*xs):
    return all(len(x) for x in xs)


REG = {"sk": [("CT-p50", ("patch", 450), ("patch", 50)), ("CT-p5", ("patch", 450), ("patch", 5)),
              ("CT-r50", ("random", 450), ("random", 50)),
              ("CM-450", ("random", 450), ("patch", 450)), ("CM-50", ("random", 50), ("patch", 50))],
       "wcte": [(f"WT-{m[0]}-{u:g}", (m, 450), (m, u)) for m in ("patch", "random") for u in (50, 5, 1.5)]
       + [(f"WM-{u:g}", ("random", u), ("patch", u)) for u in (450, 5)]}
con = {}
for det, lst in REG.items():
    n = len(ref.get(det, []))
    if not n:
        continue
    idx = S.boot_index(n)
    cd, registered = {}, []
    for cid, (m1, u1), (m2, u2) in lst:
        e1, e2 = E(det, m1, u1), E(det, m2, u2)
        if not need(e1, e2):
            res.setdefault("contrasts_missing", []).append(cid); continue
        own = S.contrast(e1, e2, idx)
        if REGM is None:
            cd[cid] = own; cd[cid]["registered"] = True; registered.append(cid)
        else:
            res.setdefault("mapping_check", {})[cid] = own["T_per_seed"]
    if REGM is not None:      # registered contrasts exactly as listed by run id in VERIFY_INPUTS_B
        SEED = {"A": 9000, "B": 9001, "C": 9002}
        for cid, v in REGM.items():
            if v["detector"] != det:
                continue
            ps = v["per_seed"]
            if not all(f"{p['run_1']}__step10000" in per_case and f"{p['run_2']}__step10000" in per_case for p in ps):
                res.setdefault("contrasts_missing", []).append(cid); continue
            e1 = {SEED[p["seed"]]: per_case[f"{p['run_1']}__step10000"]["c"] for p in ps}
            e2 = {SEED[p["seed"]]: per_case[f"{p['run_2']}__step10000"]["c"] for p in ps}
            cd[cid] = S.contrast(e1, e2, idx); cd[cid]["registered"] = True; registered.append(cid)
            cd[cid]["runs"] = ps
    if det == "sk":     # P2 chain and F5 (5,000-step snapshots); not registered, SE_seed from SK pool
        chain = [450, 150, 50, 15, 5]
        for u1, u2 in zip(chain, chain[1:]):
            e1, e2 = E(det, "patch", u1), E(det, "patch", u2)
            if need(e1, e2):
                cd[f"P2-{u1:g}-{u2:g}"] = S.contrast(e1, e2, idx); cd[f"P2-{u1:g}-{u2:g}"]["registered"] = False
        e1, e2 = E(det, "patch", 450, 5000), E(det, "patch", 5, 5000)
        if need(e1, e2):
            cd["CT-p5@5000"] = S.contrast(e1, e2, idx); cd["CT-p5@5000"]["registered"] = False
    sdp = S.finish(cd, registered)
    con[det] = {"SD_pooled": sdp, "contrasts": cd}
res["contrasts"] = con

# ------------------------------------------------------------------ P3 electron energy (SK patch)
p3 = {}
for target in ("deposited_MeV", "energy_MeV"):
    rel = {}
    for u in (450, 5):
        rel[u] = {}
        for (d, m, uu, s, st) in per:
            if d != "sk" or m != "patch" or abs(uu - u) > 1e-9 or st != 10000:
                continue
            case = next(c["case"] for c in cells if c["det"] == d and c["mask"] == m and abs(float(c["unit"]) - u) < 1e-9
                        and int(c["seed"]) == s and int(c["step"]) == st)
            tr, te = ld(case, TRAIN, "electron"), ld(case, TEST, "electron")
            if tr is None or te is None:
                continue
            ktr = tr["contained"].astype(bool) & (tr["deposited_MeV"] > 0)
            kte = te["contained"].astype(bool) & (te["deposited_MeV"] > 0)
            (p,), al = ridge_fit_predict(tr["c"][ktr].astype(np.float64), np.log(tr[target][ktr]),
                                         [te["c"][kte].astype(np.float64)], tr["run"][ktr])
            Et = te[target][kte]
            rel[u][s] = (np.exp(p.ravel()) - Et) / Et
            p3.setdefault("n", {})[case] = {"train": int(ktr.sum()), "test": int(kte.sum())}
    if need(rel[450], rel[5]):
        n = len(next(iter(rel[450].values())))
        r = S.resolution_contrast(rel[450], rel[5], S.boot_index(n, seed=S.BOOT_SEED + 1))
        Tv = list(r["T_per_seed"].values())
        r["SE_seed"] = float(np.std(Tv, ddof=1) / np.sqrt(len(Tv))) if len(Tv) > 1 else float("nan")
        se = np.nanmax([r["SE_boot"], r["SE_seed"]])
        r["N_sigma"] = float(abs(r["T_bar"]) / se)
        r["same_sign_all_seeds"] = len({np.sign(v) for v in Tv}) == 1
        r["clear"] = bool(r["N_sigma"] >= 2 and r["same_sign_all_seeds"])
        p3[target] = r
res["P3"] = p3

# ------------------------------------------------------------------ K1, K2, F1, F5, outcome
sk = con.get("sk", {}).get("contrasts", {})


def g(k):
    return sk.get(k)


crit = {}
if g("CT-p50") and g("CT-p5"):
    small = lambda c: abs(c["T_bar"]) < 0.05 or not c["clear"]
    crit["K1"] = bool(small(g("CT-p50")) and small(g("CT-p5")))
    chain = [k for k in sk if k.startswith("P2-")]
    reversal = [k for k in chain if sk[k]["T_bar"] >= 0.03 and sk[k]["clear"]]
    crit["P2_chain_complete"] = len(chain) == 4
    crit["P2_clear_reversals"] = reversal
    crit["P2_holds"] = not reversal
    crit["K2"] = bool(reversal or -g("CT-p5")["T_bar"] < 0.05)
    crit["F1"] = bool(g("CT-p5")["T_bar"] <= -0.10 and g("CT-p5")["N_sigma"] >= 3 and crit["P2_holds"])
    p3c = p3.get("deposited_MeV")
    crit["P3_confirmed"] = None if p3c is None else bool(p3c["T_bar"] > 0 and p3c["clear"])
    ct = g("CT-p5")
    crit["A_weak"] = bool(-0.10 < ct["T_bar"] <= -0.05 and ct["clear"] and crit["P3_confirmed"])
    masks = [g(k) for k in ("CM-450", "CM-50") if g(k)]
    mask_alt = any(c["T_bar"] <= -0.10 and c["clear"] for c in masks)
    if crit["K1"] or crit["K2"]:
        outcome = "A-alt" if mask_alt else "P2"
    elif crit["F1"]:
        outcome = "A' strong"
    elif crit["A_weak"]:
        outcome = "A' weak"
    else:
        outcome = "P2"
    crit["outcome_from_K1_K2_F1_F5"] = outcome
    crit["outcome_caveat"] = "assumes K3 (probe robustness), F4 and K4(b) not fired; not recomputed here"
    if g("CT-p5@5000"):
        m5, m10 = abs(g("CT-p5@5000")["T_bar"]), abs(ct["T_bar"])
        crit["F5"] = ("the loss scale sets the order and rate of learning" if m10 <= m5 / 2 else
                      "at this budget the loss scale selects the solution" if m10 >= m5 else
                      "undetermined at 10,000 steps")
        crit["F5_values"] = {"abs_T_5000": m5, "abs_T_10000": m10}
    else:
        crit["F5"] = None
    crit["P1_signs"] = {k: c["T_bar"] < 0 for d in con for k, c in con[d]["contrasts"].items()
                        if k.startswith(("CT-p", "CT-r", "WT")) and "@" not in k}
res["criteria"] = crit
res["step0_alias"] = ALIAS


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


json.dump(clean(res), open(OUT, "w"), indent=1)
print(json.dumps(clean(crit), indent=1))
