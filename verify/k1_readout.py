"""K1: independent vertex readouts on the dumped pooled statistics, plus controls and the
pre-registered invariants I1-I6.

Ridge (own implementation): features standardised on probe-train; one alpha shared by the
three vertex coordinates, chosen by 5-fold run-grouped CV on probe-train (folds = runs dealt
round-robin after sorting by run id) over logspace(-6, 4, 21), minimising the summed squared
error; intercept unpenalised.

Usage: python k1_readout.py DUMPDIR OUT.json
"""
import json
import os
import sys

import numpy as np

from cases import cases
from common import Cohort, Geometry

ALPHAS = np.logspace(-6, 4, 21)
DUMP, OUT = sys.argv[1], sys.argv[2]


def ridge_path(X, Y, alphas):
    """Coefficients for all alphas via one SVD (X, Y already centred)."""
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y
    return [Vt.T @ ((s / (s * s + a))[:, None] * UtY) for a in alphas]


def fit_predict(Xtr, Ytr, Xte, groups, alphas=ALPHAS):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd < 1e-12] = 1.0
    Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
    ug = np.unique(groups)
    fold_of_group = {g: i % 5 for i, g in enumerate(ug)}
    fold = np.array([fold_of_group[g] for g in groups])
    sse = np.zeros(len(alphas))
    for f in range(5):
        tr, va = fold != f, fold == f
        xm, ym = Ztr[tr].mean(0), Ytr[tr].mean(0)
        for i, B in enumerate(ridge_path(Ztr[tr] - xm, Ytr[tr] - ym, alphas)):
            sse[i] += (((Ztr[va] - xm) @ B + ym - Ytr[va]) ** 2).sum()
    best = int(np.argmin(sse))
    xm, ym = Ztr.mean(0), Ytr.mean(0)
    B = ridge_path(Ztr - xm, Ytr - ym, [alphas[best]])[0]
    return (Zte - xm) @ B + ym, float(alphas[best])


def metrics(pred, true, r_fid, radial=False):
    err = np.linalg.norm(pred - true, axis=1)
    out = {"mean_err_m": float(err.mean()), "median_err_m": float(np.median(err)),
           "err_over_rfid": float(err.mean() / r_fid)}
    if radial:
        rt, rp = np.linalg.norm(true, axis=1), np.linalg.norm(pred, axis=1)
        q = np.quantile(rt, [0, .25, .5, .75, 1])
        bins = np.clip(np.searchsorted(q, rt, side="right") - 1, 0, 3)
        out["radial_bias_by_true_r_quartile_m"] = [float((rp - rt)[bins == b].mean()) for b in range(4)]
        out["true_r_quartile_edges_m"] = q.tolist()
    return out


def raw_moments(det, split, geo, stats=None):
    """(d): hit-masked spatial moments of the 9 standardised raw features (no encoder)."""
    coh = Cohort(det, split)
    up = geo.unit_pos
    feats = []
    for k in range(coh.n):
        p, q, t = coh.hits(k)
        f = np.column_stack([up[p], geo.dirs[p], np.log(np.maximum(q, 1e-6)), t / geo.time_scale_ns,
                             np.ones(len(p))])
        feats.append((p, f))
    if stats is None:  # per-column mean / sd over probe-train hit tokens
        allf = np.concatenate([f for _, f in feats])
        mu, sd = allf.mean(0), allf.std(0)
        sd[sd < 1e-12] = 1.0
        stats = (mu, sd)
    mu, sd = stats
    X = np.zeros((coh.n, 9 * 4))
    for k, (p, f) in enumerate(feats):
        fs = (f - mu) / sd
        X[k, :9] = fs.mean(0)
        for c in range(3):
            X[k, 9 * (c + 1):9 * (c + 2)] = (fs * up[p, c][:, None]).mean(0)
    return X, coh, stats


res = {"protocol": {"alphas": [float(a) for a in ALPHAS], "cv": "5-fold run-grouped on probe-train",
                    "targets": "vertex_m (x,y,z)", "shuffle_seed": 12345}, "cases": {}, "raw": {}, "mean_predictor": {}}
geos = {}
for c in cases():
    det = c["det"]
    if det not in geos:
        geo = geos[det] = Geometry(det)
        Xtr_d, ctr, st = raw_moments(det, "train", geo)
        Xte_d, cte, _ = raw_moments(det, "test", geo, st)
        pred, a = fit_predict(Xtr_d, ctr.vertex, Xte_d, ctr.run)
        res["raw"][det] = {**metrics(pred, cte.vertex, geo.r_fid), "alpha": a, "r_fid_m": geo.r_fid}
        mp = np.repeat(ctr.vertex.mean(0, keepdims=True), cte.n, 0)
        res["mean_predictor"][det] = metrics(mp, cte.vertex, geo.r_fid)
        print(det, "raw", res["raw"][det]["mean_err_m"], "mean", res["mean_predictor"][det]["mean_err_m"], flush=True)
    geo = geos[det]
    if not all(os.path.exists(f"{DUMP}/{c['case']}__{s}.npz") for s in ("train", "test")):
        continue
    f = {s: np.load(f"{DUMP}/{c['case']}__{s}.npz") for s in ("train", "test")}
    ytr, yte, g = f["train"]["vertex"], f["test"]["vertex"], f["train"]["run"]
    forms = {"a_allpmt_mean": lambda d: d["mean_all"].astype(np.float64),
             "b_hit_mean": lambda d: d["mean_hit"].astype(np.float64),
             "c_hit_moments": lambda d: np.concatenate([d["mean_hit"], d["mom_hit"].reshape(len(d["mean_hit"]), -1)], 1).astype(np.float64)}
    out = {"det": det, "arm": c["arm"], "seed": c["seed"], "n_train": int(len(ytr)), "n_test": int(len(yte))}
    for name, fn in forms.items():
        pred, a = fit_predict(fn(f["train"]), ytr, fn(f["test"]), g)
        out[name] = {**metrics(pred, yte, geo.r_fid, radial=name != "b_hit_mean"), "alpha": a}
    rng = np.random.default_rng(12345)
    pred, a = fit_predict(forms["c_hit_moments"](f["train"]), ytr[rng.permutation(len(ytr))],
                          forms["c_hit_moments"](f["test"]), g)
    out["shuffled_label_c"] = {**metrics(pred, yte, geo.r_fid), "alpha": a}
    res["cases"][c["case"]] = out
    print(c["case"], {k: round(out[k]["mean_err_m"], 4) for k in list(forms) + ["shuffled_label_c"]}, flush=True)

# ------------------------------------------------------------------ invariants I1-I6
C = res["cases"]
E = lambda d, arm, s, form="c_hit_moments": C.get(f"{d}__{arm}__s{s}", {}).get(form, {}).get("mean_err_m")
inv = {}
r = [(k, v["a_allpmt_mean"]["mean_err_m"] / v["c_hit_moments"]["mean_err_m"]) for k, v in C.items() if v["det"] in ("sk", "juno", "hk")]
inv["I1"] = {"pass": bool(r) and all(x >= 3 for _, x in r), "ratio_a_over_c": dict(r), "min": min(x for _, x in r) if r else None}
r = {f"{d}_s{s}": (E(d, "mae_mse", s), E(d, "init", s)) for d in geos for s in (9000, 9001)}
inv["I2"] = {"pass": all(a is not None and b is not None and a < b for a, b in r.values()), "mae_vs_init_c": r}
r = {d: (E(d, "sigreg_jepa", 9000), E(d, "init", 9000)) for d in geos}
inv["I3"] = {"pass": all(a is not None and b is not None and a > b for a, b in r.values()), "sigreg_vs_init_c": r}
if "sk" in geos and "hk" in geos:
    target = geos["hk"].r_fid / geos["sk"].r_fid
    r = {}
    for k, v in C.items():
        if v["det"] == "hk":
            sk = C.get(k.replace("hk__", "sk__", 1))
            if sk:
                r[k[4:]] = v["a_allpmt_mean"]["mean_err_m"] / sk["a_allpmt_mean"]["mean_err_m"]
    inv["I4"] = {"pass": bool(r) and all(abs(x / target - 1) <= 0.2 for x in r.values()),
                 "rfid_ratio_hk_sk": target, "err_a_ratio_hk_sk": r}
r = {k: v["shuffled_label_c"]["mean_err_m"] / res["mean_predictor"][v["det"]]["mean_err_m"] for k, v in C.items()}
inv["I5"] = {"pass": bool(r) and all(abs(x - 1) <= 0.10 for x in r.values()), "shuffled_over_mean": r}
r = {f"{d}_s{s}": (E(d, "init", s), res["raw"][d]["mean_err_m"]) for d in ("sk", "juno", "hk") if d in geos for s in (9000, 9001)}
inv["I6"] = {"pass": all(a is not None and a < b for a, b in r.values()), "init_c_vs_raw_d": r}
res["init_seed_spread"] = {d: {form: abs((E(d, "init", 9000, form) or np.nan) - (E(d, "init", 9001, form) or np.nan))
                               for form in ("a_allpmt_mean", "c_hit_moments")} for d in geos}
res["invariants"] = inv
for k, v in inv.items():
    print(k, "PASS" if v["pass"] else "FAIL")
json.dump(res, open(OUT, "w"), indent=1)
