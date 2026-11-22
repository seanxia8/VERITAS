"""VA2: part-A quantities and invariants J1-J9 (runbook §2 VA2) from the dumps.

Per case: readouts (a), (c) (Ridge fitted on dev_probe_train photon bombs), tested on
dev_probe_test and on sealed_test; f_event; controls (shuffled-label Ridge on (c), seed 12345;
predict-the-mean).  Baseline (d) once per detector.  Invariants on every available test set;
an invariant passes only if it holds on every test set evaluated.

Usage: python va2_readouts.py CASES.json DUMPDIR RAWDIR OUT.json [SEALED_GROUP_LABEL]
Dump files: DUMPDIR/<case>__<group>__photon.npz ; raw: RAWDIR/<det>__<group>__photon.npz
"""
import json
import os
import sys

import numpy as np

from readout_spec import ridge_fit_predict, vertex_error

CASES, DUMP, RAW, OUT = sys.argv[1:5]
TESTS = ["dev_probe_test"] + (sys.argv[5].split(",") if len(sys.argv) > 5 else [])
TRAIN = "dev_probe_train"
N_FEV = 200


def load(path):
    return np.load(path) if os.path.exists(path) else None


def f_event(a, msq):
    total = float((msq.mean(0) - a.mean(0) ** 2).sum())
    return float(a.var(0).sum() / total)


cases = json.load(open(CASES))["cases"]
res = {"protocol": {"ridge": "per-coordinate alpha in {1e-6,1e-4,1e-2,1,100}, 5-fold run-grouped CV, "
                             "standardised on dev_probe_train", "train": TRAIN, "tests": TESTS,
                    "shuffle_seed": 12345, "f_event_events": f"first {N_FEV} photon bombs of dev_probe_test"},
       "cases": {}, "raw_d": {}, "predict_mean": {}}
ref_ids = {}
for c in cases:
    det, name = c["det"], c["case"]
    tr = load(f"{DUMP}/{name}__{TRAIN}__photon.npz")
    te = {t: load(f"{DUMP}/{name}__{t}__photon.npz") for t in TESTS}
    te = {t: v for t, v in te.items() if v is not None}
    if tr is None or not te:
        print("skip", name, flush=True)
        continue
    for t, v in [(TRAIN, tr)] + list(te.items()):  # event alignment across cases
        key = (det, t)
        if key not in ref_ids:
            ref_ids[key] = v["event_ids"]
        assert np.array_equal(ref_ids[key], v["event_ids"]), (name, t)
    if det not in res["raw_d"]:
        rtr = load(f"{RAW}/{det}__{TRAIN}__photon.npz")
        rte = {t: load(f"{RAW}/{det}__{t}__photon.npz") for t in te}
        assert np.array_equal(rtr["event_ids"], tr["event_ids"])
        tl = [t for t in te if rte[t] is not None]
        preds, al = ridge_fit_predict(rtr["d"], rtr["vertex_m"], [rte[t]["d"] for t in tl], rtr["run"])
        res["raw_d"][det] = {t: {"err_m": vertex_error(p, rte[t]["vertex_m"]), "alpha": al} for t, p in zip(tl, preds)}
        mu = tr["vertex_m"].mean(0)
        res["predict_mean"][det] = {t: vertex_error(np.repeat(mu[None], len(v["vertex_m"]), 0), v["vertex_m"])
                                    for t, v in te.items()}
    y, g = tr["vertex_m"], tr["run"]
    tl = list(te)
    out = {k: c[k] for k in ("det", "arm", "seed", "kind", "run_id", "window", "readout", "sha256")}
    for form in ("a", "c"):
        preds, al = ridge_fit_predict(tr[form].astype(np.float64), y, [te[t][form].astype(np.float64) for t in tl], g)
        out[form] = {t: {"err_m": vertex_error(p, te[t]["vertex_m"]), "alpha": al} for t, p in zip(tl, preds)}
    perm = np.random.default_rng(12345).permutation(len(y))
    preds, al = ridge_fit_predict(tr["c"].astype(np.float64), y[perm], [te[t]["c"].astype(np.float64) for t in tl], g)
    out["shuffled_c"] = {t: {"err_m": vertex_error(p, te[t]["vertex_m"]), "alpha": al} for t, p in zip(tl, preds)}
    d0 = te["dev_probe_test"]
    out["f_event"] = f_event(d0["a"][:N_FEV].astype(np.float64), d0["msq"][:N_FEV].astype(np.float64))
    out["f_event_all_test"] = f_event(d0["a"].astype(np.float64), d0["msq"].astype(np.float64))
    out["n"] = {TRAIN: int(len(y)), **{t: int(len(te[t]["vertex_m"])) for t in tl}}
    res["cases"][name] = out
    print(name, {t: (round(out["a"][t]["err_m"], 4), round(out["c"][t]["err_m"], 4)) for t in tl},
          round(out["f_event"], 5), flush=True)

# ------------------------------------------------------------------ invariants
C = res["cases"]
DETS = ("wcte", "sk", "juno", "hk")


def E(name, t, form="c"):
    v = C.get(name)
    return None if v is None or t not in v[form] else v[form][t]["err_m"]


def runs(det, arm):
    return sorted(k for k, v in C.items() if v["det"] == det and v["arm"] == arm and v["kind"] == "final"
                  and "__ts" not in k)


def leg(det, s):
    return f"{det}__legacy_init__s{s}"


def check(rows):
    """rows: list of (label, bool or None) -> pass only if all known and True.  Rows on a test set
    that does not exist for that detector (sealed_test for HK/JUNO, null in VERIFY_INPUTS_A) are
    'not evaluated', not failures."""
    def absent(label):
        t, _, rest = label.partition(":")
        return t in TESTS and t != "dev_probe_test" and rest.split("__")[0].split(" ")[0] in NO_TEST.get(t, ())
    na = [l for l, r in rows if r is None and absent(l)]
    rows = [(l, r) for l, r in rows if l not in na]
    return {"pass": bool(rows) and all(r is True for _, r in rows), "detail": dict(rows), "not_evaluated": na}


tests = [t for t in TESTS if any(t in v["c"] for v in C.values())]
NO_TEST = {t: tuple(d for d in DETS if not any(v["det"] == d and t in v["c"] for v in C.values())) for t in TESTS}
res["test_sets_absent"] = NO_TEST
inv = {}
rows = []
for t in tests:
    for d in DETS:
        for r in runs(d, "mae_repaired"):
            e, mi = E(r, t), E(r.replace("__final", "__matched_init"), t)
            for s in (9000, 9001):
                l = E(leg(d, s), t)
                rows.append((f"{t}:{r}<legacy{s}", None if None in (e, l) else e < l))
            rows.append((f"{t}:{r}<matched_init", None if None in (e, mi) else e < mi))
inv["J1"] = check(rows)
rows = []
for t in tests:
    for d in ("wcte", "sk"):
        for r in runs(d, "sigreg_jepa"):
            e = E(r, t)
            others = [k for k, v in C.items() if v["det"] == d and v["kind"] == "final" and k != r
                      and "__ts" not in k]
            rows.append((f"{t}:{r} worst trained", None if e is None else all(e > E(o, t) for o in others)))
            for s in (9000, 9001):
                l = E(leg(d, s), t)
                rows.append((f"{t}:{r}>legacy{s}", None if None in (e, l) else e > l))
        if not runs(d, "sigreg_jepa"):
            rows.append((f"{t}:{d} no sigreg_jepa run", None))
inv["J2"] = check(rows)
rows = []
for t in tests:
    for d in ("wcte", "sk"):
        rs = runs(d, "mae_mse")
        if not rs:
            rows.append((f"{t}:{d} no mae_mse run", None))
        for r in rs:
            for s in (9000, 9001):
                e, l = E(r, t), E(leg(d, s), t)
                rows.append((f"{t}:{r}<legacy{s}", None if None in (e, l) else e < l))
inv["J3"] = check(rows)
rows = []
for d in ("wcte", "sk"):
    for r in runs(d, "sigreg_jepa"):
        rows.append((f"{r} f_event<0.005", C[r]["f_event"] < 0.005))
for d in DETS:
    rs = runs(d, "jepa_repaired")
    if not rs:
        rows.append((f"{d} no jepa_repaired run", None))
    for r in rs:
        rows.append((f"{r} f_event>0.03", C[r]["f_event"] > 0.03))
inv["J4"] = check(rows)
rows = []
for t in tests:
    for d in DETS:
        for s in (9000, 9001):
            l, rd = E(leg(d, s), t), res["raw_d"].get(d, {}).get(t, {}).get("err_m")
            rows.append((f"{t}:{d} legacy{s}(c)<(d)", None if None in (l, rd) else l < rd))
inv["J5"] = check(rows)
rows = []
for t in tests:
    for k in C:
        a, c = E(k, t, "a"), E(k, t)
        if a is not None:
            rows.append((f"{t}:{k} a/c={a / c:.3f}", a / c >= 1.5))
inv["J6"] = check(rows)
rows = []
for t in tests:
    for k, v in C.items():
        s, m = E(k, t, "shuffled_c"), res["predict_mean"][v["det"]].get(t)
        if s is not None:
            rows.append((f"{t}:{k} shuf/mean={s / m:.3f}", abs(s / m - 1) <= 0.10))
inv["J7"] = check(rows)
rows = []
for t in tests:
    for d in DETS:
        rs = runs(d, "jepa_repaired")
        for r in rs:
            for s in (9000, 9001):
                e, l = E(r, t), E(leg(d, s), t)
                rows.append((f"{t}:{r}<legacy{s}", None if None in (e, l) else e < l))
inv["J8"] = check(rows)
rows = []
sealed = [t for t in tests if t != "dev_probe_test"]
for t in sealed:
    for k in C:
        a, b = E(k, t), E(k, "dev_probe_test")
        if a is not None:
            rows.append((f"{k} {t}/dev={a / b:.3f}", abs(a / b - 1) <= 0.10))
inv["J9"] = check(rows) if sealed else {"pass": None, "detail": "no sealed_test evaluated"}
res["invariants"] = inv
for k, v in inv.items():
    print(k, v["pass"], [x for x, ok in (v["detail"].items() if isinstance(v["detail"], dict) else []) if ok is not True][:8])
json.dump(res, open(OUT, "w"), indent=1)
