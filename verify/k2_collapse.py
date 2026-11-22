"""K2: token-variance decomposition from first principles on the 200 dumped probe-test events.

For tokens z[e, p, d] (population variances, summed over d = trace):
  total        = Var_{e,p}(z)
  static       = Var_p( mean_e z[e, p] )          (layout: per-PMT mean over events)
  across_event = mean_p Var_e( z[e, p] )          (same PMT, across events)
  total = static + across_event exactly (law of total variance; checked).
Hit-only: the same over hit tokens, PMTs weighted by their hit counts.
Event-mean spread = tr Cov_e( mean_p z[e, p] ) / total.

Usage: python k2_collapse.py DUMPDIR OUT.json
"""
import json
import os
import sys

import numpy as np

from cases import cases

DUMP, OUT = sys.argv[1], sys.argv[2]


def decompose(n_e, s1, s2, counts=None):
    """s1, s2: per-PMT sums of z and z^2 over events, (P, D); counts: per-PMT event counts."""
    cnt = np.full(s1.shape[0], float(n_e)) if counts is None else counts.astype(float)
    keep = cnt > 0
    s1, s2, cnt = s1[keep], s2[keep], cnt[keep]
    ntok = cnt.sum()
    gmean = s1.sum(0) / ntok
    total = (s2.sum(0) / ntok - gmean ** 2).sum()
    pm = s1 / cnt[:, None]
    w = cnt / ntok
    static = (w[:, None] * (pm - gmean) ** 2).sum()
    across = (w[:, None] * (s2 / cnt[:, None] - pm ** 2)).sum()
    return {"total": float(total), "static_frac": float(static / total), "across_event_frac": float(across / total),
            "closure": float((static + across) / total), "n_tokens": float(ntok)}


res = {"cases": {}}
for c in cases():
    p = f"{DUMP}/{c['case']}__k2.npz"
    if not os.path.exists(p):
        continue
    k = np.load(p)
    n = int(k["n_events"])
    allp = decompose(n, k["s1"], k["s2"])
    hit = decompose(n, k["h1"], k["h2"], k["hc"])
    m = k["mean_all"]
    spread = float(np.trace(np.cov(m.T, bias=True)) / allp["total"])
    res["cases"][c["case"]] = {"det": c["det"], "arm": c["arm"], "seed": c["seed"], "n_events": n,
                               "all_pmt": allp, "hit_only": hit, "event_mean_spread": spread}
    print(c["case"], round(allp["static_frac"], 5), "%.3e" % allp["across_event_frac"], round(hit["static_frac"], 5),
          "%.3e" % spread, flush=True)

C = res["cases"]
i7 = {}
for d in ("wcte", "sk", "hk"):
    s, mm = C.get(f"{d}__sigreg_jepa__s9000"), C.get(f"{d}__mae_mse__s9000")
    if s and mm:
        ratio = mm["all_pmt"]["across_event_frac"] / s["all_pmt"]["across_event_frac"]
        i7[d] = {"across_ratio_mae_over_sigreg": ratio, "sigreg_static_frac": s["all_pmt"]["static_frac"],
                 "pass": bool(ratio >= 50 and s["all_pmt"]["static_frac"] >= 0.95)}
res["invariants"] = {"I7": {"pass": len(i7) == 3 and all(v["pass"] for v in i7.values()), "detail": i7}}
j, others = C.get("juno__sigreg_jepa__s9000"), [C.get(f"{d}__sigreg_jepa__s9000") for d in ("sk", "hk")]
if j and all(others):
    res["invariants"]["I8"] = {
        "independent_part": "JUNO sigreg across-event fraction vs SK/HK (committed comparison in V6)",
        "juno_across": j["all_pmt"]["across_event_frac"], "sk_hk_across": [o["all_pmt"]["across_event_frac"] for o in others],
        "juno_less_extreme": bool(j["all_pmt"]["across_event_frac"] > max(o["all_pmt"]["across_event_frac"] for o in others))}
print(json.dumps(res["invariants"], indent=1))
json.dump(res, open(OUT, "w"), indent=1)
