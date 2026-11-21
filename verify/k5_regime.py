"""K5: occupancy regime of the full probe-test cohort (photon events = the probe cohort;
all domains reported as a secondary row).  Quantiles 5/25/50/75/95 %.

Usage: python k5_regime.py OUT.json
"""
import json
import sys

import numpy as np

from common import DET, Cohort, Geometry

Q = [5, 25, 50, 75, 95]
out = {"quantiles_pct": Q, "detectors": {}}
for det in DET:
    geo = Geometry(det)
    res = {}
    for label, dom in (("photon", b"photon"), ("all_domains", None)):
        coh = Cohort(det, "test", domain=dom) if dom else None
        if coh is None:  # every event of the split
            coh = Cohort(det, "test", domain=b"photon")
            coh.idx = np.arange(len(coh.offsets) - 1); coh.n = len(coh.idx)
        pe, nh = np.zeros(coh.n), np.zeros(coh.n)
        for k in range(coh.n):
            p, q, _ = coh.hits(k)
            pe[k], nh[k] = q.sum(), len(np.unique(p))
        res[label] = {"n_events": int(coh.n),
                      "total_pe": np.percentile(pe, Q).tolist(),
                      "n_hit_pmts": np.percentile(nh, Q).tolist(),
                      "hit_fraction": np.percentile(nh / geo.n, Q).tolist()}
    res["n_pmt"] = geo.n
    out["detectors"][det] = res
    print(det, json.dumps(res["photon"]), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
