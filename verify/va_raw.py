"""Raw-feature baseline (d) features (33 per event, runbook §1) for one detector cohort, CPU.

Usage: python va_raw.py DET PACK GROUP OUT.npz [--domain photon] [--threads 2]
"""
import argparse
import json

import numpy as np
import torch

from lucid_common import LCohort, LGeometry, build_features
from readout_spec import raw_baseline

ap = argparse.ArgumentParser()
ap.add_argument("det"); ap.add_argument("pack"); ap.add_argument("group"); ap.add_argument("out")
ap.add_argument("--domain", default="photon"); ap.add_argument("--threads", type=int, default=2)
a = ap.parse_args()
torch.set_num_threads(a.threads)
geo = LGeometry(a.det, a.pack)
truth = ("vertex_m",) if a.domain == "photon" else ("vertex_m", "energy_MeV", "direction_m")
coh = LCohort(a.pack, a.group, domain=a.domain.encode(), truth=truth)
X = np.zeros((coh.n, 33))
bs = 16
for s in range(0, coh.n, bs):
    ks = list(range(s, min(s + bs, coh.n)))
    f = torch.from_numpy(build_features(geo, coh, ks)).double()
    X[ks] = raw_baseline(f).numpy()
np.savez(a.out, d=X, event_ids=coh.event_ids, run=coh.run, **coh.truth,
         meta=json.dumps({"det": a.det, "pack": a.pack, "group": a.group, "domain": a.domain, "n": int(coh.n)}))
print("DONE", a.det, a.group, coh.n, flush=True)
