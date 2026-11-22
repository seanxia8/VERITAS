"""Encoder forward (black box) on full, unmasked events of one cohort; keeps per event:
  a      (128)  all-PMT mean token (readout (a))
  msq    (128)  all-PMT mean of z^2 (for f_event: total variance over any event subset)
  c      (768)  readout (c) [m1, m2, m3, vec(C)] (runbook §1)
plus event ids, run index and truth (vertex_m; energy_MeV for non-photon domains).

Usage: python va_dump.py CASES.json CASE PACK GROUP OUT.npz [--device cuda:0] [--domain photon] [--threads 6]
"""
import argparse
import json
import time

import numpy as np
import torch

from lucid_common import LCohort, LGeometry, build_features, load_encoder, sha256
from readout_spec import spatial_moments

ap = argparse.ArgumentParser()
ap.add_argument("cases"); ap.add_argument("case"); ap.add_argument("pack"); ap.add_argument("group")
ap.add_argument("out"); ap.add_argument("--device", default="cpu"); ap.add_argument("--domain", default="photon")
ap.add_argument("--threads", type=int, default=6); ap.add_argument("--batch", type=int, default=0)
ap.add_argument("--no-sha", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)
case = {c["case"]: c for c in json.load(open(a.cases))["cases"]}[a.case]
det = case["det"]
if case["ckpt"] and not a.no_sha:
    assert sha256(case["ckpt"]) == case["sha256"], "checkpoint sha256 changed"
geo = LGeometry(det, a.pack)
truth = ("vertex_m",) if a.domain == "photon" else ("vertex_m", "energy_MeV", "direction_m")
coh = LCohort(a.pack, a.group, domain=a.domain.encode(), truth=truth)
model, _ = load_encoder(det, case["ckpt"], window=case["window"], shift=case["shift"], seed=case["seed"],
                        device=a.device)
bs = a.batch or {"wcte": 64, "sk": 8, "juno": 8, "hk": 4}[det]
A = np.zeros((coh.n, 128), np.float32); M = np.zeros_like(A); Cf = np.zeros((coh.n, 768), np.float32)
t0 = time.time()
for s in range(0, coh.n, bs):
    ks = list(range(s, min(s + bs, coh.n)))
    x = torch.from_numpy(build_features(geo, coh, ks)).to(a.device)
    with torch.inference_mode():
        if case["readout"] == "posfree":
            x_in = torch.cat((torch.zeros_like(x[..., :6]), x[..., 6:]), -1)
        else:
            x_in = x
        z = model.encode(x_in)
        if case["readout"] == "layernorm":   # LucidArm.encode_tokens convention for jepa_repaired
            z = torch.nn.functional.layer_norm(z, (z.size(-1),))
        z = z.double(); f = x.double()
        A[ks] = z.mean(1).cpu().numpy()
        M[ks] = (z * z).mean(1).cpu().numpy()
        Cf[ks] = spatial_moments(z, f)[0].cpu().numpy()
    if s % (bs * 100) == 0:
        print(f"{a.case} {a.group} {s}/{coh.n} {time.time() - t0:.0f}s", flush=True)
extra = {k: v for k, v in coh.truth.items()}
np.savez(a.out, a=A, msq=M, c=Cf, event_ids=coh.event_ids, run=coh.run, **extra,
         meta=json.dumps({"case": a.case, "pack": a.pack, "group": a.group, "domain": a.domain,
                          "n": int(coh.n), "seconds": time.time() - t0, "device": a.device}))
print("DONE", a.case, a.group, coh.n, round(time.time() - t0), flush=True)
