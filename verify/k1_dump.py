"""K1/K2 token dump: run the repository encoder (black box) on full, unmasked inputs of
every photon probe-train / probe-test event and keep exact per-event pooled statistics
(storing the full float32 tokens would be 39-72 GB per SK/HK case):

  mean_all  (128)      mean over all PMTs of z
  mean_hit  (128)      mean over hit PMTs of z
  mom_hit   (3, 128)   mean over hit PMTs of z * x~, z * y~, z * z~ (x~ = position / (R, R, H/2))

and, for the first 200 probe-test photon events, the per-PMT first and second moments
needed for the K2 variance decomposition (all PMTs and hit PMTs only).

Usage: python k1_dump.py CASE OUTDIR [threads]
"""
import json
import sys
import time

import numpy as np
import torch

from cases import cases
from common import Cohort, Geometry, build_features, load_encoder, sha256

N_K2 = 200
BATCH = {"wcte": 32, "sk": 6, "juno": 6, "hk": 3}

case = {c["case"]: c for c in cases()}[sys.argv[1]]
outdir = sys.argv[2]
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 5)
det = case["det"]
if case["ckpt"]:
    got = sha256(case["ckpt"])
    assert got == case["sha256"], f"checkpoint sha256 mismatch {got}"
geo = Geometry(det)
model, _ = load_encoder(det, case["ckpt"], case["seed"], case["window"], case["shift"])
pos = torch.as_tensor(geo.unit_pos, dtype=torch.float64)  # (N, 3)
res = {"case": case, "time_scale_ns": geo.time_scale_ns}
t0 = time.time()
for split in ("train", "test"):
    coh = Cohort(det, split)
    D = 128
    mean_all = np.zeros((coh.n, D)); mean_hit = np.zeros((coh.n, D)); mom = np.zeros((coh.n, 3, D))
    if split == "test":
        n2 = min(N_K2, coh.n)
        s1 = torch.zeros(geo.n, D, dtype=torch.float64); s2 = torch.zeros_like(s1)
        h1 = torch.zeros_like(s1); h2 = torch.zeros_like(s1); hc = torch.zeros(geo.n, dtype=torch.float64)
    bs = BATCH[det]
    for a in range(0, coh.n, bs):
        ks = list(range(a, min(a + bs, coh.n)))
        x = torch.from_numpy(build_features(geo, coh, ks))
        with torch.inference_mode():
            z = model.encode(x)
            if case["readout"] == "layernorm":  # LucidArm.encode_tokens convention for jepa_repaired
                z = torch.nn.functional.layer_norm(z, (z.size(-1),))
        z = z.double()
        hit = x[..., 8].double()  # (B, N)
        nh = hit.sum(1).clamp_min(1)
        mean_all[ks] = z.mean(1).numpy()
        mean_hit[ks] = ((z * hit[..., None]).sum(1) / nh[:, None]).numpy()
        for c in range(3):
            w = hit * pos[:, c][None]
            mom[ks, c] = ((z * w[..., None]).sum(1) / nh[:, None]).numpy()
        if split == "test":
            for b, k in enumerate(ks):
                if k < n2:
                    zz = z[b]; hh = hit[b]
                    s1 += zz; s2 += zz * zz
                    h1 += zz * hh[:, None]; h2 += zz * zz * hh[:, None]; hc += hh
        if a % (bs * 50) == 0:
            print(f"{case['case']} {split} {a}/{coh.n} {time.time() - t0:.0f}s", flush=True)
    np.savez(f"{outdir}/{case['case']}__{split}.npz", mean_all=mean_all.astype(np.float32),
             mean_hit=mean_hit.astype(np.float32), mom_hit=mom.astype(np.float32),
             event_ids=coh.event_ids, run=coh.run, vertex=coh.vertex)
    if split == "test":
        np.savez(f"{outdir}/{case['case']}__k2.npz", n_events=n2, s1=s1.numpy(), s2=s2.numpy(),
                 h1=h1.numpy(), h2=h2.numpy(), hc=hc.numpy(), mean_all=mean_all[:n2])
    res[split] = {"n_events": int(coh.n), "n_runs": int(len(np.unique(coh.run)))}
res["seconds"] = time.time() - t0
json.dump(res, open(f"{outdir}/{case['case']}__meta.json", "w"), indent=1, default=str)
print("DONE", case["case"], res["seconds"], flush=True)
