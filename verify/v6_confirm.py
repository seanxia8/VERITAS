"""V6 discrepancy investigation for K1 (c)/(d): an own re-implementation of the committed
"spatial moment" construction (read from ext_r2_repair_probes / r2_probe_repair only after
the blind seal b755b23), to confirm that the difference is the feature definition.

Committed latent form = [hit mean, charge-weighted mean (w ~ q^3 over hits), early-time mean
(w ~ q^3 exp(-4 (t - t_first)/T) over prompt hits t >= 0), cross-moments with positions
centred on the event's hit centroid] (6 x 128).  Committed raw form = the same on the
[ln q, t/T, hit] columns plus 15 summaries (hit fraction, ln n_hit, mean/sd of ln q over hits,
mean/sd of t over prompt hits, hit / charge-weighted / early-weighted position centroids).

Usage:
  python v6_confirm.py raw OUT.json                    # (d') all detectors, no encoder
  python v6_confirm.py centred DUMPDIR OUT.json        # (c) with centred cross, from the dump
  python v6_confirm.py enc CASE OUTDIR [threads]       # (c') committed latent form via encoder
"""
import json
import sys

import numpy as np

from common import DET, Cohort, Geometry, build_features
from ridge_core import fit_predict


def weights(f):
    """f: (B, N, 9) features -> hit mask, charge weights, early weights (B, N)."""
    hit = f[..., 8] > 0.5
    prompt = hit & (f[..., 7] >= 0)
    first = np.min(np.where(prompt, f[..., 7], np.inf), axis=1, keepdims=True)
    first = np.where(np.isfinite(first), first, 0.0)

    def norm(logw, mask):
        m = np.max(np.where(mask, logw, -np.inf), axis=1, keepdims=True)
        w = np.where(mask, np.exp(logw - np.where(np.isfinite(m), m, 0)), 0.0)
        s = w.sum(1, keepdims=True)
        return w / np.where(s > 0, s, 1)
    wq = norm(3.0 * f[..., 6], hit)
    we = norm(3.0 * f[..., 6] - 4.0 * (f[..., 7] - first), prompt)
    we = np.where(prompt.any(1, keepdims=True), we, wq)
    return hit, prompt, wq, we


def committed_style(tok, f, summaries=False):
    hit, prompt, wq, we = weights(f)
    wh = hit / np.maximum(hit.sum(1, keepdims=True), 1)
    pos = f[..., :3]
    cen = (wh[..., None] * pos).sum(1)
    cross = np.einsum("bnc,bnd->bcd", (pos - cen[:, None]) * wh[..., None], tok).reshape(len(f), -1)
    parts = [np.einsum("bnd,bn->bd", tok, w) for w in (wh, wq, we)] + [cross]
    if summaries:
        def ms(v, m):
            c = np.maximum(m.sum(1), 1); mu = (v * m).sum(1) / c
            return [mu, np.sqrt(((v - mu[:, None]) ** 2 * m).sum(1) / c)]
        nh = hit.sum(1)
        s = [nh / f.shape[1], np.log(np.maximum(nh, 1))] + ms(f[..., 6], hit) + ms(f[..., 7], prompt)
        parts += [np.column_stack(s), cen, np.einsum("bnc,bn->bc", pos, wq), np.einsum("bnc,bn->bc", pos, we)]
    return np.concatenate(parts, 1)


def metric(pred, y):
    return float(np.linalg.norm(pred - y, axis=1).mean())


mode = sys.argv[1]
if mode == "raw":
    out = {}
    for det in DET:
        geo = Geometry(det); X = {}; Y = {}; G = {}
        for split in ("train", "test"):
            coh = Cohort(det, split); rows = []
            for a in range(0, coh.n, 16):
                ks = list(range(a, min(a + 16, coh.n)))
                f = build_features(geo, coh, ks).astype(np.float64)
                rows.append(committed_style(f[..., 6:9], f, summaries=True))
            X[split], Y[split], G[split] = np.concatenate(rows), coh.vertex, coh.run
        pred, a = fit_predict(X["train"], Y["train"], X["test"], G["train"])
        out[det] = {"d_prime_committed_style_raw_m": metric(pred, Y["test"]), "alpha": a, "n_features": X["train"].shape[1]}
        print(det, out[det], flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
elif mode == "centred":
    from cases import cases
    dump, out = sys.argv[2], {}
    cent = {}
    for c in cases():
        det = c["det"]
        if det not in cent:  # per-event hit centroid in unit coordinates, from the data
            geo = Geometry(det); cent[det] = {}
            for split in ("train", "test"):
                coh = Cohort(det, split)
                cent[det][split] = np.array([geo.unit_pos[coh.hits(k)[0]].mean(0) for k in range(coh.n)])
        F = {}
        for split in ("train", "test"):
            d = np.load(f"{dump}/{c['case']}__{split}.npz")
            mh, mom = d["mean_hit"].astype(np.float64), d["mom_hit"].astype(np.float64)
            centred = mom - cent[det][split][:, :, None] * mh[:, None, :]
            F[split] = (np.concatenate([mh, centred.reshape(len(mh), -1)], 1), d["vertex"], d["run"])
        pred, a = fit_predict(F["train"][0], F["train"][1], F["test"][0], F["train"][2])
        out[c["case"]] = {"c_centred_m": metric(pred, F["test"][1]), "alpha": a}
        print(c["case"], out[c["case"]], flush=True)
    json.dump(out, open(sys.argv[3], "w"), indent=1)
elif mode == "enc":
    import torch
    from cases import cases
    from common import load_encoder
    case = {c["case"]: c for c in cases()}[sys.argv[2]]
    torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 5)
    geo = Geometry(case["det"])
    model, _ = load_encoder(case["det"], case["ckpt"], case["seed"], case["window"], case["shift"])
    X, Y, G = {}, {}, {}
    for split in ("train", "test"):
        coh = Cohort(case["det"], split); rows = []
        bs = 32 if case["det"] == "wcte" else 6
        for a in range(0, coh.n, bs):
            ks = list(range(a, min(a + bs, coh.n)))
            x = torch.from_numpy(build_features(geo, coh, ks))
            with torch.inference_mode():
                z = model.encode(x)
                if case["readout"] == "layernorm":
                    z = torch.nn.functional.layer_norm(z, (z.size(-1),))
            rows.append(committed_style(z.double().numpy(), x.double().numpy()))
        X[split], Y[split], G[split] = np.concatenate(rows), coh.vertex, coh.run
    pred, a = fit_predict(X["train"], Y["train"], X["test"], G["train"])
    res = {"case": case["case"], "c_prime_committed_style_latent_m": metric(pred, Y["test"]), "alpha": a}
    print(res, flush=True)
    json.dump(res, open(f"{sys.argv[3]}/{case['case']}__confirm.json", "w"), indent=1)
