"""K0: independent encoder-input rebuild vs the repository feature builder (allowed code).

Usage: python k0_features.py OUT.json
"""
import json
import sys

import numpy as np

from common import DET, Cohort, Geometry, REPO, build_features

sys.path.insert(0, f"{REPO}/src")
from jepa_noise import r2_store  # noqa: E402  (allowed: feature builder under test)
from jepa_noise.r2_geometry import load_r2_geometry  # noqa: E402

COLS = ["pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z", "ln_q", "time", "hit"]
out = {}
for det in DET:
    geo = Geometry(det)
    coh = Cohort(det, "test")
    rgeo = load_r2_geometry(det)
    # first photon runs of the test split, first 64 photon events
    runs = sorted(set(coh.run[:64].tolist()))
    full = r2_store.open_cohort(DET[det]["h5"], "dev_probe_test", truth_fields=["domain"])
    scaling = r2_store.scaling_for(rgeo, geo.time_scale_ns)
    feats, ids = [], []
    for r in runs:
        b = r2_store.tokenise_sparse(full.run_store(r), rgeo, scaling=scaling)
        feats.append(b.features.numpy()); ids.append(b.event_ids)
    repo_f = np.concatenate(feats); repo_ids = np.concatenate(ids)
    ks = [k for k in range(64) if coh.event_ids[k] in set(repo_ids.tolist())]
    mine = build_features(geo, coh, ks)
    row = {int(e): i for i, e in enumerate(repo_ids)}
    repo = repo_f[[row[int(coh.event_ids[k])] for k in ks]]
    # geometry ordering check: repository PMT table vs npz rows
    perm_ok = bool(np.allclose(rgeo.positions_m, geo.pos, atol=1e-9))
    dup = 0
    for k in ks:
        p, _, _ = coh.hits(k)
        dup += len(p) - len(np.unique(p))
    res = {"n_events": len(ks), "n_dropped_by_repo_nmin": 64 - len(ks), "geometry_rows_identical": perm_ok,
           "duplicate_pmt_hits": int(dup), "time_scale_ns": geo.time_scale_ns,
           "scale_R_R_Hhalf": geo.scale.tolist(), "columns": {}}
    worst = 0.0
    for c, name in enumerate(COLS):
        d = np.abs(mine[..., c].astype(np.float64) - repo[..., c].astype(np.float64))
        ref = np.abs(repo[..., c]).max()
        rel = float(d.max() / ref) if ref > 0 else float(d.max())
        res["columns"][name] = {"max_abs_diff": float(d.max()), "max_abs_value": float(ref), "rel": rel}
        worst = max(worst, rel)
    res["max_rel"] = worst
    res["pass"] = bool(worst <= 1e-5)
    out[det] = res
    print(det, json.dumps(res), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
