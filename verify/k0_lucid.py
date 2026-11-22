"""K0 on LUCiD data: the independent 9-feature rebuild vs the repository tokeniser (the object
under test, ``r2_store.tokenise_sparse`` with ``data_roots.geometry`` and the lucid_v1 time rule).

200 events per detector from dev_probe_test (all domains, first runs) + 200 photon bombs.
Usage: python k0_lucid.py OUT.json
"""
import json
import sys

import numpy as np

from lucid_common import DETS, PACK, REPO, LCohort, LGeometry, build_features

sys.path.insert(0, f"{REPO}/src")
from jepa_noise import data_roots, r2_store  # noqa: E402
from jepa_noise import lucid_mc_generator as mg  # noqa: E402  (time rule only)

COLS = ["pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z", "ln_q", "time", "hit"]
N = 200
out = {}
for det in DETS:
    geo = LGeometry(det)
    rgeo = data_roots.geometry(det)
    ts = float(mg.time_rule(det, mg.load_config())["time_scale_ns"])
    scaling = r2_store.scaling_for(rgeo, ts)
    full = r2_store.open_cohort(PACK[det], "dev_probe_test", truth_fields=["domain"])
    res = {"time_scale_ns_pack": geo.time_scale_ns, "time_scale_ns_repo": ts,
           "geometry_positions_max_abs_diff_m": float(np.abs(rgeo.positions_m - geo.pos).max()),
           "geometry_directions_max_abs_diff": float(np.abs(rgeo.directions - geo.dirs).max()),
           "scale_repo": [scaling.radial_scale_m, scaling.radial_scale_m, scaling.axial_scale_m],
           "scale_mine": geo.scale.tolist()}
    for label, domain in (("all_domains", None), ("photon", b"photon")):
        coh = LCohort(PACK[det], "dev_probe_test", domain=domain, truth=())
        ks = list(range(N))
        runs = sorted(set(coh.run[ks].tolist()))
        feats, ids = [], []
        for r in runs:
            b = r2_store.tokenise_sparse(full.run_store(r), rgeo, scaling=scaling, n_min=0)
            feats.append(b.features.numpy()); ids.append(b.event_ids)
        repo_f = np.concatenate(feats); row = {int(e): i for i, e in enumerate(np.concatenate(ids))}
        ks = [k for k in ks if int(coh.event_ids[k]) in row]
        mine = build_features(geo, coh, ks)
        repo = repo_f[[row[int(coh.event_ids[k])] for k in ks]]
        cols = {}
        for c, name in enumerate(COLS):
            d = np.abs(mine[..., c].astype(np.float64) - repo[..., c].astype(np.float64))
            cols[name] = {"max_abs_diff": float(d.max()), "bitwise_equal": bool(np.array_equal(mine[..., c], repo[..., c]))}
        res[label] = {"n_events": len(ks), "n_missing_in_repo": N - len(ks), "columns": cols,
                      "bitwise_equal_all": bool(np.array_equal(mine, repo))}
    res["pass"] = bool(res["all_domains"]["bitwise_equal_all"] and res["photon"]["bitwise_equal_all"]
                       and res["all_domains"]["n_missing_in_repo"] == 0 and res["photon"]["n_missing_in_repo"] == 0)
    out[det] = res
    print(det, json.dumps(res), flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
