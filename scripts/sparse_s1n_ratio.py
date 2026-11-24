#!/usr/bin/env python3
"""r2-sparse W2 (§2.2 S1n): fix the WCTE negative-sampling ratio r before any training.

Kept fraction of masked unhit PMTs per event: min(n_unhit_masked, ceil(r * n_hit_masked)) / n_unhit_masked,
with the training patch mask (spatial_patch, mask seed = the dataset seed) and the training
acceptance (>= N_MIN masked hits). SK reference: ``lucid_v1`` ``dev_probe_train`` at r = 2 (SK pack,
SK geometry); WCTE: ``lucid_v1_w2`` ``dev_train`` at r in {0.25, 0.5, 1, 2}. r = the value whose
median kept fraction is closest to SK's. Base env.

    sparse_s1n_ratio.py --out results/round2/20261124_sparse/S1N_RATIO.json [--wcte-runs 20]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import data_roots as dr  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import r2_store as rs  # noqa: E402
from jepa_noise.lucid_views import spatial_patch_mask  # noqa: E402

RATIOS = (0.25, 0.5, 1.0, 2.0)


def kept(n_h, n_u, r):
    return min(n_u, math.ceil(r * n_h)) / n_u


def fractions(stores, geometry, ratios, mask_seed):
    out = {r: [] for r in ratios}
    n_ev = n_acc = 0
    occ = []
    for store in stores:
        for k, eid in enumerate(store.event_ids):
            n_ev += 1
            hit = np.zeros(geometry.n_pmt, bool)
            hit[store.pmt_id[store.offsets[k]:store.offsets[k + 1]]] = True
            occ.append(hit.mean())
            m = spatial_patch_mask(geometry, int(eid), hit, seed=mask_seed)
            if not m.accepted:
                continue
            n_acc += 1
            tgt = np.zeros(geometry.n_pmt, bool)
            tgt[m.target] = True
            n_h, n_u = int((tgt & hit).sum()), int((tgt & ~hit).sum())
            if n_u == 0:
                for r in ratios:
                    out[r].append(1.0)
                continue
            for r in ratios:
                out[r].append(kept(n_h, n_u, r))
    return ({str(r): {"median": float(np.median(v)), "mean": float(np.mean(v))} for r, v in out.items()},
            {"n_events": n_ev, "n_accepted": n_acc, "median_occupancy": float(np.median(occ))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wcte-runs", type=int, default=20)
    a = ap.parse_args()
    sk_geo = dr.geometry("sk", dr.LUCID_V1)
    sk = rs.open_cohort(dr.pack("sk", dr.LUCID_V1), "dev_probe_train", truth_fields=())
    sk_stores = [sk.run_store(i) for i in range(sk.n_runs)]
    sk_frac, sk_info = fractions(sk_stores, sk_geo, (2.0,), lb.DEFAULT_SEED)
    from jepa_noise.ext_r2_repair_train import WcteTrainingShards
    wc = WcteTrainingShards(dr.wcte_train_shards(dr.LUCID_V1_W2), data_kind=dr.LUCID_V1_W2)
    wc_geo = dr.geometry("wcte", dr.LUCID_V1_W2)
    stores = [wc.run_store(i) for i in range(min(a.wcte_runs, len(wc.rows)))]
    wc_frac, wc_info = fractions(stores, wc_geo, RATIOS, int(wc.manifest["dataset_seed"]))
    target = sk_frac["2.0"]["median"]
    best = min(RATIOS, key=lambda r: (abs(wc_frac[str(r)]["median"] - target), r))
    res = {"rule": "r in {0.25, 0.5, 1, 2} whose median kept fraction of masked unhit PMTs on WCTE lucid_v1_w2 "
                   "dev_train is closest to SK's median kept fraction at r = 2 (SK lucid_v1 dev_probe_train)",
           "sk_reference": {"r": 2.0, "kept_fraction": sk_frac["2.0"], **sk_info},
           "wcte_dev_train": {"kept_fraction": wc_frac, "n_runs_used": len(stores), **wc_info},
           "chosen_r": best, "acceptance": "training acceptance (spatial_patch, >= 20 masked hits)"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"chosen_r": best, "sk_median": target,
                      "wcte": {k: v["median"] for k, v in wc_frac.items()}}))


if __name__ == "__main__":
    main()
