#!/usr/bin/env python3
"""r2-sparse-sk T7.3 exploratory diagnostics (labelled exploratory; validation cohort only, never sealed).

* ``fown``: own-block mass fraction of SK-S2L (seeds 9000/9001/9002, step 10,000) on SK ``dev_probe_test``
  (the first 5 events of every run), with r2-sparse2's definition (PREREG appendix B there): f_own = own /
  (mass on all hidden hits), chance = query-weighted n_own / n_hidden, ratio = f_own / chance; also own / total.
* ``misid``: mu mis-ID at 99 % e efficiency versus event brightness for SK-C2, SK-S1L, SK-S1A (and SK-D0): from
  the evaluator's validation outputs (``validation.npz``: p(mu), is_mu, event ids, step 10,000), brightness =
  number of hit PMTs of the event in the SK ``lucid_v1`` ``dev_probe_test`` pack; three brightness terciles of
  the mu/e events, mis-ID with the e-efficiency threshold set within each tercile.
* the time span of SK-S2L's 32-hit blocks is the T0 measurement (``T0_HITS_SK.json``; the blocks are the training
  blocks of the same data).

    sparse_sk_explore.py --out EXPLORATORY.json
"""
from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from jepa_noise import data_roots as dr  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import r2_store as rs  # noqa: E402
from jepa_noise.lucid_views import ViewMasker  # noqa: E402

RET = Path("/ceph/dwong/jepa/returns")
TRAIN = RET / "r2_sparse_sk_2026-11-26/train/runs"
EVAL = [RET / "r2_sparse_sk_2026-11-26/eval", RET / "r2_follow_2026-11-24/eval", RET / "r2_mask_2026-11-23/eval",
        RET / "r2_tier1_2026-11-22/eval"]
SEEDS = {"A": 9000, "B": 9001, "C": 9002}
CELLS = {"SK-D0": "sk__mae_repaired__s{s}", "SK-C2": "sk__mae_repaired__s{s}__lfixed__psent",
         "SK-S1L": "sk__sparse_mae__s{s}__elin__dleg__qall", "SK-S1A": "sk__sparse_mae__s{s}__elin__dleg__abs__qall"}
KEYS = ("own", "other", "unhit", "queries", "chance_block_sum", "n_blocks", "chance_query_sum", "own_over_hidden_sum")


def fown(per_run=5, chunk=4):
    import tier1_eval as te
    out = {}
    pack = dr.pack("sk", dr.LUCID_V1)
    cohort = rs.open_cohort(pack, "dev_probe_test")
    for lab, s in SEEDS.items():
        rd = TRAIN / f"sk__sparse_time__s{s}__elin__dfull"
        if not (rd / "snapshot_step_10000.pt").is_file():
            continue
        arm, geometry, scaling, meta = te.load_model(Namespace(legacy_init=False, run_dir=str(rd), step=10000))
        arm.eval()
        masker = ViewMasker(geometry, "spatial_patch", seed=lb.DEFAULT_SEED, n_min=0)
        acc = {k: 0.0 for k in KEYS}
        n = 0
        for i in range(cohort.n_runs):
            ids = cohort.run_truth(i)["event_ids"][:per_run]
            batch = rs.tokenise_sparse(cohort.run_store(i), geometry, event_ids=ids, scaling=scaling, masker=masker)
            for k0 in range(0, len(batch.event_ids), chunk):
                from dataclasses import replace
                sub = replace(batch, **{f: getattr(batch, f)[k0:k0 + chunk] for f in ("features", "hidden")},
                              event_ids=batch.event_ids[k0:k0 + chunk])
                with torch.no_grad():
                    st = arm.block_mass_stats(sub)
                for k in KEYS:
                    acc[k] += st[k]
            n += len(batch.event_ids)
        q = max(acc["queries"], 1.0)
        f = acc["own_over_hidden_sum"] / q
        ch = acc["chance_query_sum"] / q
        out[lab] = {"events": n, "f_own": f, "chance": ch, "ratio": f / ch if ch else None,
                    "own_over_total": acc["own"] / q, "mass_on_hidden": (acc["own"] + acc["other"]) / q,
                    "mass_on_unhit": acc["unhit"] / q, "chance_block_mean": acc["chance_block_sum"] / max(acc["n_blocks"], 1)}
    return out


def find_eval(rid):
    for r in EVAL:
        p = r / rid / "step_10000" / "validation.npz"
        if p.is_file():
            return p
    return None


def misid_vs_brightness():
    from tier1_eval import misid
    cohort = rs.open_cohort(dr.pack("sk", dr.LUCID_V1), "dev_probe_test", truth_fields=())
    nhit = dict(zip(np.asarray(cohort.event_ids).tolist(), np.diff(np.asarray(cohort.offsets)).tolist()))
    out = {}
    for cell, pat in CELLS.items():
        per = {}
        for lab, s in SEEDS.items():
            p = find_eval(pat.format(s=s))
            if p is None:
                continue
            with np.load(p) as z:
                ids = z["event_ids"][np.isin(z["domain"], ("muon", "electron"))]
                pm, mu = z["pid_p_mu"], z["pid_is_mu"].astype(bool)
            b = np.asarray([nhit[int(e)] for e in ids], float)
            edges = np.quantile(b, [0, 1 / 3, 2 / 3, 1])
            rows = []
            for j in range(3):
                sel = (b >= edges[j]) & ((b < edges[j + 1]) if j < 2 else (b <= edges[j + 1]))
                rows.append({"hits_range": [float(edges[j]), float(edges[j + 1])], "n_mu": int((sel & mu).sum()),
                             "n_e": int((sel & ~mu).sum()), "misid_mu": misid(pm[sel], mu[sel])[0]})
            per[lab] = {"overall": misid(pm, mu)[0], "terciles": rows}
        if per:
            per["seed_mean_terciles"] = [float(np.mean([per[l]["terciles"][j]["misid_mu"] for l in per if l in SEEDS]))
                                         for j in range(3)]
        out[cell] = per
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-fown", action="store_true")
    a = ap.parse_args()
    t0 = json.loads((REPO / "results/round2/20261126_sparse_sk/T0_HITS_SK.json").read_text())
    res = {"label": "exploratory (runbook §T7.3); validation cohort only",
           "s2l_block_span_ns": t0["span_ns_32_consecutive_hits"],
           "misid_vs_brightness": misid_vs_brightness()}
    if not a.skip_fown:
        res["s2l_fown"] = fown()
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"fown": {k: v.get("ratio") for k, v in res.get("s2l_fown", {}).items()},
                      "misid_terciles": {c: v.get("seed_mean_terciles") for c, v in res["misid_vs_brightness"].items()}}))


if __name__ == "__main__":
    main()
