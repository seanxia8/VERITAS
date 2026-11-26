#!/usr/bin/env python3
"""r2-sparse-sk T0: hit statistics of the lucid_v1 ``dev_train`` photon bombs (CPU, read-only).

Runbook ``RUNBOOK_2026-11-26_R2_SPARSE_SK.md`` §T0. Per detector (SK registered; HK and JUNO for
Stages H and J): hits per event (median, q10, q90), occupancy (hit PMTs / all PMTs), the time span
of 32 consecutive hits in time order (S2L's blocks), and the default spatial-patch layout seen by
the absence tokens (patches per event, hit patches, hit fraction of a patch).

    sparse_sk_t0.py --detector sk --max-events 20000 --out T0_SK_HITS.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import data_roots as dr  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import r2_store as rs  # noqa: E402
from jepa_noise.lucid_views import default_patch_size, token_hilbert_order  # noqa: E402

BLOCK = 32


def q(x):
    x = np.asarray(x, dtype=np.float64)
    if not x.size:
        return {"n": 0}
    return {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.1)),
            "q90": float(np.quantile(x, 0.9)), "mean": float(x.mean()), "n": int(x.size)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", required=True, choices=("sk", "hk", "juno"))
    ap.add_argument("--max-events", type=int, default=20000,
                    help="first N photon bombs of dev_train in pack order (0 = all)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    kind = dr.LUCID_V1
    h5 = dr.pack(args.detector, kind)
    found = dr.check_pack_kind(h5, kind)
    geometry = dr.geometry(args.detector, kind)
    n_pmt = int(geometry.n_pmt)
    size = default_patch_size(n_pmt)
    order = token_hilbert_order(geometry)
    count = n_pmt // size
    patch_of = np.full(n_pmt, count, dtype=np.int64)          # tail beyond count * size = patch "count"
    patch_of[order[:count * size]] = np.repeat(np.arange(count), size)
    cohort = rs.open_cohort(h5, "dev_train", truth_fields=("domain",))
    domain = np.asarray(cohort.truth["domain"]).astype(str)
    photon = np.flatnonzero(domain == lb.DOMAIN_PHOTON)
    if args.max_events:
        photon = photon[:args.max_events]
    n_hits, spans, hit_patches, patch_frac, dup = [], [], [], [], 0
    for e in photon:
        lo, hi = int(cohort.offsets[e]), int(cohort.offsets[e + 1])
        pmt = np.asarray(cohort.pmt_id[lo:hi])
        t = np.asarray(cohort.time_ns[lo:hi], dtype=np.float64)
        u, last = np.unique(pmt[::-1], return_index=True)   # one token per PMT (last write wins)
        dup += int(pmt.size - u.size)
        t = np.sort(t[::-1][last])
        n_hits.append(t.size)
        for k in range(t.size // BLOCK):
            spans.append(t[(k + 1) * BLOCK - 1] - t[k * BLOCK])
        per_patch = np.bincount(patch_of[u], minlength=count + 1)[:count]
        hit_patches.append(int((per_patch > 0).sum()))
        patch_frac.extend((per_patch / size).tolist())
    n_hits = np.asarray(n_hits, dtype=np.float64)
    out = {"detector": args.detector, "data_kind": kind, "pack": str(h5), "pack_dataset_kind": found,
           "cohort": "dev_train", "domain": lb.DOMAIN_PHOTON, "n_photon_bombs_in_cohort": int((domain == lb.DOMAIN_PHOTON).sum()),
           "n_events_measured": int(photon.size), "n_pmt": n_pmt,
           "duplicate_pmt_entries": dup, "hits_per_event": q(n_hits), "occupancy": q(n_hits / n_pmt),
           "unhit_fraction_median_event": float(1.0 - np.median(n_hits) / n_pmt),
           "span_ns_32_consecutive_hits": q(spans),
           "patches": {"patch_size": size, "full_patches": count, "tail_pmts": int(n_pmt - count * size),
                       "hit_patches_per_event": q(hit_patches),
                       "hit_fraction_per_patch": q(patch_frac)}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("detector", "n_events_measured", "hits_per_event", "occupancy",
                                           "span_ns_32_consecutive_hits")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
