#!/usr/bin/env python3
"""r2-sparse2 W1: time-block width for S2Lt (CPU, before any training).

Runbook ``RUNBOOK_2026-11-25_R2_SPARSE2.md`` §W1. On ``lucid_v1_w2`` WCTE ``dev_train`` (photon
bombs; the WCTE training cohort holds photon bombs only):

* the time span of the S2 blocks (32 consecutive hits in time order), median / q10 / q90;
* for Delta in {2, 5, 10} ns: non-empty Delta-wide bins per event (bins start at -110 ns) and hits per
  non-empty bin, median / q10 / q90;
* the registered rule: Delta = 5 ns unless the median event has fewer than 6 non-empty bins, then the
  largest Delta in {5, 2} that gives >= 6.

And, for the r2-sparse S2 recipe (read-only; ``dev_probe_test``, the r2-sparse diagnostic event set:
the first 8 events of every run, 1,000 events), the chance level of the own-block mass fraction:
the mean over hidden blocks of (hits in the block) / (all candidates' hits = all hidden hits).

    sparse2_w1.py --s2-run <r2-sparse train/runs>/wcte__sparse_time__s9000__dw2 --out W1.json
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
from jepa_noise import sparse_arms as sa  # noqa: E402

WIDTHS = (2.0, 5.0, 10.0)
T0_NS = -110.0


def q(x):
    x = np.asarray(x, dtype=np.float64)
    return {"median": float(np.median(x)), "q10": float(np.quantile(x, 0.1)),
            "q90": float(np.quantile(x, 0.9)), "mean": float(x.mean()), "n": int(x.size)}


def event_times(store, i):
    lo, hi = int(store.offsets[i]), int(store.offsets[i + 1])
    pmt = np.asarray(store.pmt_id[lo:hi])
    t = np.asarray(store.time_ns[lo:hi], dtype=np.float64)
    # tokens keep one entry per PMT (the last write wins in _event_features); packs hold one digit per PMT
    _, last = np.unique(pmt[::-1], return_index=True)
    return np.sort(t[::-1][last]), int(pmt.size - last.size)


def measure_train(n_runs):
    root = dr.wcte_train_shards(dr.LUCID_V1_W2)
    manifest = json.loads((root / "manifest.json").read_text())
    dr.check_manifest_kind(manifest, dr.LUCID_V1_W2)
    spans, n_hits, dup = [], [], 0
    bins = {w: [] for w in WIDTHS}
    per_bin = {w: [] for w in WIDTHS}
    t_min, t_max = np.inf, -np.inf
    for row in manifest["runs"][:n_runs]:
        store = lb.read_sparse_hits(root / "runs" / row["run_dir"] / "hits.npz")
        for i in range(len(store.event_ids)):
            t, d = event_times(store, i)
            dup += d
            if t.size == 0:
                continue
            t_min, t_max = min(t_min, t[0]), max(t_max, t[-1])
            n_hits.append(t.size)
            nb = t.size // sa.S2_BLOCK
            for k in range(nb):
                spans.append(t[(k + 1) * sa.S2_BLOCK - 1] - t[k * sa.S2_BLOCK])
            for w in WIDTHS:
                b = np.floor((t - T0_NS) / w).astype(np.int64)
                _, counts = np.unique(b, return_counts=True)
                bins[w].append(counts.size)
                per_bin[w].extend(counts.tolist())
    out = {"cohort": "dev_train", "data_kind": dr.LUCID_V1_W2, "dataset_kind_manifest": manifest["dataset_kind"],
           "lucid_sha": manifest["lucid"]["lucid_sha"], "n_runs": n_runs, "n_events": len(n_hits),
           "duplicate_pmt_entries": dup, "time_range_ns": [float(t_min), float(t_max)],
           "hits_per_event": q(n_hits), "span_ns_32_hit_blocks": q(spans),
           "bins": {f"{w:g}": {"nonempty_bins_per_event": q(bins[w]), "hits_per_nonempty_bin": q(per_bin[w])}
                    for w in WIDTHS}}
    med5 = out["bins"]["5"]["nonempty_bins_per_event"]["median"]
    if med5 >= 6:
        delta, why = 5.0, f"median event has {med5:g} >= 6 non-empty 5 ns bins"
    else:
        med2 = out["bins"]["2"]["nonempty_bins_per_event"]["median"]
        delta = 2.0 if med2 >= 6 else None
        why = f"5 ns gives {med5:g} < 6; 2 ns gives {med2:g}"
    out["rule"] = {"delta_ns": delta, "reason": why}
    return out


def s2_chance(s2_run):
    import torch
    from jepa_noise.lucid_views import ViewMasker
    rec = json.loads((Path(s2_run) / "recipe.json").read_text())["recipe"]
    if rec["data_kind"] != dr.LUCID_V1_W2:
        raise ValueError("expected a lucid_v1_w2 S2 run")
    mask_seed = int(rec["r2_sparse_flags"]["mask_seed"])
    geometry = dr.geometry("wcte", dr.LUCID_V1_W2)
    scaling = rs.scaling_for(geometry, rec["scaling"]["time_scale_ns"])
    pack = dr.pack("wcte", dr.LUCID_V1_W2)
    dr.check_pack_kind(pack, dr.LUCID_V1_W2)
    cohort = rs.open_cohort(pack, "dev_probe_test")
    masker = ViewMasker(geometry, "spatial_patch", seed=lb.DEFAULT_SEED, n_min=0)
    ratios, per_query, n_ev = [], [], 0
    for i in range(cohort.n_runs):
        ids = cohort.run_truth(i)["event_ids"][:8]
        b = rs.tokenise_sparse(cohort.run_store(i), geometry, event_ids=ids, scaling=scaling, masker=masker)
        hit = b.features[..., 8] > 0.5
        n_hits = hit.sum(1)
        for eid, n in zip(b.event_ids.tolist(), n_hits.tolist()):
            n_full = int(n) // sa.S2_BLOCK
            rng = lb.keyed_rng(mask_seed, sa.NS_TIME_BLOCKS, int(eid))
            hidden = rng.permutation(n_full)[:(n_full + 1) // 2] if n_full else []
            if len(hidden):
                n_hidden = sa.S2_BLOCK * len(hidden)
                ratios.extend([sa.S2_BLOCK / n_hidden] * len(hidden))
                per_query.extend([sa.S2_BLOCK / n_hidden] * n_hidden)
        n_ev += len(b.event_ids)
        if n_ev >= 1000:
            break
    del torch
    return {"run": str(s2_run), "cohort": "dev_probe_test", "n_events": n_ev, "mask_seed": mask_seed,
            "chance_mean_over_hidden_blocks": float(np.mean(ratios)),
            "chance_mean_over_queries": float(np.mean(per_query)), "n_hidden_blocks": len(ratios)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-runs", type=int, default=100)
    ap.add_argument("--s2-run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    out = {"w1": measure_train(a.n_runs), "s2_chance": s2_chance(a.s2_run)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out["w1"]["rule"]), json.dumps(out["s2_chance"]))


if __name__ == "__main__":
    main()
