#!/usr/bin/env python3
"""r2-tier1 T6: extract yardstick events (dev_probe_test photon bombs) for the LUCiD-env fits.

Writes ``yardstick_events_<det>.npz`` (group storage only; LUCiD-generated data are never committed):
per event the truth (vertex, yield, source time t0 on the absolute clock = emission offset), the
observed prompt-window hits exactly as the packs hold them (round-2 PMT index, charge) but on the
absolute clock (trigger-relative time + recomputed NHits trigger), the LUCiD sensor index of each hit,
and per detector the prompt window, the LUCiD dark rate and the charge-weighted centroid start.
The first ``n_y1`` events (keyed choice) are Y1; the first ``n_y2`` of those are Y2.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_mc_generator as mg  # noqa: E402

CACHE = Path("/ceph/dwong/jepa/data/lucid_v1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=("sk", "wcte", "juno", "hk"))
    ap.add_argument("--n-y1", type=int, default=300)
    ap.add_argument("--n-y2", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    det = a.detector
    cfg = mg.load_config()
    geometry, l2r = mg.lucid_geometry(det, cfg)
    r2l = np.empty_like(l2r)
    r2l[l2r] = np.arange(l2r.size)
    rule = mg.time_rule(det, cfg)
    runs = mg.cohort_runs("dev_probe_test")
    rng = lb.keyed_rng(mg.DATASET_SEED, "r2-tier1/yardstick", det)
    order = rng.permutation(len(runs))
    events = []
    for ri in order:
        run = runs[int(ri)]
        cache = pickle.loads((CACHE / det / "cache" / "dev_probe_test" /
                              (run.run_id.replace("/", "__") + ".pkl")).read_bytes())
        lrun = mg.run_from_cache(cache, run, geometry, cfg, rule, with_twin=False)
        photon = [k for k, t in enumerate(lrun.truths) if t.domain == lb.DOMAIN_PHOTON]
        per_run = -(-a.n_y1 // len(runs))  # two keyed photon bombs per run for 300 of 200 runs
        for k in rng.choice(photon, size=min(per_run, len(photon)), replace=False):
            k = int(k)
            t, h = lrun.truths[k], lrun.hits[k]
            if h.trigger_ns is None or h.pmt_id.size < 5 or len(events) >= a.n_y1:
                continue
            events.append({"event_id": int(lrun.event_ids[k]), "run_id": run.run_id,
                           "vertex": np.asarray(t.vertex_m, float), "yield": float(t.yield_photons),
                           "t0": float(t.emission_offset_ns), "trigger_ns": float(h.trigger_ns),
                           "pmt_r2": np.asarray(h.pmt_id, np.int64),
                           "pmt_lucid": r2l[np.asarray(h.pmt_id, np.int64)],
                           "q": np.asarray(h.charge_pe, float), "t_abs": np.asarray(h.absolute_time_ns, float)})
        if len(events) >= a.n_y1:
            break
    pos = geometry.positions_m
    flat = {"event_id": np.array([e["event_id"] for e in events]),
            "vertex": np.stack([e["vertex"] for e in events]),
            "yield": np.array([e["yield"] for e in events]),
            "t0": np.array([e["t0"] for e in events]),
            "trigger_ns": np.array([e["trigger_ns"] for e in events]),
            "offsets": np.cumsum([0] + [e["pmt_r2"].size for e in events]),
            "pmt_r2": np.concatenate([e["pmt_r2"] for e in events]),
            "pmt_lucid": np.concatenate([e["pmt_lucid"] for e in events]),
            "q": np.concatenate([e["q"] for e in events]),
            "t_abs": np.concatenate([e["t_abs"] for e in events]),
            "centroid": np.stack([(pos[e["pmt_r2"]] * e["q"][:, None]).sum(0) / e["q"].sum() for e in events]),
            "positions_lucid_order": pos[l2r],
            "run_ids": np.array([e["run_id"] for e in events])}
    meta = {"detector": det, "n_y1": len(events), "n_y2": min(a.n_y2, len(events)),
            "prompt_window_ns": rule["prompt_window_ns"], "time_scale_ns": rule["time_scale_ns"],
            "n_pmt": int(geometry.n_pmt), "radius_m": float(geometry.radius_m),
            "half_height_m": float(geometry.half_height_m), "cohort": "dev_probe_test",
            "dark_rate_khz": 4.2, "selection": "two keyed photon bombs per run, keyed run order; Y2 = the first n_y2"}
    np.savez_compressed(a.out, meta=json.dumps(meta), **flat)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
