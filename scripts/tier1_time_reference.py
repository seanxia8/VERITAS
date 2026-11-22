#!/usr/bin/env python3
"""r2-tier1 T5.1: common-shift share of the token time on dev_probe_train photon bombs.

V_tot   = variance of the time feature tau = t / T_scale over all prompt hits of all photon events;
V_shift = variance across events of the per-event median prompt-hit tau.
Also the SD of (NHits trigger time - earliest physics digit time), recomputed exactly from LUCiD's
cached digits (``/ceph/dwong/jepa/data/lucid_v1/<det>/cache/dev_probe_train/*.pkl``) with the same
``veritas_readout`` the packs were written with (the packs store only trigger-relative times).
Read-only on all inputs; writes one JSON.
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

from jepa_noise import data_roots as dr  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_mc_generator as mg  # noqa: E402
from jepa_noise import r2_store as rs  # noqa: E402
from jepa_noise.ext_r2_repair_train import geometry_and_scaling  # noqa: E402


def share(detector):
    geometry, scaling = geometry_and_scaling(detector, "lucid_v1")
    T = float(scaling.time_scale_ns)
    cohort = rs.open_cohort(dr.pack(detector, "lucid_v1"), "dev_probe_train")
    taus, medians = [], []
    for i in range(cohort.n_runs):
        truth, store = cohort.run_truth(i), cohort.run_store(i)
        for k in np.flatnonzero(np.asarray(truth["domain"]).astype(str) == lb.DOMAIN_PHOTON):
            t = np.asarray(store.time_ns[store.offsets[k]:store.offsets[k + 1]], float) / T
            if t.size:
                taus.append(t)
                medians.append(float(np.median(t)))
    allt = np.concatenate(taus)
    v_tot, v_shift = float(allt.var()), float(np.var(medians))
    # trigger - earliest physics digit, from the cached LUCiD digits
    cfg = mg.load_config()
    rule = mg.time_rule(detector, cfg)
    n, w = int(rule["n"]), float(rule["trigger_window_ns"])
    from noise_module_lucid.readout import nhits_trigger_time
    offsets = []
    cache_dir = mg.DATA_ROOT / detector / "cache" / "dev_probe_train" if hasattr(mg, "DATA_ROOT") else \
        Path("/ceph/dwong/jepa/data/lucid_v1") / detector / "cache" / "dev_probe_train"
    for pkl in sorted(cache_dir.glob("*.pkl")):
        cache = pickle.loads(pkl.read_bytes())
        for truth, dig in zip(cache["truths"], cache["digits"]):
            dom = truth.domain if hasattr(truth, "domain") else truth["domain"]
            if dom != lb.DOMAIN_PHOTON:
                continue
            s, t, phys = np.asarray(dig["sensor"]), np.asarray(dig["t"], float), np.asarray(dig["physics"], bool)
            order = np.lexsort((t, s))
            s, t, phys = s[order], t[order], phys[order]
            first = np.ones(s.size, bool)
            first[1:] = s[1:] != s[:-1]
            t1, p1 = t[first], phys[first]
            trig = nhits_trigger_time(t1, n=n, window_ns=w)
            if trig is None or not p1.any():
                continue
            offsets.append(float(trig) - float(t1[p1].min()))
    off = np.asarray(offsets)
    return {"detector": detector, "time_scale_ns": T, "n_events": len(medians), "n_hits": int(allt.size),
            "V_tot": v_tot, "V_shift": v_shift, "share": v_shift / v_tot,
            "median_tau_sd_ns": float(np.std(medians) * T), "tau_sd_ns": float(allt.std() * T),
            "trigger_minus_first_physics_ns": {"n": int(off.size), "mean": float(off.mean()),
                                               "sd": float(off.std()),
                                               "q05_q50_q95": [float(np.quantile(off, q)) for q in (0.05, 0.5, 0.95)]}
            if off.size else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detectors", nargs="+", default=["wcte", "sk", "juno", "hk"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"cohort": "dev_probe_train", "domain": "photon", "rule_threshold": 0.10, "detectors": {}}
    for det in a.detectors:
        out["detectors"][det] = share(det)
        print(json.dumps({k: v for k, v in out["detectors"][det].items() if k != "detector"}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
