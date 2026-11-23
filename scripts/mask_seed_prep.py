#!/usr/bin/env python3
"""r2-mask M3.2: starting points for the readout-seeded LUCiD fit (torch env, validation only).

For the Y2 events of the r2-tier1 yardstick input (the first ``n_y2`` = 200 ``dev_probe_test``
photon bombs per detector) this writes, keyed by event id:

* ``start_truth``    (a) the true vertex;
* ``start_centroid`` (b) the charge-weighted centroid of the hit PMTs (from the yardstick input);
* ``start_readout``  (c) the frozen spatial-moment Ridge readout (READOUT_SPEC_v1, A.2 + Ridge
  protocol, ``tier1_eval.ridge_fit``) of ``<det>__mae_repaired__s9000`` at step 10,000, fitted on
  ``dev_probe_train`` photon bombs;
* ``start_init``     (d) the same readout on that run's matched init (step-0 snapshot);
* ``start_raw``      (e) the raw-feature baseline (A.3, ``raw_spatial_moment``) Ridge;
* ``lnN_start``      ln N from a linear fit of ln N_gamma on ln(total charge) over the
  ``dev_probe_train`` photon bombs (total charge from the pack, the same definition for test events).

The t0 start (median time residual at the starting vertex) needs the absolute clock and is computed
inside the LUCiD job. No sealed data are read (``dev_probe_train`` and ``dev_probe_test`` only).

Usage: mask_seed_prep.py --detector sk --run-dir <runs>/sk__mae_repaired__s9000 \
           --yardstick <yardstick_events_sk.npz> --out <starts_sk.npz> [--device cpu]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))


def _tier1_eval():
    spec = importlib.util.spec_from_file_location("tier1_eval", REPO / "scripts" / "tier1_eval.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def readout_predictions(te, run_dir, step, event_ids, device, chunk):
    """Ridge predictions (spatial_moment_latent and raw_spatial_moment) for the given test events."""
    import torch
    from jepa_noise import lucid_bombs as lb
    from jepa_noise import r2_store as rs
    ns = argparse.Namespace(run_dir=run_dir, step=step, legacy_init=False, detector=None, seed=None)
    arm, geometry, scaling, meta = te.load_model(ns)
    arm = arm.to(torch.device(device)).eval()
    pack = te.cohort_pack(meta["detector"], "validation")
    data = {}
    for which, name in (("train", te.TRAIN_COHORT), ("test", te.COHORTS["validation"])):
        cohort = rs.open_cohort(pack, name)
        data[which] = te.stream_cohort(arm, geometry, scaling, cohort, device=torch.device(device),
                                       chunk=chunk, photons_only=True, attentive=False, layout=False,
                                       time_reference=meta.get("time_reference", "trigger"))
    tr, ts = data["train"], data["test"]
    ph_tr = tr["domain"] == lb.DOMAIN_PHOTON
    pos = {int(e): i for i, e in enumerate(ts["event_ids"])}
    rows = np.array([pos[int(e)] for e in event_ids])
    out = {}
    for form in ("spatial_moment_latent", "raw_spatial_moment"):
        fitted = te.ridge_fit(tr[form][ph_tr], tr["vertex"][ph_tr], tr["run_ids"][ph_tr], ("x", "y", "z"))
        out[form] = te.ridge_predict(fitted, ts[form][rows], 3)
        out[f"{form}_alpha"] = [f.get("alpha") for f in fitted["fits"]]
    out["vertex_check"] = ts["vertex"][rows]
    out["meta"] = {k: meta[k] for k in ("run_id", "step") if k in meta}
    return out


def total_charge(cohort):
    """{event_id: total charge (PE)} and {event_id: yield} for the photon bombs of a cohort."""
    from jepa_noise import lucid_bombs as lb
    q, n = {}, {}
    for i in range(len(cohort.run_ids)):
        truth = cohort.run_truth(i)
        store = cohort.run_store(i)
        dom = truth["domain"].astype(str)
        for k in np.flatnonzero(dom == lb.DOMAIN_PHOTON):
            e = int(store.event_ids[k])
            span = slice(int(store.offsets[k]), int(store.offsets[k + 1]))
            q[e] = float(np.sum(store.charge_pe[span]))
            if "yield_photons" in truth:
                n[e] = float(truth["yield_photons"][k])
    return q, n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=("sk", "wcte"))
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--yardstick", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args(argv)
    from jepa_noise import data_roots as dr
    from jepa_noise import r2_store as rs
    te = _tier1_eval()
    y = np.load(a.yardstick, allow_pickle=False)
    meta = json.loads(str(y["meta"]))
    if meta["detector"] != a.detector:
        raise SystemExit("yardstick input is for another detector")
    n = int(meta["n_y2"])
    event_ids = np.asarray(y["event_id"][:n], np.int64)
    final = readout_predictions(te, a.run_dir, 10000, event_ids, a.device, a.chunk)
    init = readout_predictions(te, a.run_dir, 0, event_ids, a.device, a.chunk)
    if not np.allclose(final["vertex_check"], y["vertex"][:n], atol=1e-4):
        raise SystemExit("vertex frames of the pack and the yardstick input differ")
    pack = dr.pack(a.detector, dr.LUCID_V1) if a.detector != "wcte" else dr.wcte_probe_pack(dr.LUCID_V1)
    tr_cohort = rs.open_cohort(pack, "dev_probe_train", truth_fields=("domain", "yield_photons"))
    te_cohort = rs.open_cohort(pack, "dev_probe_test", truth_fields=("domain", "yield_photons"))
    q_tr, n_tr = total_charge(tr_cohort)
    q_te, _ = total_charge(te_cohort)
    keys = sorted(q_tr)
    lq = np.log([max(q_tr[k], 1e-6) for k in keys])
    ln_n = np.log([n_tr[k] for k in keys])
    slope, intercept = np.polyfit(lq, ln_n, 1)
    q_test = np.array([q_te[int(e)] for e in event_ids])
    ln_n_start = intercept + slope * np.log(np.maximum(q_test, 1e-6))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    info = {"detector": a.detector, "run_dir": str(a.run_dir), "n_events": n,
            "readout": "READOUT_SPEC_v1 spatial_moment_latent Ridge (tier1_eval.ridge_fit)",
            "alpha_readout": final["spatial_moment_latent_alpha"],
            "alpha_init": init["spatial_moment_latent_alpha"],
            "alpha_raw": final["raw_spatial_moment_alpha"],
            "lnN_fit": {"slope": float(slope), "intercept": float(intercept), "n_train": len(keys),
                        "rms_train": float(np.std(ln_n - (intercept + slope * lq)))},
            "mean_err_m": {}}
    starts = {"start_truth": y["vertex"][:n], "start_centroid": y["centroid"][:n],
              "start_readout": final["spatial_moment_latent"], "start_init": init["spatial_moment_latent"],
              "start_raw": final["raw_spatial_moment"]}
    for k, v in starts.items():
        info["mean_err_m"][k] = float(np.linalg.norm(v - y["vertex"][:n], axis=1).mean())
    np.savez(a.out, event_id=event_ids, lnN_start=ln_n_start, meta=json.dumps(info), **starts)
    print(json.dumps(info, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
