#!/usr/bin/env python
"""r2-lucid generation CLI (runs in the LUCiD environment).

  calibrate  photon-bomb detection efficiency at random fiducial vertices (yield-range pilot, L2)
  run        LUCiD shards for runs of a cohort (one shard directory per run; resumable, atomic)
  wcte-manifest  top-level manifest of the WCTE training shard directory (round-1 layout)

Probe cohorts are simulated once; the ``_s030`` twin cohort (same event IDs) is written from the
same LUCiD digits with its own detector state, as round 2 does with its keyed draws.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_mc_generator as mg  # noqa: E402


def _kernel(args, det):
    return mg.LucidKernel(det, args.buffer)


def calibrate(args):
    cfg = mg.load_config(args.config)
    det = args.detector
    geometry, l2r = mg.lucid_geometry(det, cfg)
    kernel = _kernel(args, det)
    rows = []
    t0 = time.time()
    for i in range(args.n):
        rng = lb.keyed_rng(mg.DATASET_SEED, "lucid_mc/calibrate", det, i)
        v = mg.rg.fiducial_vertex(geometry, rng, mg.rg.FIDUCIAL_FRACTION)
        truth = mg.Truth(f"calibrate/{det}", i, lb.DOMAIN_PHOTON, v, float(args.yield_photons), 0.0)
        dep = kernel.simulate(truth, mg.event_seed(mg.DATASET_SEED, "calibrate", i, det))
        rows.append({"vertex": v.tolist(), "eff": dep["weight"].size / args.yield_photons})
    wall = time.time() - t0
    eff = np.array([r["eff"] for r in rows])
    out = {"detector": det, "yield_photons": args.yield_photons, "n": args.n,
           "eff_quantiles_5_50_95": [float(np.quantile(eff, q)) for q in (0.05, 0.5, 0.95)],
           "eff_mean": float(eff.mean()), "wall_s": wall,
           "photons_per_s": args.n * args.yield_photons / wall, "kernel": kernel.version,
           "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}), flush=True)


def _provenance(kernel, det):
    import os
    return {"detector": det, "kernel": kernel.version,
            "code_sha": os.environ.get("JEPA_CODE_SHA") or lb.current_code_sha(REPO)}


def run(args):
    cfg = mg.load_config(args.config)
    det = args.detector
    geometry, l2r = mg.lucid_geometry(det, cfg)
    rule = mg.time_rule(det, cfg)
    shard_root = Path(args.shards)
    cache_root = Path(args.cache)
    kernel = None
    runs = mg.cohort_runs(args.cohort)
    lo, hi = (int(x) for x in args.runs.split(":"))
    twin_cohorts = [args.cohort]
    if args.cohort in ("dev_probe_train", "dev_probe_test") and not args.no_s030:
        twin_cohorts.append(args.cohort + "_s030")
    for i in range(lo, min(hi, len(runs))):
        targets = []
        for c in twin_cohorts:
            r = mg.cohort_runs(c)[i]
            d = shard_root / c / r.run_id.replace("/", "__")
            if not d.exists():
                targets.append((c, r, d))
        if not targets:
            print(f"skip {runs[i].run_id}: shards exist", flush=True)
            continue
        cache_file = cache_root / args.cohort / (runs[i].run_id.replace("/", "__") + ".pkl")
        t0 = time.time()
        if cache_file.exists():
            cache = pickle.loads(cache_file.read_bytes())
        else:
            if kernel is None:
                kernel = _kernel(args, det)
            select = None if args.stride is None else np.arange(0, runs[i].n_events, args.stride)
            cache = mg.simulate_run_digits(kernel, geometry, l2r, cfg, det, runs[i], limit=args.limit,
                                           select=select)
            cache["provenance"] = _provenance(kernel, det)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_file.with_suffix(".pkl.tmp")
            tmp.write_bytes(pickle.dumps(cache))
            tmp.rename(cache_file)
        sim_s = time.time() - t0
        for c, r, d in targets:
            lrun = mg.run_from_cache(cache, r, geometry, cfg, rule, with_twin=mg.cohort_has_twin(c))
            m = mg.write_run(d, lrun, geometry, cfg, rule, cache["provenance"])
            print(json.dumps({"cohort": c, "run": r.run_id, "events": m["n_events"],
                              "hits": m["n_hits"], "sim_s": round(sim_s, 1),
                              "regime": m["regime"], "no_trigger": m["n_no_trigger"]}), flush=True)


def wcte_manifest(args):
    """Top-level manifest for the WCTE training launcher (round-1 shard-directory layout)."""
    root = Path(args.shards) / "dev_train"
    runs = mg.cohort_runs("dev_train")
    rows = []
    for r in runs:
        d = root / r.run_id.replace("/", "__")
        m = json.loads((d / "manifest.json").read_text())
        row = dict(m["runs"][0])
        row.update(run_dir=d.name, manifest_sha256=lb._sha((d / "manifest.json").read_bytes()))
        rows.append(row)
    first = json.loads((root / rows[0]["run_dir"] / "manifest.json").read_text())
    manifest = {"schema": "jepa-lucid-dataset-v1", "dataset_kind": mg.DATASET_KIND,
                "sampling_tier": mg.SAMPLING_TIER, "cohort": "dev_train", "law": "F1",
                "setting": 0.05, "n_events": int(sum(r["n_events"] for r in rows)),
                "n_runs": len(rows), "run_length": 1000, "dataset_seed": mg.DATASET_SEED,
                "registered": False, "reported_results": False,
                "geometry_sha256": first["geometry_sha256"], "lucid": first["lucid"],
                "readout_label": first["readout_label"], "runs": rows}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    lines = [f"{lb._sha((root / r['run_dir'] / 'manifest.json').read_bytes())}  runs/{r['run_dir']}/manifest.json"
             for r in rows]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    print(f"wrote {root / 'manifest.json'} ({manifest['n_events']} events)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("calibrate", "run", "wcte-manifest"):
        s = sub.add_parser(name)
        s.add_argument("--detector", choices=sorted(mg.SPECS), default="wcte")
        s.add_argument("--config")
        s.add_argument("--buffer", type=int, default=65536)
        if name == "calibrate":
            s.add_argument("--n", type=int, default=100)
            s.add_argument("--yield-photons", type=float, default=20000.0)
            s.add_argument("--out", required=True)
        else:
            s.add_argument("--shards", required=True)
        if name == "run":
            s.add_argument("--cohort", choices=sorted(mg.COHORTS), required=True)
            s.add_argument("--runs", default="0:1000")
            s.add_argument("--cache", required=True)
            s.add_argument("--limit", type=int)
            s.add_argument("--stride", type=int)
            s.add_argument("--no-s030", action="store_true")
    args = p.parse_args()
    {"calibrate": calibrate, "run": run, "wcte-manifest": wcte_manifest}[args.cmd](args)


if __name__ == "__main__":
    main()
