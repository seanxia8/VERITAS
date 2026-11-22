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


def selfcheck(args):
    """G3: mean sampled per-PMT PE (200 events) vs LUCiD's expected-mode response, 3 positions."""
    import os
    import jax
    cfg = mg.load_config(args.config)
    det = args.detector
    geometry, l2r = mg.lucid_geometry(det, cfg)
    kernel = _kernel(args, det)
    spec = mg.SPECS[det]
    cwd = os.getcwd(); os.chdir(mg.LUCID_ROOT)
    try:
        from lucid.simulation import setup_event_simulator
        from lucid.sources.calibration_sources import isotropic_source
        esim = setup_event_simulator(str(kernel.geom_cfg), args.buffer, source="light_source",
                                     transport="expected", detection="expected",
                                     hits="first_arrival", K=12, temperature=0.0,
                                     physics_config=str(kernel.phys_cfg),
                                     default_detector_params=True,
                                     wavelength_mode=bool(spec["wavelength_mode"]))
    finally:
        os.chdir(cwd)
    r_f = mg.rg.fiducial_radius_m(geometry)
    positions = [np.zeros(3), np.array([0.5 * r_f, 0.0, 0.0]), np.array([0.0, 0.9 * r_f, 0.0])]
    y = float(np.sqrt(np.prod(cfg["yield_range_photons"][det])))
    out = {"detector": det, "yield_photons": y, "n_events": args.n, "positions": []}
    for k, pos in enumerate(positions):
        expected = np.zeros(kernel.n_sensors)
        n_exp = 8
        for j in range(n_exp):  # average expected-mode calls (independent wavelength/scatter draws)
            res = esim(isotropic_source(pos, intensity=y), jax.random.PRNGKey(1000 + 10 * k + j))
            expected += np.asarray(res.charge, float) / n_exp
        counts = np.zeros(kernel.n_sensors)
        for i in range(args.n):
            truth = mg.Truth(f"selfcheck/{det}", i, lb.DOMAIN_PHOTON, pos, y, 0.0)
            dep = kernel.simulate(truth, mg.event_seed(mg.DATASET_SEED, "selfcheck", 1000 * k + i, det))
            fin = np.isfinite(dep["t_reco"])
            counts += np.bincount(dep["sensor"][fin], minlength=kernel.n_sensors)
        mean = counts / args.n
        lit = expected > 0
        # Pearson ceiling from photon statistics alone: var(signal) / (var(signal) + mean Poisson var)
        ceiling = float(np.sqrt(np.var(expected) / (np.var(expected) + np.mean(expected) / args.n)))
        # per-PMT consistency: sampled-mean noise e/n plus the expected-mode MC noise e*Y/R
        rays = n_exp * args.buffer
        var = expected / args.n + expected * y / rays
        chi2 = float(np.mean((mean[lit] - expected[lit]) ** 2 / var[lit]))
        out["positions"].append({"position_m": pos.tolist(),
                                 "pearson": float(np.corrcoef(mean, expected)[0, 1]),
                                 "pearson_poisson_ceiling": ceiling,
                                 "chi2_per_pmt": chi2, "expected_mode_rays": int(rays),
                                 "mean_ratio_total": float(mean.sum() / expected.sum()),
                                 "total_sampled_pe": float(mean.sum()),
                                 "total_expected_pe": float(expected.sum()),
                                 "n_lit_expected": int(lit.sum())})
        print(json.dumps(out["positions"][-1]), flush=True)
    p = Path(args.shards).parent / f"selfcheck_{det}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")


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
    for name in ("calibrate", "run", "wcte-manifest", "selfcheck"):
        s = sub.add_parser(name)
        s.add_argument("--detector", choices=sorted(mg.SPECS), default="wcte")
        s.add_argument("--config")
        s.add_argument("--buffer", type=int, default=65536)
        if name == "wcte-manifest":
            s.add_argument("--cache")
        if name == "calibrate":
            s.add_argument("--n", type=int, default=100)
            s.add_argument("--yield-photons", type=float, default=20000.0)
            s.add_argument("--out", required=True)
        else:
            s.add_argument("--shards", required=True)
        if name == "selfcheck":
            s.add_argument("--n", type=int, default=200)
            s.add_argument("--cache")
        if name == "run":
            s.add_argument("--cohort", choices=sorted(mg.COHORTS), required=True)
            s.add_argument("--runs", default="0:1000")
            s.add_argument("--cache", required=True)
            s.add_argument("--limit", type=int)
            s.add_argument("--stride", type=int)
            s.add_argument("--no-s030", action="store_true")
    # r2-tier1 T6: yardsticks with LUCiD's expectation model and time term (tier1_yardstick.py)
    y = sub.add_parser("yardstick")
    y.add_argument("--input", required=True)
    y.add_argument("--out", default="returns/yardstick")
    y.add_argument("--start", type=int, default=0)
    y.add_argument("--stop", type=int, default=10**9)
    y.add_argument("--parts", nargs="+", choices=("y1", "y2"), default=["y1", "y2"])
    y.add_argument("--n-rays", type=int, default=65536)
    y.add_argument("--n-keys", type=int, default=4)
    y.add_argument("--n-rep", type=int, default=32)
    y.add_argument("--maxiter", type=int, default=40)
    y.add_argument("--n-centroid", type=int, default=50)
    y.add_argument("--sigma-tts-ns", type=float, default=None,
                   help="diagnostic: time-term resolution (default: LUCiD working point 2.1 ns)")
    y.add_argument("--shards")  # appended by lucid_job.sh; unused
    y.add_argument("--cache")
    args = p.parse_args()
    if args.cmd == "yardstick":
        from jepa_noise import tier1_yardstick
        return tier1_yardstick.run(args)
    {"calibrate": calibrate, "run": run, "wcte-manifest": wcte_manifest,
     "selfcheck": selfcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
