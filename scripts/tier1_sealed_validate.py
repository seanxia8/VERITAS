#!/usr/bin/env python3
"""r2-tier1 T2.2: validate a sealed cohort against dev_probe_test (generation QA; no model, no readout).

G2 timing-residual peak and late-light fraction, and G4 regime quantiles, computed with the r2-lucid
gate code (``scripts/lucid_validate.py``) on the cached LUCiD digits of both cohorts, must agree
within 10 % (the residual peak, which sits near 0 ns, within 10 % or one 0.2 ns histogram bin);
G6 (no analytic physics in the generation path) is the static/runtime test of the generator; event
IDs and run keys must be disjoint from every existing cohort.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import lucid_validate as lv  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_mc_generator as mg  # noqa: E402

DATA = Path("/ceph/dwong/jepa/data/lucid_v1")


def caches(det, cohort, n_runs):
    files = sorted((DATA / det / "cache" / cohort).glob("*.pkl"))
    step = max(1, len(files) // n_runs)
    return [pickle.loads(f.read_bytes()) for f in files[::step][:n_runs]], len(files)


def close(a, b, tol=0.10):
    return bool(abs(a - b) <= tol * max(abs(b), 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True)
    ap.add_argument("--n-runs", type=int, default=60)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    det = a.detector
    cfg = mg.load_config()
    geometry, _ = mg.lucid_geometry(det, cfg)
    rule = mg.time_rule(det, cfg)
    ref, n_ref = caches(det, "dev_probe_test", a.n_runs)
    sea, n_sea = caches(det, "sealed_test", 3 * a.n_runs)
    digitizer = json.loads(json.dumps(ref[0]["provenance"]["kernel"]["digitizer"]))
    g2r, _ = lv.g2(det, ref, geometry, cfg, digitizer)
    g2s, _ = lv.g2(det, sea, geometry, cfg, digitizer)
    g4r, g4s = lv.g4(det, ref, cfg, rule), lv.g4(det, sea, cfg, rule)
    checks = {
        "G2_residual_peak": abs(g2s["residual_peak_ns"] - g2r["residual_peak_ns"]) <= max(0.2, 0.1 * abs(g2r["residual_peak_ns"])),
        "G2_late_hit_fraction": close(g2s["late_hit_fraction_gt10ns_first_hits"], g2r["late_hit_fraction_gt10ns_first_hits"]),
        "G2_late_deposit_fraction": close(g2s["late_deposit_fraction_gt10ns_mean"], g2r["late_deposit_fraction_gt10ns_mean"]),
        "G4_true_pe_q_photon": all(close(x, y) for x, y in zip(g4s["true_pe_q_photon"], g4r["true_pe_q_photon"])),
        "G4_true_pe_q_all": all(close(x, y) for x, y in zip(g4s["true_pe_q_all"], g4r["true_pe_q_all"])),
    }
    # disjointness: event IDs and run keys versus every registered cohort
    sealed_runs = mg.cohort_runs("sealed_test")
    others = [r for c in lb.COHORTS.values() for r in lb.make_runs(c, "F1", 0.05)]
    s_ids = set()
    for f in sorted((DATA / det / "cache" / "sealed_test").glob("*.pkl")):
        s_ids.update(int(e) for e in pickle.loads(f.read_bytes())["event_ids"])
    overlap_ids = sum(1 for r in others for e in (r.event_start, r.event_stop - 1)
                      if min(s_ids) <= e <= max(s_ids))
    keys_overlap = len({r.run_id for r in sealed_runs} & {r.run_id for r in others})
    checks["disjoint_event_ids"] = overlap_ids == 0 and min(s_ids) >= 1_300_000 and max(s_ids) < 1_330_000
    checks["disjoint_run_keys"] = keys_overlap == 0 and all(r.event_namespace == "lucid/sealed_test" for r in sealed_runs)
    checks["complete_600_runs"] = n_sea == 600 and len(s_ids) == 30_000
    res = {"detector": det, "reference": "dev_probe_test", "n_runs_reference": len(ref), "n_runs_sealed_sampled": len(sea),
           "n_sealed_cache_files": n_sea, "n_sealed_event_ids": len(s_ids),
           "G2": {"reference": g2r, "sealed": g2s}, "G4": {"reference": g4r, "sealed": g4s},
           "G6": {"test": "src/jepa_noise/tests/test_lucid_mc_generator.py::test_generation_path_imports_no_toy_physics",
                  "forbidden": list(mg.FORBIDDEN_LB_NAMES)},
           "checks": checks, "pass": bool(all(checks.values()))}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps({"detector": det, "checks": checks, "pass": res["pass"]}))


if __name__ == "__main__":
    main()
