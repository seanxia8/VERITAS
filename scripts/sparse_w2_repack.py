#!/usr/bin/env python
"""r2-sparse W2 step 1-3: re-pack WCTE cohorts from the cached LUCiD digits as ``lucid_v1_w2``.

Runs in the LUCiD environment (the dark-noise extension calls LUCiD's digitizer). For each run of a
cohort, in run order: read the read-only ``lucid_v1`` cache pickle, extend the dark noise over the
uncovered parts of the new window (LUCiD's own chain), apply the run's detector state and the
VERITAS readout with the new prompt window (``lucid_w2.veritas_readout_w2``), and write a shard in
the ``lucid_v1`` layout under ``/ceph/dwong/jepa/data/lucid_v1_w2/wcte/shards/<cohort>/``. Existing
shards are skipped, never rewritten. Toy physics is booby-trapped for the whole run (gate G6, run
time) and ``r2_generator`` must never be imported.

    sparse_w2_repack.py --cohort dev_train --runs 0:17 --pre-ns 110 --post-ns 50
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_mc_generator as mg  # noqa: E402
from jepa_noise import lucid_w2 as w2  # noqa: E402

COHORTS = ("dev_train", "dev_probe_train", "dev_probe_test", "sealed_test")
SRC = Path("/ceph/dwong/jepa/data/lucid_v1/wcte/cache")


class ToyPhysicsCalled(RuntimeError):
    pass


def _trap(name):
    def fail(*_a, **_k):
        raise ToyPhysicsCalled(f"G6: toy physics lucid_bombs.{name} called in the W2 path")
    return fail


def arm_g6_trap():
    trapped = []
    for name in mg.FORBIDDEN_LB_NAMES:
        if hasattr(lb, name):
            setattr(lb, name, _trap(name))
            trapped.append(name)
    return trapped


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", choices=COHORTS, required=True)
    ap.add_argument("--runs", default="0:100000")
    ap.add_argument("--pre-ns", type=float, required=True)
    ap.add_argument("--post-ns", type=float, required=True)
    ap.add_argument("--out", type=Path, default=w2.W2_ROOT / "wcte" / "shards")
    ap.add_argument("--access-log", type=Path, default=None,
                    help="sealed_test: append one line (packing only, no metrics) before reading")
    a = ap.parse_args()
    trapped = arm_g6_trap()
    det = "wcte"
    cfg = mg.load_config()
    geometry, _ = mg.lucid_geometry(det, cfg)
    rule = w2.window_rule(mg.time_rule(det, cfg), a.pre_ns, a.post_ns)
    pad = float(cfg["dark_pad_no_trigger_ns"])
    ref = lb.generator_reference(mg.toy_generator_config())
    width = lb.fixed_timing_width_factor(geometry, ref)
    fixed_gain = lb.fixed_charge_gain(geometry, ref)
    runs = mg.cohort_runs(a.cohort)
    lo, hi = (int(x) for x in a.runs.split(":"))
    with_twin = mg.cohort_has_twin(a.cohort)
    code_sha = os.environ.get("JEPA_CODE_SHA") or lb.current_code_sha(REPO)
    if a.cohort == "sealed_test":
        if a.access_log is None:
            raise SystemExit("sealed_test re-pack needs --access-log")
        from datetime import datetime, timezone
        import fcntl
        a.access_log.parent.mkdir(parents=True, exist_ok=True)
        with a.access_log.open("a") as h:
            fcntl.flock(h, fcntl.LOCK_EX)
            h.write(f"- {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} | {code_sha[:12]} | wcte sealed_test "
                    f"runs {lo}:{min(hi, len(runs))} | W2 re-pack from the lucid_v1 LUCiD cache, window "
                    f"[-{a.pre_ns:g}, +{a.post_ns:g}] ns (packing only, no metrics) | scripts/sparse_w2_repack.py\n")
    digitizer_checked = False
    for i in range(lo, min(hi, len(runs))):
        run = runs[i]
        d = a.out / a.cohort / run.run_id.replace("/", "__")
        if d.exists():
            print(f"skip {run.run_id}: shard exists", flush=True)
            continue
        t0 = time.time()
        cache = pickle.loads((SRC / a.cohort / (run.run_id.replace("/", "__") + ".pkl")).read_bytes())
        digitizer = cache["provenance"]["kernel"]["digitizer"]
        if not digitizer_checked:
            from lucid.simulation.digitizer import resolve_model_config
            ref_dig = resolve_model_config(mg.SPECS[det]["digitizer"])
            if json.dumps(ref_dig, sort_keys=True, default=str) != json.dumps(digitizer, sort_keys=True, default=str):
                raise RuntimeError("cached digitizer differs from LUCiD's resolved model config")
            if cache["provenance"]["kernel"]["lucid_sha"] != mg.LUCID_SHA:
                raise RuntimeError("cache LUCiD SHA mismatch")
            digitizer_checked = True
        if len(cache["event_ids"]) != run.n_events:
            raise RuntimeError(f"{run.run_id}: cache has {len(cache['event_ids'])} of {run.n_events} events")
        state = run.state(geometry)
        gains = fixed_gain * np.asarray(state.gains, float)
        record = {"n_events": 0, "n_no_trigger": 0, "n_prompt_dropped": 0, "n_dark_moved_trigger": 0}
        readouts, twins, ext_stats = [], ([] if with_twin else None), []
        for eid, dig in zip(cache["event_ids"], cache["digits"]):
            rng = np.random.default_rng(mg.event_seed(mg.DATASET_SEED, run.run_id, int(eid), "dark_ext"))
            ext_dig, info = w2.dark_extension(dig, rule, digitizer, width, geometry.n_pmt, rng, pad)
            r = w2.veritas_readout_w2(ext_dig, geometry.n_pmt, gains, rule, cfg["readout_start_ns"], record)
            tc = info["trigger_cached"]
            info["trigger_changed"] = int((tc is None) != (r.hits.trigger_ns is None) or
                                          (tc is not None and abs(tc - r.hits.trigger_ns) > 1e-9))
            readouts.append(r)
            ext_stats.append(info)
            if with_twin:
                twins.append(w2.veritas_readout_w2(ext_dig, geometry.n_pmt, fixed_gain, rule, cfg["readout_start_ns"]))
        prov = dict(cache["provenance"], repack_code_sha=code_sha)
        m = w2.write_run_w2(d, run, np.asarray(cache["event_ids"], np.int64), cache["truths"], cache["stats"],
                            readouts, twins, state.digest, geometry, cfg, rule, prov, record, ext_stats)
        if "jepa_noise.r2_generator" in sys.modules:
            raise RuntimeError("G6: r2_generator was imported")
        print(json.dumps({"cohort": a.cohort, "run": run.run_id, "events": m["n_events"], "hits": m["n_hits"],
                          "dark_extension": m["dark_extension"], "no_trigger": m["n_no_trigger"],
                          "s": round(time.time() - t0, 1), "g6_trapped": len(trapped)}), flush=True)


if __name__ == "__main__":
    main()
