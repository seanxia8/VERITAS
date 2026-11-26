#!/usr/bin/env python3
"""r2-sparse W2 steps 3 and 5: pack the ``lucid_v1_w2`` WCTE shards and validate them against ``lucid_v1``.

Base env (r2_store imports torch). Writes, never overwriting:

* ``/ceph/dwong/jepa/data/lucid_v1_w2/wcte/wcte_lucid_v1_w2.h5`` (dev_probe_train, dev_probe_test),
* ``.../wcte_lucid_v1_w2_sealed.h5`` (sealed_test; access logged, packing and structural checks only),
* ``.../wcte_train/manifest.json`` (+ ``runs`` -> ``../shards/dev_train``), the training directory,
* ``DATA_MANIFEST_wcte_w2.json``, ``W2_VALIDATION.json`` and ``SHA256SUMS``.

Each pack also carries the extra per-hit fields (``extra_*``) next to the standard hit arrays.

    sparse_w2_pack.py pack --out-manifest results/round2/20261124_sparse --access-log <log>
    sparse_w2_pack.py validate --cohort dev_probe_train --out <json> [--access-log <log>]

r2-sparse-sk J0: ``--detector juno`` (before the subcommand) packs and validates the JUNO re-pack the same
way. JUNO keeps its ``lucid_v1`` layout: ``juno_lucid_v1_w2.h5`` holds dev_train, dev_probe_train and
dev_probe_test (training reads dev_train from the pack), ``juno_lucid_v1_w2_sealed.h5`` the sealed_test;
no training-shard directory. v1 hits beyond a narrower v2 window end are counted apart (WCTE only).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_mc_generator as mg  # noqa: E402
from jepa_noise import lucid_w2 as w2  # noqa: E402

DET = "wcte"
V1 = Path("/ceph/dwong/jepa/data/lucid_v1/wcte")
V2 = w2.W2_ROOT / "wcte"


def set_detector(det):
    """r2-sparse-sk J0: point the module at one detector (default WCTE, unchanged)."""
    global DET, V1, V2, COMMON_HI_NS
    DET = det
    V1 = Path("/ceph/dwong/jepa/data/lucid_v1") / det
    V2 = w2.W2_ROOT / det
    if det != "wcte":
        COMMON_HI_NS = None   # set from the v1 / v2 windows in validate()
PROBE = ("dev_probe_train", "dev_probe_test")
ALL = ("dev_train",) + PROBE + ("sealed_test",)
COMMON_HI_NS = 50.0
TRIGGER_ZONE = (-20.0, 10.0)
OLD_WINDOW = (-20.0, 60.0)


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_access(path, what):
    import fcntl
    if path is None:
        raise SystemExit("a sealed_test read needs --access-log")
    sha = os.environ.get("JEPA_CODE_SHA") or lb.current_code_sha(REPO)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as h:
        fcntl.flock(h, fcntl.LOCK_EX)
        h.write(f"- {utc()} | {sha[:12]} | {DET} sealed_test | {what} | scripts/sparse_w2_pack.py\n")


def run_dir(root, cohort, run):
    return root / ("shards" if root == V1 else w2.SHARD_DIR) / cohort / run.run_id.replace("/", "__")


def complete(cohort):
    return all((run_dir(V2, cohort, r) / "manifest.json").is_file() for r in mg.cohort_runs(cohort))


def add_extras(h5, cohort, runs):
    import h5py
    from jepa_noise.r2_store import COMPRESSION
    parts = {k: [] for k in w2.EXTRA_FIELDS}
    counts = []
    for r in runs:
        with np.load(run_dir(V2, cohort, r) / "hits_extra.npz") as e:
            for k in w2.EXTRA_FIELDS:
                parts[k].append(e[k])
            counts.append(int(e["offsets"][-1]))
    with h5py.File(h5, "a") as f:
        g = f[cohort]
        n_hits = int(g["pmt_id"].shape[0])
        if sum(counts) != n_hits:
            raise ValueError(f"{cohort}: extras have {sum(counts)} rows for {n_hits} hits")
        for k in w2.EXTRA_FIELDS:
            g.create_dataset("extra_" + k, data=np.concatenate(parts[k]), **COMPRESSION)


def pack(a):
    from jepa_noise import r2_store as rs
    cfg = mg.load_config()
    geometry, _ = mg.lucid_geometry(DET, cfg)
    rule = json.loads((run_dir(V2, "dev_probe_train", mg.cohort_runs("dev_probe_train")[0]) / "manifest.json")
                      .read_text())["time_rule"]
    missing = [c for c in ALL if not complete(c)]
    if missing:
        raise SystemExit(f"incomplete cohorts {missing}")
    ext = "r2_sparse_2026-11-24" if DET == "wcte" else "r2_sparse_sk_2026-11-26"
    attrs = {"detector": DET, "extension": ext, "dataset_kind": w2.DATASET_KIND_W2,
             "source_dataset_kind": mg.DATASET_KIND, "sampling_tier": mg.SAMPLING_TIER,
             "lucid_sha": mg.LUCID_SHA, "lucid_v1_config_sha256": cfg["_sha256"],
             "geometry_sha256": geometry.digest, "time_rule": rule, "veritas_steps": list(mg.VERITAS_STEPS),
             "registered": False, "extra_fields": list(w2.EXTRA_FIELDS)}
    files, summary = {}, {}
    t0 = time.time()
    main_cohorts = PROBE if DET == "wcte" else ("dev_train",) + PROBE
    for name, cohorts, sealed in ((f"{DET}_lucid_v1_w2.h5", main_cohorts, False),
                                  (f"{DET}_lucid_v1_w2_sealed.h5", ("sealed_test",), True)):
        h5 = V2 / name
        if h5.exists():
            raise SystemExit(f"{h5} exists; never rewritten")
        if sealed:
            log_access(a.access_log, f"pack lucid_v1_w2 sealed shards into {DET}_lucid_v1_w2_sealed.h5 "
                                     "(packing + byte verification only, no metrics)")
        spec = {c: list(mg.cohort_runs(c)) for c in cohorts}
        tmp = h5.with_name(h5.name + ".building")
        if tmp.exists():
            tmp.unlink()
        summary[name] = rs.pack_detector(tmp, V2 / w2.SHARD_DIR, spec,
                                         attrs=dict(attrs, sealed=sealed, cohorts=list(cohorts)))
        for c in cohorts:
            add_extras(tmp, c, spec[c])
        verified = rs.verify_pack(tmp, V2 / w2.SHARD_DIR, spec)
        os.rename(tmp, h5)
        os.chmod(h5, 0o444)
        files[name] = {"path": str(h5), "sha256": rs.sha256_file(h5), "bytes": h5.stat().st_size,
                       "verify": verified["verified_runs"]}
    if DET != "wcte":
        return _finish_manifest(a, cfg, rule, files, summary, t0)
    # training directory (round-1 layout read by ext_r2_repair_train.WcteTrainingShards)
    root = V2 / "wcte_train"
    root.mkdir(parents=True, exist_ok=True)
    link = root / "runs"
    if not link.exists():
        link.symlink_to(Path("..") / w2.SHARD_DIR / "dev_train")
    rows = []
    for r in mg.cohort_runs("dev_train"):
        d = run_dir(V2, "dev_train", r)
        m = json.loads((d / "manifest.json").read_text())
        row = dict(m["runs"][0])
        row.update(run_dir=d.name, manifest_sha256=lb._sha((d / "manifest.json").read_bytes()))
        rows.append(row)
    first = json.loads((run_dir(V2, "dev_train", mg.cohort_runs("dev_train")[0]) / "manifest.json").read_text())
    man = {"schema": "jepa-lucid-dataset-v1", "dataset_kind": w2.DATASET_KIND_W2,
           "source_dataset_kind": mg.DATASET_KIND, "sampling_tier": mg.SAMPLING_TIER, "cohort": "dev_train",
           "law": "F1", "setting": 0.05, "n_events": int(sum(r["n_events"] for r in rows)), "n_runs": len(rows),
           "run_length": 1000, "dataset_seed": mg.DATASET_SEED, "registered": False, "reported_results": False,
           "geometry_sha256": first["geometry_sha256"], "lucid": first["lucid"], "time_rule": rule,
           "readout_label": first["readout_label"], "runs": rows}
    target = root / "manifest.json"
    if target.exists():
        raise SystemExit(f"{target} exists; never rewritten")
    tmp = root / "manifest.json.building"
    tmp.write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
    os.rename(tmp, target)
    files["wcte_train_manifest"] = {"path": str(target), "sha256": lb._sha(target.read_bytes()),
                                    "n_events": man["n_events"]}
    sums = "".join(f"{v['sha256']}  {Path(v['path']).relative_to(V2)}\n" for v in files.values())
    (V2 / "SHA256SUMS").write_text(sums)
    manifest = {"extension": "r2_sparse_2026-11-24", "registered": False, "detector": "wcte",
                "dataset_kind": w2.DATASET_KIND_W2, "source": "lucid_v1 cached LUCiD digits (/ceph/dwong/jepa/"
                "data/lucid_v1/wcte/cache, read-only)", "lucid": first["lucid"], "lucid_sha": mg.LUCID_SHA,
                "time_rule": rule, "window_ns": rule["prompt_window_ns"],
                "yield_range_photons": cfg["yield_range_photons"]["wcte"], "geometry": cfg["geometry"]["wcte"],
                "cohorts": {c: {"n_runs": len(mg.cohort_runs(c)),
                                "n_events": int(sum(x.n_events for x in mg.cohort_runs(c)))} for c in ALL},
                "files": files, "pack_summary": summary, "extra_fields": list(w2.EXTRA_FIELDS),
                "dark_extension": dark_totals(), "pack_wall_s": time.time() - t0,
                "per_run_sha256sums": {c: [{"run_id": r.run_id, "sha256sums": (run_dir(V2, c, r) / "SHA256SUMS")
                                            .read_text()} for r in mg.cohort_runs(c)] for c in ALL}}
    for out in (V2 / "DATA_MANIFEST_wcte_w2.json", Path(a.out_manifest) / "DATA_MANIFEST_wcte_w2.json"):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("window_ns", "cohorts", "dark_extension")}, default=str))


def _finish_manifest(a, cfg, rule, files, summary, t0):
    """r2-sparse-sk J0: SHA256SUMS and DATA_MANIFEST for a non-WCTE detector (no training directory)."""
    first = json.loads((run_dir(V2, "dev_train", mg.cohort_runs("dev_train")[0]) / "manifest.json").read_text())
    sums = "".join(f"{v['sha256']}  {Path(v['path']).relative_to(V2)}\n" for v in files.values())
    target = V2 / "SHA256SUMS"
    if target.exists():
        raise SystemExit(f"{target} exists; never rewritten")
    target.write_text(sums)
    manifest = {"extension": "r2_sparse_sk_2026-11-26", "registered": False, "detector": DET,
                "dataset_kind": w2.DATASET_KIND_W2, "source": f"lucid_v1 cached LUCiD digits (/ceph/dwong/jepa/"
                f"data/lucid_v1/{DET}/cache, read-only)", "lucid": first["lucid"], "lucid_sha": mg.LUCID_SHA,
                "time_rule": rule, "window_ns": rule["prompt_window_ns"],
                "yield_range_photons": cfg["yield_range_photons"][DET], "geometry": cfg["geometry"][DET],
                "cohorts": {c: {"n_runs": len(mg.cohort_runs(c)),
                                "n_events": int(sum(x.n_events for x in mg.cohort_runs(c)))} for c in ALL},
                "files": files, "pack_summary": summary, "extra_fields": list(w2.EXTRA_FIELDS),
                "dark_extension": dark_totals(), "pack_wall_s": time.time() - t0,
                "per_run_sha256sums": {c: [{"run_id": r.run_id, "sha256sums": (run_dir(V2, c, r) / "SHA256SUMS")
                                            .read_text()} for r in mg.cohort_runs(c)] for c in ALL}}
    name = f"DATA_MANIFEST_{DET}_w2.json"
    for out in (V2 / name, Path(a.out_manifest) / name):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("window_ns", "cohorts", "dark_extension")}, default=str))


def dark_totals():
    out = {}
    for c in ALL:
        tot = {}
        for r in mg.cohort_runs(c):
            m = json.loads((run_dir(V2, c, r) / "manifest.json").read_text())
            for k, v in m["dark_extension"].items():
                if not k.endswith("_mean"):
                    tot[k] = tot.get(k, 0) + v
            tot["n_events"] = tot.get("n_events", 0) + m["n_events"]
        out[c] = tot
    return out


def validate(a):
    """W2 step 5 against lucid_v1, per cohort (v1 readout recomputed from the cache for physics flags)."""
    cfg = mg.load_config()
    geometry, _ = mg.lucid_geometry(DET, cfg)
    rule1 = mg.time_rule(DET, cfg)
    cohort = a.cohort
    if cohort == "sealed_test":
        log_access(a.access_log, "W2 structural validation v2 vs v1 shards and cache (event ids, truths, "
                                 "hit identity, triggers, dark counts; no model metrics)")
    runs = mg.cohort_runs(cohort)
    if a.limit_runs:
        runs = runs[:a.limit_runs]
    ref = lb.generator_reference(mg.toy_generator_config())
    fixed_gain = lb.fixed_charge_gain(geometry, ref)
    c = {"n_events": 0, "truth_mismatch_runs": 0, "event_id_mismatch_runs": 0, "v1_hits": 0,
         "v1_hits_common": 0, "v1_hits_50_60_signal": 0, "v1_hits_50_60_dark": 0, "v1_hits_missing": 0,
         "v1_hits_changed": 0, "v1_hits_changed_dark_ext_first": 0, "v1_hits_changed_other": 0,
         "v1_shard_vs_cache_mismatch": 0, "trigger_equal_events": 0, "signal_ge_v1_common_events": 0,
         "signal_ge_v1_total_events": 0, "v2_hits": 0, "v2_signal_hits": 0, "v1_signal_hits": 0,
         "v2_dark_hits": 0, "v1_dark_hits": 0, "min_q": np.inf, "n_q_nonpositive": 0, "n_later_positive": 0,
         "dark_ext_first_hits": 0, "trigger_shift_events": 0,
         "v1_hits_missing_dropped_ext_first": 0}
    lo, hi = None, None
    dark_times = []
    for r in runs:
        d1, d2 = run_dir(V1, cohort, r), run_dir(V2, cohort, r)
        h1, h2 = lb.read_sparse_hits(d1 / "hits.npz"), lb.read_sparse_hits(d2 / "hits.npz")
        m2 = json.loads((d2 / "manifest.json").read_text())
        lo, hi = m2["time_rule"]["prompt_window_ns"]
        with np.load(d1 / "events.npz") as e1, np.load(d2 / "events.npz") as e2:
            same = set(e1.files) == set(e2.files) and all(
                e1[k].dtype == e2[k].dtype and e1[k].tobytes() == e2[k].tobytes() for k in e1.files)
        c["truth_mismatch_runs"] += int(not same)
        c["event_id_mismatch_runs"] += int(not np.array_equal(h1.event_ids, h2.event_ids))
        ex = np.load(d2 / "hits_extra.npz")
        phys2, ext2, nl2 = ex["physics_first"], ex["dark_ext_first"], ex["n_later"]
        cache = pickle.loads((V1 / "cache" / cohort / (r.run_id.replace("/", "__") + ".pkl")).read_bytes())
        state = r.state(geometry)
        gains = fixed_gain * np.asarray(state.gains, float)
        for k, (eid, dig) in enumerate(zip(cache["event_ids"], cache["digits"])):
            c["n_events"] += 1
            s1, e1_ = int(h1.offsets[k]), int(h1.offsets[k + 1])
            s2, e2_ = int(h2.offsets[k]), int(h2.offsets[k + 1])
            p1, q1, t1 = h1.pmt_id[s1:e1_], h1.charge_pe[s1:e1_], h1.time_ns[s1:e1_]
            p2, q2, t2 = h2.pmt_id[s2:e2_], h2.charge_pe[s2:e2_], h2.time_ns[s2:e2_]
            # v1 recomputed from the cache (physics flags); must equal the v1 shard
            v1 = mg.veritas_readout(dig, geometry.n_pmt, gains, rule1, cfg["readout_start_ns"])
            if not (np.array_equal(v1.pmt_id, p1) and np.array_equal(v1.charge_pe, q1)
                    and np.array_equal(v1.time_ns, t1)):
                c["v1_shard_vs_cache_mismatch"] += 1
            sig1 = _physics_first(dig, v1.pmt_id)
            c["v1_hits"] += p1.size
            c["v2_hits"] += p2.size
            # float32 token times: a digit just above +50 ns can read 50.0, so 50.0 counts as outside
            # r2-sparse-sk: a v2 window that ends later than v1's (JUNO) keeps every v1 hit in common
            hi_c = COMMON_HI_NS if COMMON_HI_NS is not None else (hi if hi < rule1["prompt_window_ns"][1] else None)
            late = (t1 >= hi_c) if hi_c is not None else np.zeros(t1.shape, bool)
            c["v1_hits_50_60_signal"] += int((late & sig1).sum())
            c["v1_hits_50_60_dark"] += int((late & ~sig1).sum())
            com = ~late
            c["v1_hits_common"] += int(com.sum())
            pos = np.searchsorted(p2, p1[com])
            found = (pos < p2.size) & (p2[np.minimum(pos, p2.size - 1)] == p1[com])
            dropped = ex["dropped_ext_pmt"][ex["dropped_ext_offsets"][k]:ex["dropped_ext_offsets"][k + 1]]
            miss_drop = ~found & np.isin(p1[com], dropped)
            c["v1_hits_missing_dropped_ext_first"] += int(miss_drop.sum())
            c["v1_hits_missing"] += int((~found & ~miss_drop).sum())
            j = pos[found]
            same_hit = (q2[j] == q1[com][found]) & (t2[j] == t1[com][found])
            changed = ~same_hit
            c["v1_hits_changed"] += int(changed.sum())
            ext_j = ext2[s2:e2_][j]
            c["v1_hits_changed_dark_ext_first"] += int((changed & ext_j).sum())
            other = changed & ~ext_j
            # a trigger shift moves every common token time by one constant and keeps the charges
            shift_event = False
            if other.any():
                keep = ~ext_j
                dts = (t2[j][keep].astype(np.float64) - t1[com][found][keep].astype(np.float64))
                shift_event = bool(np.all(q2[j][keep] == q1[com][found][keep]) and np.ptp(dts) < 1e-3
                                   and abs(dts[0]) > 0)
            c["trigger_shift_events"] += int(shift_event)
            c["v1_hits_changed_other"] += 0 if shift_event else int(other.sum())
            c["trigger_equal_events"] += int(not other.any())
            ps2 = phys2[s2:e2_]
            n_sig2 = int(ps2.sum())
            c["v2_signal_hits"] += n_sig2
            c["v1_signal_hits"] += int(sig1.sum())
            n_drop_sig = int((sig1 & com & np.isin(p1, dropped)).sum())
            c["signal_ge_v1_common_events"] += int(n_sig2 >= int((sig1 & com).sum()) - n_drop_sig)
            c["signal_ge_v1_total_events"] += int(n_sig2 >= int(sig1.sum()))
            c["v2_dark_hits"] += int((~ps2).sum())
            c["v1_dark_hits"] += int((~sig1).sum())
            dark_times.append(t2[~ps2])
            c["dark_ext_first_hits"] += int(ext2[s2:e2_].sum())
            c["n_later_positive"] += int((nl2[s2:e2_] > 0).sum())
            if q2.size:
                c["min_q"] = min(c["min_q"], float(q2.min()))
                c["n_q_nonpositive"] += int((q2 <= 0).sum())
    dt = np.concatenate(dark_times) if dark_times else np.empty(0)
    edges = np.arange(lo, hi + 1e-9, 10.0)
    hist, _ = np.histogram(dt, bins=edges)
    # bins touching [-20, +10) ns hold the NHits-trigger artefact (a dark digit that completes a
    # trigger cluster becomes the trigger, tau = 0; identical in v1) and are excluded
    use = (edges[1:] <= TRIGGER_ZONE[0]) | (edges[:-1] >= TRIGGER_ZONE[1])
    expected = hist[use].mean() if use.any() else 0.0
    dev = hist[use] / max(expected, 1e-9) - 1.0
    chi2 = float(((hist[use] - expected) ** 2 / max(expected, 1e-9)).sum())
    n = c["n_events"]
    res = {"cohort": cohort, "window_ns": [lo, hi], "counts": {k: (float(v) if isinstance(v, float) else int(v))
                                                              for k, v in c.items()},
           "dark_time_hist_edges_ns": edges.tolist(), "dark_time_hist": hist.tolist(),
           "dark_uniformity_chi2": chi2, "dark_uniformity_dof": int(use.sum() - 1),
           "dark_uniformity_bins_used": use.tolist(), "dark_uniformity_max_rel_dev": float(np.abs(dev).max()),
           "dark_uniformity_rel_dev": dev.tolist(),
           "dark_hits_per_event_v2": c["v2_dark_hits"] / max(n, 1),
           "dark_hits_per_event_v1": c["v1_dark_hits"] / max(n, 1),
           "trigger_equal_fraction": c["trigger_equal_events"] / max(n, 1)}
    from scipy.stats import chi2 as chi2_dist
    res["dark_uniformity_p"] = float(chi2_dist.sf(chi2, max(int(use.sum()) - 1, 1)))
    checks = {
        "event_ids_and_truths_identical": c["truth_mismatch_runs"] == 0 and c["event_id_mismatch_runs"] == 0,
        "v1_shards_equal_cache_recompute": c["v1_shard_vs_cache_mismatch"] == 0,
        "v1_hits_present_bit_identical_except_dark_ext_first_and_trigger_shift": c["v1_hits_missing"] == 0
        and c["v1_hits_changed_other"] == 0,
        "trigger_equal_ge_99_9pct": res["trigger_equal_fraction"] >= 0.999,
        "signal_hits_ge_v1_on_common_window": c["signal_ge_v1_common_events"] == n,
        "every_hit_q_positive": c["n_q_nonpositive"] == 0,
        "dark_uniform_within_10pct_outside_trigger_zone": res["dark_uniformity_max_rel_dev"] <= 0.10,
    }
    res["checks"] = checks
    res["pass"] = bool(all(checks.values()))
    res["notes"] = {"signal_ge_v1_total_events_fraction": c["signal_ge_v1_total_events"] / max(n, 1),
                    "v1_hits_50_60": "v1 hits with token time in (50, 60] ns are outside the W1 window "
                                     "(A = 50 ns) and expected absent; counted separately"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"cohort": cohort, "pass": res["pass"], "checks": checks,
                      "trigger_equal_fraction": res["trigger_equal_fraction"],
                      "dark_v2": res["dark_hits_per_event_v2"], "dark_v1": res["dark_hits_per_event_v1"],
                      "p": res["dark_uniformity_p"]}))


def _physics_first(dig, pmts):
    """Physics flag of each PMT's earliest cached digit, for the given (sorted) PMT ids."""
    s, t, phys = np.asarray(dig["sensor"]), np.asarray(dig["t"], float), np.asarray(dig["physics"], bool)
    order = np.lexsort((t, s))
    s, phys = s[order], phys[order]
    first = np.ones(s.size, bool)
    first[1:] = s[1:] != s[:-1]
    lookup = dict(zip(s[first].tolist(), phys[first].tolist()))
    return np.asarray([lookup[int(p)] for p in pmts], bool)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", choices=("wcte", "juno"), default="wcte")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pack")
    p.add_argument("--out-manifest", required=True)
    p.add_argument("--access-log", type=Path, required=True)
    v = sub.add_parser("validate")
    v.add_argument("--cohort", choices=ALL, required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--access-log", type=Path)
    v.add_argument("--limit-runs", type=int)
    a = ap.parse_args()
    set_detector(a.detector)
    return pack(a) if a.cmd == "pack" else validate(a)


if __name__ == "__main__":
    main()
