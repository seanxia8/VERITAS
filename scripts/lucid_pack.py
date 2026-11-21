#!/usr/bin/env python3
"""r2-lucid L6: pack LUCiD shards into the round-2 HDF5 schema and write DATA_MANIFEST_<det>.json.

Runs in the native env (r2_store imports torch). Packs ``<det>_lucid_v1.h5`` (SK/JUNO/HK: dev_train
and the four probe cohorts; WCTE: the probe cohorts), verifies every run array for array
(``r2_store.verify_pack``), and for WCTE writes the round-1 training directory ``wcte_train``
(manifest.json + ``runs`` -> ``../shards/dev_train``). Written to ``.tmp`` and renamed.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_mc_generator as mg  # noqa: E402
from jepa_noise import r2_store as rs  # noqa: E402

DATA = Path("/ceph/dwong/jepa/data/lucid_v1")


def cohorts_for(det, include_train=True):
    names = ["dev_probe_train", "dev_probe_test", "dev_probe_train_s030", "dev_probe_test_s030"]
    if det != "wcte" and include_train:
        names = ["dev_train"] + names
    return {c: list(mg.cohort_runs(c)) for c in names}


def complete(det, cohort):
    runs = mg.cohort_runs(cohort)
    d = DATA / det / "shards" / cohort
    return all((d / r.run_id.replace("/", "__") / "manifest.json").exists() for r in runs)


def wcte_train(det):
    root = DATA / det / "wcte_train"
    root.mkdir(parents=True, exist_ok=True)
    link = root / "runs"
    if not link.exists():
        link.symlink_to(Path("..") / "shards" / "dev_train")
    rows = []
    for r in mg.cohort_runs("dev_train"):
        d = DATA / det / "shards" / "dev_train" / r.run_id.replace("/", "__")
        m = json.loads((d / "manifest.json").read_text())
        row = dict(m["runs"][0]); row.update(run_dir=d.name, manifest_sha256=lb._sha((d / "manifest.json").read_bytes()))
        rows.append(row)
    first = json.loads((DATA / det / "shards" / "dev_train" / rows[0]["run_dir"] / "manifest.json").read_text())
    manifest = {"schema": "jepa-lucid-dataset-v1", "dataset_kind": mg.DATASET_KIND,
                "sampling_tier": mg.SAMPLING_TIER, "cohort": "dev_train", "law": "F1", "setting": 0.05,
                "n_events": int(sum(r["n_events"] for r in rows)), "n_runs": len(rows), "run_length": 1000,
                "dataset_seed": mg.DATASET_SEED, "registered": False, "reported_results": False,
                "geometry_sha256": first["geometry_sha256"], "lucid": first["lucid"],
                "readout_label": first["readout_label"], "runs": rows}
    tmp = root / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    tmp.rename(root / "manifest.json")
    return root, manifest


def regime(det, cohorts):
    pe, dom, ntrig, nev, untrig, phot, calls = [], [], 0, 0, 0, 0, 0
    for c in ("dev_train", "dev_probe_train", "dev_probe_test"):
        if c not in cohorts:
            continue
        for r in cohorts[c]:
            d = DATA / det / "shards" / c / r.run_id.replace("/", "__")
            ev = np.load(d / "events.npz")
            pe.append(ev["expected_pe"]); dom.append(ev["domain"])
            m = json.loads((d / "manifest.json").read_text())
            ntrig += m["n_no_trigger"]; nev += m["n_events"]
            untrig += m["regime"]["n_lucid_untriggered"]; phot += m["regime"]["photons"]
            calls += m["regime"]["kernel_calls"]
    pe, dom = np.concatenate(pe), np.concatenate(dom)
    q = lambda x: [float(np.quantile(x, p)) for p in (0.05, 0.5, 0.95)] if x.size else None
    return {"true_pe_q_all": q(pe), "true_pe_q_photon": q(pe[dom == "photon"]),
            "true_pe_q_muon": q(pe[dom == "muon"]), "true_pe_q_electron": q(pe[dom == "electron"]),
            "n_events": int(nev), "n_no_nhits_trigger": int(ntrig), "n_lucid_untriggered": int(untrig),
            "photons_simulated": int(phot), "kernel_calls": int(calls)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--detector", required=True, choices=sorted(mg.SPECS))
    ap.add_argument("--out-manifest", default=str(REPO / "results/round2/20261121_lucid"))
    a = ap.parse_args()
    det = a.detector
    cfg = mg.load_config()
    geometry, _ = mg.lucid_geometry(det, cfg)
    cohorts = cohorts_for(det)
    missing = [c for c in cohorts if not complete(det, c)]
    if det == "wcte" and not complete(det, "dev_train"):
        missing.append("dev_train")
    if missing:
        raise SystemExit(f"{det}: incomplete cohorts {missing}")
    h5 = DATA / det / f"{det}_lucid_v1.h5"
    attrs = {"detector": det, "extension": "r2_lucid_2026-11-21", "dataset_kind": mg.DATASET_KIND,
             "sampling_tier": mg.SAMPLING_TIER, "lucid_sha": mg.LUCID_SHA,
             "lucid_v1_config_sha256": cfg["_sha256"], "geometry_sha256": geometry.digest,
             "time_rule": mg.time_rule(det, cfg), "veritas_steps": list(mg.VERITAS_STEPS),
             "registered": False}
    t0 = time.time()
    if not h5.exists():
        rs.pack_detector(h5, DATA / det / "shards", cohorts, attrs=attrs)
    verified = rs.verify_pack(h5, DATA / det / "shards", cohorts)
    files = {"pack": {"path": str(h5), "sha256": rs.sha256_file(h5), "bytes": h5.stat().st_size}}
    if det == "wcte":
        root, man = wcte_train(det)
        files["wcte_train_manifest"] = {"path": str(root / "manifest.json"),
                                        "sha256": rs.sha256_file(root / "manifest.json"),
                                        "n_events": man["n_events"]}
    runs = {}
    for c, rr in dict(cohorts, **({"dev_train": list(mg.cohort_runs("dev_train"))} if det == "wcte" else {})).items():
        runs[c] = [{"run_id": r.run_id, "sha256sums": (DATA / det / "shards" / c / r.run_id.replace("/", "__") / "SHA256SUMS").read_text()}
                   for r in rr]
    first = json.loads((DATA / det / "shards" / "dev_probe_train" / "lucid__dev_probe_train__0000" / "manifest.json").read_text())
    manifest = {"extension": "r2_lucid_2026-11-21", "registered": False, "detector": det,
                "dataset_kind": mg.DATASET_KIND, "sampling_tier": mg.SAMPLING_TIER,
                "lucid": first["lucid"], "time_rule": mg.time_rule(det, cfg),
                "yield_range_photons": cfg["yield_range_photons"][det],
                "geometry": cfg["geometry"][det], "files": files,
                "cohorts": {c: {"n_runs": len(r), "n_events": int(sum(x.n_events for x in r))}
                            for c, r in cohorts.items()},
                "verify": verified if isinstance(verified, dict) else str(verified),
                "regime": regime(det, dict(cohorts, **({"dev_train": list(mg.cohort_runs("dev_train"))} if det == "wcte" else {}))),
                "per_run_sha256sums": runs, "pack_wall_s": time.time() - t0}
    out = Path(a.out_manifest) / f"DATA_MANIFEST_{det}.json"
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: manifest[k] for k in ("detector", "cohorts", "regime")}, default=str))


if __name__ == "__main__":
    main()
