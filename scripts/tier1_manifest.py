#!/usr/bin/env python3
"""r2-tier1 T2.4 / T7.3: metric-free verification manifests for the blind recomputation (r2-tier1v).

Manifest A: every existing LUCiD case of the r2-lucid headline table (legacy init 9000/9001; trained
default-recipe runs of every arm and seed) at step 10,000 and its own step-0 snapshot (matched init).
Manifest B: every E-cell of this campaign (including the existing runs that count as cells) at steps
0, 5,000 and 10,000, plus the registered contrasts by run id.
Fields: run id, detector, arm, seed, step, checkpoint path and SHA256, encoder configuration, scaling,
model construction, pack paths and SHA256, cohort names. **No metric values and no result paths.**
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

LUCID_RUNS = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/train/runs")
TIER1_RUNS = Path("/ceph/dwong/jepa/returns/r2_tier1_2026-11-22/train/runs")
DATA = Path("/ceph/dwong/jepa/data/lucid_v1")
DETS = ("wcte", "sk", "juno", "hk")
_SHA = {}


def sha256(path):
    path = Path(path)
    if path not in _SHA:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        _SHA[path] = h.hexdigest()
    return _SHA[path]


def packs(det):
    main = DATA / det / f"{det}_lucid_v1.h5"
    sealed = DATA / det / f"{det}_lucid_v1_sealed.h5"
    out = {"dev_probe_train": {"path": str(main), "cohort": "dev_probe_train"},
           "dev_probe_test": {"path": str(main), "cohort": "dev_probe_test"},
           "sealed_test": {"path": str(sealed), "cohort": "sealed_test"} if sealed.is_file() else None}
    for v in out.values():
        if v:
            v["sha256"] = sha256(v["path"])
    return out


def scaling_for(det):
    from jepa_noise.ext_r2_repair_train import geometry_and_scaling
    from dataclasses import asdict
    return asdict(geometry_and_scaling(det, "lucid_v1")[1])


def run_case(directory, step):
    rec = json.loads((directory / "run_record.json").read_text())
    recipe = rec["recipe"]
    rid = recipe["identity"]["run_id"]
    det = rid.split("__")[0]
    ck = directory / f"snapshot_step_{step:05d}.pt"
    return {"run_id": rid, "detector": det, "arm": recipe["identity"]["arm"],
            "seed": int(recipe["identity"]["seed"]), "step": step,
            "checkpoint": str(ck), "checkpoint_sha256": sha256(ck) if ck.is_file() else None,
            "config_sha256": rec["config_sha256"], "encoder": recipe["encoder"], "scaling": recipe["scaling"],
            "recipe_variant": {k: recipe[k] for k in ("view_mode", "time_loss_scale_ns", "mask",
                                                      "charge_loss_weight", "time_reference") if k in recipe},
            "construction": "jepa_noise.ext_r2.build_r2_arm(arm, geometry, seed, scaling=scaling, "
                            "encoder_config=EncoderConfig(**encoder)); load_state_dict(torch.load(checkpoint)['model']); "
                            "token embeddings = arm.encode_tokens(features) of the full unmasked event",
            "time_reference_for_encoder_input": recipe.get("time_reference", "trigger")}


def init_case(det, seed):
    return {"run_id": f"{det}__init__s{seed}", "detector": det, "arm": "init", "seed": seed, "step": 0,
            "checkpoint": None, "checkpoint_sha256": None, "encoder": "EncoderConfig() defaults (window 64)",
            "scaling": scaling_for(det),
            "construction": "jepa_noise.ext_r2.build_r2_arm('init', geometry, seed, scaling=scaling) "
                            "(legacy init: sigreg_jepa at the seed, no training, default encoder window 64)"}


def manifest_a():
    cases = []
    for det in DETS:
        for seed in (9000, 9001):
            cases.append(init_case(det, seed))
        for d in sorted(LUCID_RUNS.glob(f"{det}__*")):
            rid = d.name
            if rid.count("__") != 2 or not (d / "run_record.json").is_file():
                continue  # suffixed sweep runs are E-cells (manifest B)
            for step in (10000, 0):
                cases.append(run_case(d, step))
    return cases


def manifest_b():
    import tier1_gate as tg
    cells = []
    roots = [LUCID_RUNS, TIER1_RUNS]
    seen = set()
    for root in roots:
        for d in sorted(root.glob("*__mae_repaired__s*")):
            if d.name.split("__")[0] not in ("sk", "wcte") or not (d / "run_record.json").is_file():
                continue
            rec = json.loads((d / "run_record.json").read_text())
            if not rec.get("complete") or d.name in seen:
                continue
            seen.add(d.name)
            for step in (0, 5000, 10000):
                cells.append(run_case(d, step))
    reg = {cid: {"detector": det, "per_seed": [{"seed": s, "run_1": a, "run_2": b,
                                                "definition": "T = e(run_2)/e(run_1) - 1"} for s, a, b in pairs]}
           for cid, (det, pairs) in tg.registered(seeds_wc=("A", "B", "C")).items()}
    return cells, reg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=("A", "B"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.which == "A":
        cases = manifest_a()
        body = {"manifest": "VERIFY_INPUTS_A", "metric_free": True, "cases": cases}
    else:
        cases, reg = manifest_b()
        body = {"manifest": "VERIFY_INPUTS_B", "metric_free": True, "cases": cases, "registered_contrasts": reg}
    dets = sorted({c["detector"] for c in cases})
    body.update({"phase": "r2-tier1", "runbook": "docs/round2/RUNBOOK_2026-11-22_R2_TIER1_GATE.md",
                 "readout_spec": "docs/round2/READOUT_SPEC_v1.md", "packs": {d: packs(d) for d in dets},
                 "probe_protocol": "fit on dev_probe_train, select on dev_probe_test, apply to sealed_test",
                 "note": "no metric values and no result paths by design"})
    text = json.dumps(body, indent=1, sort_keys=True) + "\n"
    for bad in ("mean_m", "median_m", "spatial Ridge", "/eval/"):
        if bad in text:
            raise SystemExit(f"manifest would contain {bad!r}")
    Path(a.out).write_text(text)
    print(a.which, len(cases), "cases")


if __name__ == "__main__":
    main()
