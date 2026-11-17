"""Run Part A6 on the final checkpoints verified by the repair campaign."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import ext_r2
from . import ext_r2_repair_diagnostic_helpers as diag
from . import r2_generator as gen
from . import r2_geometry as rg
from . import r2_store as rs
from .ext_r2_repair_diagnostics import DETECTORS, EXTENSION, ROOT, write_json
from .ext_r2_repair_probes import PROBE_ROOT, dataset_path, make_model


def _load(detector, arm):
    row = json.loads((PROBE_ROOT / f"by_arm/{detector}__{arm}__s9000.json").read_text())
    path = Path(row["checkpoint"])
    geometry = rg.load_r2_geometry(detector)
    rule = gen.time_rule_for(geometry)
    scaling = rs.scaling_for(geometry, rule.time_scale_ns)
    model, info = make_model(arm, 9000, geometry, scaling, [path], "cpu")
    if info["checkpoint_sha256"] != row["checkpoint_sha256"]:
        raise ValueError(f"{detector}/{arm}: checkpoint changed since frozen probe")
    return model, geometry, rule, scaling, info


def _micro_batch(path, detector):
    for name in ("run_record.json", "run_status.json", "ext_summary.json"):
        record = path.parent / name
        if not record.exists():
            continue
        meta = json.loads(record.read_text())
        for candidate in (meta, meta.get("config"), meta.get("training")):
            if isinstance(candidate, dict) and "micro_batch" in candidate:
                return int(candidate["micro_batch"])
    if detector != "wcte":
        raise ValueError(f"{path}: missing the original training micro-batch")
    return int(ext_r2.run_config()["training"]["micro_batch"])


def _reference(path, tag, info, micro_batch):
    evidence = path / f"{tag}.json"
    saved = json.loads(evidence.read_text())
    heldout = saved["heldout_objective"]
    if (saved["checkpoint_sha256"] != info["checkpoint_sha256"]
            or int(saved["step"]) != ext_r2.STEPS
            or heldout["micro_batch"] != micro_batch
            or heldout["n_micro_batches"] * micro_batch != diag.HELDOUT_STEPS * ext_r2.BATCH):
        raise ValueError(f"{evidence}: frozen checkpoint or held-out protocol mismatch")
    return heldout | {"verified_heldout_reference": str(evidence),
                      "verified_heldout_reference_sha256": rs.sha256_file(evidence)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-reference", type=Path)
    args = parser.parse_args()
    out = ROOT / "diagnostics" / "a6"
    out.mkdir(parents=True, exist_ok=True)
    summary = {"extension": EXTENSION, "heldout": {}}
    diag.training_logs(ROOT / "figures", summary)
    for detector in DETECTORS:
        bugcheck = out / f"{detector}__bugcheck.json"
        if not bugcheck.exists():
            diag.bugcheck(detector, out, model_loader=_load)
        cohort = (rs.open_cohort(dataset_path(detector), "dev_probe_test")
                  if args.heldout_reference is None else None)
        for arm in ("sigreg_jepa", "target_posfree"):
            model, geometry, _, scaling, info = _load(detector, arm)
            path = Path(info["checkpoint"])
            micro_batch = _micro_batch(path, detector)
            tag = f"{detector}__{arm}__s9000"
            if args.heldout_reference is None:
                rec = diag.heldout_objective(model, geometry, scaling, cohort, micro_batch)
            else:
                rec = _reference(args.heldout_reference, tag, info, micro_batch)
            lines = (path.parent / "train_log.jsonl").read_text().splitlines()
            last = json.loads(lines[-1])
            rec.update(checkpoint=str(path), training_micro_batch=micro_batch,
                       last_logged=last.get("metrics", last), last_logged_step=last.get("step"))
            summary["heldout"][f"{detector}__{arm}__s9000"] = rec
    write_json(ROOT / "diagnostics" / "a6_training_checks.json", summary)


if __name__ == "__main__":
    main()
