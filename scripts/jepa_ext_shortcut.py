#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Shortcut-collapse follow-up CLI (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md`` (Part B).
Extension name: ``shortcut_followup_2026-11-07``; variant ``target_posfree``.

Modes
-----
dry-run       the §9.2 (a)-(c) gate, the setup path of every mode and the
              executed-import closure assertion; writes ``plan.json`` and exits 2
              only on a closure/setup failure (a missing acceptance record is the
              normal pre-acceptance state ``gate_pending_acceptance``).
preflight     §2.3 read-only data hashes, the S1 launch rule, the §8.1 cost
              projection, the entries preflight and the readout items; writes
              ``<ext>/plan.json``.
manipulation  §5.2 manipulation check on the training cell (CPU); writes
              ``<ext>/manipulation.json`` and exits 2 on a failure (S9).
reproduce     §5.3 reproduction gate (``ext_h12.REPRODUCTION``, GPU through the
              launcher); writes ``<ext>/reproduction.json``, exits 2 on a failure.
smoke         the single authorised 20-step smoke (seed 9000) in its own root
              under ``<ext>/smoke``; archived, never resumed.
train         one authorised run (seeds 9000/9001, 10 000 steps) in its own root
              under ``<ext>/train/runs/`` with a ``variant.json`` sidecar.
evaluate      one evaluation job over both run directories: ``ext_h12.evaluate_run``
              (full input and half pass) plus the label-pass ``f_event`` over the
              same 64 events as §5.2; writes ``<ext>/eval/eval.json``.
r1            R-1 (CPU, report only): reads the Part A repair pilot read-only and
              is ``not_run`` until its outputs exist; writes ``<ext>/r1.json``.
report        assembles ``results/jepa/20261107_shortcut/``; ``--r1-only`` fills
              only the R-1 block later, when the pilot outputs exist.

Addendum (7 Nov 2026, v1; ``RUNBOOK_2026-11-07_REPAIR_ADDENDUM.md``): the three
``addendum-*`` modes run under the shared addendum budget (0.35 L40S h, 3 wall-h
CPU, ledger ``/ceph/dwong/jepa/returns/addendum_20261107_jobs.jsonl``) and never
consult the closed Part B (1.5 h) cap.

addendum-evaluate
              D5: one evaluation job for the two existing run directories, exactly
              Part B §5.3 -- full input and half pass via ``ext_h12.evaluate_run``
              plus the label-pass ``f_event`` over the §5.2 64 events; R2-a/R2-b/
              R2-c follow Part B §5.4 unchanged.  Writes
              ``<ext>/eval/addendum_eval.json``.
addendum-r1   Addendum §2: re-reads the pilot's updated ``x1.json``/``x2.json``/
              ``readings.json`` read-only and re-emits R-1 with the D1-corrected
              ``rho_LR``; the Part B §3 rule, families and table are unchanged.
              Writes ``<ext>/r1_addendum.json`` AND the stable results hand-off
              ``results/jepa/20261107_shortcut/r1_addendum.json`` (reading at
              top-level ``reading``/``r1``/``R-1``) for Part A's superseded table.
addendum-report
              Addendum §5/§10: preserves the 7 Nov ``readings.json`` write-once
              under ``<ext>/addendum/preserved_7nov/`` (sha256 sidecar) and then
              updates ``results/jepa/20261107_shortcut/`` in place (``README.md``,
              ``readings.json``, ``report.json``, ``runs.json``, ``OUTCOME.json``,
              the stable ``r1_addendum.json``) with the R2 readings, the corrected
              R-1 and the "Superseded 7 Nov values" table, and writes
              ``ADDENDUM_2026-11-07.json`` with the R-2 evaluation record.

All three addendum modes run the addendum-only entries preflight before work
(live CephFS quota xattrs or guard snapshot; projected writes = the addendum's
own 200; pass iff free >= 30 000 and free - projected >= 30 000; a failure is the
entries-preflight STOP) and record it in their payloads.  The combined-preflight
entries block is history only and is not the addendum STOP basis.

Every mode re-checks the executed-import closure at its end (§9.2).  Nothing here
edits a registered module, config, prediction or verdict, and no data is ever
generated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_noise import ext_h12 as eh  # noqa: E402
from jepa_noise import ext_shortcut as xs  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402

CONFIG_DEFAULT = str(ROOT / xs.CONFIG_REL)
GENERATOR_DEFAULT = str(ROOT / xs.GENERATOR_REL)
GEOMETRY_DEFAULT = xs.GEOMETRY_DEFAULT
DATA_ROOT_DEFAULT = xs.DATA_ROOT_DEFAULT
EXT_DEFAULT = str(xs.EXT_ROOT)
READOUT_EXT_DEFAULT = str(xs.READOUT_EXT)


# ----------------------------------------------------------------------------- io
def write_json(path, payload) -> None:
    xs.write_json(path, payload)


def read_json(path, default=None):
    return xs.read_json(path, default)


def _out_json(raw, default_name, *, default_dir=None) -> Path:
    """``--out`` may be a JSON file or a directory (``<dir>/<default_name>``)."""
    if raw is None:
        if default_dir is None:
            raise SystemExit("this mode needs --out")
        return Path(default_dir) / default_name
    path = Path(raw)
    if path.suffix == ".json":
        return path
    return path / default_name


def load_setup(args, *, device_default="auto"):
    """The shared setup path: config, generator, arms, geometry, runtime."""
    config = lr.load_run_config(getattr(args, "config", None) or CONFIG_DEFAULT)
    gen = lb.load_generator_config(str(getattr(args, "generator_config", None)
                                       or GENERATOR_DEFAULT))
    arms_config = la.load_config(getattr(args, "arms_config", None) or None)
    geometry = lb.load_wcte_geometry(str(getattr(args, "geometry", None)
                                         or GEOMETRY_DEFAULT))
    device = lr.configure_runtime(getattr(args, "device", None) or device_default)
    return config, gen, arms_config, geometry, device


def repo_root(args) -> Path:
    return Path(getattr(args, "root", None) or ROOT).resolve()


def ext_root(args) -> Path:
    return Path(getattr(args, "ext", None) or EXT_DEFAULT)


def data_root(args) -> str:
    return getattr(args, "data_root", None) or DATA_ROOT_DEFAULT


def readout_items(args) -> Path:
    return Path(getattr(args, "readout_items", None)
                or (Path(getattr(args, "readout_ext", None) or READOUT_EXT_DEFAULT)
                    / "items.json"))


def _closure_end(payload, path, root, *, mode) -> int:
    """§9.2 closure re-check at the END of a mode; a violation is the closure STOP."""
    closure = xs.import_closure(root)
    payload["import_closure"] = closure
    write_json(path, payload)
    if not closure["pass"]:
        print(f"[shortcut] {mode}: import-closure STOP: "
              f"{closure['violations']} (outputs quarantined, §8.4)", file=sys.stderr,
              flush=True)
        return 2
    return 0


def _guard(fn, args, *, mode, dry=False):
    """Run a mode body, writing a JSON error payload instead of a traceback."""
    started = time.time()
    try:
        return fn(args, started)
    except SystemExit:
        raise
    except Exception as problem:  # visible, never silently skipped
        payload = {"registered": False, "extension": xs.EXTENSION, "mode": mode,
                   "error": f"{type(problem).__name__}: {problem}",
                   "wall_seconds": round(time.time() - started, 3)}
        out = getattr(args, "out", None)
        if out is not None:
            try:
                write_json(_out_json(out, "error.json"), payload)
            except Exception:  # pragma: no cover - best effort
                pass
        print(f"[shortcut] {mode} failed: {payload['error']}", file=sys.stderr,
              flush=True)
        return 2


# --------------------------------------------------------------------- dry-run
def mode_dry_run(args) -> int:
    """§9.2: gate + the setup path of every mode + the executed-import closure."""
    started = time.time()
    root = repo_root(args)
    gate = xs.assert_authorised("dry-run", root=root, dry_run=True)
    payload = {"registered": False, "extension": xs.EXTENSION, "mode": "dry-run",
               "runbook": xs.RUNBOOK_REL, "root": str(root),
               "code_sha": lr.git_head(root), "git_status": lr.git_status(root),
               "gate": gate, "setup": {}, "errors": [], "closure": None,
               "wall_seconds": None}
    try:
        config, gen, arms_config, geometry, device = load_setup(args,
                                                                device_default="cpu")
        payload["setup"]["objects"] = {
            "config": str(getattr(args, "config", None) or CONFIG_DEFAULT),
            "generator_config": str(getattr(args, "generator_config", None)
                                    or GENERATOR_DEFAULT),
            "geometry": str(getattr(args, "geometry", None) or GEOMETRY_DEFAULT),
            "geometry_digest": getattr(geometry, "digest", None),
            "n_pmt": int(geometry.n_pmt), "device": str(device)}
        payload["setup"]["preflight"] = xs.preflight(
            root=root, data_root=data_root(args), ext_root=ext_root(args),
            readout_items=readout_items(args), dry_run=True)
        cell = lr.dev_cell(config)
        payload["setup"]["cell"] = {"cell_id": cell.cell_id, "split": cell.split,
                                    "occupancy": cell.occupancy,
                                    "s_arm": cell.s_arm, "s_p": cell.s_p}
        # variant: build the seed-9000 arm and install the label-pass rebinding
        torch.manual_seed(xs.SMOKE_SEED)
        np.random.seed(xs.SMOKE_SEED % 2 ** 32)
        arm = lr.build_arm(xs.ARM_NAME, geometry, arms_config, cell,
                           xs.SMOKE_SEED).to(device)
        xs.attach_posfree_target(arm)
        payload["setup"]["variant"] = {
            "arm": xs.ARM_NAME, "variant": xs.VARIANT, "installed": True,
            "label_pass": "posfree_features(batch.features) (columns 0-5 zeroed)",
            "label_detached": False, "d_model": int(getattr(arm, "d_model", 0) or 0)}
        if not payload["setup"]["preflight"]["data"].get("pass"):
            payload["errors"].append("§2.3 data preflight failed (STOP for R-2)")
        else:
            train_ds, probes = xs.open_datasets(cell, config, gen, geometry,
                                                data_root(args))
            payload["setup"]["datasets"] = {
                "train": train_ds.key,
                "probes": {role: dataset.key for role, dataset in probes.items()}}
            batches = xs.label_batches(train_ds, geometry, config, device=device)
            arm.eval()
            payload["setup"]["manipulation"] = xs.manipulation_check(
                arm, batches[0], extra_batches=batches[1:], device=device)
        payload["setup"]["r1"] = xs.pilot_status(root)["reason"] or "complete"
        payload["setup"]["report"] = {
            "ext_root": str(ext_root(args)),
            "results_dir": str(root / xs.RESULTS_DIR),
            "readout_items": str(readout_items(args))}
        # addendum plan (A4: the addendum-only entries preflight is the STOP
        # basis; the combined-preflight block above is history only)
        entries = xs.addendum_entries_preflight()
        payload["setup"]["addendum"] = {
            "path": xs.ADDENDUM_REL, "sha256": xs.ADDENDUM_SHA256,
            "gpu_cap_hours": xs.ADDENDUM_GPU_CAP_H,
            "cpu_cap_hours": xs.ADDENDUM_CPU_CAP_H,
            "gpu_projections_hours": dict(xs.ADDENDUM_COST_BY_ITEM),
            "ledger": str(xs.ADDENDUM_JOBS),
            "s1": xs.addendum_s1_check(root),
            "entries_preflight": entries}
    except SystemExit as problem:
        payload["errors"].append(f"SystemExit({problem.code})")
    except Exception as problem:
        payload["errors"].append(f"{type(problem).__name__}: {problem}")
    payload["closure"] = xs.import_closure(root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    out = _out_json(getattr(args, "out", None), "plan.json")
    write_json(out, payload)
    summary = {"plan": str(out), "gate_status": gate["status"],
               "authorised": gate["authorised"],
               "checks": {key: gate["checks"][key].get("pass")
                          for key in ("a", "b", "c")},
               "closure_pass": payload["closure"]["pass"],
               "closure_forced_imports": [
                   row["path"] for row in payload["closure"]["forced_import_exceptions"]],
               "part_b_forced_transitive_imports": [
                   row["path"] for row in
                   payload["closure"]["part_b_forced_transitive_imports"]],
               "closure_violations": payload["closure"]["violations"],
               "setup_errors": payload["errors"],
               "manipulation_pass": (payload["setup"].get("manipulation") or {}).get(
                   "pass"),
               "addendum_entries_pass": (payload["setup"].get(
                   "addendum") or {}).get("entries_preflight", {}).get("pass"),
               "addendum_entries_basis": (payload["setup"].get(
                   "addendum") or {}).get("entries_preflight", {}).get("snapshot",
                                                                        {}).get("basis"),
               "addendum_s1_pass": (payload["setup"].get(
                   "addendum") or {}).get("s1", {}).get("pass"),
               "wall_seconds": payload["wall_seconds"]}
    print(json.dumps(summary, indent=1), flush=True)
    if payload["errors"] or not payload["closure"]["pass"]:
        print("[shortcut] dry-run FAILED (exit 2)", file=sys.stderr, flush=True)
        return 2
    if gate["status"] == "gate_pending_acceptance":
        print("[shortcut] dry-run: gate pending acceptance (normal before the lead's "
              "record)", flush=True)
    return 0


# ------------------------------------------------------------------- preflight
def mode_preflight(args, started) -> int:
    root = repo_root(args)
    gate = xs.assert_authorised("preflight", root=root)
    xs.assert_cpu_budget(ext_root(args), xs.COST_BY_JOB["preflight"])
    payload = xs.preflight(root=root, data_root=data_root(args),
                           ext_root=ext_root(args),
                           readout_items=readout_items(args))
    payload.update(gate=gate, mode="preflight",
                   wall_seconds=round(time.time() - started, 3))
    out = _out_json(getattr(args, "out", None), "plan.json",
                    default_dir=ext_root(args))
    xs.record_job(ext_root(args), mode="preflight", kind="cpu",
                  wall_seconds=payload["wall_seconds"],
                  extra={"out": str(out), "pass": payload["pass"]})
    code = _closure_end(payload, out, root, mode="preflight")
    print(json.dumps({"out": str(out), "pass": payload["pass"],
                      "data_pass": payload["data"].get("pass"),
                      "s1": payload["s1"].get("pass"),
                      "s1_sum": payload["s1"].get("sum_hours"),
                      "caps": payload["caps"].get("pass"),
                      "entries": payload["entries"].get("pass"),
                      "readout_items": payload["readout_items"]["exists"],
                      "closure_pass": payload["import_closure"]["pass"]},
                     default=float), flush=True)
    return code if code else (0 if payload["pass"] else 2)


# ---------------------------------------------------------------- manipulation
def mode_manipulation(args, started) -> int:
    root = repo_root(args)
    gate = xs.assert_authorised("manipulation", root=root)
    xs.assert_cpu_budget(ext_root(args), xs.COST_BY_JOB["manipulation"])
    config, gen, arms_config, geometry, device = load_setup(args, device_default="cpu")
    payload = xs.manipulation_run(config, gen, arms_config, geometry,
                                  data_root(args), root=root, device=device)
    payload.update(gate=gate, mode="manipulation",
                   wall_seconds=round(time.time() - started, 3))
    out = _out_json(getattr(args, "out", None), "manipulation.json",
                    default_dir=ext_root(args))
    xs.record_job(ext_root(args), mode="manipulation", kind="cpu",
                  wall_seconds=payload["wall_seconds"],
                  extra={"out": str(out), "pass": payload["pass"]})
    code = _closure_end(payload, out, root, mode="manipulation")
    if code:
        return code
    print(json.dumps({"out": str(out), "pass": payload["pass"],
                      "checks": payload["checks"],
                      "label_max_abs_delta": payload["identity_blindness"][
                          "label_max_abs_delta"],
                      "context_max_abs_delta": payload["identity_blindness"][
                          "context_max_abs_delta"],
                      "input_proj_grad_absmax": payload["label_gradient"][
                          "input_proj_grad_absmax"],
                      "label_f_layout": payload["label_fractions"].get("f_layout"),
                      "label_f_event": payload["label_fractions"].get("f_event"),
                      "reasons": payload["reasons"]}, default=float), flush=True)
    if not payload["pass"]:
        print("[shortcut] manipulation check FAILED (S9); stop and report.",
              file=sys.stderr, flush=True)
        return 2
    return 0


# ------------------------------------------------------------------- reproduce
def mode_reproduce(args, started) -> int:
    root = repo_root(args)
    gate = xs.assert_authorised("reproduce", root=root)
    xs.s1_check_or_stop(root)
    xs.assert_gpu_budget(ext_root(args), xs.COST_BY_JOB["reproduce"])
    _require_manipulation(ext_root(args))
    config, gen, arms_config, geometry, device = load_setup(args)
    items = readout_items(args)
    if not items.is_file():
        raise SystemExit(f"[shortcut] readout items {items} is missing; the "
                         f"reproduction gate needs it")
    probes = eh.load_probes(config, gen, geometry, data_root(args))
    reproduction = xs.reproduce(config, arms_config, geometry, probes, items,
                               device=device, root=root)
    reproduction.update(gate=gate, mode="reproduce",
                        wall_seconds=round(time.time() - started, 3),
                        code_sha=lr.git_head(root), readout_items=str(items))
    out = _out_json(getattr(args, "out", None), "reproduction.json",
                    default_dir=ext_root(args))
    xs.record_job(ext_root(args), mode="reproduce", kind="gpu",
                  wall_seconds=reproduction["wall_seconds"],
                  extra={"out": str(out),
                         "pass": reproduction.get("reproduction_pass")})
    code = _closure_end(reproduction, out, root, mode="reproduce")
    print(json.dumps({"out": str(out),
                      "reproduction_pass": reproduction["reproduction_pass"],
                      "items": reproduction["items"],
                      "closure_pass": reproduction["import_closure"]["pass"]},
                     default=float), flush=True)
    if code:
        return code
    if not reproduction["reproduction_pass"]:
        print(f"[shortcut] §5.3 reproduction FAILED (S9 after one fix attempt): "
              f"{reproduction['items']}", file=sys.stderr, flush=True)
        return 2
    return 0


# ----------------------------------------------------------------------- smoke
def mode_smoke(args, started) -> int:
    root = repo_root(args)
    gate = xs.assert_authorised("smoke", root=root)
    steps = int(getattr(args, "steps", None) or xs.SMOKE_STEPS)
    seed = int(getattr(args, "seed", None) or xs.SMOKE_SEED)
    out_root = Path(getattr(args, "out", None) or (ext_root(args) / "smoke"))
    xs.assert_launch_allowed("smoke", seed=seed, steps=steps, out_root=out_root)
    xs.s1_check_or_stop(root)
    _require_plan(ext_root(args))
    _require_manipulation(ext_root(args))
    config, gen, arms_config, geometry, device = load_setup(args)
    result = xs.run_one_training(
        mode="smoke", seed=seed, steps=steps, out_root=out_root, config=config,
        gen=gen, arms_config=arms_config, geometry=geometry, device=device,
        data_root=data_root(args), config_path=getattr(args, "config", None)
        or CONFIG_DEFAULT, code_sha=lr.git_head(root), status_text=lr.git_status(root),
        root=root, ext_root=ext_root(args))
    summary = dict(result["summary"])
    summary.update(gate_status=gate["status"], mode="smoke",
                   wall_seconds=round(time.time() - started, 3))
    payload = {"registered": False, "extension": xs.EXTENSION, "mode": "smoke",
               "gate": gate, "run_id": result["run_id"],
               "run_dir": str(result["run_dir"]), "seed": seed, "steps": steps,
               "summary": summary, "code_sha": lr.git_head(root)}
    out = out_root / f"{result['run_id']}.json"
    xs.record_job(ext_root(args), mode="smoke", kind="gpu",
                  wall_seconds=summary["wall_seconds"],
                  extra={"run_id": result["run_id"], "out": str(out)})
    code = _closure_end(payload, out, root, mode="smoke")
    passed = bool(summary.get("metrics_finite")
                  and summary.get("status") == "complete"
                  and summary.get("steps_completed") == steps
                  and summary.get("resumed_from_step") == 0
                  and not summary.get("wall_cap_exceeded"))
    print(json.dumps({"run_dir": str(result["run_dir"]), "passed": passed,
                      "steps": summary.get("steps_completed"),
                      "resumed_from_step": summary.get("resumed_from_step"),
                      "metrics_finite": summary.get("metrics_finite"),
                      "wall_h": (summary.get("wall_seconds") or 0) / 3600.0},
                     default=float), flush=True)
    if code:
        return code
    if not passed:
        print("[shortcut] smoke acceptance FAILED", file=sys.stderr, flush=True)
        return 2
    return 0


# ----------------------------------------------------------------------- train
def mode_train(args, started) -> int:
    root = repo_root(args)
    gate = xs.assert_authorised("train", root=root)
    steps = int(getattr(args, "steps", None) or xs.TRAIN_STEPS)
    seed = int(args.seed)
    out_root = Path(getattr(args, "out", None) or (ext_root(args) / "train"))
    xs.assert_launch_allowed("train", seed=seed, steps=steps, out_root=out_root)
    xs.s1_check_or_stop(root)
    _require_plan(ext_root(args))
    _require_manipulation(ext_root(args))
    config, gen, arms_config, geometry, device = load_setup(args)
    result = xs.run_one_training(
        mode="train", seed=seed, steps=steps, out_root=out_root, config=config,
        gen=gen, arms_config=arms_config, geometry=geometry, device=device,
        data_root=data_root(args), config_path=getattr(args, "config", None)
        or CONFIG_DEFAULT, code_sha=lr.git_head(root), status_text=lr.git_status(root),
        root=root, ext_root=ext_root(args))
    summary = dict(result["summary"])
    summary.update(gate_status=gate["status"], mode="train",
                   wall_seconds=round(time.time() - started, 3))
    payload = {"registered": False, "extension": xs.EXTENSION, "mode": "train",
               "gate": gate, "run_id": result["run_id"],
               "run_dir": str(result["run_dir"]), "seed": seed, "steps": steps,
               "summary": summary, "code_sha": lr.git_head(root),
               "variant_sidecar": str(result["run_dir"] / "variant.json")}
    out = out_root / f"{result['run_id']}.json"
    xs.record_job(ext_root(args), mode="train", kind="gpu",
                  wall_seconds=summary["wall_seconds"],
                  extra={"run_id": result["run_id"], "seed": seed,
                         "out": str(out)})
    code = _closure_end(payload, out, root, mode="train")
    passed = bool(summary.get("status") == "complete"
                  and summary.get("metrics_finite")
                  and summary.get("steps_completed") == steps
                  and summary.get("resumed_from_step") == 0
                  and not summary.get("wall_cap_exceeded"))
    print(json.dumps({"run_dir": str(result["run_dir"]), "seed": seed,
                      "passed": passed, "steps": summary.get("steps_completed"),
                      "resumed_from_step": summary.get("resumed_from_step"),
                      "checkpoint": summary.get("final_checkpoint_sha256"),
                      "wall_h": (summary.get("wall_seconds") or 0) / 3600.0,
                      "wall_cap_exceeded": summary.get("wall_cap_exceeded")},
                     default=float), flush=True)
    if code:
        return code
    if not passed:
        print("[shortcut] train acceptance FAILED (STOP)", file=sys.stderr, flush=True)
        return 2
    return 0


# -------------------------------------------------------------------- evaluate
def _run_dirs(args) -> list:
    values = getattr(args, "run", None) or []
    if not values:
        train_root = ext_root(args) / "train" / "runs"
        found = sorted(path for path in train_root.glob("*") if path.is_dir())
        if not found:
            raise SystemExit(f"[shortcut] no run directories under {train_root}; "
                             f"train both seeds first")
        return found
    dirs = []
    for value in values:
        path = Path(value)
        if not path.is_dir():
            raise SystemExit(f"--run {path} is not a directory")
        dirs.append(path)
    return dirs


def mode_evaluate(args, started) -> int:
    root = repo_root(args)
    gate = xs.assert_authorised("evaluate", root=root)
    xs.s1_check_or_stop(root)
    xs.assert_gpu_budget(ext_root(args), xs.COST_BY_JOB["evaluate"])
    config, gen, arms_config, geometry, device = load_setup(args)
    cell = lr.dev_cell(config)
    train_ds, probes = xs.open_datasets(cell, config, gen, geometry, data_root(args))
    batches = xs.label_batches(train_ds, geometry, config, device=device)
    run_dirs = _run_dirs(args)
    runs = xs.evaluate_run_dirs(run_dirs, probes, geometry, config, arms_config,
                                device=device, batches=batches, root=root)
    payload = {"registered": False, "extension": xs.EXTENSION, "mode": "evaluate",
               "gate": gate, "runbook": xs.RUNBOOK_REL,
               "code_sha": lr.git_head(root), "device": str(device),
               "cell_id": cell.cell_id, "label_events": int(sum(
                   batch.features.size(0) for batch in batches)),
               "runs": runs, "complete": True,
               "wall_seconds": round(time.time() - started, 3)}
    out_dir = Path(getattr(args, "out", None) or (ext_root(args) / "eval"))
    out = out_dir / "eval.json"
    xs.record_job(ext_root(args), mode="evaluate", kind="gpu",
                  wall_seconds=payload["wall_seconds"],
                  extra={"out": str(out), "runs": sorted(runs)})
    code = _closure_end(payload, out, root, mode="evaluate")
    print(json.dumps({"out": str(out), "seeds": sorted(runs),
                      "label_events": payload["label_events"],
                      "closure_pass": payload["import_closure"]["pass"],
                      "wall_h": payload["wall_seconds"] / 3600.0},
                     default=float), flush=True)
    if code:
        return code
    if sorted(runs) != ["9000", "9001"]:
        print(f"[shortcut] evaluate needs both seeds, got {sorted(runs)}",
              file=sys.stderr, flush=True)
        return 2
    return 0


# ------------------------------------------------------------------------- r1
def mode_r1(args, started) -> int:
    root = repo_root(args)
    gate = xs.assert_authorised("r1", root=root)
    xs.assert_cpu_budget(ext_root(args), xs.COST_BY_JOB["r1"])
    payload = xs.r1_payload(root, ext_root=xs.PILOT_EXT_ROOT)
    payload.update(gate=gate, mode="r1",
                   wall_seconds=round(time.time() - started, 3),
                   code_sha=lr.git_head(root))
    out = _out_json(getattr(args, "out", None), "r1.json", default_dir=ext_root(args))
    xs.record_job(ext_root(args), mode="r1", kind="cpu",
                  wall_seconds=payload["wall_seconds"],
                  extra={"out": str(out), "status": payload["status"]})
    code = _closure_end(payload, out, root, mode="r1")
    print(json.dumps({"out": str(out),
                      "results_copy": (payload.get("results_copy") or {}).get(
                          "path"),
                      "status": payload["status"],
                      "reason": payload.get("reason"),
                      "reading": (payload.get("reading") or {}).get("verdict"),
                      "closure_pass": payload["import_closure"]["pass"]},
                     default=float), flush=True)
    return code


# --------------------------------------------------------------------- report
def mode_report(args, started) -> int:
    root = repo_root(args)
    gate = xs.assert_authorised("report", root=root)
    xs.assert_cpu_budget(ext_root(args), xs.COST_BY_JOB["report"])
    payload = xs.assemble_report(root=root, data_dir=ext_root(args),
                                 results_dir=getattr(args, "out", None),
                                 ext_root=ext_root(args),
                                 r1_only=bool(getattr(args, "r1_only", False)))
    payload.update(gate=gate, mode="report",
                   wall_seconds=round(time.time() - started, 3))
    path = Path(payload["results_dir"]) / "report.json"
    xs.record_job(ext_root(args), mode="report", kind="cpu",
                  wall_seconds=payload["wall_seconds"],
                  extra={"results_dir": payload["results_dir"],
                         "r1_only": payload.get("r1_only")})
    code = _closure_end(payload, path, root, mode="report")
    print(json.dumps({"results_dir": payload["results_dir"],
                      "r1_only": payload.get("r1_only"),
                      "r1_status": payload.get("r1_status"),
                      "too_big": payload.get("too_big"),
                      "closure_pass": payload["import_closure"]["pass"]},
                     default=float), flush=True)
    if code:
        return code
    return 2 if payload.get("too_big") else 0


# --------------------------------------------------- addendum (7 Nov 2026, v1)
def _book_addendum(part, mode, kind, seconds) -> dict:
    return xs.record_addendum_job(part, mode=mode, kind=kind, wall_seconds=seconds)


def mode_addendum_evaluate(args, started) -> int:
    """D5: the one R-2 evaluation job for the two run directories (§6)."""
    root = repo_root(args)
    gate = xs.assert_authorised("addendum-evaluate", root=root)
    s1 = xs.addendum_s1_check_or_stop(root)
    # A4 (review round 7): the addendum-only entries preflight is the STOP basis,
    # called before the job; the combined-preflight block is history only.
    entries = xs.assert_addendum_entries_preflight()
    budget = xs.assert_addendum_budget(
        "gpu", xs.ADDENDUM_COST_BY_ITEM["r2_evaluation"], part="B")
    config, gen, arms_config, geometry, device = load_setup(args)
    cell = lr.dev_cell(config)
    train_ds, probes = xs.open_datasets(cell, config, gen, geometry, data_root(args))
    batches = xs.label_batches(train_ds, geometry, config, device=device)
    run_dirs = _run_dirs(args)
    job_started = time.time()
    try:
        runs = xs.evaluate_run_dirs(run_dirs, probes, geometry, config, arms_config,
                                    device=device, batches=batches, root=root,
                                    mode="addendum-evaluate")
    except BaseException:
        _book_addendum("B", "addendum-evaluate", "gpu",
                       round(time.time() - job_started, 3))
        raise
    payload = {"registered": False, "extension": xs.EXTENSION,
               "mode": "addendum-evaluate",
               "addendum": {"path": xs.ADDENDUM_REL,
                            "sha256": xs.ADDENDUM_SHA256,
                            "date": xs.ADDENDUM_DATE,
                            "author_line": xs.ADDENDUM_AUTHOR_LINE},
               "gate": gate, "s1": s1, "entries": entries,
               "budget_before": budget,
               "code_sha": lr.git_head(root), "device": str(device),
               "cell_id": cell.cell_id,
               "run_dirs": [str(path) for path in run_dirs],
               "label_events": int(sum(batch.features.size(0) for batch in batches)),
               "runs": runs, "readings": xs.r2_readings(runs),
               "statistic_audit": xs.statistic_audit(runs),
               "complete": True, "wall_seconds": round(time.time() - started, 3)}
    job = _book_addendum("B", "addendum-evaluate", "gpu", payload["wall_seconds"])
    payload["addendum_job"] = job
    payload["budget_after"] = xs.addendum_hours()
    out_dir = Path(getattr(args, "out", None) or (ext_root(args) / "eval"))
    out = out_dir / "addendum_eval.json"
    code = _closure_end(payload, out, root, mode="addendum-evaluate")
    print(json.dumps({"out": str(out), "seeds": sorted(runs),
                      "run_dirs": payload["run_dirs"],
                      "label_events": payload["label_events"],
                      "r2_a": payload["readings"]["r2_a"].get("class"),
                      "r2_b": payload["readings"]["r2_b"].get("verdict"),
                      "budget_gpu_hours": payload["budget_after"]["gpu_hours"],
                      "budget_gpu_cap_hours": xs.ADDENDUM_GPU_CAP_H,
                      "closure_pass": payload["import_closure"]["pass"],
                      "wall_h": payload["wall_seconds"] / 3600.0},
                     default=float), flush=True)
    if code:
        return code
    missing = [seed for seed in ("9000", "9001") if seed not in sorted(runs)]
    if missing:
        print(f"[shortcut] addendum-evaluate needs both run directories (missing "
              f"seed(s) {missing}); the two seeds are seeds 9000 and 9001",
              file=sys.stderr, flush=True)
        return 2
    if payload["readings"].get("status") != "complete":
        print(f"[shortcut] addendum-evaluate: R2 readings "
              f"{payload['readings'].get('status')!r}", file=sys.stderr, flush=True)
        return 2
    return 0


def mode_addendum_r1(args, started) -> int:
    """Addendum §2: the R-1 rerun with the D1-corrected ``rho_LR`` (read-only)."""
    root = repo_root(args)
    gate = xs.assert_authorised("addendum-r1", root=root)
    entries = xs.assert_addendum_entries_preflight()
    budget = xs.assert_addendum_budget(
        "cpu", xs.ADDENDUM_COST_BY_JOB["addendum-r1"], part="B")
    status = xs.pilot_status(root)
    job_started = time.time()
    try:
        payload = xs.r1_payload(root, mode="addendum-r1",
                                ext_root=xs.PILOT_EXT_ROOT)
    except BaseException:
        _book_addendum("B", "addendum-r1", "cpu",
                       round(time.time() - job_started, 3))
        raise
    payload.update(gate=gate, mode="addendum-r1", entries=entries,
                   budget_before=budget,
                   wall_seconds=round(time.time() - started, 3),
                   code_sha=lr.git_head(root),
                   addendum={"path": xs.ADDENDUM_REL,
                             "sha256": xs.ADDENDUM_SHA256,
                             "date": xs.ADDENDUM_DATE,
                             "author_line": xs.ADDENDUM_AUTHOR_LINE,
                             "read_only": "the pilot's x1.json/x2.json/"
                                          "readings.json are never written"})
    payload["pilot_files"] = {
        name: {"path": row["path"],
               "sha256": (lr.sha256_file(Path(row["path"]))
                          if row.get("exists") else None)}
        for name, row in (status.get("files") or {}).items()}
    out = _out_json(getattr(args, "out", None), "r1_addendum.json",
                    default_dir=ext_root(args))
    job = _book_addendum("B", "addendum-r1", "cpu", payload["wall_seconds"])
    payload["addendum_job"] = job
    code = _closure_end(payload, out, root, mode="addendum-r1")
    # A2 (review round 7): the payload is ALSO written to the stable results path
    # results/jepa/20261107_shortcut/r1_addendum.json, with the reading under the
    # documented top-level keys read by Part A's superseded table.
    payload["results_copy"] = xs.write_results_r1(root, payload)
    xs.write_json(out, payload)
    reading = payload.get("reading") or {}
    print(json.dumps({"out": str(out),
                      "results_copy": (payload.get("results_copy") or {}).get(
                          "path"),
                      "status": payload["status"],
                      "reason": payload.get("reason"),
                      "verdict": reading.get("verdict"),
                      "generative_median_rho_lr":
                          (reading.get("generative_median") or {}).get("rho_lr"),
                      "frozen_target_rho_lr":
                          (reading.get("frozen_target") or {}).get("rho_lr"),
                      "closure_pass": payload["import_closure"]["pass"]},
                     default=float), flush=True)
    if code:
        return code
    # R-1 not_run (pilot outputs missing) is not a STOP (Part B §3).
    return 0


def mode_addendum_report(args, started) -> int:
    """Addendum §5/§10: update the Part B results in place and record D5."""
    root = repo_root(args)
    gate = xs.assert_authorised("addendum-report", root=root)
    entries = xs.assert_addendum_entries_preflight()
    budget = xs.assert_addendum_budget(
        "cpu", xs.ADDENDUM_COST_BY_JOB["addendum-report"], part="B")
    job_started = time.time()
    try:
        payload = xs.assemble_addendum_report(
            root=root, data_dir=ext_root(args),
            results_dir=getattr(args, "out", None), ext_root=ext_root(args),
            budget=budget)
    except BaseException:
        _book_addendum("B", "addendum-report", "cpu",
                       round(time.time() - job_started, 3))
        raise
    payload.update(gate=gate, mode="addendum-report", entries=entries,
                   budget_before=budget,
                   wall_seconds=round(time.time() - started, 3))
    job = _book_addendum("B", "addendum-report", "cpu", payload["wall_seconds"])
    payload["addendum_job"] = job
    payload["addendum_hours"] = xs.addendum_hours()
    path = Path(payload["results_dir"]) / "report.json"
    code = _closure_end(payload, path, root, mode="addendum-report")
    print(json.dumps({"results_dir": payload["results_dir"],
                      "entries_pass": (payload.get("entries") or {}).get("pass"),
                      "results_r1": (payload.get("results_r1") or {}).get("path"),
                      "preserved_7nov": (payload.get("preserved_7nov") or {}).get(
                          "path"),
                      "seven_nov_source": payload.get("seven_nov_source"),
                      "r2_status": payload.get("r2_status"),
                      "r1_status": payload.get("r1_status"),
                      "complete": payload.get("complete"),
                      "superseded": len(payload.get("superseded") or []),
                      "statistic_audit_pass": payload.get("statistic_audit_pass"),
                      "addendum_file": payload.get("addendum_file"),
                      "too_big": payload.get("too_big"),
                      "closure_pass": payload["import_closure"]["pass"]},
                     default=float), flush=True)
    if code:
        return code
    return 2 if payload.get("too_big") else 0


# ------------------------------------------------------------------- helpers
def _require_plan(ext_root_path) -> dict:
    path = Path(ext_root_path) / "plan.json"
    plan = read_json(path, {}) or {}
    if not plan:
        raise SystemExit(f"[shortcut] missing {path}; run preflight first")
    if not plan.get("pass"):
        raise SystemExit(f"{path}: preflight has not passed")
    return plan


def _require_manipulation(ext_root_path) -> dict:
    path = Path(ext_root_path) / "manipulation.json"
    payload = read_json(path, {}) or {}
    if not payload.get("pass"):
        raise SystemExit(f"{path}: the §5.2 manipulation check has not passed (a "
                         f"failure after one fix attempt is S9 for R-2)")
    return payload


# ---------------------------------------------------------------------- parser
def _common(parser):
    parser.add_argument("--root", default=None, help="repository root (default: this repo)")
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--generator-config", default=GENERATOR_DEFAULT)
    parser.add_argument("--arms-config", default=None)
    parser.add_argument("--geometry", default=GEOMETRY_DEFAULT)
    parser.add_argument("--ext", default=EXT_DEFAULT)
    parser.add_argument("--readout-ext", default=READOUT_EXT_DEFAULT)
    parser.add_argument("--readout-items", default=None)


def parser():
    p = argparse.ArgumentParser(prog="jepa_ext_shortcut")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--out", required=True)
    _common(d)
    d.add_argument("--device", default="cpu")
    pf = sub.add_parser("preflight")
    pf.add_argument("--out", default=None)
    _common(pf)
    pf.add_argument("--device", default="cpu")
    m = sub.add_parser("manipulation")
    m.add_argument("--out", default=None)
    _common(m)
    m.add_argument("--device", default="cpu")
    rp = sub.add_parser("reproduce")
    rp.add_argument("--out", default=None)
    _common(rp)
    rp.add_argument("--device", default="auto")
    s = sub.add_parser("smoke")
    s.add_argument("--seed", type=int, default=xs.SMOKE_SEED)
    s.add_argument("--steps", type=int, default=xs.SMOKE_STEPS)
    s.add_argument("--out", default=None)
    _common(s)
    s.add_argument("--device", default="auto")
    t = sub.add_parser("train")
    t.add_argument("--seed", required=True, type=int)
    t.add_argument("--steps", type=int, default=xs.TRAIN_STEPS)
    t.add_argument("--out", default=None)
    _common(t)
    t.add_argument("--device", default="auto")
    e = sub.add_parser("evaluate")
    e.add_argument("--run", action="append", default=None)
    e.add_argument("--out", default=None)
    _common(e)
    e.add_argument("--device", default="auto")
    r1 = sub.add_parser("r1")
    r1.add_argument("--out", default=None)
    _common(r1)
    r1.add_argument("--device", default="cpu")
    rep = sub.add_parser("report")
    rep.add_argument("--out", default=None)
    rep.add_argument("--r1-only", action="store_true")
    _common(rep)
    rep.add_argument("--device", default="cpu")
    # ---- addendum (7 Nov 2026, v1): the 0.35 h GPU / 3 wall-h CPU budget ----
    ae = sub.add_parser(
        "addendum-evaluate",
        help="D5: the one R-2 evaluation job for the two run directories "
             "(full input + half pass via ext_h12.evaluate_run, plus the "
             "label-pass f_event over the §5.2 events); addendum 0.35 h GPU "
             "budget, never the closed Part B 1.5 h cap",
        description="D5 R-2 evaluation (addendum §6; evaluation only, no "
                    "training).  One job for the two existing run directories, "
                    "exactly Part B §5.3: ext_h12.evaluate_run full input and half "
                    "pass, plus the label-pass f_event over the §5.2 64 events; "
                    "R2-a/R2-b/R2-c follow Part B §5.4 unchanged.  Runs under the "
                    "shared addendum budget (refuses SystemExit(2) 'not_run (cap)' "
                    "if the 0.35 L40S-h cap would be crossed) and does not consult "
                    "the closed Part B cap.")
    ae.add_argument("--run", action="append", default=None,
                    help="a completed run directory (repeat once per seed; "
                         "default: every directory under <ext>/train/runs, i.e. "
                         "the two ext_shortcut_20261107 runs)")
    ae.add_argument("--out", default=None,
                    help="output directory (default <ext>/eval); writes "
                         "addendum_eval.json")
    _common(ae)
    ae.add_argument("--device", default="auto")
    ar = sub.add_parser(
        "addendum-r1",
        help="Addendum §2: re-read the pilot's updated x1.json/x2.json/"
             "readings.json (read-only) and re-emit R-1 with the D1-corrected "
             "rho_LR; §3 rule, families and table unchanged; addendum CPU budget",
        description="R-1 rerun (addendum §2/§8 step 3).  Re-reads the Part A "
                    "pilot's updated x1.json/x2.json/readings.json read-only and "
                    "re-emits R-1 with the corrected rho_LR.  Writes "
                    "<ext>/r1_addendum.json under the addendum CPU budget; the "
                    "pilot results and <ext> are never written.")
    ar.add_argument("--out", default=None,
                    help="output JSON or directory (default <ext>/r1_addendum.json)")
    _common(ar)
    ar.add_argument("--device", default="cpu")
    arep = sub.add_parser(
        "addendum-report",
        help="Addendum §5/§10: update results/jepa/20261107_shortcut/ in place "
             "(README, readings, report, runs, OUTCOME) with the R2 readings, the "
             "corrected R-1 and the superseded-7-Nov table; write "
             "ADDENDUM_2026-11-07.json",
        description="Addendum report (addendum §5 D4/§10).  Updates "
                    "results/jepa/20261107_shortcut/{README.md,readings.json,"
                    "report.json,runs.json,OUTCOME.json} in place with the R2 "
                    "readings (D5), the corrected R-1 (D1), the Part B "
                    "\"Superseded 7 Nov values\" table and the readings "
                    "completeness; writes ADDENDUM_2026-11-07.json with the R-2 "
                    "evaluation record.  Part B has no figures (the six are "
                    "Part A's).")
    arep.add_argument("--out", default=None,
                      help="results directory (default results/jepa/"
                           "20261107_shortcut)")
    _common(arep)
    arep.add_argument("--device", default="cpu")
    return p


BODIES = {"dry-run": mode_dry_run, "preflight": mode_preflight,
          "manipulation": mode_manipulation, "reproduce": mode_reproduce,
          "smoke": mode_smoke, "train": mode_train, "evaluate": mode_evaluate,
          "r1": mode_r1, "report": mode_report,
          "addendum-evaluate": mode_addendum_evaluate,
          "addendum-r1": mode_addendum_r1,
          "addendum-report": mode_addendum_report}


def main(argv=None):
    args = parser().parse_args(argv)
    body = BODIES[args.mode]
    if args.mode == "dry-run":
        return body(args)
    return _guard(body, args, mode=args.mode)


if __name__ == "__main__":
    sys.exit(main())
