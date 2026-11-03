#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""MAE/JEPA follow-up CLI (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-03_MAE_JEPA_FOLLOWUP.md`` (DRAFT v7,
sha256 ``063e9087...``).  Extension name ``mae_jepa_followup_2026-11-03``.

Modes
-----
dry-run    the §6.2 (a)-(e) gate, the setup path of every mode, and the executed-
           import closure assertion (§6.2 (d)); writes ``plan`` and exits 2 only
           on a closure/setup failure (a missing acceptance record is the normal
           pre-acceptance state ``gate_pending_acceptance``).
inventory  §3 A0 preflight (read-only): ``inventory.json`` + ``plan.json``
           (hashes, pairing, projection against the 5.5 h cap, entries).
reproduce  §3 A1 reproduction gate: the five s = 0.05 rows and the two s = 0.30
           rows; exit 2 on a failure (S1/S9).
readout    one Stage A readout: ``--item {A-1,A-1b,A-2,A-3,A-4}``; writes
           ``<ext>/stageA/<item>/readout_<item>.json``.  With ``--include-t1``
           the completed T-1 checkpoints are added to the A-1/A-2 rows (booked
           to the T-1 cap, §5).
train-t1   one authorised T-1 run or the single 20-step smoke (§4); refuses an
           existing run directory, checks the s = 0.30 manifest hashes before
           ``datasets_for``, and runs the host manifest comparison afterwards.
report     merges the readouts into ``results/jepa/20261103_followup/``.

Nothing here edits a registered module, config, prediction or verdict, and no
data is ever generated: the s = 0.30 sets are opened only through their
``registered=False`` keys after the read-only manifest-hash check.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_noise import ext_followup as ef  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402

CONFIG_DEFAULT = str(ROOT / ef.CONFIG_REL)
GENERATOR_DEFAULT = str(ROOT / ef.GEN_REL)
ARMS_DEFAULT = str(ROOT / ef.ARMS_REL)
DATA_ROOT_DEFAULT = ef.DATA_ROOT_DEFAULT
EXT_DEFAULT = str(ef.EXT_ROOT)


# ----------------------------------------------------------------------------- io
def write_json(path, payload) -> None:
    ef.write_json(path, payload)


def read_json(path, default=None):
    return ef.read_json(path, default)


def load_setup(args, *, device_default="auto"):
    """The shared setup path: config, generator, arms, geometry, runtime."""
    config = lr.load_run_config(getattr(args, "config", None) or CONFIG_DEFAULT)
    gen = lb.load_generator_config(str(getattr(args, "generator_config", None)
                                       or GENERATOR_DEFAULT))
    arms_config = la.load_config(getattr(args, "arms_config", None) or ARMS_DEFAULT)
    geometry = lb.load_wcte_geometry(str(getattr(args, "geometry", None)
                                         or ef.GEOMETRY_DEFAULT))
    device = lr.configure_runtime(getattr(args, "device", None) or device_default)
    return config, gen, arms_config, geometry, device


def repo_root(args) -> Path:
    return Path(getattr(args, "root", None) or ROOT).resolve()


def ext_root(args) -> Path:
    return Path(getattr(args, "ext", None) or EXT_DEFAULT)


def data_root(args) -> str:
    return getattr(args, "data_root", None) or DATA_ROOT_DEFAULT


# --------------------------------------------------------------------- dry-run
def mode_dry_run(args) -> int:
    """§6.2: gate + execute the setup path of every mode + import-closure assertion."""
    started = time.time()
    root = repo_root(args)
    gate = ef.assert_authorised("dry-run", root=root, dry_run=True)
    payload = {"registered": False, "extension": ef.EXTENSION, "mode": "dry-run",
               "runbook": ef.RUNBOOK_REL, "root": str(root),
               "code_sha": lr.git_head(root), "git_status": lr.git_status(root),
               "gate": gate, "setup": {}, "errors": [], "closure": None,
               "wall_seconds": None}
    try:
        config, gen, arms_config, geometry, device = load_setup(args, device_default="cpu")
        payload["setup"]["objects"] = {
            "config": str(getattr(args, "config", None) or CONFIG_DEFAULT),
            "generator_config": str(getattr(args, "generator_config", None)
                                    or GENERATOR_DEFAULT),
            "arms_config": str(getattr(args, "arms_config", None) or ARMS_DEFAULT),
            "geometry": str(getattr(args, "geometry", None) or ef.GEOMETRY_DEFAULT),
            "geometry_digest": getattr(geometry, "digest", None),
            "n_pmt": int(geometry.n_pmt), "device": str(device)}
        payload["setup"]["data"] = ef.data_status(data_root(args))
        payload["setup"]["inventory"] = ef.build_inventory(
            root, hash_files=True, log=lambda *a, **k: None)["counts"]
        payload["setup"]["pairing"] = ef.pairing_check(
            data_root(args), limit=int(args.pairing_limit),
            log=lambda *a, **k: None)
        probes005 = ef.open_probe_sets(config, gen, geometry, data_root(args),
                                       setting=ef.S005, log=lambda *a, **k: None)
        probes030 = ef.open_probe_sets(config, gen, geometry, data_root(args),
                                       setting=ef.S030, log=lambda *a, **k: None)
        payload["setup"]["probe_sets"] = {
            "s005": {role: dataset.key for role, dataset in probes005.items()},
            "s030": {role: dataset.key for role, dataset in probes030.items()}}
        inventory = ef.build_inventory(root, hash_files=False,
                                       log=lambda *a, **k: None)
        # §3 A1: the random-init floor arm is the cheapest full arm-setup path.
        init_row = dict(inventory["rows"]["init9000"])
        init_row["status"] = "unverified (dry-run)"
        arm = ef.build_item_arm(init_row, config=config, gen=gen,
                                arms_config=arms_config, geometry=geometry,
                                data_root=data_root(args), device=device,
                                log=lambda *a, **k: None)
        payload["setup"]["init_arm"] = {"arm": init_row["arm"],
                                        "d_model": int(getattr(arm, "d_model", 0) or 0)}
        model = getattr(arm, "jepa", arm)
        # §3 A-1b marginal reader setup (one run only: a hash-checked read, no data)
        marginals = ef.read_dev_marginals(config, geometry, data_root(args), max_runs=1,
                                          log=lambda *a, **k: None)
        payload["setup"]["marginals"] = {"key": marginals["key"],
                                         "n_pmt": marginals["n_pmt"],
                                         "domains_present": marginals["domains_present"],
                                         "sampled_runs": 1}
        # §3 A-1b label collector + median/arch evaluation contexts (one chunk only)
        import contextlib as _contextlib
        with _contextlib.ExitStack() as stack:
            stack.enter_context(ef.median_time_context())
            stack.enter_context(ef.item_eval_context({"eval": {}}))
            rows = ef.collect_label_rows(model, probes005["test"], config, geometry,
                                         max_label_events=8, chunk_events=8,
                                         device=device, log=lambda *a, **k: None)
        payload["setup"]["label_rows"] = {"n_events": rows["n_events"],
                                          "n_rows": int(rows["y"].shape[0]),
                                          "label_dim": rows["label_dim"]}
        # a small frozen-encoder readout sample exercises the §3 streaming core
        sample = ef.evaluate_item_row(init_row, config=config, gen=gen,
                                      arms_config=arms_config, geometry=geometry,
                                      probes=probes005, data_root=data_root(args),
                                      device=device, with_mlp=False, max_events=8,
                                      log=lambda *a, **k: None)
        payload["setup"]["readout_sample"] = {
            "label": sample.get("label"),
            "vertex_resolution_m": (sample.get("full_input") or {}).get(
                "vertex_resolution_m"),
            "n_events": sample.get("n_events")}
        # §4 T-1: the two new cells and the exact run_one kwargs (no launch)
        cells = ef.t1_cells(config)
        payload["setup"]["t1"] = {
            variant: {"cell_id": cell.cell_id, "setting": cell.setting,
                      "s_arm": cell.s_arm, "split": cell.split,
                      "run_id": lr.run_id_for(cell.cell_id, ef.T1_ARM, 9000),
                      "kwargs": sorted(ef.t1_run_kwargs(
                          cell=cell, seed=9000, config=config,
                          arms_config=arms_config, geometry=geometry,
                          loader_dataset="<loader>", probe_datasets="<probes>",
                          out_root="<out>", data_root="<data>", config_sha="<sha>",
                          code_sha="<code>", status_text="<status>", device=device,
                          steps=ef.T1_STEPS))}
            for variant, cell in cells.items()}
        # §6.5 report paths (resolved, nothing read)
        payload["setup"]["report"] = {
            "ext_root": str(ext_root(args)),
            "results_dir": str(repo_root(args) / ef.RESULTS_DIR),
            "readouts": [str(ef._readout_path(ext_root(args), item))
                         for item in ef.READOUT_ITEMS]}
        payload["setup"]["entries"] = ef.entries_preflight(log=lambda *a, **k: None)
        payload["setup"]["projection"] = ef.stage_a_projection()
    except SystemExit as problem:
        payload["errors"].append(f"SystemExit({problem.code})")
    except Exception as problem:
        payload["errors"].append(f"{type(problem).__name__}: {problem}")
    payload["closure"] = ef.import_closure(root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(args.out), payload)
    summary = {"plan": str(args.out),
               "gate_status": gate["status"],
               "authorised": gate["authorised"],
               "closure_pass": payload["closure"]["pass"],
               "closure_forced_imports": [row["path"] for row in
                                          payload["closure"]["forced_import_exceptions"]],
               "closure_violations": payload["closure"]["violations"],
               "setup_errors": payload["errors"],
               "wall_seconds": payload["wall_seconds"]}
    print(json.dumps(summary, indent=1), flush=True)
    if payload["errors"] or not payload["closure"]["pass"]:
        print("[dry-run] FAILED (exit 2)", file=sys.stderr, flush=True)
        return 2
    if gate["status"] == "gate_pending_acceptance":
        print("[dry-run] gate pending acceptance (normal before the lead's record)",
              flush=True)
    return 0


# ------------------------------------------------------------------- inventory
def mode_inventory(args) -> int:
    root = repo_root(args)
    gate = ef.assert_authorised("inventory", root=root)
    inventory = ef.build_inventory(root)
    out = Path(args.out) if args.out else ext_root(args) / "stageA" / "A0"
    write_json(out / "inventory.json", inventory)
    plan = ef.build_plan(root, data_root(args), inventory=inventory,
                         pairing_limit=args.pairing_limit, log=print)
    write_json(out / "plan.json", plan)
    print(json.dumps({"out": str(out),
                      "inventory": inventory["counts"],
                      "gate": gate["status"],
                      "data_pass": (plan.get("data") or {}).get("pass"),
                      "pairing_pass": (plan.get("pairing") or {}).get("pass"),
                      "projected_l40s_hours":
                          (plan.get("stage_a") or {}).get("projected_l40s_hours"),
                      "stage_a_pass": (plan.get("stage_a") or {}).get("pass"),
                      "entries_pass": (plan.get("entries") or {}).get("pass"),
                      "s1_pass": (plan.get("s1") or {}).get("pass"),
                      "pass": plan.get("pass"), "stop": plan.get("stop")},
                     default=float), flush=True)
    return 0 if plan.get("pass") else 2


# ------------------------------------------------------------------- reproduce
def mode_reproduce(args) -> int:
    root = repo_root(args)
    ef.assert_authorised("reproduce", root=root)
    config, gen, arms_config, geometry, device = load_setup(args)
    payload = ef.reproduction(root=root, config=config, gen=gen,
                              arms_config=arms_config, geometry=geometry,
                              data_root=data_root(args), device=device, log=print)
    out = Path(args.out) if args.out else ext_root(args) / "stageA" / "A1"
    out = out if out.suffix == ".json" else out / "reproduction.json"
    payload["wall_seconds"] = None
    write_json(out, payload)
    print(json.dumps({"out": str(out),
                      "s005_pass": payload["s005_pass"],
                      "s030_pass": payload["s030_pass"],
                      "reproduction_pass": payload["reproduction_pass"],
                      "s005": payload["s005"], "s030": payload["s030"]},
                     default=float), flush=True)
    return 0 if payload["reproduction_pass"] else 2


# --------------------------------------------------------------------- readout
READOUT_BODIES = {
    "A-1": ef.readout_a1,
    "A-1b": ef.readout_a1b,
    "A-2": ef.readout_a2,
    "A-3": ef.readout_a3,
    "A-4": ef.readout_a4,
}


def mode_readout(args) -> int:
    root = repo_root(args)
    if args.item not in READOUT_BODIES:
        raise SystemExit(f"[followup] unknown --item {args.item!r}; "
                         f"expected {ef.READOUT_ITEMS}")
    ef.assert_authorised("readout", root=root)
    config, gen, arms_config, geometry, device = load_setup(args)
    inventory = ef.build_inventory(root)
    labels = None
    if args.include_t1 and args.item in ("A-1", "A-2"):
        extra = ef.t1_inventory_rows(ext_root(args))
        inventory["rows"].update(extra)
        # B4: pass the T-1 labels explicitly; they are evaluated on the s = 0.30
        # probes and booked to the T-1 cap (§5).
        labels = ef.readout_label_list(args.item, inventory["rows"], include_t1=True)
        print(json.dumps({"include_t1": sorted(extra), "labels": labels}), flush=True)
    out_dir = Path(args.out) if args.out else ext_root(args) / "stageA" / args.item
    body = READOUT_BODIES[args.item]
    payload = body(root=root, inventory=inventory, config=config, gen=gen,
                   arms_config=arms_config, geometry=geometry,
                   data_root=data_root(args), device=device, labels=labels, log=print)
    payload["wall_seconds"] = None
    path = out_dir / f"readout_{args.item}.json"
    write_json(path, payload)
    print(json.dumps({"out": str(path), "item": args.item,
                      "complete": payload.get("complete"),
                      "unavailable": payload.get("unavailable"),
                      "readings_keys": sorted((payload.get("readings") or {}))},
                     default=float), flush=True)
    return 0 if payload.get("complete", True) else 2


# -------------------------------------------------------------------- train-t1
def mode_train_t1(args) -> int:
    root = repo_root(args)
    config, gen, arms_config, geometry, device = load_setup(args)
    smoke = bool(args.smoke)
    steps = ef.T1_SMOKE_STEPS if smoke else int(args.steps)
    out_root = (ext_root(args) / "smoke") if smoke else (ext_root(args) / "train")
    if args.out is not None and Path(args.out).resolve() != out_root.resolve():
        raise SystemExit(f"[followup] T-1 output root must be {out_root}, got {args.out}")
    if args.plan:
        plan = read_json(args.plan, {}) or {}
        if not (plan.get("entries") or {}).get("pass"):
            raise SystemExit(f"{args.plan}: the §5 entries projection has not passed")
        if not plan.get("pass"):
            raise SystemExit(f"{args.plan}: A0 preflight has not passed")
    entries = ef.entries_preflight(log=print)
    if not entries["pass"]:
        raise SystemExit(f"[followup] entries preflight failed (STOP): {entries}")
    if not entries.get("gpu_slot_available"):
        raise SystemExit("[followup] guard HOLD: no free GPU slot for this job; the "
                         "view-design jobs are expected live jobs, so wait for a slot "
                         "and retry (§5 host rules; not a STOP)")
    payload = ef.train_t1(root=root, config=config, gen=gen, arms_config=arms_config,
                          geometry=geometry, data_root=data_root(args),
                          config_path=getattr(args, "config", None) or CONFIG_DEFAULT,
                          out_root=out_root, variant=args.variant, seed=int(args.seed),
                          steps=steps, device=device, host_check=True,
                          ext_root=ext_root(args), log=print)
    host = payload.get("host_check") or {}
    cost = payload.get("cost") or {}
    summary = dict(payload.get("summary") or {})
    summary["host_check"] = host
    summary["run_dir"] = payload.get("run_dir")
    write_json(Path(payload["run_dir"]) / "ext_summary.json", summary)
    print(json.dumps({"run_id": payload["run_id"], "run_dir": payload["run_dir"],
                      "variant": args.variant, "seed": int(args.seed),
                      "steps": steps, "smoke": smoke,
                      "host_check_pass": host.get("pass"),
                      "unexpected_differences": host.get("unexpected_differences"),
                      "required": host.get("required"),
                      "a1_s030_gate": payload.get("a1_gate"),
                      "t1_cost": cost}, default=float), flush=True)
    ok = bool(host.get("pass") and cost.get("cap_pass", True))
    return 0 if ok else 2


# ---------------------------------------------------------------------- report
def mode_report(args) -> int:
    root = repo_root(args)
    ef.assert_authorised("report", root=root)
    payload = ef.assemble_report(root=root, ext_root=ext_root(args),
                                 results_dir=args.out, log=print)
    return 0 if (payload.get("reproduction_pass") and not payload.get("too_big")) else 2


# ---------------------------------------------------------------------- parser
def _common(parser):
    parser.add_argument("--root", default=None, help="repository root (default: this repo)")
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--generator-config", default=GENERATOR_DEFAULT)
    parser.add_argument("--arms-config", default=ARMS_DEFAULT)
    parser.add_argument("--geometry", default=ef.GEOMETRY_DEFAULT)
    parser.add_argument("--ext", default=EXT_DEFAULT)


def parser():
    p = argparse.ArgumentParser(prog="jepa_ext_followup")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--out", required=True)
    d.add_argument("--pairing-limit", type=int, default=8)
    _common(d)
    d.add_argument("--device", default="cpu")
    i = sub.add_parser("inventory")
    i.add_argument("--out", default=None)
    i.add_argument("--pairing-limit", type=int, default=None)
    _common(i)
    i.add_argument("--device", default="cpu")
    rp = sub.add_parser("reproduce")
    rp.add_argument("--out", default=None)
    _common(rp)
    rp.add_argument("--device", default="auto")
    r = sub.add_parser("readout")
    r.add_argument("--item", required=True, choices=list(ef.READOUT_ITEMS))
    r.add_argument("--out", default=None)
    r.add_argument("--include-t1", action="store_true",
                   help="add the completed T-1 checkpoints to the A-1/A-2 rows")
    _common(r)
    r.add_argument("--device", default="auto")
    t = sub.add_parser("train-t1")
    t.add_argument("--variant", required=True, choices=list(ef.T1_VARIANTS))
    t.add_argument("--seed", required=True, type=int)
    t.add_argument("--steps", type=int, default=ef.T1_STEPS)
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--out", default=None)
    t.add_argument("--plan", default=None)
    _common(t)
    t.add_argument("--device", default="auto")
    rep = sub.add_parser("report")
    rep.add_argument("--out", default=None)
    _common(rep)
    rep.add_argument("--device", default="cpu")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    handlers = {"dry-run": mode_dry_run, "inventory": mode_inventory,
                "reproduce": mode_reproduce, "readout": mode_readout,
                "train-t1": mode_train_t1, "report": mode_report}
    return handlers[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
