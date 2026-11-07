#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paper 3 repair-pilot CLI (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-07_REPAIR_AND_SHORTCUT.md`` (Part A).
Extension name ``paper3_repair_pilot_2026-11-06``.

Modes
-----
dry-run    the §9.2 (a)-(c) gate, the setup path of every mode, and the executed-
           import closure assertion; writes a summary JSON.  A missing acceptance
           record is the normal pre-acceptance state ``gate_pending_acceptance``.
preflight  X0-a (CPU, read-only): ``plan.json`` + ``inventory.json``.
reproduce  X0-b reproduction gate (GPU, one job): ``reproduction.json``.
x1         X1 calibration readings: ``x1.json`` and the embedding cache.
x2         X2 equivariance / latent repair / identification: ``x2.json``.
x3-fisher  X3-a Fisher accounting and P0: ``x3_fisher.json``.
x3         X3-b invariant variants and LEACE: ``x3.json``.
report     merges everything into ``results/jepa/20261106_repair/``.

No training mode exists, no data is generated, and no registered module, config,
prediction or verdict is edited.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_noise import ext_followup as ef  # noqa: E402
from jepa_noise import ext_repair as rep  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402

CONFIG_DEFAULT = str(ROOT / rep.CONFIG_REL)
GENERATOR_DEFAULT = str(ROOT / rep.GEN_REL)
ARMS_DEFAULT = str(ROOT / rep.ARMS_REL)
DATA_ROOT_DEFAULT = rep.DATA_ROOT_DEFAULT
EXT_DEFAULT = str(rep.EXT_ROOT)


# ----------------------------------------------------------------------------- io
def write_json(path, payload) -> None:
    rep.write_json(path, payload)


def read_json(path, default=None):
    return rep.read_json(path, default)


def load_setup(args, *, device_default="auto"):
    """The shared setup path: config, generator, arms, geometry, runtime."""
    config = lr.load_run_config(getattr(args, "config", None) or CONFIG_DEFAULT)
    gen = lb.load_generator_config(str(getattr(args, "generator_config", None)
                                       or GENERATOR_DEFAULT))
    arms_config = la.load_config(getattr(args, "arms_config", None) or ARMS_DEFAULT)
    geometry = lb.load_wcte_geometry(str(getattr(args, "geometry", None)
                                         or rep.GEOMETRY_DEFAULT))
    device = lr.configure_runtime(getattr(args, "device", None) or device_default)
    return config, gen, arms_config, geometry, device


def repo_root(args) -> Path:
    return Path(getattr(args, "root", None) or ROOT).resolve()


def ext_root(args) -> Path:
    return Path(getattr(args, "ext", None) or EXT_DEFAULT)


def data_root(args) -> str:
    return getattr(args, "data_root", None) or DATA_ROOT_DEFAULT


def _output_dir(args, default_relative) -> Path:
    return Path(args.out) if args.out else ext_root(args) / default_relative


# --------------------------------------------------------------------- dry-run
def mode_dry_run(args) -> int:
    """§9.2: gate + execute the setup path of every mode + import-closure assertion."""
    started = time.time()
    root = repo_root(args)
    gate = rep.assert_authorised("dry-run", root=root, dry_run=True)
    payload = {"registered": False, "extension": rep.EXTENSION, "mode": "dry-run",
               "runbook": rep.RUNBOOK_REL, "root": str(root),
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
            "geometry": str(getattr(args, "geometry", None) or rep.GEOMETRY_DEFAULT),
            "geometry_digest": getattr(geometry, "digest", None),
            "n_pmt": int(geometry.n_pmt), "n_modules": int(geometry.n_modules),
            "device": str(device)}
        payload["setup"]["data"] = rep.data_status(data_root(args))
        probes = {"s005": rep.open_probe_sets(config, gen, geometry, data_root(args),
                                              setting=rep.S005,
                                              log=lambda *a, **k: None),
                  "s030": rep.open_probe_sets(config, gen, geometry, data_root(args),
                                              setting=rep.S030,
                                              log=lambda *a, **k: None)}
        inventory = rep.part_a_inventory(root, hash_files=False,
                                         log=lambda *a, **k: None)
        particle_truth = {role: ef.load_particle_truth(data_root(args),
                                                       rep.PROBE_KEYS_S005[role])
                          for role in ("train", "test")}
        modes = [mode for mode in rep.MODES if mode != "dry-run"]
        for mode in modes:
            setup = rep._mode_setup_objects(
                mode, root=root, config=config, gen=gen, arms_config=arms_config,
                geometry=geometry, data_root=data_root(args), ext_root=ext_root(args),
                inventory=inventory, probes=probes, particle_truth=particle_truth,
                log=lambda *a, **k: None)
            payload["setup"][mode] = {"plan": setup["plan"],
                                      "data_pass": setup["data"]["pass"],
                                      "inventory": setup["inventory_counts"],
                                      "probes": setup["probes"]}
        # exercise the §2.5 transform end-to-end on one run of one probe set
        row = inventory["rows"].get("init9000")
        with ef.item_eval_context(row):
            arm = ef.build_item_arm(row, config=config, gen=gen,
                                    arms_config=arms_config, geometry=geometry,
                                    data_root=data_root(args), device=device,
                                    log=lambda *a, **k: None)
        model = getattr(arm, "jepa", arm)
        condition = rep.build_condition("N", probes["s005"]["test"], geometry)
        sample = rep.stream_condition_embeddings(
            model, probes["s005"]["test"], geometry, condition, split="within",
            config=config, device=device, max_events=8, max_runs=1,
            log=lambda *a, **k: None)
        payload["setup"]["sample"] = {
            "label": row.get("label"), "condition": condition.name,
            "n_events": int(sample["events"].shape[0]),
            "embedding_dim": int(sample["events"].shape[1]),
            "reused": sample.get("cache_status")}
    except SystemExit as problem:
        payload["errors"].append(f"SystemExit({problem.code})")
    except Exception as problem:
        payload["errors"].append(f"{type(problem).__name__}: {problem}")
    payload["closure"] = rep.import_closure(root)
    payload["wall_seconds"] = round(time.time() - started, 3)
    out = Path(args.out)
    if out.suffix != ".json":
        out = out / "dry_run.json"
    write_json(out, payload)
    summary = {"plan": str(out), "gate_status": gate["status"],
               "authorised": gate["authorised"],
               "closure_pass": payload["closure"]["pass"],
               "closure_violations": payload["closure"]["violations"],
               "closure_forced_imports": [row["path"] for row in
                                          payload["closure"]["forced_import_exceptions"]],
               "setup_errors": payload["errors"],
               "modes": sorted(payload["setup"]),
               "wall_seconds": payload["wall_seconds"]}
    print(json.dumps(summary, indent=1), flush=True)
    if payload["errors"] or not payload["closure"]["pass"]:
        print("[dry-run] FAILED (exit 2)", file=sys.stderr, flush=True)
        return 2
    if gate["status"] == "gate_pending_acceptance":
        print("[dry-run] gate pending acceptance (normal before the lead's record)",
              flush=True)
    return 0


# ------------------------------------------------------------------- preflight
def mode_preflight(args) -> int:
    root = repo_root(args)
    config, gen, arms_config, geometry, device = load_setup(args, device_default="cpu")
    inventory = rep.part_a_inventory(root, log=print)
    out = _output_dir(args, "x0")
    write_json(out / "inventory.json", inventory)
    payload = rep.preflight(root=root, data_root=data_root(args), config=config, gen=gen,
                            geometry=geometry, inventory=inventory,
                            limit=args.limit, log=print)
    write_json(out / "plan.json", payload)
    print(json.dumps({"out": str(out / "plan.json"), "pass": payload["pass"],
                      "stops": payload["stops"],
                      "data_pass": payload["data"]["pass"],
                      "gpu_projection": payload["gpu_projection"],
                      "s1": payload["s1"], "closure_pass":
                      payload["closure"]["pass"]}, default=float), flush=True)
    return 0 if payload["pass"] else 2


# ------------------------------------------------------------------- reproduce
def mode_reproduce(args) -> int:
    root = repo_root(args)
    config, gen, arms_config, geometry, device = load_setup(args)
    inventory = rep.part_a_inventory(root, log=print)
    payload = rep.reproduction(root=root, config=config, gen=gen,
                               arms_config=arms_config, geometry=geometry,
                               inventory=inventory, data_root=data_root(args),
                               device=device, ext_root=ext_root(args),
                               limit=args.limit, log=print)
    out = _output_dir(args, "x0")
    path = out / "reproduction.json"
    write_json(path, payload)
    print(json.dumps({"out": str(path), "pass": payload["pass"],
                      "tolerance_m": payload["tolerance_m"],
                      "models": {label: row.get("pass") for label, row
                                 in payload["models"].items()},
                      "identity_pass": payload["identity"]["pass"]},
                     default=float), flush=True)
    return 0 if payload["pass"] else 2


# -------------------------------------------------------------------------- x1
def mode_x1(args) -> int:
    root = repo_root(args)
    config, gen, arms_config, geometry, device = load_setup(args)
    inventory = rep.part_a_inventory(root, log=print)
    payload = rep.x1(root=root, config=config, gen=gen, arms_config=arms_config,
                     geometry=geometry, inventory=inventory, data_root=data_root(args),
                     device=device, ext_root=ext_root(args),
                     include_optional=not args.no_optional, limit=args.limit, log=print)
    out = _output_dir(args, "x1")
    path = out / "x1.json"
    write_json(path, payload)
    print(json.dumps({"out": str(path), "complete": payload["complete"],
                      "unavailable": payload["unavailable"],
                      "families": sorted((payload["readings"] or {}).get("families")
                                         or {})}, default=float), flush=True)
    return 0 if payload["complete"] else 2


# -------------------------------------------------------------------------- x2
def mode_x2(args) -> int:
    root = repo_root(args)
    config, gen, arms_config, geometry, device = load_setup(args)
    inventory = rep.part_a_inventory(root, log=print)
    payload = rep.x2(root=root, config=config, gen=gen, arms_config=arms_config,
                     geometry=geometry, inventory=inventory, data_root=data_root(args),
                     device=device, ext_root=ext_root(args),
                     include_optional=not args.no_optional, limit=args.limit, log=print)
    out = _output_dir(args, "x2")
    path = out / "x2.json"
    write_json(path, payload)
    print(json.dumps({"out": str(path), "unavailable": payload["unavailable"],
                      "families": sorted((payload["readings"] or {}).get("families")
                                         or {})}, default=float), flush=True)
    return 0 if not payload["unavailable"] else 2


# -------------------------------------------------------------------- x3-fisher
def mode_x3_fisher(args) -> int:
    root = repo_root(args)
    config, gen, arms_config, geometry, device = load_setup(args, device_default="cpu")
    payload = rep.fisher_accounting(geometry, gen, n_events=int(args.events),
                                    limit=args.limit, root=root, log=print)
    out = _output_dir(args, "x3")
    path = out / "x3_fisher.json"
    write_json(path, payload)
    print(json.dumps({"out": str(path), "n_events": payload["n_events"],
                      "p0": payload["p0"],
                      "models": sorted(payload["models"]),
                      "closure_pass": payload["closure"]["pass"]},
                     default=float), flush=True)
    return 0 if payload["p0"]["pass"] else 2


# -------------------------------------------------------------------------- x3
def mode_x3(args) -> int:
    root = repo_root(args)
    config, gen, arms_config, geometry, device = load_setup(args)
    inventory = rep.part_a_inventory(root, log=print)
    fisher = read_json(ext_root(args) / "x3" / "x3_fisher.json", None)
    payload = rep.x3(root=root, config=config, gen=gen, arms_config=arms_config,
                     geometry=geometry, inventory=inventory, data_root=data_root(args),
                     device=device, ext_root=ext_root(args), fisher_payload=fisher,
                     limit=args.limit, log=print)
    out = _output_dir(args, "x3")
    path = out / "x3.json"
    write_json(path, payload)
    print(json.dumps({"out": str(path),
                      "families": sorted(payload["families"]),
                      "unavailable": payload["unavailable"],
                      "closure_pass": payload["closure"]["pass"]},
                     default=float), flush=True)
    return 0


# ---------------------------------------------------------------------- report
def mode_report(args) -> int:
    root = repo_root(args)
    payload = rep.assemble_report(root=root, ext_root=ext_root(args),
                                  results_dir=args.out, log=print)
    print(json.dumps({"path": payload["path"], "complete": payload["complete"],
                      "reproduction_pass": payload["reproduction_pass"],
                      "figures": payload["figures"].get("n_figures"),
                      "closure_pass": payload["closure"]["pass"]},
                     default=float), flush=True)
    return 0 if (payload.get("reproduction_pass") and not payload.get("too_big")) else 2


# ---------------------------------------------------------------------- parser
def _common(parser):
    parser.add_argument("--root", default=None, help="repository root (default: this repo)")
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--generator-config", default=GENERATOR_DEFAULT)
    parser.add_argument("--arms-config", default=ARMS_DEFAULT)
    parser.add_argument("--geometry", default=rep.GEOMETRY_DEFAULT)
    parser.add_argument("--ext", default=EXT_DEFAULT)


def _limit(parser):
    parser.add_argument("--limit", type=int, default=None,
                        help="engineering sample cap (never used for a reported run)")


def parser():
    p = argparse.ArgumentParser(prog="jepa_ext_repair")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--out", required=True)
    _common(d)
    d.add_argument("--device", default="cpu")
    pre = sub.add_parser("preflight")
    pre.add_argument("--out", default=None)
    _limit(pre)
    _common(pre)
    pre.add_argument("--device", default="cpu")
    rp = sub.add_parser("reproduce")
    rp.add_argument("--out", default=None)
    _limit(rp)
    _common(rp)
    rp.add_argument("--device", default="auto")
    x1 = sub.add_parser("x1")
    x1.add_argument("--out", default=None)
    x1.add_argument("--no-optional", action="store_true")
    _limit(x1)
    _common(x1)
    x1.add_argument("--device", default="auto")
    x2 = sub.add_parser("x2")
    x2.add_argument("--out", default=None)
    x2.add_argument("--no-optional", action="store_true")
    _limit(x2)
    _common(x2)
    x2.add_argument("--device", default="auto")
    xf = sub.add_parser("x3-fisher")
    xf.add_argument("--out", default=None)
    xf.add_argument("--events", type=int, default=1000)
    _limit(xf)
    _common(xf)
    xf.add_argument("--device", default="cpu")
    x3 = sub.add_parser("x3")
    x3.add_argument("--out", default=None)
    _limit(x3)
    _common(x3)
    x3.add_argument("--device", default="auto")
    report = sub.add_parser("report")
    report.add_argument("--out", default=None)
    _common(report)
    report.add_argument("--device", default="cpu")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    handlers = {"dry-run": mode_dry_run, "preflight": mode_preflight,
                "reproduce": mode_reproduce, "x1": mode_x1, "x2": mode_x2,
                "x3-fisher": mode_x3_fisher, "x3": mode_x3, "report": mode_report}
    return handlers[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
