#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paper 3 P2 CLI (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-09_P2_INFOLOSS.md``.
Extension ``paper3_p2_infoloss_2026-11-09``.  Stage 1: no training, no data
generation; every factor is an exact transform of the stored hits.

Modes
-----
dry-run     the §8.2 gate (pre-acceptance ``gate_pending_acceptance``), the setup
            path of every mode, sample computations on synthetic/toy arrays and
            the executed-import closure; writes a summary JSON.
preflight   X0-a (CPU, read-only): hashes, identity self-checks, trigger
            efficiency and window-edge counts, the GPU projection, entries, S1;
            writes ``<ext>/x0/plan.json`` and ``inventory.json``.
reproduce   X0-b (GPU): the three Part A N vertex values and the two new models'
            full-input vertex; writes ``<ext>/x0/reproduction.json``.
new-models  (GPU) the two ``target_posfree`` models' Part A sets and their
            corrected X2-b fit; writes ``<ext>/new_models.json``.
conditions  (GPU) the seven §2.3 condition sets and the R1 test set for the 15
            core models; writes ``<ext>/conditions.json``.
p2          (CPU) the §5 readings from the cached embeddings; writes
            ``<ext>/p2.json``.
x3p         (CPU) the §6 X3-P readings; writes ``<ext>/x3p.json``.
report      (CPU) assembles ``results/jepa/20261109_p2/``; writes ``<ext>/report.json``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_noise import ext_p2 as p2x  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402

CONFIG_DEFAULT = str(ROOT / p2x.CONFIG_REL)
GENERATOR_DEFAULT = str(ROOT / p2x.GENERATOR_REL)
GEOMETRY_DEFAULT = p2x.GEOMETRY_DEFAULT
DATA_ROOT_DEFAULT = p2x.DATA_ROOT_DEFAULT


def write_json(path, payload) -> None:
    p2x.write_json(path, payload)


def read_json(path, default=None):
    return p2x.read_json(path, default)


def _out_json(raw, default_name, *, default_dir=None) -> Path:
    if raw is None:
        if default_dir is None:
            raise SystemExit("this mode needs --out")
        return Path(default_dir) / default_name
    path = Path(raw)
    if path.suffix == ".json":
        return path
    return path / default_name


def load_setup(args, *, device_default="cpu"):
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


def data_root(args) -> str:
    return getattr(args, "data_root", None) or DATA_ROOT_DEFAULT


def _closure_end(payload, path, root, *, mode) -> int:
    closure = p2x.import_closure(root)
    payload["import_closure"] = closure
    write_json(path, payload)
    if not closure["pass"]:
        print(f"[p2] {mode}: import-closure STOP: {closure['violations']} "
              f"(outputs quarantined, §8.2)", file=sys.stderr, flush=True)
        return 2
    return 0


def _guard(fn, args, *, mode):
    started = time.time()
    try:
        return fn(args, started)
    except SystemExit:
        raise
    except Exception as problem:
        payload = {"registered": False, "extension": p2x.EXTENSION, "mode": mode,
                   "error": f"{type(problem).__name__}: {problem}",
                   "wall_seconds": round(time.time() - started, 3)}
        out = getattr(args, "out", None)
        if out is not None:
            try:
                write_json(_out_json(out, "error.json"), payload)
            except Exception:  # pragma: no cover - best effort
                pass
        print(f"[p2] {mode} failed: {payload['error']}", file=sys.stderr, flush=True)
        return 2


# --------------------------------------------------------------------- dry-run
def mode_dry_run(args) -> int:
    started = time.time()
    root = repo_root(args)
    gate = p2x.assert_authorised("dry-run", root=root, dry_run=True)
    payload = {"registered": False, "extension": p2x.EXTENSION, "mode": "dry-run",
               "runbook": p2x.RUNBOOK_REL, "root": str(root),
               "code_sha": lr.git_head(root), "git_status": lr.git_status(root),
               "gate": gate, "setup": {}, "samples": {}, "errors": [], "closure": None,
               "wall_seconds": None}
    out = _out_json(args.out, "dry_run.json")
    try:
        config, gen, arms_config, geometry, device = load_setup(args, device_default="cpu")
        payload["setup"]["objects"] = {
            "config": CONFIG_DEFAULT, "generator_config": GENERATOR_DEFAULT,
            "geometry": str(getattr(geometry, "name", "wcte")),
            "n_pmt": int(geometry.n_pmt), "n_modules": int(geometry.n_modules)}
        payload["setup"]["inventory"] = p2x.p2_inventory(root, hash_files=False)
        payload["setup"]["gpu_projection"] = p2x.gpu_set_projection()
        payload["setup"]["cpu_projection"] = p2x.cpu_projection()
        payload["setup"]["s1"] = p2x.p2_s1_check(root)
        payload["setup"]["entries"] = p2x.entries_preflight(log=lambda *a, **k: None)
        drift = p2x.build_drift_conditions()
        payload["setup"]["conditions"] = {name: condition.as_dict()
                                          for name, condition in drift.items()}
        # sample hit-set transform on a synthetic two-hit event (no data written)
        pmt = np.array([3, 7, 11, 15, 19], np.int32)
        charge = np.array([1.5, 2.0, 3.0, 8.0, 12.0], np.float32)
        times = np.array([0.0, 1.0, 2.0, 3.0, 5.0], np.float32)
        identity = p2x.transform_event(p2x.identity_condition(), "sample", pmt, charge,
                                       times, geometry, gen, p2x.pmt_rule())
        payload["samples"]["identity"] = {
            "kept": None if identity is None else int(identity["pmt_id"].size),
            "trigger": None if identity is None else identity["trig_rel"]}
        for name in ("T0.5", "T1.0", "T1.0w", "K5"):
            result = p2x.transform_event(drift[name], "sample", pmt, charge, times,
                                         geometry, gen, p2x.pmt_rule())
            payload["samples"][name] = {"kept": None if result is None
                                        else int(result["pmt_id"].size)}
        # sample probe fit, R2 design, LEACE erasure and the P2-e statistic
        rng = np.random.default_rng(11)
        rows = {"events": rng.standard_normal((60, 8)),
                "run_ids": np.repeat([f"run{i}" for i in range(5)], 12),
                "event_ids": np.arange(60), "vertex": rng.standard_normal((60, 3)),
                "log_yield": rng.standard_normal(60), "n_hit": rng.integers(0, 40, 60)}
        fitted = p2x._raw_vertex_probe(rows)
        payload["samples"]["probe"] = {"status": fitted.get("status"),
                                       "n_targets": len(fitted.get("fits") or [])}
        design = p2x.r2_design(rows, drift["K5"],
                               {run: np.full(int(geometry.n_modules), 0.1)
                                for run in np.unique(rows["run_ids"])},
                               float(np.mean(rows["n_hit"])), geometry)
        payload["samples"]["r2_design_shape"] = list(design.shape)
        x = rng.standard_normal((80, 16)); z = rng.standard_normal((80, 1)) @ np.ones((1, 16))
        eraser = p2x.er.leace_fit(x, z, rank=1)
        erased = p2x.er.leace_erase(eraser, x)
        payload["samples"]["leace"] = {"rank": int(eraser["rank"]),
                                       "relative_shift": float(
                                           np.linalg.norm(erased - x) / np.linalg.norm(x))}
        toy_x = np.array([0.1, 0.4, 0.5, 0.7, 0.9]); toy_y = np.array([0.2, 0.3, 0.6, 0.8, 0.85])
        payload["samples"]["spearman"] = p2x.permutation_spearman(toy_x, toy_y, level="dry",
                                                                  permutations=200)
        payload["samples"]["p2e_class"] = p2x.p2e_class({
            level: {"rho_s": 0.6, "p": 0.01, "n": 10}
            for level in ("K5", "K15", "K30")})
        # plan presence: a report cannot run without the stage payloads
        payload["setup"]["preflight_plan_present"] = (
            Path(p2x.EXT_ROOT) / "x0" / "plan.json").is_file()
    except SystemExit:
        raise
    except Exception as problem:
        payload["errors"].append(f"{type(problem).__name__}: {problem}")
    payload["wall_seconds"] = round(time.time() - started, 3)
    code = _closure_end(payload, out, root, mode="dry-run")
    return 2 if (payload["errors"] or code != 0) else 0


# ------------------------------------------------------------------- GPU modes
def mode_reproduce(args, started) -> int:
    config, gen, arms_config, geometry, device = load_setup(args)
    payload = p2x.reproduce(repo_root(args), config=config, gen=gen,
                            arms_config=arms_config, geometry=geometry,
                            data_root=data_root(args), device=device,
                            limit=args.limit)
    return _finish(payload, args, "reproduce")


def mode_new_models(args, started) -> int:
    config, gen, arms_config, geometry, device = load_setup(args)
    payload = p2x.new_models(repo_root(args), config=config, gen=gen,
                             arms_config=arms_config, geometry=geometry,
                             data_root=data_root(args), device=device,
                             limit=args.limit)
    return _finish(payload, args, "new-models")


def mode_conditions(args, started) -> int:
    config, gen, arms_config, geometry, device = load_setup(args)
    payload = p2x.conditions(repo_root(args), config=config, gen=gen,
                             arms_config=arms_config, geometry=geometry,
                             data_root=data_root(args), device=device,
                             limit=args.limit)
    return _finish(payload, args, "conditions")


# ------------------------------------------------------------------- CPU modes
def mode_preflight(args, started) -> int:
    config, gen, arms_config, geometry, device = load_setup(args, device_default="cpu")
    payload = p2x.preflight(repo_root(args), config=config, gen=gen,
                            arms_config=arms_config, geometry=geometry,
                            data_root=data_root(args), device=device, limit=args.limit)
    return _finish(payload, args, "preflight")


def mode_p2(args, started) -> int:
    config, gen, arms_config, geometry, device = load_setup(args, device_default="cpu")
    payload = p2x.p2(repo_root(args), config=config, gen=gen,
                     arms_config=arms_config, geometry=geometry,
                     data_root=data_root(args), device=device, limit=args.limit)
    return _finish(payload, args, "p2")


def mode_x3p(args, started) -> int:
    config, gen, arms_config, geometry, device = load_setup(args, device_default="cpu")
    payload = p2x.x3p(repo_root(args), config=config, gen=gen,
                      arms_config=arms_config, geometry=geometry,
                      data_root=data_root(args), device=device, limit=args.limit)
    return _finish(payload, args, "x3p")


def mode_report(args, started) -> int:
    payload = p2x.report(repo_root(args))
    return _finish(payload, args, "report")


def _finish(payload, args, mode) -> int:
    out = getattr(args, "out", None)
    if out:
        write_json(_out_json(out, f"{mode}.json"), payload)
    root = repo_root(args)
    closure = p2x.import_closure(root)
    if not closure["pass"]:
        print(f"[p2] {mode}: import-closure STOP: {closure['violations']}",
              file=sys.stderr)
        return 2
    if payload.get("pass") is False:
        return 2
    return 0


# ------------------------------------------------------------------------ CLI
def _common(parser):
    parser.add_argument("--root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--generator-config", default=None)
    parser.add_argument("--arms-config", default=None)
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--limit", type=int, default=None)


def parser():
    p = argparse.ArgumentParser(prog="jepa_ext_p2")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--out", required=True)
    _common(d)
    d.add_argument("--device", default="cpu")
    for name, default_device in (("preflight", "cpu"), ("reproduce", "auto"),
                                 ("new-models", "auto"), ("conditions", "auto"),
                                 ("p2", "cpu"), ("x3p", "cpu"), ("report", "cpu")):
        mode = sub.add_parser(name)
        mode.add_argument("--out", default=None)
        _common(mode)
        mode.add_argument("--device", default=default_device)
    return p


BODIES = {"dry-run": mode_dry_run, "preflight": mode_preflight,
          "reproduce": mode_reproduce, "new-models": mode_new_models,
          "conditions": mode_conditions, "p2": mode_p2, "x3p": mode_x3p,
          "report": mode_report}


def main(argv=None):
    args = parser().parse_args(argv)
    body = BODIES[args.mode]
    if args.mode == "dry-run":
        return body(args)
    return _guard(body, args, mode=args.mode)


if __name__ == "__main__":
    sys.exit(main())
