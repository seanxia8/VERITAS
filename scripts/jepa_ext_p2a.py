#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paper 3 P2 addendum CLI (CPU only, ``paper3_p2_addendum_2026-11-11``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-11_P2_ADDENDUM.md``.

Modes
-----
dry-run      gate (pre-acceptance ``gate_pending_acceptance``), setup, toy
             A1/A2/A3/A6 samples and the executed-import closure.
equivalence  A4: fold-equivalence against the committed P2 payloads.
p2a          A1/A2/A3 readings from the P2 caches.
shift        A6 X3-P shift shares.
report       assemble ``results/jepa/20261111_p2a/``.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_noise import ext_p2a as p2a  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402

CONFIG_DEFAULT = str(ROOT / p2a.p2x.CONFIG_REL)
GENERATOR_DEFAULT = str(ROOT / p2a.p2x.GENERATOR_REL)
GEOMETRY_DEFAULT = p2a.p2x.GEOMETRY_DEFAULT
DATA_ROOT_DEFAULT = p2a.p2x.DATA_ROOT_DEFAULT


def write_json(path, payload) -> None:
    p2a.write_json(path, payload)


def read_json(path, default=None):
    return p2a.read_json(path, default)


DEFAULT_NAMES = {
    "dry-run": "dry_run.json",
    "equivalence": "equivalence.json",
    "p2a": "p2a.json",
    "shift": "x3p_shift.json",
    "report": "report.json",
}


def _out_json(raw, mode, *, default_dir=None) -> Path:
    """Write ``<mode>`` to the §6 file name when ``--out`` is a directory (M11)."""
    name = DEFAULT_NAMES.get(mode, f"{mode}.json")
    if raw is None:
        if default_dir is None:
            raise SystemExit("this mode needs --out")
        return Path(default_dir) / name
    path = Path(raw)
    return path if path.suffix == ".json" else path / name


def load_setup(args):
    config = lr.load_run_config(getattr(args, "config", None) or CONFIG_DEFAULT)
    gen = lb.load_generator_config(str(getattr(args, "generator_config", None)
                                       or GENERATOR_DEFAULT))
    arms = la.load_config(getattr(args, "arms_config", None) or None)
    geometry = lb.load_wcte_geometry(str(getattr(args, "geometry", None)
                                         or GEOMETRY_DEFAULT))
    return config, gen, arms, geometry


def repo_root(args) -> Path:
    return Path(getattr(args, "root", None) or ROOT).resolve()


def _finish(payload, args, mode) -> int:
    out = getattr(args, "out", None)
    if out:
        write_json(_out_json(out, mode), payload)
    root = repo_root(args)
    closure = p2a.import_closure(root)
    if not closure["pass"]:
        print(f"[p2a] {mode}: import-closure STOP: {closure['violations']}",
              file=sys.stderr)
        return 2
    if payload.get("pass") is False:
        return 2
    return 0


def _guard(fn, args, *, mode):
    started = time.time()
    try:
        return fn(args, started)
    except SystemExit:
        raise
    except Exception as problem:
        print(f"[p2a] {mode} failed: {type(problem).__name__}: {problem}",
              file=sys.stderr)
        out = getattr(args, "out", None)
        if out:
            try:
                write_json(_out_json(out, mode).with_name("error.json"),
                           {"registered": False, "extension": p2a.EXTENSION,
                            "mode": mode,
                            "error": f"{type(problem).__name__}: {problem}",
                            "wall_seconds": round(time.time() - started, 3)})
            except Exception:
                pass
        return 2


# --------------------------------------------------------------------- dry-run
def mode_dry_run(args) -> int:
    started = time.time()
    root = repo_root(args)
    gate = p2a.assert_authorised("dry-run", root=root, dry_run=True)
    payload = {"registered": False, "extension": p2a.EXTENSION, "mode": "dry-run",
               "runbook": p2a.RUNBOOK_REL, "root": str(root),
               "code_sha": lr.git_head(root), "git_status": lr.git_status(root),
               "gate": gate, "setup": {}, "samples": {}, "errors": [],
               "closure": None, "wall_seconds": None}
    out = _out_json(args.out, "dry-run")
    try:
        config, gen, arms, geometry = load_setup(args)
        payload["setup"]["objects"] = {
            "n_pmt": int(geometry.n_pmt), "n_modules": int(geometry.n_modules),
            "config": CONFIG_DEFAULT, "generator_config": GENERATOR_DEFAULT}
        p2 = p2a.committed_p2(root)
        payload["setup"]["committed_p2"] = {
            "models": len(p2.get("models") or {}),
            "primary_verdict": ((p2.get("P2-e") or {}).get("class") or {}).get(
                "verdict")}
        drift = p2a.p2x.build_drift_conditions()
        rows = {"run_ids": np.asarray([f"r{i // 12}" for i in range(60)], dtype=object),
                "event_ids": np.arange(60), "n_hit": np.linspace(1.0, 30.0, 60)}
        design_t = p2a.fair_design(rows, drift["T1.0"], None, 10.0, geometry)
        design_k = p2a.fair_design(rows, drift["K5"],
                                   {run: np.full(int(geometry.n_modules), 0.1)
                                    for run in ("r0", "r1", "r2", "r3", "r4")},
                                   10.0, geometry)
        payload["samples"]["fair_design"] = {"T_columns": int(design_t.shape[1]),
                                             "K_columns": int(design_k.shape[1])}
        toy_p2_cell = {"delta_c_m": 0.05, "v_c_m": 0.2,
                       "refit": {"v_R3_m": 0.18}, "latent": {"rho_R2": 0.4},
                       "r1": {"rho_R1": None}}
        payload["samples"]["p2f_u"] = p2a.p2f_u(toy_p2_cell, {"rho_R3u": 0.9,
                                                              "information_destroying_u": True})
        rng = np.random.default_rng(3)
        basis = p2a.er.random_control_basis(rng, 8, 2)
        payload["samples"]["shift_share"] = p2a.shift_share(
            np.eye(8), rng.standard_normal((8, 20)), basis)
        payload["samples"]["eta_rule"] = p2a.eta_readings(
            {"models": {}}, {}, {})["rule"]
        payload["setup"]["equivalence_present"] = (
            root / p2a.RESULTS_DIR / "equivalence.json").is_file()
        payload["setup"]["shift_present"] = (
            root / p2a.RESULTS_DIR / "x3p_shift.json").is_file()
    except SystemExit:
        raise
    except Exception as problem:
        payload["errors"].append(f"{type(problem).__name__}: {problem}")
    payload["wall_seconds"] = round(time.time() - started, 3)
    code = 0
    closure = p2a.import_closure(root)
    payload["closure"] = closure
    write_json(out, payload)
    if not closure["pass"]:
        return 2
    return 2 if payload["errors"] else code


# ------------------------------------------------------------------ main modes
def mode_equivalence(args, started) -> int:
    config, gen, arms, geometry = load_setup(args)
    payload = p2a.run_equivalence(repo_root(args), config=config, gen=gen,
                                  geometry=geometry, data_root=args.data_root)
    return _finish(payload, args, "equivalence")


def mode_p2a(args, started) -> int:
    config, gen, arms, geometry = load_setup(args)
    payload = p2a.run_p2a(repo_root(args), config=config, gen=gen, geometry=geometry,
                          data_root=args.data_root)
    return _finish(payload, args, "p2a")


def mode_shift(args, started) -> int:
    config, gen, arms, geometry = load_setup(args)
    payload = p2a.shift_readings(repo_root(args), config=config, gen=gen,
                                 geometry=geometry, data_root=args.data_root)
    return _finish(payload, args, "shift")


def mode_report(args, started) -> int:
    root = repo_root(args)
    results = root / p2a.RESULTS_DIR
    p2a_payload = read_json(results / "p2a.json", None)
    equivalence = read_json(results / "equivalence.json", None)
    shift = read_json(results / "x3p_shift.json", None)
    if p2a_payload is None or equivalence is None or shift is None:
        raise RuntimeError("report needs p2a.json, equivalence.json and x3p_shift.json")
    extra = read_json(results / "execution_extra.json", {}) or {}
    payload = p2a.assemble_report(root, p2a=p2a_payload, equivalence=equivalence,
                                  shift=shift, extra=extra)
    return _finish(payload, args, "report")


# ------------------------------------------------------------------------ CLI
def _common(parser):
    parser.add_argument("--root", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--generator-config", default=None)
    parser.add_argument("--arms-config", default=None)
    parser.add_argument("--geometry", default=None)


def parser():
    p = argparse.ArgumentParser(prog="jepa_ext_p2a")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--out", required=True)
    _common(d)
    for name in ("equivalence", "p2a", "shift", "report"):
        mode = sub.add_parser(name)
        mode.add_argument("--out", default=None)
        _common(mode)
    return p


BODIES = {"dry-run": mode_dry_run, "equivalence": mode_equivalence, "p2a": mode_p2a,
          "shift": mode_shift, "report": mode_report}


def main(argv=None):
    args = parser().parse_args(argv)
    body = BODIES[args.mode]
    if args.mode == "dry-run":
        return body(args)
    return _guard(body, args, mode=args.mode)


if __name__ == "__main__":
    sys.exit(main())
