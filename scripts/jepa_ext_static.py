# SPDX-License-Identifier: MIT
"""Frozen static-target CLI (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md`` (2 Nov 2026, revision 2).

Modes
-----
dry-run   authorisation gate, §3.3 data preflight, split/occupancy pin (within +
          natural) and the §2 manipulation check (zeroed columns 6-8, label event
          fraction <= 1e-3 via ``er.layout_fractions``, per-PMT variance > 0 via
          ``er.layout_state``); writes ``plan.json`` and exits 2 on failure.
smoke     the single authorised 20-step smoke, out ``<ext>/smoke/``.
train     one authorised run (seeds 9000/9001), out ``<ext>/train/static_target/``;
          refuses an existing run directory and requires ``resumed_from_step == 0``.
evaluate  Q1/Q2 readout protocol for the run ``final.pt`` files plus the §4.2
          reproduction (exit 2 on mismatch).
report    ``results/jepa/20261102_static/``: README, runs.json, readings.json
          (``static_target_r``), reproduction.json, plan.json.
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
from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import ext_static as es  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402

CONFIG_PATH = ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
GENERATOR_PATH = ROOT / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
GEOMETRY_DEFAULT = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
DATA_ROOT_DEFAULT = es.DATA_ROOT_DEFAULT
EXT_ROOT_DEFAULT = str(es.EXT_ROOT)
READOUT_EXT_DEFAULT = str(es.READOUT_EXT)
READOUT_RESULTS = ROOT / "results/jepa/20261101_readout"


# ----------------------------------------------------------------------------- io
def write_json(path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(lr._safe(payload), indent=1, default=float) + "\n")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


# ---------------------------------------------------------------------- setup
def load_setup(args):
    config = lr.load_run_config(getattr(args, "config", None) or CONFIG_PATH)
    gen = lb.load_generator_config(str(getattr(args, "generator_config", None)
                                       or GENERATOR_PATH))
    arms_config = la.load_config()
    geometry = lb.load_wcte_geometry(str(getattr(args, "geometry", None)
                                         or GEOMETRY_DEFAULT))
    device = lr.configure_runtime(getattr(args, "device", "auto") or "auto")
    return config, gen, arms_config, geometry, device


def _readout_items(args) -> Path:
    return Path(args.readout_items or (Path(args.readout_ext or READOUT_EXT_DEFAULT)
                                       / "items.json"))


def _plan_path(args) -> Path:
    return Path(args.plan) if getattr(args, "plan", None) else (
        Path(args.ext or EXT_ROOT_DEFAULT) / "plan.json")


def _load_plan(args) -> dict:
    path = _plan_path(args)
    if not path.is_file():
        raise SystemExit(f"missing {path}; run dry-run before smoke/train")
    plan = read_json(path)
    if not (plan.get("manipulation") or {}).get("pass"):
        raise SystemExit(f"{path}: the §2 manipulation check has not passed; stop")
    return plan


def _readout_floors(args) -> tuple:
    matrix = read_json(Path(args.readout or READOUT_RESULTS) / "matrix.json")
    entry = (matrix.get("items") or {}).get("init9000") or {}
    forms = entry.get("forms") or {}
    return forms.get("full_input") or {}, forms.get("half_pass") or {}


def _run_dirs(args) -> list:
    dirs = []
    for value in getattr(args, "run", None) or []:
        path = Path(value)
        if not path.is_dir():
            raise SystemExit(f"--run {path} is not a directory")
        dirs.append(path)
    if not dirs:
        raise SystemExit("evaluate needs at least one --run <run directory>")
    return dirs


# --------------------------------------------------------------------- modes
def mode_dry_run(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    data_root = args.data_root or DATA_ROOT_DEFAULT
    plan = {"registered": False, "extension": es.EXTENSION, "mode": "dry-run",
            "runbook": "docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md (revision 2)",
            "code_sha": lr.git_head(ROOT), "git_status": lr.git_status(ROOT),
            "data_root": str(data_root), "wall_seconds": None}
    try:
        plan["gate"] = es.assert_authorised(ROOT)
    except SystemExit:
        plan["gate"] = {"authorised": False}
        plan["wall_seconds"] = round(time.time() - started, 3)
        write_json(args.out, plan)
        print("[dry-run] authorisation gate FAILED (exit 2)", file=sys.stderr, flush=True)
        return 2
    plan["data"] = eh.preflight_data(Path(data_root))
    if not plan["data"].get("pass"):
        plan["wall_seconds"] = round(time.time() - started, 3)
        write_json(args.out, plan)
        print("[dry-run] §3.3 data preflight FAILED; stop and report.", file=sys.stderr,
              flush=True)
        return 2
    try:
        plan["splits"] = es.assert_static_splits(config)
    except RuntimeError as error:
        plan["splits"] = {"pass": False, "reason": str(error)}
        plan["wall_seconds"] = round(time.time() - started, 3)
        write_json(args.out, plan)
        print(f"[dry-run] split/occupancy assertion FAILED: {error}", file=sys.stderr,
              flush=True)
        return 2
    items = _readout_items(args)
    plan["readout"] = {"readout_ext": str(Path(args.readout_ext or READOUT_EXT_DEFAULT)),
                       "items_json": str(items), "items_json_exists": items.is_file(),
                       "readout_ext_exists": Path(args.readout_ext
                                                  or READOUT_EXT_DEFAULT).is_dir()}
    try:
        plan["manipulation"] = es.manipulation_check(config, gen, arms_config, geometry,
                                                     data_root, device="cpu", log=print)
    except Exception as problem:  # a failed check is visible, never silently skipped
        plan["manipulation"] = {"pass": False,
                                "error": f"{type(problem).__name__}: {problem}"}
    plan["wall_seconds"] = round(time.time() - started, 3)
    write_json(args.out, plan)
    print(json.dumps({"plan": str(args.out),
                      "authorised": plan["gate"]["authorised"],
                      "data_pass": plan["data"]["pass"],
                      "splits": plan["splits"],
                      "manipulation_pass": plan["manipulation"]["pass"],
                      "label_event_fraction":
                          plan["manipulation"].get("label_event_fraction"),
                      "per_pmt_variance": plan["manipulation"].get("per_pmt_variance")},
                     default=float), flush=True)
    if not plan["manipulation"]["pass"]:
        print("[dry-run] §2 manipulation check FAILED (exit 2); stop and report.",
              file=sys.stderr, flush=True)
        return 2
    return 0


def mode_smoke(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    _load_plan(args)
    steps = int(args.steps or 20)
    result = es.run_one_training(seed=int(args.seed), out_root=Path(args.out),
                                 config=config, gen=gen, arms_config=arms_config,
                                 geometry=geometry, device=device,
                                 data_root=args.data_root or DATA_ROOT_DEFAULT,
                                 config_path=getattr(args, "config", None) or CONFIG_PATH,
                                 code_sha=lr.git_head(ROOT),
                                 status_text=lr.git_status(ROOT), steps=steps,
                                 probe_max_events=int(args.probe_max_events),
                                 log=print)
    run_dir = Path(result["run_dir"])
    summary = dict(result["summary"])
    forms = None
    error = None
    try:
        probes = eh.load_probes(config, gen, geometry, args.data_root or DATA_ROOT_DEFAULT)
        item = {"label": f"static_target_smoke_{args.seed}", "arm": es.ARM_NAME,
                "split": "within", "seed": int(args.seed), "beta": None,
                "source": str(run_dir / "final.pt")}
        metrics = eh.evaluate_run(item, probes, geometry, config, arms_config,
                                  device=device, max_events=int(args.probe_max_events))
        forms = {form: {key: value for key, value in cell.items()
                        if key in ("vertex_resolution_m", "token_log_charge_r2",
                                   "effective_rank", "event_sigreg", "module_gain_r2",
                                   "layout")}
                 for form, cell in metrics["forms"].items()}
    except Exception as problem:  # visible, not fatal
        error = f"{type(problem).__name__}: {problem}"
    summary.update(forms=forms, evaluation_error=error,
                   wall_seconds=round(time.time() - started, 3))
    write_json(run_dir / "ext_summary.json", summary)
    passed = bool(summary.get("metrics_finite") and summary.get("status") == "complete"
                  and summary.get("steps_completed") == steps
                  and summary.get("final_checkpoint_sha256") and error is None)
    print(json.dumps({"smoke": str(run_dir), "passed": passed,
                      "steps": summary.get("steps_completed"),
                      "status": summary.get("status"), "forms": forms,
                      "evaluation_error": error,
                      "wall_seconds": summary["wall_seconds"]}, default=float), flush=True)
    if not passed:
        print("[smoke] acceptance FAILED", file=sys.stderr, flush=True)
        return 2
    return 0


def mode_train(args) -> int:
    config, gen, arms_config, geometry, device = load_setup(args)
    started = time.time()
    _load_plan(args)
    result = es.run_one_training(seed=int(args.seed), out_root=Path(args.out),
                                 config=config, gen=gen, arms_config=arms_config,
                                 geometry=geometry, device=device,
                                 data_root=args.data_root or DATA_ROOT_DEFAULT,
                                 config_path=getattr(args, "config", None) or CONFIG_PATH,
                                 code_sha=lr.git_head(ROOT),
                                 status_text=lr.git_status(ROOT), steps=args.steps,
                                 probe_max_events=None, log=print)
    summary = result["summary"]
    tracked = {key: value for key, value in summary.items()
               if key != "final_train_metrics"}
    tracked["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(args.out) / f"{result['run_id']}.json", tracked)
    print(json.dumps({"run_dir": str(result["run_dir"]), "seed": int(args.seed),
                      "steps": summary.get("steps_completed"),
                      "resumed_from_step": summary.get("resumed_from_step"),
                      "checkpoint": summary.get("final_checkpoint_sha256"),
                      "metrics_finite": summary.get("metrics_finite"),
                      "wall_seconds": tracked["wall_seconds"]}, default=float),
          flush=True)
    return 0 if summary.get("status") == "complete" and summary.get("metrics_finite") else 2


def mode_evaluate(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    readout_ext = Path(args.readout_ext or READOUT_EXT_DEFAULT)
    items = _readout_items(args)
    if not readout_ext.is_dir():
        raise SystemExit(f"readout extension root {readout_ext} is missing")
    if not items.is_file():
        raise SystemExit(f"readout items.json {items} is missing")
    out = Path(args.out)
    probes = eh.load_probes(config, gen, geometry, args.data_root or DATA_ROOT_DEFAULT)
    existing = read_json(out / "eval.json") if (out / "eval.json").is_file() else {}
    runs = dict(existing.get("runs") or {})
    run_dirs = _run_dirs(args)
    runs.update(es.evaluate_run_dirs(run_dirs, probes, geometry, config, arms_config,
                                     device=device))
    reproduction = eh.reproduction(config, arms_config, geometry, probes, items,
                                   device=device)
    payload = {"registered": False, "extension": es.EXTENSION, "mode": "evaluate",
               "runbook": "docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md (revision 2)",
               "code_sha": lr.git_head(ROOT), "device": str(device), "runs": runs,
               "reproduction": reproduction, "complete": True,
               "wall_seconds": round(time.time() - started, 3)}
    write_json(out / "eval.json", payload)
    write_json(out / "reproduction.json", reproduction)
    es.write_variants_json(Path(args.ext or EXT_ROOT_DEFAULT), run_dirs)
    print(json.dumps({"out": str(out), "runs": sorted(runs),
                      "reproduction_pass": reproduction["reproduction_pass"],
                      "wall_seconds": payload["wall_seconds"]}, default=float),
          flush=True)
    if not reproduction["reproduction_pass"]:
        print(f"[evaluate] §4.2 reproduction FAILED: {reproduction['items']}",
              file=sys.stderr, flush=True)
        return 2
    return 0


def _fmt(value, digits=6):
    if value is None:
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def mode_report(args) -> int:
    started = time.time()
    out = Path(args.out)
    eval_payload = read_json(Path(args.eval) / "eval.json")
    if not eval_payload.get("complete"):
        raise SystemExit("evaluate output is not complete")
    reproduction = read_json(args.reproduction) if args.reproduction else (
        eval_payload.get("reproduction") or {})
    plan = read_json(args.plan) if args.plan else _load_plan(args)
    init_full, init_half = _readout_floors(args)
    runs = eval_payload["runs"]
    if sorted(runs) != ["9000", "9001"]:
        raise SystemExit(f"report needs both seeds, got {sorted(runs)}")
    reading = es.static_target_r(runs)
    runs_table = {}
    for seed, run in runs.items():
        full, half = run["full_input"], run["half_pass"]
        layout = full.get("layout") or {}
        runs_table[seed] = {
            "variant": run["variant"], "seed": int(seed), "run_id": run.get("run_id"),
            "wall_seconds": (run.get("ext_summary") or {}).get("wall_seconds"),
            "checkpoint_sha256": run.get("checkpoint_sha256"),
            "resumed_from_step": (run.get("ext_summary") or {}).get("resumed_from_step"),
            "full_input": {"vertex_resolution_m": full.get("vertex_resolution_m"),
                           "token_log_charge_r2": full.get("token_log_charge_r2"),
                           "effective_rank": full.get("effective_rank"),
                           "participation_ratio": full.get("participation_ratio"),
                           "event_sigreg": full.get("event_sigreg"),
                           "module_gain_r2": full.get("module_gain_r2"),
                           "f_layout": layout.get("f_layout"),
                           "f_event": layout.get("f_event")},
            "half_pass": {"vertex_resolution_m": half.get("vertex_resolution_m"),
                          "token_log_charge_r2": half.get("token_log_charge_r2"),
                          "effective_rank": half.get("effective_rank")},
            "baselines": {"init_full_input_vertex_m":
                              init_full.get("vertex_resolution_m"),
                          "init_half_pass_vertex_m":
                              init_half.get("vertex_resolution_m")}}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "runs.json", {"registered": False, "extension": es.EXTENSION,
                                   "manipulation": plan.get("manipulation"),
                                   "splits": plan.get("splits"),
                                   "runs": runs_table, "reading": reading})
    write_json(out / "readings.json", {"registered": False, "extension": es.EXTENSION,
                                       "static_target_r": reading})
    write_json(out / "reproduction.json", reproduction)
    write_json(out / "plan.json", plan)
    readme = _readme(runs_table, reading, reproduction, plan)
    (out / "README.md").write_text(readme)
    sizes = {path.name: path.stat().st_size for path in sorted(out.glob("*.json"))}
    too_big = {name: size for name, size in sizes.items() if size > 1_000_000}
    print(json.dumps({"out": str(out), "static_target_r": reading["verdict"],
                      "reproduction_pass": reproduction.get("reproduction_pass"),
                      "sizes": sizes, "too_big": too_big,
                      "wall_seconds": round(time.time() - started, 3)}, default=float),
          flush=True)
    ok = bool(not too_big and reproduction.get("reproduction_pass"))
    return 0 if ok else 2


def _readme(runs: dict, reading: dict, reproduction: dict, plan: dict) -> str:
    manipulation = plan.get("manipulation") or {}
    lines = [
        "# Frozen static-target test (target-content ablation) — 2 Nov 2026",
        "",
        "`registered: false`. Extension `static_target_2026-11-02`, variant "
        "`sigreg_jepa_static_target` (`arm_name` stays `sigreg_jepa`).",
        "Runbook: `docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md` (revision 2, sha256 "
        "`516e408e8ba35d4f1de475afefb4420e726a3386d0824a7462c20414ddb5b456`).",
        "Author-approved training: exactly two runs (seeds 9000/9001) and one 20-step "
        "smoke; no other training. The frozen seed-0 random encoder (the same net as E3's "
        "frozen target) produces the label from geometry-only features (event columns 6-8 "
        "zeroed in the target pass); the context encoder keeps the full features.",
        "", f"- code SHA: `{lr.git_head(ROOT)}`",
        f"- §4.2 reproduction: "
        f"{'PASS' if reproduction.get('reproduction_pass') else 'FAIL'}",
        "",
        "## §2 manipulation check (implementation guard; within + natural occupancy)", "",
        "| check | value |", "|---|---|"]
    for key, value in (manipulation.get("checks") or {}).items():
        lines.append(f"| {key} | {value} |")
    lines += [f"| label event fraction | {_fmt(manipulation.get('label_event_fraction'))} "
              f"(max {_fmt(manipulation.get('label_event_fraction_max'), 1)}) |",
              f"| per-PMT label variance | {_fmt(manipulation.get('per_pmt_variance'))} |",
              f"| static deviation (max abs) | "
              f"{_fmt(manipulation.get('static_deviation_max_abs'))} |",
              f"| pass | {manipulation.get('pass')} |", ""]
    lines += ["## §2 reading (two-sided, descriptive contrast against E3)", "",
              f"- **static_target_r = {reading['verdict']}**", f"- {reading['direction']}",
              "",
              "**Asymmetry caveat (runbook §1).** With a static detached label the MSE "
              "optimum is a constant output, so the prediction loss gives the context "
              "encoder no event-specific signal and the SIGReg target term is a constant "
              "with zero gradient (only the context half regularises). `recovered` is "
              "therefore nearly unreachable and would more likely signal a leak; "
              "`not recovered` cannot distinguish staticity from a mere absence of target "
              "event-information. This is not a verdict about H2.",
              "",
              "| seed | full vertex m | full token R2 | half vertex m | half token R2 | "
              "f_layout | f_event | eff. rank |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, run in runs.items():
        full, half = run["full_input"], run["half_pass"]
        lines.append(f"| {seed} | {_fmt(full.get('vertex_resolution_m'))} | "
                     f"{_fmt((full.get('token_log_charge_r2') or {}).get('r2'))} | "
                     f"{_fmt(half.get('vertex_resolution_m'))} | "
                     f"{_fmt((half.get('token_log_charge_r2') or {}).get('r2'))} | "
                     f"{_fmt(full.get('f_layout'))} | {_fmt(full.get('f_event'))} | "
                     f"{_fmt(full.get('effective_rank'))} |")
    lines += ["", "Floors/criteria: " + reading["criteria"], "",
              "## §4.2 reproduction", "",
              "| item | expected m | recomputed m | |diff| m | pass |",
              "|---|---:|---:|---:|---|"]
    for label, row in (reproduction.get("items") or {}).items():
        lines.append(f"| {label} | {_fmt(row.get('expected_m'))} | "
                     f"{_fmt(row.get('got_m'))} | {_fmt(row.get('abs_diff_m'))} | "
                     f"{row.get('pass')} |")
    lines += ["", "## Outputs", "",
              "- `runs.json` — per seed: wall, full/half readout, f_layout, f_event, "
              "effective rank, resumed_from_step; the manipulation check and split pin.",
              "- `readings.json` — the pinned `static_target_r` reading.",
              "- `reproduction.json` — §4.2 check.",
              "- `plan.json` — gate, data preflight, split/occupancy, manipulation check.",
              "- `README.md` — this file.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- parser
def _common(p):
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--generator-config", default=str(GENERATOR_PATH))
    p.add_argument("--geometry", default=GEOMETRY_DEFAULT)
    p.add_argument("--ext", default=EXT_ROOT_DEFAULT)
    p.add_argument("--readout-ext", default=READOUT_EXT_DEFAULT)
    p.add_argument("--readout-items", default=None)
    p.add_argument("--readout", default=str(READOUT_RESULTS))
    p.add_argument("--plan", default=None)


def parser():
    p = argparse.ArgumentParser(prog="jepa_ext_static")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--out", required=True)
    _common(d)
    d.add_argument("--device", default="cpu")
    s = sub.add_parser("smoke")
    s.add_argument("--seed", type=int, default=9000)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=20)
    s.add_argument("--probe-max-events", type=int, default=200)
    _common(s)
    s.add_argument("--device", default="auto")
    t = sub.add_parser("train")
    t.add_argument("--seed", required=True, type=int)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=None)
    _common(t)
    t.add_argument("--device", default="auto")
    e = sub.add_parser("evaluate")
    e.add_argument("--run", action="append", required=True)
    e.add_argument("--out", required=True)
    _common(e)
    e.add_argument("--device", default="auto")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--eval", required=True)
    r.add_argument("--reproduction", default=None)
    _common(r)
    r.add_argument("--device", default="cpu")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return {"dry-run": mode_dry_run, "smoke": mode_smoke, "train": mode_train,
            "evaluate": mode_evaluate, "report": mode_report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
