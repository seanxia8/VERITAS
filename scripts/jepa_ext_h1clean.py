# SPDX-License-Identifier: MIT
"""Clean H1 causal-test CLI: predictor position-query noise (extension,
``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md`` (2 Nov 2026, revision 3).

Modes
-----
dry-run   hard-gate check, §3.5 data preflight, checkpoint/hash manifest and the
          §3.4 σ manipulation check (untrained seed-9000 arm, one fixed saved
          batch: patched/unpatched prediction MSE must be >= 1.5; if σ=0.10 fails
          the pre-declared ladder allows σ=0.15 once, otherwise exit 2).
smoke     the single authorised 20-step smoke, out ``<ext>/smoke/``.
train     one authorised run (seeds 9000/9001), out ``<ext>/train/query_noise/``;
          refuses an existing run directory and requires ``resumed_from_step == 0``.
evaluate  Q1/Q2 readout protocol for the run ``final.pt`` files (full_input +
          half_pass, photon domain) plus the §4.2 reproduction (exit 2 on mismatch).
report    ``results/jepa/20261102_h1clean/``: README, runs.json, readings.json
          (H1-clean-r), reproduction.json, plan.json.
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

from jepa_noise import ext_h1clean as ec  # noqa: E402
from jepa_noise import ext_h12 as eh  # noqa: E402
from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402

CONFIG_PATH = ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
GENERATOR_PATH = ROOT / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
GEOMETRY_DEFAULT = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
DATA_ROOT_DEFAULT = ec.DATA_ROOT_DEFAULT
EXT_ROOT_DEFAULT = str(ec.EXT_ROOT)
READOUT_EXT_DEFAULT = str(ec.READOUT_EXT)
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


def _split(value: str) -> list:
    return [part.strip() for part in str(value).split(",") if part.strip()]


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
    manipulation = plan.get("manipulation") or {}
    if not manipulation.get("pass") or manipulation.get("sigma_used") is None:
        raise SystemExit(f"{path}: the §3.4 manipulation check has not passed; stop")
    return plan


def _readout_floors(args) -> tuple:
    """The init9000 full/half readout rows from the committed Q1/Q2 matrix."""
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
    plan = {"registered": False, "extension": ec.EXTENSION, "mode": "dry-run",
            "runbook": "docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md (revision 3)",
            "code_sha": lr.git_head(ROOT), "git_status": lr.git_status(ROOT),
            "data_root": str(data_root), "wall_seconds": None}
    try:
        plan["gate"] = ec.assert_authorised(ROOT)
    except ec.AuthorisationError as error:
        plan["gate"] = {"authorised": False, "reason": str(error)}
        write_json(args.out, plan)
        print(f"[dry-run] hard gate FAILED: {error}", file=sys.stderr, flush=True)
        return 2
    plan["data"] = eh.preflight_data(Path(data_root))
    if not plan["data"].get("pass"):
        plan["wall_seconds"] = round(time.time() - started, 3)
        write_json(args.out, plan)
        print("[dry-run] §3.5 data preflight FAILED; stop and report (no regeneration).",
              file=sys.stderr, flush=True)
        return 2
    # Checkpoint/hash manifest: the Q1/Q2 items the evaluation/reproduction reuse.
    manifest = {}
    try:
        items = read_json(_readout_items(args))["items"]
        for label, item in items.items():
            row = {"arm": item.get("arm"), "split": item.get("split"),
                   "seed": item.get("seed"), "init": item.get("init")}
            source = item.get("ckpt") or item.get("source")
            if source:
                row["source"] = str(source)
                row["source_sha256"] = lr.sha256_file(Path(source))
                expected = item.get("ckpt_sha256") or item.get("expected_sha256")
                row["expected_sha256"] = expected
                row["pass"] = bool(expected is None
                                   or row["source_sha256"] == expected)
            else:
                row["pass"] = True
            manifest[label] = row
        plan["checkpoints"] = {"items_json": str(_readout_items(args)),
                               "items": manifest,
                               "pass": all(row["pass"] for row in manifest.values())}
    except (OSError, ValueError, KeyError) as error:
        plan["checkpoints"] = {"items_json": str(_readout_items(args)),
                               "pass": False, "reason": str(error)}
    if not plan["checkpoints"]["pass"]:
        plan["wall_seconds"] = round(time.time() - started, 3)
        write_json(args.out, plan)
        print(f"[dry-run] checkpoint manifest FAILED: {plan['checkpoints']}",
              file=sys.stderr, flush=True)
        return 2
    plan["manipulation"] = ec.manipulation_check(config, gen, arms_config, geometry,
                                                 data_root, device="cpu",
                                                 sigmas=ec.SIGMA_LADDER, log=print)
    plan["sigma"] = plan["manipulation"].get("sigma_used")
    plan["wall_seconds"] = round(time.time() - started, 3)
    write_json(args.out, plan)
    print(json.dumps({"plan": str(args.out),
                      "authorised": plan["gate"]["authorised"],
                      "data_pass": plan["data"]["pass"],
                      "checkpoint_pass": plan["checkpoints"]["pass"],
                      "manipulation_pass": plan["manipulation"]["pass"],
                      "sigma": plan["sigma"],
                      "ratios": [check["ratio"] for check in
                                 plan["manipulation"]["checks"]]}, default=float),
          flush=True)
    if not plan["manipulation"]["pass"]:
        print("[dry-run] §3.4 manipulation check FAILED at sigma=0.15; stop and report.",
              file=sys.stderr, flush=True)
        return 2
    return 0


def mode_smoke(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    plan = _load_plan(args)
    sigma = float(plan["manipulation"]["sigma_used"])
    steps = int(args.steps or 20)
    result = ec.run_one_training(seed=int(args.seed), out_root=Path(args.out),
                                 config=config, gen=gen, arms_config=arms_config,
                                 geometry=geometry, device=device,
                                 data_root=args.data_root or DATA_ROOT_DEFAULT,
                                 config_path=getattr(args, "config", None) or CONFIG_PATH,
                                 code_sha=lr.git_head(ROOT),
                                 status_text=lr.git_status(ROOT), sigma=sigma,
                                 steps=steps,
                                 probe_max_events=int(args.probe_max_events),
                                 log=print)
    run_dir = Path(result["run_dir"])
    summary = dict(result["summary"])
    forms = None
    error = None
    try:
        probes = eh.load_probes(config, gen, geometry, args.data_root or DATA_ROOT_DEFAULT)
        item = {"label": f"query_noise_smoke_{args.seed}", "arm": ec.ARM_NAME,
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
    print(json.dumps({"smoke": str(run_dir), "passed": passed, "sigma": sigma,
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
    plan = _load_plan(args)
    sigma = float(plan["manipulation"]["sigma_used"])
    result = ec.run_one_training(seed=int(args.seed), out_root=Path(args.out),
                                 config=config, gen=gen, arms_config=arms_config,
                                 geometry=geometry, device=device,
                                 data_root=args.data_root or DATA_ROOT_DEFAULT,
                                 config_path=getattr(args, "config", None) or CONFIG_PATH,
                                 code_sha=lr.git_head(ROOT),
                                 status_text=lr.git_status(ROOT), sigma=sigma,
                                 steps=args.steps, probe_max_events=None, log=print)
    summary = result["summary"]
    tracked = {key: value for key, value in summary.items()
               if key != "final_train_metrics"}
    tracked["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(args.out) / f"{result['run_id']}.json", tracked)
    print(json.dumps({"run_dir": str(result["run_dir"]), "seed": int(args.seed),
                      "sigma": sigma, "steps": summary.get("steps_completed"),
                      "resumed_from_step": summary.get("resumed_from_step"),
                      "checkpoint": summary.get("final_checkpoint_sha256"),
                      "metrics_finite": summary.get("metrics_finite"),
                      "wall_seconds": tracked["wall_seconds"]}, default=float),
          flush=True)
    return 0 if summary.get("status") == "complete" and summary.get("metrics_finite") else 2


def mode_evaluate(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    out = Path(args.out)
    probes = eh.load_probes(config, gen, geometry, args.data_root or DATA_ROOT_DEFAULT)
    existing = read_json(out / "eval.json") if (out / "eval.json").is_file() else {}
    runs = dict(existing.get("runs") or {})
    run_dirs = _run_dirs(args)
    runs.update(ec.evaluate_run_dirs(run_dirs, probes, geometry, config, arms_config,
                                     device=device))
    reproduction = eh.reproduction(config, arms_config, geometry, probes,
                                   _readout_items(args), device=device)
    payload = {"registered": False, "extension": ec.EXTENSION, "mode": "evaluate",
               "runbook": "docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md (revision 3)",
               "code_sha": lr.git_head(ROOT), "device": str(device),
               "runs": runs, "reproduction": reproduction, "complete": True,
               "wall_seconds": round(time.time() - started, 3)}
    write_json(out / "eval.json", payload)
    write_json(out / "reproduction.json", reproduction)
    ext_root = Path(args.ext or EXT_ROOT_DEFAULT)
    ec.write_variants_json(ext_root, run_dirs)
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
    plan = read_json(_load_plan(args))
    init_full, init_half = _readout_floors(args)
    runs = eval_payload["runs"]
    if sorted(runs) != ["9000", "9001"]:
        raise SystemExit(f"report needs both seeds, got {sorted(runs)}")
    manipulation_pass = bool(plan.get("manipulation", {}).get("pass"))
    reading = ec.reading(runs, init_full, init_half,
                         manipulation_pass=manipulation_pass)
    runs_table = {}
    for seed, run in runs.items():
        full, half = run["full_input"], run["half_pass"]
        layout = full.get("layout") or {}
        runs_table[seed] = {
            "variant": run["variant"], "seed": int(seed), "sigma": run.get("sigma"),
            "run_id": run.get("run_id"),
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
    write_json(out / "runs.json", {"registered": False, "extension": ec.EXTENSION,
                                   "runs": runs_table, "reading": reading})
    write_json(out / "readings.json", {"registered": False, "extension": ec.EXTENSION,
                                       "readings": {"H1-clean-r": reading}})
    write_json(out / "reproduction.json", reproduction)
    write_json(out / "plan.json", plan)
    readme = _readme(runs_table, reading, reproduction, plan)
    (out / "README.md").write_text(readme)
    sizes = {path.name: path.stat().st_size for path in sorted(out.glob("*.json"))}
    too_big = {name: size for name, size in sizes.items() if size > 1_000_000}
    print(json.dumps({"out": str(out), "H1-clean-r": reading["verdict"],
                      "sigma": plan.get("sigma"),
                      "reproduction_pass": reproduction.get("reproduction_pass"),
                      "sizes": sizes, "too_big": too_big,
                      "wall_seconds": round(time.time() - started, 3)}, default=float),
          flush=True)
    ok = bool(not too_big and reproduction.get("reproduction_pass"))
    return 0 if ok else 2


def _readme(runs: dict, reading: dict, reproduction: dict, plan: dict) -> str:
    manipulation = plan.get("manipulation") or {}
    lines = [
        "# H1-clean causal test: predictor position-query noise — 2 Nov 2026",
        "",
        "`registered: false`. Extension `h1_clean_2026-11-02`, variant "
        "`sigreg_jepa_query_noise` (`arm_name` stays `sigreg_jepa`).",
        "Runbook: `docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md` (revision 3, sha256 "
        "`e02214204b7a3965665faa459adc718d2f9c061860ef66a1cb84a3d1940812c6`).",
        "Author-approved training: exactly two runs (seeds 9000/9001) and one 20-step "
        "smoke; no other training. All six geometry features stay in the encoder input; "
        "i.i.d. N(0, sigma^2) noise is added to the predictor's position-query inputs "
        "during training only.",
        "", f"- code SHA: `{lr.git_head(ROOT)}`",
        f"- sigma used: **{plan.get('sigma')}** scaled units",
        f"- §4.2 reproduction: "
        f"{'PASS' if reproduction.get('reproduction_pass') else 'FAIL'}",
        "", "## §3.4 manipulation check (untrained seed-9000 arm, fixed saved batch)", "",
        "| sigma | MSE without noise | MSE with noise | ratio | pass |",
        "|---:|---:|---:|---:|---|"]
    for check in manipulation.get("checks", []):
        lines.append(f"| {check['sigma']} | {_fmt(check['mse_without_noise'])} | "
                     f"{_fmt(check['mse_with_noise'])} | {_fmt(check['ratio'], 4)} | "
                     f"{check['pass']} |")
    lines += ["", "## §6 reading (one-sided)", "",
              f"- **H1-clean-r = {reading['verdict']}** "
              f"(eh.recover: {reading['eh_recover_verdict']}; manipulation "
              f"{'passed' if reading['manipulation_pass'] else 'did not pass'}).",
              f"- {reading['interpretation']}", "",
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
    lines += ["", "Baselines: `init9000` full-input vertex "
              f"{_fmt(runs['9000']['baselines']['init_full_input_vertex_m'])} m, half-pass "
              f"{_fmt(runs['9000']['baselines']['init_half_pass_vertex_m'])} m "
              "(committed Q1/Q2 matrix).", "",
              "## §4.2 reproduction", "",
              "| item | expected m | recomputed m | |diff| m | pass |",
              "|---|---:|---:|---:|---|"]
    for label, row in (reproduction.get("items") or {}).items():
        lines.append(f"| {label} | {_fmt(row.get('expected_m'))} | "
                     f"{_fmt(row.get('got_m'))} | {_fmt(row.get('abs_diff_m'))} | "
                     f"{row.get('pass')} |")
    lines += ["", "## Outputs", "",
              "- `runs.json` — per seed: wall, full/half readout, f_layout, f_event, "
              "effective rank, resumed_from_step.",
              "- `readings.json` — the one-sided H1-clean-r reading.",
              "- `reproduction.json` — §4.2 check.",
              "- `plan.json` — data/checkpoint preflight and the manipulation check.",
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
    p = argparse.ArgumentParser(prog="jepa_ext_h1clean")
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
