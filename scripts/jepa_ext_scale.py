# SPDX-License-Identifier: MIT
"""Scale-control CLI (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-02_SCALE.md`` (2 Nov 2026, reviewed, authorised).

Modes
-----
data      generate/verify the authorised ``dev_train_300k`` cohort (300 runs x 1000
          events, namespace ``lucid/dev_train_300k``, event_start 1 300 000) with the
          exact ``lr.ensure_dataset`` call; entries projection checked *before* any
          write; writes ``cohort.json``.
dry-run   authorisation gate, §3.3 data/probe preflight (read-only, never regenerate),
          §3.4 entries projection from the live free count (stop if guard HOLD or
          projected free entries < 30 000); writes ``plan.json``, exits 2 on failure.
smoke     the single authorised 20-step smoke, out ``<ext>/smoke/``.
train     the two authorised runs (seeds 9000/9001), 30 000 steps via
          ``run_one(steps=30000)``, out ``<ext>/train/scale300k/``; refuses an existing
          run directory and asserts ``resumed_from_step == 0``.
evaluate  Q1/Q2 readout protocol for the run ``final.pt`` files + §4.2 reproduction.
report    ``results/jepa/20261102_scale/``: README, runs.json, readings.json
          (``scale_r``), reproduction.json, plan.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_noise import ext_h12 as eh  # noqa: E402
from jepa_noise import ext_scale as es  # noqa: E402
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
    if not (plan.get("entries") or {}).get("pass"):
        raise SystemExit(f"{path}: the §3.4 entries projection has not passed; stop")
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
def mode_data(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    try:
        gate = es.assert_authorised(ROOT)
    except SystemExit:
        print("[data] authorisation gate FAILED (exit 2)", file=sys.stderr, flush=True)
        return 2
    live = es.live_free_entries()
    projection = es.entries_projection(live["free_entries"])
    if live.get("guard_level") == "HOLD" or not projection["pass"]:
        plan = {"registered": False, "extension": es.EXTENSION, "mode": "data",
                "gate": gate, "entries": {"live": live, "projection": projection},
                "pass": False}
        write_json(args.out, plan)
        print(f"[data] §3.4 entries preflight FAILED before write: live={live} "
              f"projection={projection}", file=sys.stderr, flush=True)
        return 2
    cohort = es.ensure_cohort(config, gen, geometry, args.data_root or DATA_ROOT_DEFAULT,
                              code_sha=lr.git_head(ROOT), log=print)
    cohort = dict(cohort, gate=gate, entries_checked=projection,
                  wall_seconds=round(time.time() - started, 3))
    write_json(args.out, cohort)
    print(json.dumps({"cohort": args.out, "key": cohort["key"],
                      "manifest_sha256": cohort["manifest_sha256"],
                      "n_events": cohort["n_events"], "n_runs": cohort["n_runs"],
                      "entries": cohort["entries"], "gb": cohort["gb"],
                      "created": cohort["created_this_invocation"],
                      "wall_seconds": cohort["wall_seconds"]}, default=float),
          flush=True)
    return 0 if cohort["validation"]["pass"] else 2


def mode_dry_run(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    data_root = args.data_root or DATA_ROOT_DEFAULT
    plan = {"registered": False, "extension": es.EXTENSION, "mode": "dry-run",
            "runbook": "docs/jepa/RUNBOOK_2026-11-02_SCALE.md",
            "code_sha": lr.git_head(ROOT), "git_status": lr.git_status(ROOT),
            "data_root": str(data_root), "steps": es.STEPS,
            "config_steps": es.CONFIG_STEPS, "wall_seconds": None,
            "floors": {"full": es.INIT_FULL_VERTEX_M, "half": es.INIT_HALF_VERTEX_M}}
    try:
        plan["gate"] = es.assert_authorised(ROOT)
    except SystemExit:
        plan["gate"] = {"authorised": False}
        plan["wall_seconds"] = round(time.time() - started, 3)
        write_json(args.out, plan)
        print("[dry-run] authorisation gate FAILED (exit 2)", file=sys.stderr, flush=True)
        return 2
    plan["cohort"] = {"expected_key": es.expected_key(),
                      "directory": str(es.cohort_directory(data_root))}
    manifest = es.cohort_directory(data_root) / "manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        plan["cohort"].update({"present": True,
                               "manifest_sha256": lr.sha256_file(manifest),
                               "validation": es.validate_cohort_manifest(payload)})
    else:
        plan["cohort"]["present"] = False
    probe_status = es.probe_sets_status(Path(data_root))
    plan["probes"] = probe_status
    items = _readout_items(args)
    plan["readout"] = {"readout_ext": str(Path(args.readout_ext or READOUT_EXT_DEFAULT)),
                       "items": str(items), "items_exists": items.is_file()}
    try:
        live = es.live_free_entries()
        projection = es.entries_projection(live["free_entries"])
        plan["entries"] = {"live": live, "projection": projection,
                           "pass": bool(live.get("guard_level") != "HOLD"
                                        and projection["pass"])}
    except Exception as problem:
        plan["entries"] = {"pass": False,
                           "error": f"{type(problem).__name__}: {problem}"}
    plan["wall_seconds"] = round(time.time() - started, 3)
    write_json(args.out, plan)
    print(json.dumps({"plan": str(args.out), "authorised": plan["gate"]["authorised"],
                      "cohort_present": plan["cohort"].get("present"),
                      "cohort_valid": (plan["cohort"].get("validation") or {}).get("pass"),
                      "probes_pass": probe_status["pass"],
                      "readout_items": plan["readout"]["items_exists"],
                      "entries": plan["entries"]}, default=float), flush=True)
    failed = (not plan["gate"].get("authorised") or not probe_status["pass"]
              or not plan["readout"]["items_exists"] or not plan["entries"].get("pass")
              or not (plan["cohort"].get("validation") or {}).get("pass", True))
    if failed:
        print("[dry-run] preflight FAILED (exit 2)", file=sys.stderr, flush=True)
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
        item = {"label": f"scale300k_smoke_{args.seed}", "arm": es.ARM_NAME,
                "split": "within", "seed": int(args.seed), "beta": None,
                "source": str(run_dir / "final.pt")}
        metrics = eh.evaluate_run(item, probes, geometry, config, arms_config,
                                  device=device, max_events=int(args.probe_max_events))
        forms = {form: {key: value for key, value in cell.items()
                        if key in ("vertex_resolution_m", "token_log_charge_r2",
                                   "effective_rank", "event_sigreg", "module_gain_r2",
                                   "layout")}
                 for form, cell in metrics["forms"].items()}
    except Exception as problem:
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
                                 status_text=lr.git_status(ROOT), steps=es.STEPS,
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
               "runbook": "docs/jepa/RUNBOOK_2026-11-02_SCALE.md",
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
    runs = eval_payload["runs"]
    if sorted(runs) != ["9000", "9001"]:
        raise SystemExit(f"report needs both seeds, got {sorted(runs)}")
    for seed, run in runs.items():
        finite = (run.get("ext_summary") or {}).get("metrics_finite")
        if finite is False:
            raise SystemExit(f"seed {seed}: non-finite training metrics")
    reading = es.scale_r(runs)
    cohort = plan.get("cohort") or {}
    runs_table = {}
    for seed, run in runs.items():
        full, half = run["full_input"], run["half_pass"]
        layout = full.get("layout") or {}
        summary = run.get("ext_summary") or {}
        runs_table[seed] = {
            "seed": int(seed), "wall_seconds": summary.get("wall_seconds"),
            "steps": summary.get("steps_completed"),
            "resumed_from_step": summary.get("resumed_from_step"),
            "metrics_finite": summary.get("metrics_finite"),
            "cohort_manifest_sha256": summary.get("cohort_manifest_sha256")
            or cohort.get("manifest_sha256"),
            "checkpoint_sha256": run.get("checkpoint_sha256"),
            "full_input": {"vertex_m": full.get("vertex_resolution_m"),
                           "token_charge_r2":
                               (full.get("token_log_charge_r2") or {}).get("r2")},
            "half_pass": {"vertex_m": half.get("vertex_resolution_m"),
                          "token_charge_r2":
                              (half.get("token_log_charge_r2") or {}).get("r2")},
            "f_layout": layout.get("f_layout"), "f_event": layout.get("f_event"),
            "eff_rank": full.get("effective_rank")}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "runs.json", {"registered": False, "extension": es.EXTENSION,
                                   "cohort": cohort, "runs": runs_table,
                                   "reading": reading})
    write_json(out / "readings.json", {"registered": False, "extension": es.EXTENSION,
                                       "criteria": reading["criteria"],
                                       "verdict": reading["verdict"],
                                       "per_seed": reading["per_seed"],
                                       "direction": reading["direction"]})
    write_json(out / "reproduction.json", reproduction)
    write_json(out / "plan.json", plan)
    readme = _readme(runs_table, reading, reproduction, plan)
    (out / "README.md").write_text(readme)
    sizes = {path.name: path.stat().st_size for path in sorted(out.glob("*.json"))}
    too_big = {name: size for name, size in sizes.items() if size > 1_000_000}
    print(json.dumps({"out": str(out), "scale_r": reading["verdict"],
                      "reproduction_pass": reproduction.get("reproduction_pass"),
                      "sizes": sizes, "too_big": too_big,
                      "wall_seconds": round(time.time() - started, 3)}, default=float),
          flush=True)
    ok = bool(not too_big and reproduction.get("reproduction_pass"))
    return 0 if ok else 2


def _readme(runs: dict, reading: dict, reproduction: dict, plan: dict) -> str:
    cohort = plan.get("cohort") or {}
    entries = plan.get("entries") or {}
    lines = [
        "# Scale control (300k events / 30k steps) — 2 Nov 2026",
        "",
        "`registered: false`. Extension `scale_2026-11-02`, variant "
        "`sigreg_jepa_scale300k` (`arm_name` stays `sigreg_jepa`).",
        "Runbook: `docs/jepa/RUNBOOK_2026-11-02_SCALE.md` (sha256 "
        "`073d1532eb603565b7deede465161c924d8eb1195b9b15dc1dad322e2d8a1c70`).",
        "Author-approved training: exactly two 30 000-step runs (seeds 9000/9001) on the new "
        "`registered=False` `dev_train_300k` cohort and one 20-step smoke; no other training.",
        "",
        f"- code SHA: `{lr.git_head(ROOT)}`",
        f"- cohort: `{cohort.get('expected_key')}` "
        f"(manifest `{_fmt(cohort.get('manifest_sha256'))}`, present="
        f"{cohort.get('present')}, valid="
        f"{(cohort.get('validation') or {}).get('pass')})",
        f"- entries projection: live {_fmt((entries.get('live') or {}).get('free_entries'))} "
        f"-> projected {_fmt((entries.get('projection') or {}).get('projected_after'))} "
        f"(floor {_fmt((entries.get('projection') or {}).get('floor'))})",
        f"- §4.2 reproduction: "
        f"{'PASS' if reproduction.get('reproduction_pass') else 'FAIL'}",
        "", "## §2 reading (two-sided, both seeds)", "",
        f"- **scale_r = {reading['verdict']}**", f"- {reading['direction']}", "",
        "| seed | wall s | steps | resumed | full vertex m | full token R2 | "
        "half vertex m | half token R2 | f_layout | f_event | eff. rank |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for seed, run in runs.items():
        lines.append(
            f"| {seed} | {_fmt(run.get('wall_seconds'), 1)} | {run.get('steps')} | "
            f"{run.get('resumed_from_step')} | {_fmt(run['full_input'].get('vertex_m'))} | "
            f"{_fmt(run['full_input'].get('token_charge_r2'))} | "
            f"{_fmt(run['half_pass'].get('vertex_m'))} | "
            f"{_fmt(run['half_pass'].get('token_charge_r2'))} | "
            f"{_fmt(run.get('f_layout'))} | {_fmt(run.get('f_event'))} | "
            f"{_fmt(run.get('eff_rank'))} |")
    lines += ["", "Criteria: " + reading["criteria"]["rule"], "",
              "The effective epochs are nominally unchanged (300k x 3.2 = 30k x 32; the "
              "loader samples with replacement), so the run separates unique-data/steps "
              "scale from the development-scale collapse.", "",
              "## §4.2 reproduction", "",
              "| item | expected m | recomputed m | |diff| m | pass |",
              "|---|---:|---:|---:|---|"]
    for label, row in (reproduction.get("items") or {}).items():
        lines.append(f"| {label} | {_fmt(row.get('expected_m'))} | "
                     f"{_fmt(row.get('got_m'))} | {_fmt(row.get('abs_diff_m'))} | "
                     f"{row.get('pass')} |")
    lines += ["", "## Outputs", "",
              "- `runs.json` — per seed: wall, steps, resumed_from_step, metrics_finite, "
              "cohort manifest sha256, full/half readout, f_layout, f_event, eff_rank.",
              "- `readings.json` — the pinned `scale_r` verdict and per-seed values.",
              "- `reproduction.json` — §4.2 check.",
              "- `plan.json` — gate, cohort, probes, readout and entries projection.",
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
    p = argparse.ArgumentParser(prog="jepa_ext_scale")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("data")
    d.add_argument("--out", required=True)
    _common(d)
    d.add_argument("--device", default="cpu")
    r = sub.add_parser("dry-run")
    r.add_argument("--out", required=True)
    _common(r)
    r.add_argument("--device", default="cpu")
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
    _common(t)
    t.add_argument("--device", default="auto")
    e = sub.add_parser("evaluate")
    e.add_argument("--run", action="append", required=True)
    e.add_argument("--out", required=True)
    _common(e)
    e.add_argument("--device", default="auto")
    v = sub.add_parser("report")
    v.add_argument("--out", required=True)
    v.add_argument("--eval", required=True)
    v.add_argument("--reproduction", default=None)
    _common(v)
    v.add_argument("--device", default="cpu")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return {"data": mode_data, "dry-run": mode_dry_run, "smoke": mode_smoke,
            "train": mode_train, "evaluate": mode_evaluate,
            "report": mode_report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
