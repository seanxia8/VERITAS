# SPDX-License-Identifier: MIT
"""View-design CLI (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md`` (2 Nov 2026, reviewed, authorised).

Modes
-----
dry-run   authorisation gate, committed dev-cohort/probe preflight (read-only), §3.5
          entries projection from the live guard count (stop on HOLD or projected
          free entries < 30 000), readout items check; writes ``plan.json``, exit 2
          on failure.
smoke     the single authorised 20-step ``module_parity`` seed-9000 smoke, out
          ``<ext>/smoke``.
train     one authorised run (``--split``, ``--seed``), out per-split root
          ``<ext>/train/views/<split>``; refuses an existing run directory and
          asserts ``resumed_from_step == 0``; ``<ext>`` derived from ``--out``.
evaluate  readout with the **registered within probe** (item split pinned to
          ``within``); writes ``<ext>/eval/<split>/eval.json`` and the root
          ``<ext>/eval/reproduction.json``.
report    requires all eight runs; writes ``results/jepa/20261103_view/`` with the
          pinned schemas (ranking + recommendation).
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
from jepa_noise import ext_view as ev  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402

CONFIG_PATH = ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
GENERATOR_PATH = ROOT / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
GEOMETRY_DEFAULT = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
DATA_ROOT_DEFAULT = ev.DATA_ROOT_DEFAULT
EXT_ROOT_DEFAULT = str(ev.EXT_ROOT)
READOUT_EXT_DEFAULT = str(ev.READOUT_EXT)
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
        raise SystemExit(f"{path}: the §3.5 entries projection has not passed; stop")
    return plan


def _ext_from_split_root(out_root: Path) -> Path:
    """``<ext>/train/views/<split>`` -> ``<ext>`` (two levels up, §3.3)."""
    out_root = Path(out_root).resolve()
    if out_root.parent.name != "views" or out_root.parent.parent.name != "train":
        raise SystemExit(f"[view] expected <ext>/train/views/<split>, got {out_root}")
    return out_root.parent.parent.parent


def _readout_floors(args) -> tuple:
    matrix = read_json(Path(args.readout or READOUT_RESULTS) / "matrix.json")
    entry = (matrix.get("items") or {}).get("init9000") or {}
    forms = entry.get("forms") or {}
    return forms.get("full_input") or {}, forms.get("half_pass") or {}


# --------------------------------------------------------------------- modes
def mode_dry_run(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    data_root = args.data_root or DATA_ROOT_DEFAULT
    plan = {"registered": False, "extension": ev.EXTENSION, "mode": "dry-run",
            "runbook": "docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md",
            "code_sha": lr.git_head(ROOT), "git_status": lr.git_status(ROOT),
            "data_root": str(data_root), "splits": list(ev.SPLIT_NAMES),
            "steps": ev.STEPS, "wall_seconds": None,
            "floors": {"full": ev.INIT_FULL_VERTEX_M, "half": ev.INIT_HALF_VERTEX_M},
            "random_half_salt": ev.RANDOM_HALF_SALT}
    try:
        plan["gate"] = ev.assert_authorised(ROOT)
    except SystemExit:
        plan["gate"] = {"authorised": False}
        plan["wall_seconds"] = round(time.time() - started, 3)
        write_json(args.out, plan)
        print("[dry-run] authorisation gate FAILED (exit 2)", file=sys.stderr, flush=True)
        return 2
    status = ev.probe_status(Path(data_root))
    plan["probes"] = status
    preflight = eh.preflight_data(Path(data_root))
    plan["data"] = preflight
    items = _readout_items(args)
    plan["readout"] = {"readout_ext": str(Path(args.readout_ext or READOUT_EXT_DEFAULT)),
                       "items": str(items), "items_exists": items.is_file()}
    try:
        live = ev.live_free_entries()
        projection = ev.entries_projection(live["free_entries"])
        plan["entries"] = {"live": live, "projection": projection,
                           "pass": bool(live.get("guard_level") != "HOLD"
                                        and projection["pass"])}
    except Exception as problem:
        plan["entries"] = {"pass": False,
                           "error": f"{type(problem).__name__}: {problem}"}
    plan["wall_seconds"] = round(time.time() - started, 3)
    write_json(args.out, plan)
    print(json.dumps({"plan": str(args.out), "authorised": plan["gate"]["authorised"],
                      "probes_pass": status["pass"], "data_pass": preflight.get("pass"),
                      "readout_items": plan["readout"]["items_exists"],
                      "entries": plan["entries"]}, default=float), flush=True)
    if (not plan["gate"].get("authorised") or not status["pass"]
            or not preflight.get("pass") or not plan["readout"]["items_exists"]
            or not plan["entries"].get("pass")):
        print("[dry-run] preflight FAILED (exit 2)", file=sys.stderr, flush=True)
        return 2
    return 0


def mode_smoke(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    split = args.split or "module_parity"
    if split not in ev.SPLIT_NAMES:
        raise SystemExit(f"[view] unknown split {split!r}")
    out = Path(args.out)
    ext_root = (Path(args.ext) if args.ext else _ext_from_split_root(
        out if out.parent.name == "views" else out))
    plan = Path(args.plan) if args.plan else ext_root / "plan.json"
    if not plan.is_file():
        raise SystemExit(f"missing {plan}; run dry-run before smoke")
    if not (read_json(plan).get("entries") or {}).get("pass"):
        raise SystemExit(f"{plan}: the §3.5 entries projection has not passed; stop")
    steps = int(args.steps or 20)
    result = ev.run_one_training(
        split=split, seed=int(args.seed), out_root=out, config=config, gen=gen,
        arms_config=arms_config, geometry=geometry, device=device,
        data_root=args.data_root or DATA_ROOT_DEFAULT,
        config_path=getattr(args, "config", None) or CONFIG_PATH,
        code_sha=lr.git_head(ROOT), status_text=lr.git_status(ROOT), steps=steps,
        probe_max_events=int(args.probe_max_events), log=print)
    run_dir = Path(result["run_dir"])
    summary = dict(result["summary"])
    forms = None
    error = None
    try:
        probes = eh.load_probes(config, gen, geometry, args.data_root or DATA_ROOT_DEFAULT)
        item = {"label": f"view_{split}_smoke_{args.seed}", "arm": ev.ARM_NAME,
                "split": "within", "seed": int(args.seed), "beta": None,
                "source": str(run_dir / "final.pt")}
        metrics = eh.evaluate_run(item, probes, geometry, config, arms_config,
                                  device=device,
                                  max_events=int(args.probe_max_events))
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
    print(json.dumps({"smoke": str(run_dir), "split": split, "passed": passed,
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
    out = Path(args.out)
    ext_root = Path(args.ext) if args.ext else _ext_from_split_root(out)
    plan = Path(args.plan) if args.plan else ext_root / "plan.json"
    if not plan.is_file():
        raise SystemExit(f"missing {plan}; run dry-run before train")
    if not (read_json(plan).get("entries") or {}).get("pass"):
        raise SystemExit(f"{plan}: the §3.5 entries projection has not passed; stop")
    expected = ev.out_root_for(ext_root, args.split).resolve()
    if out.resolve() != expected:
        raise SystemExit(f"[view] train --out must be the per-split root {expected}, "
                         f"got {out}")
    result = ev.run_one_training(
        split=args.split, seed=int(args.seed), out_root=out, config=config, gen=gen,
        arms_config=arms_config, geometry=geometry, device=device,
        data_root=args.data_root or DATA_ROOT_DEFAULT,
        config_path=getattr(args, "config", None) or CONFIG_PATH,
        code_sha=lr.git_head(ROOT), status_text=lr.git_status(ROOT), steps=None,
        probe_max_events=None, log=print)
    summary = result["summary"]
    tracked = {key: value for key, value in summary.items() if key != "final_train_metrics"}
    tracked["wall_seconds"] = round(time.time() - started, 3)
    write_json(out / f"{result['run_id']}.json", tracked)
    print(json.dumps({"run_dir": str(result["run_dir"]), "split": args.split,
                      "seed": int(args.seed), "steps": summary.get("steps_completed"),
                      "resumed_from_step": summary.get("resumed_from_step"),
                      "checkpoint": summary.get("final_checkpoint_sha256"),
                      "metrics_finite": summary.get("metrics_finite"),
                      "wall_seconds": tracked["wall_seconds"]}, default=float),
          flush=True)
    return 0 if summary.get("status") == "complete" and summary.get("metrics_finite") else 2


def mode_evaluate(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    split = args.split
    if split not in ev.SPLIT_NAMES:
        raise SystemExit(f"[view] unknown split {split!r}")
    ext_root = Path(args.ext) if args.ext else Path(args.out).parent.parent
    readout_ext = Path(args.readout_ext or READOUT_EXT_DEFAULT)
    items = _readout_items(args)
    if not readout_ext.is_dir():
        raise SystemExit(f"readout extension root {readout_ext} is missing")
    if not items.is_file():
        raise SystemExit(f"readout items.json {items} is missing")
    run_dirs = [Path(value) for value in args.run]
    for run_dir in run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"--run {run_dir} is not a directory")
    probes = eh.load_probes(config, gen, geometry, args.data_root or DATA_ROOT_DEFAULT)
    out = Path(args.out)
    runs = ev.evaluate_run_dirs(split, run_dirs, probes, geometry, config, arms_config,
                                device=device, max_events=args.max_events)
    reproduction = eh.reproduction(config, arms_config, geometry, probes, items,
                                   device=device)
    payload = {"registered": False, "extension": ev.EXTENSION, "mode": "evaluate",
               "runbook": "docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md",
               "split": split, "probe_split": "within",
               "code_sha": lr.git_head(ROOT), "device": str(device), "runs": runs,
               "reproduction": reproduction, "complete": True,
               "wall_seconds": round(time.time() - started, 3)}
    write_json(out / "eval.json", payload)
    write_json(out / "reproduction.json", reproduction)
    write_json(out.parent / "reproduction.json", reproduction)
    ev.write_variants_json(ext_root, run_dirs)
    print(json.dumps({"out": str(out), "split": split, "runs": sorted(runs),
                      "probe_split": "within",
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
    eval_root = Path(args.eval)
    plan = read_json(args.plan) if args.plan else _load_plan(args)
    reproduction = read_json(args.reproduction) if args.reproduction else (
        read_json(eval_root / "reproduction.json")
        if (eval_root / "reproduction.json").is_file() else {})
    init_full, init_half = _readout_floors(args)
    readings, runs_table = {}, {}
    for split in ev.SPLIT_NAMES:
        split_eval = eval_root / split / "eval.json"
        if not split_eval.is_file():
            raise SystemExit(f"report needs {split_eval}; missing")
        payload = read_json(split_eval)
        if not payload.get("complete"):
            raise SystemExit(f"{split_eval} is not complete")
        rows = payload.get("runs") or {}
        if sorted(rows) != ["9000", "9001"]:
            raise SystemExit(f"{split_eval}: report needs both seeds, got {sorted(rows)}")
        for seed, run in rows.items():
            summary = run.get("ext_summary") or {}
            if summary.get("metrics_finite") is False:
                raise SystemExit(f"{split} seed {seed}: non-finite training metrics")
            full, half = run["full_input"], run["half_pass"]
            layout = full.get("layout") or {}
            runs_table[f"{split}__{seed}"] = {
                "split": split, "seed": int(seed),
                "wall_seconds": summary.get("wall_seconds"),
                "steps": summary.get("steps_completed"),
                "resumed_from_step": summary.get("resumed_from_step"),
                "metrics_finite": summary.get("metrics_finite"),
                "probe_split": run.get("probe_split", "within"),
                "checkpoint_sha256": run.get("checkpoint_sha256"),
                "full_input": {"vertex_m": full.get("vertex_resolution_m"),
                               "token_charge_r2":
                                   (full.get("token_log_charge_r2") or {}).get("r2")},
                "half_pass": {"vertex_m": half.get("vertex_resolution_m"),
                              "token_charge_r2":
                                  (half.get("token_log_charge_r2") or {}).get("r2")},
                "f_layout": layout.get("f_layout"),
                "f_event": layout.get("f_event"),
                "eff_rank": full.get("effective_rank")}
        reading = ev.view_r(rows, init_full=init_full, init_half=init_half)
        reading["split"] = split
        readings[split] = reading
    ranked, recommendation = ev.rank_splits(readings)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "runs.json", {"registered": False, "extension": ev.EXTENSION,
                                   "splits": list(ev.SPLIT_NAMES),
                                   "random_half_salt": ev.RANDOM_HALF_SALT,
                                   "runs": runs_table,
                                   "readings": {split: {"verdict": readings[split]["verdict"],
                                                        "per_seed": readings[split]["per_seed"]}
                                                for split in readings},
                                   "ranking": ranked, "recommendation": recommendation})
    write_json(out / "readings.json", {"registered": False,
                                       "extension": ev.EXTENSION,
                                       "criteria": {
                                           "floors": {"full": ev.INIT_FULL_VERTEX_M,
                                                      "half": ev.INIT_HALF_VERTEX_M,
                                                      "charge_r2_min": ev.CHARGE_R2_MIN},
                                           "rule": readings[ev.SPLIT_NAMES[0]]["criteria"]["rule"]},
                                       "splits": {split: {"verdict": readings[split]["verdict"],
                                                          "per_seed": readings[split]["per_seed"]}
                                                  for split in readings},
                                       "ranking": ranked, "recommendation": recommendation})
    write_json(out / "reproduction.json", reproduction)
    write_json(out / "plan.json", plan)
    readme = _readme(runs_table, readings, ranked, recommendation, reproduction, plan)
    (out / "README.md").write_text(readme)
    sizes = {path.name: path.stat().st_size for path in sorted(out.glob("*.json"))}
    too_big = {name: size for name, size in sizes.items() if size > 1_000_000}
    print(json.dumps({"out": str(out),
                      "verdicts": {split: readings[split]["verdict"] for split in readings},
                      "ranking": [row["split"] for row in ranked],
                      "recommendation": recommendation,
                      "reproduction_pass": reproduction.get("reproduction_pass"),
                      "sizes": sizes, "too_big": too_big,
                      "wall_seconds": round(time.time() - started, 3)}, default=float),
          flush=True)
    ok = bool(not too_big and reproduction.get("reproduction_pass"))
    return 0 if ok else 2


def _readme(runs, readings, ranked, recommendation, reproduction, plan) -> str:
    lines = [
        "# View design: four candidate splits — 2 Nov 2026",
        "",
        "`registered: false`. Extension `view_design_2026-11-03`; all runs `arm_name = "
        "`sigreg_jepa`; only the runtime mask/split differs (process-level `SplitMasker` "
        "patch of `lucid_run.ViewMasker` + `point_encoder.ViewMasker`, training-only).",
        "Runbook: `docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md` (sha256 "
        "`993755d7dc77b917a29566d4cf9e79b25a0146e1364b3b91b06b76456e2920e6`).",
        "Author-approved training: exactly eight 10 000-step runs (four splits × seeds "
        "9000/9001) on the committed 100k dev cohort and one 20-step `module_parity` "
        "seed-9000 smoke; no other training and no new data (masks are runtime only).",
        "",
        f"- code SHA: `{lr.git_head(ROOT)}`",
        f"- random_half salt: {ev.RANDOM_HALF_SALT}; probe split for every evaluation: "
        f"`within` (registered)",
        f"- §4.2 reproduction: "
        f"{'PASS' if reproduction.get('reproduction_pass') else 'FAIL'}",
        "", "## Per-split reading (two-sided, both seeds)", "",
        "| split | verdict | seed | full vertex m | full token R2 | half vertex m | "
        "half token R2 | f_layout | f_event | eff. rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split in ev.SPLIT_NAMES:
        for seed in ("9000", "9001"):
            row = runs.get(f"{split}__{seed}")
            if row is None:
                continue
            lines.append(f"| {split} | {readings[split]['verdict']} | {seed} | "
                         f"{_fmt(row['full_input']['vertex_m'])} | "
                         f"{_fmt(row['full_input']['token_charge_r2'])} | "
                         f"{_fmt(row['half_pass']['vertex_m'])} | "
                         f"{_fmt(row['half_pass']['token_charge_r2'])} | "
                         f"{_fmt(row.get('f_layout'))} | {_fmt(row.get('f_event'))} | "
                         f"{_fmt(row.get('eff_rank'))} |")
    lines += ["", "| split | wall s | steps | resumed |", "|---|---:|---:|---:|"]
    for key, row in runs.items():
        lines.append(f"| {row['split']} | {_fmt(row.get('wall_seconds'), 1)} | "
                     f"{row.get('steps')} | {row.get('resumed_from_step')} |")
    lines += ["", "## Ranking and recommendation", "",
              "| rank | split | verdict | mean full vertex m | max | min |",
              "|---:|---|---|---:|---:|---:|"]
    for position, row in enumerate(ranked, start=1):
        lines.append(f"| {position} | {row['split']} | {row['verdict']} | "
                     f"{_fmt(row['mean_full_vertex_m'])} | "
                     f"{_fmt(row['max_full_vertex_m'])} | {_fmt(row['min_full_vertex_m'])} |")
    lines += ["", f"**Recommendation:** {recommendation}", "",
              "Criteria: " + readings[ev.SPLIT_NAMES[0]]["criteria"]["rule"], "",
              "Caveat (runbook §2): `half_pass` always uses the registered `within` target "
              "mask, so for `module_parity`/`random_half` it is the same fixed probe as for "
              "every other run — comparable, but not \"the pass the run saw during training\".",
              "", "## §4.2 reproduction", "",
              "| item | expected m | recomputed m | |diff| m | pass |",
              "|---|---:|---:|---:|---|"]
    for label, row in (reproduction.get("items") or {}).items():
        lines.append(f"| {label} | {_fmt(row.get('expected_m'))} | "
                     f"{_fmt(row.get('got_m'))} | {_fmt(row.get('abs_diff_m'))} | "
                     f"{row.get('pass')} |")
    lines += ["", "## Outputs", "",
              "- `runs.json` — keyed `<split>__<seed>`; wall, steps, resumed_from_step, "
              "metrics_finite, probe split, full/half vertex, token charge R², f_layout, "
              "f_event, eff_rank, readings, ranking, recommendation.",
              "- `readings.json` — per-split `view_r` verdicts/per-seed rows, ranking and "
              "recommendation (pinned schema).",
              "- `reproduction.json` — §4.2 check.",
              "- `plan.json` — gate, probes, data and entries projection.",
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
    p = argparse.ArgumentParser(prog="jepa_ext_view")
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("dry-run")
    d.add_argument("--out", required=True)
    _common(d)
    d.add_argument("--device", default="cpu")
    s = sub.add_parser("smoke")
    s.add_argument("--split", default="module_parity", choices=sorted(ev.SPLIT_NAMES))
    s.add_argument("--seed", type=int, default=9000)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=20)
    s.add_argument("--probe-max-events", type=int, default=200)
    _common(s)
    s.add_argument("--device", default="auto")
    t = sub.add_parser("train")
    t.add_argument("--split", required=True, choices=sorted(ev.SPLIT_NAMES))
    t.add_argument("--seed", required=True, type=int)
    t.add_argument("--out", required=True)
    _common(t)
    t.add_argument("--device", default="auto")
    e = sub.add_parser("evaluate")
    e.add_argument("--split", required=True, choices=sorted(ev.SPLIT_NAMES))
    e.add_argument("--run", action="append", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--max-events", type=int, default=None)
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
