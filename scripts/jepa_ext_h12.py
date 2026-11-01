# SPDX-License-Identifier: MIT
"""H1/H2 causal-test extension CLI (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-01_H12_CAUSAL.md`` (1 Nov 2026, reviewed).

Modes
-----
preflight  §3.4 read-only preflight of the three development dataset keys (no
           regeneration; absent/different -> exit 2 and stop).
e0e1       Phase 0: E0-a/b/c and E1 on the ``fix9000`` (and rebuilt ``init9000``)
           encoders; evaluation only.
smoke      one 20-step smoke per patched variant (``--probe-max-events 200``),
           out ``<ext>/smoke/<variant>/``; finite ``ext_summary.json``.
train      one approved run: E2 ``geomblind`` seeds 9000/9001 or E3
           ``frozen_target`` seeds 9000/9001, each with its own output root.
evaluate   §3.3: evaluate every ``final.pt`` with the Q1/Q2 readout protocol
           (E2 with ``--zero-geometry``), §4.2 reproduction (exit 2 on mismatch).
report     §7: ``results/jepa/20261101_h12/`` with ``README.md``,
           ``e0e1.json``, ``runs.json``, ``decision.json``, ``reproduction.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jepa_noise import ext_h12 as eh  # noqa: E402
from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402

CONFIG_PATH = ROOT / "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
GENERATOR_PATH = ROOT / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"
GEOMETRY_DEFAULT = "/ceph/dwong/jepa/external/LUCiD/config/WCTE_geom_config.json"
DATA_ROOT_DEFAULT = eh.DATA_ROOT_DEFAULT
EXT_ROOT_DEFAULT = str(eh.EXT_ROOT)
READOUT_EXT_DEFAULT = str(eh.READOUT_EXT)
READOUT_RESULTS = ROOT / "results/jepa/20261101_readout"


# ----------------------------------------------------------------------------- io
def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(lr._safe(payload), indent=1, default=float) + "\n")
    tmp.replace(path)


def _split(value: str) -> list:
    return [part.strip() for part in str(value).split(",") if part.strip()]


# ---------------------------------------------------------------------- setup
def load_setup(args):
    config = lr.load_run_config(getattr(args, "config", None) or CONFIG_PATH)
    gen = lb.load_generator_config(str(getattr(args, "generator_config", None) or GENERATOR_PATH))
    arms_config = la.load_config()
    geometry = lb.load_wcte_geometry(str(getattr(args, "geometry", None) or GEOMETRY_DEFAULT))
    device = lr.configure_runtime(getattr(args, "device", "auto") or "auto")
    return config, gen, arms_config, geometry, device


def _readout_items(args) -> Path:
    return Path(args.readout_ext or READOUT_EXT_DEFAULT) / "items.json"


def _readout_matrix(args) -> dict:
    path = Path(args.readout or READOUT_RESULTS) / "matrix.json"
    return json.loads(path.read_text())


# ----------------------------------------------------------------- readout rows
def _readout_rows(matrix: dict) -> dict:
    """fix9000/fix9001/mae9000/init9000 rows from the committed Q1/Q2 results."""
    rows = {}
    for label in ("fix9000", "fix9001", "mae9000", "init9000"):
        entry = matrix["items"].get(label) or {}
        rows[label] = entry.get("forms") or {}
    if not rows["init9000"]:
        raise RuntimeError("the committed Q1/Q2 matrix.json carries no init9000 rows")
    return rows


def _full_half(forms: dict) -> tuple:
    full = forms.get("full_input") or {}
    half = forms.get("half_pass") or {}
    return full, half


# ---------------------------------------------------------------- run discovery
def discover_runs(ext_root: Path) -> dict:
    """The four approved runs grouped by variant, discovered from their sidecars."""
    found = {variant: {} for variant in eh.VARIANTS}
    for variant in eh.VARIANTS:
        base = Path(ext_root) / "train" / variant / "runs"
        if not base.is_dir():
            continue
        for run_dir in sorted(path for path in base.iterdir() if path.is_dir()):
            sidecar_path = run_dir / "variant.json"
            if not sidecar_path.is_file():
                continue
            sidecar = json.loads(sidecar_path.read_text())
            if sidecar.get("variant") != variant:
                raise RuntimeError(f"{run_dir}: sidecar variant {sidecar.get('variant')!r} "
                                   f"!= directory variant {variant!r}")
            seed = int(sidecar["seed"])
            if seed in found[variant]:
                raise RuntimeError(f"duplicate {variant} seed {seed}: {run_dir}")
            found[variant][seed] = {"run_dir": run_dir, "sidecar": sidecar}
    return found


def smoke_runs(ext_root: Path) -> dict:
    found = {}
    for variant in eh.VARIANTS:
        base = Path(ext_root) / "smoke" / variant / "runs"
        if not base.is_dir():
            continue
        for run_dir in sorted(path for path in base.iterdir() if path.is_dir()):
            summary = run_dir / "ext_summary.json"
            found[f"{variant}__{run_dir.name}"] = {
                "variant": variant, "run_dir": str(run_dir),
                "summary": json.loads(summary.read_text()) if summary.is_file() else None}
    return found


# --------------------------------------------------------------------- modes
def mode_preflight(args) -> int:
    started = time.time()
    data = eh.preflight_data(args.data_root or DATA_ROOT_DEFAULT)
    data["wall_seconds"] = round(time.time() - started, 3)
    data["code_sha"] = lr.git_head(ROOT)
    out = Path(args.out or (Path(args.ext or EXT_ROOT_DEFAULT) / "preflight.json"))
    write_json(out, data)
    print(json.dumps({"out": str(out), "pass": data["pass"],
                      "sets": {key: value["pass"] for key, value in data["sets"].items()}},
                     default=float), flush=True)
    if not data["pass"]:
        print("§3.4 data unavailable/different; stop and report (no regeneration).",
              file=sys.stderr, flush=True)
        return 2
    return 0


def mode_e0e1(args) -> int:
    config, gen, arms_config, geometry, device = load_setup(args)
    ext = Path(args.ext or EXT_ROOT_DEFAULT)
    probe = getattr(args, "position_probe", "linear") or "linear"
    base_out = Path(args.out) if args.out else (ext / "e0e1.json")
    position_out = Path(args.position_out) if args.position_out else (
        ext / "e0e1_position_probe.json")
    label_cache = Path(args.label_cache) if args.label_cache else (ext / "e0e1_rows.npz")
    code_origin = bool(getattr(args, "code_origin", False))
    code_origin_out = Path(args.code_origin_out) if getattr(args, "code_origin_out", None) else (
        ext / "e0e1_code_origin.json")
    payload = eh.e0e1(config, gen, arms_config, geometry,
                      args.data_root or DATA_ROOT_DEFAULT,
                      args.readout_items or str(_readout_items(args)),
                      (base_out if (probe == "linear" and not code_origin) else None),
                      device=device, max_label_events=int(args.max_label_events),
                      position_probe=probe,
                      position_out=(position_out if probe != "linear" else None),
                      label_cache=(label_cache if (probe != "linear" or code_origin) else None),
                      code_origin=code_origin,
                      code_origin_out=(code_origin_out if code_origin else None),
                      log=print)
    summary = {"E0": payload["E0"], "E1": payload["E1"],
               "readings": payload["readings"],
               "position_probe": payload.get("position_probe"),
               "code_origin": payload.get("code_origin")}
    print(json.dumps(summary, default=float), flush=True)
    return 0


def mode_smoke(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    steps = int(args.steps or 20)
    out_root = Path(args.out)
    result = eh.train_variant(variant=args.variant, seed=int(args.seed), out_root=out_root,
                              config=config, gen=gen, arms_config=arms_config,
                              geometry=geometry, device=device,
                              data_root=args.data_root or DATA_ROOT_DEFAULT,
                              config_path=getattr(args, "config", None) or CONFIG_PATH,
                              code_sha=lr.git_head(ROOT), status_text=lr.git_status(ROOT),
                              steps=steps, probe_max_events=int(args.probe_max_events),
                              log=print)
    summary = dict(result["summary"])
    forms = None
    error = None
    try:
        probes = eh.load_probes(config, gen, geometry, args.data_root or DATA_ROOT_DEFAULT)
        item = {"label": f"smoke_{args.variant}_{args.seed}", "arm": eh.ARM_NAME,
                "split": "within", "seed": int(args.seed), "beta": None,
                "source": str(Path(result["run_dir"]) / "final.pt")}
        metrics = er.evaluate_item(item, probes, geometry, config, arms_config,
                                   device=device, forms=("full_input", "half_pass"),
                                   chunk_events=8, max_events=int(args.probe_max_events),
                                   domain=er.PRIMARY_DOMAIN)
        forms = {form: {key: value for key, value in cell.items()
                        if key in ("vertex_resolution_m", "token_log_charge_r2",
                                   "effective_rank", "event_sigreg", "module_gain_r2")}
                 for form, cell in metrics["forms"].items()}
    except Exception as problem:  # a smoke evaluation failure must be visible, not fatal
        error = f"{type(problem).__name__}: {problem}"
    summary.update(forms=forms, evaluation_error=error,
                   wall_seconds=round(time.time() - started, 3), steps=steps,
                   probe_max_events=int(args.probe_max_events))
    write_json(Path(result["run_dir"]) / "ext_summary.json", summary)
    if error is None and forms is not None:
        finite = all(value is None or np.isfinite(float(value))
                     for cell in forms.values() for key, value in cell.items()
                     if isinstance(value, (int, float)))
        summary["metrics_finite"] = bool(summary.get("metrics_finite", True) and finite)
        write_json(Path(result["run_dir"]) / "ext_summary.json", summary)
    passed = bool(summary.get("metrics_finite")
                  and summary.get("status") == "complete"
                  and summary.get("final_checkpoint_sha256"))
    print(json.dumps({"smoke": str(result["run_dir"]), "passed": passed,
                      "status": summary.get("status"), "forms": forms,
                      "evaluation_error": error, "wall_seconds": summary["wall_seconds"]},
                     default=float), flush=True)
    if not passed:
        print("[smoke] acceptance FAILED (§4.6)", file=sys.stderr, flush=True)
        return 2
    return 0


def mode_train(args) -> int:
    config, gen, arms_config, geometry, device = load_setup(args)
    started = time.time()
    result = eh.train_variant(variant=args.variant, seed=int(args.seed),
                              out_root=Path(args.out), config=config, gen=gen,
                              arms_config=arms_config, geometry=geometry, device=device,
                              data_root=args.data_root or DATA_ROOT_DEFAULT,
                              config_path=getattr(args, "config", None) or CONFIG_PATH,
                              code_sha=lr.git_head(ROOT), status_text=lr.git_status(ROOT),
                              steps=args.steps, probe_max_events=None, log=print)
    summary = result["summary"]
    tracked = {key: value for key, value in summary.items() if key != "final_train_metrics"}
    tracked["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(args.out) / f"{result['run_dir'].name}.json", tracked)
    print(json.dumps({"run_dir": str(result["run_dir"]), "variant": args.variant,
                      "seed": int(args.seed), "steps": summary.get("steps_completed"),
                      "checkpoint": summary.get("final_checkpoint_sha256"),
                      "metrics_finite": summary.get("metrics_finite"),
                      "wall_seconds": tracked["wall_seconds"]}, default=float), flush=True)
    return 0


def _item_for(run: dict, variant: str) -> dict:
    source = Path(run["run_dir"]) / "final.pt"
    if not source.is_file():
        raise RuntimeError(f"{run['run_dir']}: final.pt is missing")
    return {"label": f"{variant}_{run['sidecar']['seed']}", "arm": eh.ARM_NAME,
            "split": "within", "seed": int(run["sidecar"]["seed"]), "beta": None,
            "source": str(source)}


def mode_evaluate(args) -> int:
    started = time.time()
    config, gen, arms_config, geometry, device = load_setup(args)
    ext = Path(args.ext or EXT_ROOT_DEFAULT)
    probes = eh.load_probes(config, gen, geometry, args.data_root or DATA_ROOT_DEFAULT)
    matrix = _readout_matrix(args)
    runs = discover_runs(ext)
    out = {"registered": False, "extension": eh.EXTENSION, "mode": "evaluate",
           "runbook": "docs/jepa/RUNBOOK_2026-11-01_H12_CAUSAL.md",
           "code_sha": lr.git_head(ROOT), "device": str(device), "runs": {},
           "reproduction": {}, "complete": False}
    for variant in eh.VARIANTS:
        for seed, run in sorted(runs[variant].items()):
            label = f"{variant}_{seed}"
            item = _item_for(run, variant)
            metrics = eh.evaluate_run(item, probes, geometry, config, arms_config,
                                      device=device,
                                      zero_geometry=(variant == "geomblind"))
            full, half = _full_half(metrics["forms"])
            out["runs"][label] = {
                "variant": variant, "seed": int(seed), "split": "within",
                "checkpoint": str(Path(run["run_dir"]) / "final.pt"),
                "checkpoint_sha256": lr.sha256_file(Path(run["run_dir"]) / "final.pt"),
                "sidecar": run["sidecar"], "zero_geometry": variant == "geomblind",
                "full_input": full, "half_pass": half,
                "ext_summary": json.loads((Path(run["run_dir"]) / "ext_summary.json").read_text())
                if (Path(run["run_dir"]) / "ext_summary.json").is_file() else None}
            write_json(Path(args.out) / "eval.json", out)
            print(f"[evaluate] {label}: full vertex="
                  f"{full.get('vertex_resolution_m')} full lq="
                  f"{(full.get('token_log_charge_r2') or {}).get('r2')} half vertex="
                  f"{half.get('vertex_resolution_m')} half lq="
                  f"{(half.get('token_log_charge_r2') or {}).get('r2')}", flush=True)
    reproduction = eh.reproduction(config, arms_config, geometry, probes,
                                   _readout_items(args), device=device)
    out["reproduction"] = reproduction
    out["complete"] = True
    out["wall_seconds"] = round(time.time() - started, 3)
    write_json(Path(args.out) / "eval.json", out)
    write_json(Path(args.out) / "reproduction.json", reproduction)
    # variants.json: the realised variant map of the extension (§3.2).
    variants = {"registered": False, "extension": eh.EXTENSION,
                "variants": {variant: {"out_root": str(ext / "train" / variant),
                                       "seeds": sorted(runs[variant])}
                             for variant in eh.VARIANTS},
                "smoke": sorted(smoke_runs(ext)),
                "run_dirs": {f"{variant}_{seed}": str(run["run_dir"])
                             for variant in eh.VARIANTS
                             for seed, run in runs[variant].items()}}
    write_json(ext / "variants.json", variants)
    print(json.dumps({"complete": True, "runs": sorted(out["runs"]),
                      "reproduction_pass": reproduction["reproduction_pass"],
                      "wall_seconds": out["wall_seconds"]}, default=float), flush=True)
    if not reproduction["reproduction_pass"]:
        print(f"[evaluate] §4.2 reproduction FAILED: {reproduction['items']}",
              file=sys.stderr, flush=True)
        return 2
    return 0


def mode_report(args) -> int:
    started = time.time()
    ext = Path(args.ext or EXT_ROOT_DEFAULT)
    out_dir = Path(args.out)
    eval_payload = json.loads(Path(args.eval or (ext / "eval.json")).read_text())
    if not eval_payload.get("complete"):
        raise SystemExit("evaluate output is not complete")
    reproduction = eval_payload["reproduction"]
    e0e1_path = Path(args.e0e1 or (ext / "e0e1.json"))
    e0e1 = json.loads(e0e1_path.read_text()) if e0e1_path.is_file() else None
    matrix = _readout_matrix(args)
    readout = _readout_rows(matrix)
    init_full, init_half = _full_half(readout["init9000"])
    # Recovery: E2-r and E3-r from the evaluated finals.
    full_by_variant, half_by_variant, recovered = {}, {}, {}
    for variant in eh.VARIANTS:
        full_rows, half_rows = {}, {}
        for label, run in eval_payload["runs"].items():
            if run["variant"] != variant:
                continue
            seed = int(run["seed"])
            full_rows[seed] = run["full_input"]
            half_rows[seed] = run["half_pass"]
        if not full_rows:
            raise SystemExit(f"no evaluated runs for variant {variant}")
        full_by_variant[variant] = full_rows
        half_by_variant[variant] = half_rows
        recovered[variant] = eh.recover(full_rows, half_rows, init_full, init_half)
    verdict = eh.decision(recovered["geomblind"], recovered["frozen_target"])
    # Per-run comparison with the paired fix row and the init baseline.
    runs_table = {}
    for label, run in eval_payload["runs"].items():
        seed = int(run["seed"])
        fix_full, fix_half = _full_half(readout.get(f"fix{seed}", {}))
        full, half = run["full_input"], run["half_pass"]
        run_dir = Path(run.get("checkpoint", "")).parent
        runs_table[label] = {
            "variant": run["variant"], "seed": seed,
            "wall_seconds": (run.get("ext_summary") or {}).get("wall_seconds"),
            "train_log_check": eh.train_log_check(run_dir),
            "checkpoint_sha256": run["checkpoint_sha256"],
            "zero_geometry": run["zero_geometry"],
            "full_input": {"vertex_resolution_m": full.get("vertex_resolution_m"),
                           "token_log_charge_r2": full.get("token_log_charge_r2"),
                           "effective_rank": full.get("effective_rank"),
                           "event_sigreg": full.get("event_sigreg"),
                           "module_gain_r2": full.get("module_gain_r2"),
                           "layout": full.get("layout")},
            "half_pass": {"vertex_resolution_m": half.get("vertex_resolution_m"),
                          "token_log_charge_r2": half.get("token_log_charge_r2"),
                          "effective_rank": half.get("effective_rank"),
                          "event_sigreg": half.get("event_sigreg"),
                          "module_gain_r2": half.get("module_gain_r2")},
            "comparison": {
                "fix_full_input_vertex_m": fix_full.get("vertex_resolution_m"),
                "init_full_input_vertex_m": init_full.get("vertex_resolution_m"),
                "init_half_pass_vertex_m": init_half.get("vertex_resolution_m"),
                "full_input_vertex_delta_vs_fix_m": (
                    None if full.get("vertex_resolution_m") is None
                    or fix_full.get("vertex_resolution_m") is None
                    else full["vertex_resolution_m"] - fix_full["vertex_resolution_m"]),
                "half_pass_vertex_delta_vs_fix_m": (
                    None if half.get("vertex_resolution_m") is None
                    or fix_half.get("vertex_resolution_m") is None
                    else half["vertex_resolution_m"] - fix_half["vertex_resolution_m"]),
            }}
    decision_payload = {"registered": False, "extension": eh.EXTENSION,
                        "runbook": "docs/jepa/RUNBOOK_2026-11-01_H12_CAUSAL.md",
                        "code_sha": lr.git_head(ROOT),
                        "E2-r": recovered["geomblind"]["verdict"],
                        "E3-r": recovered["frozen_target"]["verdict"],
                        "verdict": verdict["verdict"],
                        "recovery": recovered,
                        "readout_baselines": {
                            "init9000_full_input_vertex_m": init_full.get("vertex_resolution_m"),
                            "init9000_half_pass_vertex_m": init_half.get("vertex_resolution_m"),
                            "fix9000_full_input_vertex_m":
                                _full_half(readout["fix9000"])[0].get("vertex_resolution_m"),
                            "fix9001_full_input_vertex_m":
                                _full_half(readout["fix9001"])[0].get("vertex_resolution_m")},
                        "conjunction": verdict["conjunction"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "runs.json", {"registered": False, "extension": eh.EXTENSION,
                                       "runs": runs_table,
                                       "recovered": recovered, "decision": decision_payload})
    write_json(out_dir / "decision.json", decision_payload)
    write_json(out_dir / "reproduction.json", reproduction)
    if e0e1 is not None:
        write_json(out_dir / "e0e1.json", e0e1)
    readme = _readme(decision_payload, runs_table, e0e1, reproduction, matrix)
    (out_dir / "README.md").write_text(readme)
    sizes = {path.name: path.stat().st_size for path in sorted(out_dir.glob("*.json"))}
    print(json.dumps({"out": str(out_dir), "E2-r": decision_payload["E2-r"],
                      "E3-r": decision_payload["E3-r"],
                      "verdict": decision_payload["verdict"],
                      "reproduction_pass": reproduction["reproduction_pass"],
                      "sizes": sizes,
                      "wall_seconds": round(time.time() - started, 3)}, default=float),
          flush=True)
    ok = bool(not any(size > 1_000_000 for size in sizes.values())
              and reproduction["reproduction_pass"])
    return 0 if ok else 2


def _readme(decision: dict, runs: dict, e0e1: dict | None, reproduction: dict,
            matrix: dict) -> str:
    lines = [
        "# H1/H2 causal tests: E2 geometry feature ablation + E3 frozen target — 1 Nov 2026",
        "",
        "`registered: false`. Extension `h12_causal_2026-11-01`.",
        "Runbook: `docs/jepa/RUNBOOK_2026-11-01_H12_CAUSAL.md` (sha256 "
        "`aab229a08cd9dca6288dd0143ff0b75efa4d5e4cea0330830d56727a2c51436f`).",
        "Author-approved training: exactly E2 seeds 9000/9001, E3 seeds 9000/9001 and one "
        "20-step smoke per variant; no other training.",
        "", f"- final code SHA: `{decision.get('code_sha')}`",
        f"- §4.2 reproduction: "
        f"{'PASS' if reproduction['reproduction_pass'] else 'FAIL'}",
        "", "## §6 verdict", "",
        f"- **E2-r** (geometry feature ablation): **{decision['E2-r']}**",
        f"- **E3-r** (frozen random target): **{decision['E3-r']}**",
        f"- **decision**: **{decision['verdict']}**",
        "", "## Phase 0 priors (E0/E1)", ""]
    if e0e1:
        lines += ["| reading | value |", "|---|---|"]
        e0, e1 = e0e1.get("E0", {}), e0e1.get("E1", {})
        lines += [f"| E0-a `f_layout` | {e0.get('f_layout')} |",
                  f"| E0-b `f_event` | {e0.get('f_event')} |",
                  f"| E0-c position-only label ridge R2 | "
                  f"{(e0.get('position_only_label_ridge') or {}).get('r2')} |",
                  f"| E1 invariant fraction | {e1.get('invariant_fraction')} |",
                  f"| E1 position-explained R2 | {e1.get('position_explained_r2')} |",
                  f"| E1 label-variance ratio vs init | {e1.get('label_variance_ratio')} |",
                  f"| E1 n label rows / events | {e1.get('n_label_rows')} / "
                  f"{e1.get('n_events')} |",
                  "", f"Prior readings: {json.dumps(e0e1.get('readings'), default=float)}",
                  ""]
    else:
        lines += ["E0/E1 output is missing from `<ext>`; the Phase 0 priors are not reported.", ""]
    lines += ["## E2/E3 runs", "",
              "| run | variant | seed | wall s | full vertex m | full token R2 | "
              "half vertex m | half token R2 | zero geometry |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for label, run in runs.items():
        full, half = run["full_input"], run["half_pass"]
        lines.append(
            f"| {label} | {run['variant']} | {run['seed']} | {run.get('wall_seconds')} | "
            f"{full.get('vertex_resolution_m')} | "
            f"{(full.get('token_log_charge_r2') or {}).get('r2')} | "
            f"{half.get('vertex_resolution_m')} | "
            f"{(half.get('token_log_charge_r2') or {}).get('r2')} | {run['zero_geometry']} |")
    lines += ["", "Training-log NaN check (§4.6): "
              + ", ".join(f"{label}={row['train_log_check']['finite']}"
                          for label, row in runs.items()) + " (last logged metrics).", ""]
    lines += ["", "Baselines: `init9000` full_input vertex "
              f"{decision['readout_baselines']['init9000_full_input_vertex_m']} m, "
              "half_pass vertex "
              f"{decision['readout_baselines']['init9000_half_pass_vertex_m']} m; "
              "`fix9000` full_input vertex "
              f"{decision['readout_baselines']['fix9000_full_input_vertex_m']} m, "
              "`fix9001` "
              f"{decision['readout_baselines']['fix9001_full_input_vertex_m']} m "
              "(Q1/Q2 committed matrix).", "",
              "## §6 recovery detail", ""]
    for variant, recovered in decision["recovery"].items():
        lines.append(f"- **{variant}**: {recovered['verdict']} "
                     f"(floors {json.dumps(recovered['floors'])})")
        for seed, row in recovered["seeds"].items():
            lines.append(f"  - seed {seed}: full vertex {row['full_input_vertex_m']} m "
                         f"(ok={row['vertex_ok']}), full token R2 "
                         f"{row['full_input_token_r2']} (ok={row['charge_ok']}), half vertex "
                         f"{row['half_pass_vertex_m']} m / half R2 "
                         f"{row['half_pass_token_r2']} (half ok={row['half_ok']})")
    lines += ["", "E2 is a **feature ablation** (token order, parity mask and predictor "
              "position queries remain geometry-defined), so a positive E2 reads \"consistent "
              "with H1\". E3 also changes the SIGReg target term (the frozen label feeds both "
              "the prediction MSE and the SIGReg target term).", "",
              "## §4.2 reproduction", "",
              "| item | expected m | recomputed m | |diff| m | pass |",
              "|---|---:|---:|---:|---|"]
    for label, row in reproduction["items"].items():
        lines.append(f"| {label} | {row['expected_m']} | {row['got_m']} | "
                     f"{row['abs_diff_m']} | {row['pass']} |")
    lines += ["", "## Outputs", "",
              "- `e0e1.json` — Phase 0 priors (E0-a/b/c, E1) and their readings.",
              "- `runs.json` — per run: variant, seed, wall, full_input/half_pass rows, "
              "comparison with fix/init, recovery detail.",
              "- `decision.json` — §6 E2-r/E3-r with the actual values and the author's "
              "conjunction.",
              "- `reproduction.json` — §4.2 check (exit 2 on mismatch).",
              "- `README.md` — this file.", ""]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- parser
def _common(p):
    p.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--generator-config", default=str(GENERATOR_PATH))
    p.add_argument("--geometry", default=GEOMETRY_DEFAULT)
    p.add_argument("--readout-ext", default=READOUT_EXT_DEFAULT)
    p.add_argument("--readout", default=str(READOUT_RESULTS))
    p.add_argument("--ext", default=EXT_ROOT_DEFAULT)


def parser():
    p = argparse.ArgumentParser(prog="jepa_ext_h12")
    sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("preflight")
    f.add_argument("--out", default=None)
    _common(f)
    e = sub.add_parser("e0e1")
    e.add_argument("--out", default=None)
    e.add_argument("--readout-items", default=None)
    e.add_argument("--max-label-events", type=int, default=2000)
    e.add_argument("--position-probe", choices=eh.POSITION_PROBES, default="linear",
                   help="position-probe addendum (1 Nov 2026): linear is the committed "
                        "E0/E1 behaviour; mlp/knn add the nonlinear probes")
    e.add_argument("--position-out", default=None,
                   help="side-by-side addendum JSON (default <ext>/e0e1_position_probe.json)")
    e.add_argument("--label-cache", default=None,
                   help="label-row cache reused by the mlp/knn addendum runs "
                        "(default <ext>/e0e1_rows.npz)")
    e.add_argument("--code-origin", action="store_true",
                   help="code-origin addendum (2 Nov 2026): fit the per-PMT invariant "
                        "component m_p from per-PMT fixed features with linear/MLP/kNN")
    e.add_argument("--code-origin-out", default=None,
                   help="code-origin JSON (default <ext>/e0e1_code_origin.json)")
    _common(e)
    e.add_argument("--device", default="cpu")
    s = sub.add_parser("smoke")
    s.add_argument("--variant", required=True, choices=eh.VARIANTS)
    s.add_argument("--seed", type=int, default=9000)
    s.add_argument("--out", required=True)
    s.add_argument("--steps", type=int, default=20)
    s.add_argument("--probe-max-events", type=int, default=200)
    _common(s)
    s.add_argument("--device", default="auto")
    t = sub.add_parser("train")
    t.add_argument("--variant", required=True, choices=eh.VARIANTS)
    t.add_argument("--seed", required=True, type=int)
    t.add_argument("--out", required=True)
    t.add_argument("--steps", type=int, default=None)
    _common(t)
    t.add_argument("--device", default="auto")
    v = sub.add_parser("evaluate")
    v.add_argument("--out", required=True)
    _common(v)
    v.add_argument("--device", default="auto")
    r = sub.add_parser("report")
    r.add_argument("--out", required=True)
    r.add_argument("--eval", default=None)
    r.add_argument("--e0e1", default=None)
    _common(r)
    r.add_argument("--device", default="cpu")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    return {"preflight": mode_preflight, "e0e1": mode_e0e1, "smoke": mode_smoke,
            "train": mode_train, "evaluate": mode_evaluate,
            "report": mode_report}[args.mode](args)


if __name__ == "__main__":
    sys.exit(main())
