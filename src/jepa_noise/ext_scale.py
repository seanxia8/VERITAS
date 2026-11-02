# SPDX-License-Identifier: MIT
"""Scale-control extension (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-02_SCALE.md`` (2 Nov 2026, reviewed, authorised).
Extension name: ``scale_2026-11-02``.

Variant ``sigreg_jepa_scale300k`` (variant label only; ``arm_name`` stays
``"sigreg_jepa"``): the development cell is trained on a new ``registered=False``
300 000-event cohort in the unused event range (namespace ``lucid/dev_train_300k``,
``event_start`` 1 300 000, 300 runs × 1 000 events) for 30 000 steps via the
``run_one(steps=30000)`` override; the config's ``training.steps`` stays 10 000.
Everything else (probe sets, hyper-parameters, cell, seeds 9000/9001) is unchanged.

The gate, cohort spec, entries preflight, thin training/evaluation wrappers and the
two-sided ``scale_r`` reading live here.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from . import ext_h12 as eh
from . import ext_readout as er
from . import lucid_bombs as lb
from . import lucid_run as lr

EXTENSION = "scale_2026-11-02"
VARIANT = "scale300k"
ARM_NAME = "sigreg_jepa"
CELL_ID = "dev_s005_within"
EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_scale_20261102")
READOUT_EXT = Path("/ceph/dwong/jepa/returns/ext_readout_20261101")
RESULTS_DIR = "results/jepa/20261102_scale"
RUNBOOK_REL = "docs/jepa/RUNBOOK_2026-11-02_SCALE.md"
DECISIONS_REL = "docs/jepa/DECISIONS.md"
#: Author line (verbatim, runbook §0) — note the substring "我授权" does NOT occur.
AUTHOR_LINE = "B 的下一批训练我全部授权"
#: B4a row of the batch plan, quoted verbatim in runbook §0.
B4A_ROW = "| B4a | scale：300k events / 30k steps，2 seeds | 2 | 3×0.6 + eval 0.15 | **≈3.9** |"
B4A_ROW_TAIL = "1.7（生成）+0.2 | 数据 6.9 GB；每个新数据集吃 entries |"
COHORT = {"namespace": "lucid/dev_train_300k", "event_start": 1300000,
          "n_events": 300000, "run_length": 1000}
COHORT_NAME = "dev_train_300k"
LAW, SETTING, SEED = "F1", 0.05, 20261017
STEPS = 30000
CONFIG_STEPS = 10000
#: §2 floors / charge threshold for the two-sided reading.
INIT_FULL_VERTEX_M = 0.160072
INIT_HALF_VERTEX_M = 0.174821
CHARGE_R2_MIN = 0.5
#: §3.4 entries projection (cohort / clone / suite / extension) and floor.
ENTRIES_COHORT = 1812
ENTRIES_CLONE = 3059
ENTRIES_SUITE = 1825
ENTRIES_EXT = 1000
ENTRIES_FLOOR = 30000
DATA_ROOT_DEFAULT = eh.DATA_ROOT_DEFAULT
JOBS_DIR = Path("/ceph/dwong/jepa/jobs")
GUARD_JSON = JOBS_DIR / "guard_deepthought2.json"
#: Probe-set manifest sha256 (committed, read-only; never regenerated).
PROBE_MANIFESTS = {"dev_probe_train__F1__0.05__s20261017":
                   "093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3",
                   "dev_probe_test__F1__0.05__s20261017":
                   "c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a"}
DEV_TRAIN_MANIFEST = "b1b3f4f401bcee48c49c347b67d7f5cbd4410213f08ef8b9d2848edc9f2cd13a"


# --------------------------------------------------------------------- gate
def _section_zero(text: str) -> str:
    start = text.find("## 0.")
    if start < 0:
        start = text.find("## 0 ")
    if start < 0:
        return ""
    end = text.find("\n## ", start + 3)
    return text[start: end if end >= 0 else None]


def assert_authorised(root=None) -> dict:
    """Hard gate (§0): the full author line + B4a row in the runbook §0 and the line
    in ``DECISIONS.md``; exits 2 otherwise.  Uses its own ``RUNBOOK_REL``.
    """
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    runbook = root / RUNBOOK_REL
    decisions = root / DECISIONS_REL
    problems = []
    if not runbook.is_file() or not decisions.is_file():
        problems.append(f"missing {runbook} or {decisions}")
    else:
        section = _section_zero(runbook.read_text())
        if AUTHOR_LINE not in section:
            problems.append(f"{RUNBOOK_REL} §0 does not contain {AUTHOR_LINE!r}")
        if B4A_ROW not in section or B4A_ROW_TAIL not in section:
            problems.append(f"{RUNBOOK_REL} §0 does not contain the B4a row text")
        if "NOT AUTHORISED" in section:
            problems.append(f"{RUNBOOK_REL} §0 still carries a NOT AUTHORISED banner")
        if AUTHOR_LINE not in decisions.read_text():
            problems.append(f"{DECISIONS_REL} does not contain {AUTHOR_LINE!r}")
    if problems:
        for problem in problems:
            print(f"[scale] authorisation gate: {problem}", file=sys.stderr)
        raise SystemExit(2)
    return {"authorised": True, "runbook": str(runbook),
            "runbook_sha256": lr.sha256_file(runbook), "decisions": str(decisions),
            "author_line": AUTHOR_LINE, "b4a_row": B4A_ROW}


# ---------------------------------------------------------------- cohort §2/§3
def cohort_spec() -> dict:
    return dict(COHORT)


def expected_key() -> str:
    return lr.dataset_key(COHORT_NAME, LAW, SETTING, SEED, None,
                          COHORT["n_events"], COHORT["run_length"], registered=False)


def cohort_directory(data_root) -> Path:
    return Path(data_root) / "lucid" / expected_key()


def validate_cohort_manifest(manifest: dict, *, key: str | None = None) -> dict:
    """Validate the 300k cohort manifest (namespace via the run event ranges)."""
    key = expected_key() if key is None else key
    runs = manifest.get("runs") or []
    event_starts = [int(row.get("event_id_start", -1)) for row in runs]
    checks = {
        "key": manifest.get("key") == key,
        "n_events": int(manifest.get("n_events", -1)) == COHORT["n_events"],
        "n_runs": int(manifest.get("n_runs", -1)) == COHORT["n_events"] // COHORT["run_length"],
        "run_length": int(manifest.get("run_length", -1)) == COHORT["run_length"],
        "registered_false": manifest.get("registered") is False,
        "event_start": bool(event_starts) and min(event_starts) == COHORT["event_start"],
        "law": manifest.get("law") == LAW,
        "setting": manifest.get("setting") is not None
        and abs(float(manifest["setting"]) - SETTING) < 1e-12,
    }
    return {"key": key, "checks": checks, "pass": all(checks.values()),
            "namespace": COHORT["namespace"], "event_start": COHORT["event_start"],
            "n_events": COHORT["n_events"], "n_runs": COHORT["n_events"] // COHORT["run_length"],
            "run_length": COHORT["run_length"], "registered": False}


def cohort_usage(directory) -> dict:
    """Entries (files + directories) and bytes of a cohort directory."""
    directory = Path(directory)
    entries = 0
    total = 0
    if directory.is_dir():
        for base, dirs, files in os.walk(directory):
            entries += len(dirs) + len(files)
            for name in files:
                try:
                    total += (Path(base) / name).stat().st_size
                except FileNotFoundError:
                    continue
    return {"entries": int(entries), "bytes": int(total), "gb": total / 1e9}


def live_free_entries(guard_json=GUARD_JSON) -> dict:
    """Live free-entry count from the guard status (ceph quota), with a fail-closed error."""
    guard_json = Path(guard_json)
    if not guard_json.is_file():
        raise RuntimeError(f"guard status {guard_json} is missing; cannot check entries")
    status = json.loads(guard_json.read_text())
    free = (status.get("caps") or {}).get("free_entries")
    if free is None:
        raise RuntimeError(f"{guard_json}: guard status has no caps.free_entries")
    return {"free_entries": int(free), "guard_level": status.get("level"),
            "guard_reasons": status.get("reasons"),
            "guard_timestamp": status.get("timestamp"),
            "guard_json": str(guard_json)}


def entries_projection(free_before: int, *, cohort: int = ENTRIES_COHORT,
                       clone: int = ENTRIES_CLONE, suite: int = ENTRIES_SUITE,
                       extension: int = ENTRIES_EXT, floor: int = ENTRIES_FLOOR) -> dict:
    """§3.4 projection: cohort + fresh clone + suite + <ext> against the free count."""
    projected = int(free_before) - (int(cohort) + int(clone) + int(suite) + int(extension))
    return {"free_before": int(free_before), "cohort": int(cohort),
            "clone": int(clone), "suite": int(suite), "extension": int(extension),
            "projected_after": int(projected), "floor": int(floor),
            "pass": bool(projected >= int(floor) and int(free_before) >= int(floor))}


def probe_sets_status(data_root) -> dict:
    """Read-only probe-set check: both directories and manifests match (no generation)."""
    root = Path(data_root) / "lucid"
    out = {"data_root": str(data_root), "sets": {}, "regeneration": "forbidden"}
    for key, expected in PROBE_MANIFESTS.items():
        directory = root / key
        manifest = directory / "manifest.json"
        row = {"directory": str(directory), "exists": directory.is_dir(),
               "manifest": str(manifest), "exists_manifest": manifest.is_file(),
               "expected_sha256": expected}
        if manifest.is_file():
            row["manifest_sha256"] = lr.sha256_file(manifest)
            row["pass"] = row["manifest_sha256"] == expected
        else:
            row["manifest_sha256"] = None
            row["pass"] = False
        out["sets"][key] = row
    dev_train = root / f"dev_train__{LAW}__{SETTING:g}__s{SEED}"
    out["dev_train_100k"] = {"directory": str(dev_train),
                             "exists_manifest": (dev_train / "manifest.json").is_file()}
    out["pass"] = all(row["pass"] for row in out["sets"].values())
    return out


def ensure_cohort(config, gen, geometry, data_root, *, code_sha=None, log=print) -> dict:
    """The exact authorised ``ensure_dataset`` call; returns the cohort record."""
    data_root = Path(data_root)
    expected_dir = cohort_directory(data_root)
    manifest = expected_dir / "manifest.json"
    created = not manifest.is_file()
    if created:
        live = live_free_entries()
        projection = entries_projection(live["free_entries"])
        if live.get("guard_level") == "HOLD" or not projection["pass"]:
            raise RuntimeError(f"§3.4 entries preflight failed before writing the cohort: "
                               f"live={live} projection={projection}")
        log(f"[scale] generating the 300k cohort: live free {live['free_entries']}, "
            f"projected after {projection['projected_after']}")
    train_ds = lr.ensure_dataset(
        data_root=data_root, geometry=geometry, generator_config=gen,
        cohort_spec=cohort_spec(), cohort_name=COHORT_NAME, law=LAW,
        setting=SETTING, seed=SEED, registered=False, code_sha=code_sha,
        verify_hashes=False, log=log)
    payload = json.loads((train_ds.directory / "manifest.json").read_text())
    validation = validate_cohort_manifest(payload, key=train_ds.key)
    if not validation["pass"]:
        raise RuntimeError(f"cohort manifest validation failed: {validation}")
    usage = cohort_usage(train_ds.directory)
    return {"registered": False, "extension": EXTENSION, "key": train_ds.key,
            "directory": str(train_ds.directory),
            "resolved_directory": str(train_ds.directory.resolve()),
            "manifest_sha256": train_ds.manifest_sha256(),
            "namespace": COHORT["namespace"], "event_start": COHORT["event_start"],
            "n_events": COHORT["n_events"], "n_runs": COHORT["n_events"] // COHORT["run_length"],
            "run_length": COHORT["run_length"], "created_this_invocation": created,
            "validation": validation, "entries": usage["entries"],
            "gb": usage["gb"], "steps_scheduled": STEPS,
            "config_steps": CONFIG_STEPS}


# ----------------------------------------------------------------- run dirs
def run_id_for(seed) -> str:
    return lr.run_id_for(CELL_ID, ARM_NAME, int(seed))


def run_dir_for(out_root, seed) -> Path:
    return Path(out_root) / "runs" / run_id_for(seed)


def prepare_run_dir(out_root, seed, *, steps=None, device=None) -> tuple:
    """Refuse an existing run directory and write the per-run variant sidecar."""
    run_id, run_dir = run_id_for(seed), run_dir_for(out_root, seed)
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} already exists; refusing to launch (run_one would "
                           f"auto-resume it). Archive or remove it first.")
    run_dir.mkdir(parents=True, exist_ok=False)
    sidecar = {"registered": False, "extension": EXTENSION, "variant": VARIANT,
               "seed": int(seed), "arm": ARM_NAME, "cell_id": CELL_ID,
               "steps": steps, "device": str(device), "out_root": str(out_root),
               "run_id": run_id, "cohort": expected_key()}
    (run_dir / "variant.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return run_id, run_dir


def summarize_run(run_dir, *, seed, wall_seconds=None, steps=None,
                  resumed_from_step=0, cohort=None, forms=None) -> dict:
    run_dir = Path(run_dir)
    check = eh.train_log_check(run_dir)
    checkpoint = run_dir / "final.pt"
    status = json.loads((run_dir / "run_status.json").read_text()) \
        if (run_dir / "run_status.json").is_file() else {}
    return {"registered": False, "extension": EXTENSION, "variant": VARIANT,
            "seed": int(seed), "run_id": run_dir.name, "arm": ARM_NAME,
            "cell_id": CELL_ID, "steps_scheduled": steps,
            "steps_completed": status.get("steps_completed"),
            "resumed_from_step": int(resumed_from_step),
            "status": status.get("outcome"),
            "cohort": (cohort or {}).get("key"),
            "cohort_manifest_sha256": (cohort or {}).get("manifest_sha256"),
            "final_train_metrics": check.get("final_metrics"),
            "non_finite_keys": check.get("non_finite_keys"),
            "metrics_finite": check.get("finite"),
            "final_checkpoint_sha256": (lr.sha256_file(checkpoint)
                                        if checkpoint.is_file() else None),
            "wall_seconds": wall_seconds, "forms": forms}


# ------------------------------------------------------------------- training
def _require_cohort(data_root) -> dict:
    directory = cohort_directory(data_root)
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise SystemExit(f"[scale] the 300k cohort manifest {manifest} is missing; run the "
                         f"data mode first")
    payload = json.loads(manifest.read_text())
    validation = validate_cohort_manifest(payload)
    if not validation["pass"]:
        raise SystemExit(f"[scale] cohort manifest validation failed: {validation}")
    return {"key": expected_key(), "directory": str(directory),
            "manifest_sha256": lr.sha256_file(manifest),
            "validation": validation}


def _require_probes(config, data_root) -> dict:
    status = probe_sets_status(data_root)
    if not status["pass"]:
        raise SystemExit(f"[scale] §3.3 probe-set preflight failed (never regenerate): "
                         f"{status}")
    preflight = eh.preflight_data(Path(data_root))
    if not preflight.get("pass"):
        raise SystemExit(f"[scale] §3.3 data preflight failed: {preflight}")
    return {"probes": status, "preflight": preflight}


def load_cohort_and_probes(config, gen, geometry, data_root, *, code_sha, log=print):
    """Verify (never regenerate) the 300k cohort and the read-only probe sets."""
    cohort = _require_cohort(data_root)
    probe_info = _require_probes(config, data_root)
    train_ds = lr.ensure_dataset(
        data_root=Path(data_root), geometry=geometry, generator_config=gen,
        cohort_spec=cohort_spec(), cohort_name=COHORT_NAME, law=LAW,
        setting=SETTING, seed=SEED, registered=False, code_sha=code_sha,
        verify_hashes=False, log=log)
    cell = lr.dev_cell(config)
    _, probes = eh.datasets_for(cell, config, gen, geometry, Path(data_root),
                                code_sha=None)
    cohort.update({"entries": cohort_usage(train_ds.directory)["entries"],
                   "gb": cohort_usage(train_ds.directory)["gb"]})
    return cohort, train_ds, probes, probe_info


def loader_config(config: dict, n_runs: int) -> dict:
    """In-memory training config whose tokenised-run cache covers the whole cohort.

    The registered config caches 128 tokenised runs.  With a 300-run cohort the
    random 8-run choice per batch misses the cache ~57 % of the time and tokenising
    a fresh run (~0.25 s) dominates the step time (>1.9 s/step, far outside the
    per-run cap).  Raising ``data.loader_cache_runs`` to the cohort size tokenises
    every run once (about 75 s of warm-up) and leaves the sampling protocol, batch
    composition, hyper-parameters and the frozen config file untouched; ``run_one``
    still records the config file's sha256.
    """
    n_runs = int(n_runs)
    current = int(config["data"].get("loader_cache_runs", 12))
    if n_runs <= current:
        return config
    import copy
    out = copy.deepcopy(config)
    out["data"]["loader_cache_runs"] = n_runs
    return out


def run_one_training(*, seed, out_root, config, gen, arms_config, geometry, device,
                     data_root, config_path, code_sha, status_text, steps=None,
                     probe_max_events=None, log=print) -> dict:
    """The authorised ``run_one`` (steps=30000 for the runs, 20 for the smoke)."""
    gate = assert_authorised()
    cohort, train_ds, probes, probe_info = load_cohort_and_probes(
        config, gen, geometry, data_root, code_sha=code_sha, log=log)
    scheduled = int(STEPS if steps is None else steps)
    if scheduled != 20 and scheduled != STEPS:
        raise ValueError(f"scale training is authorised only for {STEPS} steps "
                         f"(or the 20-step smoke), got {scheduled}")
    run_config = loader_config(config, len(train_ds.runs))
    run_id, run_dir = prepare_run_dir(out_root, seed, steps=scheduled, device=device)
    started = __import__("time").time()
    result = lr.run_one(
        arm_name=ARM_NAME, cell=lr.dev_cell(config), seed=int(seed), beta=None,
        config=run_config, arms_config=arms_config, geometry=geometry,
        loader_dataset=train_ds, probe_datasets=probes, dark_bank=None,
        out_root=Path(out_root), data_root=Path(data_root),
        config_sha=lr.sha256_file(config_path), code_sha=code_sha,
        status_text=status_text, device=device, steps=scheduled, micro_batch=None,
        resume=None, stop_after=None, checkpoint_minutes=None, evaluate=False,
        allow_pending=False, step_callback=None, probe_max_events=probe_max_events,
        save_embeddings=False, physics_method="ridge", tf32=False, log=log)
    status = json.loads((run_dir / "run_status.json").read_text())
    from . import ext_h1clean as ec
    resumed = ec.assert_fresh_resume(status, run_id)
    summary = summarize_run(run_dir, seed=seed,
                            wall_seconds=round(__import__("time").time() - started, 3),
                            steps=scheduled, resumed_from_step=resumed,
                            cohort=cohort)
    summary["smoke"] = scheduled == 20
    summary["loader_cache_runs"] = int(run_config["data"].get("loader_cache_runs", 12))
    summary["cohort_runs"] = int(len(train_ds.runs))
    summary["probes_preflight"] = {"pass": probe_info["preflight"].get("pass")}
    (run_dir / "ext_summary.json").write_text(json.dumps(summary, indent=1,
                                                         default=float) + "\n")
    return {"result": result, "run_dir": run_dir, "summary": summary,
            "run_id": run_id, "seed": int(seed), "cohort": cohort}


# ------------------------------------------------------------- evaluation §2
def evaluate_run_dirs(run_dirs, probes, geometry, config, arms_config, *, device,
                      max_events=None) -> dict:
    runs = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        sidecar_path = run_dir / "variant.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
        seed = int(sidecar.get("seed", 0))
        source = run_dir / "final.pt"
        if not source.is_file():
            raise RuntimeError(f"{run_dir}: final.pt is missing")
        item = {"label": f"scale300k_{seed}", "arm": ARM_NAME, "split": "within",
                "seed": seed, "beta": None, "source": str(source)}
        metrics = eh.evaluate_run(item, probes, geometry, config, arms_config,
                                  device=device, max_events=max_events)
        runs[str(seed)] = {
            "variant": VARIANT, "seed": seed, "run_id": run_dir.name,
            "run_dir": str(run_dir), "checkpoint_sha256": lr.sha256_file(source),
            "full_input": metrics["forms"]["full_input"],
            "half_pass": metrics["forms"]["half_pass"],
            "ext_summary": (json.loads((run_dir / "ext_summary.json").read_text())
                            if (run_dir / "ext_summary.json").is_file() else None)}
    return runs


def write_variants_json(ext_root, run_dirs) -> dict:
    variants = {"registered": False, "extension": EXTENSION, "variant": VARIANT,
                "runs": {}}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        sidecar_path = run_dir / "variant.json"
        if sidecar_path.is_file():
            variants["runs"][run_dir.name] = json.loads(sidecar_path.read_text())
    path = Path(ext_root) / "variants.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(variants, indent=1, default=float) + "\n")
    os.replace(tmp, path)
    return variants


# --------------------------------------------------------- reading scale_r §2
def scale_r(runs: dict, *, full_floor: float = INIT_FULL_VERTEX_M,
            half_floor: float = INIT_HALF_VERTEX_M,
            charge_min: float = CHARGE_R2_MIN) -> dict:
    """The pinned two-sided ``scale_r`` reading (§2), over ``eh.recover``'s rows."""
    rows = {}
    for seed, run in runs.items():
        full, half = run["full_input"], run["half_pass"]
        full_v = full.get("vertex_resolution_m")
        full_r2 = (full.get("token_log_charge_r2") or {}).get("r2")
        half_v = half.get("vertex_resolution_m")
        half_r2 = (half.get("token_log_charge_r2") or {}).get("r2")
        recovers = bool(full_v is not None and float(full_v) < float(full_floor)
                        and full_r2 is not None and float(full_r2) >= float(charge_min)
                        and half_v is not None and float(half_v) < float(half_floor)
                        and half_r2 is not None and float(half_r2) >= float(charge_min))
        fails = bool((full_v is not None and float(full_v) >= float(full_floor))
                     or full_r2 is None or float(full_r2) < float(charge_min))
        rows[str(seed)] = {"full_input_vertex_m": full_v,
                           "full_input_token_r2": full_r2,
                           "half_pass_vertex_m": half_v,
                           "half_pass_token_r2": half_r2,
                           "recovers": recovers, "fails": fails}
    values = list(rows.values())
    if values and all(row["recovers"] for row in values):
        verdict = "recovered"
        direction = ("recovered: the collapse does not persist at 3x data and 3x steps "
                     "-> at least partly a scale artifact (pre-declared follow-up: 100k "
                     "events at 30k steps)")
    elif values and all(row["fails"] for row in values):
        verdict = "not recovered"
        direction = ("not recovered: the collapse persists at 3x data and 3x steps "
                     "(optional pre-declared follow-up: 300k events at 10k steps)")
    else:
        verdict = "unclear"
        direction = "unclear (partial recovery, seed disagreement or half-pass contradiction)"
    return {"registered": False, "extension": EXTENSION, "id": "scale_r",
            "question": "is the collapse a scale artifact? (two-sided, both seeds)",
            "criteria": {"floors": {"full_input_m": float(full_floor),
                                    "half_pass_m": float(half_floor)},
                         "charge_r2_min": float(charge_min),
                         "rule": "recovered: both seeds full vertex < %.6f, full token "
                                 "R2 >= %.1f, half vertex < %.6f, half token R2 >= %.1f; "
                                 "not recovered: both seeds full vertex >= %.6f or full "
                                 "token R2 < %.1f" % (full_floor, charge_min, half_floor,
                                                      charge_min, full_floor, charge_min)},
            "verdict": verdict, "direction": direction, "per_seed": rows}
