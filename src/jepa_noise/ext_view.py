# SPDX-License-Identifier: MIT
"""View-design candidates extension (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md`` (2 Nov 2026, reviewed,
authorised).  Extension name: ``view_design_2026-11-03``.

Eight ``sigreg_jepa`` runs (10 000 steps, seeds 9000/9001) on the committed 100k
development cohort, one per candidate split: the registered ``within`` and
``across`` splits plus two new runtime-only splits registered by a process-level
``ViewMasker`` patch (never edited in place):

* ``module_parity``: whole modules by the **dense** ``geometry.module_index``
  parity (even -> context, odd -> target);
* ``random_half``: a seeded permutation of the 1 995 PMTs (salt 20260101), first
  997 -> context, remaining 998 -> target, fixed across events.

The patch replaces both ``lucid_run.ViewMasker`` and ``point_encoder.ViewMasker``
(the two import-time bindings), is installed around training/smoke only, and is
restored in a ``finally``.  Evaluation uses the registered ``within`` probe split
with the item split pinned explicitly.

This module holds the gate, ``SPLIT_NAMES``/``SplitMasker``/``masker_patch``, the
entries preflight, thin wrappers, sidecars and the per-split ``view_r`` reading.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np

from . import ext_h12 as eh
from . import lucid_bombs as lb
from . import lucid_run as lr
from . import lucid_views as lv
from . import point_encoder as pe

EXTENSION = "view_design_2026-11-03"
ARM_NAME = "sigreg_jepa"
CELL_ID = "dev_s005_within"
EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_view_20261103")
READOUT_EXT = Path("/ceph/dwong/jepa/returns/ext_readout_20261101")
RESULTS_DIR = "results/jepa/20261103_view"
RUNBOOK_REL = "docs/jepa/RUNBOOK_2026-11-03_VIEW_DESIGN.md"
DECISIONS_REL = "docs/jepa/DECISIONS.md"
AUTHOR_LINE = "B 的下一批训练我全部授权"
C_ROW = ("| C | view design：4 个候选 split × 2 seeds × 10k steps | 8 | 0.75/run | "
         "**≈6.0** | ~0（复用 dev 数据，split 是 runtime mask） | 低风险 |")
SPLIT_NAMES = ("within", "across", "module_parity", "random_half")
NEW_SPLITS = ("module_parity", "random_half")
RANDOM_HALF_SALT = 20260101
STEPS = 10000
INIT_FULL_VERTEX_M = float(eh.INIT_FULL_VERTEX_M)      # 0.160072, measured rebuild
INIT_HALF_VERTEX_M = float(eh.INIT_HALF_VERTEX_M)      # 0.174821
CHARGE_R2_MIN = float(eh.CHARGE_R2_MIN)
ENTRIES_CLONE = 3059
ENTRIES_SUITE = 1825
ENTRIES_EXT = 1000
ENTRIES_FLOOR = 30000
DATA_ROOT_DEFAULT = eh.DATA_ROOT_DEFAULT
JOBS_DIR = Path("/ceph/dwong/jepa/jobs")
GUARD_JSON = JOBS_DIR / "guard_deepthought2.json"
#: Read-only probe-set manifest hashes (committed; never regenerated here).
PROBE_MANIFESTS = {"dev_probe_train__F1__0.05__s20261017":
                   "093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3",
                   "dev_probe_test__F1__0.05__s20261017":
                   "c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a"}


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
    """Hard gate (§0): full author line + C row in runbook §0, line in DECISIONS."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    runbook, decisions = root / RUNBOOK_REL, root / DECISIONS_REL
    problems = []
    if not runbook.is_file() or not decisions.is_file():
        problems.append(f"missing {runbook} or {decisions}")
    else:
        section = _section_zero(runbook.read_text())
        if AUTHOR_LINE not in section:
            problems.append(f"{RUNBOOK_REL} §0 does not contain {AUTHOR_LINE!r}")
        if C_ROW not in section:
            problems.append(f"{RUNBOOK_REL} §0 does not contain the C row")
        if "NOT AUTHORISED" in section:
            problems.append(f"{RUNBOOK_REL} §0 still carries a NOT AUTHORISED banner")
        if AUTHOR_LINE not in decisions.read_text():
            problems.append(f"{DECISIONS_REL} does not contain {AUTHOR_LINE!r}")
    if problems:
        for problem in problems:
            print(f"[view] authorisation gate: {problem}", file=sys.stderr)
        raise SystemExit(2)
    return {"authorised": True, "runbook": str(runbook),
            "runbook_sha256": lr.sha256_file(runbook), "decisions": str(decisions),
            "author_line": AUTHOR_LINE, "c_row": C_ROW}


# ------------------------------------------------------- split / masker patch
class SplitMasker(lv.ViewMasker):
    """``ViewMasker`` accepting the two new runtime splits (§3.1 exact behaviour)."""

    def __init__(self, geometry, split, *, seed=0, replicate=0, cell="", n_min=lv.N_MIN,
                 max_redraws=lv.MAX_REDRAWS, occupancy="natural",
                 n_target_modules=None, module_bits=lv.MODULE_HILBERT_BITS):
        if split not in SPLIT_NAMES:
            raise ValueError(f"unregistered split {split!r}")
        # bypass ViewMasker.__init__'s split-in-SPLITS check; copy its field setup
        self.geometry, self.split, self.seed = geometry, split, int(seed)
        self.replicate, self.cell = int(replicate), cell
        self.n_min, self.max_redraws = int(n_min), int(max_redraws)
        self.occupancy = occupancy
        self.n_target_modules = n_target_modules
        self.module_bits = int(module_bits)
        self.counter = lv.ViewExclusionCounter(n_min=self.n_min,
                                               max_redraws=self.max_redraws)
        self.fraction_context = self.fraction_target = 0.0
        self.n_seen = self.n_balanced = 0

    def mask(self, event_id, hit_pmt):
        if self.split in ("within", "across"):
            return super().mask(event_id, hit_pmt)
        hit = lv._hit_array(self.geometry, hit_pmt)
        if self.split == "module_parity":
            in_context = (self.geometry.module_index % 2) == 0
            salt = None
        else:  # random_half
            rng = lb.keyed_rng(int(self.seed), "lucid/views", "random_half",
                               RANDOM_HALF_SALT)
            order = rng.permutation(self.geometry.n_pmt)
            in_context = np.zeros(self.geometry.n_pmt, dtype=bool)
            in_context[order[: self.geometry.n_pmt // 2]] = True
            salt = RANDOM_HALF_SALT
        context = np.flatnonzero(in_context).astype(np.int64)
        target = np.flatnonzero(~in_context).astype(np.int64)
        context_hits, target_hits = int(hit[context].sum()), int(hit[target].sum())
        accepted = target_hits >= self.n_min
        key = {"split": self.split, "geometry_sha256": self.geometry.digest,
               "event_id": int(event_id), "seed": int(self.seed), "salt": salt,
               "occupancy": "natural", "n_min": int(self.n_min),
               "context_hits": context_hits, "target_hits": target_hits,
               "module_index_parity": ("even=context,odd=target"
                                       if self.split == "module_parity" else None)}
        mask = lv.ViewMask(self.split, int(event_id), context, target, accepted, 0,
                           context_hits, target_hits, None, (), np.empty(0, np.int64), key)
        self.counter.record_within(target_hits)
        self.n_seen += 1
        if accepted:
            fr = mask.fractions(hit)
            self.fraction_context += fr["context"]
            self.fraction_target += fr["target"]
        return mask


@contextlib.contextmanager
def masker_patch(log=print):
    """Patch both import-time ``ViewMasker`` bindings; restored in a ``finally``."""
    old_lr, old_pe = lr.ViewMasker, pe.ViewMasker
    lr.ViewMasker = pe.ViewMasker = SplitMasker
    try:
        yield
    finally:
        lr.ViewMasker, pe.ViewMasker = old_lr, old_pe


def masker_patch_active() -> bool:
    return lr.ViewMasker is SplitMasker and pe.ViewMasker is SplitMasker


def cell_for(config, split: str):
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    return dataclasses.replace(lr.dev_cell(config), split=split)


# ---------------------------------------------------------------- preflight
def live_free_entries(guard_json=GUARD_JSON) -> dict:
    guard_json = Path(guard_json)
    if not guard_json.is_file():
        raise RuntimeError(f"guard status {guard_json} is missing; cannot check entries")
    status = json.loads(guard_json.read_text())
    free = (status.get("caps") or {}).get("free_entries")
    if free is None:
        raise RuntimeError(f"{guard_json}: guard status has no caps.free_entries")
    return {"free_entries": int(free), "guard_level": status.get("level"),
            "guard_reasons": status.get("reasons"),
            "guard_timestamp": status.get("timestamp"), "guard_json": str(guard_json)}


def entries_projection(free_before: int, *, clone: int = ENTRIES_CLONE,
                       suite: int = ENTRIES_SUITE, extension: int = ENTRIES_EXT,
                       floor: int = ENTRIES_FLOOR) -> dict:
    projected = int(free_before) - (int(clone) + int(suite) + int(extension))
    return {"free_before": int(free_before), "clone": int(clone), "suite": int(suite),
            "extension": int(extension), "projected_after": int(projected),
            "floor": int(floor),
            "pass": bool(projected >= int(floor) and int(free_before) >= int(floor))}


def probe_status(data_root) -> dict:
    root = Path(data_root) / "lucid"
    out = {"data_root": str(data_root), "sets": {}}
    for key, expected in PROBE_MANIFESTS.items():
        manifest = root / key / "manifest.json"
        row = {"manifest": str(manifest), "exists": manifest.is_file(),
               "expected_sha256": expected}
        if manifest.is_file():
            row["manifest_sha256"] = lr.sha256_file(manifest)
            row["pass"] = row["manifest_sha256"] == expected
        else:
            row["manifest_sha256"] = None
            row["pass"] = False
        out["sets"][key] = row
    out["pass"] = all(row["pass"] for row in out["sets"].values())
    return out


def require_data(config, gen, geometry, data_root, *, code_sha=None, log=print):
    """`eh.preflight_data` **before** `eh.datasets_for` (§3.5: never regenerate)."""
    status = probe_status(data_root)
    if not status["pass"]:
        raise SystemExit(f"[view] probe-set preflight failed (never regenerate): {status}")
    preflight = eh.preflight_data(Path(data_root))
    if not preflight.get("pass"):
        raise SystemExit(f"[view] dev data preflight failed: {preflight}")
    cell = lr.dev_cell(config)
    train_ds, probes = eh.datasets_for(cell, config, gen, geometry, Path(data_root),
                                       code_sha=code_sha)
    return train_ds, probes, {"probes": status, "preflight": preflight}


# ----------------------------------------------------------------- run dirs
def run_id_for(seed) -> str:
    return lr.run_id_for(CELL_ID, ARM_NAME, int(seed))


def out_root_for(ext_root, split) -> Path:
    return Path(ext_root) / "train" / "views" / split


def run_dir_for(out_root, seed) -> Path:
    return Path(out_root) / "runs" / run_id_for(seed)


def sidecar_payload(split, seed, out_root, steps, device) -> dict:
    return {"registered": False, "extension": EXTENSION, "split": split,
            "seed": int(seed), "arm": ARM_NAME, "cell_id": CELL_ID,
            "out_root": str(out_root), "run_id": run_id_for(seed),
            "steps": int(steps), "device": str(device)}


def prepare_run_dir(out_root, split, seed, *, steps=STEPS, device=None) -> tuple:
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split {split!r}")
    run_id, run_dir = run_id_for(seed), run_dir_for(out_root, seed)
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} already exists; refusing to launch (run_one would "
                           f"auto-resume it). Archive or remove it first.")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "variant.json").write_text(json.dumps(
        sidecar_payload(split, seed, Path(out_root), steps, device), indent=1) + "\n")
    return run_id, run_dir


def summarize_run(run_dir, *, split, seed, wall_seconds=None, steps=None,
                  resumed_from_step=0) -> dict:
    run_dir = Path(run_dir)
    check = eh.train_log_check(run_dir)
    checkpoint = run_dir / "final.pt"
    status = json.loads((run_dir / "run_status.json").read_text()) \
        if (run_dir / "run_status.json").is_file() else {}
    return {"registered": False, "extension": EXTENSION, "split": split,
            "seed": int(seed), "run_id": run_dir.name, "arm": ARM_NAME,
            "cell_id": CELL_ID, "steps_scheduled": steps,
            "steps_completed": status.get("steps_completed"),
            "resumed_from_step": int(resumed_from_step),
            "status": status.get("outcome"),
            "final_train_metrics": check.get("final_metrics"),
            "non_finite_keys": check.get("non_finite_keys"),
            "metrics_finite": check.get("finite"),
            "final_checkpoint_sha256": (lr.sha256_file(checkpoint)
                                        if checkpoint.is_file() else None),
            "wall_seconds": wall_seconds}


# ----------------------------------------------------------------- training
def run_one_training(*, split, seed, out_root, config, gen, arms_config, geometry,
                     device, data_root, config_path, code_sha, status_text,
                     steps=None, probe_max_events=None, log=print) -> dict:
    """The exact ``run_one`` call of §3.2 with the masker patch installed."""
    gate = assert_authorised()
    if split not in SPLIT_NAMES:
        raise SystemExit(f"[view] unknown split {split!r}")
    train_ds, probes, data_info = require_data(config, gen, geometry, data_root,
                                               code_sha=code_sha, log=log)
    cell = cell_for(config, split)
    scheduled = int(STEPS if steps is None else steps)
    run_id, run_dir = prepare_run_dir(Path(out_root), split, seed, steps=scheduled,
                                      device=device)
    started = __import__("time").time()
    with masker_patch(log=log):
        result = lr.run_one(
            arm_name=ARM_NAME, cell=cell, seed=int(seed), beta=None, config=config,
            arms_config=arms_config, geometry=geometry, loader_dataset=train_ds,
            probe_datasets=probes, dark_bank=None, out_root=Path(out_root),
            data_root=Path(data_root), config_sha=lr.sha256_file(config_path),
            code_sha=code_sha, status_text=status_text, device=device,
            steps=None if steps is None else steps, micro_batch=None, resume=None,
            stop_after=None, checkpoint_minutes=None, evaluate=False,
            allow_pending=False, step_callback=None, probe_max_events=probe_max_events,
            save_embeddings=False, physics_method="ridge", tf32=False, log=log)
    from . import ext_h1clean as ec
    resumed = ec.assert_fresh_resume(result["training"], run_id)
    summary = summarize_run(run_dir, split=split, seed=seed,
                            wall_seconds=round(__import__("time").time() - started, 3),
                            steps=scheduled, resumed_from_step=resumed)
    summary["smoke"] = scheduled == 20
    summary["masker_patch_active"] = False  # restored after training
    summary["probes_preflight"] = {"pass": data_info["preflight"].get("pass")}
    (run_dir / "ext_summary.json").write_text(json.dumps(summary, indent=1,
                                                         default=float) + "\n")
    return {"result": result, "run_dir": run_dir, "summary": summary,
            "run_id": run_id, "seed": int(seed), "split": split, "gate": gate,
            "cell_split": cell.split}


# --------------------------------------------------------------- evaluation
def evaluate_run_dirs(split, run_dirs, probes, geometry, config, arms_config, *,
                      device, max_events=None) -> dict:
    """Evaluate with the **registered within probe**; no masker patch installed."""
    runs = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        candidates = sorted((run_dir / "runs").glob("*")) if (run_dir / "runs").is_dir() \
            else [run_dir]
        for candidate in candidates:
            sidecar_path = candidate / "variant.json"
            if not sidecar_path.is_file():
                raise SystemExit(f"[view] {sidecar_path} missing")
            sidecar = json.loads(sidecar_path.read_text())
            if sidecar.get("split") != split:
                raise SystemExit(f"[view] {sidecar_path}: split "
                                 f"{sidecar.get('split')!r} != {split!r}")
            seed = int(sidecar["seed"])
            source = candidate / "final.pt"
            if not source.is_file():
                raise SystemExit(f"[view] {candidate}: final.pt is missing")
            item = {"label": candidate.name, "arm": ARM_NAME, "split": "within",
                    "seed": seed, "source": str(source)}
            metrics = eh.evaluate_run(item, probes, geometry, config, arms_config,
                                      device=device, max_events=max_events)
            runs[str(seed)] = {
                "split": split, "seed": seed, "run_id": candidate.name,
                "run_dir": str(candidate), "probe_split": "within",
                "checkpoint_sha256": lr.sha256_file(source),
                "full_input": metrics["forms"]["full_input"],
                "half_pass": metrics["forms"]["half_pass"],
                "ext_summary": (json.loads((candidate / "ext_summary.json").read_text())
                                if (candidate / "ext_summary.json").is_file() else None)}
    return runs


def write_variants_json(ext_root, run_dirs) -> dict:
    variants = {"registered": False, "extension": EXTENSION, "runs": {}}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        candidates = sorted((run_dir / "runs").glob("*")) if (run_dir / "runs").is_dir() \
            else [run_dir]
        for candidate in candidates:
            sidecar_path = candidate / "variant.json"
            if sidecar_path.is_file():
                variants["runs"][str(candidate)] = json.loads(sidecar_path.read_text())
    path = Path(ext_root) / "variants.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.is_file():
        existing = json.loads(path.read_text())
        variants["runs"] = dict(existing.get("runs") or {}, **variants["runs"])
    tmp.write_text(json.dumps(variants, indent=1, default=float) + "\n")
    os.replace(tmp, path)
    return variants


# -------------------------------------------------------- reading view_r §2
def view_r(split_runs: dict, *, full_floor=INIT_FULL_VERTEX_M,
           half_floor=INIT_HALF_VERTEX_M, charge_min=CHARGE_R2_MIN,
           init_full=None, init_half=None) -> dict:
    """Per-split two-sided ``view_r`` over the same rows as ``eh.recover`` (§2)."""
    full_rows = {seed: run["full_input"] for seed, run in split_runs.items()}
    half_rows = {seed: run["half_pass"] for seed, run in split_runs.items()}
    recovered = eh.recover(full_rows, half_rows, init_full or {}, init_half or {})
    rows = {}
    for seed, row in recovered["seeds"].items():
        full_v, full_r2 = row["full_input_vertex_m"], row["full_input_token_r2"]
        half_v, half_r2 = row["half_pass_vertex_m"], row["half_pass_token_r2"]
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
        verdict, direction = "recovered", "recovered: the collapse does not persist for this view split"
    elif values and all(row["fails"] for row in values):
        verdict, direction = "not recovered", "not recovered: the collapse persists for this view split"
    else:
        verdict, direction = "unclear", "unclear (partial recovery, seed disagreement or half-pass contradiction)"
    return {"registered": False, "extension": EXTENSION, "split": None,
            "criteria": {"floors": {"full": float(full_floor),
                                    "half": float(half_floor),
                                    "charge_r2_min": float(charge_min)},
                         "rule": "recovered: both seeds full vertex strictly < %.6f m, full "
                                 "token ln q R2 >= %.1f, half vertex < %.6f m, half token "
                                 "R2 >= %.1f; not recovered: for each seed full vertex >= "
                                 "%.6f m or full token R2 < %.1f (both seeds); else unclear"
                                 % (full_floor, charge_min, half_floor, charge_min,
                                    full_floor, charge_min)},
            "verdict": verdict, "direction": direction, "per_seed": rows}


def rank_splits(splits: dict) -> tuple:
    """Rank by mean full-input vertex (ascending); recommends the best split."""
    ranked = []
    for split, reading in splits.items():
        values = [row["full_input_vertex_m"] for row in reading["per_seed"].values()
                  if row["full_input_vertex_m"] is not None]
        if values:
            ranked.append({"split": split, "verdict": reading["verdict"],
                           "mean_full_vertex_m": float(np.mean(values)),
                           "max_full_vertex_m": float(np.max(values)),
                           "min_full_vertex_m": float(np.min(values))})
    ranked.sort(key=lambda row: row["mean_full_vertex_m"])
    if not ranked:
        recommendation = "none (no evaluated split has a full-input vertex)"
    else:
        best = next((row for row in ranked if row["verdict"] == "recovered"), None)
        if best is not None:
            recommendation = (f"use the {best['split']} split as the designed view "
                              f"(both seeds recovered, mean full vertex "
                              f"{best['mean_full_vertex_m']:.6f} m)")
        else:
            recommendation = ("no split in this set makes the physics learnable at the "
                             "development budget; the view is not the lever (the "
                             "target-content mechanism is) — best-ranked is "
                             f"{ranked[0]['split']} (mean full vertex "
                             f"{ranked[0]['mean_full_vertex_m']:.6f} m), still above the floor")
    return ranked, recommendation
