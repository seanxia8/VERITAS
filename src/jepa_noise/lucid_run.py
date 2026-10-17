"""WP5 LUCiD runner (§3.5.6): cells, cached generation, training, probes, verdicts.
Modes: ``grid`` (per-cell independent training; refuses while the optimizer is
``pending_review``), ``dev`` (development-cohort steps 1-3; step 4 is refused —
it does not exist in §3.5.6), ``smoke`` (tiny, smoke optimizer), ``scaling``
(§3.5.5, no training; records ``does_not_fit``), ``aggregate`` (read-only; never
edits ``PREDICTION_STATUS.json``).  ``grid --dry-run`` validates the plan without
training or writing.  One process = one GPU job.  Binding: runbook v4 §3.5.3-
§3.5.9 and §6.1-§6.2; every parameter the runbook does not fix is listed in
``unresolved_questions`` and in ``docs/jepa/WP5_RUNNER.md``, never decided here.
"""
from __future__ import annotations
import argparse, json, math, os, signal, sys, time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Sequence
import numpy as np
import torch
from . import lucid_arms as la
from . import lucid_bombs as lb
from . import point_encoder as pe
from .bridge_run import (atomic_write_json, atomic_write_text, git_head, git_status,
                         host_name, sha256_file, utcnow, write_sha256sums)
from .lucid_views import ViewMask, ViewMasker
SCHEMA_RUN, SCHEMA_CHECKPOINT = "jepa-lucid-run-v1", "jepa-lucid-checkpoint-v1"
SCHEMA_DATASET, SCHEMA_AGGREGATE = "jepa-lucid-dataset-v1", "jepa-lucid-aggregate-v1"
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "lucid_run_v0.json"
ROOT = Path(__file__).resolve().parents[2]
READOUT_LABEL = lb.READOUT_LABEL
DEV2 = ("sigreg_jepa", "mae_mse")
DEV3 = ("sigreg_jepa_anchored", "anchor_only", "sigreg_jepa_anchored_dark")
# --- config -----------------------------------------------------------------
def load_run_config(path=None) -> dict:
    """Load and validate the registered plan against §3.5.6's fixed numbers."""
    target = Path(CONFIG_PATH if path is None else path)
    c = json.loads(target.read_text())
    for block in ("training", "cells", "probe_sets", "data", "views", "evaluation",
                  "scaling", "hosts", "unresolved_questions"):
        if block not in c:
            raise ValueError(f"{target}: missing {block!r}")
    t, m, d = c["training"], c["cells"]["main"], c["cells"]["development"]
    if int(t["steps"]) != 10000 or int(t["batch"]) != 32:
        raise ValueError(f"{target}: training must stay at 10 000 steps / batch 32")
    if not t.get("fp32") or t.get("tf32") or not t.get("final_step_freeze") or t.get("best_checkpoint"):
        raise ValueError(f"{target}: fp32, TF32 off, final-step freeze, no best checkpoint")
    if t["optimizer"] not in ("pending_review", "adamw", "muon_adamw"):
        raise ValueError(f"{target}: unknown optimizer {t['optimizer']!r}")
    if t["optimizer"] == "pending_review" and any(v != "pending_review" for v in t["optimizer_detail"].values()):
        raise ValueError(f"{target}: pending_review optimizer carries numerics")
    if [float(x) for x in m["spreads"]] != [0.02, 0.05, 0.10] or list(m["splits"]) != ["across", "within"]:
        raise ValueError(f"{target}: main cells must be the registered spreads x splits")
    if float(m["s_p"]) != lb.F3_SPREAD:
        raise ValueError(f"{target}: s_p is frozen at {lb.F3_SPREAD}")
    if set(m["arms_seeds"]) != set(la.ARMS):
        raise ValueError(f"{target}: arms_seeds must name the seven arms")
    for arm, n in m["arms_seeds"].items():
        if int(n) != (5 if arm in ("sigreg_jepa", "sigreg_jepa_anchored", "mae_mse") else 3):
            raise ValueError(f"{target}: {arm} seed count differs from §3.5.6")
    if list(c["cells"]["secondary_order"]) != ["r11", "r11b", "dark_anchor", "law_t", "occupancy"]:
        raise ValueError(f"{target}: secondary order differs from §3.5.6")
    if tuple(d["step2_arms"]) != DEV2 or tuple(d["step3_arms"]) != DEV3 \
            or [float(x) for x in d["beta_grid"]] != [0.1, 1, 10, 100, 1000]:
        raise ValueError(f"{target}: development cohort differs from §3.5.6")
    forms = c["evaluation"]["form_kinds"]
    if set(forms) != set(la.PROBE_FORMS) or c["evaluation"]["primary_form"] != "run_mean" \
            or any(set(k) - set(la.PROBE_KINDS) for k in forms.values()):
        raise ValueError(f"{target}: probe forms/kinds differ from §3.5.6")
    return c
# --- cells ------------------------------------------------------------------
@dataclass(frozen=True)
class Cell:
    cell_id: str; role: str; law: str; setting: float | None; s_arm: float; split: str
    s_p: float; arms: tuple; seeds: dict; train_cohort: str; probe_set: str
    occupancy: str = "natural"; robustness_set: str | None = None
    baseline_set: str | None = None; extrapolation_set: str | None = None
    r6_r7_eligible: bool = False; criterion: str | None = None; settings_source: str | None = None
    @property
    def n_runs(self) -> int:
        return int(sum(int(self.seeds[a]) for a in self.arms))
    def runs(self):
        for arm in self.arms:
            for seed in range(int(self.seeds[arm])):
                yield {"cell_id": self.cell_id, "arm": arm, "seed": seed, "beta": None}
@dataclass(frozen=True)
class ProbeSet:
    key: str; law: str; setting: float | None; setting_kind: str | None
    cohort: str | None; dev: bool; test_only: bool
    def cohorts(self) -> dict:
        p = "dev_" if self.dev else ""
        return ({"train": None, "test": p + "robustness"} if self.cohort == "robustness"
                else {"train": p + "probe_train", "test": p + "probe_test"})
def _probe_sets(c: dict) -> dict:
    return {k: ProbeSet(k, s["law"], None if "setting" not in s else float(s["setting"]),
                        s.get("setting_kind"), s.get("cohort"), bool(s.get("dev", False)),
                        bool(s.get("test_only", False))) for k, s in c["probe_sets"].items()}
def enumerate_cells(c: dict) -> list:
    """The six main cells then the five secondary cells of §3.5.6."""
    m = c["cells"]["main"]
    seeds = {a: int(n) for a, n in m["arms_seeds"].items()}
    probes = {float(k): v for k, v in m["probe_set_by_spread"].items()}
    eligible = {float(x) for x in m["r6_r7_eligible_spreads"]}
    out = [Cell(f"main_s{int(round(s * 100)):03d}_{split}", "main", "F1", s, s, split,
                float(m["s_p"]), tuple(la.ARMS), dict(seeds), m["train_cohort"], probes[s],
                robustness_set=m["robustness_set"], baseline_set=m["baseline_set"],
                r6_r7_eligible=s in eligible)
           for s in [float(x) for x in m["spreads"]] for split in m["splits"]]
    ids = {"r11": "r11_f1f2f3_within", "r11b": "r11b_law_within",
           "dark_anchor": "dark_anchor_s010_within", "law_t": "law_t_aging_within",
           "occupancy": "occupancy_s005_within_balanced"}
    for key in c["cells"]["secondary_order"]:
        sp = c["cells"]["secondary"][key]; arms = tuple(sp["arms"])
        setting = None if key == "law_t" else float(sp["setting"])
        s_arm = {"law_t": float(lb.T_SPREAD), "r11b": float(lb.R11B_SPREAD),
                 "r11": 0.05}.get(key, setting or 0.0)
        out.append(Cell(ids[key], key, sp["law"], setting, s_arm, sp["split"],
                        float(sp["s_p"]), arms, {a: int(sp["seeds"]) for a in arms},
                        sp["train_cohort"], sp["probe_set"], sp.get("occupancy", "natural"),
                        extrapolation_set=sp.get("extrapolation_set"), criterion=sp.get("criterion"),
                        settings_source=sp.get("settings_source")))
    return out
def development_runs(c: dict) -> list:
    """The 17 development trainings: 2 probe-decision + 15 beta."""
    d = c["cells"]["development"]
    runs = [{"cell_id": "dev_s005_within", "step": 2, "arm": a,
             "seed": int(d["step2_seed"]), "beta": None} for a in d["step2_arms"]]
    for a in d["step3_arms"]:
        for i, b in enumerate(d["beta_grid"]):
            runs.append({"cell_id": "dev_s005_within", "step": 3, "arm": a,
                         "seed": int(d["step3_seed_base"]) + i, "beta": float(b)})
    return runs
def dev_cell(c: dict) -> Cell:
    d = c["cells"]["development"]
    arms = tuple(d["step2_arms"]) + tuple(d["step3_arms"])
    return Cell("dev_s005_within", "development", "F1", float(d["spread"]), float(d["spread"]),
                d["split"], float(d["s_p"]), arms, {a: 1 for a in arms},
                d["train_cohort"], d["probe_set"], criterion="development")
def run_id_for(cell_id: str, arm: str, seed: int, beta=None) -> str:
    rid = f"{cell_id}__{arm}__s{int(seed):04d}"
    return rid if beta is None else f"{rid}__b{beta:g}"
def _settings_for(law: str, kind: str | None, n_runs: int, seed: int):
    if law != "T":
        return None
    if kind == "in_range":
        return tuple(lb.law_T_ratios(n_runs, kind="in_range", seed=seed))
    if kind == "extrapolation":
        return tuple(lb.law_T_ratios(n_runs, kind="extrapolation", seed=seed))
    return tuple(lb.law_T_training_ratios(n_runs))
def dataset_key(cohort, law, setting, seed, kind=None, n_events=None, run_length=None, registered=True):
    tag = "training" if law == "T" and kind is None else (
        kind if law == "T" else ("none" if setting is None else f"{setting:g}"))
    key = f"{cohort}__{law}__{tag}__s{int(seed)}"
    return key if registered else f"{key}__n{n_events}x{run_length}"
# --- runtime ----------------------------------------------------------------
def configure_runtime(device="auto") -> torch.device:
    """fp32, TF32 off, deterministic, single BLAS thread (§6.1)."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1, user_api="blas")
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    torch.set_num_threads(1)
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"device {device!r} requested but CUDA is unavailable")
    return resolved
def _safe(value):
    """Recursively replace non-finite floats with None (the result writer rejects nan)."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.floating):
        return None if not math.isfinite(float(value)) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value
def _inside(path, root) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents
def _guard(path: Path, *roots) -> Path:
    if not any(_inside(path, r) for r in roots):
        raise RuntimeError(f"refusing to write outside the out/data roots: {path}")
    return Path(path)
# --- datasets ---------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    key: str; directory: Path; cohort: str; law: str; setting: float | None
    setting_kind: str | None; seed: int; n_events: int; run_length: int
    runs: tuple; manifest: dict
    @property
    def n_runs(self) -> int:
        return len(self.runs)
    def manifest_sha256(self) -> str:
        return sha256_file(self.directory / "manifest.json")
def _cohort(spec: dict) -> lb.CohortSpec:
    return lb.CohortSpec(str(spec["namespace"]), int(spec["event_start"]),
                         int(spec["n_events"]), int(spec["run_length"]))
def ensure_dataset(*, data_root, geometry, generator_config, cohort_spec, cohort_name, law,
                   setting=None, seed, settings=None, setting_kind=None, registered=True,
                   code_sha=None, verify_hashes=False, log=print) -> Dataset:
    """Generate or reuse one dataset; existing runs are verified, never overwritten."""
    data_root = Path(data_root).resolve(); data_root.mkdir(parents=True, exist_ok=True)
    spec = _cohort(cohort_spec)
    runs = (lb.make_runs(spec, law, settings=tuple(float(x) for x in settings), seed=int(seed))
            if settings is not None else lb.make_runs(spec, law, setting=setting, seed=int(seed)))
    key = dataset_key(cohort_name, law, setting, seed, setting_kind, spec.n_events,
                      spec.run_length, registered)
    directory = data_root / "lucid" / key; manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        _verify_dataset(manifest, directory, geometry, generator_config, cohort_name, law,
                        setting, seed, setting_kind, len(runs), runs, verify_hashes)
        return Dataset(key, directory, cohort_name, law, setting, setting_kind, int(seed),
                       spec.n_events, spec.run_length, tuple(manifest["runs"]), manifest)
    directory.mkdir(parents=True, exist_ok=True); rows = []
    for i, run in enumerate(runs):
        run_dir = directory / "runs" / run.run_id.replace("/", "__"); path = run_dir / "manifest.json"
        if path.exists():
            payload = json.loads(path.read_text()); lb.verify_manifest(payload, [run], geometry, seed=int(seed))
            if verify_hashes:
                lb.verify_sha256sums(run_dir)
        else:
            generated = lb.generate_run(geometry, generator_config, run, with_twin=True)
            payload = lb.write_generated_run(run_dir, generated, geometry, generator_config,
                                             seed=int(seed), code_sha=code_sha)
            log(f"[data] {key} run {i + 1}/{len(runs)}: {generated.n_events} events, {generated.n_hits} hits")
        rows.append({"run_id": run.run_id, "run_dir": run_dir.name,
                     "event_id_start": run.event_start, "event_id_stop": run.event_stop,
                     "n_events": run.n_events, "setting": run.setting,
                     "state_sha256": payload["runs"][0]["state_sha256"],
                     "event_ids_sha256": payload["runs"][0]["event_ids_sha256"],
                     "manifest_sha256": sha256_file(path)})
    manifest = {"schema": SCHEMA_DATASET, "key": key, "cohort": cohort_name, "law": law,
                "setting": setting, "setting_kind": setting_kind, "dataset_seed": int(seed),
                "n_events": spec.n_events, "n_runs": len(runs), "run_length": spec.run_length,
                "n_pmt": int(geometry.n_pmt), "registered": bool(registered),
                "readout_label": READOUT_LABEL, "reported_results": False,
                "geometry_sha256": geometry.digest, "geometry_summary": geometry.summary(),
                "generator_config_sha256": generator_config.digest,
                "yield_range_photons": [float(generator_config.y_lo), float(generator_config.y_hi)],
                "yield_range_status": lb.YIELD_RANGE_STATUS_IN_FORCE, "code_sha": code_sha,
                "created_utc": utcnow(), "runs": rows}
    atomic_write_json(manifest_path, manifest)
    atomic_write_text(directory / "SHA256SUMS", f"{sha256_file(manifest_path)}  manifest.json\n")
    return Dataset(key, directory, cohort_name, law, setting, setting_kind, int(seed),
                   spec.n_events, spec.run_length, tuple(rows), manifest)
def _verify_dataset(m, directory, geometry, generator_config, cohort, law, setting, seed,
                    kind, n_runs, runs, verify_hashes) -> None:
    expected = {"schema": SCHEMA_DATASET, "cohort": cohort, "law": law, "setting": setting,
                "setting_kind": kind, "dataset_seed": int(seed), "n_runs": int(n_runs),
                "n_pmt": int(geometry.n_pmt), "readout_label": READOUT_LABEL,
                "geometry_sha256": geometry.digest, "generator_config_sha256": generator_config.digest}
    for name, value in expected.items():
        if m.get(name) != value:
            raise RuntimeError(f"dataset {directory.name}: {name} {m.get(name)!r} != cached {value!r}; "
                               "refusing to regenerate silently")
    for row, run in zip(m["runs"], runs):
        if row["run_id"] != run.run_id:
            raise RuntimeError(f"dataset {directory.name}: run id mismatch")
        run_dir = directory / "runs" / row["run_dir"]
        lb.verify_manifest(json.loads((run_dir / "manifest.json").read_text()), [run], geometry, seed=int(seed))
        if verify_hashes:
            lb.verify_sha256sums(run_dir)
def _run_truth(row: dict, dataset: Dataset, geometry) -> dict:
    directory = dataset.directory / "runs" / row["run_dir"]
    with np.load(directory / "events.npz", allow_pickle=False) as data:
        event_ids, vertex = np.asarray(data["event_ids"], np.int64), np.asarray(data["vertex_m"], float)
        yields = np.asarray(data["yield_photons"], float)
    spec = lb.RunSpec(row["run_id"], dataset.manifest["cohort"], dataset.law, float(row["setting"]),
                      int(row["event_id_start"]), int(row["n_events"]), int(dataset.seed))
    truth = la.truth_from_state(geometry, spec.state(geometry), vertex_m=vertex, yield_photons=yields)
    truth["event_ids"] = event_ids
    return truth
# --- loader -----------------------------------------------------------------
def _hit_pattern(store, index, n_pmt):
    span = slice(int(store.offsets[index]), int(store.offsets[index + 1]))
    hit = np.zeros(n_pmt, bool); hit[np.asarray(store.pmt_id[span], np.int64)] = True
    return hit
def _masks_for(run_dir, geometry, *, split, seed, occupancy, n_min, event_ids):
    store = lb.read_sparse_hits(run_dir / "hits.npz")
    wanted = {int(x) for x in np.asarray(event_ids).ravel()}
    masker = ViewMasker(geometry, split, seed=seed, occupancy=occupancy, n_min=n_min)
    out = {}
    for i, event_id in enumerate(store.event_ids):
        if int(event_id) in wanted:
            mask = masker.mask(int(event_id), _hit_pattern(store, i, geometry.n_pmt))
            if mask.accepted:
                out[int(event_id)] = mask
    return out
def _tokenise_run(run_dir, geometry, *, split, seed, occupancy, n_min, scaling, limit=None, masker=None):
    masker = masker or ViewMasker(geometry, split, seed=seed, occupancy=occupancy, n_min=n_min)
    return pe.tokenise(run_dir, geometry, split=split, seed=seed, occupancy=occupancy,
                       scaling=scaling, masker=masker, limit=limit)
def _stack(batches, rows):
    sel = [(b, np.asarray(r, np.int64)) for b, r in zip(batches, rows) if np.asarray(r).size]
    if not sel:
        raise RuntimeError("empty batch: every event was excluded by the N_min rule")
    first = sel[0][0]
    return pe.LucidBatch(
        features=torch.cat([b.features[r] for b, r in sel]), target=torch.cat([b.target[r] for b, r in sel]),
        hidden=torch.cat([b.hidden[r] for b, r in sel]),
        context_index=torch.cat([b.context_index[r] for b, r in sel]),
        target_index=torch.cat([b.target_index[r] for b, r in sel]),
        hit_flags=np.concatenate([b.hit_flags[r] for b, r in sel]),
        log_charge=np.concatenate([b.log_charge[r] for b, r in sel]),
        event_ids=np.concatenate([b.event_ids[r] for b, r in sel]),
        keys=tuple(k for b, r in sel for k in (b.keys[int(i)] for i in r)),
        report=dict(first.report), scaling=first.scaling, split=first.split, occupancy=first.occupancy)
def _slice(batch, start, stop):
    s = slice(int(start), int(stop))
    return pe.LucidBatch(features=batch.features[s], target=batch.target[s], hidden=batch.hidden[s],
                         context_index=batch.context_index[s], target_index=batch.target_index[s],
                         hit_flags=batch.hit_flags[s], log_charge=batch.log_charge[s],
                         event_ids=batch.event_ids[s], keys=batch.keys[s], report=batch.report,
                         scaling=batch.scaling, split=batch.split, occupancy=batch.occupancy)
def _to_device(batch, device):
    if batch is None or device.type == "cpu":
        return batch
    return replace(batch, features=batch.features.to(device), target=batch.target.to(device),
                   hidden=batch.hidden.to(device), context_index=batch.context_index.to(device),
                   target_index=batch.target_index.to(device))
class _FixedMasker:
    """Replays cached masks so the nominal twin keeps the original view."""
    def __init__(self, geometry, split, masks, occupancy):
        self.geometry, self.split, self.masks, self.occupancy = geometry, split, masks, occupancy
        self.cell, self.counter = "", None
    def mask(self, event_id, hit_pmt=None):
        m = self.masks[int(event_id)]
        return ViewMask(m.split, m.event_id, m.context, m.target, True, m.attempts,
                        m.context_hits, m.target_hits, m.offset, m.attempt_hits,
                        m.extra_hidden, dict(m.key))
    def report(self):
        return {"split": self.split, "occupancy": self.occupancy, "n_min": 0,
                "geometry_sha256": self.geometry.digest, "n_events": len(self.masks)}
class TrainingLoader:
    """Deterministic loader: 4 events from each of 8 runs per batch; keyed by (seed, step)."""
    def __init__(self, dataset, geometry, *, split, seed, occupancy="natural", n_min=lb.N_MIN,
                 batch=32, twin=False, cache_runs=12, scaling=None):
        self.dataset, self.geometry, self.split, self.seed = dataset, geometry, split, int(seed)
        self.occupancy, self.n_min, self.batch, self.twin = occupancy, int(n_min), int(batch), bool(twin)
        self.cache_runs, self.scaling, self.rows = int(cache_runs), scaling, list(dataset.runs)
        self._cache, self._order = {}, []
    def _entry(self, index):
        key = self.rows[index]["run_dir"]
        if key in self._cache:
            return self._cache[key]
        run_dir = self.dataset.directory / "runs" / key
        batch = _tokenise_run(run_dir, self.geometry, split=self.split, seed=int(self.dataset.seed),
                              occupancy=self.occupancy, n_min=self.n_min, scaling=self.scaling)
        masks = _masks_for(run_dir, self.geometry, split=self.split, seed=int(self.dataset.seed),
                           occupancy=self.occupancy, n_min=self.n_min, event_ids=batch.event_ids)
        entry = {"batch": batch, "masks": masks, "twin": None, "run_dir": run_dir}
        self._cache[key] = entry; self._order.append(key)
        while len(self._order) > self.cache_runs:
            self._cache.pop(self._order.pop(0), None)
        return entry
    def batch_at(self, step):
        rng = lb.keyed_rng(self.seed, "lucid/batches", int(step))
        chosen = np.sort(rng.choice(len(self.rows), min(8, len(self.rows)), replace=False))
        per = max(1, self.batch // chosen.size)
        entries, rows, twins, twin_rows = [], [], [], []
        for index in chosen:
            row = self.rows[int(index)]
            take = np.sort(int(row["event_id_start"]) + rng.permutation(int(row["n_events"]))[:per])
            entry = self._entry(int(index)); found = np.flatnonzero(np.isin(entry["batch"].event_ids, take))
            if not found.size:
                continue
            entries.append(entry["batch"]); rows.append(found)
            if self.twin:
                if entry["twin"] is None:
                    entry["twin"] = _tokenise_run(
                        entry["run_dir"], self.geometry, split=self.split, seed=int(self.dataset.seed),
                        occupancy=self.occupancy, n_min=0, scaling=self.scaling,
                        masker=_FixedMasker(self.geometry, self.split, entry["masks"], self.occupancy))
                twins.append(entry["twin"])
                twin_rows.append(np.flatnonzero(np.isin(entry["twin"].event_ids,
                                                        entry["batch"].event_ids[found])))
        return _stack(entries, rows), (_stack(twins, twin_rows) if self.twin else None)
def ensure_dark_bank(*, data_root, geometry, generator_config, seed, records, code_sha=None):
    """Noise-only windows from the generator's own readout at yield = 0 (flagged choice)."""
    directory = Path(data_root).resolve() / "lucid" / f"dark_noise__s{int(seed)}__n{int(records)}"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["geometry_sha256"] != geometry.digest:
            raise RuntimeError("dark bank: cached geometry differs; refusing silently")
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    state = lb.nominal_state(geometry, "lucid/dark", seed=int(seed))
    readout = lb.make_readout(geometry, generator_config)
    hits = [readout(state, replace(lb.draw_event(geometry, generator_config, "lucid/dark", i,
                                                 seed=int(seed)), yield_photons=0.0))
            for i in range(int(records))]
    lb.write_sparse_hits(directory / "hits.npz", lb.sparse_from_readouts(np.arange(int(records), np.int64), hits))
    atomic_write_json(manifest_path, {
        "schema": "jepa-lucid-dark-bank-v0", "seed": int(seed), "n_records": int(records),
        "readout_label": READOUT_LABEL, "reported_results": False, "geometry_sha256": geometry.digest,
        "generator_config_sha256": generator_config.digest,
        "method": "readout_synthetic_unrepaired at yield_photons = 0, nominal state",
        "status": "engineering_choice_flagged", "code_sha": code_sha, "created_utc": utcnow()})
    write_sha256sums(directory)
    return directory
# --- model / training -------------------------------------------------------
def build_arm(arm, geometry, arms_config, cell, seed, beta=None, dark_bank=None, encoder_config=None):
    settings = la.ArmSettings.from_config(arms_config, cell=(cell.s_arm, cell.s_p), beta=beta)
    return la.build_arm(arm, geometry, settings=settings, beta=beta, seed=int(seed),
                        dark_bank=dark_bank, encoder_config=encoder_config)
def build_optimizers(arm, config, *, allow_pending=False):
    """Registered AdamW/Muon groups; refuses while the numerics are pending_review."""
    t = config["training"]
    if t["optimizer"] == "pending_review":
        if not allow_pending:
            raise RuntimeError("training.optimizer is pending_review: the WP5 config review must freeze "
                               "optimizer/lr/schedule before any registered or development training "
                               "(§3.5.5; gate to the Lead)")
        spec, label, family = dict(t["smoke_optimizer"]), "engineering_smoke_only_not_registered", "adamw"
    else:
        spec, label = dict(t["optimizer_detail"]), t["optimizer"]
        family = t["optimizer_detail"].get("family", t["optimizer"])
    lr, groups = spec.get("learning_rate"), arm.parameter_groups()
    if not isinstance(lr, (int, float)) or float(lr) <= 0:
        raise RuntimeError(f"optimizer {label!r}: learning_rate is not frozen")
    betas, wd = tuple(spec.get("betas", (0.9, 0.999))), float(spec.get("weight_decay", 0.0))
    if family == "adamw":
        opts = [torch.optim.AdamW(list(groups["muon"]) + list(groups["adamw"]), lr=float(lr),
                                  betas=betas, weight_decay=wd)]
    elif family == "muon_adamw":
        from reconstruction_model.model import Muon
        opts = [Muon(groups["muon"], lr=float(spec.get("muon_learning_rate", lr)),
                     ns_steps=int(spec.get("muon_ns_steps", 5))),
                torch.optim.AdamW(groups["adamw"], lr=float(lr), betas=betas, weight_decay=wd)]
    else:
        raise RuntimeError(f"unknown optimizer family {family!r}")
    return opts, {"family": family, "label": label, "learning_rate": float(lr),
                  "betas": list(betas), "weight_decay": wd, "schedule": spec.get("schedule"),
                  "warmup_steps": int(spec.get("warmup_steps", 0)),
                  "gradient_clip": spec.get("gradient_clip"),
                  "pending_review": t["optimizer"] == "pending_review",
                  "groups": {k: len(v) for k, v in groups.items()}}
def _checkpoint(arm, optimizers, *, identity, step, elapsed, config_sha):
    payload = {"schema": SCHEMA_CHECKPOINT, "config_sha256": config_sha, "identity": dict(identity),
               "step": int(step), "steps_scheduled": int(identity["steps_scheduled"]),
               "elapsed_seconds": float(elapsed),
               "model": {k: v.detach().cpu() for k, v in arm.state_dict().items()},
               "optimizers": [o.state_dict() for o in optimizers]}
    generator = getattr(arm, "_generator", None)
    if generator is not None:
        payload["generator_state"] = generator.get_state()
    return payload
def _save(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp"); torch.save(payload, tmp); os.replace(tmp, path)
def _validate(payload, *, identity, config_sha):
    if payload.get("schema") != SCHEMA_CHECKPOINT:
        raise RuntimeError("not a LUCiD WP5 checkpoint")
    for name in ("run_id", "cell_id", "arm", "seed", "beta", "steps_scheduled"):
        if payload.get("identity", {}).get(name) != identity.get(name):
            raise RuntimeError(f"checkpoint {name} differs from the requested run")
    if payload.get("config_sha256") != config_sha:
        raise RuntimeError("checkpoint config sha256 differs")
def train_loop(*, arm, loader, steps, optimizers, checkpoint_path, identity, config_sha,
               checkpoint_minutes=30.0, checkpoint_every=500, stop_after=None, resume=None,
               step_callback=None, log_path=None, accum_steps=1, oracle=False, log_every=None) -> dict:
    """Deterministic training; SIGTERM -> checkpoint + interrupted; the final step is kept."""
    start, elapsed_before, resumed = 0, 0.0, None
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        _validate(payload, identity=identity, config_sha=config_sha)
        arm.load_state_dict(payload["model"], strict=True)
        for optimizer, state in zip(optimizers, payload["optimizers"]):
            optimizer.load_state_dict(state)
        generator = getattr(arm, "_generator", None)
        if generator is not None and payload.get("generator_state") is not None:
            generator.set_state(payload["generator_state"])
        start, resumed = int(payload["step"]), int(payload["step"])
        elapsed_before = float(payload.get("elapsed_seconds", 0.0))
        if start > int(steps):
            raise RuntimeError("checkpoint step beyond the scheduled steps")
    stop = {"signal": None}
    previous = signal.signal(signal.SIGTERM, lambda signum, _f: stop.__setitem__("signal", int(signum)))
    log_every = max(1, int(steps) // 100) if log_every is None else int(log_every)
    next_timed = time.perf_counter() + checkpoint_minutes * 60.0 if checkpoint_minutes else math.inf
    started, step, interrupted, last = time.perf_counter(), int(start), False, {}
    try:
        while step < int(steps):
            if stop["signal"] is not None or (stop_after is not None and step >= int(stop_after)):
                interrupted = True; break
            batch, twin = loader.batch_at(step)
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            accum = max(1, int(accum_steps)); micro = max(1, math.ceil(int(batch.features.size(0)) / accum))
            total, metrics = 0.0, {}
            for chunk in range(accum):
                begin, end = chunk * micro, min((chunk + 1) * micro, int(batch.features.size(0)))
                if begin >= end:
                    continue
                out = arm(_slice(batch, begin, end), oracle=oracle,
                          twin_batch=(_slice(twin, begin, end) if twin is not None else None), step=step)
                loss = out["loss"]
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at step {step + 1}")
                (loss / accum).backward(); total += float(loss.detach()) / accum
                for key, value in out["metrics"].items():
                    metrics[key] = metrics.get(key, 0.0) + float(value) / accum
            if identity.get("gradient_clip"):
                torch.nn.utils.clip_grad_norm_(arm.parameters(), float(identity["gradient_clip"]))
            for optimizer in optimizers:
                optimizer.step()
            if arm.arm == "ijepa_ema":
                arm.update_target(step)
            last, step = dict(metrics, loss=total), step + 1
            if step % log_every == 0 or step == int(steps):
                line = {"step": step, "elapsed_seconds": round(time.perf_counter() - started + elapsed_before, 2),
                        "metrics": last}
                if log_path is not None:
                    with Path(log_path).open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(line, sort_keys=True) + "\n")
            if step_callback is not None:
                step_callback(step)
            timed = time.perf_counter() >= next_timed
            if (step % int(checkpoint_every) == 0 or timed) and step < int(steps):
                _save(checkpoint_path, _checkpoint(arm, optimizers, identity=identity, step=step,
                                                   elapsed=time.perf_counter() - started + elapsed_before,
                                                   config_sha=config_sha))
                if timed:
                    next_timed = time.perf_counter() + checkpoint_minutes * 60.0
        elapsed = time.perf_counter() - started + elapsed_before
        _save(checkpoint_path, _checkpoint(arm, optimizers, identity=identity, step=step,
                                           elapsed=elapsed, config_sha=config_sha))
    finally:
        signal.signal(signal.SIGTERM, previous)
    return {"step": int(step), "interrupted": bool(interrupted),
            "interrupted_by": "sigterm" if stop["signal"] is not None else ("stop_after" if interrupted else None),
            "resumed_from_step": resumed, "elapsed_seconds": round(elapsed, 3), "metrics": last}
def _status(*, run_id, arm, cell_id, seed, beta, outcome, started_at, started_epoch, steps,
            scheduled, device, reason=None, extra=None):
    payload = {"schema": "jepa-lucid-run-status-v0", "run_id": run_id, "arm": arm,
               "cell_id": cell_id, "seed": int(seed), "beta": beta, "host": host_name(),
               "device": device, "outcome": outcome, "reason": reason, "steps_completed": int(steps),
               "steps_scheduled": int(scheduled), "started_at_utc": started_at,
               "finished_at_utc": utcnow(), "wall_seconds": round(time.time() - started_epoch, 2),
               "registered": False}
    payload.update(extra or {}); return payload
def _append(root, row):
    with (Path(root) / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
def run_one(*, arm_name, cell, seed, beta, config, arms_config, geometry, loader_dataset,
            probe_datasets, dark_bank, out_root, data_root, config_sha, code_sha, status_text,
            device, steps=None, micro_batch=None, resume=None, stop_after=None,
            checkpoint_minutes=None, evaluate=False, allow_pending=False, step_callback=None,
            probe_max_events=None, save_embeddings=True, log=print) -> dict:
    """Train, freeze and optionally evaluate one run; writes only under out/data."""
    out_root, data_root = Path(out_root).resolve(), Path(data_root).resolve()
    run_id = run_id_for(cell.cell_id, arm_name, seed, beta)
    run_dir = _guard(out_root / "runs" / run_id, out_root, data_root); run_dir.mkdir(parents=True, exist_ok=True)
    steps = int(config["training"]["steps"] if steps is None else steps)
    batch = int(config["training"]["batch"]); micro = batch if micro_batch is None else int(micro_batch)
    if micro < 1 or batch % micro:
        raise ValueError(f"micro_batch {micro} must divide {batch}")
    started_at, started_epoch = utcnow(), time.time()
    identity = {"run_id": run_id, "cell_id": cell.cell_id, "arm": arm_name, "seed": int(seed),
                "beta": beta, "steps_scheduled": int(steps), "gradient_clip": None}
    status_path = run_dir / "run_status.json"
    atomic_write_json(status_path, _status(run_id=run_id, arm=arm_name, cell_id=cell.cell_id,
                                           seed=seed, beta=beta, outcome="running", started_at=started_at,
                                           started_epoch=started_epoch, steps=0, scheduled=steps,
                                           device=str(device)))
    row = {"schema": SCHEMA_RUN, "run_id": run_id, "arm": arm_name, "cell_id": cell.cell_id,
           "role": cell.role, "law": cell.law, "setting": cell.setting, "split": cell.split,
           "s_arm": cell.s_arm, "s_p": cell.s_p, "occupancy": cell.occupancy, "seed": int(seed),
           "beta": beta, "config_sha256": config_sha, "code_sha": code_sha, "git_status": status_text,
           "clean_tree": status_text.strip() == "", "host": host_name(), "device": str(device),
           "readout_label": READOUT_LABEL, "reported_results": False, "geometry_sha256": geometry.digest,
           "geometry_summary": geometry.summary(), "loader_dataset": loader_dataset.key,
           "loader_dataset_manifest_sha256": loader_dataset.manifest_sha256(),
           "probe_datasets": {k: v.key for k, v in probe_datasets.items()},
           "probe_dataset_sha256": {k: v.manifest_sha256() for k, v in probe_datasets.items()},
           "steps": steps, "batch": batch, "micro_batch": micro, "gradient_accumulation": batch // micro,
           "optimizer": None, "started_at_utc": started_at, "finished_at_utc": None,
           "wall_seconds": None, "steps_completed": 0, "registered": False}
    try:
        torch.manual_seed(int(seed)); np.random.seed(int(seed) % 2 ** 32)
        arm = build_arm(arm_name, geometry, arms_config, cell, seed, beta=beta, dark_bank=dark_bank)
        optimizers, optimiser = build_optimizers(arm, config, allow_pending=allow_pending)
        row["optimizer"], identity["gradient_clip"] = optimiser, optimiser.get("gradient_clip")
        loader = TrainingLoader(loader_dataset, geometry, split=cell.split, seed=int(seed),
                                occupancy=cell.occupancy, n_min=int(config["views"]["n_min"]),
                                batch=batch, twin=(arm_name == la.ORACLE_ARM),
                                cache_runs=int(config["data"].get("loader_cache_runs", 12)))
        checkpoint_path = run_dir / "checkpoint.pt"
        training = train_loop(
            arm=arm, loader=loader, steps=steps, optimizers=optimizers, checkpoint_path=checkpoint_path,
            identity=identity, config_sha=config_sha,
            checkpoint_minutes=float(config["training"]["checkpoint_minutes"]
                                     if checkpoint_minutes is None else checkpoint_minutes),
            checkpoint_every=int(config["training"]["checkpoint_every_steps"]), stop_after=stop_after,
            resume=resume, step_callback=step_callback, log_path=run_dir / "train_log.jsonl",
            accum_steps=batch // micro, oracle=(arm_name == la.ORACLE_ARM))
        row["steps_completed"], row["resumed_from_step"] = int(training["step"]), training["resumed_from_step"]
        extra = {"optimizer": optimiser, "steps_completed": int(training["step"]),
                 "checkpoint_sha256": sha256_file(checkpoint_path),
                 "resumed_from_step": training["resumed_from_step"]}
        if training["interrupted"]:
            row.update(outcome="interrupted", interrupted_by=training["interrupted_by"],
                       finished_at_utc=utcnow(), wall_seconds=round(time.time() - started_epoch, 2))
            atomic_write_json(status_path, _status(run_id=run_id, arm=arm_name, cell_id=cell.cell_id,
                                                   seed=seed, beta=beta, outcome="interrupted",
                                                   started_at=started_at, started_epoch=started_epoch,
                                                   steps=training["step"], scheduled=steps, device=str(device),
                                                   reason=training["interrupted_by"], extra=extra))
            write_sha256sums(run_dir); _append(out_root, row)
            return {"manifest": row, "training": training, "run_dir": run_dir}
        if int(training["step"]) != int(steps):
            raise RuntimeError(f"{run_id}: stopped at {training['step']}, not {steps}")
        final_path = run_dir / "final.pt"
        _save(final_path, _checkpoint(arm, optimizers, identity=identity, step=steps,
                                      elapsed=training["elapsed_seconds"], config_sha=config_sha))
        row["final_checkpoint_sha256"], row["final_step"] = sha256_file(final_path), int(steps)
        if evaluate:
            probes = evaluate_arm_run(arm, arm_name=arm_name, cell=cell, geometry=geometry,
                                      probe_datasets=probe_datasets, config=config, device=device,
                                      mask_seed=int(config["views"]["mask_seed"]),
                                      max_events=probe_max_events, save_embeddings=save_embeddings,
                                      run_dir=run_dir, physics_method=physics_method)
            row["probes"] = {k: (v.get("r2_mean") if isinstance(v, dict) else v)
                             for k, v in probes["results"].items()}
            row["physics"], row["probe_files"] = probes.get("physics", {}), probes.get("files", {})
        row["registered"] = bool(not geometry.synthetic and cell.role != "development"
                                 and int(steps) == int(config["training"]["steps"])
                                 and probe_max_events is None
                                 and config["training"]["optimizer"] != "pending_review"
                                 and status_text.strip() == "")
        row.update(outcome="complete", finished_at_utc=utcnow(),
                   wall_seconds=round(time.time() - started_epoch, 2))
        atomic_write_json(status_path, _status(run_id=run_id, arm=arm_name, cell_id=cell.cell_id,
                                               seed=seed, beta=beta, outcome="complete",
                                               started_at=started_at, started_epoch=started_epoch,
                                               steps=steps, scheduled=steps, device=str(device), extra=extra))
        write_sha256sums(run_dir); _append(out_root, row)
        return {"manifest": row, "training": training, "run_dir": run_dir}
    except Exception as error:
        row.update(outcome="failed", failure=f"{type(error).__name__}: {error}",
                   finished_at_utc=utcnow(), wall_seconds=round(time.time() - started_epoch, 2))
        atomic_write_json(status_path, _status(run_id=run_id, arm=arm_name, cell_id=cell.cell_id,
                                               seed=seed, beta=beta, outcome="failed",
                                               started_at=started_at, started_epoch=started_epoch,
                                               steps=row.get("steps_completed", 0), scheduled=steps,
                                               device=str(device), reason=row["failure"]))
        _append(out_root, row); raise
# --- evaluation -------------------------------------------------------------
def available_kinds(cell, config):
    kinds = {form: list(names) for form, names in config["evaluation"]["form_kinds"].items()}
    for names in kinds.values():
        if cell.law not in ("F2", "F2p") and cell.role != "r11b" and "f2_variable" in names:
            names.remove("f2_variable")
        if cell.law != "T" and "age" in names:
            names.remove("age")
    return kinds
def _probe_batch(run_dir, geometry, *, split, mask_seed, occupancy, scaling, limit=None):
    return pe.tokenise(run_dir, geometry, split=split, seed=mask_seed, occupancy=occupancy,
                       scaling=scaling, masker=ViewMasker(geometry, split, seed=mask_seed,
                                                          occupancy=occupancy, n_min=0), limit=limit)
def _f32(data):
    return la.RunProbeData(data.run_id, data.event_ids, np.asarray(data.tokens, np.float32),
                           np.asarray(data.context_tokens, np.float32), np.asarray(data.raw, np.float32),
                           data.hit, data.hidden, data.module_index, data.truth)
def _concat(items):
    if not items:
        raise ValueError("no probe rows")
    return la.ProbeDataset(items[0].kind, items[0].form, np.concatenate([i.x for i in items]),
                           np.concatenate([i.y for i in items]), np.concatenate([i.run_ids for i in items]),
                           items[0].target_names, {"key": np.concatenate([i.meta["key"] for i in items])})
def probe_tables(model, dataset, geometry, *, split, kinds_by_form, mask_seed, max_events=None, device=None):
    """All four §3.5.6 probe forms; probe events are never excluded on N_min."""
    scaling, tables, device = pe.FeatureScaling.from_geometry(geometry), {}, device or torch.device("cpu")
    for form, kinds in kinds_by_form.items():
        accum = {kind: [] for kind in kinds}
        for row in dataset.runs:
            batch = _probe_batch(dataset.directory / "runs" / row["run_dir"], geometry, split=split,
                                 mask_seed=mask_seed, occupancy="natural", scaling=scaling, limit=max_events)
            data = _f32(la.probe_data(model, _to_device(batch, device), _run_truth(row, dataset, geometry),
                                      geometry, run_id=row["run_id"]))
            for kind in kinds:
                accum[kind].append(la.build_probe_dataset([data], kind, form))
        for kind in kinds:
            tables[(kind, form)] = _concat(accum[kind])
    return tables
def _score(train, test):
    return {f"{k}::{f}": la.probe_r2(train[k], test[k]) for k in sorted(train) if k in test}
def _vertex_error(predictions, truth):
    predictions, truth = np.asarray(predictions, float), np.asarray(truth, float)
    keep = np.isfinite(predictions).all(1) & np.isfinite(truth).all(1)
    if not keep.any():
        return {"status": "not_evaluable", "reason": "no finite vertex prediction", "resolution": None}
    errors = np.linalg.norm(predictions[keep] - truth[keep], axis=1)
    return {"status": "evaluated", "resolution": float(errors.mean()), "median": float(np.median(errors)),
            "n": int(errors.size), "failure_rate": float(1.0 - keep.mean())}
def _vertex_resolution(train, test, method="ridge", seed=0):
    if method == "mlp":
        fitted = _fit_mlp(train, seed=seed)
        return (_vertex_error(fitted["predict"](test.x), test.y) if fitted.get("status") == "evaluated"
                else {"status": "not_evaluable", "reason": fitted.get("reason"), "resolution": None})
    return _vertex_error(la.probe_predictions(la.fit_probe(train), test), test.y)
def _fit_mlp(train, *, seed=0, hidden=256, epochs=200, patience=10):
    """§3.5.6 MLP physics probe: 2 x 256, early stopping on 10% of probe-train."""
    x, y = np.asarray(train.x, float), np.asarray(train.y, float)
    if x.shape[0] < 10:
        return {"status": "not_evaluable", "reason": "too few probe-train rows"}
    centre, scale = x.mean(0), x.std(0); scale[scale == 0] = 1.0; y_centre = y.mean(0, keepdims=True)
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], hidden), torch.nn.GELU(),
                                torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
                                torch.nn.Linear(hidden, y.shape[1]))
    xt = torch.as_tensor((x - centre) / scale, dtype=torch.float32)
    yt = torch.as_tensor(y - y_centre, dtype=torch.float32)
    cut, best, state, bad = max(1, int(0.9 * x.shape[0])), math.inf, None, 0
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(int(epochs)):
        model.train(); optimizer.zero_grad()
        torch.nn.functional.mse_loss(model(xt[:cut]), yt[:cut]).backward(); optimizer.step(); model.eval()
        with torch.no_grad():
            validation = float(torch.nn.functional.mse_loss(model(xt[cut:]), yt[cut:]))
        if validation < best:
            best, state, bad = validation, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    def predict(values):
        with torch.no_grad():
            return model(torch.as_tensor((np.asarray(values, float) - centre) / scale,
                                         dtype=torch.float32)).numpy() + y_centre
    return {"status": "evaluated", "predict": predict}
def _r11_block(train_module, test_module, train_f2, test_f2):
    """§3.5.9 R11 cross-leakage from the two run-mean probes."""
    module_pred = la.probe_predictions(la.fit_probe(train_module), test_module)
    f2_pred = la.probe_predictions(la.fit_probe(train_f2), test_f2).ravel()
    runs = np.asarray(test_module.run_ids)
    f2_by_run = dict(zip(np.asarray(test_f2.run_ids).tolist(), test_f2.y[:, 0].tolist()))
    f2p_by_run = dict(zip(np.asarray(test_f2.run_ids).tolist(), f2_pred.tolist()))
    unique = sorted(set(runs.tolist()))
    mod_mean = {run: float(test_module.y[runs == run, 0].mean()) for run in unique}
    return {"leakage_i": la.squared_correlation(module_pred[:, 0], [f2_by_run[r] for r in runs]),
            "leakage_ii": la.squared_correlation([f2p_by_run[r] for r in unique],
                                                [mod_mean[r] for r in unique]),
            "n_run_module_rows": int(module_pred.shape[0]), "n_runs": len(unique)}
def evaluate_arm_run(arm, *, arm_name, cell, geometry, probe_datasets, config, device, mask_seed,
                     max_events=None, save_embeddings=True, run_dir=None, physics_method="ridge",
                     compute_reference=False):
    """The four probe forms over the six targets, the physics probe and R11 leakage."""
    model = getattr(arm, "jepa", arm); kinds = available_kinds(cell, config)
    train = probe_tables(model, probe_datasets["train"], geometry, split=cell.split, kinds_by_form=kinds,
                         mask_seed=mask_seed, max_events=max_events, device=device)
    test = probe_tables(model, probe_datasets["test"], geometry, split=cell.split, kinds_by_form=kinds,
                        mask_seed=mask_seed, max_events=max_events, device=device)
    results = _score(train, test); physics = {"model_vertex_probe": physics_method}
    if ("vertex", "per_event") in train:
        physics.update(_vertex_resolution(train[("vertex", "per_event")],
                                          test[("vertex", "per_event")], method=physics_method))
    if compute_reference:
        physics["reference"] = _reference_vertex_resolution(probe_datasets["test"], geometry, config)
    r11 = (_r11_block(train[("module_gain", "run_mean")], test[("module_gain", "run_mean")],
                      train[("f2_variable", "run_mean")], test[("f2_variable", "run_mean")])
           if ("module_gain", "run_mean") in train and ("f2_variable", "run_mean") in train else {})
    payload = {"schema": "jepa-lucid-probes-v0", "arm": arm_name, "cell_id": cell.cell_id,
               "split": cell.split, "primary_form": la.PRIMARY_PROBE_FORM, "results": results,
               "physics": physics, "r11": r11, "registered": False, "readout_label": READOUT_LABEL}
    if run_dir is not None:
        if save_embeddings:
            means = _embedding_means(model, probe_datasets["test"], geometry, split=cell.split,
                                     mask_seed=mask_seed, max_events=max_events, device=device)
            np.savez_compressed(Path(run_dir) / "embeddings.npz", run_ids=np.asarray(means["run_ids"]),
                                mean_tokens=np.asarray(means["tokens"], np.float16))
            payload["files"] = {"probes": "probes.json", "embeddings": "embeddings.npz"}
        atomic_write_json(Path(run_dir) / "probes.json", _safe(payload))
    return payload
def _embedding_means(model, dataset, geometry, *, split, mask_seed, max_events=None, device=None):
    scaling, ids, means = pe.FeatureScaling.from_geometry(geometry), [], []
    for row in dataset.runs:
        batch = _probe_batch(dataset.directory / "runs" / row["run_dir"], geometry, split=split, mask_seed=mask_seed,
                             occupancy="natural", scaling=scaling, limit=max_events)
        with torch.no_grad():
            tokens = model.encode(_to_device(batch, device or torch.device("cpu")).features).cpu().numpy()
        ids.append(row["run_id"]); means.append(tokens.mean(axis=(0, 1)).astype(np.float32))
    return {"run_ids": ids, "tokens": np.stack(means)}
def _reference_vertex_resolution(dataset, geometry, config):
    """The VERITAS time-of-flight fit on the same events (no gain correction)."""
    try:
        from noise_module_lucid.consequence import FitConfig, fit_vertex
    except Exception as error:
        return {"status": "not_evaluable", "reason": f"consequence unavailable: {error}", "resolution": None}
    path = config["evaluation"].get("physics_reference_path"); v_group = None
    if path:
        try:
            payload = json.loads(Path(path).read_text())
            v_group = payload.get("v_group") or payload.get("v_group_m_per_ns")
        except Exception as error:
            return {"status": "not_evaluable", "reason": f"reference.json unreadable: {error}", "resolution": None}
    fit_config = FitConfig(v_group=float(v_group)) if v_group else FitConfig()
    errors, failures, total = [], 0, 0
    for row in dataset.runs:
        directory = dataset.directory / "runs" / row["run_dir"]; store = lb.read_sparse_hits(directory / "hits.npz")
        with np.load(directory / "events.npz", allow_pickle=False) as data:
            truth = np.asarray(data["vertex_m"], float)
        for i in range(store.event_ids.size):
            span = slice(int(store.offsets[i]), int(store.offsets[i + 1])); total += 1
            fit = fit_vertex(geometry.positions_m, np.asarray(store.pmt_id[span], int),
                             np.asarray(store.time_ns[span], float),
                             np.asarray(store.charge_pe[span], float), fit_config)
            if fit.ok:
                errors.append(float(np.linalg.norm(fit.vertex - truth[i])))
            else:
                failures += 1
    return {"status": "evaluated" if errors else "not_evaluable",
            "resolution": float(np.mean(errors)) if errors else None,
            "failure_rate": failures / max(1, total), "n_ok": len(errors), "n_events": total,
            "v_group": float(fit_config.v_group),
            "v_group_source": str(path) if path else "FitConfig default"}
def ensure_floor(*, arm_name, cell, geometry, config, probe_datasets, mask_seed, config_sha, out_root,
                 data_root, max_events=None, device=None, log=print):
    """The §3.5.6 random-init floor, cached per (arm, cell); init seed = first seed (0)."""
    out_root, data_root = Path(out_root).resolve(), Path(data_root).resolve()
    path = _guard(out_root / "floors" / f"{arm_name}__{cell.cell_id}.json", out_root, data_root)
    identity = {"arm": arm_name, "cell_id": cell.cell_id, "seed": 0, "config_sha256": config_sha,
                "geometry_sha256": geometry.digest,
                "probe_sets": {k: v.key for k, v in probe_datasets.items()},
                "probe_sha256": {k: v.manifest_sha256() for k, v in probe_datasets.items()}}
    if path.exists():
        cached = json.loads(path.read_text())
        if cached.get("identity") != identity:
            raise RuntimeError(f"floor cache {path} belongs to a different arm/cell/config")
        return cached
    model = la.random_init_floor(arm_name, geometry, seed=0); kinds = available_kinds(cell, config)
    train = probe_tables(model, probe_datasets["train"], geometry, split=cell.split, kinds_by_form=kinds,
                         mask_seed=mask_seed, max_events=max_events, device=device)
    test = probe_tables(model, probe_datasets["test"], geometry, split=cell.split, kinds_by_form=kinds,
                        mask_seed=mask_seed, max_events=max_events, device=device)
    payload = {"schema": "jepa-lucid-floor-v0", "identity": identity, "results": _score(train, test),
               "physics": (_vertex_resolution(train[("vertex", "per_event")], test[("vertex", "per_event")])
                           if ("vertex", "per_event") in train else {}),
               "registered": False, "readout_label": READOUT_LABEL}
    path.parent.mkdir(parents=True, exist_ok=True); atomic_write_json(path, payload); write_sha256sums(path.parent)
    log(f"[floor] {arm_name} / {cell.cell_id}: {len(payload['results'])} probe cells")
    return payload
# --- verdicts ---------------------------------------------------------------
def _sd(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return float(np.std(v, ddof=1)) if len(v) > 1 else None
def _mean(v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None
def _floor(*v):
    v = [float(x) for x in v if x is not None and np.isfinite(x)]
    return max(v) if v else None
def _r2(tables, run_id, kind, form):
    r = tables["probes"].get(run_id, {}).get("results", {}).get(f"{kind}::{form}")
    return r.get("r2_mean") if r and r.get("status") == "evaluated" else None
def _floor_r2(tables, arm, cell_id, kind, form):
    r = tables["floors"].get(f"{arm}::{cell_id}", {}).get("results", {}).get(f"{kind}::{form}")
    return r.get("r2_mean") if r and r.get("status") == "evaluated" else None
def _res(tables, run_id):
    v = tables["probes"].get(run_id, {}).get("physics", {}).get("resolution")
    return None if v is None else float(v)
def _values(tables, cell_id, arm, seeds, kind, form):
    return [_r2(tables, run_id_for(cell_id, arm, s), kind, form) for s in range(int(seeds))]
def _excess(values, floor):
    return [None if v is None or floor is None else v - floor for v in values]
def _missing(tables, specs, kind=None, form=None):
    out = []
    for cell_id, arm, seed in specs:
        rid = run_id_for(cell_id, arm, int(seed))
        if rid not in tables["runs"]:
            out.append(rid)
        elif kind is not None and _r2(tables, rid, kind, form) is None:
            out.append(f"{rid}:{kind}::{form}")
    return out
def _median_sd(tables, cell_id, arm_seeds, kind, form):
    spreads = [s for a, n in arm_seeds.items() if (s := _sd(_values(tables, cell_id, a, n, kind, form))) is not None]
    return float(np.median(spreads)) if spreads else None
def _verdict(passed, reason, **values):
    return {"status": "pass" if passed else "fail", "missing": [], "reason": None if passed else reason,
            "values": values}
def _gain_verdict(tables, config, cell, seeds, arm_a, arm_b, sign, reason):
    """Shared R7/R7b core: per-seed ordering, 2x seed floor and the mean comparison."""
    missing = _missing(tables, [(cell, arm_a, s) for s in range(seeds)]
                       + [(cell, arm_b, s) for s in range(seeds)], "module_gain", "run_mean")
    if missing:
        return None, {"status": "not_evaluable", "reason": "incomplete cell/seed set", "missing": missing}
    a = _values(tables, cell, arm_a, seeds, "module_gain", "run_mean")
    b = _values(tables, cell, arm_b, seeds, "module_gain", "run_mean")
    every = all(x is not None and y is not None and sign * (x - y) > 0 for x, y in zip(a, b))
    floor = _floor(_sd(a), _sd(b), _median_sd(tables, cell, config["cells"]["main"]["arms_seeds"],
                                              "module_gain", "run_mean"))
    delta = _mean(a) - _mean(b)
    beyond = bool(floor is not None and abs(delta) > 2 * floor)
    return {"a": a, "b": b, "every": every, "floor": floor, "delta": delta, "beyond": beyond}, None
def verdict_r6(tables, config):
    """§3.5.9 R6: within excess > across excess at every eligible s, beyond the floor."""
    main, seeds = config["cells"]["main"], 5
    spreads = [float(x) for x in main["r6_r7_eligible_spreads"]]
    cells = [f"main_s{int(round(s * 100)):03d}_{split}" for s in spreads for split in ("within", "across")]
    missing = _missing(tables, [(c, "sigreg_jepa", s) for c in cells for s in range(seeds)],
                       "module_gain", "propagated")
    if missing:
        return {"status": "not_evaluable", "reason": "incomplete cell/seed set", "missing": missing}
    medians = {c: _median_sd(tables, c, main["arms_seeds"], "module_gain", "propagated") for c in cells}
    rows, passed = [], True
    for s in spreads:
        w, a = f"main_s{int(round(s * 100)):03d}_within", f"main_s{int(round(s * 100)):03d}_across"
        floors = {c: _floor_r2(tables, "sigreg_jepa", c, "module_gain", "propagated") for c in (w, a)}
        ex_w = _excess(_values(tables, w, "sigreg_jepa", seeds, "module_gain", "propagated"), floors[w])
        ex_a = _excess(_values(tables, a, "sigreg_jepa", seeds, "module_gain", "propagated"), floors[a])
        cross = _floor(_sd(ex_w), _sd(ex_a), medians[w], medians[a])
        delta = (_mean(ex_w) - _mean(ex_a)) if _mean(ex_w) is not None and _mean(ex_a) is not None else None
        within_floor = _floor(_sd(ex_w), medians[w])
        beyond = bool(delta is not None and cross is not None and delta > 2 * cross)
        learned = bool(_mean(ex_w) is not None and within_floor is not None and _mean(ex_w) > 2 * within_floor)
        passed = passed and beyond and learned
        rows.append({"s": s, "within": w, "across": a, "mean_excess_within": _mean(ex_w),
                     "mean_excess_across": _mean(ex_a), "delta_excess": delta, "cross_cell_floor": cross,
                     "within_cell_floor": within_floor, "delta_beyond_rule": beyond,
                     "within_learned": learned, "median_sd_within": medians[w], "median_sd_across": medians[a],
                     "values_within": ex_w, "values_across": ex_a})
    return {"status": "pass" if passed else "fail", "missing": [], "values": rows,
            "reason": None if passed else "R6 conditions not met",
            "metric": "propagated module_gain r2_mean - random-init floor"}
def verdict_r7(tables, config):
    """§3.5.9 R7: anchored module-gain R2 below plain in every seed, physics kept."""
    cell, seeds = "main_s010_within", 5
    core, missing = _gain_verdict(tables, config, cell, seeds, "sigreg_jepa", "sigreg_jepa_anchored", 1, "R7")
    if missing:
        return missing
    plain_res = [_res(tables, run_id_for(cell, "sigreg_jepa", s)) for s in range(seeds)]
    anchor_res = [_res(tables, run_id_for(cell, "sigreg_jepa_anchored", s)) for s in range(seeds)]
    ratio = (_mean(anchor_res) / _mean(plain_res)) if all(v is not None for v in plain_res + anchor_res) \
        and _mean(plain_res) else None
    physics = bool(ratio is not None and ratio <= 1.1)
    return _verdict(core["every"] and core["beyond"] and physics, "criterion not met",
                    plain=core["a"], anchored=core["b"], delta_mean=core["delta"],
                    seed_floor=core["floor"], every_seed_less=core["every"],
                    plain_resolution=plain_res, anchored_resolution=anchor_res,
                    resolution_ratio=ratio, physics_ok=physics) | {"missing": []}
def verdict_r7b(tables, config):
    """§3.5.9 R7b: the dark-only anchor does not remove module drift (seeds 0-2)."""
    cell, seeds = "main_s010_within", 3
    core, missing = _gain_verdict(tables, config, cell, seeds, "sigreg_jepa",
                                  "sigreg_jepa_anchored_dark", 1, "R7b")
    if missing:
        return missing
    threshold = (_mean(core["a"]) - 2 * core["floor"]
                 if _mean(core["a"]) is not None and core["floor"] is not None else None)
    passed = bool(threshold is not None and _mean(core["b"]) is not None
                  and _mean(core["b"]) >= threshold)
    return _verdict(passed, "dark anchor removed the drift", plain=core["a"],
                    dark_anchor=core["b"], mean_plain=_mean(core["a"]), mean_dark=_mean(core["b"]),
                    threshold=threshold, seed_floor=core["floor"]) | {"missing": []}
def _r11_conditions(tables, cell_id, arm, seeds, config):
    f1 = _values(tables, cell_id, arm, seeds, "module_gain", "run_mean")
    f2 = _values(tables, cell_id, arm, seeds, "f2_variable", "run_mean")
    mf1, mf2 = _mean(f1), _mean(f2)
    ff1 = _floor_r2(tables, arm, cell_id, "module_gain", "run_mean")
    ff2 = _floor_r2(tables, arm, cell_id, "f2_variable", "run_mean")
    arms = ({"sigreg_jepa": 5, "sigreg_jepa_anchored": 5, "mae_mse": 5}
            if not cell_id.startswith("r11b") else {arm: seeds})
    med = _median_sd(tables, cell_id, arms, "module_gain", "run_mean")
    sf1, sf2 = _floor(_sd(f1), med), _floor(_sd(f2), med)
    leak_i = [tables["probes"].get(run_id_for(cell_id, arm, s), {}).get("r11", {}).get("leakage_i")
              for s in range(seeds)]
    leak_ii = [tables["probes"].get(run_id_for(cell_id, arm, s), {}).get("r11", {}).get("leakage_ii")
               for s in range(seeds)]
    conditions = {"f1_ok": bool(mf1 is not None and mf1 >= 0.5), "f2_ok": bool(mf2 is not None and mf2 >= 0.5),
                  "f1_margin": bool(mf1 is not None and ff1 is not None and sf1 is not None and mf1 - ff1 > 2 * sf1),
                  "f2_margin": bool(mf2 is not None and ff2 is not None and sf2 is not None and mf2 - ff2 > 2 * sf2),
                  "leakage_ok": bool(_mean(leak_i) is not None and _mean(leak_ii) is not None
                                     and _mean(leak_i) <= 0.1 and _mean(leak_ii) <= 0.1)}
    return {"f1_r2": f1, "f2_r2": f2, "mean_f1": mf1, "mean_f2": mf2, "floor_f1": ff1, "floor_f2": ff2,
            "seed_floor_f1": sf1, "seed_floor_f2": sf2, "leakage_i": leak_i, "leakage_ii": leak_ii,
            "leakage_i_mean": _mean(leak_i), "leakage_ii_mean": _mean(leak_ii), **conditions,
            "all_conditions": all(conditions.values())}
def verdict_r11(tables, config):
    """§3.5.9 R11: F1 and F2 separable on the R11 cell."""
    cell, seeds = "r11_f1f2f3_within", 5
    missing = _missing(tables, [(cell, "sigreg_jepa", s) for s in range(seeds)], "module_gain", "run_mean")
    if missing:
        return {"status": "not_evaluable", "reason": "incomplete cell/seed set", "missing": missing}
    conditions = _r11_conditions(tables, cell, "sigreg_jepa", seeds, config); passed = conditions["all_conditions"]
    return {"status": "pass" if passed else "fail", "missing": [], "values": conditions,
            "reason": None if passed else "one or more R11 conditions not met"}
def verdict_r11b(tables, config):
    """§3.5.9 R11b: co-trained effects stay entangled (only if R11 passes)."""
    r11 = tables.get("verdicts", {}).get("R11") or verdict_r11(tables, config)
    if r11.get("status") != "pass":
        return {"status": "not_evaluable", "reason": "R11 does not pass", "missing": []}
    cell, seeds = "r11b_law_within", 3
    missing = _missing(tables, [(cell, "sigreg_jepa", s) for s in range(seeds)], "module_gain", "run_mean")
    if missing:
        return {"status": "not_evaluable", "reason": "incomplete cell/seed set", "missing": missing}
    conditions = _r11_conditions(tables, cell, "sigreg_jepa", seeds, config); passed = not conditions["all_conditions"]
    return {"status": "pass" if passed else "fail", "missing": [], "values": conditions,
            "reason": None if passed else "all R11 conditions still hold after co-varying training"}
def verdict_r12(tables, config):
    """§3.5.9 R12: propagated per-PMT gain at the floor in both splits, while R6 passes."""
    if not tables.get("detectability", {}).get("f3_ok"):
        return {"status": "not_evaluable", "missing": [],
                "reason": "F3 did not pass §3.5.3a (frozen decision): R12 not_evaluable, no gate"}
    r6 = tables.get("verdicts", {}).get("R6") or verdict_r6(tables, config)
    if r6.get("status") != "pass":
        return {"status": "not_evaluable", "reason": "R6 does not pass", "missing": []}
    seeds = 5
    cells = [f"main_s{int(round(float(s) * 100)):03d}_{split}"
             for s in config["cells"]["main"]["r6_r7_eligible_spreads"] for split in ("within", "across")]
    missing = _missing(tables, [(c, "sigreg_jepa", s) for c in cells for s in range(seeds)], "pmt_gain", "propagated")
    if missing:
        return {"status": "not_evaluable", "reason": "incomplete cell/seed set", "missing": missing}
    rows, passed = [], True
    for cell in cells:
        floor = _floor_r2(tables, "sigreg_jepa", cell, "pmt_gain", "propagated")
        excess = _excess(_values(tables, cell, "sigreg_jepa", seeds, "pmt_gain", "propagated"), floor)
        ok = bool(_mean(excess) is not None and _mean(excess) <= 0.05); passed = passed and ok
        rows.append({"cell": cell, "excess_mean": _mean(excess), "ok": ok,
                     "values": _values(tables, cell, "sigreg_jepa", seeds, "pmt_gain", "propagated")})
    return {"status": "pass" if passed else "fail", "missing": [], "values": rows,
            "reason": None if passed else "propagated per-PMT gain above the floor + 0.05"}
def evaluate_verdicts(tables, config):
    """R6, R7, R7b, R11, R11b, R12; each only on a complete cell/seed set."""
    verdicts = {"R6": verdict_r6(tables, config), "R7": verdict_r7(tables, config),
                "R7b": verdict_r7b(tables, config), "R11": verdict_r11(tables, config)}
    tables["verdicts"] = verdicts
    verdicts["R11b"] = verdict_r11b(tables, config); verdicts["R12"] = verdict_r12(tables, config)
    tables["verdicts"] = verdicts
    return verdicts
# --- aggregate --------------------------------------------------------------
def linear_cka(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); x, y = x - x.mean(0), y - y.mean(0)
    d = np.linalg.norm(x.T @ y, "fro")
    return (float(d ** 2 / (np.linalg.norm(x.T @ x, "fro") * np.linalg.norm(y.T @ y, "fro")))
            if d else 0.0)
def load_records(roots):
    rows, probes, floors, embeddings = {}, {}, {}, {}
    for root in roots:
        root = Path(root)
        if (root / "manifest.jsonl").exists():
            for line in (root / "manifest.jsonl").read_text().splitlines():
                row = json.loads(line) if line.strip() else None
                if row and row.get("outcome") == "complete":
                    rows[row["run_id"]] = row
        if (root / "runs").exists():
            for run_dir in sorted(p for p in (root / "runs").iterdir() if p.is_dir()):
                if (run_dir / "probes.json").exists():
                    probes[run_dir.name] = json.loads((run_dir / "probes.json").read_text())
                if (run_dir / "embeddings.npz").exists():
                    with np.load(run_dir / "embeddings.npz", allow_pickle=False) as data:
                        embeddings[run_dir.name] = {"run_ids": np.asarray(data["run_ids"]),
                                                    "tokens": np.asarray(data["mean_tokens"], float)}
        if (root / "floors").exists():
            for path in sorted((root / "floors").glob("*.json")):
                payload = json.loads(path.read_text())
                floors[f"{payload['identity']['arm']}::{payload['identity']['cell_id']}"] = payload
    return rows, probes, floors, embeddings
def _detectability(path):
    payload = json.loads(Path(path).read_text()); b, y = payload.get("branches", {}), payload.get("yields", {})
    return payload, {"status": payload.get("status"), "source": str(path), "source_sha256": sha256_file(path),
                     "f1_eligible_settings": b.get("f1_eligible_settings"),
                     "f1_excluded_settings": b.get("f1_excluded_settings"), "f2_choice": b.get("f2_choice"),
                     "f3_ok": bool(b.get("f3_ok")), "law_T_ok": bool(b.get("law_T_ok")),
                     "g6": bool(b.get("g6")), "r12_status": b.get("r12_status"),
                     "y_lo": y.get("y_lo"), "y_hi": y.get("y_hi"), "median_yield": y.get("median_yield")}
def _cka_table(embeddings, config):
    out = []
    for cell in enumerate_cells(config):
        for arm in cell.arms:
            vectors = [embeddings[rid]["tokens"] for seed in range(int(cell.seeds[arm]))
                       if (rid := run_id_for(cell.cell_id, arm, seed)) in embeddings]
            for a in range(len(vectors)):
                for b in range(a + 1, len(vectors)):
                    out.append({"cell_id": cell.cell_id, "arm": arm, "seeds": [a, b],
                                "linear_cka": linear_cka(vectors[a], vectors[b])})
    return out
def run_aggregate(args, config) -> dict:
    """Read-only aggregation; writes aggregate/fragment/update under --out only."""
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, probes, floors, embeddings = load_records(args.runs_root)
    _, detect = _detectability(args.detectability or (ROOT / config["detectability_decision"]))
    tables = {"runs": rows, "probes": probes, "floors": floors, "detectability": detect}
    verdicts = evaluate_verdicts(tables, config)
    payload = {"schema": SCHEMA_AGGREGATE, "created_utc": utcnow(),
               "registered": bool(rows) and all(r.get("registered") for r in rows.values()),
               "readout_label": READOUT_LABEL, "reported_results": False, "run_count": len(rows),
               "runs": {rid: {k: r.get(k) for k in ("cell_id", "arm", "seed", "outcome",
                                                    "registered", "wall_seconds")}
                        for rid, r in rows.items()},
               "detectability": detect, "verdicts": verdicts, "cka": _cka_table(embeddings, config)}
    atomic_write_json(out / "aggregate.json", _safe(payload))
    fragment = {"generated_utc": payload["created_utc"], "registered": payload["registered"],
                "note": "PREDICTION_STATUS.json is never edited by this tool",
                "predictions": [{"id": key, "campaign": "E2", "status": value["status"],
                                 "reason": value.get("reason"), "missing": value.get("missing", []),
                                 "seeds": sorted({int(row["seed"]) for row in rows.values()})}
                                for key, value in verdicts.items()]}
    atomic_write_json(out / "predictions_fragment.json", _safe(fragment))
    lines = ["# PREDICTIONS_UPDATE — WP5 LUCiD (generated)", "",
             f"Generated {payload['created_utc']}; registered: {str(payload['registered']).lower()}.",
             "This is a fragment: `PREDICTION_STATUS.json` is not edited.", "",
             "| ID | Status | Reason |", "|---|---|---|"]
    lines += [f"| {k} | {v['status']} | {v.get('reason') or ''} |" for k, v in verdicts.items()]
    lines += ["", "```json", json.dumps(fragment, indent=1, sort_keys=True), "```", ""]
    atomic_write_text(out / "PREDICTIONS_UPDATE.md", "\n".join(lines))
    write_sha256sums(out)
    return payload
# --- development cohort -----------------------------------------------------
def decide_physics_probe(linear_by_arm, reference_resolution, factor=3.0) -> dict:
    """§3.5.6 step 2: MLP only if BOTH linear resolutions exceed 3x the reference fit."""
    values = {k: v for k, v in linear_by_arm.items() if v is not None}
    if reference_resolution is None or len(values) < 2:
        return {"decision": "not_evaluable", "reason": "reference fit or a linear probe missing",
                "linear_vertex_resolution": linear_by_arm,
                "reference_vertex_resolution": reference_resolution, "factor": factor}
    both_worse = all(v > factor * reference_resolution for v in values.values())
    return {"decision": "mlp" if both_worse else "ridge",
            "rule": "MLP iff both linear vertex resolutions > 3 x reference fit",
            "linear_vertex_resolution": linear_by_arm, "reference_vertex_resolution": reference_resolution,
            "ratios": {k: v / reference_resolution for k, v in values.items()}, "factor": factor}
def select_beta(table: Sequence[dict], plain_resolution, ratio=1.1) -> dict:
    """§3.5.6 beta rule: the largest beta with vertex resolution <= 1.1x plain."""
    if plain_resolution is None:
        return {"selected": None, "status": "not_evaluable", "reason": "plain resolution missing"}
    qualifying = [row for row in table if row.get("resolution") is not None
                  and row["resolution"] <= ratio * plain_resolution]
    if not qualifying:
        return {"selected": None, "status": "none_qualified", "table": list(table),
                "plain_resolution": plain_resolution, "ratio": ratio}
    best = max(qualifying, key=lambda row: float(row["beta"]))
    return {"selected": float(best["beta"]), "status": "selected", "table": list(table),
            "plain_resolution": plain_resolution, "ratio": ratio}
def _write_decision(path: Path, payload: dict):
    payload = dict(payload, registered=False,
                   note="frozen development decision; registered=false until the full grid runs")
    atomic_write_json(path, payload); write_sha256sums(path.parent); return payload
def run_dev(args, config, *, geometry, gen_config, arms_config, data_root, out_root, config_sha,
            code_sha, status_text, device, dry_run=False, log=print) -> dict:
    """Development cohort steps 1-3; step 4 is not in §3.5.6 and is refused."""
    if int(args.step) == 4:
        raise RuntimeError("development step 4 (robustness seed selection) is not in runbook "
                           "§3.5.6: gate to the Lead; no decision file written")
    step = int(args.step)
    if step == 1:
        payload, summary = _detectability(args.detectability or (ROOT / config["detectability_decision"]))
        if dry_run:
            print(json.dumps({"step": 1, "detectability": summary}, indent=1, sort_keys=True)); return summary
        path = _guard(Path(out_root) / "dev" / "step1_detectability_decision.json", out_root, data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        return _write_decision(path, {"schema": "jepa-lucid-dev-step1-v0", "step": 1,
                                      "decisions": summary, "decisions_md": payload.get("decisions_md")})
    dev = dev_cell(config); plan = [r for r in development_runs(config) if r["step"] == step]
    if dry_run:
        print(json.dumps({"step": step, "cell": dev.cell_id, "runs": plan}, indent=1, sort_keys=True))
        return {"step": step, "runs": plan}
    if config["training"]["optimizer"] == "pending_review":
        raise RuntimeError("development-cohort steps 2-3 need the WP5 config review to freeze the "
                           "optimizer/lr/schedule (§3.5.5; gate to the Lead)")
    loader_ds = ensure_dataset(data_root=data_root, geometry=geometry, generator_config=gen_config,
                               cohort_spec=config["data"]["cohorts"][dev.train_cohort],
                               cohort_name=dev.train_cohort, law="F1", setting=dev.setting,
                               seed=int(config["data"]["dataset_seed"]), code_sha=code_sha, log=log)
    probe = _probe_sets(config)[dev.probe_set]
    probe_datasets = {role: ensure_dataset(
        data_root=data_root, geometry=geometry, generator_config=gen_config,
        cohort_spec=config["data"]["cohorts"][name], cohort_name=name, law=probe.law,
        setting=probe.setting, seed=int(config["data"]["dataset_seed"]), code_sha=code_sha, log=log)
        for role, name in probe.cohorts().items() if name}
    dark = None
    if any(item["arm"] in la.ANCHOR_ARMS for item in plan):
        directory = ensure_dark_bank(data_root=data_root, geometry=geometry,
                                     generator_config=gen_config,
                                     seed=int(config["data"]["dataset_seed"]),
                                     records=int(config["data"]["dark_noise"]["records"]),
                                     code_sha=code_sha)
        dark = la.dark_bank_from_store(directory / "hits.npz", geometry)
    results = {}
    for item in plan:
        results[item["arm"] + ("" if item["beta"] is None else f"@{item['beta']:g}")] = run_one(
            arm_name=item["arm"], cell=dev, seed=item["seed"], beta=item["beta"], config=config,
            arms_config=arms_config, geometry=geometry, loader_dataset=loader_ds,
            probe_datasets=probe_datasets, dark_bank=dark, out_root=out_root, data_root=data_root,
            config_sha=config_sha, code_sha=code_sha, status_text=status_text, device=device,
            evaluate=True, allow_pending=False, log=log)
    if step == 2:
        linear = {arm: results[arm]["manifest"].get("physics", {}).get("resolution")
                  for arm in config["cells"]["development"]["step2_arms"]}
        reference = _reference_vertex_resolution(probe_datasets["test"], geometry, config)
        path = _guard(Path(out_root) / "dev" / "step2_probe_decision.json", out_root, data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        return _write_decision(path, {"schema": "jepa-lucid-dev-step2-v0", "step": 2,
                                      "probe_primary_form": la.PRIMARY_PROBE_FORM,
                                      "physics_probe": decide_physics_probe(linear, reference.get("resolution")),
                                      "reference": reference, "runs": list(results)})
    plain_path = Path(args.plain_reference) if getattr(args, "plain_reference", None) else None
    plain_resolution = (json.loads(plain_path.read_text()).get("physics", {}).get("resolution")
                        if plain_path and plain_path.exists() else None)
    selections = {}
    for arm in config["cells"]["development"]["step3_arms"]:
        table = [{"beta": item["beta"], "resolution": results[f"{arm}@{item['beta']:g}"]["manifest"]
                  .get("physics", {}).get("resolution")} for item in plan if item["arm"] == arm]
        selections[arm] = select_beta(table, plain_resolution)
    path = _guard(Path(out_root) / "dev" / "step3_beta_decision.json", out_root, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    return _write_decision(path, {"schema": "jepa-lucid-dev-step3-v0", "step": 3,
                                  "plain_resolution": plain_resolution,
                                  "plain_source": str(plain_path) if plain_path else None,
                                  "selections": selections,
                                  "beta_grid": config["cells"]["development"]["beta_grid"]})
# --- scaling / smoke --------------------------------------------------------
def synthetic_layout(n_pmt: int, pmts_per_module: int = 20, radius=1.577, height=2.777):
    """A cylinder-surface synthetic layout with (n_pmt // per) modules (NOT WCTE)."""
    n_pmt, per = int(n_pmt), int(pmts_per_module)
    if n_pmt % per:
        raise ValueError("n_pmt must be divisible by pmts_per_module")
    top = int(round(0.2 * n_pmt)); bottom = int(round(0.2 * n_pmt))
    return lb.from_synthetic({"module_ids": tuple(range(n_pmt // per)), "pmts_per_module": per,
                              "surface_counts": {"top": top, "barrel": n_pmt - top - bottom,
                                                 "bottom": bottom},
                              "radius_m": radius, "height_m": height})
def estimate_activation_mb(batch: int, n_pmt: int, d_model: int, n_blocks: int) -> float:
    """Conservative linear-attention feasibility estimate (not the measured result)."""
    return 4 * int(batch) * int(n_pmt) * int(d_model) * (4 * (int(n_blocks) + 2) + 30) / 1e6
def scaling_features(geometry, batch: int, hit_fraction: float, seed: int = 0):
    rng = np.random.default_rng(int(seed)); base = np.zeros((geometry.n_pmt, pe.N_FEATURES), np.float32)
    base[:, :6] = pe.geometry_features(geometry, pe.FeatureScaling.from_geometry(geometry))
    features = np.repeat(base[None], int(batch), axis=0)
    hit = rng.random((int(batch), geometry.n_pmt)) < float(hit_fraction)
    features[..., pe.LOG_CHARGE_INDEX] = np.where(hit, rng.normal(0.0, 1.0, hit.shape), 0.0)
    features[..., pe.TIME_INDEX] = np.where(hit, rng.normal(0.5, 0.2, hit.shape), 0.0)
    features[..., pe.HIT_INDEX] = hit; half = geometry.n_pmt // 2
    context = torch.as_tensor(np.tile(np.arange(half), (batch, 1)), dtype=torch.long)
    target = torch.as_tensor(np.tile(np.arange(half, geometry.n_pmt), (batch, 1)), dtype=torch.long)
    hidden = torch.zeros((batch, geometry.n_pmt), dtype=torch.bool); hidden[:, half:] = True
    return torch.as_tensor(features), hidden, context, target
def run_scaling(args, config, *, out_root, device, log=print) -> dict:
    """§3.5.5 scaling benchmark (no training); does_not_fit is a recorded result."""
    sizes = [int(x) for x in (args.sizes or config["scaling"]["sizes"]).split(",")]
    results = []
    for n_pmt in sizes:
        host = config["scaling"]["hosts"].get(str(n_pmt), "unknown")
        cap_mb = float(args.memory_cap_mb if args.memory_cap_mb is not None
                       else config["scaling"]["memory_cap_mb"].get(host, 9600))
        estimate = estimate_activation_mb(int(args.batch), n_pmt, pe.EncoderConfig().d_model,
                                          pe.EncoderConfig().n_blocks)
        record = {"n_pmt": n_pmt, "host": host, "memory_cap_mb": cap_mb,
                  "estimated_activation_mb": round(estimate, 1), "batch": int(args.batch),
                  "hit_fraction": float(args.hit_fraction), "device": str(device),
                  "status": None, "does_not_fit": False}
        if estimate > cap_mb:
            record.update(status="does_not_fit", does_not_fit=True,
                          reason="estimated activation memory exceeds the cap")
            results.append(record); log(f"[scaling] {n_pmt}: does_not_fit"); continue
        try:
            geometry = synthetic_layout(n_pmt, int(config["scaling"]["pmts_per_module"]))
            model = pe.LucidPointJepa(pe.EncoderConfig(), geometry).to(device)
            features, hidden, context, target = scaling_features(geometry, int(args.batch),
                                                                float(args.hit_fraction))
            features, hidden = features.to(device), hidden.to(device)
            context, target = context.to(device), target.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            model.predict(model.encode(features, hidden_mask=hidden), context, target).square().mean().backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None
            record.update(status="fit", seconds=round(seconds, 3),
                          events_per_second=(round(int(args.batch) / seconds, 3) if seconds else None),
                          peak_memory_mb=round(peak, 1) if peak else None,
                          parameter_count=int(sum(p.numel() for p in model.parameters())),
                          geometry_sha256=geometry.digest)
        except (RuntimeError, MemoryError, torch.cuda.OutOfMemoryError) as error:
            record.update(status="does_not_fit", does_not_fit=True, reason=str(error)[:300])
        results.append(record); log(f"[scaling] {n_pmt}: {record['status']}")
    payload = {"schema": "jepa-lucid-scaling-v0", "created_utc": utcnow(), "registered": False,
               "no_training": True, "results": results}
    out = Path(out_root) / "scaling.json"; out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, payload); write_sha256sums(out.parent); return payload
def run_smoke(args, config, *, geometry, generator_config, arms_config, data_root, out_root,
              config_sha, code_sha, status_text, device, log=print) -> dict:
    """Tiny end-to-end run with the labelled smoke optimizer; never registered."""
    spread, runs, run_events = 0.05, int(args.runs), int(args.run_events); seed = int(config["data"]["dataset_seed"])
    cell = replace(next(c for c in enumerate_cells(config) if c.cell_id == "main_s005_within"),
                   cell_id="smoke_s005_within", arms=(args.arm,), seeds={args.arm: 1})
    loader_ds = ensure_dataset(
        data_root=data_root, geometry=geometry, generator_config=generator_config,
        cohort_spec={"namespace": "lucid/smoke_train", "event_start": 0,
                     "n_events": runs * run_events, "run_length": run_events},
        cohort_name="smoke_train", law="F1", setting=spread, seed=seed, registered=False,
        code_sha=code_sha, log=log)
    probe_datasets = {}
    if args.evaluate:
        for role, name in (("train", "smoke_probe_train"), ("test", "smoke_probe_test")):
            probe_datasets[role] = ensure_dataset(
                data_root=data_root, geometry=geometry, generator_config=generator_config,
                cohort_spec={"namespace": f"lucid/{name}", "event_start": 0,
                             "n_events": int(args.probe_events), "run_length": int(args.probe_run_events)},
                cohort_name=name, law="F1", setting=spread, seed=seed, registered=False,
                code_sha=code_sha, log=log)
    return run_one(arm_name=args.arm, cell=cell, seed=0, beta=None, config=config,
                   arms_config=arms_config, geometry=geometry, loader_dataset=loader_ds,
                   probe_datasets=probe_datasets, dark_bank=None, out_root=out_root,
                   data_root=data_root, config_sha=config_sha, code_sha=code_sha,
                   status_text=status_text, device=device, steps=int(args.steps or 500),
                   micro_batch=args.micro_batch, resume=Path(args.resume) if args.resume else None,
                   stop_after=args.stop_after, evaluate=bool(args.evaluate), allow_pending=True,
                   probe_max_events=args.probe_max_events, log=log)["manifest"]
# --- grid -------------------------------------------------------------------
def _cohort_dataset(config, data_root, geometry, gen_config, cohort_name, law, setting, seed,
                    kind=None, settings=None, code_sha=None, log=print):
    return ensure_dataset(data_root=data_root, geometry=geometry, generator_config=gen_config,
                          cohort_spec=config["data"]["cohorts"][cohort_name], cohort_name=cohort_name,
                          law=law, setting=setting, seed=seed, settings=settings, setting_kind=kind,
                          code_sha=code_sha, log=log)
def _probe_pair(config, data_root, geometry, gen_config, probe, seed, code_sha=None, log=print):
    out = {}
    for role, cohort in probe.cohorts().items():
        if cohort:
            spec = config["data"]["cohorts"][cohort]
            settings = _settings_for(probe.law, probe.setting_kind,
                                     spec["n_events"] // spec["run_length"], seed)
            out[role] = _cohort_dataset(config, data_root, geometry, gen_config, cohort, probe.law,
                                        probe.setting, seed, probe.setting_kind, settings, code_sha, log)
    return out
def run_grid(args, config, *, geometry, generator_config, arms_config, data_root, out_root,
             config_sha, code_sha, status_text, device, log=print) -> dict:
    """Per-cell independent training; refuses while the optimizer is pending_review."""
    cells = enumerate_cells(config)
    if args.cells:
        wanted = {x.strip() for x in args.cells.split(",")}
        cells = [c for c in cells if c.cell_id in wanted]
        if wanted - {c.cell_id for c in cells}:
            raise RuntimeError(f"unknown cells {sorted(wanted - {c.cell_id for c in cells})}")
    arms = set(args.arms.split(",")) if args.arms else None
    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    if args.dry_run:
        print(json.dumps({"cells": [c.cell_id for c in cells],
                          "runs": [r for c in cells for r in c.runs()
                                   if (arms is None or r["arm"] in arms)
                                   and (seeds is None or r["seed"] in seeds)]}, indent=1, sort_keys=True))
        return {"dry_run": True, "cells": [c.cell_id for c in cells]}
    if config["training"]["optimizer"] == "pending_review":
        raise RuntimeError("training.optimizer is pending_review: the WP5 config review must freeze "
                           "optimizer/lr/schedule before the grid (§3.5.5; gate to the Lead). "
                           "Use --dry-run to validate the plan.")
    beta = {}
    if args.beta_decision:
        beta = {arm: value["selected"] for arm, value in
                json.loads(Path(args.beta_decision).read_text()).get("selections", {}).items()}
    sets, seed = _probe_sets(config), int(config["data"]["dataset_seed"]); done, floors_done = [], set()
    physics_method = "ridge"
    if getattr(args, "physics_probe_decision", None):
        physics_method = json.loads(Path(args.physics_probe_decision).read_text()).get("decision", "ridge")
    for cell in cells:
        if arms and not (set(cell.arms) & arms):
            continue
        spec = config["data"]["cohorts"][cell.train_cohort]
        settings = _settings_for(cell.law, cell.settings_source, spec["n_events"] // spec["run_length"], seed)
        loader_ds = _cohort_dataset(config, data_root, geometry, generator_config, cell.train_cohort,
                                    cell.law, cell.setting, seed, cell.settings_source, settings,
                                    code_sha, log)
        probe_datasets = _probe_pair(config, data_root, geometry, generator_config, sets[cell.probe_set],
                                     seed, code_sha, log)
        for attr in ("robustness_set", "baseline_set", "extrapolation_set"):
            key = getattr(cell, attr)
            if key:
                probe_datasets[attr.split("_")[0]] = _probe_pair(
                    config, data_root, geometry, generator_config, sets[key], seed, code_sha, log)["test"]
        for arm in cell.arms:
            if arms and arm not in arms:
                continue
            if arm in la.BETA_ARMS and beta.get(arm) is None:
                raise RuntimeError(f"{arm} requires a beta decision from `dev --step 3`")
            for s in range(int(cell.seeds[arm])):
                if seeds is not None and s not in seeds:
                    continue
                dark = None
                if arm in la.ANCHOR_ARMS:
                    directory = ensure_dark_bank(data_root=data_root, geometry=geometry,
                                                 generator_config=generator_config, seed=seed,
                                                 records=int(config["data"]["dark_noise"]["records"]),
                                                 code_sha=code_sha)
                    dark = la.dark_bank_from_store(directory / "hits.npz", geometry)
                result = run_one(arm_name=arm, cell=cell, seed=s, beta=beta.get(arm), config=config,
                                 arms_config=arms_config, geometry=geometry, loader_dataset=loader_ds,
                                 probe_datasets=probe_datasets, dark_bank=dark, out_root=out_root,
                                 data_root=data_root, config_sha=config_sha, code_sha=code_sha,
                                 status_text=status_text, device=device, steps=args.steps,
                                 micro_batch=args.micro_batch,
                                 resume=Path(args.resume) if args.resume else None,
                                 stop_after=args.stop_after, evaluate=not args.no_evaluate,
                                 probe_max_events=args.probe_max_events,
                                 physics_method=physics_method, log=log)
                done.append(result["manifest"]["run_id"])
                if not args.no_evaluate and (arm, cell.cell_id) not in floors_done:
                    ensure_floor(arm_name=arm, cell=cell, geometry=geometry, config=config,
                                 probe_datasets=probe_datasets,
                                 mask_seed=int(config["views"]["mask_seed"]), config_sha=config_sha,
                                 out_root=out_root, data_root=data_root,
                                 max_events=args.probe_max_events, device=device, log=log)
                    floors_done.add((arm, cell.cell_id))
    return {"completed": done, "cells": [c.cell_id for c in cells]}
# --- CLI --------------------------------------------------------------------
def build_parser():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True); common.add_argument("--data-root", required=True)
    common.add_argument("--config", default=None); common.add_argument("--device", default="auto")
    common.add_argument("--geometry", default=None,
                        help="LUCiD config/WCTE_geom_config.json, or 'synthetic' for smoke/tests")
    common.add_argument("--generator-config", default=None); common.add_argument("--arms-config", default=None)
    common.add_argument("--steps", type=int, default=None); common.add_argument("--micro-batch", type=int, default=None)
    common.add_argument("--stop-after", type=int, default=None); common.add_argument("--resume", default=None)
    common.add_argument("--probe-max-events", type=int, default=None,
                        help="debug cap per probe run; marks every run non-registered")
    common.add_argument("--dry-run", action="store_true")
    parser = argparse.ArgumentParser(prog="python -m jepa_noise.lucid_run")
    sub = parser.add_subparsers(dest="mode", required=True)
    grid = sub.add_parser("grid", parents=[common]); grid.add_argument("--cells", default=None)
    grid.add_argument("--arms", default=None); grid.add_argument("--seeds", default=None)
    grid.add_argument("--no-evaluate", action="store_true"); grid.add_argument("--beta-decision", default=None)
    grid.add_argument("--physics-probe-decision", default=None)
    dev = sub.add_parser("dev", parents=[common]); dev.add_argument("--step", type=int, choices=(1, 2, 3, 4), required=True)
    dev.add_argument("--detectability", default=None); dev.add_argument("--plain-reference", default=None)
    smoke = sub.add_parser("smoke", parents=[common]); smoke.add_argument("--arm", default="sigreg_jepa")
    smoke.add_argument("--runs", type=int, default=2); smoke.add_argument("--run-events", type=int, default=1000)
    smoke.set_defaults(steps=500)
    smoke.add_argument("--probe-events", type=int, default=500)
    smoke.add_argument("--probe-run-events", type=int, default=50)
    smoke.add_argument("--evaluate", action="store_true"); smoke.add_argument("--modules", type=int, default=8)
    smoke.add_argument("--pmts-per-module", type=int, default=19)
    scaling = sub.add_parser("scaling", parents=[common]); scaling.add_argument("--sizes", default=None)
    scaling.add_argument("--batch", type=int, default=8); scaling.add_argument("--hit-fraction", type=float, default=0.2)
    scaling.add_argument("--memory-cap-mb", type=float, default=None)
    aggregate = sub.add_parser("aggregate"); aggregate.add_argument("--out", required=True)
    aggregate.add_argument("--runs-root", nargs="+", required=True); aggregate.add_argument("--config", default=None)
    aggregate.add_argument("--detectability", default=None)
    aggregate.add_argument("--data-root", default=None, help="accepted, never written")
    return parser
def load_geometry(args, config):
    """Real WCTE geometry (path-pinned by the generator) or a small synthetic fixture."""
    if getattr(args, "dry_run", False) and args.mode == "grid":
        return None  # a dry run validates the plan and must not touch the geometry/data roots
    if args.geometry in (None, "synthetic"):
        if args.mode != "smoke":
            raise RuntimeError("a registered/grid run needs --geometry (the WCTE config); "
                               "synthetic is smoke/test only")
        return lb.from_synthetic({"module_ids": tuple(range(int(args.modules))),
                                  "surface_counts": {"top": int(args.modules) * 4,
                                                     "barrel": int(args.modules) * 11,
                                                     "bottom": int(args.modules) * 4},
                                  "pmts_per_module": int(args.pmts_per_module)})
    return lb.load_wcte_geometry(args.geometry)
def main(argv=None) -> int:
    parser = build_parser(); args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    config = load_run_config(args.config); config_sha = sha256_file(args.config or CONFIG_PATH)
    code_sha, status_text = git_head(ROOT), git_status(ROOT)
    try:
        if args.mode == "aggregate":
            payload = run_aggregate(args, config)
            print(json.dumps({"aggregate": str(Path(args.out) / "aggregate.json"),
                              "runs": payload["run_count"],
                              "verdicts": {k: v["status"] for k, v in payload["verdicts"].items()}},
                             indent=1, sort_keys=True)); return 0
        device = configure_runtime(args.device)
        generator_config = lb.load_generator_config(
            args.generator_config or (ROOT / config["data"]["generator_config"]))
        arms_config = la.load_config(args.arms_config)
        out_root, data_root = Path(args.out), Path(args.data_root)
        if args.mode == "dev":
            geometry = (None if int(args.step) == 1 or args.dry_run
                        else load_geometry(args, config))
            payload = run_dev(args, config, geometry=geometry,
                              gen_config=generator_config, arms_config=arms_config, data_root=data_root,
                              out_root=out_root, config_sha=config_sha, code_sha=code_sha,
                              status_text=status_text, device=device, dry_run=args.dry_run)
            print(json.dumps(payload, indent=1, sort_keys=True)); return 0
        if args.mode == "scaling":
            if args.dry_run:
                print(json.dumps({"sizes": config["scaling"]["sizes"]}, indent=1)); return 0
            payload = run_scaling(args, config, out_root=out_root, device=device)
            print(json.dumps({r["n_pmt"]: r["status"] for r in payload["results"]}, indent=1)); return 0
        if args.mode == "smoke":
            if args.dry_run:
                print(json.dumps({"mode": "smoke", "steps": args.steps, "arm": args.arm}, indent=1)); return 0
            manifest = run_smoke(args, config, geometry=load_geometry(args, config),
                                 generator_config=generator_config, arms_config=arms_config,
                                 data_root=data_root, out_root=out_root, config_sha=config_sha,
                                 code_sha=code_sha, status_text=status_text, device=device)
            print(json.dumps({"run_id": manifest["run_id"], "outcome": manifest["outcome"],
                              "steps": manifest["steps_completed"]}, indent=1)); return 0
        if args.mode == "grid":
            payload = run_grid(args, config, geometry=load_geometry(args, config),
                               generator_config=generator_config, arms_config=arms_config,
                               data_root=data_root, out_root=out_root, config_sha=config_sha,
                               code_sha=code_sha, status_text=status_text, device=device)
            print(json.dumps(payload, indent=1, sort_keys=True)); return 0
    except RuntimeError as error:
        print(f"lucid_run: refused/failed: {error}", file=sys.stderr); return 2
    parser.error(f"unknown mode {args.mode!r}"); return 2
if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
