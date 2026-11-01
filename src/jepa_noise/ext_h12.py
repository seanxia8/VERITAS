# SPDX-License-Identifier: MIT
"""H1/H2 causal-test extension helpers (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-01_H12_CAUSAL.md`` (1 Nov 2026, reviewed).
Extension name: ``h12_causal_2026-11-01``.

Phase 0 (E0/E1) is evaluation-only on the ``fix9000``/``init9000`` encoders;
Phase 1 is the author-approved E2 (geometry feature ablation) and E3 (frozen
random target) training, two seeds each, on the frozen development cell
``dev_s005_within``.  The variant hooks are installed at process level on
``lucid_run.build_arm`` (precedent: ``patch_stopgrad``) and removed afterwards;
no registered file, config or verdict is edited.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from . import ext_readout as er
from . import lucid_arms as la
from . import lucid_bombs as lb
from . import lucid_run as lr
from . import point_encoder as pe

EXTENSION = "h12_causal_2026-11-01"
VARIANTS = ("geomblind", "frozen_target")
ARM_NAME = "sigreg_jepa"
CELL_ID = "dev_s005_within"
EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_h12_20261101")
READOUT_EXT = Path("/ceph/dwong/jepa/returns/ext_readout_20261101")
RESULTS_DIR = "results/jepa/20261101_h12"
#: Position-probe addendum (E0 addendum, 1 Nov 2026): three probe families.
POSITION_PROBES = ("linear", "mlp", "knn")
PROBE_K = 5
#: Code-origin addendum (2 Nov 2026): per-PMT fixed feature groups.
CODE_ORIGIN_GROUPS = ("position", "hit_rate", "charge", "parity", "module",
                      "raw_token")
CODE_ORIGIN_METHODS = ("linear", "mlp", "knn")
#: §6 thresholds (full_input primary, half_pass required secondary).
INIT_FULL_VERTEX_M = 0.160072          # the named random-init floor (rebuilt seed 9000)
INIT_HALF_VERTEX_M = 0.174821          # Part C recorded init9000 half-pass value
CHARGE_R2_MIN = 0.5
#: §4.2 reproduction (Q1/Q2 readout code re-evaluated).
REPRODUCTION = {"fix9000": 0.219258, "init9000": 0.160072}
REPRODUCTION_TOLERANCE_M = 1e-3
#: §3.4 read-only data keys and recorded manifest sha256 values.
DATA_ROOT_DEFAULT = "/ceph/dwong/jepa/data/ext_pregrid_dev_20261025"
DATA_MANIFESTS = {
    "dev_train__F1__0.05__s20261017__n100000x1000":
        "b1b3f4f401bcee48c49c347b67d7f5cbd4410213f08ef8b9d2848edc9f2cd13a",
    "dev_probe_train__F1__0.05__s20261017":
        "093ea8949b6f079841f389775bfab771b0e2d9d9cc20f4dff707bea1532ecfd3",
    "dev_probe_test__F1__0.05__s20261017":
        "c11edc711833dddc9520cb69435608ad4654f66ef80bcf49810d32fa667a8c5a",
}
CONFIG_PATH = "/ceph/dwong/jepa/code/"  # overridden by the CLI (repo-relative path)


# ---------------------------------------------------------------------- data
def load_probes(config, gen, geometry, data_root):
    """The two dev probe sets only (no ``dev_train`` build; read-only §3.4)."""
    cell = lr.dev_cell(config)
    common = dict(data_root=Path(data_root), geometry=geometry, generator_config=gen,
                  seed=int(config["data"]["dataset_seed"]), code_sha=None, log=print)
    return {role: lr.ensure_dataset(cohort_spec=config["data"]["cohorts"][name],
                                    cohort_name=name, law="F1", setting=cell.setting,
                                    registered=True, **common)
            for role, name in (("train", "dev_probe_train"), ("test", "dev_probe_test"))}


def datasets_for(cell, config, gen, geometry, data_root, code_sha=None):
    """``dev_train`` + the two probe sets (the pregrid helper's exact recipe)."""
    seed = int(config["data"]["dataset_seed"])
    common = dict(data_root=Path(data_root), geometry=geometry, generator_config=gen,
                  seed=seed, code_sha=code_sha, log=print)
    train = lr.ensure_dataset(cohort_spec=config["data"]["cohorts"][cell.train_cohort],
                              cohort_name=cell.train_cohort, law="F1", setting=cell.setting,
                              registered=False, **common)
    probes = {role: lr.ensure_dataset(cohort_spec=config["data"]["cohorts"][name],
                                      cohort_name=name, law="F1", setting=cell.setting,
                                      registered=True, **common)
              for role, name in (("train", "dev_probe_train"), ("test", "dev_probe_test"))}
    return train, probes


def preflight_data(data_root) -> dict:
    """§3.4 read-only preflight of the three development dataset keys."""
    root = Path(data_root) / "lucid"
    out = {"registered": False, "extension": EXTENSION, "data_root": str(data_root),
           "check": "§3.4 read-only development datasets", "sets": {}, "regenerated": False}
    for key, expected in DATA_MANIFESTS.items():
        directory = root / key
        manifest = directory / "manifest.json"
        entry = {"directory": str(directory), "key": key, "expected_sha256": expected,
                 "exists": directory.is_dir(), "exists_manifest": manifest.is_file()}
        if manifest.is_file():
            entry["manifest_sha256"] = lr.sha256_file(manifest)
            entry["pass"] = entry["manifest_sha256"] == expected
        else:
            entry["manifest_sha256"] = None
            entry["pass"] = False
        out["sets"][key] = entry
    out["pass"] = all(entry["pass"] for entry in out["sets"].values())
    out["regeneration"] = "forbidden (§3.4: stop and report if absent/different)"
    return out


# --------------------------------------------------------------- E2/E3 hooks
def attach_frozen_target(arm, geometry):
    """E3: freeze a seeded random second encoder as the prediction label source.

    The frozen encoder is built under ``torch.manual_seed(0)`` with the RNG state
    saved and restored, held with ``object.__setattr__`` so it is **outside** the
    module tree (``arm.state_dict()`` is unchanged), moved with the arm via an
    ``_apply`` override, and used by a rebound ``_encode_views`` for the label of
    both loss terms (prediction MSE and the SIGReg target term).
    """
    state = torch.get_rng_state()
    try:
        torch.manual_seed(0)
        frozen = pe.LucidPointJepa(arm.jepa.config, geometry, arm.jepa.scaling)
    finally:
        torch.set_rng_state(state)
    frozen.eval()
    frozen.requires_grad_(False)
    object.__setattr__(arm, "_ext_frozen_target", frozen)
    original_apply = arm._apply

    def _apply(fn):
        frozen._apply(fn)
        return original_apply(fn)

    object.__setattr__(arm, "_apply", _apply)

    def _encode_views(batch, need_full=True):
        hidden_index = arm._hidden_index(batch)
        context = arm.jepa.encode(batch.features, hidden_mask=batch.hidden)
        prediction = arm.jepa.predict(context, batch.context_index, hidden_index)
        if not need_full:
            return context, prediction, None, hidden_index, None
        full = arm.jepa.encode(batch.features)
        with torch.no_grad():
            label_full = frozen.encode(batch.features, hidden_mask=~batch.target)
        gather = hidden_index.unsqueeze(-1).expand(-1, -1, arm.d_model)
        labels = label_full.gather(1, gather).detach()
        return context, prediction, labels, hidden_index, full

    object.__setattr__(arm, "_encode_views", _encode_views)
    return arm


def apply_variant(arm, variant, geometry):
    """Install the E2 or E3 runtime hook on a freshly built arm."""
    if variant == "geomblind":
        original = er.install_geometry_blind(arm.jepa)
        object.__setattr__(arm, "_ext_geometry_blind", original)
        return arm
    if variant == "frozen_target":
        return attach_frozen_target(arm, geometry)
    raise ValueError(f"unknown H1/H2 variant {variant!r}; expected {VARIANTS}")


@contextlib.contextmanager
def build_arm_patch(variant, geometry, log=print):
    """Process-level wrapper on ``lucid_run.build_arm`` that applies the variant."""
    original = lr.build_arm

    def build(arm_name, geometry_, arms_config, cell, seed, beta=None, dark_bank=None,
              encoder_config=None):
        made = original(arm_name, geometry_, arms_config, cell, seed, beta=beta,
                        dark_bank=dark_bank, encoder_config=encoder_config)
        made = apply_variant(made, variant, geometry_)
        log(f"[h12] build_arm patch: variant={variant} run={lr.run_id_for(cell.cell_id, arm_name, seed)}")
        return made

    lr.build_arm = build
    try:
        yield
    finally:
        lr.build_arm = original


# ------------------------------------------------------------------ run dirs
def run_id_for(seed) -> str:
    return lr.run_id_for(CELL_ID, ARM_NAME, int(seed))


def run_dir_for(out_root, seed) -> Path:
    return Path(out_root) / "runs" / run_id_for(seed)


def prepare_run_dir(out_root, variant, seed, *, extra=None) -> tuple:
    """Refuse an existing run directory (§4.5) and write the variant sidecar."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    run_id, run_dir = run_id_for(seed), run_dir_for(out_root, seed)
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} already exists; refusing to launch (run_one would "
                           f"auto-resume it). Archive or remove it first (§4.5).")
    run_dir.mkdir(parents=True, exist_ok=False)
    sidecar = {"registered": False, "extension": EXTENSION, "variant": variant,
               "seed": int(seed), "arm": ARM_NAME, "cell_id": CELL_ID,
               "out_root": str(out_root), "run_id": run_id}
    sidecar.update(extra or {})
    (run_dir / "variant.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return run_id, run_dir


# ------------------------------------------------------------------- training
def train_variant(*, variant, seed, out_root, config, gen, arms_config, geometry, device,
                  data_root, config_path, code_sha, status_text, steps=None,
                  probe_max_events=None, log=print) -> dict:
    """One approved run (or the 20-step smoke) through the registered ``run_one``."""
    cell = lr.dev_cell(config)
    train_ds, probes = datasets_for(cell, config, gen, geometry, Path(data_root),
                                    code_sha=None)
    scheduled = int(config["training"]["steps"] if steps is None else steps)
    run_id, run_dir = prepare_run_dir(out_root, variant, seed,
                                     extra={"steps": scheduled, "device": str(device)})
    started = time.time()
    try:
        with build_arm_patch(variant, geometry, log=log):
            result = lr.run_one(
                arm_name=ARM_NAME, cell=cell, seed=int(seed), beta=None, config=config,
                arms_config=arms_config, geometry=geometry, loader_dataset=train_ds,
                probe_datasets=probes, dark_bank=None, out_root=Path(out_root),
                data_root=Path(data_root), config_sha=lr.sha256_file(config_path),
                code_sha=code_sha, status_text=status_text, device=device,
                steps=scheduled, micro_batch=None, resume=None, stop_after=None,
                checkpoint_minutes=None, evaluate=False, allow_pending=False,
                step_callback=None, probe_max_events=probe_max_events,
                save_embeddings=False, physics_method="ridge", tf32=False, log=log)
    except Exception:
        (run_dir / "variant.json").write_text(json.dumps(
            {"registered": False, "extension": EXTENSION, "variant": variant,
             "seed": int(seed), "failed": True}, indent=1) + "\n")
        raise
    summary = summarize_run(run_dir, variant=variant, seed=seed,
                            wall_seconds=round(time.time() - started, 3))
    (run_dir / "ext_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    return {"result": result, "run_dir": run_dir, "summary": summary, "variant": variant,
            "seed": int(seed)}


def summarize_run(run_dir: Path, *, variant: str, seed: int, wall_seconds=None,
                  forms: dict | None = None) -> dict:
    """Finite ``ext_summary.json`` payload for a smoke or an approved run."""
    run_dir = Path(run_dir)
    log_path = run_dir / "train_log.jsonl"
    rows = ([json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
            if log_path.is_file() else [])
    final = rows[-1] if rows else {}
    # train_log rows are {"step": .., "elapsed_seconds": .., "metrics": {loss, prediction, ...}}
    metrics = dict(final.get("metrics") or {})
    metrics.setdefault("step", final.get("step"))
    metrics.setdefault("elapsed_seconds", final.get("elapsed_seconds"))
    finite = all(math.isfinite(float(value)) for value in metrics.values()
                 if isinstance(value, (int, float)))
    checkpoint = run_dir / "final.pt"
    status = json.loads((run_dir / "run_status.json").read_text()) \
        if (run_dir / "run_status.json").is_file() else {}
    summary = {"registered": False, "extension": EXTENSION, "variant": variant,
               "seed": int(seed), "run_id": run_dir.name, "arm": ARM_NAME,
               "cell_id": CELL_ID, "split": "within",
               "final_train_metrics": final, "metrics_finite": bool(finite),
               "n_log_rows": len(rows), "wall_seconds": wall_seconds,
               "status": status.get("outcome"), "steps_completed": status.get("steps_completed"),
               "final_checkpoint_sha256": (lr.sha256_file(checkpoint)
                                           if checkpoint.is_file() else None),
               "forms": forms}
    return summary


def train_log_check(run_dir: Path) -> dict:
    """Scan ``train_log.jsonl``: row count, final metrics and a real NaN check."""
    run_dir = Path(run_dir)
    path = run_dir / "train_log.jsonl"
    if not path.is_file():
        return {"status": "missing", "n_rows": 0, "final_metrics": None,
                "non_finite_keys": [], "finite": None}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    final = (rows[-1].get("metrics") or {}) if rows else {}
    bad = sorted(key for key, value in final.items()
                 if isinstance(value, (int, float)) and not math.isfinite(float(value)))
    return {"status": "checked", "n_rows": len(rows), "final_metrics": final,
            "non_finite_keys": bad, "finite": not bad}




# ------------------------------------------ code-origin addendum (2 Nov 2026)
def code_origin_feature_groups(features: dict, geometry, present) -> dict:
    """Per-PMT fixed feature groups for the ``present`` PMTs (sorted order).

    ``features`` carries per-PMT accumulators over the same probe-test events as the
    label rows: ``hit_count`` (n_pmt,), ``charge_sum`` (n_pmt, mean ln q sums),
    ``raw_sum`` (n_pmt, 9 raw-token feature sums over hit occurrences), ``n_events``
    and the scaled ``positions`` (n_pmt, 3).  Groups: scaled position (3), hit rate
    (1), mean hit ln q (1), within-target parity (1), module one-hot (n_modules) and
    the mean raw-token feature vector (9); ``all`` concatenates them.
    """
    present = np.asarray(present, np.int64)
    n_events = max(int(features["n_events"]), 1)
    counts = np.asarray(features["hit_count"], float)[present]
    charge = np.asarray(features["charge_sum"], float)[present]
    raw = np.asarray(features["raw_sum"], float)[present]
    counts_safe = np.maximum(counts, 1.0)[:, None]
    pmt_id = np.asarray(geometry.pmt_id_in_mpmt)[present]
    module = np.asarray(geometry.module_index)[present]
    n_modules = int(geometry.n_modules)
    one_hot = np.zeros((len(present), n_modules))
    one_hot[np.arange(len(present)), module] = 1.0
    groups = {
        "position": np.asarray(features["positions"], float)[present],
        "hit_rate": (counts / n_events)[:, None],
        "charge": (charge / np.maximum(counts, 1.0))[:, None],
        "parity": ((pmt_id % 2) != 0).astype(float)[:, None],
        "module": one_hot,
        "raw_token": raw / counts_safe,
    }
    groups["all"] = np.concatenate([groups[name] for name in CODE_ORIGIN_GROUPS], axis=1)
    return groups


def code_origin_feature_names(features: dict, geometry) -> dict:
    """Names/dimensions of the feature groups (for the JSON)."""
    raw_names = list(features.get("raw_feature_names") or
                     ("pos_x", "pos_y", "pos_z", "dir_x", "dir_y", "dir_z",
                      "log_charge", "time_100ns", "hit"))
    return {"position": ["pos_x", "pos_y", "pos_z"],
            "hit_rate": ["hit_rate"],
            "charge": ["mean_hit_log_charge"],
            "parity": ["within_target_parity (pmt_id_in_mpmt % 2)"],
            "module": f"one-hot module id ({int(geometry.n_modules)} modules)",
            "raw_token": raw_names,
            "all": "concatenation of position, hit_rate, charge, parity, module, raw_token"}


def collect_per_pmt_features(config, gen, geometry, data_root, *,
                             max_label_events: int = 2000, chunk_events: int = 8,
                             device="cpu", log=print) -> dict:
    """Per-PMT fixed features over the same probe-test events as the label rows.

    Read-only data pass (tokenisation only, no encoder): per-PMT hit counts, summed
    hit ln q and summed raw-token features, and the number of probe events used.
    """
    device = device if isinstance(device, torch.device) else torch.device(device)
    cell = lr.dev_cell(config)
    scaling = pe.FeatureScaling.from_geometry(geometry)
    _, probes = None, load_probes(config, gen, geometry, data_root)
    test = probes["test"]
    n_pmt = int(geometry.n_pmt)
    n_features = int(pe.N_FEATURES)
    hit_count = np.zeros(n_pmt, np.float64)
    charge_sum = np.zeros(n_pmt, np.float64)
    raw_sum = np.zeros((n_pmt, n_features), np.float64)
    n_events = 0
    for row in test.runs:
        if n_events >= int(max_label_events):
            break
        source = test.directory / "runs" / row["run_dir"]
        ids = lr._run_event_ids(source)
        allowed = lr._domain_event_ids(source, er.PRIMARY_DOMAIN)
        if allowed is not None:
            ids = np.intersect1d(ids, allowed)
        if ids.size == 0:
            continue
        ids = ids[: int(max_label_events) - n_events]
        for start in range(0, int(ids.size), int(chunk_events)):
            batch = lr._probe_batch(source, geometry, split=cell.split,
                                    mask_seed=int(config["views"]["mask_seed"]),
                                    occupancy="natural", scaling=scaling,
                                    event_ids=ids[start:start + int(chunk_events)])
            if batch.event_ids.size == 0:
                continue
            hit = np.asarray(batch.hit_flags, bool)
            log_charge = np.asarray(batch.log_charge, float)
            raw = batch.features.detach().to(torch.float32).cpu().numpy()
            event_index, pmt_index = np.nonzero(hit)
            np.add.at(hit_count, pmt_index, 1.0)
            np.add.at(charge_sum, pmt_index, log_charge[event_index, pmt_index])
            np.add.at(raw_sum, pmt_index, raw[event_index, pmt_index])
            n_events += int(batch.event_ids.size)
            log(f"[code-origin] {n_events} probe events, {int(hit_count.sum())} hit rows")
    return {"hit_count": hit_count, "charge_sum": charge_sum, "raw_sum": raw_sum,
            "n_events": int(n_events),
            "positions": pe.geometry_features(geometry, scaling)[:, :3].astype(np.float64),
            "raw_feature_names": list(pe.FEATURES)}


def code_origin_probes(rows: dict, features: dict, geometry,
                       probes=CODE_ORIGIN_METHODS, *, seed: int = 0, log=print) -> dict:
    """Fit the invariant component ``m_p`` from the per-PMT fixed feature groups.

    ``m_p`` = mean label over the rows of each PMT (the E1 invariant component);
    80/20 split by PMT (the position-probe protocol, seed 0); linear / MLP (2 x 256,
    early stopping on 10 % of the training PMTs) / kNN (k = 5) per group and for the
    combined feature set, for ``fix9000`` and the rebuilt ``init9000`` reference.
    """
    probes = tuple(probes)
    for probe in probes:
        if probe not in CODE_ORIGIN_METHODS:
            raise ValueError(f"unknown code-origin probe {probe!r}; expected "
                             f"{CODE_ORIGIN_METHODS}")
    started = time.time()
    present, order, cut, is_train = _pmt_split(rows["pmts"], rows["n_pmt"], seed=seed)
    _, m_fix, n_present = _group_rows_by_pmt(np.asarray(rows["positions"], float),
                                             np.asarray(rows["y"], float),
                                             np.asarray(rows["pmts"], np.int64),
                                             int(rows["n_pmt"]))
    _, m_init, _ = _group_rows_by_pmt(np.asarray(rows["positions"], float),
                                      np.asarray(rows["init_y"], float),
                                      np.asarray(rows["pmts"], np.int64),
                                      int(rows["n_pmt"]))
    present_ids = np.flatnonzero(np.bincount(np.asarray(rows["pmts"], np.int64),
                                             minlength=int(rows["n_pmt"])) > 0)
    if not np.array_equal(present, present_ids):  # both sorted: defensive
        raise RuntimeError("code-origin: PMT order mismatch")
    groups = code_origin_feature_groups(features, geometry, present)
    validation = max(1, int(round(0.1 * cut)))
    train_pmts = order[:cut]
    fit_mask = np.isin(present, train_pmts[:cut - validation])
    val_mask = np.isin(present, train_pmts[cut - validation:])
    test_mask = ~is_train
    block = {"registered": False, "extension": EXTENSION,
             "check": "code-origin probes for the event-invariant per-PMT component m_p "
                      "(0 training, 2 Nov 2026)",
             "sampling": dict(rows["sampling"]),
             "fix9000": rows.get("fix9000", {}),
             "init9000": {"rebuilt": True, "seed": 9000},
             "features": {"groups": code_origin_feature_names(features, geometry),
                          "n_features_all": int(groups["all"].shape[1])},
             "split": {"by": "PMT", "seed": int(seed), "n_pmt": int(n_present),
                       "n_pmt_train": int(is_train.sum()),
                       "n_pmt_test": int(test_mask.sum()),
                       "n_val_pmt": int(val_mask.sum()),
                       "protocol": f"80/20 by PMT (same split as the position-probe "
                                   f"addendum); MLP early-stops on 10 % of the training "
                                   f"PMTs; kNN k = {PROBE_K}"},
             "probes": list(probes)}
    results = {"fix9000": {}, "init9000": {}}
    for label, m_labels in (("fix9000", m_fix), ("init9000", m_init)):
        for name in CODE_ORIGIN_GROUPS + ("all",):
            values = groups[name]
            entry = {}
            if "linear" in probes:
                fit = _multioutput_ridge(values[is_train], m_labels[is_train],
                                         values[test_mask], m_labels[test_mask])
                entry["linear"] = _record(fit)
            if "mlp" in probes:
                prediction, epochs_run = _mlp_fit(values[fit_mask], m_labels[fit_mask],
                                                  values[val_mask], m_labels[val_mask],
                                                  values[test_mask], seed=seed)
                record = _r2_mean(m_labels[test_mask], prediction)
                record.update(epochs_run=epochs_run, n_train=int(fit_mask.sum()),
                              n_val=int(val_mask.sum()), n_test=int(test_mask.sum()))
                entry["mlp"] = _record(record, n_train=record["n_train"],
                                       n_test=record["n_test"])
                entry["mlp"]["n_val"] = record["n_val"]
            if "knn" in probes:
                prediction = _knn_predict(values[is_train], m_labels[is_train],
                                          values[test_mask], k=PROBE_K)
                record = _r2_mean(m_labels[test_mask], prediction)
                entry["knn5"] = _record(record, n_train=int(is_train.sum()),
                                        n_test=int(test_mask.sum()))
                entry["knn5"]["k"] = PROBE_K
            results[label][name] = entry
            log(f"[code-origin] {label} {name}: "
                f"{ {method: round(cell['r2'], 4) if cell.get('r2') is not None else None for method, cell in entry.items()} }")
    block["probes_by_label"] = results
    block["wall_seconds"] = round(time.time() - started, 3)
    return block

# ------------------------------------------- position-probe addendum (1 Nov 2026)
def _r2_mean(y_test, prediction) -> dict:
    """Mean per-dimension R² (undefined dimensions are dropped, never set to 0)."""
    y_test = np.asarray(y_test, float)
    prediction = np.asarray(prediction, float)
    if y_test.ndim == 1:
        y_test = y_test[:, None]
    per_dim = []
    for column in range(y_test.shape[1]):
        sst = float(((y_test[:, column] - y_test[:, column].mean()) ** 2).sum())
        per_dim.append(float(1 - ((y_test[:, column] - prediction[:, column]) ** 2).sum()
                             / sst) if sst > 0 else None)
    finite = [value for value in per_dim if value is not None]
    return {"r2": float(np.mean(finite)) if finite else None, "r2_per_dim": per_dim}


def _record(result: dict, *, n_train=None, n_test=None) -> dict:
    """Small side-by-side record for one probe fit."""
    record = {"r2": result.get("r2"), "r2_per_dim": result.get("r2_per_dim"),
              "n_train": int(n_train if n_train is not None else result.get("n_train", 0)),
              "n_test": int(n_test if n_test is not None else result.get("n_test", 0))}
    if "alpha" in result:
        record["alpha"] = result.get("alpha")
    if "epochs_run" in result:
        record["epochs_run"] = result.get("epochs_run")
    return record


def _mlp_fit(x_train, y_train, x_val, y_val, x_test, *, seed: int = 0, hidden: int = 256,
             epochs: int = 200, patience: int = 10, batch: int = 4096) -> tuple:
    """2 x 256 MLP on standardised inputs with early stopping on ``(x_val, y_val)``.

    Same architecture / patience / seed / input standardisation as the Q1/Q2
    ``mlp_probe``; separate validation rows because the addendum's early-stopping
    split is by event (row-level) or by PMT (invariant component).  Returns
    ``(predictions on x_test, epochs_run)``.
    """
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, float)
    if y_train.ndim == 1:
        y_train = y_train[:, None]
    centre, scale = x_train.mean(0), x_train.std(0)
    scale[scale == 0] = 1.0
    y_centre = y_train.mean(0, keepdims=True)
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(x_train.shape[1], hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, hidden), torch.nn.GELU(),
        torch.nn.Linear(hidden, y_train.shape[1]))
    xt = torch.as_tensor((x_train - centre) / scale, dtype=torch.float32)
    yt = torch.as_tensor(y_train - y_centre, dtype=torch.float32)
    xv = torch.as_tensor((np.asarray(x_val, float) - centre) / scale, dtype=torch.float32)
    yv = torch.as_tensor(np.asarray(y_val, float) - y_centre, dtype=torch.float32)
    xtest = torch.as_tensor((np.asarray(x_test, float) - centre) / scale,
                            dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.functional.mse_loss
    generator = torch.Generator().manual_seed(int(seed) + 1)
    best, state, bad, epochs_run = math.inf, None, 0, 0
    for epoch in range(int(epochs)):
        epochs_run = epoch + 1
        model.train()
        order = torch.randperm(len(xt), generator=generator)
        for start in range(0, len(xt), int(batch)):
            index = order[start:start + int(batch)]
            optimizer.zero_grad()
            loss_fn(model(xt[index]), yt[index]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = float(loss_fn(model(xv), yv))
        if validation < best:
            best, state, bad = validation, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= int(patience):
                break
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        prediction = model(xtest).numpy() + y_centre
    return prediction, epochs_run


def _knn_predict(x_train, y_train, x_test, *, k: int = PROBE_K) -> np.ndarray:
    """kNN (Euclidean on standardised positions) mean of the k nearest train PMTs."""
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, float)
    x_test = np.asarray(x_test, float)
    centre, scale = x_train.mean(0), x_train.std(0)
    scale[scale == 0] = 1.0
    a = (x_train - centre) / scale
    b = (x_test - centre) / scale
    distance = ((b[:, None, :] - a[None, :, :]) ** 2).sum(axis=-1)
    take = min(int(k), len(a))
    index = np.argsort(distance, axis=1, kind="stable")[:, :take]
    return y_train[index].mean(axis=1)


def _event_split(events, fraction: float = 0.8) -> tuple:
    """Deterministic 80/20 split by event id (the E0-c split, sorted-event order)."""
    unique = np.unique(np.asarray(events, np.int64))
    cut = int(float(fraction) * len(unique))
    train_events = set(unique[:cut].tolist())
    is_train = np.array([int(value) in train_events for value in events], bool)
    return unique, cut, is_train


def _pmt_split(pmts, n_pmt: int, *, seed: int = 0, fraction: float = 0.8) -> tuple:
    """Deterministic 80/20 split by PMT index (seeded permutation of the present PMTs)."""
    present = np.flatnonzero(np.bincount(np.asarray(pmts, np.int64), minlength=n_pmt) > 0)
    rng = np.random.default_rng(int(seed))
    order = present[rng.permutation(len(present))]
    cut = int(float(fraction) * len(order))
    is_train = np.isin(present, order[:cut])
    return present, order, cut, is_train


def position_probes(rows: dict, probes=("mlp",), *, seed: int = 0, log=print) -> dict:
    """Position -> label probes answering whether the invariant component is a
    (possibly nonlinear) function of the PMT position.

    Three probe families are reported side by side when requested:
    ``linear`` (always; the E0-c and E1 protocol), ``mlp`` (2 x 256, early stopping
    on 10 % of the training events for the row-level probe and 10 % of the training
    PMTs for the per-PMT probe) and ``knn`` (k = 5, 80/20 split by PMT).  The
    per-PMT target is ``m_p`` = mean label over the rows of PMT ``p`` (the E1
    invariant component); the row-level target is the label vector itself.
    """
    probes = tuple(probes)
    for probe in probes:
        if probe not in POSITION_PROBES:
            raise ValueError(f"unknown position probe {probe!r}; expected {POSITION_PROBES}")
    started = time.time()
    x = np.asarray(rows["x"], float)
    y = np.asarray(rows["y"], float)
    init_y = np.asarray(rows["init_y"], float)
    pmts = np.asarray(rows["pmts"], np.int64)
    events = np.asarray(rows["events"], np.int64)
    positions = np.asarray(rows["positions"], float)
    n_pmt = int(rows["n_pmt"])
    block = {"registered": False, "extension": EXTENSION,
             "check": "position probe addendum to E0/E1 (0 training, 1 Nov 2026)",
             "sampling": dict(rows["sampling"]),
             "fix9000": rows.get("fix9000", {}),
             "init9000": {"rebuilt": True, "seed": 9000},
             "probes": list(probes)}
    # (a) row-level position -> label, 80/20 by event (the E0-c split) + MLP.
    unique, cut, is_train = _event_split(events)
    row = {"split": "80/20 by event (same split as E0-c); MLP early-stops on the "
                    "last 10 % of the training events",
           "n_events": int(len(unique)), "n_events_train": int(cut),
           "n_events_test": int(len(unique) - cut)}
    linear = _multioutput_ridge(x[is_train], y[is_train], x[~is_train], y[~is_train])
    row["linear"] = _record(linear)
    if "mlp" in probes:
        train_events = unique[:cut]
        validation = max(1, int(round(0.1 * cut)))
        fit_events = set(train_events[:cut - validation].tolist())
        val_events = set(train_events[cut - validation:].tolist())
        fit_mask = np.isin(events, list(fit_events))
        val_mask = np.isin(events, list(val_events))
        prediction, epochs_run = _mlp_fit(x[fit_mask], y[fit_mask], x[val_mask], y[val_mask],
                                          x[~is_train], seed=seed)
        record = _r2_mean(y[~is_train], prediction)
        record.update(epochs_run=epochs_run, n_train=int(fit_mask.sum()),
                      n_val=int(val_mask.sum()), n_test=int((~is_train).sum()))
        row["mlp"] = _record(record, n_train=record["n_train"], n_test=record["n_test"])
        row["mlp"]["n_val"] = record["n_val"]
        log(f"[position-probe] row-level mlp r2={row['mlp']['r2']} "
            f"(linear {row['linear']['r2']})")
    block["row_level"] = row
    # (b)/(c) per-PMT invariant component m_p: fix9000 and the init9000 reference.
    _, m_fix, n_present = _group_rows_by_pmt(positions, y, pmts, n_pmt)
    _, m_init, _ = _group_rows_by_pmt(positions, init_y, pmts, n_pmt)
    present, order, pmt_cut, is_train_pmt = _pmt_split(pmts, n_pmt, seed=seed)
    block["per_pmt"] = {
        "split": "80/20 by PMT (seeded permutation, seed %d); the invariant component "
                 "m_p = mean label over the rows of each PMT (E1)" % int(seed),
        "n_pmt": int(n_present), "n_pmt_train": int(is_train_pmt.sum()),
        "n_pmt_test": int((~is_train_pmt).sum())}
    for name, m_labels in (("fix9000", m_fix), ("init9000", m_init)):
        entry = {}
        linear = _multioutput_ridge(positions[present][is_train_pmt],
                                    m_labels[is_train_pmt],
                                    positions[present][~is_train_pmt],
                                    m_labels[~is_train_pmt])
        entry["linear"] = _record(linear)
        if "mlp" in probes:
            train_pmts = order[:pmt_cut]
            validation = max(1, int(round(0.1 * pmt_cut)))
            fit_mask = np.isin(present, train_pmts[:pmt_cut - validation])
            val_mask = np.isin(present, train_pmts[pmt_cut - validation:])
            prediction, epochs_run = _mlp_fit(positions[present][fit_mask],
                                              m_labels[fit_mask],
                                              positions[present][val_mask],
                                              m_labels[val_mask],
                                              positions[present][~is_train_pmt], seed=seed)
            record = _r2_mean(m_labels[~is_train_pmt], prediction)
            record.update(epochs_run=epochs_run, n_train=int(fit_mask.sum()),
                          n_val=int(val_mask.sum()), n_test=int((~is_train_pmt).sum()))
            entry["mlp"] = _record(record, n_train=record["n_train"],
                                   n_test=record["n_test"])
            entry["mlp"]["n_val"] = record["n_val"]
        if "knn" in probes:
            prediction = _knn_predict(positions[present][is_train_pmt],
                                      m_labels[is_train_pmt],
                                      positions[present][~is_train_pmt], k=PROBE_K)
            record = _r2_mean(m_labels[~is_train_pmt], prediction)
            entry["knn5"] = _record(record, n_train=int(is_train_pmt.sum()),
                                    n_test=int((~is_train_pmt).sum()))
            entry["knn5"]["k"] = PROBE_K
        block["per_pmt"][name] = entry
        log(f"[position-probe] per-PMT {name}: linear r2={entry['linear']['r2']} "
            f"mlp={((entry.get('mlp') or {}).get('r2'))} "
            f"knn5={((entry.get('knn5') or {}).get('r2'))}")
    block["wall_seconds"] = round(time.time() - started, 3)
    return block


def _merge(dst: dict, src: dict) -> dict:
    """Recursive new-wins merge (keeps earlier probe blocks in the side-by-side file)."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value
    return dst


def merge_position_payload(path, block: dict) -> dict:
    """Write/merge the addendum JSON so the mlp and knn runs end up side by side."""
    path = Path(path)
    payload = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text())
        except (ValueError, OSError):
            payload = {}
    _merge(payload, block)
    payload.setdefault("registered", False)
    payload.setdefault("extension", EXTENSION)
    row = payload.get("row_level") or {}
    per_pmt = payload.get("per_pmt") or {}
    probes = {"linear": True, "mlp": any("mlp" in cell for cell in
                                        (row, per_pmt.get("fix9000") or {},
                                         per_pmt.get("init9000") or {})),
              "knn": any("knn5" in cell for cell in
                         (row, per_pmt.get("fix9000") or {},
                          per_pmt.get("init9000") or {}))}
    payload["probes_present"] = sorted(name for name, present in probes.items() if present)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(lr._safe(payload), indent=1, default=float) + "\n")
    os.replace(tmp, path)
    return payload


# ---------------------------------------------------------------- E0/E1 (Phase 0)
def _multioutput_ridge(x_train, y_train, x_test, y_test, alphas=er.RIDGE_ALPHAS) -> dict:
    """Standardised multi-output ridge; alpha on the last 20 % of train, R² per dim."""
    x_train = np.asarray(x_train, float)
    y_train = np.asarray(y_train, float)
    x_test, y_test = np.asarray(x_test, float), np.asarray(y_test, float)
    centre, scale = x_train.mean(0), x_train.std(0)
    scale[scale == 0] = 1.0
    z_train, z_test = (x_train - centre) / scale, (x_test - centre) / scale
    cut = max(1, int(0.8 * len(z_train)))

    def fit(z, y, alpha):
        gram = z.T @ z + alpha * np.eye(z.shape[1])
        coefficients = np.linalg.solve(gram, z.T @ (y - y.mean(0)))
        return coefficients, y.mean(0)

    best = None
    for alpha in alphas:
        coefficients, offset = fit(z_train[:cut], y_train[:cut], alpha)
        error = float(((z_train[cut:] @ coefficients + offset - y_train[cut:]) ** 2).mean())
        if best is None or error < best[1]:
            best = (alpha, error)
    coefficients, offset = fit(z_train, y_train, best[0])
    prediction = z_test @ coefficients + offset
    per_dim = []
    for column in range(y_test.shape[1]):
        sst = float(((y_test[:, column] - y_test[:, column].mean()) ** 2).sum())
        per_dim.append(float(1 - ((y_test[:, column] - prediction[:, column]) ** 2).sum() / sst)
                      if sst > 0 else None)
    finite = [value for value in per_dim if value is not None]
    return {"r2": float(np.mean(finite)) if finite else None, "alpha": best[0],
            "r2_per_dim": per_dim, "n_train": int(len(y_train)),
            "n_test": int(len(y_test))}


def _group_rows_by_pmt(positions, labels, pmts, n_pmt):
    """Per-PMT mean label rows (m_p) and their PMT indices."""
    means = np.zeros((n_pmt, labels.shape[1]))
    counts = np.bincount(np.asarray(pmts, np.int64), minlength=n_pmt)
    np.add.at(means, np.asarray(pmts, np.int64), labels)
    keep = counts > 0
    return positions[keep], means[keep] / counts[keep][:, None], int(keep.sum())


def _collect_label_rows(config, gen, arms_config, geometry, data_root, readout_items, *,
                        device, max_label_events: int = 2000, chunk_events: int = 8,
                        label_cache=None, log=print) -> dict:
    """Collect the E0/E1 label rows (cached under ``<ext>/e0e1_rows.npz`` if asked)."""
    items = json.loads(Path(readout_items).read_text())["items"]
    fix = dict(items["fix9000"])
    fix["source"] = fix.get("ckpt") or fix["source"]
    if fix.get("ckpt") and lr.sha256_file(Path(fix["ckpt"])) != fix["ckpt_sha256"]:
        raise RuntimeError("E0/E1: the verified fix9000 checkpoint copy has changed")
    fix_sha = fix.get("ckpt_sha256") or fix.get("source_sha256")
    if label_cache is not None:
        cached = _read_label_cache(Path(label_cache), fix_sha=fix_sha,
                                   max_label_events=int(max_label_events))
        if cached is not None:
            log(f"[e0e1] reusing cached label rows {label_cache}")
            return cached
    fix_arm = er.item_arm(fix, config, arms_config, geometry, device=device)
    init_arm = er.item_arm({"label": "init9000", "arm": ARM_NAME, "split": "within",
                            "seed": 9000, "beta": None, "source": None},
                           config, arms_config, geometry, device=device)
    _, probes = None, load_probes(config, gen, geometry, data_root)
    test = probes["test"]
    cell = lr.dev_cell(config)
    mask_seed = int(config["views"]["mask_seed"])
    scaling = pe.FeatureScaling.from_geometry(geometry)
    positions = pe.geometry_features(geometry, scaling)[:, :3].astype(np.float64)
    layout = er.layout_state(geometry.n_pmt, 0)
    rng = np.random.default_rng(er.SAMPLE_SEED)
    x_rows, y_rows, init_rows, pmt_rows, event_rows = [], [], [], [], []
    n_events = 0
    event_offset = 0
    for row in test.runs:
        if n_events >= int(max_label_events):
            break
        source = test.directory / "runs" / row["run_dir"]
        truth = lr._run_truth(row, test, geometry)
        ids = lr._run_event_ids(source)
        allowed = lr._domain_event_ids(source, er.PRIMARY_DOMAIN)
        if allowed is not None:
            ids = np.intersect1d(ids, allowed)
        if ids.size == 0:
            continue
        ids = ids[: int(max_label_events) - n_events]
        for start in range(0, int(ids.size), int(chunk_events)):
            batch = lr._probe_batch(source, geometry, split=cell.split, mask_seed=mask_seed,
                                    occupancy="natural", scaling=scaling,
                                    event_ids=ids[start:start + int(chunk_events)])
            if batch.event_ids.size == 0:
                continue
            batch = lr._to_device(batch, device)
            with torch.no_grad():
                tokens = fix_arm.jepa.encode(batch.features)
                fix_labels = fix_arm.jepa.encode(batch.features, hidden_mask=~batch.target)
                init_labels = init_arm.jepa.encode(batch.features, hidden_mask=~batch.target)
            tokens_np = tokens.detach().to(torch.float32).cpu().numpy()
            if layout["dim"] == 0:
                layout = er.layout_state(geometry.n_pmt, tokens_np.shape[-1])
            er.layout_add(layout, tokens_np)
            fix_np = fix_labels.detach().to(torch.float32).cpu().numpy()
            init_np = init_labels.detach().to(torch.float32).cpu().numpy()
            hidden = np.asarray(batch.hidden.detach().cpu(), bool)
            hit = np.asarray(batch.hit_flags, bool)
            candidate = hidden & hit
            for event_index in range(candidate.shape[0]):
                positions_hit = np.flatnonzero(candidate[event_index])
                if positions_hit.size == 0:
                    continue
                take = min(32, int(positions_hit.size))
                pick = (rng.choice(positions_hit, size=take, replace=False)
                        if positions_hit.size > take else positions_hit)
                for pmt in pick:
                    x_rows.append(positions[pmt])
                    y_rows.append(fix_np[event_index, pmt])
                    init_rows.append(init_np[event_index, pmt])
                    pmt_rows.append(int(pmt))
                    event_rows.append(int(event_offset))
            event_offset += int(batch.event_ids.size)
            n_events += int(batch.event_ids.size)
            log(f"[e0e1] {n_events} events, {len(y_rows)} label rows")
    del fix_arm, init_arm
    if not y_rows:
        raise RuntimeError("E0-c has no label rows; nothing could be sampled")
    rows = {"x": np.asarray(x_rows, np.float64), "y": np.asarray(y_rows, np.float64),
            "init_y": np.asarray(init_rows, np.float64),
            "pmts": np.asarray(pmt_rows, np.int64),
            "events": np.asarray(event_rows, np.int64),
            "positions": positions, "n_pmt": int(geometry.n_pmt), "n_events": int(n_events),
            "layout": layout,
            "fix9000": {"source": fix.get("source"),
                        "source_sha256": fix.get("source_sha256") or fix.get("ckpt_sha256"),
                        "ckpt_sha256": fix.get("ckpt_sha256")},
            "sampling": {"max_label_events": int(max_label_events),
                         "tokens_per_event": 32, "seed": er.SAMPLE_SEED}}
    if label_cache is not None:
        _write_label_cache(Path(label_cache), rows, fix_sha=fix_sha)
    return rows


def _read_label_cache(path: Path, *, fix_sha, max_label_events: int):
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("fix_sha256") != fix_sha or \
                    int(meta.get("max_label_events", -1)) != int(max_label_events):
                return None
            return {"x": np.asarray(data["x"], np.float64),
                    "y": np.asarray(data["y"], np.float64),
                    "init_y": np.asarray(data["init_y"], np.float64),
                    "pmts": np.asarray(data["pmts"], np.int64),
                    "events": np.asarray(data["events"], np.int64),
                    "positions": np.asarray(data["positions"], np.float64),
                    "n_pmt": int(data["n_pmt"]), "n_events": int(data["n_events"]),
                    "layout": None, "cache": str(path),
                    "sampling": meta.get("sampling") or {},
                    "fix9000": meta.get("fix9000") or {}}
    except (OSError, ValueError, KeyError):
        return None


def _write_label_cache(path: Path, rows: dict, *, fix_sha) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    meta = {"fix_sha256": fix_sha, "max_label_events": int(rows["sampling"]["max_label_events"]),
            "sampling": rows["sampling"], "fix9000": rows.get("fix9000", {})}
    with open(tmp, "wb") as handle:
        np.savez(handle, x=rows["x"], y=rows["y"], init_y=rows["init_y"],
                 pmts=rows["pmts"], events=rows["events"],
                 positions=rows["positions"], n_pmt=np.asarray(rows["n_pmt"]),
                 n_events=np.asarray(rows["n_events"]),
                 meta=np.asarray(json.dumps(meta)))
    os.replace(tmp, path)


def e0e1(config, gen, arms_config, geometry, data_root, readout_items, out_path=None, *,
         device, max_label_events: int = 2000, chunk_events: int = 8, log=print,
         position_probe: str = "linear", position_out=None, label_cache=None,
         code_origin: bool = False, code_origin_out=None,
         code_origin_methods=CODE_ORIGIN_METHODS) -> dict:
    """E0-a/b/c and E1 on the ``fix9000`` (and rebuilt ``init9000``) encoders.

    ``position_probe`` extends the mode with the 1 Nov 2026 position-probe addendum
    (``mlp``: row-level MLP and per-PMT MLP; ``knn``: per-PMT kNN); the default
    ``linear`` keeps the committed E0/E1 output unchanged.
    """
    if position_probe not in POSITION_PROBES:
        raise ValueError(f"unknown position probe {position_probe!r}; "
                         f"expected {POSITION_PROBES}")
    started = time.time()
    rows = _collect_label_rows(config, gen, arms_config, geometry, data_root,
                               readout_items, device=device,
                               max_label_events=int(max_label_events),
                               chunk_events=int(chunk_events),
                               label_cache=(label_cache if (position_probe != "linear"
                                                            or code_origin) else None),
                               log=log)
    fix = rows.get("fix9000") or {}
    layout = rows.get("layout")
    if layout is None:
        # cached rows do not carry the layout statistics; recompute E0-a/b from the
        # cached forward pass is not needed for the addendum, so mark them null.
        fractions = {"f_layout": None, "f_event": None, "n_dropped": None}
    else:
        fractions = er.layout_fractions(layout)
    x, y = rows["x"], rows["y"]
    init_y = rows["init_y"]
    pmts = rows["pmts"]
    event_index = rows["events"]
    positions = rows["positions"]
    geometry_n_pmt = int(rows["n_pmt"])
    n_events = int(rows["n_events"])
    # E0-c: 80/20 split by event, no event leaks across the split.
    unique_events = np.unique(event_index)
    cut = int(0.8 * len(unique_events))
    train_events = set(unique_events[:cut].tolist())
    is_train = np.array([int(value) in train_events for value in event_index])
    e0c = _multioutput_ridge(x[is_train], y[is_train], x[~is_train], y[~is_train])
    # E1: label decomposition on the same rows.
    var_rows = y.var(axis=0, ddof=1)
    init_var_rows = init_y.var(axis=0, ddof=1)
    m_positions, m_labels, n_pmt_rows = _group_rows_by_pmt(positions, y, pmts,
                                                           geometry.n_pmt)
    var_m = m_labels.var(axis=0, ddof=1)
    keep = var_rows > 0
    invariant = float(np.mean(var_m[keep] / var_rows[keep])) if keep.any() else None
    position_fit = _multioutput_ridge(m_positions[:, :3], m_labels, m_positions[:, :3],
                                      m_labels)
    label_variance = float(np.mean(var_rows))
    init_label_variance = float(np.mean(init_var_rows))
    ratio = (label_variance / init_label_variance) if init_label_variance > 0 else None
    e0 = {"f_layout": fractions.get("f_layout"), "f_event": fractions.get("f_event"),
          "n_dropped": fractions.get("n_dropped"),
          "n_events_token_sample": int(fractions.get("n_events") or 0),
          "n_tokens": int(fractions.get("n_tokens") or 0),
          "f_layout_per_dim": fractions.get("f_layout_per_dim"),
          "f_event_per_dim": fractions.get("f_event_per_dim"),
          "position_only_label_ridge": e0c}
    e1 = {"invariant_fraction": invariant, "position_explained_r2": position_fit["r2"],
          "position_explained_alpha": position_fit["alpha"],
          "label_variance": label_variance, "init_label_variance": init_label_variance,
          "label_variance_ratio": ratio, "n_label_rows": int(y.shape[0]),
          "n_pmt_present": n_pmt_rows, "n_events": int(n_events),
          "label_dim": int(y.shape[1])}
    readings = {}
    layout_high = (e0["f_layout"] is not None and e0["f_layout"] >= 0.5)
    event_low = (e0["f_event"] is not None and e0["f_layout"] is not None
                 and e0["f_event"] <= 0.5 * e0["f_layout"])
    e0c_r2 = (e0c.get("r2") if e0c else None)
    if layout_high and event_low and e0c_r2 is not None and e0c_r2 >= 0.5:
        readings["E0-1"] = "H1 prior supported"
    elif (e0["f_layout"] is not None and e0["f_layout"] <= 0.2
          and e0["f_event"] is not None and e0["f_event"] <= 0.1):
        readings["E0-1"] = "H2 prior supported (low layout + low event)"
    else:
        readings["E0-1"] = "unclear"
    if (invariant is not None and invariant >= 0.5 and position_fit["r2"] is not None
            and position_fit["r2"] >= 0.8):
        readings["E1-1"] = "H1 prior supported"
    elif (ratio is not None and ratio <= 0.1 and position_fit["r2"] is not None
          and position_fit["r2"] <= 0.2):
        readings["E1-1"] = "H2 prior supported"
    else:
        readings["E1-1"] = "unclear"
    payload = {"registered": False, "extension": EXTENSION, "phase": "E0/E1",
               "cell": CELL_ID, "primary_domain": er.PRIMARY_DOMAIN,
               "fix9000": {"source": fix.get("source"),
                           "source_sha256": fix.get("source_sha256") or fix.get("ckpt_sha256"),
                           "ckpt_sha256": fix.get("ckpt_sha256")},
               "init9000": {"rebuilt": True, "seed": 9000},
               "sampling": dict(rows["sampling"]),
               "E0": e0, "E1": e1, "readings": readings,
               "wall_seconds": round(time.time() - started, 3)}
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload, indent=1, default=float) + "\n")
    if position_probe != "linear":
        block = position_probes(rows, probes=(position_probe,), seed=er.SAMPLE_SEED, log=log)
        target = Path(position_out) if position_out is not None else (
            Path(out_path).parent / "e0e1_position_probe.json"
            if out_path is not None else EXT_ROOT / "e0e1_position_probe.json")
        merged = merge_position_payload(target, block)
        payload["position_probe"] = {"path": str(target),
                                     "requested": position_probe,
                                     "probes_present": merged.get("probes_present")}
    else:
        payload["position_probe"] = None
    if code_origin:
        per_pmt = collect_per_pmt_features(config, gen, geometry, data_root,
                                           max_label_events=int(max_label_events),
                                           chunk_events=int(chunk_events),
                                           device=device, log=log)
        block = code_origin_probes(rows, per_pmt, geometry,
                                   probes=tuple(code_origin_methods),
                                   seed=er.SAMPLE_SEED, log=log)
        target = Path(code_origin_out) if code_origin_out is not None else (
            Path(out_path).parent / "e0e1_code_origin.json"
            if out_path is not None else EXT_ROOT / "e0e1_code_origin.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_text(json.dumps(lr._safe(block), indent=1, default=float) + "\n")
        os.replace(tmp, target)
        payload["code_origin"] = {"path": str(target),
                                  "probes": list(block["probes"]),
                                  "features": block["features"]["groups"],
                                  "n_features_all": block["features"]["n_features_all"]}
    else:
        payload["code_origin"] = None
    return payload


# --------------------------------------------------------------- evaluation §3.3
def evaluate_run(item, probes, geometry, config, arms_config, *, device,
                 zero_geometry=False, max_events=None) -> dict:
    """Q1/Q2 readout protocol for one trained ``final.pt`` (§3.3)."""
    return er.evaluate_item(item, probes, geometry, config, arms_config, device=device,
                            forms=("full_input", "half_pass"),
                            chunk_events=32, max_events=max_events,
                            domain=er.PRIMARY_DOMAIN, zero_geometry=zero_geometry)


def reproduction(config, arms_config, geometry, probes, readout_items, *, device,
                 tol=REPRODUCTION_TOLERANCE_M) -> dict:
    """§4.2: the Q1/Q2 readout code re-evaluates fix9000 and the rebuilt init9000."""
    items = json.loads(Path(readout_items).read_text())["items"]
    fix = dict(items["fix9000"])
    fix["source"] = fix.get("ckpt") or fix["source"]
    init = {"label": "init9000", "arm": ARM_NAME, "split": "within", "seed": 9000,
            "beta": None, "source": None}
    rows, ok = {}, True
    for label, item in (("fix9000", fix), ("init9000", init)):
        metrics = er.evaluate_item(item, probes, geometry, config, arms_config,
                                   device=device, forms=("full_input",),
                                   chunk_events=32, max_events=None,
                                   domain=er.PRIMARY_DOMAIN)
        got = metrics["forms"]["full_input"]["vertex_resolution_m"]
        target = REPRODUCTION[label]
        diff = None if got is None else float(abs(got - target))
        passed = bool(got is not None and diff <= float(tol))
        ok = ok and passed
        rows[label] = {"expected_m": float(target), "got_m": got, "abs_diff_m": diff,
                       "pass": passed}
    return {"registered": False, "extension": EXTENSION, "check": "§4.2 reproduction",
            "tolerance_m": float(tol), "items": rows, "reproduction_pass": bool(ok)}


# ----------------------------------------------------------------- readings §6
def recover(full_rows: dict, half_rows: dict, init_full: dict, init_half: dict) -> dict:
    """Three-valued E2-r/E3-r recovery over the two seeds (§6)."""
    seeds = sorted(full_rows)
    vertex_floor = init_full.get("vertex_resolution_m") or INIT_FULL_VERTEX_M
    half_floor = init_half.get("vertex_resolution_m") or INIT_HALF_VERTEX_M
    rows = {}
    for seed in seeds:
        full = full_rows[seed]
        half = half_rows[seed]
        rows[str(seed)] = {
            "full_input_vertex_m": full.get("vertex_resolution_m"),
            "full_input_token_r2": (full.get("token_log_charge_r2") or {}).get("r2"),
            "half_pass_vertex_m": half.get("vertex_resolution_m"),
            "half_pass_token_r2": (half.get("token_log_charge_r2") or {}).get("r2"),
            "vertex_ok": bool(full.get("vertex_resolution_m") is not None
                              and full["vertex_resolution_m"] <= vertex_floor),
            "charge_ok": bool((full.get("token_log_charge_r2") or {}).get("r2") is not None
                              and full["token_log_charge_r2"]["r2"] >= CHARGE_R2_MIN),
            "half_ok": bool(half.get("vertex_resolution_m") is not None
                            and half["vertex_resolution_m"] < half_floor
                            and (half.get("token_log_charge_r2") or {}).get("r2") is not None
                            and half["token_log_charge_r2"]["r2"] >= CHARGE_R2_MIN),
        }
    vertex = [row["vertex_ok"] for row in rows.values()]
    charge = [row["charge_ok"] for row in rows.values()]
    half = [row["half_ok"] for row in rows.values()]
    vertex_floor_used = {"full_input_m": vertex_floor, "half_pass_m": half_floor}
    if all(vertex) and all(charge) and all(half):
        verdict = "supported"
    elif not all(half):
        verdict = "not supported"
    elif (any(vertex) and not all(vertex)) or (any(charge) and not all(charge)) \
            or all(vertex) or all(charge):
        verdict = "partial"
    else:
        verdict = "not supported"
    return {"verdict": verdict, "seeds": rows, "floors": vertex_floor_used,
            "criteria": "supported: both seeds full_input vertex <= init full floor AND "
                        "full_input token R2 >= 0.5 AND half_pass does not contradict "
                        "(half vertex < init half value and half R2 >= 0.5)"}


DECISION = {
    ("supported", "supported"): "H1 + H2 both active (most likely)",
    ("supported", "not supported"): "H1 dominant (deterministic-geometry shortcut)",
    ("not supported", "supported"): "H2 dominant (trainable-target trivial solution)",
    ("not supported", "not supported"): ("neither H1 nor H2; move to H4 (architecture/masking) "
                                         "and H5 (scale)"),
}


def decision(e2: dict, e3: dict) -> dict:
    """The author's conjunction table (§6); any partial -> unclear."""
    pair = (e2["verdict"], e3["verdict"])
    if "partial" in pair:
        verdict = "unclear (a partial recovery; report and design the next runbook: a third " \
                  "seed or the geometry-randomisation variant; no extra training here)"
    else:
        verdict = DECISION[pair]
    return {"registered": False, "extension": EXTENSION, "E2-r": e2["verdict"],
            "E3-r": e3["verdict"], "verdict": verdict,
            "conjunction": "author's §6 table: E2/E3 supported/partial/not supported"}
