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
    metrics = {key: value for key, value in final.items()
               if isinstance(value, (int, float))}
    finite = all(math.isfinite(float(value)) for value in metrics.values())
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


def e0e1(config, gen, arms_config, geometry, data_root, readout_items, out_path, *,
         device, max_label_events: int = 2000, chunk_events: int = 8, log=print) -> dict:
    """E0-a/b/c and E1 on the ``fix9000`` (and rebuilt ``init9000``) encoders."""
    started = time.time()
    items = json.loads(Path(readout_items).read_text())["items"]
    fix = dict(items["fix9000"])
    fix["source"] = fix.get("ckpt") or fix["source"]
    if fix.get("ckpt") and lr.sha256_file(Path(fix["ckpt"])) != fix["ckpt_sha256"]:
        raise RuntimeError("E0/E1: the verified fix9000 checkpoint copy has changed")
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
    x = np.asarray(x_rows, np.float64)
    y = np.asarray(y_rows, np.float64)
    init_y = np.asarray(init_rows, np.float64)
    pmts = np.asarray(pmt_rows, np.int64)
    event_index = np.asarray(event_rows, np.int64)
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
    fractions = er.layout_fractions(layout)
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
               "sampling": {"max_label_events": int(max_label_events),
                            "tokens_per_event": 32, "seed": er.SAMPLE_SEED,
                            "token_rows": "all PMT tokens of the cut-0.8 train events"},
               "E0": e0, "E1": e1, "readings": readings,
               "wall_seconds": round(time.time() - started, 3)}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(payload, indent=1, default=float) + "\n")
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
