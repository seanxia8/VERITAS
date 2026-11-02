# SPDX-License-Identifier: MIT
"""Frozen static-target extension (extension, ``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md`` (2 Nov 2026, revision 2,
authorised).  Extension name: ``static_target_2026-11-02``.

Variant ``sigreg_jepa_static_target`` (variant label only; ``arm_name`` stays
``"sigreg_jepa"``): the prediction label comes from a frozen random second encoder
(the same ``torch.manual_seed(0)`` net as the H12 E3 frozen target), fed a **clone**
of ``batch.features`` whose event columns 6-8 (ln q, time, hit) are zeroed; the
context encoder keeps the full features.  For the ``within`` split with natural
occupancy the target-only label is therefore exactly event-invariant (a function of
the fixed PMT geometry only), while everything else is unchanged.

Freezing / no-grad / detach / device plumbing reuses
``ext_h12.attach_frozen_target``; the label is installed by rebinding
``arm._encode_views`` (the E3 precedent) through a process-level ``build_arm``
wrapper in the CLI.  The gate, manipulation check, sidecars and the
``static_target_r`` reading live here.
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from . import ext_h12 as eh
from . import ext_readout as er
from . import lucid_run as lr
from . import point_encoder as pe

EXTENSION = "static_target_2026-11-02"
VARIANT = "static_target"
ARM_NAME = "sigreg_jepa"
CELL_ID = "dev_s005_within"
EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_static_20261102")
READOUT_EXT = Path("/ceph/dwong/jepa/returns/ext_readout_20261101")
RESULTS_DIR = "results/jepa/20261102_static"
RUNBOOK_REL = "docs/jepa/RUNBOOK_2026-11-02_STATIC_TARGET.md"
DECISIONS_REL = "docs/jepa/DECISIONS.md"
#: Author line (verbatim, runbook §0).
AUTHOR_LINE = "frozen static-target 变体训练 我授权"
#: Event columns of ``point_encoder.FEATURES`` zeroed in the target pass: 6 ln q,
#: 7 time_100ns, 8 hit (columns 0-5 = scaled position/direction stay).
EVENT_COLUMNS = slice(6, 9)
#: §2 pinned thresholds for the two-sided ``static_target_r`` reading.
INIT_FULL_VERTEX_M = 0.160072
INIT_HALF_VERTEX_M = 0.174821
CHARGE_R2_MIN = 0.5
MAX_LABEL_EVENT_FRACTION = 1e-3
DATA_ROOT_DEFAULT = eh.DATA_ROOT_DEFAULT


# --------------------------------------------------------------------- gate
def _section_zero(text: str) -> str:
    """The runbook's §0 block (from its heading to the next ``## `` heading)."""
    start = text.find("## 0.")
    if start < 0:
        start = text.find("## 0 ")
    if start < 0:
        return ""
    end = text.find("\n## ", start + 3)
    return text[start: end if end >= 0 else None]


def assert_authorised(root=None) -> dict:
    """Hard gate (§0): banner-absent §0 with the author line + DECISIONS line.

    Exits with status 2 (``SystemExit(2)``) otherwise.  Uses its own
    ``RUNBOOK_REL``/``AUTHOR_LINE`` (not ``ext_h1clean.assert_authorised``, which
    hard-codes the H1-clean runbook).
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
            problems.append(f"{RUNBOOK_REL} §0 does not contain the author line "
                            f"{AUTHOR_LINE!r}")
        if "NOT AUTHORISED" in section:
            problems.append(f"{RUNBOOK_REL} §0 still carries a NOT AUTHORISED banner")
        if AUTHOR_LINE not in decisions.read_text():
            problems.append(f"{DECISIONS_REL} does not contain the author line")
    if problems:
        for problem in problems:
            print(f"[static-target] authorisation gate: {problem}", file=sys.stderr)
        raise SystemExit(2)
    return {"authorised": True, "runbook": str(runbook),
            "runbook_sha256": lr.sha256_file(runbook),
            "decisions": str(decisions),
            "author_line": AUTHOR_LINE}


# ------------------------------------------------- frozen static target (§2)
def static_target_features(features: torch.Tensor) -> torch.Tensor:
    """A clone of ``features`` with the event columns 6-8 zeroed (never mutates)."""
    if features.size(-1) < EVENT_COLUMNS.stop:
        raise ValueError("static target needs at least nine feature columns")
    clone = features.clone()
    clone[..., EVENT_COLUMNS] = 0.0
    return clone


def attach_static_target(arm, geometry):
    """Install the frozen static (geometry-only) target label on a built arm.

    Reuses ``eh.attach_frozen_target`` for the frozen seed-0 net / ``requires_grad`` /
    ``eval`` / ``object.__setattr__`` / ``_apply`` device plumbing, then rebinds
    ``_encode_views`` so the label is the frozen target-only encoding of the
    zeroed-event clone (columns 6-8), gathered at ``hidden_index`` and detached.
    """
    arm = eh.attach_frozen_target(arm, geometry)
    frozen = arm.__dict__["_ext_frozen_target"]

    def _encode_views(batch, need_full=True):
        hidden_index = arm._hidden_index(batch)
        context = arm.jepa.encode(batch.features, hidden_mask=batch.hidden)
        prediction = arm.jepa.predict(context, batch.context_index, hidden_index)
        if not need_full:
            return context, prediction, None, hidden_index, None
        full = arm.jepa.encode(batch.features)
        with torch.no_grad():
            label_features = static_target_features(batch.features)
            label_full = frozen.encode(label_features, hidden_mask=~batch.target)
        gather = hidden_index.unsqueeze(-1).expand(-1, -1, arm.d_model)
        labels = label_full.gather(1, gather).detach()
        return context, prediction, labels, hidden_index, full

    object.__setattr__(arm, "_encode_views", _encode_views)
    return arm


@contextlib.contextmanager
def build_arm_patch(log=print):
    """Process-level wrapper on ``lucid_run.build_arm`` installing the variant."""
    original = lr.build_arm

    def build(arm_name, geometry, arms_config, cell, seed, beta=None, dark_bank=None,
              encoder_config=None):
        made = original(arm_name, geometry, arms_config, cell, seed, beta=beta,
                        dark_bank=dark_bank, encoder_config=encoder_config)
        attach_static_target(made, geometry)
        log(f"[static-target] build_arm patch: variant={VARIANT} "
            f"run={lr.run_id_for(cell.cell_id, arm_name, seed)}")
        return made

    lr.build_arm = build
    try:
        yield
    finally:
        lr.build_arm = original


# ------------------------------------------------------ manipulation check §2
def static_label_manipulation(arm, batch, *,
                              max_event_fraction: float = MAX_LABEL_EVENT_FRACTION) -> dict:
    """Implementation guard on one batch: zeroed clone, static label, varied per PMT."""
    frozen = arm.__dict__["_ext_frozen_target"]
    features = batch.features
    original = features.detach().clone()
    label_features = static_target_features(features)
    checks = {
        "event_columns_zero": bool(torch.equal(
            label_features[..., EVENT_COLUMNS].detach(),
            torch.zeros_like(label_features[..., EVENT_COLUMNS]))),
        "geometry_columns_unchanged": bool(torch.equal(
            label_features[..., :EVENT_COLUMNS.start].detach(),
            features[..., :EVENT_COLUMNS.start].detach())),
        "original_features_unchanged": bool(torch.equal(features.detach(), original)),
    }
    with torch.no_grad():
        label_tokens = frozen.encode(label_features, hidden_mask=~batch.target)
    tokens = label_tokens.detach().to(torch.float32).cpu().numpy().astype(np.float64)
    fractions = er.layout_and_event_fractions(tokens.mean(axis=1), tokens)
    state = er.layout_state(tokens.shape[1], tokens.shape[2])
    er.layout_add(state, tokens)
    n_events = max(int(state["n_events"]), 1)
    m_p = state["sum_pd"] / n_events
    per_dim_pmt_variance = m_p.var(axis=0, ddof=1)
    per_pmt_variance = (float(np.mean(per_dim_pmt_variance))
                        if per_dim_pmt_variance.size else None)
    event_mean = tokens.mean(axis=0, keepdims=True)
    static_deviation = float(np.abs(tokens - event_mean).max())
    event_fraction = fractions.get("f_event")
    passed = bool(all(checks.values())
                  and event_fraction is not None
                  and event_fraction <= float(max_event_fraction)
                  and per_pmt_variance is not None
                  and per_pmt_variance > 0.0)
    reasons = []
    if not all(checks.values()):
        reasons.append("clone/feature check failed")
    if event_fraction is None:
        reasons.append("label event fraction not evaluable (constant representation?)")
    elif event_fraction > float(max_event_fraction):
        reasons.append(f"label event fraction {event_fraction:.3e} > {max_event_fraction:g}")
    if per_pmt_variance is None or per_pmt_variance <= 0.0:
        reasons.append("per-PMT label variance is not positive")
    return {"checks": checks, "label_event_fraction": event_fraction,
            "label_event_fraction_max": float(max_event_fraction),
            "per_pmt_variance": per_pmt_variance,
            "static_deviation_max_abs": static_deviation,
            "n_events": int(tokens.shape[0]), "n_tokens": int(tokens.shape[1]),
            "pass": passed, "reasons": reasons,
            "note": "within + natural occupancy; the target-only label is a function "
                    "of the fixed PMT geometry only (§2)"}


def assert_static_splits(config) -> dict:
    """Pin the split/occupancy to ``within`` + ``natural`` (§2); stop otherwise."""
    cell = lr.dev_cell(config)
    occupancy = str(getattr(cell, "occupancy", "natural") or "natural")
    info = {"split": str(cell.split), "occupancy": occupancy, "cell_id": cell.cell_id,
            "pass": bool(str(cell.split) == "within" and occupancy == "natural")}
    if not info["pass"]:
        raise RuntimeError(f"static target requires within+natural, got {info}")
    return info


def manipulation_check(config, gen, arms_config, geometry, data_root, *, device="cpu",
                       log=print) -> dict:
    """§2 manipulation check on one fixed saved batch with the untrained seed-9000 arm."""
    device = device if isinstance(device, torch.device) else torch.device(device)
    splits = assert_static_splits(config)
    cell = lr.dev_cell(config)
    train_ds, _ = eh.datasets_for(cell, config, gen, geometry, Path(data_root))
    torch.manual_seed(9000)
    np.random.seed(9000 % 2 ** 32)
    arm = lr.build_arm(ARM_NAME, geometry, arms_config, cell, 9000).to(device)
    attach_static_target(arm, geometry)
    arm.eval()
    loader = lr.TrainingLoader(train_ds, geometry, split=cell.split, seed=9000,
                               occupancy=cell.occupancy,
                               n_min=int(config["views"]["n_min"]),
                               batch=int(config["training"]["batch"]))
    batch, _ = loader.batch_at(0)
    batch = lr._to_device(batch, device)
    check = static_label_manipulation(arm, batch)
    check.update({"registered": False, "extension": EXTENSION,
                  "check": "§2 manipulation check (implementation guard)",
                  "batch": {"step": 0, "n_events": int(batch.features.size(0)),
                            "label": "TrainingLoader.batch_at(0), seed 9000"},
                  "splits": splits})
    log(f"[static-target] manipulation check: f_event={check['label_event_fraction']} "
        f"per-PMT variance={check['per_pmt_variance']} pass={check['pass']}")
    return check


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
               "run_id": run_id}
    (run_dir / "variant.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return run_id, run_dir


def summarize_run(run_dir, *, seed, wall_seconds=None, steps=None,
                  resumed_from_step=0, forms=None) -> dict:
    """Finite ``ext_summary.json`` payload for a smoke or an approved run."""
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
            "final_train_metrics": check.get("final_metrics"),
            "non_finite_keys": check.get("non_finite_keys"),
            "metrics_finite": check.get("finite"),
            "final_checkpoint_sha256": (lr.sha256_file(checkpoint)
                                        if checkpoint.is_file() else None),
            "wall_seconds": wall_seconds, "forms": forms}


# ------------------------------------------------------------------- training
def run_one_training(*, seed, out_root, config, gen, arms_config, geometry, device,
                     data_root, config_path, code_sha, status_text, steps=None,
                     probe_max_events=None, log=print) -> dict:
    """The registered ``run_one`` with the frozen static target installed.

    The gate is called here (library entry) and the split/occupancy is pinned.
    """
    gate = assert_authorised()
    preflight = eh.preflight_data(Path(data_root))
    if not preflight.get("pass"):
        raise RuntimeError(f"§3.3 data preflight failed: {preflight}")
    splits = assert_static_splits(config)
    cell = lr.dev_cell(config)
    train_ds, probes = eh.datasets_for(cell, config, gen, geometry, Path(data_root),
                                       code_sha=None)
    scheduled = int(config["training"]["steps"] if steps is None else steps)
    run_id, run_dir = prepare_run_dir(out_root, seed, steps=scheduled, device=device)
    started = time.time()
    with build_arm_patch(log=log):
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
    status = json.loads((run_dir / "run_status.json").read_text())
    from . import ext_h1clean as ec
    resumed = ec.assert_fresh_resume(status, run_id)
    summary = summarize_run(run_dir, seed=seed,
                            wall_seconds=round(time.time() - started, 3),
                            steps=scheduled, resumed_from_step=resumed)
    summary["splits"] = splits
    summary["smoke"] = bool(steps is not None and int(steps) == 20)
    (run_dir / "ext_summary.json").write_text(json.dumps(summary, indent=1,
                                                         default=float) + "\n")
    return {"result": result, "run_dir": run_dir, "summary": summary,
            "run_id": run_id, "seed": int(seed)}


# ------------------------------------------------------------- evaluation §2
def evaluate_run_dirs(run_dirs, probes, geometry, config, arms_config, *, device,
                      max_events=None) -> dict:
    """Q1/Q2 readout protocol (full_input + half_pass) for every ``final.pt``."""
    runs = {}
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        sidecar_path = run_dir / "variant.json"
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
        seed = int(sidecar.get("seed", 0))
        source = run_dir / "final.pt"
        if not source.is_file():
            raise RuntimeError(f"{run_dir}: final.pt is missing")
        item = {"label": f"static_target_{seed}", "arm": ARM_NAME, "split": "within",
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


# ------------------------------------------------------- reading §2 (two-sided)
def static_target_r(runs: dict, *, full_floor: float = INIT_FULL_VERTEX_M,
                    half_floor: float = INIT_HALF_VERTEX_M,
                    charge_min: float = CHARGE_R2_MIN) -> dict:
    """The pinned two-sided ``static_target_r`` reading (§2).

    ``recovered``: both seeds full vertex strictly < floor, full token R2 >= 0.5,
    half vertex < half floor and half token R2 >= 0.5.  ``not recovered``: both
    seeds fail the full-input criterion (vertex >= floor or token R2 < 0.5).
    Anything else is ``unclear``.
    """
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
                           "full_criterion": ("vertex+charge" if recovers else
                                              ("failed" if fails else "partial")),
                           "recovers": recovers, "fails": fails}
    values = list(rows.values())
    if values and all(row["recovers"] for row in values):
        verdict = "recovered"
        direction = ("recovered (nearly unreachable by construction — treat as a leak "
                     "detector, not as evidence against staticity)")
    elif values and all(row["fails"] for row in values):
        verdict = "not recovered"
        direction = ("not recovered (cannot distinguish staticity from a mere absence of "
                     "target event-information; two-sided descriptive contrast against E3)")
    else:
        verdict = "unclear"
        direction = "unclear (partial recovery, seed disagreement or half-pass contradiction)"
    return {"registered": False, "extension": EXTENSION, "id": "static_target_r",
            "question": "does an event-invariant (frozen static) target alone prevent "
                        "learning? (two-sided descriptive contrast against E3)",
            "criteria": "recovered: both seeds full vertex < %.6f, full token R2 >= %.1f, "
                        "half vertex < %.6f, half token R2 >= %.1f; not recovered: both "
                        "seeds full vertex >= %.6f or full token R2 < %.1f"
                        % (full_floor, charge_min, half_floor, charge_min, full_floor,
                           charge_min),
            "verdict": verdict, "direction": direction, "seeds": rows,
            "floors": {"full_input_m": float(full_floor),
                       "half_pass_m": float(half_floor),
                       "charge_r2_min": float(charge_min)}}
