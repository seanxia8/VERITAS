# SPDX-License-Identifier: MIT
"""Clean H1 causal-test extension: predictor position-query noise (extension,
``registered: false``).

Runbook: ``docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md`` (2 Nov 2026, revision 3,
authorised).  Extension name: ``h1_clean_2026-11-02``.

Variant ``sigreg_jepa_query_noise``: all six geometry features stay in the
encoder input; the only change is i.i.d. ``N(0, sigma^2)`` noise added to the
**predictor's position-query inputs during training only** (``arm.jepa.predict``
installed at process level through the ``lucid_run.build_arm`` wrapper and
restored in a ``finally``).  The registered ``arm_name`` stays ``"sigreg_jepa"``
(``lucid_arms.build_arm`` rejects names outside ``ALL_ARMS``).

The H12 extension primitives are reused read-only (``eh.preflight_data``,
``eh.load_probes``, ``eh.datasets_for``, ``eh.evaluate_run``,
``eh.reproduction``, ``eh.recover``); the H12 training helpers are *not* reused
because they hardcode the H12 variant names.
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

from . import ext_h12 as eh
from . import ext_readout as er
from . import lucid_run as lr

EXTENSION = "h1_clean_2026-11-02"
VARIANT = "query_noise"
ARM_NAME = "sigreg_jepa"
CELL_ID = "dev_s005_within"
EXT_ROOT = Path("/ceph/dwong/jepa/returns/ext_h1clean_20261102")
READOUT_EXT = Path("/ceph/dwong/jepa/returns/ext_readout_20261101")
RESULTS_DIR = "results/jepa/20261102_h1clean"
RUNBOOK_REL = "docs/jepa/RUNBOOK_2026-11-02_H1_CLEAN.md"
DECISIONS_REL = "docs/jepa/DECISIONS.md"
AUTHOR_LINE = "我授权，你就按这个顺序做：A1（你一句话）+ A2（论文 A 审阅）"
AUTHORISATION_SUBSTRING = "我授权"
#: §3.4 pre-declared sigma ladder (only the first step may be used, once).
SIGMA = 0.10
SIGMA_LADDER = (0.10, 0.15)
MANIPULATION_RATIO_MIN = 1.5
#: §6 floors (decision.json / Q1/Q2 committed matrix).
INIT_FULL_VERTEX_M = 0.160072
INIT_HALF_VERTEX_M = 0.174821
CHARGE_R2_MIN = 0.5
DATA_ROOT_DEFAULT = eh.DATA_ROOT_DEFAULT


# --------------------------------------------------------------------- gate
class AuthorisationError(RuntimeError):
    """The committed runbook or DECISIONS.md does not carry the author line."""


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
    """Hard gate (§0): the committed runbook §0 and DECISIONS.md carry the line."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    runbook = root / RUNBOOK_REL
    decisions = root / DECISIONS_REL
    if not runbook.is_file() or not decisions.is_file():
        raise AuthorisationError(f"missing {runbook} or {decisions}")
    section = _section_zero(runbook.read_text())
    if AUTHORISATION_SUBSTRING not in section:
        raise AuthorisationError(
            f"{RUNBOOK_REL} §0 does not contain the author line "
            f"{AUTHORISATION_SUBSTRING!r}; wrong SHA or unauthorised runbook")
    if AUTHOR_LINE not in decisions.read_text():
        raise AuthorisationError(f"{DECISIONS_REL} does not contain the author line")
    return {"authorised": True, "runbook": str(runbook),
            "runbook_sha256": lr.sha256_file(runbook),
            "decisions": str(decisions)}


# ----------------------------------------------------------------- the patch
def install_query_noise(arm, sigma: float = SIGMA):
    """Add i.i.d. ``N(0, sigma^2)`` to the predictor's position queries.

    Replaces ``arm.jepa.predict`` with a function that gathers the context exactly
    as the original does and adds the noise to ``self.query_features[target_index]``
    before the query projection.  The encoder input, label and predictor
    architecture are untouched.  Returns the original bound method.
    """
    jepa = arm.jepa
    original = jepa.predict
    sigma = float(sigma)

    def predict(readout, context_index, target_index):
        width = readout.size(-1)
        context = readout.gather(1, context_index.unsqueeze(-1).expand(-1, -1, width))
        queries = jepa.query_features[target_index]
        queries = queries + torch.randn_like(queries) * sigma
        return jepa.predictor(context, queries)

    jepa.predict = predict
    return original


def restore_predict(arm, original) -> None:
    """Remove a previously installed :func:`install_query_noise` patch."""
    arm.jepa.predict = original


@contextlib.contextmanager
def query_noise_patch(arm, sigma: float = SIGMA):
    """Install the query-noise patch for the duration of the ``with`` block."""
    original = install_query_noise(arm, sigma)
    try:
        yield arm
    finally:
        restore_predict(arm, original)


@contextlib.contextmanager
def build_arm_patch(sigma: float = SIGMA, log=print):
    """Process-level wrapper on ``lucid_run.build_arm`` applying the variant.

    Mirrors ``eh.build_arm_patch`` but installs the query-noise patch instead of
    the H12 E2/E3 variants; the wrapper is removed in a ``finally``.
    """
    original = lr.build_arm

    def build(arm_name, geometry, arms_config, cell, seed, beta=None, dark_bank=None,
              encoder_config=None):
        made = original(arm_name, geometry, arms_config, cell, seed, beta=beta,
                        dark_bank=dark_bank, encoder_config=encoder_config)
        install_query_noise(made, sigma)
        log(f"[h1clean] build_arm patch: variant={VARIANT} sigma={sigma} "
            f"run={lr.run_id_for(cell.cell_id, arm_name, seed)}")
        return made

    lr.build_arm = build
    try:
        yield
    finally:
        lr.build_arm = original


# ----------------------------------------------------------------- run dirs
def run_id_for(seed) -> str:
    return lr.run_id_for(CELL_ID, ARM_NAME, int(seed))


def run_dir_for(out_root, seed) -> Path:
    return Path(out_root) / "runs" / run_id_for(seed)


def prepare_run_dir(out_root, seed, *, sigma: float = SIGMA, steps=None,
                    device=None) -> tuple:
    """Refuse an existing run directory and write the per-run variant sidecar."""
    run_id, run_dir = run_id_for(seed), run_dir_for(out_root, seed)
    if run_dir.exists():
        raise RuntimeError(f"{run_dir} already exists; refusing to launch (run_one would "
                           f"auto-resume it). Archive or remove it first.")
    run_dir.mkdir(parents=True, exist_ok=False)
    sidecar = {"registered": False, "extension": EXTENSION, "variant": VARIANT,
               "seed": int(seed), "arm": ARM_NAME, "cell_id": CELL_ID,
               "sigma": float(sigma), "steps": steps, "device": str(device),
               "out_root": str(out_root), "run_id": run_id}
    (run_dir / "variant.json").write_text(json.dumps(sidecar, indent=1) + "\n")
    return run_id, run_dir


def assert_fresh_resume(status: dict, run_id: str = "") -> int:
    """The run must not have resumed: ``_checkpoint`` does not save the global RNG."""
    resumed = status.get("resumed_from_step")
    if resumed is None:
        raise RuntimeError(f"{run_id}: run status has no resumed_from_step")
    if int(resumed) != 0:
        raise RuntimeError(f"{run_id}: resumed_from_step={resumed} != 0; the query-noise "
                           f"stream cannot be replayed from a checkpoint")
    return int(resumed)


# ------------------------------------------------------------ manipulation §3.4
def _prediction_mse(arm, batch) -> float:
    with torch.no_grad():
        out = arm._forward_sigreg(batch)
    return float(torch.nn.functional.mse_loss(out["prediction"],
                                              out["target_tokens"]).detach())


def _mse_with_and_without_noise(arm, batch, sigma: float) -> dict:
    """Prediction MSE with and without the query noise on one fixed batch.

    The patched pass reseeds the global torch RNG (seed 0) right before its one
    noise draw, so the ratio is deterministic for a fixed arm/batch.
    """
    unpatched = _prediction_mse(arm, batch)
    with query_noise_patch(arm, sigma):
        torch.manual_seed(0)
        patched = _prediction_mse(arm, batch)
    ratio = (patched / unpatched) if unpatched > 0 else None
    return {"mse_without_noise": unpatched, "mse_with_noise": patched,
            "ratio": ratio, "sigma": float(sigma),
            "pass": bool(ratio is not None and ratio >= MANIPULATION_RATIO_MIN)}


def manipulation_check(config, gen, arms_config, geometry, data_root, *, device="cpu",
                       sigmas=SIGMA_LADDER, log=print) -> dict:
    """σ ladder on one fixed saved batch of the untrained seed-9000 arm (§3.4)."""
    cell = lr.dev_cell(config)
    train_ds, _ = eh.datasets_for(cell, config, gen, geometry, Path(data_root))
    torch.manual_seed(9000)
    np.random.seed(9000 % 2 ** 32)
    arm = lr.build_arm(ARM_NAME, geometry, arms_config, cell, 9000).to(device)
    arm.eval()
    loader = lr.TrainingLoader(train_ds, geometry, split=cell.split, seed=9000,
                               occupancy=cell.occupancy,
                               n_min=int(config["views"]["n_min"]),
                               batch=int(config["training"]["batch"]))
    batch, _ = loader.batch_at(0)
    batch = lr._to_device(batch, device)
    checks = []
    chosen = None
    for sigma in sigmas:
        check = _mse_with_and_without_noise(arm, batch, float(sigma))
        checks.append(check)
        log(f"[h1clean] manipulation check sigma={sigma}: mse {check['mse_without_noise']:.6g}"
            f" -> {check['mse_with_noise']:.6g} ratio={check['ratio']} "
            f"({'PASS' if check['pass'] else 'FAIL'})")
        if check["pass"]:
            chosen = float(sigma)
            break
    return {"registered": False, "extension": EXTENSION,
            "check": "§3.4 manipulation check (untrained seed-9000 arm, fixed saved batch)",
            "batch": {"step": 0, "n_events": int(batch.features.size(0)),
                      "label": "TrainingLoader.batch_at(0), seed 9000"},
            "ratio_min": MANIPULATION_RATIO_MIN,
            "checks": checks, "sigma_used": chosen,
            "pass": chosen is not None}


# ------------------------------------------------------------------- training
def run_one_training(*, seed, out_root, config, gen, arms_config, geometry, device,
                     data_root, config_path, code_sha, status_text, sigma,
                     steps=None, probe_max_events=None, log=print) -> dict:
    """The real registered run with the query-noise patch installed."""
    gate = assert_authorised()
    preflight = eh.preflight_data(Path(data_root))
    if not preflight.get("pass"):
        raise RuntimeError(f"§3.5 data preflight failed: {preflight}")
    cell = lr.dev_cell(config)
    train_ds, probes = eh.datasets_for(cell, config, gen, geometry, Path(data_root),
                                       code_sha=None)
    scheduled = int(config["training"]["steps"] if steps is None else steps)
    run_id, run_dir = prepare_run_dir(out_root, seed, sigma=sigma, steps=scheduled,
                                     device=device)
    started = time.time()
    with build_arm_patch(sigma, log=log):
        result = lr.run_one(
            arm_name=ARM_NAME, cell=cell, seed=int(seed), beta=None, config=config,
            arms_config=arms_config, geometry=geometry, loader_dataset=train_ds,
            probe_datasets=probes, dark_bank=None, out_root=Path(out_root),
            data_root=Path(data_root), config_sha=lr.sha256_file(config_path),
            code_sha=code_sha, status_text=status_text, device=device, steps=scheduled,
            micro_batch=None, resume=None, stop_after=None, checkpoint_minutes=None,
            evaluate=False, allow_pending=False, step_callback=None,
            probe_max_events=probe_max_events, save_embeddings=False,
            physics_method="ridge", tf32=False, log=log)
    status = json.loads((run_dir / "run_status.json").read_text())
    resumed = assert_fresh_resume(status, run_id)
    summary = summarize_run(run_dir, seed=seed, sigma=sigma,
                            wall_seconds=round(time.time() - started, 3),
                            steps=scheduled, resumed_from_step=resumed)
    summary["smoke"] = bool(steps is not None and int(steps) == 20)
    (run_dir / "ext_summary.json").write_text(json.dumps(summary, indent=1,
                                                         default=float) + "\n")
    return {"result": result, "run_dir": run_dir, "summary": summary,
            "run_id": run_id, "seed": int(seed)}


def summarize_run(run_dir, *, seed, sigma, wall_seconds=None, steps=None,
                  resumed_from_step=0, forms=None) -> dict:
    """Finite ``ext_summary.json`` payload for a smoke or an approved run."""
    run_dir = Path(run_dir)
    log_path = run_dir / "train_log.jsonl"
    rows = ([json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
            if log_path.is_file() else [])
    check = eh.train_log_check(run_dir)
    checkpoint = run_dir / "final.pt"
    status = json.loads((run_dir / "run_status.json").read_text()) \
        if (run_dir / "run_status.json").is_file() else {}
    return {"registered": False, "extension": EXTENSION, "variant": VARIANT,
            "seed": int(seed), "run_id": run_dir.name, "arm": ARM_NAME,
            "cell_id": CELL_ID, "sigma": float(sigma), "steps_scheduled": steps,
            "steps_completed": status.get("steps_completed"),
            "resumed_from_step": int(resumed_from_step),
            "status": status.get("outcome"), "n_log_rows": len(rows),
            "final_train_metrics": check.get("final_metrics"),
            "non_finite_keys": check.get("non_finite_keys"),
            "metrics_finite": check.get("finite"),
            "final_checkpoint_sha256": (lr.sha256_file(checkpoint)
                                        if checkpoint.is_file() else None),
            "wall_seconds": wall_seconds, "forms": forms}


# ------------------------------------------------------------- evaluation §3.3
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
        item = {"label": f"query_noise_{seed}", "arm": ARM_NAME, "split": "within",
                "seed": seed, "beta": None, "source": str(source)}
        metrics = eh.evaluate_run(item, probes, geometry, config, arms_config,
                                  device=device, max_events=max_events)
        runs[str(seed)] = {
            "variant": VARIANT, "seed": seed, "sigma": float(sidecar.get("sigma", SIGMA)),
            "run_id": run_dir.name, "run_dir": str(run_dir),
            "checkpoint_sha256": lr.sha256_file(source),
            "full_input": metrics["forms"]["full_input"],
            "half_pass": metrics["forms"]["half_pass"],
            "ext_summary": (json.loads((run_dir / "ext_summary.json").read_text())
                            if (run_dir / "ext_summary.json").is_file() else None),
        }
    return runs


def write_variants_json(ext_root, run_dirs) -> dict:
    """The aggregated variant map (§2) from the per-run sidecars."""
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


# --------------------------------------------------------------- reading §6
def reading(runs: dict, init_full: dict, init_half: dict, *,
            manipulation_pass: bool) -> dict:
    """One-sided H1-clean-r reading (§6), using ``eh.recover``'s criteria."""
    full_rows = {int(seed): row["full_input"] for seed, row in runs.items()}
    half_rows = {int(seed): row["half_pass"] for seed, row in runs.items()}
    recovered = eh.recover(full_rows, half_rows, init_full, init_half)
    raw = recovered["verdict"]
    if raw == "supported":
        verdict = "supported"
        interpretation = ("the exact-position query route contributes to the collapse "
                          "(mechanism result; not evidence that the label is a position "
                          "function)")
    elif raw == "partial":
        verdict = "unclear (partial recovery)"
        interpretation = "one of the two criteria holds for both seeds, or the seeds disagree"
    else:
        verdict = "not supported" if manipulation_pass else "unclear (non-discriminating)"
        interpretation = ("the query route is not sufficient; this is NOT evidence for H2"
                          + (" (the manipulation check passed, so record 'query route not "
                             "supported')" if manipulation_pass else
                             " (the manipulation check did not pass)"))
    return {"registered": False, "extension": EXTENSION, "id": "H1-clean-r",
            "question": "does the predictor's exact target-position query route cause the "
                        "collapse? (one-sided)",
            "criteria": "supported: both seeds full_input vertex <= %.6f AND token ln q "
                        "R2 >= %.1f AND half_pass vertex < %.6f with half R2 >= %.1f"
                        % (INIT_FULL_VERTEX_M, CHARGE_R2_MIN, INIT_HALF_VERTEX_M,
                           CHARGE_R2_MIN),
            "verdict": verdict, "eh_recover_verdict": raw,
            "manipulation_pass": bool(manipulation_pass),
            "interpretation": interpretation, "seeds": recovered["seeds"],
            "floors": recovered["floors"]}
