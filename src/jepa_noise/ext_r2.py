# SPDX-License-Identifier: MIT
"""Round 2: the round-1 recipe on SK, HK and JUNO (extension ``r2_multidetector_2026-11-11``).

Binding: ``docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md`` (sha256 366f068...).
``registered: false``.  New code only; every round-1 file stays byte-identical.

Pieces (runbook section in brackets):

* arms and their encoders [§5.1]: ``sigreg_jepa``, ``target_posfree``, ``mae_mse``,
  ``init`` (no training) and the receptive-field control ``mae_mse_rf``;
* training [§2.1, §5.2] with the round-1 optimiser, loop and checkpoints, fed by
  :class:`r2_store.R2TrainingLoader` (4 events from each of 8 runs);
* the loss-share logger [§6], the readings R2-a..R2-e [§7], the conditional seed-9002 rule
  and the HK fallback [§8], the forbidden-path guard [§0 "not authorised"];
* evaluation [§6] (vertex full input + half pass, light yield, token ln q R², f_layout /
  f_event, effective rank, module gain, mu/e AUC, direction MLP, deposited energy) built on
  the round-1 probe functions;
* the X0 preflight [§4.2] (table, trigger check, yield range, Gate V, benchmark);
* the CLI [§4.1]: dry-run, preflight, generate, pack, smoke, train, evaluate, report.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import lucid_bombs as lb
from . import r2_geometry as rg

EXTENSION = "r2_multidetector_2026-11-11"
RUNBOOK_RELPATH = "docs/round2/RUNBOOK_2026-11-11_R2_MULTIDETECTOR.md"
RUNBOOK_SHA256 = "366f0680a5b262a90774bf725fd6ee75c516766789a605807b22978630572afe"
DETECTOR_ORDER = ("sk", "juno", "hk")
MAIN_ARMS = ("sigreg_jepa", "target_posfree", "mae_mse")
RF_ARM = "mae_mse_rf"
INIT_ARM = "init"
TRAINED_ARMS = MAIN_ARMS + (RF_ARM,)
ALL_ARMS = MAIN_ARMS + (INIT_ARM, RF_ARM)
MAIN_SEEDS = (9000, 9001)
INIT_SEEDS = (9000, 9001, 9002, 9003, 9004)
RF_SEEDS = (9000,)
CONDITIONAL_SEED = 9002
STEPS = 10_000
SMOKE_STEPS = 20
BATCH = 32
MICRO_BATCH_CHOICES = (8, 4, 2, 1)
GPU_MEMORY_FRACTION_CAP = 0.45
L40S_MEMORY_GIB = 44.4  # usable L40S memory reported by torch (46,068 MiB card)
WALL_CAP_FACTOR = 1.5
GPU_CAP_H = 60.0
LEDGER_LINE = "R2_l40s_deepthought2"
ROUND1_MAE_SEED_SPREAD_M = 0.015
WCTE_ORDER = ("mae_mse", "target_posfree", "init", "sigreg_jepa")  # best -> worst (Q4)
F_EVENT_INVARIANT, F_EVENT_DEPENDENT = 1e-3, 0.01
TOKEN_R2_RECOVERED = 0.5
BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED = 1000, 20261103

DATA_ROOT = Path("/ceph/dwong/jepa/data/round2")
GEN_ROOT = Path("/work/dwong/r2_gen")
RETURNS_ROOT = Path(f"/ceph/dwong/jepa/returns/{EXTENSION}")
ROUND1_DATA_ROOT = Path("/ceph/dwong/jepa/data")
ROUND1_CHECKPOINT_ROOTS = (Path("/ceph/dwong/jepa/returns"), Path("/work/dwong/jepa_a10"),
                           Path("/ceph/dwong/jepa/code"))
RUN_CONFIG_RELPATH = "src/jepa_noise/configs/lucid_run_v0_mixed_calibrated.json"
ARMS_CONFIG_RELPATH = "src/jepa_noise/configs/lucid_arms_v0.json"
PHOTONSIM_DIR = Path("/work/dwong/pmt_multienergy_20261016/source_balanced")
GATE_V_HOLDOUT = tuple(f"e500_{i:03d}.npz" for i in range(10, 20))
UPSTREAM_MANIFEST_RELPATH = "configs/lucid_pmt/upstream_manifest.json"


# ======================================================================= guards
class ForbiddenPath(PermissionError):
    pass


def check_write_path(path) -> Path:
    """Refuse any write under round-1 data, round-1 checkpoints or code clones (§0).

    Both the normalised path and its symlink-resolved target are checked."""
    p = Path(os.path.abspath(str(path)))
    real = Path(os.path.realpath(str(path)))
    if real != p:
        check_write_path(real)
    if p.is_relative_to(ROUND1_DATA_ROOT) and not p.is_relative_to(DATA_ROOT):
        raise ForbiddenPath(f"{p}: round-1 data is read-only (runbook §0)")
    for root in ROUND1_CHECKPOINT_ROOTS:
        if p.is_relative_to(root) and not p.is_relative_to(RETURNS_ROOT):
            raise ForbiddenPath(f"{p}: round-1 checkpoints/results are read-only (runbook §0)")
    return p


def allowed_training(arm: str, seed: int, *, detector: str, triggered=()) -> None:
    """Only the §5.1 runs and the §8 conditional seed-9002 runs may train."""
    if detector not in DETECTOR_ORDER:
        raise ValueError(f"training on {detector!r} is not in §5 (only {DETECTOR_ORDER})")
    if arm == INIT_ARM:
        raise ValueError("init is never trained (§5.1)")
    if arm in MAIN_ARMS and int(seed) in MAIN_SEEDS:
        return
    if arm == RF_ARM and int(seed) in RF_SEEDS:
        return
    if int(seed) == CONDITIONAL_SEED and arm in MAIN_ARMS and arm in set(triggered):
        return
    raise ValueError(f"training {arm} seed {seed} on {detector} is not authorised (§5.1, §8)")


# ======================================================================= configs and arms
def run_config() -> dict:
    return json.loads((lb._repo_root() / RUN_CONFIG_RELPATH).read_text())


def arms_config() -> dict:
    return json.loads((lb._repo_root() / ARMS_CONFIG_RELPATH).read_text())


def encoder_config_for(arm: str, geometry):
    """``EncoderConfig()`` defaults; ``mae_mse_rf`` changes only window and shift (§2.2)."""
    from .point_encoder import EncoderConfig
    if arm == RF_ARM:
        w, s = rg.rf_window_rule(geometry.n_pmt)
        return EncoderConfig(window=int(w), shift=int(s))
    return EncoderConfig()


def base_arm_name(arm: str) -> str:
    """The round-1 arm a round-2 arm is built from."""
    return {"target_posfree": "sigreg_jepa", "init": "sigreg_jepa", RF_ARM: "mae_mse"}.get(arm, arm)


def build_r2_arm(arm: str, geometry, seed: int, *, scaling=None, arms_cfg=None, run_cfg=None):
    """Build one §5.1 arm on a round-2 geometry with the round-1 builders."""
    import torch
    from . import lucid_arms as la
    from . import lucid_run as lr
    from .ext_shortcut import attach_posfree_target
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown round-2 arm {arm!r}")
    run_cfg = run_config() if run_cfg is None else run_cfg
    arms_cfg = arms_config() if arms_cfg is None else arms_cfg
    cell = lr.dev_cell(run_cfg)
    settings = la.ArmSettings.from_config(arms_cfg, cell=(cell.s_arm, cell.s_p))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % 2 ** 32)
    made = la.build_arm(base_arm_name(arm), geometry, settings=settings, seed=int(seed),
                        encoder_config=encoder_config_for(arm, geometry), scaling=scaling)
    if arm == "target_posfree":
        attach_posfree_target(made)
    return made


def config_diff(a, b) -> dict:
    """Field-by-field difference of two dataclass configs."""
    da, db = dataclasses.asdict(a), dataclasses.asdict(b)
    return {k: (da[k], db[k]) for k in da if da[k] != db[k]}


# ======================================================================= loss shares (§6)
LOSS_TERMS = ("charge", "hit", "time")


def loss_shares(lines, *, first: int = 100, last: int = 100) -> dict:
    """Share of each MAE loss term (ln q, hit, time) in the total, start and end means.

    ``lines`` are the per-step ``train_log.jsonl`` rows (``{"step", "metrics"}``).  The
    shares of a window are the window-mean term over the window-mean sum of the terms.
    """
    rows = sorted((dict(r) for r in lines), key=lambda r: int(r["step"]))
    if not rows:
        raise ValueError("loss_shares: no log rows")
    out = {}
    for label, window in (("start", rows[:int(first)]), ("end", rows[-int(last):])):
        means = {t: float(np.mean([float(r["metrics"][t]) for r in window])) for t in LOSS_TERMS}
        total = sum(means.values())
        out[label] = {"steps": [int(window[0]["step"]), int(window[-1]["step"])],
                      "n_steps": len(window), "mean_terms": means,
                      "mean_loss": float(np.mean([float(r["metrics"]["loss"]) for r in window])),
                      "shares": {t: (means[t] / total if total > 0 else None) for t in LOSS_TERMS}}
    return out


def time_feature_std(time_ns, time_scale_ns: float) -> float:
    """Standard deviation of the time feature over hit tokens (§6 diagnostic)."""
    x = np.asarray(time_ns, dtype=np.float64) / float(time_scale_ns)
    return float(x.std())


# ======================================================================= readings (§7)
def reading_r2a(f_event_by_seed: dict) -> dict:
    values = [float(f_event_by_seed[s]) for s in sorted(f_event_by_seed)]
    if all(v <= F_EVENT_INVARIANT for v in values):
        cls = "event-invariant"
    elif all(v >= F_EVENT_DEPENDENT for v in values):
        cls = "event-dependent"
    else:
        cls = "intermediate"
    return {"reading": "R2-a", "class": cls, "f_event": {str(s): float(f_event_by_seed[s])
                                                         for s in sorted(f_event_by_seed)}}


def reading_r2b(posfree: dict, init: dict) -> dict:
    """``posfree[s] = {vertex, token_r2, f_event}``; ``init[s] = {vertex}`` (seed-matched)."""
    seeds = sorted(posfree)
    rec = [posfree[s]["vertex"] < init[s]["vertex"] and posfree[s]["token_r2"] >= TOKEN_R2_RECOVERED
           and posfree[s]["f_event"] >= F_EVENT_DEPENDENT for s in seeds]
    notrec = [posfree[s]["vertex"] >= init[s]["vertex"] or posfree[s]["token_r2"] < TOKEN_R2_RECOVERED
              for s in seeds]
    cls = "recovered" if all(rec) else ("not recovered" if all(notrec) else "unclear")
    return {"reading": "R2-b", "class": cls, "seeds": [int(s) for s in seeds],
            "per_seed": {str(s): {**{k: float(v) for k, v in posfree[s].items()},
                                  "init_vertex": float(init[s]["vertex"])} for s in seeds}}


def reading_r2c(diffs: dict) -> dict:
    """``diffs[s] = {"mean_difference_m": MAE - init, "ci95_m": [lo, hi]}`` (seed-matched)."""
    seeds = sorted(diffs)
    yes = [d["mean_difference_m"] < 0 and d["ci95_m"][1] < 0 for d in (diffs[s] for s in seeds)]
    no = [d["mean_difference_m"] >= 0 and d["ci95_m"][0] > 0 for d in (diffs[s] for s in seeds)]
    cls = "yes" if all(yes) else ("no" if all(no) else "unclear")
    return {"reading": "R2-c", "class": cls, "seeds": [int(s) for s in seeds],
            "per_seed": {str(s): diffs[s] for s in seeds}}


def kendall_tau(order_a, order_b) -> float:
    """Kendall tau between two orderings of the same items (no ties)."""
    items = list(order_a)
    if sorted(items) != sorted(order_b):
        raise ValueError("kendall_tau: orderings must hold the same items")
    ra = {x: i for i, x in enumerate(order_a)}
    rb = {x: i for i, x in enumerate(order_b)}
    conc = disc = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = (ra[items[i]] - ra[items[j]]) * (rb[items[i]] - rb[items[j]])
            conc += s > 0
            disc += s < 0
    n = len(items) * (len(items) - 1) / 2
    return float((conc - disc) / n)


def reading_r2d(vertex_by_arm: dict) -> dict:
    """Seed-mean vertex error per arm -> order (best first) and Kendall tau vs WCTE."""
    means = {arm: float(np.mean(list(v.values()))) for arm, v in vertex_by_arm.items()}
    order = sorted(WCTE_ORDER, key=lambda a: means[a])
    return {"reading": "R2-d", "order": order, "seed_mean_vertex_m": means,
            "wcte_order": list(WCTE_ORDER), "kendall_tau": kendall_tau(order, WCTE_ORDER)}


def reading_r2e(diff: dict, rf: dict, mae: dict) -> dict:
    """``mae_mse_rf`` - ``mae_mse`` (seed 9000), descriptive; confounding sentence rule."""
    out = {"reading": "R2-e", "vertex": diff,
           "token_r2_difference": rf["token_r2"] - mae["token_r2"],
           "f_event_difference": rf["f_event"] - mae["f_event"],
           "particle_difference": {k: rf["particle"][k] - mae["particle"][k]
                                   for k in rf.get("particle", {}) if k in mae.get("particle", {})
                                   and rf["particle"][k] is not None
                                   and mae["particle"][k] is not None}}
    lo, hi = diff["ci95_m"]
    excludes = lo > 0 or hi < 0
    out["confounded_by_receptive_field"] = bool(
        excludes and abs(diff["mean_difference_m"]) > ROUND1_MAE_SEED_SPREAD_M)
    out["by_m"] = float(diff["mean_difference_m"])
    return out


def floor_band(init_metrics: dict) -> dict:
    """Mean and range of the five ``init`` encoders per metric (descriptive)."""
    keys = sorted({k for m in init_metrics.values() for k in m})
    out = {}
    for k in keys:
        vals = [float(m[k]) for m in init_metrics.values() if m.get(k) is not None]
        if vals:
            out[k] = {"mean": float(np.mean(vals)), "min": float(min(vals)),
                      "max": float(max(vals)), "n": len(vals)}
    return out


# ======================================================================= §8 conditional seed 9002
def seed9002_triggers(readings: dict, vertex_by_arm: dict) -> list:
    """Arms that get seed 9002 on this detector, each with its trigger (§8)."""
    out = {}

    def add(arm, why):
        out.setdefault(arm, []).append(why)
    if readings["R2-b"]["class"] == "unclear":
        add("target_posfree", "R2-b unclear")
    if readings["R2-c"]["class"] == "unclear":
        add("mae_mse", "R2-c unclear")
    a = [float(v) for v in vertex_by_arm["mae_mse"].values()]
    b = [float(v) for v in vertex_by_arm["target_posfree"].values()]
    if max(min(a), min(b)) <= min(max(a), max(b)):
        add("mae_mse", "mae_mse / target_posfree vertex seed ranges overlap")
        add("target_posfree", "mae_mse / target_posfree vertex seed ranges overlap")
    if readings["R2-a"]["class"] == "intermediate":
        add("sigreg_jepa", "R2-a intermediate")
    return [{"arm": arm, "seed": CONDITIONAL_SEED, "triggers": why}
            for arm, why in sorted(out.items(), key=lambda kv: MAIN_ARMS.index(kv[0]))]


def plan_seed9002(triggers: list, remaining_h: float, train_h: dict, eval_h: float) -> list:
    """Train a triggered run only while the remaining cap holds it plus its evaluation."""
    left = float(remaining_h)
    plan = []
    for t in triggers:
        cost = float(train_h[t["arm"]]) + float(eval_h)
        if cost <= left:
            plan.append({**t, "decision": "train", "projected_h": cost})
            left -= cost
        else:
            plan.append({**t, "decision": "skip", "projected_h": cost,
                         "reason": f"remaining cap {left:.2f} h < {cost:.2f} h (run + evaluation)"})
    return plan


def hk_fallback(remaining_h: float, main_run_h: float, rf_run_h: float, eval_h: float) -> dict:
    """§8 fallback before HK training, in its fixed order."""
    full = 6 * main_run_h + rf_run_h + eval_h
    if remaining_h >= full:
        return {"step": 0, "runs": "six main runs + mae_mse_rf", "projected_h": full}
    no_rf = 6 * main_run_h + eval_h
    if remaining_h >= no_rf:
        return {"step": 1, "runs": "six main runs (mae_mse_rf on HK dropped)", "projected_h": no_rf}
    three = 3 * main_run_h + eval_h
    if remaining_h >= three:
        return {"step": 2, "runs": "three main runs, seed 9000 only", "projected_h": three}
    return {"step": 3, "runs": "none", "stop": "S1", "projected_h": three}


# ======================================================================= training
def run_name(detector: str, arm: str, seed: int, smoke: bool = False) -> str:
    return f"{detector}__{arm}__s{int(seed)}" + ("__smoke" if smoke else "")


# ======================================================================= budget enforcement (§8, §9)
class StopError(RuntimeError):
    """A runbook §9 stop; ``stop`` is its id (S1..S9)."""

    def __init__(self, stop: str, reason: str):
        super().__init__(f"{stop}: {reason}")
        self.stop, self.reason = stop, reason


#: Design-time projections of runbook §8 (replaced by smoke measurements once they exist):
#: main-arm training per detector (upper ends) / 6 runs, ``mae_mse_rf``, evaluation 5 h / 3.
DESIGN_MAIN_RUN_H = {"sk": 17.0 / 6, "juno": 15.0 / 6, "hk": 30.0 / 6}
DESIGN_RF_RUN_H = {"sk": 2.0, "juno": 1.7, "hk": 4.0}
DESIGN_EVAL_H = 5.0 / 3
DESIGN_EVAL_MODEL_H = DESIGN_EVAL_H / 12  # one model: 7 trained + 5 init per detector
CPU_GENERATION_CAP_H = 48.0
CPU_GENERATION_USED_ROUND1_H = 18.43
CEPH_QUOTA_ROOT = Path("/ceph/dwong")
CEPH_FREE_ENTRIES_FLOOR = 30_000


#: GPU profiles (addendum RUNBOOK_2026-11-12_R2_JUNO_TITAN.md): SK (and HK) on the deepthought2
#: L40S under the 60 h line; JUNO on Deepthought's GTX Titans under its own 100 h line. The
#: process profile comes from ``R2_GPU_PROFILE`` (default ``l40s``: round-2 behaviour unchanged).
TITAN_ADDENDUM_SHA256 = "0a8d76e980e94007699f920dc809a00767e7843f6611840a8c614cdf7d8d277f"  # v2 (v1 f381955d...)
GPU_PROFILES = {
    "l40s": {"line": LEDGER_LINE, "cap_h": GPU_CAP_H, "file": "RESOURCE_LEDGER_R2.json",
             "detectors": DETECTOR_ORDER, "mem_limit_gib": GPU_MEMORY_FRACTION_CAP * L40S_MEMORY_GIB,
             "gpu": "NVIDIA L40S (deepthought2)"},
    "titan": {"line": "R2_titan_deepthought", "cap_h": 100.0, "file": "RESOURCE_LEDGER_R2_TITAN.json",
              "detectors": ("juno",), "mem_limit_gib": 11.0,
              "gpu": "NVIDIA GeForce GTX TITAN X (Deepthought)"},
}
DETECTOR_PROFILE = {"sk": "l40s", "hk": "l40s", "juno": "titan"}
#: Titan evaluation reservation: the measured Titan WCTE-regression evaluation time per model,
#: scaled by the token count (N_PMT_det / 1,995), x 12 models (7 trained + 5 init).
WCTE_N_PMT = 1995


def gpu_profile_name() -> str:
    name = os.environ.get("R2_GPU_PROFILE", "l40s")
    if name not in GPU_PROFILES:
        raise ValueError(f"unknown R2_GPU_PROFILE {name!r}")
    return name


def gpu_profile() -> dict:
    return GPU_PROFILES[gpu_profile_name()]


def cap_h() -> float:
    return float(gpu_profile()["cap_h"])


def ledger_path(root) -> Path:
    return Path(root) / "ledger" / gpu_profile()["file"]


def ledger_load(root) -> dict:
    p = ledger_path(root)
    if p.is_file():
        return json.loads(p.read_text())
    prof = gpu_profile()
    if gpu_profile_name() == "l40s":
        return {"line": LEDGER_LINE, "cap_l40s_h": GPU_CAP_H, "entries": [], "cpu_entries": [],
                "hk_fallback": None}
    return {"line": prof["line"], "cap_h": prof["cap_h"], "gpu": prof["gpu"], "entries": [],
            "cpu_entries": [], "addendum_sha256": TITAN_ADDENDUM_SHA256}


def eval_total_h(root, detector: str) -> float:
    """Evaluation reservation of one detector under the current profile (§8 / Titan addendum)."""
    if gpu_profile_name() == "l40s":
        return DESIGN_EVAL_H
    wcte = [float(e["wall_s"]) for e in ledger_load(root)["entries"]
            if e.get("kind") == "eval" and e.get("detector") == "wcte" and e.get("wall_s")]
    if not wcte:
        return 12 * DESIGN_EVAL_MODEL_H * 10.0  # before the Titan measurement: 10x the L40S design
    scale = rg.DESIGN_TABLE[detector]["n_pmt"] / WCTE_N_PMT
    return 12 * max(wcte) * scale / 3600.0


def eval_model_h(root, detector: str) -> float:
    return eval_total_h(root, detector) / 12.0


def _ledger_update(root, fn) -> dict:
    """Read-modify-write the ledger under an exclusive lock (two GPU jobs may finish together)."""
    import fcntl
    p = check_write_path(ledger_path(root))
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = ledger_load(root)
        fn(ledger)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")
        tmp.replace(p)
    return ledger


def ledger_add(root, entry: dict, *, cpu: bool = False) -> dict:
    stamped = {**entry, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if not cpu and gpu_profile_name() != "l40s":
        stamped["gpu"] = gpu_profile()["gpu"]
    return _ledger_update(root, lambda l: l["cpu_entries" if cpu else "entries"].append(stamped))


def booked_gpu_h(ledger) -> float:
    return float(sum(float(e.get("gpu_h", 0.0)) for e in ledger["entries"]))


def run_complete(ledger, detector: str, arm: str, seed: int) -> bool:
    return any(e.get("kind") == "train" and e.get("complete") and e["detector"] == detector
               and e["arm"] == arm and int(e["seed"]) == int(seed) for e in ledger["entries"])


def smoke_projection_h(root, detector: str, arm: str, card_gib: float | None = None):
    """10,000-step projection from the detector/arm smoke at the chosen micro-batch, or None."""
    smokes = []
    for p in sorted((Path(root) / "smokes").glob(f"{detector}__{arm}__s*__smoke*/run_record.json")):
        smokes.append(json.loads(p.read_text()))
    if not smokes:
        return None
    try:
        micro = choose_micro_batch(smokes, card_gib)
    except RuntimeError:
        return None
    rec = [s for s in smokes if int(s["micro_batch"]) == micro][-1]
    return {"micro_batch": micro, "seconds_per_step": float(rec["seconds_per_step"]),
            "projected_h": float(rec["seconds_per_step"]) * STEPS / 3600.0,
            "peak_memory_gib": rec["peak_memory_gib"]}


def run_projection_h(root, detector: str, arm: str) -> float:
    s = smoke_projection_h(root, detector, arm)
    if s is not None:
        return s["projected_h"]
    return DESIGN_RF_RUN_H[detector] if arm == RF_ARM else DESIGN_MAIN_RUN_H[detector]


def mark_stopped(root, detector: str, stop: str, reason: str) -> dict:
    """Record a §9 detector stop (e.g. S2): its runs and evaluation are no longer owed."""
    return _ledger_update(root, lambda l: l.setdefault("stopped", {}).__setitem__(
        detector, {"stop": stop, "reason": reason}))


def mark_run_stopped(root, detector: str, arm: str, seed: int, stop: str, reason: str) -> dict:
    """Record a §9 stop for one planned run (e.g. S1 for ``mae_mse_rf`` when the cap cannot
    hold it after the main runs): the run is no longer reserved and its training is refused."""
    return _ledger_update(root, lambda l: l.setdefault("run_stops", []).append(
        {"detector": detector, "arm": arm, "seed": int(seed), "stop": stop, "reason": reason,
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))


def run_stopped(ledger, detector: str, arm: str, seed: int):
    return next((r for r in ledger.get("run_stops") or [] if r["detector"] == detector
                 and r["arm"] == arm and int(r["seed"]) == int(seed)), None)


def resume_stopped(root, detector: str, authority: str) -> dict:
    """Lift a recorded detector stop by a later author decision (e.g. the Gate V addendum):
    the original stop record is kept under ``resumed`` with the authority that lifted it."""
    def fn(l):
        rec = (l.get("stopped") or {}).pop(detector, None)
        if rec is None:
            raise ValueError(f"{detector} is not stopped")
        l.setdefault("resumed", {})[detector] = {
            "original_stop": rec, "authority": authority,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return _ledger_update(root, fn)


def is_stopped(ledger, detector: str) -> bool:
    return detector in (ledger.get("stopped") or {})


def planned_runs(ledger, detector: str) -> list:
    """The §5.1 runs of a detector, cut by a recorded HK fallback (§8); none if stopped."""
    if is_stopped(ledger, detector):
        return []
    runs = [(a, s) for a in MAIN_ARMS for s in MAIN_SEEDS] + [(RF_ARM, 9000)]
    fb = ledger.get("hk_fallback") if detector == "hk" else None
    if fb is not None:
        if fb["step"] >= 1:
            runs = [r for r in runs if r[0] != RF_ARM]
        if fb["step"] >= 2:
            runs = [r for r in runs if r[1] == 9000]
        if fb["step"] >= 3:
            runs = []
    return [r for r in runs if run_stopped(ledger, detector, *r) is None]


def eval_booked_h(ledger, detector: str) -> float:
    return float(sum(float(e.get("gpu_h", 0.0)) for e in ledger["entries"]
                     if e.get("kind") == "eval" and e["detector"] == detector))


def reserved_gpu_h(root, ledger, *, exclude=None, hk_minimal: bool = False) -> float:
    """GPU hours still owed to higher-priority work: every planned main/rf run not yet
    complete and the remaining evaluation of each detector (§8 priority)."""
    total = 0.0
    for det in gpu_profile()["detectors"]:
        runs = planned_runs(ledger, det)
        if det == "hk" and hk_minimal and ledger.get("hk_fallback") is None:
            runs = [(a, 9000) for a in MAIN_ARMS]  # the smallest HK plan before S1 (§8 step 2)
        for arm, seed in runs:
            if exclude == (det, arm, seed) or run_complete(ledger, det, arm, seed):
                continue
            total += run_projection_h(root, det, arm)
        if not is_stopped(ledger, det):
            total += max(0.0, eval_total_h(root, det) - eval_booked_h(ledger, det))
    return total


def hk_fallback_decision(root) -> dict:
    """§8: before HK training, re-project HK from the measured SK and JUNO step times and
    record the fallback step in the ledger (computed once)."""
    ledger = ledger_load(root)
    if ledger.get("hk_fallback") is not None:
        return ledger["hk_fallback"]
    n_pmt = {d: rg.DESIGN_TABLE[d]["n_pmt"] for d in DETECTOR_ORDER}
    scaled = []
    for det in ("sk", "juno"):
        for arm in MAIN_ARMS:
            s = smoke_projection_h(root, det, arm)
            if s is not None:
                scaled.append(s["projected_h"] * n_pmt["hk"] / n_pmt[det])
    hk_meas = [p["projected_h"] for p in (smoke_projection_h(root, "hk", a) for a in MAIN_ARMS)
               if p is not None]
    main_h = max(scaled + hk_meas) if (scaled or hk_meas) else DESIGN_MAIN_RUN_H["hk"]
    rf_s = smoke_projection_h(root, "hk", RF_ARM)
    rf_h = rf_s["projected_h"] if rf_s else main_h * 1.45
    other = 0.0
    for det in ("sk", "juno"):
        for arm, seed in planned_runs(ledger, det):
            if not run_complete(ledger, det, arm, seed):
                other += run_projection_h(root, det, arm)
        if not is_stopped(ledger, det):
            other += max(0.0, DESIGN_EVAL_H - eval_booked_h(ledger, det))
    remaining = GPU_CAP_H - booked_gpu_h(ledger) - other
    decision = hk_fallback(remaining, main_h, rf_h, DESIGN_EVAL_H)
    decision.update({"remaining_h": remaining, "hk_main_run_h": main_h, "hk_rf_run_h": rf_h,
                     "projection_inputs": {"sk_juno_scaled_by_n_pmt": scaled,
                                           "hk_smoke": hk_meas},
                     "rule": "max of SK/JUNO smoke step times scaled by N_PMT and the HK smoke"})
    _ledger_update(root, lambda l: l.__setitem__("hk_fallback", decision))
    return decision


def seed9002_decision(root, detector: str, arm: str) -> dict:
    """Derive the §8 seed-9002 plan for this detector from its evaluated readings."""
    readings = detector_readings(root, detector)
    if readings.get("status") != "evaluated":
        raise ValueError(f"{detector}: readings incomplete; seed 9002 needs the evaluation first")
    ledger = ledger_load(root)
    triggers = seed9002_triggers(readings, {a: {s: readings["metrics"][f"{a}:{s}"]["vertex_full_m"]
                                                for s in MAIN_SEEDS} for a in WCTE_ORDER})
    remaining = cap_h() - booked_gpu_h(ledger) - reserved_gpu_h(root, ledger)
    done = [t for t in triggers if run_complete(ledger, detector, t["arm"], CONDITIONAL_SEED)]
    pending = [t for t in triggers if t not in done]
    plan = plan_seed9002(pending, remaining,
                         {t["arm"]: run_projection_h(root, detector, t["arm"]) for t in pending},
                         2 * (DESIGN_EVAL_MODEL_H if gpu_profile_name() == "l40s"
                              else eval_model_h(root, detector)))  # the arm + its init 9002
    entry = next((p for p in plan if p["arm"] == arm), None)
    return {"triggers": triggers, "plan": plan, "remaining_h": remaining, "entry": entry}


def gpu_precheck(root, detector: str, arm: str, seed: int, projected_h: float) -> dict:
    """Refuse a run the 60 h cap cannot hold after the §8 fallback (S1)."""
    ledger = ledger_load(root)
    booked = booked_gpu_h(ledger)
    reserved = reserved_gpu_h(root, ledger, exclude=(detector, arm, seed),
                              hk_minimal=(detector != "hk"))
    if int(seed) == CONDITIONAL_SEED:
        reserved = reserved_gpu_h(root, ledger, exclude=(detector, arm, seed))
    total = booked + float(projected_h) + reserved
    cap = cap_h()
    if total > cap:
        stop = "S1" if gpu_profile_name() == "l40s" else "T-S1"
        raise StopError(stop, f"booked {booked:.2f} + this run {projected_h:.2f} + still owed "
                              f"{reserved:.2f} = {total:.2f} {gpu_profile()['line']} h > cap {cap}")
    return {"booked_h": booked, "projected_h": float(projected_h), "reserved_h": reserved,
            "total_h": total, "cap_h": cap, "ledger_line": gpu_profile()["line"]}


def ceph_free_entries(quota_root=CEPH_QUOTA_ROOT) -> int:
    q = str(quota_root)
    return (int(os.getxattr(q, "ceph.quota.max_files")) - int(os.getxattr(q, "ceph.dir.rfiles"))
            - int(os.getxattr(q, "ceph.dir.rsubdirs")))


def storage_check(disk_bytes: float, *, entries_needed: int = 500,
                  work_root="/work/dwong", ceph_root=CEPH_QUOTA_ROOT) -> dict:
    """S5: free bytes >= 2 x the projection on /work and /ceph; Ceph entries >= 30,000 after."""
    import shutil
    out = {"projection_bytes": float(disk_bytes)}
    ok = True
    for label, root in (("work", work_root), ("ceph", ceph_root)):
        free = shutil.disk_usage(str(root)).free
        out[f"{label}_free_bytes"] = int(free)
        ok = ok and free >= 2.0 * float(disk_bytes)
    try:
        entries = ceph_free_entries(ceph_root)
    except OSError:
        entries = None
    out["ceph_free_entries"] = entries
    out["entries_needed"] = int(entries_needed)
    ok = ok and entries is not None and entries - int(entries_needed) >= CEPH_FREE_ENTRIES_FLOOR
    out["pass"], out["stop"] = bool(ok), (None if ok else "S5")
    return out


def cpu_check(root, projected_wall_h: float) -> dict:
    """S7: the generation projection must fit the remaining round-1 generation cap."""
    ledger = ledger_load(root)
    used = float(sum(float(e.get("wall_h", 0.0)) for e in ledger["cpu_entries"]
                     if e.get("kind") == "generation"))
    remaining = CPU_GENERATION_CAP_H - CPU_GENERATION_USED_ROUND1_H - used
    ok = float(projected_wall_h) <= remaining
    return {"projected_wall_h": float(projected_wall_h), "round2_used_wall_h": used,
            "remaining_wall_h": remaining, "pass": bool(ok), "stop": None if ok else "S7"}


#: Addendum RUNBOOK_2026-11-12_R2_GATEV_CEILING.md (author, 12 Nov 2026 10:08): Gate V is
#: informational for SK, JUNO and HK; S2 no longer applies. Gate V is still required in the
#: X0 record (it is reported) but its result no longer blocks generation.
ADDENDUM_GATEV_SHA256 = "d237ec2b3059f5359b51e20e2047b78454264b214cd73c8699d7d7030cca582c"
GATE_V_INFORMATIONAL = True


def x0_clearance(x0: dict) -> dict:
    """The X0 record must show no stop before generation; returns the yield range.

    S3 is judged on the trigger efficiencies only (§9), so an X0 record written before that
    fix (whose ``trigger.stop`` also counted dark-only windows) is re-judged from its
    efficiencies. Gate V is informational (addendum)."""
    keys = ("geometry", "yield_range", "trigger", "gate_v", "storage", "cpu")
    missing = [k for k in keys if x0.get(k) is None]
    if missing:
        raise ValueError(f"X0 record lacks {missing}")
    stops = {k: (x0.get(k) or {}).get("stop") for k in keys}
    eff = (x0["trigger"] or {}).get("efficiency")
    if eff:
        stops["trigger"] = None if all(float(v) >= 0.99 for v in eff.values()) else "S3"
    if GATE_V_INFORMATIONAL:
        stops["gate_v"] = None
    fired = {k: v for k, v in stops.items() if v}
    if fired:
        raise StopError(sorted(fired.values())[0], f"X0 stop(s) {fired}; no generation")
    return {"y_lo": float(x0["yield_range"]["y_lo"]), "y_hi": float(x0["yield_range"]["y_hi"])}


def train_r2(*, detector: str, arm: str, seed: int, h5_path, out_root, steps: int = STEPS,
             micro_batch: int = 8, device: str = "cuda", smoke: bool = False,
             resume: bool = True, log=print) -> dict:
    """One round-2 training run with the round-1 optimiser, loop and checkpoints.

    A main run needs its detector/arm smoke (projection, micro-batch), passes the §8/§9
    budget checks (S1, HK fallback, seed-9002 plan), runs under the 1.5 x projection wall cap
    and books its GPU hours in the ledger.
    """
    import hashlib
    import signal
    import torch
    from . import lucid_run as lr
    from . import r2_generator as gen
    from . import r2_store as rs
    budget = None
    if DETECTOR_PROFILE.get(detector) != gpu_profile_name():
        raise ValueError(f"{detector} trains under GPU profile {DETECTOR_PROFILE.get(detector)!r}, "
                         f"this process is {gpu_profile_name()!r} (set R2_GPU_PROFILE)")
    if is_stopped(ledger_load(out_root), detector):
        raise StopError(ledger_load(out_root)["stopped"][detector]["stop"],
                        f"{detector} is stopped; no training")
    rs_ = None if smoke else run_stopped(ledger_load(out_root), detector, arm, seed)
    if rs_ is not None:
        raise StopError(rs_["stop"], f"{detector}/{arm}/{seed} is stopped: {rs_['reason']}")
    if not smoke:
        triggered = ()
        if int(seed) == CONDITIONAL_SEED and arm in MAIN_ARMS:
            decision = seed9002_decision(out_root, detector, arm)
            entry = decision["entry"]
            if entry is None or entry["decision"] != "train":
                raise ValueError(f"seed 9002 for {arm} on {detector} is not planned: {entry}")
            triggered = (arm,)
        allowed_training(arm, seed, detector=detector, triggered=triggered)
        if int(steps) != STEPS:
            raise ValueError("a main run trains the §2.1 10,000 steps")
    elif int(steps) != SMOKE_STEPS:
        raise ValueError("a smoke trains 20 steps (§5.2)")
    if arm not in TRAINED_ARMS:
        raise ValueError(f"{arm} is not a trained arm")
    out_root = check_write_path(out_root)
    if not smoke:
        proj = smoke_projection_h(out_root, detector, arm)
        if proj is None:
            raise ValueError(f"{detector}/{arm}: no smoke projection yet (§5.2)")
        if int(micro_batch) != proj["micro_batch"]:
            raise ValueError(f"micro-batch must be the smoke choice {proj['micro_batch']}")
        if detector == "hk":
            fb = hk_fallback_decision(out_root)
            if fb.get("stop"):
                raise StopError("S1", f"HK fallback exhausted: {fb}")
            if int(seed) != CONDITIONAL_SEED and (arm, int(seed)) not in planned_runs(
                    ledger_load(out_root), "hk"):
                raise ValueError(f"HK fallback step {fb['step']} excludes {arm} seed {seed}")
        budget = gpu_precheck(out_root, detector, arm, seed, proj["projected_h"])
        prior_h = sum(float(e.get("gpu_h", 0.0)) for e in ledger_load(out_root)["entries"]
                      if e.get("kind") == "train" and e["detector"] == detector
                      and e["arm"] == arm and int(e["seed"]) == int(seed))
        # the cap counts every invocation of this run (a resumed run keeps its earlier hours)
        wall_cap_s = WALL_CAP_FACTOR * proj["projected_h"] * 3600.0 - prior_h * 3600.0
        if wall_cap_s <= 0:
            raise ValueError(f"{detector}/{arm}/{seed}: wall cap already used ({prior_h:.2f} h)")
    run_dir = check_write_path(Path(out_root) / ("smokes" if smoke else "runs")
                               / (run_name(detector, arm, seed, smoke)
                                  + (f"__mb{int(micro_batch)}" if smoke else "")))
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    geometry = rg.load_r2_geometry(detector)
    rule = gen.time_rule_for(geometry)
    scaling = rs.scaling_for(geometry, rule.time_scale_ns)
    cfg = run_config()
    arm_module = build_r2_arm(arm, geometry, seed, scaling=scaling, run_cfg=cfg).to(device)
    optimizers, optimiser, scheduler = lr.build_optimizers(arm_module, cfg)
    cohort = rs.open_cohort(h5_path, "dev_train", truth_fields=())
    loader = rs.R2TrainingLoader(cohort, geometry, seed=int(seed), scaling=scaling,
                                 n_min=int(cfg["views"]["n_min"]), batch=BATCH)
    if BATCH % int(micro_batch):
        raise ValueError("micro_batch must divide 32")
    identity = {"run_id": run_name(detector, arm, seed, smoke), "cell_id": f"r2_{detector}_within",
                "arm": arm, "seed": int(seed), "beta": None, "steps_scheduled": int(steps),
                "gradient_clip": optimiser.get("gradient_clip")}
    r2cfg = {"extension": EXTENSION, "runbook_sha256": RUNBOOK_SHA256, "detector": detector,
             "arm": arm, "encoder": dataclasses.asdict(encoder_config_for(arm, geometry)),
             "time_scale_ns": rule.time_scale_ns, "training": cfg["training"],
             "h5_sha256_prefix": None}
    config_sha = hashlib.sha256(json.dumps(r2cfg, sort_keys=True).encode()).hexdigest()
    checkpoint = run_dir / "checkpoint.pt"
    if smoke and checkpoint.exists():
        raise RuntimeError(f"{run_dir}: a smoke is never resumed (§5.2)")
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    capped = {"hit": False}

    def wall_cap(step):
        # The round-1 loop checkpoints and stops cleanly on SIGTERM; the cap uses that path.
        if not smoke and not capped["hit"] and time.time() - t0 > wall_cap_s:
            capped["hit"] = True
            os.kill(os.getpid(), signal.SIGTERM)
    try:
        result = lr.train_loop(arm=arm_module, loader=loader, steps=int(steps),
                               optimizers=optimizers, checkpoint_path=checkpoint,
                               identity=identity, config_sha=config_sha,
                               checkpoint_minutes=float(cfg["training"]["checkpoint_minutes"]),
                               checkpoint_every=int(cfg["training"]["checkpoint_every_steps"]),
                               resume=(checkpoint if (resume and checkpoint.exists() and not smoke)
                                       else None),
                               log_path=run_dir / "train_log.jsonl",
                               accum_steps=BATCH // int(micro_batch), log_every=1,
                               scheduler=scheduler, step_callback=wall_cap)
    except BaseException as error:  # a crashed run still books its GPU time
        ledger_add(out_root, {"kind": "smoke" if smoke else "train", "detector": detector,
                              "arm": arm, "seed": int(seed), "micro_batch": int(micro_batch),
                              "gpu_h": (time.time() - t0) / 3600.0, "complete": False,
                              "crashed": repr(error)[:300], "precheck": budget})
        raise
    wall = time.time() - t0
    complete = (not result["interrupted"]) and int(result["step"]) == int(steps)
    ledger_add(out_root, {"kind": "smoke" if smoke else "train", "detector": detector,
                          "arm": arm, "seed": int(seed), "micro_batch": int(micro_batch),
                          "gpu_h": wall / 3600.0, "wall_s": wall, "steps": int(result["step"]),
                          "complete": bool(complete), "wall_cap_hit": capped["hit"],
                          "precheck": budget})
    peak = (torch.cuda.max_memory_allocated() / 2 ** 30) if device.startswith("cuda") else None
    record = {"extension": EXTENSION, "detector": detector, "arm": arm, "seed": int(seed),
              "smoke": bool(smoke), "steps": int(result["step"]), "steps_scheduled": int(steps),
              "micro_batch": int(micro_batch), "accumulation": BATCH // int(micro_batch),
              "wall_seconds": wall, "seconds_per_step": wall / max(1, int(result["step"]) -
                                                                  int(result["resumed_from_step"] or 0)),
              "peak_memory_gib": peak, "device": device, "optimizer": optimiser,
              "config_sha256": config_sha, "encoder": r2cfg["encoder"],
              "time_scale_ns": rule.time_scale_ns, "interrupted": result["interrupted"],
              "resumed_from_step": result["resumed_from_step"], "code_sha": lb.current_code_sha(),
              "checkpoint": str(checkpoint), "registered": False, "complete": bool(complete),
              "wall_cap_hit": capped["hit"], "budget_precheck": budget,
              "gpu_model": (torch.cuda.get_device_name() if device.startswith("cuda") else None),
              "gpu_profile": gpu_profile_name(), "ledger_line": gpu_profile()["line"]}
    (run_dir / "run_record.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    log(json.dumps({k: record[k] for k in ("detector", "arm", "seed", "steps", "wall_seconds",
                                           "peak_memory_gib")}))
    return record


def choose_micro_batch(smokes: list, card_gib: float | None = None) -> int:
    """Largest of 8, 4, 2, 1 whose smoke peak stays <= 45 % of the L40S (§5.2), or <= 11 GiB
    on the Titans (Titan addendum §2.5); ``card_gib`` keeps the 45 % rule for a given card."""
    limit = (GPU_MEMORY_FRACTION_CAP * float(card_gib) if card_gib is not None
             else float(gpu_profile()["mem_limit_gib"]))
    ok = [int(s["micro_batch"]) for s in smokes
          if s.get("peak_memory_gib") is not None and float(s["peak_memory_gib"]) <= limit]
    for m in MICRO_BATCH_CHOICES:
        if m in ok:
            return m
    raise RuntimeError("no micro-batch fits 45 % of the L40S")


# ======================================================================= evaluation (§6)
def _truth_for_run(geometry, cohort_obj, cohort_name: str, index: int) -> dict:
    from . import lucid_arms as la
    from . import r2_generator as gen
    run = gen.cohort_runs(cohort_name)[index]
    if run.run_id != cohort_obj.run_ids[index]:
        raise ValueError("cohort run order differs from the generator's")
    t = cohort_obj.run_truth(index)
    truth = la.truth_from_state(geometry, run.state(geometry), vertex_m=t["vertex_m"],
                                yield_photons=t["yield_photons"])
    truth["event_ids"] = t["event_ids"]
    return truth, t


def stream_forms(model, geometry, scaling, probe_sets: dict, *, device, chunk_events: int = 8,
                 tokens_per_event: int = 32, sample_seed: int = 0, half: bool = True,
                 particles: bool = True) -> dict:
    """The round-1 ``ext_readout.stream_model_forms`` (full input + half pass) on HDF5 cohorts.

    ``probe_sets = {"train": R2Cohort, "test": R2Cohort}`` with names recorded in
    ``cohort.name``.  Also streams the full-input event means of the muon/electron events
    (the particle tasks of §6).
    """
    import torch
    from . import ext_readout as er
    from . import lucid_arms as la
    from . import lucid_run as lr
    from . import r2_store as rs
    from .lucid_views import ViewMasker
    module_index = np.asarray(geometry.module_index, np.int64)
    half_encoder = er.HalfPassEncoder(model, er.within_target_mask(geometry)).to(device) if half else None
    out = {}
    for which in ("train", "test"):
        cohort = probe_sets[which]
        base_name = cohort.name
        rng = np.random.default_rng(sample_seed)
        acc = {"full": [], "half": [], "runs": [], "event_ids": [], "vertex": [], "log_yield": [],
               "tokens": [], "token_y": [], "token_runs": []}
        part = {"events": [], "run_ids": [], "event_ids": [], "particle": [], "direction": [],
                "deposited_MeV": [], "contained": [], "n_hit": [], "vertex": []}
        layout = er.layout_state(geometry.n_pmt, 0)
        gain_acc = la.ProbeAccumulator("module_gain", "run_mean")
        for i in range(cohort.n_runs):
            truth, raw = _truth_for_run(geometry, cohort, base_name, i)
            run_id = cohort.run_ids[i]
            store = cohort.run_store(i)
            domain = np.asarray(raw["domain"]).astype(str)
            n_hit_all = np.diff(store.offsets)
            photon_ids = raw["event_ids"][domain == lb.DOMAIN_PHOTON]
            partial = None
            for start in range(0, photon_ids.size, chunk_events):
                ids = photon_ids[start:start + chunk_events]
                batch = rs.tokenise_sparse(store, geometry, event_ids=ids, scaling=scaling,
                                           masker=ViewMasker(geometry, "within", seed=0, n_min=0))
                batch = lr._to_device(batch, torch.device(device))
                sel = la._select_truth(truth, np.asarray(batch.event_ids))
                hit = np.asarray(batch.hit_flags, bool)
                log_charge = np.asarray(batch.log_charge, float)
                with torch.no_grad():
                    tokens = model.encode(batch.features).to(torch.float32).cpu().numpy()
                    half_tokens = (half_encoder.encode(batch.features).to(torch.float32).cpu().numpy()
                                   if half else None)
                acc["full"].append(la._mean_tokens(tokens, None).astype(np.float32))
                if half:
                    acc["half"].append(la._mean_tokens(half_tokens, None).astype(np.float32))
                acc["runs"].append(np.full(len(batch.event_ids), run_id, dtype=object))
                acc["event_ids"].append(np.asarray(batch.event_ids))
                acc["vertex"].append(np.asarray(sel["vertex"], float))
                acc["log_yield"].append(np.asarray(sel["log_yield"], float))
                if layout["dim"] == 0:
                    layout = er.layout_state(geometry.n_pmt, tokens.shape[-1])
                er.layout_add(layout, tokens)
                for e in range(hit.shape[0]):
                    positions = np.flatnonzero(hit[e])
                    if positions.size == 0:
                        continue
                    take = min(int(tokens_per_event), int(positions.size))
                    pick = (rng.choice(positions, size=take, replace=False)
                            if positions.size > take else positions)
                    acc["tokens"].append(tokens[e, pick])
                    acc["token_y"].append(log_charge[e, pick])
                    acc["token_runs"].append(np.full(take, run_id, dtype=object))
                data = la.RunProbeData(run_id=run_id, event_ids=np.asarray(batch.event_ids),
                                       tokens=tokens, context_tokens=None, raw=None, hit=hit,
                                       hidden=np.asarray(batch.hidden.cpu(), bool),
                                       module_index=module_index, truth=sel)
                p = la.probe_run_mean_partial(data, "module_gain")
                partial = p if partial is None else la._merge_run_mean_partial(partial, p)
            if partial is not None:
                features, targets, _key, names = la.finalise_run_mean(partial)
                if features.shape[0]:
                    gain_acc.target_names = names
                    gain_acc.add(run_id, features, targets)
            if not particles:
                continue
            particle_ids = raw["event_ids"][np.isin(domain, lb.PARTICLE_DOMAINS)]
            index = {int(e): k for k, e in enumerate(raw["event_ids"])}
            for start in range(0, particle_ids.size, chunk_events):
                ids = particle_ids[start:start + chunk_events]
                batch = rs.tokenise_sparse(store, geometry, event_ids=ids, scaling=scaling,
                                           masker=ViewMasker(geometry, "within", seed=0, n_min=0))
                feats = batch.features.to(device)
                with torch.no_grad():
                    tokens = model.encode(feats).to(torch.float32).cpu().numpy()
                pos = np.asarray([index[int(e)] for e in batch.event_ids])
                part["events"].append(la._mean_tokens(tokens, None).astype(np.float32))
                part["run_ids"].append(np.full(len(pos), run_id, dtype=object))
                part["event_ids"].append(np.asarray(batch.event_ids))
                part["particle"].append(np.asarray(raw["particle"]).astype(str)[pos])
                part["direction"].append(np.asarray(raw["direction_m"], float)[pos])
                part["deposited_MeV"].append(np.asarray(raw["deposited_MeV"], float)[pos])
                part["contained"].append(np.asarray(raw["contained"], bool)[pos])
                part["n_hit"].append(n_hit_all[pos])
                part["vertex"].append(np.asarray(raw["vertex_m"], float)[pos])
        cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)
        out[which] = {k: cat(v) for k, v in acc.items()}
        out[which]["layout"] = er.layout_fractions(layout) if layout["dim"] else None
        out[which]["module_gain"] = gain_acc
        out[which]["particle"] = {k: cat(v) for k, v in part.items()}
    return out


def _vertex_fit(train_x, train_y, train_runs, test_x, test_y, test_runs) -> dict:
    from . import lucid_arms as la
    from . import lucid_run as lr
    ds = lambda x, y, r: la.ProbeDataset(kind="vertex", form="full_input", x=np.asarray(x, float),
                                         y=np.asarray(y, float),
                                         run_ids=np.asarray(r, dtype=object),
                                         target_names=("x", "y", "z"), meta={})
    fitted = la.fit_probe(ds(train_x, train_y, train_runs))
    pred = la.probe_predictions(fitted, ds(test_x, test_y, test_runs))
    scored = lr._vertex_error(pred, test_y)
    errors = np.linalg.norm(np.asarray(pred, float) - np.asarray(test_y, float), axis=1)
    return {"resolution_m": scored.get("resolution"), "median_m": scored.get("median"),
            "n": scored.get("n"), "errors": errors}


def metrics_from_streams(s: dict, geometry) -> tuple[dict, dict]:
    """§6 metrics for one model; returns (metrics, per-event arrays for the bootstrap)."""
    from . import ext_followup as ef
    from . import ext_readout as er
    from . import lucid_arms as la
    tr, te = s["train"], s["test"]
    full = _vertex_fit(tr["full"], tr["vertex"], tr["runs"], te["full"], te["vertex"], te["runs"])
    halfv = (_vertex_fit(tr["half"], tr["vertex"], tr["runs"], te["half"], te["vertex"], te["runs"])
             if np.asarray(tr["half"]).size else None)
    ly_ds = lambda d: la.ProbeDataset(kind="log_yield", form="full_input",
                                      x=np.asarray(d["full"], float),
                                      y=np.asarray(d["log_yield"], float)[:, None],
                                      run_ids=np.asarray(d["runs"], dtype=object),
                                      target_names=("log_yield",), meta={})
    pred = la.probe_predictions(la.fit_probe(ly_ds(tr)), ly_ds(te))
    ly_err = np.asarray(pred, float).ravel() - np.asarray(te["log_yield"], float)
    token = er.ridge_r2(tr["tokens"], tr["token_y"], te["tokens"], te["token_y"])
    rank = er.effective_rank(te["full"])
    gain = er.module_gain_r2({"train": {"run_mean": tr["module_gain"]},
                              "test": {"run_mean": te["module_gain"]}}, "full_input")
    rf = fiducial = rg.fiducial_radius_m(geometry)
    metrics = {"vertex_full_m": full["resolution_m"], "vertex_full_median_m": full["median_m"],
               "vertex_full_over_fiducial_radius": full["resolution_m"] / fiducial,
               "vertex_half_m": None if halfv is None else halfv["resolution_m"],
               "vertex_half_over_fiducial_radius": (None if halfv is None
                                                    else halfv["resolution_m"] / fiducial),
               "fiducial_radius_m": rf, "n_photon_test": int(full["n"] or 0),
               "light_yield_rms": float(np.sqrt(np.mean(ly_err[np.isfinite(ly_err)] ** 2))),
               "token_log_charge_r2": token.get("r2"), "token_ridge_alpha": token.get("alpha"),
               "f_layout": (te["layout"] or {}).get("f_layout"),
               "f_event": (te["layout"] or {}).get("f_event"),
               "effective_rank": rank.get("effective_rank"),
               "module_gain_r2": gain.get("value")}
    p_tr, p_te = tr["particle"], te["particle"]
    particle = {}
    if np.asarray(p_tr["events"]).size and np.asarray(p_te["events"]).size:
        y_tr = (np.asarray(p_tr["particle"]).astype(str) == lb.DOMAIN_MUON).astype(int)
        y_te = (np.asarray(p_te["particle"]).astype(str) == lb.DOMAIN_MUON).astype(int)
        auc = ef._logistic_auc(np.asarray(p_tr["events"], float), y_tr, p_tr["run_ids"],
                               np.asarray(p_te["events"], float), y_te)
        auc.pop("probability", None)
        particle["mue_auc"] = auc.get("auc")
        for dom in lb.PARTICLE_DOMAINS:
            mtr = np.asarray(p_tr["particle"]).astype(str) == dom
            mte = np.asarray(p_te["particle"]).astype(str) == dom
            d = ef._direction_tasks(np.asarray(p_tr["events"], float)[mtr],
                                    np.asarray(p_tr["direction"], float)[mtr],
                                    np.asarray(p_tr["run_ids"])[mtr],
                                    np.asarray(p_te["events"], float)[mte],
                                    np.asarray(p_te["direction"], float)[mte],
                                    np.asarray(p_te["run_ids"])[mte])
            particle[f"{dom}_direction_mlp_median_deg"] = d["mlp"]["median_deg"]
            e = ef._energy_task(p_tr["events"], p_tr["deposited_MeV"], p_tr["run_ids"],
                                mtr & np.asarray(p_tr["contained"], bool), p_te["events"],
                                p_te["deposited_MeV"], p_te["run_ids"],
                                mte & np.asarray(p_te["contained"], bool))
            particle[f"{dom}_energy_r2"] = e.get("r2_mean", e.get("r2")) \
                if e.get("status") != "not_evaluable" else None
            particle[f"{dom}_energy_status"] = e.get("status", "evaluated")
            particle[f"{dom}_n_contained_test"] = e.get("n_contained_test")
    metrics["particle"] = particle
    arrays = {"vertex_errors": full["errors"], "run_ids": np.asarray(te["runs"]).astype(str),
              "event_ids": np.asarray(te["event_ids"], np.int64),
              "half_errors": (None if halfv is None else halfv["errors"]),
              "log_yield_errors": ly_err}
    return metrics, arrays


def paired_vertex_difference(left: dict, right: dict) -> dict:
    """Run-cluster bootstrap of right - left vertex error (round-1 ``paired_run_bootstrap``)."""
    from . import ext_followup as ef
    if not np.array_equal(left["event_ids"], right["event_ids"]):
        raise ValueError("paired bootstrap needs the same probe-test events")
    return ef.paired_run_bootstrap(left["vertex_errors"], right["vertex_errors"],
                                   left["run_ids"], resamples=BOOTSTRAP_RESAMPLES,
                                   seed=BOOTSTRAP_SEED)


def evaluate_model(*, detector: str, arm: str, seed: int, h5_path, out_root, checkpoint=None,
                   device: str = "cuda", ledger_root=None, log=print) -> dict:
    """Evaluate one §5.1 model on the detector's probe cohorts; writes JSON + npz."""
    import torch
    from . import r2_generator as gen
    from . import r2_store as rs
    out_dir = check_write_path(Path(out_root) / "eval" / run_name(detector, arm, seed))
    out_dir.mkdir(parents=True, exist_ok=True)
    geometry = rg.load_r2_geometry(detector)
    rule = gen.time_rule_for(geometry)
    scaling = rs.scaling_for(geometry, rule.time_scale_ns)
    model = build_r2_arm(arm, geometry, seed, scaling=scaling)
    ckpt_sha = None
    if arm != INIT_ARM:
        if checkpoint is None:
            raise ValueError(f"{arm} needs its frozen final checkpoint")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        ident = payload["identity"]
        if int(ident.get("seed", seed)) != int(seed):
            raise RuntimeError(f"checkpoint seed {ident.get('seed')} != {seed}")
        if int(payload["step"]) != STEPS or ident["arm"] != arm:
            raise RuntimeError("checkpoint is not the frozen final step of this arm")
        model.load_state_dict(payload["model"])
        ckpt_sha = rs.sha256_file(checkpoint)
    model = model.to(device).eval()
    probes = {"train": rs.open_cohort(h5_path, "dev_probe_train"),
              "test": rs.open_cohort(h5_path, "dev_probe_test")}
    t0 = time.time()
    streams = stream_forms(model.jepa, geometry, scaling, probes, device=device)
    metrics, arrays = metrics_from_streams(streams, geometry)
    record = {"extension": EXTENSION, "detector": detector, "arm": arm, "seed": int(seed),
              "checkpoint_sha256": ckpt_sha, "metrics": metrics,
              "wall_seconds": time.time() - t0, "code_sha": lb.current_code_sha(),
              "h5_sha256": None, "registered": False,
              "gpu_model": (torch.cuda.get_device_name() if str(device).startswith("cuda") else None),
              "gpu_profile": gpu_profile_name(), "ledger_line": gpu_profile()["line"]}
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=1, sort_keys=True,
                                                     default=float) + "\n")
    np.savez_compressed(out_dir / "per_event.npz",
                        **{k: v for k, v in arrays.items() if v is not None})
    if str(device).startswith("cuda"):
        ledger_add(ledger_root or out_root, {"kind": "eval", "detector": detector, "arm": arm,
                                             "seed": int(seed),
                              "gpu_h": record["wall_seconds"] / 3600.0,
                              "wall_s": record["wall_seconds"]})
    log(json.dumps({"detector": detector, "arm": arm, "seed": seed,
                    "vertex_full_m": metrics["vertex_full_m"], "f_event": metrics["f_event"]}))
    return record


# ======================================================================= X0 preflight (§4.2)
def x0_geometry(detector: str) -> dict:
    geometry = rg.load_r2_geometry(detector)
    values = rg.derived_values(geometry)
    diffs = rg.compare_with_design(values, detector)
    return {"detector": detector, "geometry": geometry.summary(), "derived": values,
            "design_table": rg.DESIGN_TABLE[detector], "differences": diffs,
            "readout_window": rg.readout_window_check(values),
            "stop": ("S4" if diffs or not rg.readout_window_check(values)["ok"] else None)}


def _fixed_yield_draws(geometry, cfg, y: float, event_id: int, seed: int, tag: str,
                      capacity_y: float | None = None):
    """Round-1 ``lucid_detectability._fixed_yield_draws``: the draws are made at a fixed
    capacity yield (so every trial yield of a bisection re-maps the same random numbers),
    then the yield is set to ``y``."""
    from . import r2_generator as gen
    cy = float(y if capacity_y is None else max(float(capacity_y), float(y)))
    cap = dataclasses.replace(cfg, y_lo=cy * (1.0 - 1e-12), y_hi=cy)
    return dataclasses.replace(gen.draw_event(geometry, cap, tag, int(event_id), seed=seed),
                               yield_photons=float(y))


def occupancy_stats(geometry, cfg, rule, y: float, n_events: int, seed: int,
                    tag: str = "r2/x0/occupancy", workers: int = 1,
                    capacity_y: float | None = None) -> dict:
    """Median hit count, mean (pseudo-)module charge and hit fraction at a fixed yield."""
    def one(i):
        from . import r2_generator as gen
        state = lb.nominal_state(geometry, tag)
        hits = gen.readout(state, _fixed_yield_draws(geometry, cfg, y, i, seed, tag, capacity_y),
                           geometry, cfg, rule)
        charge = np.bincount(geometry.module_index[np.asarray(hits.pmt_id, int)],
                             weights=hits.charge_pe, minlength=geometry.n_modules).mean()
        return len(hits), float(charge)
    rows = _parallel_map(one, range(int(n_events)), workers)
    n_hit = np.array([r[0] for r in rows], float)
    return {"yield_photons": float(y), "n_events": int(n_events),
            "median_hit_pmts": float(np.median(n_hit)),
            "median_hit_fraction": float(np.median(n_hit / geometry.n_pmt)),
            "median_module_charge_pe": float(np.median([r[1] for r in rows]))}


_POOL_FN = None


def _pool_call(i):
    return _POOL_FN(i)


def _parallel_map(fn, items, workers: int):
    items = list(items)
    if int(workers) <= 1:
        return [fn(i) for i in items]
    import multiprocessing as mp
    global _POOL_FN
    _POOL_FN = fn
    with mp.get_context("fork").Pool(int(workers)) as pool:
        return pool.map(_pool_call, items, chunksize=max(1, len(items) // (4 * int(workers))))


def compute_yield_range(geometry, cfg, rule, *, n_events: int = 1000, iterations: int = 10,
                        seed: int = lb.DEFAULT_SEED, y_lo_start: float = 6000.0,
                        y_hi_start: float = 320000.0, workers: int = 16) -> dict:
    """The round-1 §3.5.3a yield-range procedure in relative terms (runbook §2.2).

    Y_lo: median hit fraction 0.100 (= 200 / 1995); Y_hi: the lower of 0.95 median hit
    fraction or 300 PE median (pseudo-)module charge; log bisection, 10 iterations.  The
    starting bracket is the round-1 one, doubled until it contains the targets (no doubling
    on WCTE); a bracket that cannot be found in 30 doublings is S8.
    """
    cap = {"y": 4.0 * float(y_hi_start)}  # round 1: capacity 4 x the bracket's upper yield
    stats = lambda y: occupancy_stats(geometry, cfg, rule, y, n_events, seed, workers=workers,
                                      capacity_y=cap["y"])
    target_hits = 0.100 * geometry.n_pmt
    lo, hi = float(y_lo_start), float(y_hi_start)
    trail = []
    for _ in range(30):
        s = stats(hi)
        trail.append(s)
        if s["median_hit_pmts"] >= target_hits:
            break
        lo, hi = hi, hi * 2.0
        cap["y"] = max(cap["y"], 4.0 * hi)
    else:
        return {"converged": False, "stop": "S8", "reason": "no Y_lo bracket", "trail": trail}
    for _ in range(30):
        s = stats(lo)
        trail.append(s)
        if s["median_hit_pmts"] < target_hits:
            break
        lo /= 2.0
    else:
        return {"converged": False, "stop": "S8", "reason": "no Y_lo lower bracket", "trail": trail}

    def bisect(objective, target, a, b):
        for _ in range(int(iterations)):
            mid = math.sqrt(a * b)
            a, b = (mid, b) if objective(mid) < target else (a, mid)
        return math.sqrt(a * b)
    y_lo = bisect(lambda y: stats(y)["median_hit_pmts"], target_hits, lo, hi)
    upper = max(hi, 2.0 * y_lo)
    for _ in range(30):
        cap["y"] = max(cap["y"], 4.0 * upper)
        s = stats(upper)
        trail.append(s)
        if s["median_module_charge_pe"] >= 300.0 and s["median_hit_fraction"] >= 0.95:
            break
        upper *= 2.0
    else:
        return {"converged": False, "stop": "S8", "reason": "no Y_hi bracket", "y_lo": y_lo,
                "trail": trail}
    y_pe = bisect(lambda y: stats(y)["median_module_charge_pe"], 300.0, y_lo, upper)
    y_hf = bisect(lambda y: stats(y)["median_hit_fraction"], 0.95, y_lo, upper)
    y_hi = min(y_pe, y_hf)
    return {"converged": True, "stop": None, "y_lo": float(y_lo), "y_hi": float(y_hi),
            "y_hi_bound_source": "module_charge_300PE" if y_pe <= y_hf else "hit_fraction_0.95",
            "y_at_module_charge_300pe": float(y_pe), "y_at_hit_fraction_0.95": float(y_hf),
            "y_lo_check": stats(y_lo), "y_hi_check": stats(y_hi),
            "targets": {"median_hit_pmts": target_hits, "median_hit_fraction_lo": 0.100,
                        "module_charge_pe": 300.0, "hit_fraction": 0.95},
            "n_events": int(n_events), "iterations": int(iterations), "seed": int(seed),
            "bracket_trail": [{"y": t["yield_photons"], "hits": t["median_hit_pmts"]}
                              for t in trail]}


def trigger_check(geometry, cfg, rule, *, y_lo: float, n_dark: int = 10_000,
                  n_events: int = 1000, seed: int = lb.DEFAULT_SEED, workers: int = 16) -> dict:
    """Dark-only windows give no trigger; Y_lo bombs and each particle domain trigger >= 99 %."""
    from . import r2_generator as gen

    def dark(i):
        tag = "r2/x0/dark"
        d = gen.draw_event(geometry, cfg, tag, int(i), seed=seed)
        d = dataclasses.replace(d, yield_photons=0.0)
        h = gen.readout(lb.nominal_state(geometry, tag), d, geometry, cfg, rule)
        return h.trigger_ns is not None

    def bomb(i):
        tag = "r2/x0/ylo"
        h = gen.readout(lb.nominal_state(geometry, tag),
                        _fixed_yield_draws(geometry, cfg, y_lo, i, seed, tag), geometry, cfg, rule)
        return h.trigger_ns is not None

    def particle(args):
        dom, i = args
        tag = f"r2/x0/{dom}"
        d = gen.draw_event(geometry, cfg, tag, int(i), seed=seed, domain=dom)
        h = gen.readout(lb.nominal_state(geometry, tag), d, geometry, cfg, rule)
        return h.trigger_ns is not None
    dark_triggers = int(sum(_parallel_map(dark, range(int(n_dark)), workers)))
    eff = {"photon_y_lo": float(np.mean(_parallel_map(bomb, range(int(n_events)), workers)))}
    for dom in lb.PARTICLE_DOMAINS:
        eff[dom] = float(np.mean(_parallel_map(particle, [(dom, i) for i in range(int(n_events))],
                                               workers)))
    # §9 S3 is defined by the efficiencies only ("the only stops"); the dark-only result is
    # recorded as its own check and reported.
    eff_ok = all(v >= 0.99 for v in eff.values())
    return {"nhits_n": rule.n, "window_ns": rule.trigger_window_ns, "n_dark_windows": int(n_dark),
            "dark_triggers": dark_triggers, "dark_check_pass": dark_triggers == 0,
            "dark_trigger_rate_per_window": dark_triggers / float(n_dark),
            "efficiency": eff, "efficiency_pass": bool(eff_ok), "n_events": int(n_events),
            "y_lo": float(y_lo), "pass": bool(eff_ok and dark_triggers == 0),
            "stop": None if eff_ok else "S3"}


def tolerant_sensor_map(geometry, sensor_points, max_distance_m: float = 0.06) -> np.ndarray:
    """Transport sensor -> geometry index; -1 for a transport sensor absent from the geometry.

    The active geometry PMTs must be matched one-to-one within ``max_distance_m`` (the
    round-1 tolerance).  HK's LUCiD transport may carry the excluded inactive sensors.
    """
    from scipy.spatial import cKDTree
    points = np.asarray(sensor_points, float)
    dist, idx = cKDTree(geometry.positions_m).query(points)
    mapping = np.where(dist <= max_distance_m, idx, -1).astype(np.int64)
    matched = mapping[mapping >= 0]
    if np.unique(matched).size != matched.size or matched.size != geometry.n_pmt:
        raise ValueError("transport sensors do not match the active geometry one-to-one")
    return mapping


def gate_v(detector: str, cfg, *, photonsim_dir=PHOTONSIM_DIR) -> dict:
    """Gate V per detector: the 10 holdout PhotonSim events through LUCiD, round-1 metrics.

    No refit of the yield per MeV (S2 on failure).  Host-only (imports JAX via the pinned
    ``PromptTransport``).
    """
    import sys
    from . import cherenkov_validation as cv
    pinned = str(rg.lucid_config_dir().parent)  # the pinned LUCiD clone (eaf2e89), not another
    if pinned not in sys.path[:1]:
        sys.path.insert(0, pinned)
    from noise_module_lucid.lucid_transport import PromptTransport
    geometry = rg.load_r2_geometry(detector)
    config_path = rg.lucid_config_dir() / rg.DETECTORS[detector][0]
    transport = PromptTransport(config_path, lb._repo_root() / UPSTREAM_MANIFEST_RELPATH)
    mapping = tolerant_sensor_map(geometry, transport.geometry.sensor_points)
    events, dropped = [], []
    for index, name in enumerate(GATE_V_HOLDOUT):
        with np.load(Path(photonsim_dir) / name, allow_pickle=False) as data:
            raw = {k: data[k] for k in data.files}
        raw = cv.contain_event(raw, geometry)
        batch = transport.export(index, np.asarray(raw["photon_id"], np.int64),
                                 np.asarray(raw["origins_m"], float),
                                 np.asarray(raw["directions"], float),
                                 np.asarray(raw["emission_ns"], float),
                                 np.asarray(raw["wavelength_nm"], float))
        raw_sensor = np.asarray(batch.sensor_id, np.int64)
        keep = mapping[raw_sensor] >= 0  # photons on excluded (inactive) sensors are dropped
        kept = SimpleNamespace(sensor_id=raw_sensor[keep],
                               arrival_time_ns=np.asarray(batch.arrival_time_ns)[keep],
                               wavelength_nm=np.asarray(batch.wavelength_nm)[keep])
        n_dropped = int((~keep).sum())
        dropped.append(n_dropped)
        events.append(cv.event_from_transport(raw, kept, event_id=index,
                                              sensor_map=np.where(mapping >= 0, mapping, 0)))
    result = cv.validate(geometry, cfg, events)
    result["detector"] = detector
    result["holdout"] = list(GATE_V_HOLDOUT)
    result["photons_on_excluded_sensors"] = dropped
    result["transport_sensors"] = int(mapping.size)
    result["stop"] = None if result["passed"] else "S2"
    return result


def benchmark(geometry, cfg, rule, *, n_events: int = 1000, with_twin: bool = True) -> dict:
    """1,000 mixed events on one process: events/s, hits and bytes per event (§4.2 item 5)."""
    from . import r2_generator as gen
    run = lb.RunSpec("r2/x0/benchmark/0000", "r2/x0/benchmark", "F1", 0.05, 0, int(n_events),
                     lb.DEFAULT_SEED)
    t0 = time.perf_counter()
    r = gen.generate_run(geometry, cfg, run, rule, with_twin=False)
    t_main = time.perf_counter() - t0
    twin_s = None
    if with_twin:
        small = lb.RunSpec("r2/x0/benchmark/twin", "r2/x0/benchmark", "F1", 0.05, 0, 100,
                           lb.DEFAULT_SEED)
        t1 = time.perf_counter()
        gen.generate_run(geometry, cfg, small, rule, with_twin=True, store_expected_pmt=True)
        twin_s = (time.perf_counter() - t1) / 100.0
    store = lb.sparse_from_readouts(r.event_ids, r.hits)
    raw_bytes = sum(a.nbytes for a in (store.pmt_id, store.charge_pe, store.time_ns))
    return {"n_events": int(n_events), "seconds": t_main, "events_per_s": n_events / t_main,
            "probe_seconds_per_event_with_twin_and_expected_pe": twin_s,
            "hits_per_event": r.n_hits / n_events, "raw_hit_bytes_per_event": raw_bytes / n_events,
            "readout": r.readout_record.as_dict(), "peak_rss_mb": lb.peak_rss_mb()}


def projections(bench: dict, n_pmt: int, *, cores: int = 16) -> dict:
    """CPU wall and disk projection for the §3.1 cohorts of one detector."""
    train_ev, probe_ev = 100_000, 40_000
    s_train = train_ev / bench["events_per_s"]
    s_probe = probe_ev * (bench.get("probe_seconds_per_event_with_twin_and_expected_pe")
                          or 2.0 / bench["events_per_s"])
    wall_h = (s_train + s_probe) / cores / 3600.0
    # Compressed sizes: hits ~ 0.6 x raw (gzip), twins double the probe hits, expected_pe
    # is float32 per PMT (mostly small values, compresses ~3x).
    hit_b = bench["raw_hit_bytes_per_event"] * 0.6
    disk = train_ev * hit_b + probe_ev * (2 * hit_b + 4 * n_pmt / 3.0)
    return {"cpu_wall_h_16_cores": wall_h, "disk_bytes_estimate": disk,
            "cohort_events": {"dev_train": train_ev, "probe_total": probe_ev}}


# ======================================================================= readings aggregation
def load_eval(root, detector: str, arm: str, seed: int):
    """(metrics, per-event arrays) of one evaluated model, or None if absent."""
    d = Path(root) / "eval" / run_name(detector, arm, seed)
    if not (d / "metrics.json").is_file():
        return None
    metrics = json.loads((d / "metrics.json").read_text())["metrics"]
    with np.load(d / "per_event.npz", allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}
    return metrics, arrays


def detector_readings(root, detector: str, *, three_seed: bool = False) -> dict:
    """R2-a..R2-e, the floor band and the seed-9002 triggers of one detector (§7, §8).

    ``three_seed=True`` is the §7 seed-9002 version: each reading uses seed 9002 where every
    arm it reads (and the seed-matched ``init``) has a 9002 evaluation, so a single-arm
    trigger (e.g. only ``mae_mse`` for an unclear R2-c) yields a three-seed version of exactly
    the readings that concern it; the others keep two seeds and say so.
    """
    seeds = tuple(int(s) for s in MAIN_SEEDS)
    ev = {(a, s): load_eval(root, detector, a, s) for a in MAIN_ARMS + (INIT_ARM,)
          for s in seeds + (CONDITIONAL_SEED,)}
    missing = [f"{a}:{s}" for (a, s), v in ev.items() if v is None and s in seeds]
    if missing:
        return {"detector": detector, "status": "incomplete", "missing": missing}
    m = {k: v[0] for k, v in ev.items() if v is not None}
    arr = {k: v[1] for k, v in ev.items() if v is not None}

    def seeds_for(*arms):
        if three_seed and all((a, CONDITIONAL_SEED) in m for a in arms):
            return seeds + (CONDITIONAL_SEED,)
        return seeds
    out = {"detector": detector, "status": "evaluated",
           "version": "three_seed" if three_seed else "two_seed"}
    sa = seeds_for("sigreg_jepa")
    out["R2-a"] = reading_r2a({s: m[("sigreg_jepa", s)]["f_event"] for s in sa})
    sb = seeds_for("target_posfree", INIT_ARM)
    out["R2-b"] = reading_r2b(
        {s: {"vertex": m[("target_posfree", s)]["vertex_full_m"],
             "token_r2": m[("target_posfree", s)]["token_log_charge_r2"],
             "f_event": m[("target_posfree", s)]["f_event"]} for s in sb},
        {s: {"vertex": m[(INIT_ARM, s)]["vertex_full_m"]} for s in sb})
    sc = seeds_for("mae_mse", INIT_ARM)
    out["R2-c"] = reading_r2c({s: paired_vertex_difference(arr[(INIT_ARM, s)], arr[("mae_mse", s)])
                               for s in sc})
    vertex = {a: {s: m[(a, s)]["vertex_full_m"] for s in seeds_for(a)} for a in WCTE_ORDER}
    out["R2-d"] = reading_r2d(vertex)
    out["seeds_used"] = {"R2-a": list(sa), "R2-b": list(sb), "R2-c": list(sc),
                         "R2-d": {a: sorted(v) for a, v in vertex.items()}}
    rf, mae = load_eval(root, detector, RF_ARM, 9000), ev[("mae_mse", 9000)]
    if rf is not None:
        pick = lambda mm: {"token_r2": mm["token_log_charge_r2"], "f_event": mm["f_event"],
                           "particle": {k: v for k, v in mm.get("particle", {}).items()
                                        if isinstance(v, (int, float))}}
        out["R2-e"] = reading_r2e(paired_vertex_difference(mae[1], rf[1]), pick(rf[0]),
                                  pick(mae[0]))
    else:
        out["R2-e"] = {"reading": "R2-e", "status": "not run"}
    init_all = {s: load_eval(root, detector, INIT_ARM, s) for s in INIT_SEEDS}
    flat = lambda mm: {"vertex_full_m": mm["vertex_full_m"],
                       "token_log_charge_r2": mm["token_log_charge_r2"],
                       "module_gain_r2": mm.get("module_gain_r2"),
                       **{k: v for k, v in mm.get("particle", {}).items()
                          if isinstance(v, (int, float))}}
    out["floor_band"] = floor_band({s: flat(v[0]) for s, v in init_all.items() if v is not None})
    out["floor_band"]["seeds"] = [s for s, v in init_all.items() if v is not None]
    # Module gain R² minus the seed-matched init floor (§6)
    out["module_gain_minus_init"] = {
        f"{a}:{s}": (m[(a, s)].get("module_gain_r2") - m[(INIT_ARM, s)].get("module_gain_r2")
                     if m[(a, s)].get("module_gain_r2") is not None
                     and m[(INIT_ARM, s)].get("module_gain_r2") is not None else None)
        for a in MAIN_ARMS for s in seeds}
    if not three_seed:
        out["seed9002_triggers"] = seed9002_triggers(out, vertex)
    out["metrics"] = {f"{a}:{s}": v for (a, s), v in m.items()}
    if rf is not None:
        out["metrics"][f"{RF_ARM}:9000"] = rf[0]
    return out


def readings_both_versions(root, detector: str) -> dict:
    """The two-seed readings and, where any seed 9002 was evaluated, the three-seed version."""
    two = detector_readings(root, detector)
    out = {"two_seed": two, "three_seed": None}
    if two.get("status") == "evaluated" and any(
            load_eval(root, detector, a, CONDITIONAL_SEED) is not None for a in MAIN_ARMS):
        out["three_seed"] = detector_readings(root, detector, three_seed=True)
    return out


def loss_share_diagnostics(root, detector: str) -> dict:
    """§6 time-scale diagnostic: loss-term shares of mae_mse and mae_mse_rf (seed 9000)."""
    out = {}
    for arm in ("mae_mse", RF_ARM):
        log_path = Path(root) / "runs" / run_name(detector, arm, 9000) / "train_log.jsonl"
        if log_path.is_file():
            lines = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()]
            out[arm] = loss_shares(lines)
    return out


# ======================================================================= WCTE regression (§4.2)
ROUND1_LUCID_ROOT = Path("/ceph/dwong/jepa/data/ext_pregrid_dev_20261025/lucid")
ROUND1_PROBE_KEYS = {"dev_probe_train": "dev_probe_train__F1__0.05__s20261017",
                     "dev_probe_test": "dev_probe_test__F1__0.05__s20261017"}
ROUND1_RETURNS = Path("/ceph/dwong/jepa/returns")
#: label -> (round-2 arm, seed, round-1 checkpoint (relative to ROUND1_RETURNS), committed
#: full-input vertex in m, source of the committed value).
WCTE_REGRESSION = {
    "init9000": ("init", 9000, None, 0.16007169,
                 "results/jepa/20261101_readout/matrix.json items/init9000"),
    "mae_within_trigger9000": (
        "mae_mse", 9000,
        "ext_mae_dejitter_20261031/train/runs/ext_dev_s005_within_trigger__mae_mse__s9000/final.pt",
        0.10491560, "results/jepa/20261031_mae_dejitter/summary.json")}
WCTE_REGRESSION_TOLERANCE_M = 1e-3


def wcte_pack(work_dir) -> dict:
    """Pack the two round-1 WCTE probe cohorts (read-only, via symlinks) and verify them."""
    from . import r2_generator as gen
    from . import r2_store as rs
    work_dir = check_write_path(work_dir)
    shard_root = work_dir / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    for cohort, key in ROUND1_PROBE_KEYS.items():
        link = shard_root / cohort
        if not link.exists():
            link.symlink_to(ROUND1_LUCID_ROOT / key / "runs")
    cohorts = {c: gen.cohort_runs(c) for c in ROUND1_PROBE_KEYS}
    h5 = work_dir / "wcte_probe_r2.h5"
    if not h5.exists():
        geometry = rg.load_r2_geometry("wcte")
        rs.pack_detector(h5, shard_root, cohorts,
                         attrs={"extension": EXTENSION, "detector": "wcte",
                                "geometry_sha256": geometry.digest,
                                "source": "round-1 stored probe cohorts (read-only)"})
    return {"h5": str(h5), **rs.verify_pack(h5, shard_root, cohorts)}


def wcte_regression(work_dir, *, device: str = "cuda", ledger_root=RETURNS_ROOT) -> dict:
    """Reproduce init9000 and mae_within_trigger9000 full-input vertex within 1e-3 m (S6)."""
    pack = wcte_pack(work_dir)
    rows = {}
    for label, (arm, seed, ckpt, expected, source) in WCTE_REGRESSION.items():
        record = evaluate_model(detector="wcte", arm=arm, seed=seed, h5_path=pack["h5"],
                                out_root=work_dir,
                                checkpoint=None if ckpt is None else ROUND1_RETURNS / ckpt,
                                device=device, ledger_root=ledger_root)
        got = record["metrics"]["vertex_full_m"]
        rows[label] = {"expected_m": expected, "got_m": got, "abs_diff_m": abs(got - expected),
                       "pass": abs(got - expected) <= WCTE_REGRESSION_TOLERANCE_M,
                       "source": source, "checkpoint_sha256": record["checkpoint_sha256"]}
    ok = all(r["pass"] for r in rows.values())
    return {"pack": pack, "items": rows, "tolerance_m": WCTE_REGRESSION_TOLERANCE_M,
            "pass": ok, "stop": None if ok else "S6"}


# ======================================================================= CLI
def _parser():
    p = argparse.ArgumentParser(prog="python -m jepa_noise.ext_r2")
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("dry-run")
    for name in ("preflight", "generate", "pack", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--detector", required=True, choices=sorted(rg.DETECTORS))
        s.add_argument("--out", default=None)
        s.add_argument("--workers", type=int, default=16)
        s.add_argument("--h5", default=None)
        s.add_argument("--y-lo", type=float, default=None)
        s.add_argument("--y-hi", type=float, default=None)
        s.add_argument("--cohort", default=None)
        s.add_argument("--runs", default=None, help="a:b run slice (generate)")
        s.add_argument("--items", default=None, help="arm:seed[:ckpt],... (evaluate)")
        s.add_argument("--steps", default="geometry,trigger,yield,gatev,bench")
        s.add_argument("--x0", default=None, help="preflight JSON (generate)")
        s.add_argument("--device", default="cuda")
        s.add_argument("--out-dir", default=None, help="work directory (WCTE regression)")
    for name in ("smoke", "train"):
        s = sub.add_parser(name)
        s.add_argument("--detector", required=True, choices=sorted(rg.DETECTORS))
        s.add_argument("--arm", required=True, choices=TRAINED_ARMS)
        s.add_argument("--seed", type=int, default=9000)
        s.add_argument("--micro-batch", type=int, default=8)
        s.add_argument("--h5", required=True)
        s.add_argument("--out", default=str(RETURNS_ROOT))
        s.add_argument("--device", default="cuda")
    r = sub.add_parser("report")
    r.add_argument("--root", required=True)
    return p


def _generate(args):
    """Generate one detector's §3.1 cohorts into per-run shards (parallel over runs)."""
    from . import r2_generator as gen
    geometry = rg.load_r2_geometry(args.detector)
    rule = gen.time_rule_for(geometry)
    if not args.x0:
        raise SystemExit("generate needs --x0 <preflight JSON> with no stop (runbook §4.2)")
    yields = x0_clearance(json.loads(Path(args.x0).read_text()))
    cfg = gen.load_r2_config(yields["y_lo"], yields["y_hi"])
    ledger_root = Path(args.out_dir) if args.out_dir else RETURNS_ROOT
    root = check_write_path(Path(args.out or GEN_ROOT / args.detector))
    cohorts = [args.cohort] if args.cohort else list(gen.COHORTS)
    jobs = []
    for cohort in cohorts:
        runs = gen.cohort_runs(cohort)
        a, b = (0, len(runs)) if not args.runs else map(int, args.runs.split(":"))
        jobs += [(cohort, i) for i in range(a, min(b, len(runs)))]
    derived = rg.derived_values(geometry)
    code_sha = lb.current_code_sha()

    def one(job):
        cohort, i = job
        run = gen.cohort_runs(cohort)[i]
        d = root / cohort / run.run_id.replace("/", "__")
        if (d / "SHA256SUMS").is_file():
            lb.verify_sha256sums(d)
            return "kept"
        twin = gen.cohort_has_twin(cohort)
        r = gen.generate_run(geometry, cfg, run, rule, with_twin=twin, store_expected_pmt=twin)
        gen.write_generated_run(d, r, geometry, cfg, rule, code_sha=code_sha, derived=derived)
        return "made"
    t0 = time.time()
    done = _parallel_map(one, jobs, args.workers)
    ledger_add(ledger_root, {"kind": "generation", "detector": args.detector,
                             "jobs": len(jobs), "made": done.count("made"),
                             "workers": int(args.workers), "wall_h": (time.time() - t0) / 3600.0},
               cpu=True)
    print(json.dumps({"detector": args.detector, "jobs": len(jobs), "made": done.count("made"),
                      "kept": done.count("kept"), "wall_s": time.time() - t0}))


def _pack(args):
    from . import r2_generator as gen
    from . import r2_store as rs
    geometry = rg.load_r2_geometry(args.detector)
    rule = gen.time_rule_for(geometry)
    shard_root = Path(args.out or GEN_ROOT / args.detector)
    h5 = check_write_path(Path(args.h5 or DATA_ROOT / args.detector / f"{args.detector}_r2.h5"))
    cohorts = {c: gen.cohort_runs(c) for c in gen.COHORTS}
    if not args.x0:
        raise SystemExit("pack needs --x0 <preflight JSON>")
    yields = x0_clearance(json.loads(Path(args.x0).read_text()))
    cfg = gen.load_r2_config(yields["y_lo"], yields["y_hi"])
    attrs = {"extension": EXTENSION, "detector": args.detector,
             "geometry_sha256": geometry.digest, "geometry_summary": geometry.summary(),
             "generator_config_sha256": cfg.digest, "derived_values": rg.derived_values(geometry),
             "time_rule": rule.record(), "yield_range_photons": [cfg.y_lo, cfg.y_hi],
             "runbook_sha256": RUNBOOK_SHA256, "code_sha": lb.current_code_sha()}
    summary = rs.pack_detector(h5, shard_root, cohorts, attrs=attrs)
    verified = rs.verify_pack(h5, shard_root, cohorts)
    print(json.dumps({"h5": str(h5), "summary": summary, **verified,
                      "bytes": h5.stat().st_size}))


def _preflight(args):
    from . import r2_generator as gen
    out = {"detector": args.detector, "extension": EXTENSION, "code_sha": lb.current_code_sha()}
    steps = args.steps.split(",")
    if "regression" in steps:
        out["wcte_regression"] = wcte_regression(Path(args.out_dir or RETURNS_ROOT / "wcte_regression"),
                                                 device=args.device)
        steps = [s for s in steps if s != "regression"]
    geometry = rg.load_r2_geometry(args.detector)
    rule = gen.time_rule_for(geometry)
    cfg = gen.load_r2_config()
    if "geometry" in steps:
        out["geometry"] = x0_geometry(args.detector)
    if "yield" in steps:
        out["yield_range"] = compute_yield_range(geometry, cfg, rule, workers=args.workers)
    y_lo = args.y_lo or (out.get("yield_range") or {}).get("y_lo")
    y_hi = args.y_hi or (out.get("yield_range") or {}).get("y_hi")
    if "trigger" in steps and y_lo:
        out["trigger"] = trigger_check(geometry, cfg, rule, y_lo=y_lo, workers=args.workers)
    if "gatev" in steps:
        out["gate_v"] = gate_v(args.detector, cfg)
    if "bench" in steps and y_lo and y_hi:
        bench = benchmark(geometry, gen.load_r2_config(y_lo, y_hi), rule)
        out["benchmark"] = bench
        out["projection"] = projections(bench, geometry.n_pmt)
        root = Path(args.out_dir) if args.out_dir else RETURNS_ROOT
        out["storage"] = storage_check(out["projection"]["disk_bytes_estimate"])
        out["cpu"] = cpu_check(root, out["projection"]["cpu_wall_h_16_cores"])
    text = json.dumps(out, indent=1, sort_keys=True, default=float)
    if args.out:
        Path(check_write_path(args.out)).write_text(text + "\n")
    print(text)


def _dry_run():
    plan = []
    for det in DETECTOR_ORDER:
        g = rg.load_r2_geometry(det)
        v = rg.derived_values(g)
        plan.append({"detector": det, "n_pmt": v["n_pmt"], "k": v["k"], "nhits_n": v["nhits_n"],
                     "prompt_window_ns": v["prompt_window_ns"], "time_scale_ns": v["time_scale_ns"],
                     "w_rf": v["w_rf"], "diffs": rg.compare_with_design(v, det),
                     "runs": [run_name(det, a, s) for a in MAIN_ARMS for s in MAIN_SEEDS]
                     + [run_name(det, RF_ARM, 9000)]})
    print(json.dumps(plan, indent=1))


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.mode == "dry-run":
        _dry_run()
    elif args.mode == "preflight":
        _preflight(args)
    elif args.mode == "generate":
        _generate(args)
    elif args.mode == "pack":
        _pack(args)
    elif args.mode in ("smoke", "train"):
        smoke = args.mode == "smoke"
        train_r2(detector=args.detector, arm=args.arm, seed=args.seed, h5_path=args.h5,
                 out_root=args.out, steps=SMOKE_STEPS if smoke else STEPS,
                 micro_batch=args.micro_batch, device=args.device, smoke=smoke)
    elif args.mode == "evaluate":
        h5 = args.h5 or DATA_ROOT / args.detector / f"{args.detector}_r2.h5"
        for item in (args.items or "").split(","):
            if not item:
                continue
            parts = item.split(":")
            evaluate_model(detector=args.detector, arm=parts[0], seed=int(parts[1]), h5_path=h5,
                           out_root=args.out or RETURNS_ROOT,
                           checkpoint=parts[2] if len(parts) > 2 else None, device=args.device)
    elif args.mode == "report":
        out = {det: {**readings_both_versions(args.root, det),
                     "loss_shares": loss_share_diagnostics(args.root, det),
                     "ledger_booked_l40s_h": booked_gpu_h(ledger_load(args.root))}
               for det in DETECTOR_ORDER}
        print(json.dumps(out, indent=1, sort_keys=True, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
