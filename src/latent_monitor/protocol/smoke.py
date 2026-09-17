# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""One bounded CPU development smoke of the two-claim protocol (TWO_CLAIM_REVISION_PLAN §6 Phase B).

    PYTHONPATH=src python -m latent_monitor.protocol.smoke --out results/latent_monitor_smoke_dev_<date>

On the linear subject and the Tier-1 cells, at a size that finishes in well
under a minute on a laptop CPU:

1. a five-partition event-group split (reference_fit / attribution_train /
   development / calibration / evaluation), two declared held-out families,
   one **undeclared** family (the glitch: unknown, evaluation only) and one
   N+S **mixture** cell;
2. alarm-time features through the typed builder (no truth, twin or realised
   Σ reachable), the declarative and data-flow contract checked per arm;
3. Claim 1: ``generic_rich`` vs ``full_intermediate`` (+ ``intermediate_only``,
   the matched-capacity and hook-drop controls, the pass-1 diagnostics) on the
   inclusive evaluation set, the **hard matched** set and the held-out
   families; confusion matrices; ΔF1 with a group bootstrap and the unfrozen
   three-way reading;
4. abstention end to end (conformal novelty on clean calibration windows,
   unknown AUROC, risk–coverage, retained-known F1) and the joint
   detect → abstain → attribute table over clean, N, S, mixture and unknown;
5. Claim 2: the committed generic harm score against the task-sensitive score
   (paired ΔAUROC over cells with a *provisional development* κ_m), the
   hierarchical bootstrap with the cell as outer unit (descriptive where the
   outer units are too few), the four quadrants, missed harm at a cell-level
   threshold, valid-rare cell/window rejection, breakdowns, conditional triage;
6. the designed and task-specific controls (linear subject only) with the
   linearisation check;
7. FAR precision, provenance, and the dev/confirmatory gate (with the full
   run-dependency check reported as warnings).

Everything it writes is **development output and not citable**; the report
says so in its first line, and the JSON is strict (no NaN tokens).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..designed import DesignedCell, DesignedFamily, NonlinearSubjectUnsupported, NullSpaceUnavailable, linearization_check
from ..linear_subject import LinearSubject
from ..reference import fit_reference
from ..support import displaced_controls, validate_support_estimator
from ..tier1 import TARGET_NAMES, all_cells, mixture_cells, reference_cell
from .abstention import ConformalNovelty, evaluate_abstention
from .arms import (ARMS, CONTROL_ARMS, DIAGNOSTIC_ARMS, MARGINS, PRIMARY_ARMS, TUNING_GRID, ArmFitError, arm_feature_names,
                   check_arms, fit_arm, macro_f1, macro_f1_report, read_delta)
from .availability import AlarmTimeInputs, FeatureBatch, tier1_manifest
from .consequence import (MIN_OUTER_UNITS_FOR_INFERENCE, HarmThreshold, alarm_harm_matrix, all_cell_ranking, bootstrap_groups,
                          bootstrap_hierarchical, breakdown, cell_aggregate, cell_alarm_threshold, cell_weights,
                          conditional_triage, far_precision, harm_labels, missed_harm_rate_at_budget, paired_delta_auroc,
                          supporting_intermediate_cubic, valid_rare_cell_rejection, valid_rare_window_rejection)
from .features import GENERIC_COMMITTED, INTERMEDIATE_COMMITTED, AlarmTimeReference, NullCalibrator, build_alarm_time_features
from .labels import label_from_legacy
from .matching import hard_match, joint_decision_table
from .mode import ProtocolMode, RunDependencies, provenance, require_gates, require_run_dependencies
from .splits import assign_partitions, split_event_groups

FRACTIONS = {"reference_fit": 0.30, "attribution_train": 0.20, "development": 0.15, "calibration": 0.15, "evaluation": 0.20}
HELD_OUT_FAMILIES = ("line_pickup", "double_pulse")
UNDECLARED_FAMILIES = ("glitch",)
FAR_BUDGET = 0.01
ABSTENTION_ALPHA = 0.10          # development value; unfrozen
TASK_TARGET = 0                  # the declared consequence output: amplitude
CELL_AGGREGATE = "median"


def _js(o: Any):
    """Strict-JSON conversion: NaN/inf → None, numpy → python."""
    if isinstance(o, dict):
        return {str(k): _js(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_js(v) for v in o]
    if isinstance(o, np.ndarray):
        return _js(o.tolist())
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def run_smoke(out: Path, *, n_channels: int = 6, n_samples: int = 128, latent_dim: int = 5, n_groups: int = 120,
              n_noise_per_window: int = 2, seed: int = 0, mode: str = "dev", kappa_m: HarmThreshold | None = None,
              designed_norm: float = 3.0, n_boot: int = 100, repo_root: Path | str = ".") -> dict[str, Any]:
    t0 = time.time()
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    config = {"n_channels": n_channels, "n_samples": n_samples, "latent_dim": latent_dim, "n_groups": n_groups,
              "n_noise_per_window": n_noise_per_window, "seed": seed, "fractions": FRACTIONS,
              "held_out_families": list(HELD_OUT_FAMILIES), "undeclared_families": list(UNDECLARED_FAMILIES),
              "far_budget": FAR_BUDGET, "abstention_alpha": ABSTENTION_ALPHA, "task_target": TASK_TARGET,
              "cell_aggregate": CELL_AGGREGATE, "designed_norm": designed_norm, "tuning_grid": list(TUNING_GRID),
              "margins": MARGINS, "n_boot": n_boot, "min_outer_units_for_inference": MIN_OUTER_UNITS_FOR_INFERENCE}
    if kappa_m is None:
        kappa_m = HarmThreshold(1.10, "provisional_dev", units="ratio of amplitude |error| to the clean calibration mean",
                                source="proposal (16 Sep): 'provisional 10% amplitude degradation' — development only; PREREGISTRATION §4 PENDING",
                                arm="tier1", level="cell")
    gates = require_gates(mode, freezes=("core",), thresholds={"kappa_m_tier1": kappa_m}, repo_root=repo_root)
    run_deps = require_run_dependencies(mode, RunDependencies(), repo_root=repo_root, thresholds={"kappa_m_tier1": kappa_m}, out_dir=out)

    # ---------------------------------------------------------------- 1. split by event group
    groups = np.arange(n_groups)
    split = split_event_groups(groups, FRACTIONS, seed=seed, held_out_families=HELD_OUT_FAMILIES)

    # ---------------------------------------------------------------- subject, reference, alarm-time reference (reference_fit only)
    ref_cell = reference_cell(n_channels=n_channels, n_samples=n_samples)
    fit_ids = split.partitions["reference_fit"]
    X_fit, T_fit = ref_cell.batch(fit_ids)
    subject = LinearSubject.fit(X_fit, T_fit, ref_cell.geometry, ref_cell.implied_whitener(), latent_dim=latent_dim,
                                seed=seed, target_names=TARGET_NAMES)
    noise_ref_ids = np.arange(5000, 5000 + 2 * len(fit_ids))
    ref = fit_reference(subject, ref_cell, fit_ids, noise_ref_ids, seed=seed, n_bootstrap=100)
    atref = AlarmTimeReference.fit(subject, ref, X_fit, ref_cell.geometry, task_target=TASK_TARGET)
    # support-estimator control: held-out clean windows (development groups) vs displaced controls
    X_dev_clean, _ = ref_cell.batch(split.partitions["development"])
    z_dev = subject.represent(X_dev_clean, ref_cell.geometry)["z"]
    support_val = validate_support_estimator(atref.support, z_dev, displaced_controls(z_dev, atref.support, seed=seed))
    manifest = tier1_manifest(supplies_noise_only_records=True, latent_dim=latent_dim, n_targets=3, n_pcs=atref.n_pcs)

    # ---------------------------------------------------------------- 2. features for every (cell, window) via the typed builder
    cells = [ref_cell] + all_cells(ref_cell) + mixture_cells(ref_cell)
    rows: list[dict] = []
    batch: FeatureBatch | None = None
    cell_labels: dict[str, Any] = {}
    for cell in cells:
        declared = cell.label.split(":")[-1] not in UNDECLARED_FAMILIES
        lab = label_from_legacy(cell.label, cell.moved, declared=declared)
        cell_labels[cell.label] = lab
        X, T = cell.batch(groups)                                            # T (truth) stays outside the builder
        R_flat = cell.noise_batch(7000 + np.arange(n_groups * n_noise_per_window))
        R = R_flat.reshape(n_groups, n_noise_per_window, R_flat.shape[1], R_flat.shape[2])
        fb = build_alarm_time_features(subject, atref, AlarmTimeInputs(X=X, geometry=cell.geometry, noise_records=R))
        batch = fb if batch is None else batch.concat(fb)
        parts = assign_partitions([{"event_group": int(g), "family": lab.family, "declared": lab.declared} for g in groups], split)
        yhat = subject.outputs(X, cell.geometry)
        for i, g in enumerate(groups):
            rows.append({"cell": cell.label, "origin": lab.origin, "contract": lab.contract, "family": lab.family,
                         "declared": lab.declared, "event_group": int(g), "partition": parts[i],
                         "K_amp": float(abs(yhat[i, TASK_TARGET] - T[i, TASK_TARGET]))})     # evaluation-only truth, only for K
    batch = batch.lock()
    origin = np.array([r["origin"] for r in rows], dtype=object)
    contract = np.array([r["contract"] for r in rows], dtype=object)
    part = np.array([r["partition"] for r in rows], dtype=object)
    cell_of = np.array([r["cell"] for r in rows], dtype=object)
    group_of = np.array([r["event_group"] for r in rows])
    fam_of = np.array([r["family"] for r in rows], dtype=object)
    K_amp = np.array([r["K_amp"] for r in rows])
    arm_status = check_arms(manifest, tuple(ARMS) + CONTROL_ARMS[1:], batch)

    # ---------------------------------------------------------------- 3. clean-null calibration of the committed scores
    clean_cal = (cell_of == "reference") & (part == "calibration")
    clean_eval = (cell_of == "reference") & (part == "evaluation")
    n_cal = int(clean_cal.sum())
    cal = NullCalibrator().fit(batch.subset(clean_cal), list(GENERIC_COMMITTED) + list(INTERMEDIATE_COMMITTED) + ["raw_task_sensitive", "raw_task_cubic", "gr_support_novelty"])
    score_generic = cal.combine_max(batch, GENERIC_COMMITTED)                  # committed generic monitor (detector and Claim-2 baseline)
    score_task = cal.calibrate("raw_task_sensitive", batch["raw_task_sensitive"])  # task-sensitive representation score
    score_cubic = cal.calibrate("raw_task_cubic", batch["raw_task_cubic"])      # supporting intermediate score (D8)
    score_intermediate = cal.combine_max(batch, INTERMEDIATE_COMMITTED)         # supporting
    # abstention novelty: the larger of the calibrated z-support novelty and the calibrated out-of-span fraction
    # (a support move can sit in the residual rather than in z — the glitch family does; declared, unfrozen)
    cal.fit(batch.subset(clean_cal), ["gr_out_of_span"])
    novelty = np.maximum(cal.calibrate("gr_support_novelty", batch["gr_support_novelty"]), cal.calibrate("gr_out_of_span", batch["gr_out_of_span"]))
    window_thr = float(np.quantile(score_generic[clean_cal], 1 - FAR_BUDGET))
    far = far_precision(int(clean_eval.sum()), int(np.sum(score_generic[clean_eval] >= window_thr)), FAR_BUDGET)
    far["calibration_windows"] = n_cal
    far["window_threshold"] = window_thr
    far["note_smoke"] = "18-ish clean calibration windows are a plumbing check; the calibration count is an output of the development precision study (PREREGISTRATION §6)"

    # ---------------------------------------------------------------- 4. Claim 1 arms over {N, S}
    classes = ("N", "S")
    use = np.isin(contract, classes) & np.isin(part, ("attribution_train", "development", "evaluation"))
    sub = batch.subset(use)
    y = contract[use]; p = part[use]; g = group_of[use]; f = fam_of[use]; c_of = cell_of[use]
    arms_to_fit = list(PRIMARY_ARMS) + list(CONTROL_ARMS) + list(DIAGNOSTIC_ARMS)
    arm_results, arm_errors = {}, {}
    for a in arms_to_fit:
        try:
            arm_results[a] = fit_arm(a, manifest, sub, y, p, classes)
        except ArmFitError as e:
            arm_errors[a] = str(e)
    ev = p == "evaluation"
    y_ev, g_ev, f_ev = y[ev], g[ev], f[ev]
    heldout_fam = np.isin(f_ev, HELD_OUT_FAMILIES)
    sub_ev = sub.subset(ev)
    # matching variables standardised by the clean pool: use clean calibration windows for the standardisation
    clean_pool = batch.subset(clean_cal).to_dict()
    from .matching import DEFAULT_MATCHING_VARIABLES
    merged = {v: np.concatenate([clean_pool[v], sub_ev[v]]) for v in DEFAULT_MATCHING_VARIABLES}
    merged_contract = np.concatenate([np.array(["clean"] * n_cal, dtype=object), y_ev])
    hard = hard_match(merged, merged_contract, np.arange(len(merged_contract)) < n_cal, caliper=1.0, seed=seed)
    hard_rows = hard["rows"] - n_cal                                             # indices into the evaluation subset
    preds = {a: r.y_pred_eval for a, r in arm_results.items()}
    c1: dict[str, Any] = {"classes": list(classes), "arms": {}, "errors": arm_errors}
    for a, r in arm_results.items():
        role = "primary" if a in PRIMARY_ARMS else ("control" if a in CONTROL_ARMS else "diagnostic")
        c1["arms"][a] = {"role": role, "status": r.status, "lambda": r.lam, "n_features": len(r.features), "converged": r.extra["converged"],
                         "n_train": r.n_train, "n_tune": r.n_tune, "n_eval": r.n_eval, "f1_tune": r.f1_tune,
                         "f1_eval_inclusive": r.f1_eval,
                         "f1_eval_hard": macro_f1(y_ev[hard_rows], r.y_pred_eval[hard_rows], classes) if len(hard_rows) else float("nan"),
                         "f1_eval_heldout_families": macro_f1(y_ev[heldout_fam], r.y_pred_eval[heldout_fam], classes),
                         "confusion_inclusive": r.extra["confusion_eval"]}

    def delta(idx, a, b, rows_sel=None):
        base = np.arange(len(y_ev)) if rows_sel is None else rows_sel
        ii = base[idx]
        return macro_f1(y_ev[ii], preds[a][ii], classes) - macro_f1(y_ev[ii], preds[b][ii], classes)

    def delta_block(a, b, rows_sel=None, label=""):
        if a not in preds or b not in preds:
            return {"contrast": f"{a} − {b}", "undefined_reason": "an arm failed to fit"}
        sel = np.arange(len(y_ev)) if rows_sel is None else np.asarray(rows_sel)
        if len(sel) == 0:
            return {"contrast": f"{a} − {b}", "undefined_reason": "empty evaluation subset"}
        d = bootstrap_groups(lambda idx: delta(idx, a, b, sel), g_ev[sel], n_boot=n_boot, seed=seed)
        d["reading"] = read_delta(d["point"], d["low"], d["high"], n_windows=int(len(sel)), n_groups=d["n_groups"])
        d["contrast"] = f"{a} − {b}"; d["set"] = label; d["n_windows"] = int(len(sel))
        return d

    c1["primary_hard"] = delta_block("full_intermediate", "generic_rich", hard_rows, "hard matched evaluation set (Claim-1 primary estimand)")
    c1["primary_inclusive"] = delta_block("full_intermediate", "generic_rich", None, "inclusive evaluation set")
    c1["primary_heldout_families"] = delta_block("full_intermediate", "generic_rich", np.where(heldout_fam)[0], "held-out families")
    c1["control_matched_capacity"] = delta_block("full_intermediate", "generic_rich_matched", hard_rows, "hard set; generic arm with matched feature count")
    c1["control_drop_channel"] = delta_block("full_intermediate", "full_intermediate_drop_channel", hard_rows, "hard set; hook drop")
    c1["control_drop_token"] = delta_block("full_intermediate", "full_intermediate_drop_token", hard_rows, "hard set; hook drop")
    c1["control_intermediate_alone"] = delta_block("intermediate_only", "generic_rich", hard_rows, "hard set; internal hooks alone vs generic_rich")
    c1["diagnostic_noise_only_ablation"] = delta_block("all_generic", "all_generic_no_noise", None, "pass-1 diagnostic")
    c1["diagnostic_legacy_layerwise"] = delta_block("full_layerwise_legacy", "all_generic", None, "pass-1 diagnostic (contaminated arm)")
    c1["hard_matching"] = {k: v for k, v in hard.items() if k not in ("rows", "rows_N", "rows_S")}
    c1["family_breakdown_hard"] = {}
    for fam in sorted(set(f_ev[hard_rows].tolist())) if len(hard_rows) else []:
        sel = hard_rows[f_ev[hard_rows] == fam]
        c1["family_breakdown_hard"][fam] = {a: {"n": int(len(sel)), "acc": float(np.mean(preds[a][sel] == y_ev[sel]))} for a in PRIMARY_ARMS if a in preds}
    c1["hierarchical_family_outer_hard"] = bootstrap_hierarchical(lambda idx: delta(idx, "full_intermediate", "generic_rich", hard_rows),
                                                                    f_ev[hard_rows], g_ev[hard_rows], n_boot=n_boot, seed=seed) if len(hard_rows) else None
    c1["note"] = ("macro-F1 over {N, S}; arms fitted on attribution_train, tuned on development, scored once on evaluation; "
                  "reference_fit groups are never supervised examples; held-out and undeclared families entered no fit, tuning or calibration; "
                  "the Claim-1 primary estimand is on the hard matched set")

    # ---------------------------------------------------------------- 5. abstention and the joint decision table (all evaluation windows)
    conf = ConformalNovelty.fit(novelty[clean_cal], ABSTENTION_ALPHA, "calibrated support novelty (gr_support_novelty)")
    ev_all = (part == "evaluation")
    model = arm_results["full_intermediate"].model if "full_intermediate" in arm_results else None
    abst_report, joint = None, None
    if model is not None:
        X_all_ev = batch.subset(ev_all).matrix(arm_feature_names(manifest, "full_intermediate"))
        pred_all = model.predict(X_all_ev)
        cat = np.array([{"N": "N", "S": "S", "clean": "clean", "mixture": "mixture", "U": "unknown", "G": "G"}[c] for c in contract[ev_all]], dtype=object)
        known_ns = np.isin(cat, ("N", "S"))
        abst_report = evaluate_abstention(conf, novelty[ev_all][known_ns | (cat == "unknown")], contract[ev_all][known_ns | (cat == "unknown")],
                                          pred_all[known_ns | (cat == "unknown")], (cat == "unknown")[known_ns | (cat == "unknown")], classes,
                                          novelty_clean_eval=novelty[clean_eval]).to_dict()
        abst_report["support_estimator_validation"] = support_val
        joint = joint_decision_table(score_generic[ev_all], window_thr, conf.abstain(novelty[ev_all]), pred_all, cat)
        joint["n_G_windows_not_tabulated"] = int(np.sum(cat == "G"))

    # ---------------------------------------------------------------- 6. Claim 2: harm ranking over cells
    ref_amp = float(np.mean(K_amp[clean_eval]))          # paired baseline: the same evaluation events through the clean reference cell
    assert ref_amp > 0, "zero baseline: K undefined"
    c2: dict[str, Any] = {"kappa_m": kappa_m.to_dict(), "level": "cell (windows aggregated by " + CELL_AGGREGATE + ")",
                          "K_definition": "K_phys = mean |ŷ_amplitude − amplitude_truth| over the cell's evaluation windows / the same quantity on the "
                                          "clean reference cell's evaluation windows (the same events; truth is evaluation-only). Cell-level K uses the mean; "
                                          "cell-level scores use the declared aggregate",
                          "scores": {"generic_committed": f"max of clean-calibrated {list(GENERIC_COMMITTED)} — committed before any evaluation label",
                                     "task_sensitive": "clean-calibrated ‖z − z̄‖ in M_task = J_yᵀ W_y J_y with W_y one-hot on the amplitude output",
                                     "intermediate_supporting": f"max of clean-calibrated {list(INTERMEDIATE_COMMITTED)} (supporting, not primary)"},
                          "weighting": "uniform_cell (declared, unfrozen)"}
    c2_cells = [c for c in sorted(set(cell_of.tolist())) if c != "reference"]
    ev_c2 = ev_all & np.isin(cell_of, c2_cells)
    cells_arr, K_cell = cell_aggregate(K_amp[ev_c2] / ref_amp, cell_of[ev_c2], "mean")
    _, s_gen = cell_aggregate(score_generic[ev_c2], cell_of[ev_c2], CELL_AGGREGATE)
    _, s_task = cell_aggregate(score_task[ev_c2], cell_of[ev_c2], CELL_AGGREGATE)
    _, s_cubic = cell_aggregate(score_cubic[ev_c2], cell_of[ev_c2], CELL_AGGREGATE)
    _, s_int = cell_aggregate(score_intermediate[ev_c2], cell_of[ev_c2], CELL_AGGREGATE)
    harm = harm_labels(K_cell, kappa_m)
    w = cell_weights([{"weight": 1.0} for _ in cells_arr], "uniform_cell")
    cell_size = int(np.sum(ev_c2) // max(len(cells_arr), 1))
    thr_cell = cell_alarm_threshold(score_generic[clean_cal], cell_size, FAR_BUDGET, CELL_AGGREGATE, seed=seed)
    origins_c = np.array([cell_labels[c].origin for c in cells_arr], dtype=object)
    contracts_c = np.array([cell_labels[c].contract for c in cells_arr], dtype=object)
    valid_rare = np.isin(origins_c, ("S_in_span", "S_support"))
    heldout_c = np.isin([cell_labels[c].family for c in cells_arr], HELD_OUT_FAMILIES)

    def stat_delta(idx):
        sel = np.zeros(len(rows), bool); sel[np.where(ev_c2)[0][idx]] = True
        cc, Kb = cell_aggregate(K_amp[sel] / ref_amp, cell_of[sel], "mean")
        _, gb = cell_aggregate(score_generic[sel], cell_of[sel], CELL_AGGREGATE)
        _, tb = cell_aggregate(score_task[sel], cell_of[sel], CELL_AGGREGATE)
        return paired_delta_auroc(tb, gb, harm_labels(Kb, kappa_m))["delta_auroc"]

    pdelta = paired_delta_auroc(s_task, s_gen, harm, w)
    ci_h = bootstrap_hierarchical(stat_delta, cell_of[ev_c2], group_of[ev_c2], n_boot=n_boot, seed=seed)
    if pdelta["undefined_reason"]:
        ci_h["inference"] = f"undefined — point estimate undefined ({pdelta['undefined_reason']})"
    c2["primary_delta"] = {**pdelta,
                           "ci_hierarchical_outer_cell": ci_h,
                           "ci_event_groups_only_legacy": bootstrap_groups(stat_delta, group_of[ev_c2], n_boot=n_boot, seed=seed),
                           "set": "all intervention cells (development families, in-family held-out events, plus the two held-out families and the unknown)"}
    ho_idx = np.where(heldout_c)[0]
    if len(ho_idx):
        ev_ho = ev_c2 & np.isin(cell_of, cells_arr[ho_idx])
        c2["primary_delta_heldout_families_only"] = {**paired_delta_auroc(s_task[ho_idx], s_gen[ho_idx], harm[ho_idx]),
                                                     "n_cells": int(len(ho_idx)),
                                                     "note": f"the Claim-2 estimand set proper — {len(ho_idx)} held-out cells: descriptive only; below MIN_OUTER_UNITS_FOR_INFERENCE"}
    c2["secondary_intermediate_delta"] = paired_delta_auroc(s_int, s_gen, harm, w)
    c2["absolute"] = {"generic_committed": all_cell_ranking(s_gen, harm, w), "task_sensitive": all_cell_ranking(s_task, harm, w),
                      "intermediate_supporting": all_cell_ranking(s_int, harm, w)}
    # supporting intermediate score (D8): the aligned cubic companion of the task length. Reported, never primary.
    c2["supporting_intermediate_scores"] = {
        "task_length": {**all_cell_ranking(s_task, harm, w), "role": "supporting; the quadratic task-sensitive score"},
        "cubic": supporting_intermediate_cubic(s_cubic, harm, w),
        "scores": {"task_length": "clean-calibrated ‖z − z̄‖_{M_task}",
                   "cubic": "clean-calibrated raw_task_cubic = Δ_a Δ_b Δ_c Î3_abc on the M_task-whitened chart, I3 fitted on reference_fit z"}}
    c2["cell_threshold"] = thr_cell
    for name, s in (("generic_committed", s_gen), ("task_sensitive", s_task)):
        c2[f"quadrants_{name}"] = alarm_harm_matrix(s, harm, thr_cell["threshold"], w)
        c2[f"missed_harm_{name}"] = missed_harm_rate_at_budget(s, harm, thr_cell["threshold"], w)
        c2[f"valid_rare_cell_rejection_{name}"] = valid_rare_cell_rejection(s, harm, valid_rare, thr_cell["threshold"])
        c2[f"conditional_triage_{name}"] = conditional_triage(s, harm, thr_cell["threshold"], w)
        c2[f"breakdown_contract_{name}"] = breakdown(s, harm, contracts_c, w)
    benign_vr_cells = set(cells_arr[valid_rare & (harm == "benign")].tolist())
    c2["valid_rare_window_rejection_generic"] = valid_rare_window_rejection(score_generic[ev_c2], window_thr, np.isin(cell_of[ev_c2], list(benign_vr_cells)))
    c2["cells"] = {str(c): {"origin": str(o), "contract": str(k), "K": float(K), "harm": str(h), "generic": float(a), "task": float(t),
                            "cubic": float(cb), "intermediate": float(i), "held_out_family": bool(hf)}
                   for c, o, k, K, h, a, t, cb, i, hf in zip(cells_arr, origins_c, contracts_c, K_cell, harm, s_gen, s_task, s_cubic, s_int, heldout_c)}

    # ---------------------------------------------------------------- 7. designed and task-specific controls (linear only)
    des: dict[str, Any] = {"note": "constructed from the frozen head; positive controls, linear subject only (NonlinearSubjectUnsupported otherwise); "
                                   "realised consequence measured per output; linearisation error reported"}
    X_ev, T_ev = ref_cell.batch(split.partitions["evaluation"])
    r_base = subject.represent(X_ev, ref_cell.geometry)
    base_err = np.mean(np.abs(r_base["output"] - T_ev), axis=0)
    for metric in ("euclidean", "null_mahalanobis"):
        des[metric] = {}
        for kind in ("output_null", "output_aligned", "random", "task_aligned", "task_null"):
            fam = DesignedFamily(kind, designed_norm, seed, match_metric=metric, task_target=TASK_TARGET if kind.startswith("task_") else None)
            try:
                Xp = fam.perturb(subject, ref, X_ev, ref_cell.geometry)
            except (NullSpaceUnavailable, NonlinearSubjectUnsupported) as e:
                des[metric][kind] = {"skipped": str(e)}; continue
            r1 = subject.represent(Xp, ref_cell.geometry)
            per_target = np.mean(np.abs(r1["output"] - T_ev), axis=0) / base_err
            des[metric][kind] = {"consequence_ratio_targets": {n: float(v) for n, v in zip(TARGET_NAMES, per_target)},
                                 "consequence_ratio_declared_K": float(per_target[TASK_TARGET]),
                                 "linearization": linearization_check(subject, ref, fam, X_ev, ref_cell.geometry)}

    # ---------------------------------------------------------------- 8. write
    elapsed = time.time() - t0
    result = {
        "STATUS": "DEVELOPMENT SMOKE — NOT CITABLE. Exercises the two-claim protocol chain on the linear subject at toy size; no scientific conclusion.",
        "provenance": provenance(config, mode=mode, seeds=[seed], repo_root=repo_root,
                                 extra={"elapsed_s": round(elapsed, 1), "subject": "LinearSubject", "cells": [c.label for c in cells]}),
        "gates": gates, "run_dependencies": run_deps,
        "manifest": manifest.to_dict(), "alarm_time_reference": atref.to_dict(), "arm_contract_status": arm_status,
        "split": split.to_dict(), "partition_counts": {p_: int(np.sum(part == p_)) for p_ in sorted(set(part.tolist()))},
        "far": far, "claim1": c1, "abstention": abst_report, "joint_decision_table": joint, "claim2": c2, "designed": des,
        "labels": {c: lab.to_dict() for c, lab in cell_labels.items()},
    }
    result = _js(result)
    (out / "smoke_report.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (out / "smoke_report.md").write_text(render_markdown(result))
    return result


def _f(x, nd=3) -> str:
    return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def render_markdown(r: dict[str, Any]) -> str:
    pv = r["provenance"]
    L = ["# Two-claim protocol smoke — DEVELOPMENT OUTPUT, NOT CITABLE", "", f"_{r['STATUS']}_", "",
         f"latent_monitor {pv['latent_monitor_version']} · mode `{pv['mode']}` · config hash `{pv['config_hash']}` · commit `{pv['git_commit']}` · "
         f"seeds {pv['seeds']} · {pv['elapsed_s']} s on `{pv['machine']}`", ""]
    L += ["**Gate warnings (dev mode continues; confirmatory would refuse):**", ""] + [f"- {w}" for w in r["run_dependencies"]["warnings"]] + [""]
    L += ["## Information contract", "", "| arm | features | status | data-flow checked |", "|---|---:|---|---|"]
    for a, s in r["arm_contract_status"].items():
        L.append(f"| {a} | {s['n_features']} | {s['status']} | {s['dataflow_checked']} |")
    far = r["far"]
    L += ["", f"Partitions (windows): {r['partition_counts']}. Held-out families: {r['split']['held_out_families']}; undeclared: {r['provenance']['config']['undeclared_families']}.",
          f"FAR budget {far['budget']}: {far['n_false_alerts']}/{far['n_clean_windows']} clean evaluation windows alerted (realised {_f(far['realised_far'])}, "
          f"CI95 {far['ci95']}); resolvable in one step: {far['resolvable_in_one_step']}.", ""]
    c1 = r["claim1"]
    L += ["## Claim 1 — incremental attribution over {N, S}", "",
          "| arm | role | status | F1 inclusive | F1 hard | F1 held-out fam. | features | λ |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for a, c in c1["arms"].items():
        L.append(f"| {a} | {c['role']} | {c['status']} | {_f(c['f1_eval_inclusive'])} | {_f(c['f1_eval_hard'])} | {_f(c['f1_eval_heldout_families'])} | {c['n_features']} | {c['lambda']:g} |")
    if c1["errors"]:
        L += ["", "Arms that failed to fit: " + "; ".join(f"{a}: {e}" for a, e in c1["errors"].items())]
    for key in ("primary_hard", "primary_inclusive", "primary_heldout_families", "control_matched_capacity", "control_drop_channel",
                "control_drop_token", "control_intermediate_alone", "diagnostic_noise_only_ablation", "diagnostic_legacy_layerwise"):
        d = c1[key]
        if "undefined_reason" in d and d.get("point") is None:
            L.append(f"\n**{key}** — {d['contrast']}: undefined ({d['undefined_reason']})"); continue
        L.append(f"\n**{key}** — {d['contrast']} [{d['set']}, n={d['n_windows']}]: ΔF1 = {_f(d['point'])} [{_f(d['low'])}, {_f(d['high'])}] "
                 f"over {d['n_groups']} event groups → **{d['reading']}** (failed replicates {d['n_failed']})")
    hm = c1["hard_matching"]
    L += ["", f"Hard matching: {hm['n_pairs']} pairs from {hm['n_S']} S / {hm['n_N']} N windows (retained S {_f(hm['retained_fraction_S'])}, "
          f"N {_f(hm['retained_fraction_N'])}; overlap failures {hm['overlap_failures_S']}; caliper {hm['caliper']}; median pair distance {_f(hm['pair_distance_median'])})."]
    hf = c1.get("hierarchical_family_outer_hard")
    if hf:
        L.append(f"Family-outer hierarchical interval on the hard set: [{_f(hf['low'])}, {_f(hf['high'])}] — {hf['inference']}.")
    L += ["", f"_{c1['note']}_", ""]
    if r["abstention"]:
        ab = r["abstention"]; rc = ab["risk_coverage"]
        L += ["## Abstention (support novelty, conformal on clean calibration windows)", "",
              f"α = {ab['conformal']['alpha']}, threshold {_f(ab['conformal']['threshold'])} from {ab['conformal']['n_calibration']} clean windows; "
              f"finite-sample clean-retention bound {_f(ab['conformal']['finite_sample_clean_retention_bound'])}; realised clean retention {_f(ab['clean_retention_at_threshold'])}.",
              f"Support-estimator control AUROC (held-out clean vs displaced): {_f(ab['support_estimator_validation']['auroc_in_vs_out'])} "
              f"(usable: {ab['support_estimator_validation']['usable_for_abstention']}).",
              f"Unknown-family AUROC: {_f(ab['unknown_auroc']['auroc'])} ({ab['counts']['n_unknown']} unknown windows; {ab['counts']['n_unknown_abstained']} abstained, "
              f"{ab['counts']['n_known_abstained']} known abstained). Risk–coverage AUC {_f(rc['risk_coverage_auc'])}.", "",
              "| coverage | risk | retained known | retained unknown | retained-known macro-F1 |", "|---:|---:|---:|---:|---:|"]
        for c, pt in rc["points"].items():
            L.append(f"| {c} | {_f(pt['risk'])} | {pt['n_retained_known']} | {pt['n_retained_unknown']} | {_f(pt['retained_known_macro_f1'])} |")
        L.append("")
    if r["joint_decision_table"]:
        jt = r["joint_decision_table"]
        L += ["## Joint operational table (detect → abstain → attribute)", "", "| category | n | quiet | abstain | N | S | correct |", "|---|---:|---:|---:|---:|---:|---:|"]
        for c, t in jt["per_category"].items():
            if t["n"] == 0:
                L.append(f"| {c} | 0 | | | | | — |"); continue
            d = t["decisions"]
            L.append(f"| {c} | {t['n']} | {d['quiet']} | {d['abstain']} | {d['N']} | {d['S']} | {_f(t['correct_rate'])} ({'/'.join(t['correct_decisions'])}) |")
        L += ["", f"Overall correct-decision rate {_f(jt['overall_correct_rate'])}; {jt['n_G_windows_not_tabulated']} geometry windows not tabulated. _{jt['note']}_", ""]
    c2 = r["claim2"]; pd_ = c2["primary_delta"]; ci = pd_["ci_hierarchical_outer_cell"]
    L += ["## Claim 2 — harm ranking: task-sensitive score vs committed generic score", "",
          f"κ_m: {c2['kappa_m']['value']} ({c2['kappa_m']['status']}) — {c2['kappa_m']['source']}", "",
          f"**Primary ΔAUROC** = {_f(pd_['delta_auroc'])} (task {_f(pd_['auroc_task_sensitive'])} − generic {_f(pd_['auroc_generic_committed'])}; "
          f"{pd_['n_pos']} harmful / {pd_['n_neg']} benign cells) — hierarchical interval [{_f(ci['low'])}, {_f(ci['high'])}] over {ci['n_outer']} cells: **{ci['inference']}**."]
    if "primary_delta_heldout_families_only" in c2:
        h = c2["primary_delta_heldout_families_only"]
        L.append(f"Held-out families only: ΔAUROC {_f(h['delta_auroc'])} on {h['n_cells']} cells — {h['note']}.")
    si = c2["secondary_intermediate_delta"]
    if c2["primary_delta"].get("undefined_reason"):
        L.append(f"Primary undefined: {c2['primary_delta']['undefined_reason']}.")
    L += [f"Supporting: intermediate score ΔAUROC {_f(si['delta_auroc'])}; "
          f"cubic intermediate score (D8) absolute AUROC {_f(c2['supporting_intermediate_scores']['cubic']['auroc'])} "
          f"(task length {_f(c2['supporting_intermediate_scores']['task_length']['auroc'])}); "
          f"conditional triage (generic) AUROC {_f(c2['conditional_triage_generic_committed']['auroc'])}, "
          f"dropped {c2['conditional_triage_generic_committed']['n_dropped_by_conditioning']} cells ({c2['conditional_triage_generic_committed']['n_dropped_harmful']} harmful).",
          f"Cell threshold {_f(c2['cell_threshold']['threshold'])} (pseudo-cell bootstrap of {c2['cell_threshold']['n_clean_windows']} clean windows, cell size {c2['cell_threshold']['cell_size']}); "
          f"missed harm at budget: generic {_f(c2['missed_harm_generic_committed']['missed_harm_rate'])}, task {_f(c2['missed_harm_task_sensitive']['missed_harm_rate'])}; "
          f"valid-rare cell rejection: generic {_f(c2['valid_rare_cell_rejection_generic_committed']['valid_rare_rejection_rate'])}, task {_f(c2['valid_rare_cell_rejection_task_sensitive']['valid_rare_rejection_rate'])}; "
          f"valid-rare window rejection (generic) {_f(c2['valid_rare_window_rejection_generic']['valid_rare_window_rejection_rate'])}.", ""]
    for name in ("generic_committed", "task_sensitive"):
        m = c2[f"quadrants_{name}"]
        L += [f"| {name} | K < κ_m | K ≥ κ_m |", "|---|---:|---:|", f"| low alarm | {m['low_alarm_benign']['n']} | {m['low_alarm_harmful']['n']} |",
              f"| strong alarm | {m['strong_alarm_benign']['n']} | {m['strong_alarm_harmful']['n']} |", ""]
    L += ["| cell | origin | K_phys | harm | generic | task | cubic | intermediate | held-out |", "|---|---|---:|---|---:|---:|---:|---:|---|"]
    for name, cc in c2["cells"].items():
        L.append(f"| {name} | {cc['origin']} | {cc['K']:.2f} | {cc['harm']} | {cc['generic']:.2f} | {cc['task']:.2f} | {cc['cubic']:.2f} | {cc['intermediate']:.2f} | {cc['held_out_family']} |")
    L += ["", "## Designed and task-specific controls (linear subject; positive controls)", "",
          "| match metric | family | consequence ratio (amp, t0, τ) | declared-K ratio | lin. err Δy |", "|---|---|---|---:|---:|"]
    for metric in ("euclidean", "null_mahalanobis"):
        for kind, d in r["designed"][metric].items():
            if "skipped" in d:
                L.append(f"| {metric} | {kind} | skipped: {d['skipped']} | | |"); continue
            ct = ", ".join(f"{v:.2f}" for v in d["consequence_ratio_targets"].values())
            L.append(f"| {metric} | {kind} | {ct} | {d['consequence_ratio_declared_K']:.2f} | {d['linearization']['rel_err_dy']:.1e} |")
    L += ["", f"_{r['designed']['note']}_", ""]
    return "\n".join(L)


def _main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=Path("results/latent_monitor_smoke_dev_2026-09-16_pass2"))
    ap.add_argument("--mode", default="dev", choices=["dev", "confirmatory"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-groups", type=int, default=120)
    ap.add_argument("--channels", type=int, default=6)
    ap.add_argument("--samples", type=int, default=128)
    a = ap.parse_args(argv)
    r = run_smoke(a.out, seed=a.seed, n_groups=a.n_groups, n_channels=a.channels, n_samples=a.samples, mode=a.mode)
    print(r["STATUS"])
    d = r["claim1"]["primary_hard"]
    print(f"Claim 1 ΔF1 (hard set): {d.get('point')} [{d.get('low')}, {d.get('high')}] → {d.get('reading')}")
    p = r["claim2"]["primary_delta"]
    print(f"Claim 2 ΔAUROC: {p['delta_auroc']} (task {p['auroc_task_sensitive']} vs generic {p['auroc_generic_committed']}); {p['ci_hierarchical_outer_cell']['inference']}")
    print(f"-> {a.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
