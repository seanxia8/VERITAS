# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The amended protocol (plan §8.3 R2–R5): leakage, splits, metric conditioning, gates.

Classification per REVIEW_PROMPTS §B.2: ANALYTIC (closed-form answer),
CONTROL (predicted direction on a constructed case), STRUCTURAL (refusal /
shape). Each test says which.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from latent_monitor.protocol import (
    AlarmTimeContract, ConfirmatoryGateError, Feature, FeatureManifest, GroupSplit, HarmThreshold, LeakageError,
    OverlapError, PendingThresholdError, ProtocolMode, alarm_harm_matrix, all_cell_ranking, assert_disjoint,
    assign_partitions, bootstrap_groups, breakdown, conditional_triage, harm_labels, label_from_legacy,
    missed_harm_rate_at_budget, require_gates, split_event_groups, tier1_manifest, valid_rare_cell_rejection,
    weighted_auroc,
)
from latent_monitor.protocol.arms import ARMS, CONTROL_ARMS, arm_feature_names, check_arms, fit_logistic, macro_f1, read_delta
from latent_monitor.protocol.consequence import auprc
from latent_monitor.protocol.labels import CONTRACT_OF_ORIGIN, LEGACY_MOVED_TO_ORIGIN, CellLabel
from latent_monitor.protocol.mode import freeze_part


# ----------------------------------------------------------------- labels (STRUCTURAL)

def test_legacy_labels_map_onto_separate_origin_and_harm_axes():
    lab = label_from_legacy("event_in_span:double_pulse", "event_in_span")
    assert lab.origin == "S_in_span" and lab.contract == "S" and lab.harm == "undefined" and lab.family == "double_pulse"
    assert label_from_legacy("sigma_cov:corr_up", "sigma_cov").contract == "N"
    assert set(LEGACY_MOVED_TO_ORIGIN.values()) <= set(CONTRACT_OF_ORIGIN)
    with pytest.raises(ValueError):
        CellLabel("x", "unknown", "u", declared=True)          # an unknown family can never be declared
    with pytest.raises(KeyError):
        label_from_legacy("e", "E")                             # a bare 'E' has no meaning in the new namespace


# ----------------------------------------------------------------- availability (STRUCTURAL: the contract refuses leakage)

def test_alarm_time_contract_rejects_evaluation_only_and_delayed_features():
    m = tier1_manifest()
    c = AlarmTimeContract(m)
    ok = c.check("generic_rich", arm_feature_names(m, "generic_rich"))
    assert ok["status"] == "operational" and ok["dataflow_checked"] is False
    with pytest.raises(LeakageError, match="twin_dz"):
        c.check("cheating", ["z_0", "twin_dz"])
    with pytest.raises(LeakageError, match="truth_targets"):
        c.check("cheating", ["truth_targets"])
    with pytest.raises(LeakageError, match="task_metric_delayed"):
        c.check("cheating", ["task_metric_delayed"])
    with pytest.raises(KeyError):
        c.check("cheating", ["a_feature_nobody_declared"])      # undeclared = leakage by default


def test_noise_only_features_are_privileged_when_the_acquisition_has_none():
    m = tier1_manifest(supplies_noise_only_records=False)
    st = check_arms(m)
    assert st["generic_rich"]["status"] == "privileged" and "no_var_ratio" in st["generic_rich"]["privileged_features"]
    assert st["all_generic_no_noise"]["status"] == "operational"
    assert st["full_intermediate"]["status"] == "privileged"   # the intermediate arm gets exactly the same side information
    assert st["intermediate_only"]["status"] == "operational"  # internal hooks alone need no noise-only record


def test_generic_rich_and_full_intermediate_differ_only_by_internal_hooks():
    m = tier1_manifest()
    g, l = set(arm_feature_names(m, "generic_rich")), set(arm_feature_names(m, "full_intermediate"))
    assert g < l and all(n.startswith("im_") for n in l - g)
    assert not any(n.startswith(("in_", "out_", "z_", "gr_", "lw_")) for n in arm_feature_names(m, "intermediate_only"))
    assert len(arm_feature_names(m, "generic_rich_matched")) == len(arm_feature_names(m, "full_intermediate"))


# ----------------------------------------------------------------- splits (STRUCTURAL: overlap and unknown families refuse)

def test_split_is_disjoint_and_keeps_every_variant_of_a_group_together():
    split = split_event_groups(range(50), {"reference_fit": 0.3, "attribution_train": 0.2, "development": 0.15, "calibration": 0.1, "evaluation": 0.25}, seed=3)
    assert_disjoint(split)
    assert sum(len(v) for v in split.partitions.values()) == 50
    recs = [{"event_group": g, "family": fam, "declared": True} for g in range(50) for fam in ("clean", "corr_up", "geom_12", "glitch")]
    parts = assign_partitions(recs, split)
    by_group = {}
    for r, p in zip(recs, parts):
        by_group.setdefault(r["event_group"], set()).add(p)
    assert all(len(s) == 1 for s in by_group.values()), "a replay/corruption variant left its group's partition"


def test_overlapping_groups_raise():
    bad = GroupSplit(partitions={"reference_fit": np.array([1, 2, 3]), "evaluation": np.array([3, 4])})
    with pytest.raises(OverlapError, match="event group 3"):
        assert_disjoint(bad)


def test_unknown_and_held_out_families_never_enter_fit_tuning_or_calibration():
    split = split_event_groups(range(40), {"reference_fit": 0.5, "development": 0.2, "calibration": 0.1, "evaluation": 0.2},
                               seed=0, held_out_families=("line_pickup",), held_out_seeds=(7,), held_out_severities=((0.9, 1.0),))
    recs = [{"event_group": g, "family": "wimp", "declared": False} for g in range(40)]
    recs += [{"event_group": g, "family": "line_pickup", "declared": True} for g in range(40)]
    recs += [{"event_group": g, "family": "corr_up", "declared": True, "seed": 7} for g in range(40)]
    recs += [{"event_group": g, "family": "corr_up", "declared": True, "severity": 0.95} for g in range(40)]
    parts = assign_partitions(recs, split)
    assert set(parts) <= {"evaluation", "excluded"}
    n_eval_groups = len(split.partitions["evaluation"])
    assert parts.count("evaluation") == 4 * n_eval_groups
    with pytest.raises(KeyError):
        assign_partitions([{"event_group": 999, "family": "corr_up"}], split)


# ----------------------------------------------------------------- consequence (ANALYTIC + CONTROL)

def test_weighted_auroc_matches_closed_form_and_handles_ties_and_empty_classes():
    r = weighted_auroc(np.array([0.1, 0.4, 0.35, 0.8]), np.array([False, False, True, True]))
    assert r["auroc"] == pytest.approx(0.75)
    assert weighted_auroc(np.array([1.0, 1.0]), np.array([True, False]))["auroc"] == pytest.approx(0.5)
    # weights: duplicating a unit equals weighting it 2
    a = np.array([0.2, 0.9, 0.5, 0.7]); y = np.array([False, True, False, True])
    dup = weighted_auroc(np.concatenate([a, a[:1]]), np.concatenate([y, y[:1]]))["auroc"]
    assert weighted_auroc(a, y, np.array([2.0, 1, 1, 1]))["auroc"] == pytest.approx(dup)
    e = weighted_auroc(np.array([0.1, 0.2]), np.array([True, True]))
    assert np.isnan(e["auroc"]) and e["undefined_reason"]
    assert auprc(np.array([0.9, 0.8, 0.1]), np.array([True, True, False]))["auprc"] == pytest.approx(1.0)


def test_pending_kappa_m_makes_harm_undefined_and_confirmatory_refuses(tmp_path):
    with pytest.raises(PendingThresholdError):
        harm_labels(np.array([1.0, 2.0]), HarmThreshold(None, "pending", arm="tier1"))
    with pytest.raises(ValueError):
        HarmThreshold(None, "declared")
    thr = HarmThreshold(1.1, "provisional_dev")
    dev = require_gates("dev", freezes=("core",), thresholds={"k": thr}, repo_root=tmp_path)
    assert dev["citable"] is False and len(dev["warnings"]) == 1          # provisional_dev is usable in dev
    pend = require_gates("dev", freezes=("core",), thresholds={"k": HarmThreshold(None, "pending")}, repo_root=tmp_path)
    assert len(pend["warnings"]) == 2 and "pending" in pend["warnings"][1]
    with pytest.raises(ConfirmatoryGateError, match="freeze part 'core' is missing"):
        require_gates(ProtocolMode.CONFIRMATORY, freezes=("core",), thresholds={"k": thr}, repo_root=tmp_path)


def test_freeze_hash_check_detects_a_stale_protocol(tmp_path):
    src = tmp_path / "docs" / "PREREGISTRATION.md"; src.parent.mkdir()
    src.write_text("thresholds: x")
    freeze_part("core", src, tmp_path / "protocol" / "frozen", commit="deadbeef")
    ok = require_gates("confirmatory", freezes=("core",), thresholds={"k": HarmThreshold(0.2, "declared", source="req")}, repo_root=tmp_path)
    assert ok["citable"] and ok["freezes"]["core"]["commit"] == "deadbeef"
    src.write_text("thresholds: y")                              # someone edits one threshold after the freeze
    with pytest.raises(ConfirmatoryGateError, match="stale"):
        require_gates("confirmatory", freezes=("core",), thresholds={"k": HarmThreshold(0.2, "declared")}, repo_root=tmp_path)
    with pytest.raises(ConfirmatoryGateError, match="provisional_dev"):
        require_gates("confirmatory", freezes=(), thresholds={"k": HarmThreshold(0.2, "provisional_dev")}, repo_root=tmp_path)


def test_harmful_low_alarm_cells_change_the_primary_result_but_not_the_conditional_one():
    """CONTROL (plan §8.3 R3 acceptance): conditioning on the alarm hides the low-alarm harmful quadrant; the all-cell endpoint does not."""
    thr = HarmThreshold(1.1, "provisional_dev")
    # six cells: four strong-alarm (two harmful, two benign, alarm ranks them perfectly) ...
    alarm = np.array([3.0, 2.5, 1.5, 1.2])
    K = np.array([2.0, 1.5, 1.0, 0.9])
    harm = harm_labels(K, thr)
    base = all_cell_ranking(alarm, harm)["auroc"]
    cond = conditional_triage(alarm, harm, 1.0)["auroc"]
    assert base == pytest.approx(1.0) and cond == pytest.approx(1.0)
    # ... then add two low-alarm cells that are harmful (a clean rare event with a large consequence)
    alarm2 = np.concatenate([alarm, [0.2, 0.3]]); K2 = np.concatenate([K, [3.0, 2.5]])
    harm2 = harm_labels(K2, thr)
    primary = all_cell_ranking(alarm2, harm2)
    cond2 = conditional_triage(alarm2, harm2, 1.0)
    assert primary["auroc"] < 0.8, "the all-cell ranking must degrade when harm hides below the alarm threshold"
    assert cond2["auroc"] == pytest.approx(1.0), "the conditional endpoint cannot see them"
    assert cond2["n_dropped_harmful"] == 2
    m = alarm_harm_matrix(alarm2, harm2, 1.0)
    assert m["low_alarm_harmful"]["n"] == 2 and m["strong_alarm_benign"]["n"] == 2 and m["strong_alarm_harmful"]["n"] == 2 and m["low_alarm_benign"]["n"] == 0
    assert missed_harm_rate_at_budget(alarm2, harm2, 1.0)["missed_harm_rate"] == pytest.approx(0.5)
    rej = valid_rare_cell_rejection(alarm2, harm2, np.array([False, False, True, True, False, False]), 1.0)
    assert rej["valid_rare_rejection_rate"] == pytest.approx(1.0) and rej["n_valid_rare_benign"] == 2   # both benign, both alarmed
    rej2 = valid_rare_cell_rejection(np.array([0.5, 1.5]), harm_labels(np.array([0.9, 0.9]), thr), np.array([True, True]), 1.0)
    assert rej2["valid_rare_rejection_rate"] == pytest.approx(0.5)
    # undefined K stays in the report as undefined, never silently dropped
    harm3 = harm_labels(np.array([2.0, np.nan]), thr)
    assert list(harm3) == ["harmful", "undefined"] and all_cell_ranking(np.array([1.0, 2.0]), harm3)["n_undefined"] == 1


def test_breakdown_reports_single_class_strata_as_undefined_not_dropped():
    thr = HarmThreshold(1.1, "provisional_dev")
    harm = harm_labels(np.array([2.0, 0.5, 2.0, 3.0]), thr)
    b = breakdown(np.array([1.0, 0.5, 2.0, 3.0]), harm, np.array(["N", "N", "S", "S"], dtype=object))
    assert b["N"]["auroc"] == pytest.approx(1.0) and np.isnan(b["S"]["auroc"]) and b["S"]["undefined_reason"]


def test_bootstrap_resamples_groups_not_units():
    """ANALYTIC: with one unit per group the bootstrap of the mean has the textbook spread; with all units in one group it has none."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    one_per = bootstrap_groups(lambda idx: float(np.mean(x[idx])), np.arange(400), n_boot=300, seed=1)
    assert one_per["high"] - one_per["low"] == pytest.approx(2 * 1.96 * x.std() / 20, rel=0.25)
    same = bootstrap_groups(lambda idx: float(np.mean(x[idx])), np.zeros(400, dtype=int), n_boot=50, seed=1)
    assert same["high"] == same["low"] == pytest.approx(x.mean())
    crossed = bootstrap_groups(lambda idx: float(np.mean(x[idx])), np.arange(400) // 4, seeds=np.arange(400) % 4, n_boot=50, seed=1)
    assert crossed["units"].startswith("event_group × perturbation_seed")


# ----------------------------------------------------------------- arms (CONTROL: separable classes; identical budget)

def test_logistic_arm_recovers_a_separable_rule_and_delta_reading_follows_the_margins():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 3)); y = np.where(X[:, 0] > 0, "N", "S")
    m = fit_logistic(X, y, ("N", "S"), lam=1e-2)
    assert macro_f1(y, m.predict(X), ("N", "S")) > 0.95
    assert read_delta(0.15, 0.05, 0.25) == "benefit"
    assert read_delta(0.01, -0.04, 0.04) == "equivalence"
    assert read_delta(0.15, -0.02, 0.3) == "inconclusive"
    assert read_delta(-0.2, -0.3, -0.1) == "detriment"
    assert read_delta(0.02, -0.1, 0.12) == "inconclusive"
    assert read_delta(0.0, 0.0, 0.0).startswith("inconclusive")                      # degenerate interval
    assert read_delta(0.15, 0.05, 0.25, n_windows=12).startswith("inconclusive")     # below the declared minimum


def test_every_arm_is_declared_in_terms_of_manifest_groups():
    m = tier1_manifest()
    for a in tuple(ARMS) + CONTROL_ARMS:
        assert arm_feature_names(m, a), a
