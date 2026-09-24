# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Pass-2 protocol repairs (TWO_CLAIM_REVISION_PLAN §4 I1–I13). Classification: ANALYTIC / CONTROL / STRUCTURAL."""

from __future__ import annotations

import numpy as np
import pytest

from latent_monitor.protocol import (
    AlarmTimeInputs, ConfirmatoryGateError, ConformalNovelty, FeatureBatch, HarmThreshold, InvalidWeightsError, LeakageError,
    RunDependencies, bootstrap_hierarchical, cell_aggregate, cell_alarm_threshold, evaluate_abstention, far_precision,
    harm_labels, hard_match, joint_decision_table, paired_delta_auroc, require_run_dependencies, risk_coverage,
    split_event_groups, tier1_manifest, valid_rare_window_rejection, weighted_auroc,
)
from latent_monitor.protocol.arms import ARMS, ArmFitError, arm_feature_names, fit_arm, fit_logistic, macro_f1, macro_f1_report
from latent_monitor.protocol.availability import ALARM_TIME_SOURCES, AlarmTimeContract
from latent_monitor.protocol.consequence import auprc, MIN_OUTER_UNITS_FOR_INFERENCE
from latent_monitor.protocol.features import AlarmTimeReference, NullCalibrator, build_alarm_time_features
from latent_monitor.protocol.mode import freeze_part, sha256_file
from latent_monitor.protocol.splits import PARTITIONS, assign_partitions

from .conftest import EVAL_IDS, FIT_IDS


# ----------------------------------------------------------------- I2: data-flow contract (STRUCTURAL, adversarial)

def test_feature_batch_refuses_forbidden_sources_and_locks():
    b = FeatureBatch()
    b.add("in_rms", np.ones(3), "raw_window")
    with pytest.raises(LeakageError, match="forbidden source"):
        b.add("z_0", np.ones(3), "truth")
    with pytest.raises(LeakageError, match="forbidden source"):
        b.add("z_1", np.ones(3), {"raw_window", "twin"})
    with pytest.raises(LeakageError, match="no source"):
        b.add("z_2", np.ones(3), set())
    with pytest.raises(ValueError, match="rows"):
        b.add("z_3", np.ones(4), "raw_window")
    b.lock()
    with pytest.raises(RuntimeError, match="locked"):
        b.add("late", np.ones(3), "raw_window")


def test_adversarial_builder_cannot_smuggle_truth(subject, reference, ref_cell):
    """An implementation that computes an *allowed name* from truth: (a) the typed input object has no truth field, so the
    only way in is an extra argument the reviewer can see; (b) if it tags the value honestly it is refused at insertion;
    (c) if it lies about the tag nothing catches it — that residual limitation is the reason code review stays mandatory."""
    X, T = ref_cell.batch(EVAL_IDS[:8])
    inputs = AlarmTimeInputs(X=X, geometry=ref_cell.geometry)
    assert not hasattr(inputs, "truth") and not hasattr(inputs, "twin") and not hasattr(inputs, "sigma_realized")
    assert "truth" not in inputs.tags and inputs.tags <= ALARM_TIME_SOURCES
    atref = AlarmTimeReference.fit(subject, reference, ref_cell.batch(FIT_IDS[:60])[0], ref_cell.geometry)
    honest = build_alarm_time_features(subject, atref, inputs)
    assert honest.locked and all(honest.sources(n) <= ALARM_TIME_SOURCES for n in honest.names())

    def adversarial_builder(inputs: AlarmTimeInputs, truth: np.ndarray) -> FeatureBatch:   # the extra argument is the visible violation
        b = FeatureBatch(n=inputs.n)
        b.add("in_rms", truth[:, 0], "truth")                                              # honest tag → refused
        return b.lock()

    with pytest.raises(LeakageError):
        adversarial_builder(inputs, T)

    def lying_builder(inputs: AlarmTimeInputs, truth: np.ndarray) -> FeatureBatch:
        b = FeatureBatch(n=inputs.n)
        b.add("in_rms", truth[:, 0], "raw_window")                                         # a lie: not catchable by the contract
        return b.lock()

    lied = lying_builder(inputs, T)
    AlarmTimeContract(tier1_manifest()).check("input_only", ["in_rms"], lied)              # passes — the documented residual limitation
    assert np.allclose(lied["in_rms"], T[:, 0])


def test_contract_requires_a_locked_batch_and_source_consistency():
    m = tier1_manifest()
    c = AlarmTimeContract(m)
    b = FeatureBatch()
    for n in arm_feature_names(m, "input_only"):
        b.add(n, np.zeros(2), "raw_window")
    with pytest.raises(LeakageError, match="locked"):
        c.check("input_only", arm_feature_names(m, "input_only"), b)
    b.lock()
    assert c.check("input_only", arm_feature_names(m, "input_only"), b)["dataflow_checked"]
    with pytest.raises(KeyError):
        c.check("output_uncertainty", arm_feature_names(m, "output_uncertainty"), b)


# ----------------------------------------------------------------- I6: five disjoint partitions (STRUCTURAL)

def test_five_partitions_are_disjoint_and_reference_fit_is_never_trained_on():
    fr = {"reference_fit": 0.3, "attribution_train": 0.2, "development": 0.15, "calibration": 0.15, "evaluation": 0.2}
    split = split_event_groups(range(100), fr, seed=1)
    assert set(split.partitions) == set(PARTITIONS)
    ids = [set(v.tolist()) for v in split.partitions.values()]
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            assert not (ids[i] & ids[j])
    with pytest.raises(ValueError, match="unknown partitions"):
        split_event_groups(range(10), {"reference_fit": 0.5, "train": 0.5})


def test_unknown_only_evaluation_and_fit_refuses_undeclared_class():
    """Unknown windows are evaluation-only and never a training class; a training set containing one raises."""
    with pytest.raises(ArmFitError, match="undeclared class"):
        fit_logistic(np.random.default_rng(0).normal(size=(10, 2)), ["N", "S", "U"] * 3 + ["N"], ("N", "S"), 0.1)
    with pytest.raises(ArmFitError, match="lacks class"):
        fit_logistic(np.random.default_rng(0).normal(size=(6, 2)), ["N"] * 6, ("N", "S"), 0.1)


# ----------------------------------------------------------------- I9: metric edge cases (ANALYTIC)

def test_macro_f1_missing_class_policy():
    assert macro_f1(["N", "S"], ["N", "N"], ("N", "S")) == pytest.approx(np.mean([2 / 3, 0.0]))   # S present, never predicted → 0
    rep = macro_f1_report(["N", "N"], ["N", "S"], ("N", "S"))
    assert np.isnan(rep["macro_f1"]) and "absent from the truth" in rep["undefined_reason"]
    assert np.isnan(macro_f1([], [], ("N", "S")))


def test_auprc_is_permutation_invariant_under_ties():
    rng = np.random.default_rng(0)
    s = np.array([0.9, 0.9, 0.9, 0.5, 0.5, 0.1]); y = np.array([1, 0, 1, 1, 0, 0], bool)
    base = auprc(s, y)["auprc"]
    for _ in range(20):
        p = rng.permutation(len(s))
        assert auprc(s[p], y[p])["auprc"] == pytest.approx(base)
    # all tied → precision = prevalence at the single threshold
    assert auprc(np.ones(6), y)["auprc"] == pytest.approx(0.5)


def test_invalid_weights_are_refused_everywhere():
    s = np.array([0.1, 0.9]); y = np.array([False, True])
    for w in (np.array([1.0]), np.array([1.0, np.nan]), np.array([-1.0, 1.0]), np.array([0.0, 0.0])):
        with pytest.raises(InvalidWeightsError):
            weighted_auroc(s, y, w)
        with pytest.raises(InvalidWeightsError):
            auprc(s, y, w)


def test_optimiser_and_partition_failures_are_informative():
    with pytest.raises(ArmFitError, match="empty design"):
        fit_logistic(np.zeros((0, 2)), [], ("N", "S"), 0.1)
    with pytest.raises(ArmFitError, match="invalid λ"):
        fit_logistic(np.zeros((4, 2)), ["N", "S", "N", "S"], ("N", "S"), -1.0)
    m = tier1_manifest()
    b = FeatureBatch()
    for n in arm_feature_names(m, "input_only"):
        b.add(n, np.random.default_rng(0).normal(size=6), "raw_window")
    b.lock()
    y = np.array(["N", "S", "N", "S", "N", "S"], dtype=object)
    with pytest.raises(ArmFitError, match="partition 'development' is empty"):
        fit_arm("input_only", m, b, y, np.array(["attribution_train"] * 4 + ["evaluation"] * 2, dtype=object), ("N", "S"))


# ----------------------------------------------------------------- I5: hierarchical resampling (ANALYTIC + STRUCTURAL)

def test_hierarchical_bootstrap_resamples_outer_units_and_refuses_inference_when_few():
    rng = np.random.default_rng(0)
    n_outer = 20
    outer = np.repeat(np.arange(n_outer), 10); inner = np.arange(200)
    cell_effect = rng.normal(size=n_outer)[outer]
    x = cell_effect + 0.1 * rng.normal(size=200)
    hier = bootstrap_hierarchical(lambda idx: float(np.mean(x[idx])), outer, inner, n_boot=300, seed=1)
    assert hier["inference"].startswith("outer-unit bootstrap") and hier["n_outer"] == 20
    # the outer-level spread is the between-cell spread, much wider than an event-only bootstrap conditional on the cells
    from latent_monitor.protocol import bootstrap_groups
    ev_only = bootstrap_groups(lambda idx: float(np.mean(x[idx])), inner, n_boot=300, seed=1)
    assert (hier["high"] - hier["low"]) > 2 * (ev_only["high"] - ev_only["low"])
    few = bootstrap_hierarchical(lambda idx: float(np.mean(x[idx])), outer[:40], inner[:40], n_boot=50, seed=1)
    assert few["n_outer"] == 4 < MIN_OUTER_UNITS_FOR_INFERENCE and few["inference"].startswith("descriptive only")
    # failed replicates are counted, not hidden
    bad = bootstrap_hierarchical(lambda idx: float("nan"), outer, inner, n_boot=10, seed=1)
    assert bad["n_failed"] == 10 and bad["inference"] == "undefined"
    # model seeds as an outer-most level are declared in the units string
    ms = bootstrap_hierarchical(lambda idx: float(np.mean(x[idx])), outer, inner, n_boot=20, seed=1, model_seeds=np.repeat([0, 1], 100))
    assert ms["units"].startswith("model_seed") and ms["n_model_seeds"] == 2


def test_paired_delta_auroc_and_cell_level_threshold():
    thr = HarmThreshold(1.1, "provisional_dev")
    harm = harm_labels(np.array([2.0, 0.5, 1.5, 0.9]), thr)
    d = paired_delta_auroc(np.array([3, 1, 2, 0.5]), np.array([1, 3, 2, 0.5]), harm)
    assert d["delta_auroc"] == pytest.approx(1.0 - 0.5) and d["level"] == "cell"
    cells, vals = cell_aggregate(np.array([1.0, 3.0, 2.0, 10.0]), np.array(["a", "a", "b", "b"], dtype=object), "median")
    assert list(cells) == ["a", "b"] and list(vals) == [2.0, 6.0]
    t = cell_alarm_threshold(np.random.default_rng(0).normal(size=50), cell_size=8, far=0.05, n_draws=500)
    assert t["level" if "level" in t else "aggregate"] and t["resolvable"] and t["threshold"] < 2.0
    w = valid_rare_window_rejection(np.array([0.5, 1.5, 2.0]), 1.0, np.array([True, True, False]))
    assert w["valid_rare_window_rejection_rate"] == pytest.approx(0.5) and w["level"] == "window"


def test_far_precision_reports_a_binomial_interval_and_resolvability():
    r = far_precision(18, 1, 0.01)
    assert r["realised_far"] == pytest.approx(1 / 18) and r["ci95"][0] > 0 and r["ci95"][1] < 1 and not r["resolvable_in_one_step"]
    r0 = far_precision(200, 0, 0.01)
    assert r0["ci95"][0] == 0.0 and r0["resolvable_in_one_step"]


# ----------------------------------------------------------------- I1: abstention (CONTROL)

def test_conformal_abstention_calibrates_on_clean_only_and_measures_unknowns_empirically():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=200)
    conf = ConformalNovelty.fit(clean, alpha=0.1, score_name="novelty")
    assert conf.n_calibration == 200 and conf.finite_sample_bound == pytest.approx(0.9 - 1 / 201)
    fresh_clean = rng.normal(size=5000)
    assert abs(np.mean(~conf.abstain(fresh_clean)) - 0.9) < 0.03           # retention ≈ 1 − α on exchangeable clean windows
    known = rng.normal(size=100); unknown = rng.normal(size=50) + 3.0
    nov = np.concatenate([known, unknown]); is_unk = np.concatenate([np.zeros(100, bool), np.ones(50, bool)])
    y_true = np.array(["N"] * 50 + ["S"] * 50 + ["U"] * 50, dtype=object)
    y_pred = np.array(["N"] * 50 + ["S"] * 50 + ["N"] * 50, dtype=object)
    rep = evaluate_abstention(conf, nov, y_true, y_pred, is_unk, ("N", "S"), novelty_clean_eval=fresh_clean[:500])
    assert rep.unknown_auroc["auroc"] > 0.95 and rep.counts["n_unknown_abstained"] > 40
    rc = rep.risk_coverage
    assert rc["points"]["0.5"]["risk"] < rc["points"]["1.0"]["risk"]       # retained unknowns count as errors: risk rises with coverage
    assert rc["points"]["1.0"]["n_retained_unknown"] == 50
    # a novelty that cannot see unknowns gives AUROC ≈ 0.5 and no false comfort
    rep2 = evaluate_abstention(conf, np.concatenate([known, rng.normal(size=50)]), y_true, y_pred, is_unk, ("N", "S"))
    assert abs(rep2.unknown_auroc["auroc"] - 0.5) < 0.15


# ----------------------------------------------------------------- I7: hard contrasts and the joint table (STRUCTURAL + CONTROL)

def test_hard_matching_retains_overlapping_windows_and_reports_failures():
    rng = np.random.default_rng(0)
    n = 60
    f = {"in_rms": np.concatenate([rng.normal(size=20), rng.normal(size=20), rng.normal(size=20) + 6]),
         "in_kurtosis": rng.normal(size=n), "z_mahalanobis": rng.normal(size=n), "gr_output_maha": rng.normal(size=n)}
    contract = np.array(["clean"] * 20 + ["N"] * 20 + ["S"] * 20, dtype=object)
    m = hard_match(f, contract, contract == "clean", caliper=1.0)
    assert m["n_pairs"] == 0 and m["overlap_failures_S"] == 20                # S far from every N: nothing matches
    f["in_rms"][40:] = f["in_rms"][20:40]                                      # now S overlaps N exactly on one variable
    m2 = hard_match(f, contract, contract == "clean", caliper=3.0)
    assert m2["n_pairs"] > 10 and m2["retained_fraction_S"] == m2["n_pairs"] / 20 and len(m2["rows"]) == 2 * m2["n_pairs"]
    assert len(set(m2["rows_N"].tolist())) == m2["n_pairs"], "without replacement"


def test_joint_decision_table_scores_the_whole_chain_per_category():
    alarm = np.array([0.1, 2.0, 2.0, 2.0, 2.0, 0.2])
    abst = np.array([False, False, True, False, False, False])
    att = np.array(["N", "N", "N", "S", "S", "S"], dtype=object)
    cat = np.array(["clean", "N", "unknown", "S", "mixture", "unknown"], dtype=object)
    t = joint_decision_table(alarm, 1.0, abst, att, cat)
    pc = t["per_category"]
    assert pc["clean"]["correct_rate"] == 1.0 and pc["N"]["correct_rate"] == 1.0 and pc["S"]["correct_rate"] == 1.0
    assert pc["mixture"]["correct_rate"] == 1.0 and pc["unknown"]["correct_rate"] == 0.5
    assert pc["unknown"]["decisions"] == {"quiet": 1, "abstain": 1, "N": 0, "S": 0}


# ----------------------------------------------------------------- I10: null calibration to a common clean scale (ANALYTIC)

def test_null_calibrator_maps_clean_scores_to_standard_normal_scale_and_preserves_order():
    rng = np.random.default_rng(0)
    b = FeatureBatch(); b.add("a", rng.exponential(size=400), "raw_window"); b.add("b", 100 * rng.exponential(size=400), "raw_window"); b.lock()
    cal = NullCalibrator().fit(b, ["a", "b"])
    ca, cb = cal.calibrate("a", b["a"]), cal.calibrate("b", b["b"])
    assert abs(np.median(ca)) < 0.05 and abs(np.median(cb)) < 0.05 and abs(np.std(ca) - 1) < 0.1
    s = np.array([0.0, 1.0, 5.0, 50.0])
    assert np.all(np.diff(cal.calibrate("a", s)) > 0)
    with pytest.raises(ValueError, match="≥ 5"):
        NullCalibrator().fit(FeatureBatch().add("c", np.ones(3), "raw_window").lock(), ["c"])


# ----------------------------------------------------------------- I11: confirmatory dependency gate (STRUCTURAL)

def test_run_dependencies_gate_verifies_commit_environment_data_model_and_destination(tmp_path):
    import json, subprocess
    root = tmp_path
    git = lambda *a: subprocess.run(["git", "-c", "user.email=a@b", "-c", "user.name=t", *a], cwd=root, check=True, capture_output=True)
    (root / "docs").mkdir(); (root / "docs" / "PREREGISTRATION.md").write_text("core")
    (root / "uv.lock").write_text("lock"); (root / "data_manifest.json").write_text("{}")
    git("init", "-q"); git("add", "-A"); git("commit", "-qm", "x")
    thr = {"k": HarmThreshold(0.3, "declared", source="req")}
    deps = RunDependencies(freezes=("core",), data_manifest="data_manifest.json", model_hash="abc")
    dev = require_run_dependencies("dev", deps, repo_root=root, thresholds=thr, out_dir=root / "results" / "dev")
    assert dev["citable"] is False and any("freeze part 'core' is missing" in w for w in dev["warnings"])
    with pytest.raises(ConfirmatoryGateError):
        require_run_dependencies("confirmatory", deps, repo_root=root, thresholds=thr)
    # freeze core, commit the record (the freeze commit is then an ancestor of HEAD), declare every hash
    freeze_part("core", root / "docs" / "PREREGISTRATION.md", root / "protocol" / "frozen")
    git("add", "-A"); git("commit", "-qm", "freeze")
    full = RunDependencies(freezes=("core",), data_manifest="data_manifest.json", model_hash="abc",
                           frozen_environment_sha256=sha256_file(root / "uv.lock"),
                           frozen_data_manifest_sha256=sha256_file(root / "data_manifest.json"), frozen_model_sha256="abc")
    ok = require_run_dependencies("confirmatory", full, repo_root=root, thresholds=thr, out_dir=root / "results" / "confirmatory" / "run1")
    assert ok["citable"] and ok["environment_sha256"] == sha256_file(root / "uv.lock")
    # wrong destination
    with pytest.raises(ConfirmatoryGateError, match="results/confirmatory"):
        require_run_dependencies("confirmatory", full, repo_root=root, thresholds=thr, out_dir=root / "results" / "dev")
    # a dirty tree without a frozen tree hash
    (root / "uv.lock").write_text("lock2")
    with pytest.raises(ConfirmatoryGateError, match="dirty|environment lock does not match"):
        require_run_dependencies("confirmatory", full, repo_root=root, thresholds=thr, out_dir=root / "results" / "confirmatory" / "run1")
    (root / "uv.lock").write_text("lock")
    # a stale environment hash, a wrong model hash, a missing data manifest
    with pytest.raises(ConfirmatoryGateError, match="environment lock does not match"):
        require_run_dependencies("confirmatory", RunDependencies(**{**full.__dict__, "frozen_environment_sha256": "0"}), repo_root=root,
                                 thresholds=thr, out_dir=root / "results" / "confirmatory" / "run1")
    with pytest.raises(ConfirmatoryGateError, match="model hash"):
        require_run_dependencies("confirmatory", RunDependencies(**{**full.__dict__, "frozen_model_sha256": "zzz"}), repo_root=root,
                                 thresholds=thr, out_dir=root / "results" / "confirmatory" / "run1")
    with pytest.raises(ConfirmatoryGateError, match="data manifest"):
        require_run_dependencies("confirmatory", RunDependencies(**{**full.__dict__, "data_manifest": "missing.json"}), repo_root=root,
                                 thresholds=thr, out_dir=root / "results" / "confirmatory" / "run1")
    # a freeze recorded on a commit that is not in this history
    rec = json.loads((root / "protocol" / "frozen" / "core.json").read_text()); rec["commit"] = "0123456"
    (root / "protocol" / "frozen" / "core.json").write_text(json.dumps(rec)); git("commit", "-qam", "tamper")
    with pytest.raises(ConfirmatoryGateError, match="not an ancestor"):
        require_run_dependencies("confirmatory", full, repo_root=root, thresholds=thr, out_dir=root / "results" / "confirmatory" / "run1")
