# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
#
# Part of the ORACLE study. If you use this package in published work,
# please cite it: see CITATION.cff at the repository root.
"""The amended protocol layer (docs/IMPLEMENTATION_PLAN.md §8, R2–R5).

Everything here enforces a rule a reviewer can check rather than a rule an
author intends to follow:

* ``labels``        origin (N/S/G/mixture/unknown/constructed) separated from harm
                    (benign/harmful), the EF/EV namespace and the legacy-label map.
* ``availability``  the feature-availability manifest and the alarm-time
                    information contract; leakage of evaluation-only or
                    delayed-label features into an alarm-time arm raises.
* ``splits``        event-group splits: every geometry, replay and corruption
                    variant of an event stays in one partition; declared
                    held-out families / severities / seeds; unknown families
                    never enter fitting, tuning or calibration.
* ``arms``          the five comparison arms plus the noise-only ablation, one
                    classifier, identical splits and tuning budget.
* ``consequence``   Claim 2: paired ΔAUROC(K ≥ κ_m) of the task-sensitive score
                    against the committed generic score over held-out cells,
                    the four-quadrant alarm–harm matrix, missed harm at the
                    alert budget, valid-rare-cell rejection, breakdowns,
                    cell-level thresholds, FAR precision, and hierarchical
                    resampling (outer unit = cell) that refuses inference when
                    the outer units are too few.
* ``abstention``    conformal novelty on clean calibration windows, unknown
                    AUROC, risk–coverage curve/AUC, retained-known F1.
* ``matching``      hard matched N/S contrasts with retention reporting, and
                    the joint detect → abstain → attribute decision table.
* ``mode``          dev / confirmatory; confirmatory fails closed on a missing
                    freeze, a pending threshold, a dirty or mismatched tree,
                    an unfrozen environment / data / model hash, or a result
                    path outside ``results/confirmatory``.
* ``smoke``         one bounded CPU development smoke that exercises all of it.
"""

from .availability import (
    ALARM_TIME_SOURCES,
    AlarmTimeContract,
    AlarmTimeInputs,
    Feature,
    FeatureBatch,
    FeatureManifest,
    LeakageError,
    PHASES,
    tier1_manifest,
)
from .consequence import (
    MIN_OUTER_UNITS_FOR_INFERENCE,
    HarmThreshold,
    InvalidWeightsError,
    PendingThresholdError,
    alarm_harm_matrix,
    all_cell_ranking,
    bootstrap_groups,
    bootstrap_hierarchical,
    breakdown,
    cell_aggregate,
    cell_alarm_threshold,
    cell_weights,
    conditional_triage,
    far_precision,
    harm_labels,
    missed_harm_rate_at_budget,
    paired_delta_auroc,
    supporting_intermediate_cubic,
    valid_rare_cell_rejection,
    valid_rare_event_rejection,
    valid_rare_window_rejection,
    weighted_auroc,
)
from .abstention import ConformalNovelty, evaluate_abstention, risk_coverage, unknown_auroc
from .matching import hard_match, joint_decision_table
from .labels import (
    CONTRACT_OF_ORIGIN,
    HARM,
    LEGACY_MOVED_TO_ORIGIN,
    NAMESPACE,
    ORIGIN,
    CellLabel,
    label_from_legacy,
)
from .mode import (ConfirmatoryGateError, FreezeRecord, ProtocolMode, RunDependencies, provenance, require_gates,
                   require_run_dependencies)
from .splits import GroupSplit, OverlapError, assert_disjoint, assign_partitions, split_event_groups

__all__ = [
    "ALARM_TIME_SOURCES", "AlarmTimeInputs", "ConformalNovelty", "FeatureBatch", "InvalidWeightsError",
    "MIN_OUTER_UNITS_FOR_INFERENCE", "RunDependencies", "bootstrap_hierarchical", "cell_aggregate",
    "cell_alarm_threshold", "evaluate_abstention", "far_precision", "hard_match", "joint_decision_table",
    "paired_delta_auroc", "require_run_dependencies", "risk_coverage", "unknown_auroc",
    "valid_rare_cell_rejection", "valid_rare_window_rejection",
    "AlarmTimeContract", "CONTRACT_OF_ORIGIN", "CellLabel", "ConfirmatoryGateError", "Feature",
    "FeatureManifest", "FreezeRecord", "GroupSplit", "HARM", "HarmThreshold", "LEAKAGE",
    "LEGACY_MOVED_TO_ORIGIN", "LeakageError", "NAMESPACE", "ORIGIN", "OverlapError", "PHASES",
    "PendingThresholdError", "ProtocolMode", "alarm_harm_matrix", "all_cell_ranking",
    "assert_disjoint", "assign_partitions", "bootstrap_groups", "breakdown", "cell_weights",
    "conditional_triage", "harm_labels", "label_from_legacy", "missed_harm_rate_at_budget",
    "provenance", "require_gates", "split_event_groups", "tier1_manifest",
    "supporting_intermediate_cubic",
    "valid_rare_event_rejection", "weighted_auroc",
]
LEAKAGE = LeakageError
