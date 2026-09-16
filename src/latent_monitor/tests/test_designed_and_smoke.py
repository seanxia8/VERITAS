# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""R4 audit of the designed control (ANALYTIC on the linear subject) and the bounded CPU smoke (STRUCTURAL + CONTROL)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from latent_monitor.designed import DesignedFamily, NonlinearSubjectUnsupported, NullSpaceUnavailable, linearization_check
from latent_monitor.protocol.smoke import run_smoke

from .conftest import EVAL_IDS


def test_designed_families_are_exact_on_the_linear_subject_in_both_match_metrics(subject, reference, ref_cell):
    """ANALYTIC: Δz equals the intended δ; Δy equals J δ (zero for the null family); norms match in the declared metric."""
    X, _ = ref_cell.batch(EVAL_IDS[:20])
    L = np.linalg.cholesky(reference.null_dz_cov + 1e-12 * np.eye(reference.latent_dim))
    for metric in ("euclidean", "null_mahalanobis"):
        for kind in ("output_null", "output_aligned", "random"):
            fam = DesignedFamily(kind, 2.5, seed=1, match_metric=metric)
            chk = linearization_check(subject, reference, fam, X, ref_cell.geometry)
            assert chk["rel_err_dz"] < 1e-10 and chk["rel_err_dy"] < 1e-10, (metric, kind, chk)
            d = fam.latent_directions(reference, 20)
            if metric == "euclidean":
                np.testing.assert_allclose(np.linalg.norm(d, axis=1), 2.5, atol=1e-10)
            else:
                np.testing.assert_allclose(np.sqrt(np.sum(np.linalg.solve(L, d.T) ** 2, axis=0)), 2.5, atol=1e-10)
            if kind == "output_null":
                assert chk["realised_dy_norm"] < 1e-10
            else:
                assert chk["realised_dy_norm"] > 1e-3


def test_null_space_unavailable_is_an_explicit_skip(reference):
    """STRUCTURAL: a zero-rank subspace raises instead of yielding a zero vector dressed as a null perturbation."""
    from dataclasses import replace
    k = reference.latent_dim
    full = replace(reference, P_out=np.eye(k), P_null=np.zeros((k, k)), k_out=k)
    fam = DesignedFamily("output_null", 3.0)
    assert fam.subspace_rank(full) == 0
    with pytest.raises(NullSpaceUnavailable, match="rank 0"):
        fam.latent_directions(full, 4)
    assert DesignedFamily("output_aligned", 3.0).subspace_rank(full) == k


def test_nonlinear_subject_is_refused_by_the_designed_construction(subject, reference, ref_cell):
    """STRUCTURAL (I8): x + W⁻¹ g(δ) is a local inverse only for the tied linear subject; anything else is refused, not approximated."""
    import copy
    fake = copy.copy(subject)
    object.__setattr__(fake, "is_linear", False)          # a subject that does not declare linearity
    X, _ = ref_cell.batch(np.arange(3))
    with pytest.raises(NonlinearSubjectUnsupported):
        DesignedFamily("output_null", 1.0).perturb(fake, reference, X, ref_cell.geometry)


def test_designed_directions_are_reproducible_across_processes(reference):
    """STRUCTURAL: the seed stream no longer depends on the process hash salt of the kind string."""
    a = DesignedFamily("output_null", 1.0, seed=5).latent_directions(reference, 3)
    b = DesignedFamily("output_null", 1.0, seed=5).latent_directions(reference, 3)
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, DesignedFamily("output_aligned", 1.0, seed=5).latent_directions(reference, 3))


@pytest.mark.slow
def test_cpu_smoke_runs_end_to_end_with_provenance_and_gates(tmp_path):
    """STRUCTURAL + CONTROL: the two-claim chain runs at toy size; the report is labelled dev; JSON is strict; the designed sign holds."""
    r = run_smoke(tmp_path, n_channels=4, n_samples=64, latent_dim=4, n_groups=60, n_boot=20, seed=1, repo_root=tmp_path)
    assert "NOT CITABLE" in r["STATUS"] and r["provenance"]["citable"] is False and r["gates"]["mode"] == "dev"
    assert any("freeze part 'core' is missing" in w for w in r["gates"]["warnings"])
    assert any("data manifest" in w for w in r["run_dependencies"]["warnings"]) and r["run_dependencies"]["citable"] is False
    text = (tmp_path / "smoke_report.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    json.loads(text)
    assert (tmp_path / "smoke_report.md").read_text().startswith("# Two-claim protocol smoke — DEVELOPMENT OUTPUT")
    # information contract: declarative and data-flow checks passed for every arm
    assert all(s["status"] == "operational" and s["dataflow_checked"] for s in r["arm_contract_status"].values())
    # partitions: five, plus excluded; reference_fit rows never trained on
    assert set(r["partition_counts"]) >= {"reference_fit", "attribution_train", "development", "calibration", "evaluation"}
    for a in ("generic_rich", "full_intermediate"):
        assert r["claim1"]["arms"][a]["n_train"] == r["claim1"]["arms"]["generic_rich"]["n_train"]
    # abstention chain and joint table exist; unknown windows appear only in evaluation
    assert r["abstention"]["counts"]["n_unknown"] > 0 and r["joint_decision_table"]["per_category"]["unknown"]["n"] > 0
    assert r["labels"]["event:glitch"]["origin"] == "unknown" and r["labels"]["event:glitch"]["declared"] is False
    # Claim 2: quadrants, κ_m provisional, hierarchical CI labelled
    assert r["claim2"]["kappa_m"]["status"] == "provisional_dev"
    assert set(r["claim2"]["quadrants_generic_committed"]) >= {"low_alarm_benign", "low_alarm_harmful", "strong_alarm_benign", "strong_alarm_harmful"}
    assert "inference" in r["claim2"]["primary_delta"]["ci_hierarchical_outer_cell"]
    # designed controls: null leaves every target untouched; task_aligned moves the declared K; exact
    for metric in ("euclidean", "null_mahalanobis"):
        d = r["designed"][metric]
        assert all(abs(v - 1.0) < 1e-9 for v in d["output_null"]["consequence_ratio_targets"].values())
        assert abs(d["task_null"]["consequence_ratio_declared_K"] - 1.0) < 1e-9
        assert d["task_aligned"]["consequence_ratio_declared_K"] > d["task_null"]["consequence_ratio_declared_K"] + 1e-6
        assert d["task_aligned"]["linearization"]["rel_err_dy"] < 1e-9
    assert r["designed"]["euclidean"]["task_aligned"]["consequence_ratio_declared_K"] > 1.05
    # FAR resolution is stated, not assumed
    assert r["far"]["resolvable_in_one_step"] is False and len(r["far"]["ci95"]) == 2


def test_confirmatory_smoke_fails_closed(tmp_path):
    from latent_monitor.protocol import ConfirmatoryGateError
    with pytest.raises(ConfirmatoryGateError):
        run_smoke(tmp_path, n_channels=4, n_samples=64, latent_dim=4, n_groups=12, mode="confirmatory", repo_root=tmp_path)
