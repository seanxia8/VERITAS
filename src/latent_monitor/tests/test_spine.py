# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""The mathematical spine (TWO_CLAIM_REVISION_PLAN §3 M1–M4): dimensions, resolvability vs support, task-specific
controls, and the invariance group of every statistic the paper reports. ANALYTIC unless marked otherwise."""

from __future__ import annotations

import numpy as np
import pytest

from latent_monitor.designed import DesignedFamily, linearization_check
from latent_monitor.protocol.features import AlarmTimeReference, HookNull
from latent_monitor.support import SupportEstimator, displaced_controls, ledoit_wolf_shrinkage, validate_support_estimator
from latent_monitor.task_metric import TaskMetric, measurement_metric

from .conftest import EVAL_IDS, FIT_IDS


# ----------------------------------------------------------------- M1: two metrics, two dimensions

def test_measurement_and_task_metrics_have_the_right_shapes_units_and_never_mix(subject, reference, ref_cell):
    d = subject.latent_dim
    J_rec = subject.jac_recon(reference.z_mean, ref_cell.geometry)          # (C·N, d), already Σ̂^{-1/2}-whitened
    M_recon = measurement_metric(J_rec)
    assert M_recon.shape == (d, d)
    np.testing.assert_allclose(M_recon, reference.M_recon, atol=1e-9)
    # for the tied linear subject J̃_g = tile(D) and D = Aᵀ with orthonormal rows ⇒ M_recon = C · I (unit-free)
    C = ref_cell.geometry.n_channels
    np.testing.assert_allclose(M_recon, C * np.eye(d), atol=1e-9)
    J_y = subject.jac_output(reference.z_mean, ref_cell.geometry)           # (t, d) in physics-output units
    t = J_y.shape[0]
    tm = TaskMetric.one_hot(J_y, 0, subject.target_names)
    assert tm.M.shape == (d, d) and tm.W_y.shape == (t, t)
    # M_task with a one-hot weight is the outer product of one Jacobian row — nothing about Σ enters
    np.testing.assert_allclose(tm.M, np.outer(J_y[0], J_y[0]), atol=1e-12)
    # a weight of the wrong dimension (a measurement-space Σ⁻¹ of size C·N) is refused: dimensionally invalid
    with pytest.raises(ValueError, match="W_y must be"):
        TaskMetric(J_y, np.eye(C * ref_cell.n_samples))
    with pytest.raises(ValueError, match="positive semi-definite"):
        TaskMetric(J_y, -np.eye(t))
    # scaling an output's units by 10 scales its one-hot task length by 10 and leaves M_recon untouched
    tm10 = TaskMetric.one_hot(10.0 * J_y, 0)
    delta = np.ones(d)
    assert tm10.length(delta)[0] == pytest.approx(10.0 * tm.length(delta)[0])
    np.testing.assert_allclose(measurement_metric(J_rec), M_recon)


def test_resolution_weighted_task_metric_matches_the_declared_units():
    J = np.array([[1.0, 0.0], [0.0, 2.0]])
    tm = TaskMetric.from_resolutions(J, sigma=np.array([0.5, 4.0]))
    np.testing.assert_allclose(tm.M, J.T @ np.diag([4.0, 1 / 16]) @ J)
    with pytest.raises(ValueError):
        TaskMetric.from_resolutions(J, sigma=np.array([0.5, 0.0]))


# ----------------------------------------------------------------- M2: resolvability is not support

def test_resolved_projectors_are_the_legacy_names_and_support_is_a_separate_validated_estimator(subject, reference, ref_cell):
    np.testing.assert_array_equal(reference.P_resolved, reference.P_exc)
    np.testing.assert_array_equal(reference.P_weak, reference.P_unexc)
    assert reference.resolved_rank == reference.fisher_rank == subject.latent_dim
    # the tied subject resolves every direction; a support estimator can still flag windows as far from the reference
    X_fit, _ = ref_cell.batch(FIT_IDS[:80]); X_in, _ = ref_cell.batch(EVAL_IDS[:30])
    z_fit = subject.represent(X_fit, ref_cell.geometry)["z"]; z_in = subject.represent(X_in, ref_cell.geometry)["z"]
    est = SupportEstimator.fit(z_fit, k=5)
    val = validate_support_estimator(est, z_in, displaced_controls(z_in, est, sigmas=6.0))
    assert val["auroc_in_vs_out"] > 0.95 and val["usable_for_abstention"]
    # a weak displacement is not flagged as out of support — the estimator has a floor and says so
    val2 = validate_support_estimator(est, z_in, displaced_controls(z_in, est, sigmas=0.3), floor=0.9)
    assert not val2["usable_for_abstention"]
    # resolved rank says nothing about support: every window here is fully resolved (rank = latent_dim) yet the
    # displaced ones score as novel
    assert est.score(displaced_controls(z_in, est, sigmas=6.0)).min() > est.score(z_in).max()


def test_ledoit_wolf_shrinkage_is_invertible_when_d_exceeds_n_and_recovers_the_sample_covariance_when_n_is_large():
    rng = np.random.default_rng(0)
    X_small = rng.normal(size=(6, 20))
    cov, alpha = ledoit_wolf_shrinkage(X_small)
    assert 0.0 < alpha <= 1.0 and np.linalg.cond(cov) < 1e6
    X_big = rng.normal(size=(20000, 3)) @ np.array([[2.0, 0.5, 0.0], [0.0, 1.0, 0.3], [0.0, 0.0, 0.5]])
    cov_big, alpha_big = ledoit_wolf_shrinkage(X_big)
    assert alpha_big < 0.02
    np.testing.assert_allclose(cov_big, np.cov(X_big.T, bias=True), rtol=0.05, atol=0.02)


# ----------------------------------------------------------------- M3 / I8: task-specific controls target the declared K

def test_task_aligned_and_task_null_families_move_or_spare_the_declared_output(subject, reference, ref_cell):
    X, _ = ref_cell.batch(EVAL_IDS[:20])
    for metric in ("euclidean", "null_mahalanobis"):
        al = linearization_check(subject, reference, DesignedFamily("task_aligned", 2.0, match_metric=metric, task_target=0), X, ref_cell.geometry)
        nu = linearization_check(subject, reference, DesignedFamily("task_null", 2.0, match_metric=metric, task_target=0), X, ref_cell.geometry)
        assert al["rel_err_dy"] < 1e-10 and nu["rel_err_dy"] < 1e-10
        assert al["realised_dy_per_output"][0] > 1e-3, "task_aligned must move the declared output"
        assert nu["realised_dy_per_output"][0] < 1e-10, "task_null must leave the declared output unchanged to first order"
        assert nu["subspace_rank"] == subject.latent_dim - 1
    with pytest.raises(ValueError, match="task_target"):
        DesignedFamily("task_aligned", 1.0)


# ----------------------------------------------------------------- M4: invariance group of the reported statistics

def _random_orthogonal(d, rng):
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q


def test_invariance_group_of_the_reference_distances_is_orthogonal_and_isotropic_scale_not_shear():
    """Per-hook Mahalanobis with Ledoit–Wolf shrinkage: invariant to orthogonal maps and global scale; shear changes it.
    (Without shrinkage Mahalanobis is fully affine invariant; shrinkage towards the identity breaks shear invariance —
    reported as a limitation, PREREGISTRATION §2.)"""
    rng = np.random.default_rng(1)
    d, n = 4, 60
    R = rng.normal(size=(n, d)) @ np.diag([3.0, 1.0, 0.5, 0.2])
    V = rng.normal(size=(15, d)) @ np.diag([3.0, 1.0, 0.5, 0.2]) + 0.5
    base = HookNull.fit(R).mahalanobis(V)
    Q = _random_orthogonal(d, rng)
    np.testing.assert_allclose(HookNull.fit(R @ Q).mahalanobis(V @ Q), base, rtol=1e-6)
    np.testing.assert_allclose(HookNull.fit(3.0 * R).mahalanobis(3.0 * V), base, rtol=1e-6)
    S = np.eye(d); S[0, 1] = 2.0                          # shear
    sheared = HookNull.fit(R @ S).mahalanobis(V @ S)
    assert not np.allclose(sheared, base, rtol=1e-3), "shear invariance is NOT claimed; the dependence is real and reported"


def test_invariance_group_of_energy_splits_and_task_length():
    """Energy splits in the reference projectors are invariant to orthogonal maps of z (projectors rotate with it) and to
    isotropic scale; task length ‖δ‖_{M_task} is invariant to any invertible reparameterisation r ↦ A r when J_y transforms
    covariantly (J_y A⁻¹), because it is a pullback."""
    rng = np.random.default_rng(2)
    d = 5
    J = rng.normal(size=(3, d)); delta = rng.normal(size=(7, d))
    tm = TaskMetric.one_hot(J, 0)
    A = rng.normal(size=(d, d)) + 3 * np.eye(d)            # a general invertible reparameterisation (includes shear)
    tmA = TaskMetric.one_hot(J @ np.linalg.inv(A), 0)
    np.testing.assert_allclose(tmA.length(delta @ A.T), tm.length(delta), rtol=1e-9)
    # projector energy split: orthogonal-invariant, scale-invariant, not shear-invariant
    P = np.outer(J[0], J[0]) / (J[0] @ J[0])
    e = np.einsum("bi,ij,bj->b", delta, P, delta) / np.sum(delta**2, axis=1)
    Q = _random_orthogonal(d, rng)
    PQ = Q.T @ P @ Q
    np.testing.assert_allclose(np.einsum("bi,ij,bj->b", delta @ Q, PQ, delta @ Q) / np.sum((delta @ Q) ** 2, axis=1), e, rtol=1e-9)
    np.testing.assert_allclose(np.einsum("bi,ij,bj->b", 2 * delta, P, 2 * delta) / np.sum((2 * delta) ** 2, axis=1), e, rtol=1e-9)
    Sh = np.eye(d); Sh[0, 1] = 1.5
    e_sh = np.einsum("bi,ij,bj->b", delta @ Sh, P, delta @ Sh) / np.sum((delta @ Sh) ** 2, axis=1)
    assert not np.allclose(e_sh, e, rtol=1e-3)


# ----------------------------------------------------------------- I10: covariance stability

def test_hook_null_scores_are_stable_across_reference_sample_size_and_representation_rescaling():
    """Measured (d = 6): the correlation of window orderings against a 4000-window reference is 0.65–0.93 at n_ref = 60 and
    0.90–0.99 at n_ref = 300. Reference distances at the smoke's n_ref (~36) are therefore *unstable*; the report says so and the
    trained-subject run must use a reference-fit partition sized by the precision study (PREREGISTRATION §6)."""
    d = 6
    worst_60, worst_300 = 1.0, 1.0
    for seed in range(3):
        rng = np.random.default_rng(seed)
        L = np.tril(rng.normal(size=(d, d))) + 2 * np.eye(d)
        draw = lambda n: rng.normal(size=(n, d)) @ L.T
        V = draw(40) + 1.0
        big = HookNull.fit(draw(4000)).mahalanobis(V)
        c60 = np.corrcoef(HookNull.fit(draw(60)).mahalanobis(V), big)[0, 1]
        c300 = np.corrcoef(HookNull.fit(draw(300)).mahalanobis(V), big)[0, 1]
        worst_60, worst_300 = min(worst_60, c60), min(worst_300, c300)
    assert worst_300 > 0.85 and worst_300 > worst_60, "ordering stabilises with the reference size; below ~300 it is not stable"
    rng = np.random.default_rng(3)
    L = np.tril(rng.normal(size=(d, d))) + 2 * np.eye(d)
    draw = lambda n: rng.normal(size=(n, d)) @ L.T
    V = draw(40) + 1.0
    # a hook whose dimension exceeds n_ref/5 is PCA-reduced by the declared rule and still invertible
    hn = HookNull.fit(rng.normal(size=(30, 25)))
    assert hn.reduced_dim == 6 and hn.pcs is not None and np.all(np.isfinite(hn.mahalanobis(rng.normal(size=(5, 25)))))
    # rescaling a representation rescales nothing in the score
    R = draw(200)
    np.testing.assert_allclose(HookNull.fit(R * 7.0).mahalanobis(V * 7.0), HookNull.fit(R).mahalanobis(V), rtol=1e-6)
