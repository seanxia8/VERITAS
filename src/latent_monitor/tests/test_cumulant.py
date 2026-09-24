# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""Fisher-cumulant integration (D7, D8, D10; `ARITRA_CUMULANT_INTEGRATION_2026-09-17.md` §6).

Classification per `REVIEW_PROMPTS.md` §B.2: ANALYTIC (closed-form answer),
CONTROL (predicted direction on a constructed case), STRUCTURAL (refusal / shape).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from latent_monitor.cumulant import (fourth_cumulant, tensor_frobenius_deviation, third_central_moment,
                                     truncation_error_ratio)
from latent_monitor.hypergraph_subject import (FrozenPropagators, HypergraphSubject, normalized_laplacian,
                                               normalized_laplacian_order3, order3_pair_weights)
from latent_monitor.protocol.arms import arm_feature_names
from latent_monitor.protocol.availability import ALARM_TIME_SOURCES, AlarmTimeInputs, tier1_manifest
from latent_monitor.protocol.features import AlarmTimeReference, build_alarm_time_features
from latent_monitor.task_metric import TaskMetric

from .conftest import EVAL_IDS, FIT_IDS, NOISE_IDS


@pytest.fixture(scope="module")
def atref(subject, reference, ref_cell):
    X, _ = ref_cell.batch(FIT_IDS[:80])
    return AlarmTimeReference.fit(subject, reference, X, ref_cell.geometry)


# ----------------------------------------------------------------- D7: the estimator (ANALYTIC)

def test_third_cumulant_is_zero_on_gaussian_within_declared_tolerance_and_n():
    """ANALYTIC + CONTROL: the connected third cumulant of iid Gaussian data is zero; the sample
    estimator scatters by sqrt(15/n) per element, so the test declares n and a tolerance."""
    rng = np.random.default_rng(0)
    n, d = 400_000, 3
    T = third_central_moment(rng.normal(size=(n, d)))
    tol = 6.0 * np.sqrt(15.0 / n)                       # 6 sigma of the elementwise sampling error
    assert np.max(np.abs(T)) < tol
    # a skewed sample (unit-variance exponential) has a real third cumulant (2.0) and is not zero
    X = rng.exponential(1.0, size=(n, d)) - 1.0
    T_skew = third_central_moment(X)
    assert np.max(np.abs(T_skew)) > 50 * tol


def test_third_cumulant_is_orthogonal_invariant_and_shear_is_recorded_as_non_invariant():
    """ANALYTIC: the tensor transforms covariantly under an orthogonal chart change and its Frobenius
    norm is invariant; a shear changes the norm and the dependence is recorded, never hidden."""
    rng = np.random.default_rng(1)
    d = 4
    V = rng.normal(size=(20_000, d)) @ np.diag([3.0, 1.0, 0.5, 0.2])
    T = third_central_moment(V)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    Tq = third_central_moment(V @ Q)
    np.testing.assert_allclose(np.einsum("ijk,ia,jb,kc->abc", T, Q, Q, Q), Tq, rtol=1e-6, atol=1e-8)
    assert np.linalg.norm(Tq) == pytest.approx(np.linalg.norm(T), rel=1e-6)
    S = np.eye(d); S[0, 1] = 1.5
    Ts = third_central_moment(V @ S)
    assert not np.isclose(np.linalg.norm(Ts), np.linalg.norm(T), rtol=1e-3), "shear invariance is NOT claimed"


def test_fourth_companion_and_truncation_ratio_are_computed_but_supporting():
    """ANALYTIC: the fourth cumulant of Gaussian data is zero and the truncation ratio is a
    development-only scalar in (0, 1]; a cubic direction lowers it below 1."""
    rng = np.random.default_rng(2)
    n, d = 200_000, 2
    assert np.max(np.abs(fourth_cumulant(rng.normal(size=(n, d))))) < 0.05
    I2 = np.eye(d)
    I3 = np.zeros((d, d, d)); I3[0, 0, 0] = 6.0
    delta = np.array([1.0, 0.0])
    r = truncation_error_ratio(I2, I3, delta)
    assert r < 1.0 and r == pytest.approx(0.5 / (0.5 + 1.0))
    assert truncation_error_ratio(I2, np.zeros((d, d, d)), delta) == pytest.approx(1.0)


# ----------------------------------------------------------------- D7: chart rule and arm symmetry (STRUCTURAL + CONTROL)

def test_chart_rule_is_cumulant_on_the_linear_subject_and_deviation_otherwise(atref, subject, reference, ref_cell):
    assert atref.reading == "cumulant"
    assert atref.to_dict()["reading"] == "cumulant"
    fake = copy.copy(subject)
    object.__setattr__(fake, "is_linear", False)
    X, _ = ref_cell.batch(FIT_IDS[:60])
    at2 = AlarmTimeReference.fit(fake, reference, X, ref_cell.geometry)
    assert at2.reading == "deviation"
    # the whitened residual is a natural chart under the Gaussian reference regardless of subject linearity
    assert at2.to_dict()["resid_cumulant"]["reading"] == "cumulant"
    assert at2.to_dict()["hook_third_nulls"]["channel"]["reading"] == "deviation"
    assert atref.to_dict()["hook_third_nulls"]["token"]["reading"] == "cumulant"


def test_arm_symmetry_and_matched_capacity_after_the_third_moment_transform(atref, subject, ref_cell):
    """STRUCTURAL: a third-moment feature exists on both sides or neither; the capacity control is
    re-truncated so generic_rich_matched again has the feature count of full_intermediate."""
    m = tier1_manifest(latent_dim=subject.latent_dim, n_targets=3, n_pcs=atref.n_pcs)
    X, _ = ref_cell.batch(FIT_IDS[:30])
    rng = np.random.default_rng(0)
    R = rng.normal(size=(30, 2, ref_cell.geometry.n_channels, ref_cell.n_samples))
    b = build_alarm_time_features(subject, atref, AlarmTimeInputs(X=X, geometry=ref_cell.geometry, noise_records=R))
    g = set(arm_feature_names(m, "generic_rich"))
    f = set(arm_feature_names(m, "full_intermediate"))
    assert len(arm_feature_names(m, "generic_rich_matched")) == len(f)
    third_generic = {n for n in g if n.startswith("gr_resid_c3_")}
    third_intermediate = {n for n in f - g if n.endswith("_third_maha")}
    assert len(third_generic) > 0 and len(third_intermediate) == 2, "the transform must be symmetric across the arms"
    # manifest and builder agree on every alarm-time feature
    declared = {n for n, feat in m.features.items() if feat.availability == "alarm_time"}
    assert declared <= set(b.names())
    for n in declared:
        assert b.sources(n) <= ALARM_TIME_SOURCES


def test_new_features_are_tagged_alarm_time_and_never_evaluation_only(atref, subject, ref_cell):
    X, _ = ref_cell.batch(FIT_IDS[:20])
    b = build_alarm_time_features(subject, atref, AlarmTimeInputs(X=X, geometry=ref_cell.geometry))
    new = [n for n in b.names() if n.startswith("gr_resid_c3_") or n.endswith("_third_maha") or n == "raw_task_cubic"]
    assert new
    for n in new:
        assert b.sources(n) == {"raw_window", "reference_fit"}, (n, b.sources(n))
    for n in b.names():
        assert b.sources(n) <= ALARM_TIME_SOURCES


# ----------------------------------------------------------------- D8: the cubic companion (ANALYTIC + CONTROL)

def test_cubic_aligned_contracts_the_whitened_chart_and_is_calibrated_in_the_batch(atref, subject, reference, ref_cell):
    J = np.asarray(subject.jac_output(reference.z_mean, ref_cell.geometry), dtype=float)
    tm = TaskMetric.one_hot(J, 0, subject.target_names)
    A = tm.whitening()
    np.testing.assert_allclose(A @ A, tm.M, atol=1e-8)                  # A = M_task^{1/2}, the whitening chart
    rng = np.random.default_rng(3)
    d = subject.latent_dim
    I3 = third_central_moment(rng.normal(size=(200_000, d)))
    delta = rng.normal(size=(7, d))
    s = tm.cubic_aligned(delta @ A.T, I3)
    assert s.shape == (7,) and np.all(np.isfinite(s))
    np.testing.assert_allclose(tm.cubic_aligned(delta, np.zeros((d, d, d))), np.zeros(7))
    with pytest.raises(ValueError, match="I3 must be"):
        tm.cubic_aligned(delta, np.zeros((d, d)))
    X, _ = ref_cell.batch(EVAL_IDS[:15])
    b = build_alarm_time_features(subject, atref, AlarmTimeInputs(X=X, geometry=ref_cell.geometry))
    assert "raw_task_cubic" in b and np.all(np.isfinite(b["raw_task_cubic"]))


# ----------------------------------------------------------------- D10: frozen-propagator hypergraph subject (ANALYTIC + CONTROL)

def test_hypergraph_laplacians_are_correctly_normalised():
    rng = np.random.default_rng(4)
    C = 6
    A = rng.normal(size=(C, C)); A = A @ A.T
    L = normalized_laplacian(A)
    np.testing.assert_allclose(L, L.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(L), np.ones(C), atol=1e-12)
    assert np.min(np.linalg.eigvalsh(L)) > -1e-10 and np.max(np.linalg.eigvalsh(L)) < 2.0 + 1e-9
    T = rng.normal(size=(C, C, C))
    L3 = normalized_laplacian_order3(T)
    np.testing.assert_allclose(L3, L3.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(L3), np.ones(C), atol=1e-12)
    assert np.min(np.linalg.eigvalsh(L3)) > -1e-10
    np.testing.assert_array_equal(order3_pair_weights(np.zeros((C, C, C))), np.zeros((C, C)))


def test_p3_vanishes_on_gaussian_reference_and_is_nonzero_on_heavy_tails():
    """CONTROL: the order-3 propagator is linear in the third cumulant, so it is identically zero
    for a zero cumulant and near zero on Gaussian noise (declared tolerance and n); a unit-variance
    skewed sample (exponential, third cumulant 2) gives a clearly larger Laplacian."""
    rng = np.random.default_rng(5)
    C, N, m = 5, 32, 20_000
    gaussian = rng.normal(size=(m, C, N))
    fp = FrozenPropagators.fit(gaussian)
    tol = 6.0 * C * np.sqrt(15.0 / m)
    assert np.max(np.abs(fp.p3())) < tol
    fp0 = FrozenPropagators(channel_cov=fp.channel_cov, third_cumulant=np.zeros((C, C, C)))
    np.testing.assert_array_equal(fp0.p3(), np.zeros((C, C)))
    # a shared skewed component makes the cross-channel third cumulants non-zero: P3 is non-trivial
    common = rng.exponential(1.0, size=(m, N)) - 1.0
    loadings = rng.normal(size=C)
    heavy = loadings[None, :, None] * common[:, None, :] + 0.1 * rng.normal(size=(m, C, N))
    fp_h = FrozenPropagators.fit(heavy)
    assert np.max(np.abs(fp_h.p3())) > 10.0 * np.max(np.abs(fp.p3()))


def test_hypergraph_subject_conforms_to_the_subject_interface_and_is_untrained(subject, reference, ref_cell):
    fp = FrozenPropagators.fit(ref_cell.noise_batch(NOISE_IDS))
    hs = HypergraphSubject(fp, ref_cell.implied_whitener(), subject.A, subject.D, subject.O, subject.o0, subject.E,
                           target_names=subject.target_names)
    assert hs.is_linear is False, "the candidate nonlinear subject must not declare the tied-linear construction"
    X, T = ref_cell.batch(EVAL_IDS[:12])
    rep = hs.represent(X, ref_cell.geometry)
    assert rep["z"].shape == (12, subject.latent_dim)
    assert hs.outputs(X, ref_cell.geometry).shape == T.shape
    assert hs.decode(rep["z"], ref_cell.geometry).shape == X.shape
    assert hs.jac_recon(reference.z_mean, ref_cell.geometry).shape == (X.shape[1] * X.shape[2], subject.latent_dim)
    assert hs.jac_output(reference.z_mean, ref_cell.geometry).shape == (T.shape[1], subject.latent_dim)
    # gate alpha = 0 at init: the subject is exactly the tied linear AE
    np.testing.assert_allclose(rep["z"], subject.represent(X, ref_cell.geometry)["z"], atol=1e-12)
    assert hs.to_dict()["gaussian_note"].startswith("P3 vanishes")
