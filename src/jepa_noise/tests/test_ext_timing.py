"""Acceptance tests for the timing extension (runbook 2026-10-30, §3.4)."""
from pathlib import Path

import numpy as np
import pytest

from jepa_noise import ext_timing as et

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"


def _sample_sets(n=500, groups_c=8, groups_t=7, seed=0):
    """Synthetic pooled-sample sets and their exact covariance blocks."""
    rng = np.random.default_rng(seed)
    dc, dt = 3 * groups_c, 3 * groups_t
    base = rng.normal(size=(n, dc + dt))
    # Give the time coordinates a common shift so time_hit_removed has content.
    base[:, 1:dc:3] += rng.normal(size=(n, groups_c)) * 0.3
    base[:, dc + 1::3] += rng.normal(size=(n, groups_t)) * 0.3
    z = base
    z_a = base + 0.2 * rng.normal(size=base.shape)
    z_b = base + 0.3 * rng.normal(size=base.shape)
    cov = lambda x: np.cov(x, rowvar=False, ddof=1)
    S, A, B = cov(z), cov(z_a), cov(z_b)
    return z, z_a, z_b, et.blocks_from_joint(S, A, B, dc, dt), S, dc, dt


def test_variant_transform_matches_transformed_samples():
    z, z_a, z_b, blocks, S, dc, dt = _sample_sets()
    for variant in ("time_only", "charge_only", "hit_only", "no_time",
                    "time_common_removed", "time_hit_removed"):
        transformed, Tc, Tt = et.variant_blocks(
            blocks, variant, S=S, means=z, context_dim=dc, target_dim=dt)
        dc2, dt2 = Tc.shape[0], Tt.shape[0]
        for samples, prefix in ((z, "s"), (z_a, "a"), (z_b, "b")):
            z2 = np.column_stack((samples[:, :dc] @ Tc.T,
                                  samples[:, dc:] @ Tt.T))
            expected = et.blocks_from_joint(np.cov(z2, rowvar=False, ddof=1),
                                            np.cov(z2, rowvar=False, ddof=1),
                                            np.cov(z2, rowvar=False, ddof=1),
                                            dc2, dt2)
            for suffix in ("cc", "tt", "ct"):
                np.testing.assert_allclose(
                    transformed[f"{prefix}{suffix}"],
                    expected[f"s{suffix}"], atol=1e-10, rtol=0,
                    err_msg=f"{variant}/{prefix}{suffix}")


def test_time_common_removed_kills_a_constant_time_shift():
    groups = 6
    rng = np.random.default_rng(1)
    x = rng.normal(size=(40, 3 * groups))
    x[:, 1::3] = 2.5
    matrix, _ = et.common_time_map(groups)
    transformed = x @ matrix.T
    np.testing.assert_allclose(transformed[:, 1::3], 0.0, atol=1e-12, rtol=0)
    np.testing.assert_allclose(transformed[:, 0::3], x[:, 0::3], atol=0, rtol=0)


def test_trigger_deltas_sum_to_zero():
    rng = np.random.default_rng(2)
    triggers = rng.normal(size=(64, 16)) + 1000.0
    deltas = np.stack([et.trigger_deltas(row) for row in triggers])
    np.testing.assert_allclose(deltas.sum(axis=1), 0.0, atol=1e-9, rtol=0)


def _fixture():
    from jepa_noise import lucid_bombs as lb

    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(str(CONFIG))
    run_id = "test/ext_timing/0000"
    state = lb.state_for_run("F1", 0.05, run_id, geometry, seed=13)
    draws = lb.draw_event(geometry, cfg, run_id, 0, seed=13, domain="photon")
    return lb, geometry, cfg, state, draws


def test_jitter_to_token_path():
    """A +c ns trigger shift moves every shared token time by exactly -c/100."""
    pytest.importorskip("torch")
    lb, geometry, cfg, state, draws = _fixture()
    try:
        real_rule = lb.pmt_time_rule()
    except RuntimeError:
        pytest.skip("PMT NHits time rule is not merged")
    readout = lb.make_readout(geometry, cfg)
    original = lb._PMT_TIME_RULE
    try:
        reference = readout(state, draws)
        lb._PMT_TIME_RULE = et.shifted_time_rule(real_rule, 5.0)
        shifted = readout(state, draws)
    finally:
        lb._PMT_TIME_RULE = original
    assert lb._PMT_TIME_RULE is original
    common = np.intersect1d(reference.pmt_id, shifted.pmt_id)
    index_reference = {int(pmt): index for index, pmt in enumerate(reference.pmt_id)}
    index_shifted = {int(pmt): index for index, pmt in enumerate(shifted.pmt_id)}
    before = np.array([reference.time_ns[index_reference[int(pmt)]] for pmt in common])
    after = np.array([shifted.time_ns[index_shifted[int(pmt)]] for pmt in common])
    np.testing.assert_allclose(after - before, -5.0, atol=1e-9, rtol=0)
    np.testing.assert_allclose((after - before) / 100.0, -0.05, atol=1e-11, rtol=0)


def test_dejitter_removes_the_trigger_dependence():
    """§2.4 de-jitter removes the injected trigger shift from the pooled residual."""
    pytest.importorskip("torch")
    from jepa_noise.ext_predictability import resample_noise
    from jepa_noise import point_encoder as pe
    from jepa_noise.lucid_views import ViewMasker
    from scripts.jepa_ext_m_dose import pooled, tokenise_hits, view_groups

    lb, geometry, cfg, state, draws = _fixture()
    try:
        real_rule = lb.pmt_time_rule()
    except RuntimeError:
        pytest.skip("PMT NHits time rule is not merged")
    readout = lb.make_readout(geometry, cfg)
    scaling = pe.FeatureScaling.from_geometry(geometry)
    masker = ViewMasker(geometry, "within", n_min=0)
    shifts = np.array([0.0, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 0.0])
    hits_reference, hits_shifted = [], []
    original = lb._PMT_TIME_RULE
    try:
        for replicate, shift in enumerate(shifts):
            sample = resample_noise(draws, geometry, cfg, replicate, seed=13)
            lb._PMT_TIME_RULE = real_rule          # unshifted reference
            hits_reference.append(readout(state, sample))
            lb._PMT_TIME_RULE = et.shifted_time_rule(real_rule, shift)
            hits_shifted.append(readout(state, sample))
    finally:
        lb._PMT_TIME_RULE = original
    assert lb._PMT_TIME_RULE is original
    triggers_reference = np.array([hit.trigger_ns for hit in hits_reference])
    triggers_shifted = np.array([hit.trigger_ns for hit in hits_shifted])
    np.testing.assert_allclose(triggers_shifted - triggers_reference, shifts,
                               atol=1e-12, rtol=0)
    features_reference, _ = tokenise_hits(geometry, hits_reference, scaling, masker)
    features_shifted, _ = tokenise_hits(geometry, hits_shifted, scaling, masker)
    deltas_reference = et.trigger_deltas(triggers_reference)
    deltas_shifted = et.trigger_deltas(triggers_shifted)
    groups = view_groups(geometry)
    for split in ("within", "across_fixed"):
        reference = pooled(features_reference, split, groups)
        shifted = pooled(features_shifted, split, groups)
        corrected_reference = et.dejitter_pooled(reference, deltas_reference)
        corrected_shifted = et.dejitter_pooled(shifted, deltas_shifted)
        # float32 token-time quantization; tolerance recorded in DECISIONS.
        np.testing.assert_allclose(corrected_shifted[:, 1::3],
                                   corrected_reference[:, 1::3],
                                   atol=1e-6, rtol=0,
                                   err_msg=f"{split}: de-jitter not shift-independent")
        # Sanity: without the correction the same residual does carry the shift.
        raw_shifted = ((shifted - shifted.mean(axis=0)) * np.sqrt(8 / 7))
        raw_reference = ((reference - reference.mean(axis=0)) * np.sqrt(8 / 7))
        assert np.max(np.abs(raw_shifted[:, 1::3]
                             - raw_reference[:, 1::3])) > 1e-3

def test_pair_metrics_handles_missing_extra_cross_blocks():
    """The Criterion-M wrapper must accept zero, one or two extra cross blocks."""
    z, z_a, z_b, blocks, S, dc, dt = _sample_sets()
    primary = et.pair_metrics(blocks, 3)
    assert primary["nu_A"] is None and primary["nu_gain"] is None
    assert primary["nu_star_A"] is None and primary["argmax_A"] is None
    full = et.pair_metrics(blocks, 3, (blocks["act"],
                                      blocks["bct"] - blocks["act"]))
    assert len(full["nu_A"]) == 3 and len(full["nu_gain"]) == 3
    assert full["nu_star_A"] is not None and full["argmax_gain"] is not None
