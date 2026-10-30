"""Acceptance tests for the de-jitter extension (runbook 2026-10-30, §2.4)."""
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from jepa_noise import ext_dejitter as ed

REPO = Path(__file__).resolve().parents[3]
CONFIG = REPO / "src/jepa_noise/configs/lucid_bombs_v2_calibrated.json"


def _fixture():
    from jepa_noise import lucid_bombs as lb
    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(str(CONFIG))
    run_id = "test/ext_dejitter/0000"
    state = lb.state_for_run("F1", 0.05, run_id, geometry, seed=13)
    draws = lb.draw_event(geometry, cfg, run_id, 0, seed=13, domain="photon")
    return lb, geometry, cfg, state, draws


def test_shifted_trigger_moves_token_time_by_minus_c_over_100():
    pytest.importorskip("torch")
    from jepa_noise import ext_timing as et
    lb, geometry, cfg, state, draws = _fixture()
    try:
        real_rule = lb.pmt_time_rule()
    except RuntimeError:
        pytest.skip("PMT NHits time rule is not merged")
    readout = lb.make_readout(geometry, cfg)
    original = lb._PMT_TIME_RULE
    try:
        reference = readout(state, draws)
        lb._PMT_TIME_RULE = et.shifted_time_rule(real_rule, 4.0)
        shifted = readout(state, draws)
    finally:
        lb._PMT_TIME_RULE = original
    common = np.intersect1d(reference.pmt_id, shifted.pmt_id)
    index_reference = {int(pmt): index for index, pmt in enumerate(reference.pmt_id)}
    index_shifted = {int(pmt): index for index, pmt in enumerate(shifted.pmt_id)}
    before = np.array([reference.time_ns[index_reference[int(pmt)]] for pmt in common])
    after = np.array([shifted.time_ns[index_shifted[int(pmt)]] for pmt in common])
    # time_100ns units: a +c ns trigger shift moves the token time by -c/100.
    np.testing.assert_allclose((after - before) / ed.TIME_SCALE_NS, -0.04,
                               atol=1e-11, rtol=0)


def test_physical_vertex_changes_only_the_vertex_and_enforces_capacity(monkeypatch):
    lb, geometry, cfg, state, draws = _fixture()
    displaced, info = ed.sample_physical_vertex(
        draws, geometry, cfg, seed=20261017, run_id="test", event_id=3)
    assert not np.allclose(displaced.vertex_m, draws.vertex_m)
    for name in ("poisson_u", "charge_z", "transit_z", "dark_counts", "dark_times",
                 "qe_eff", "yield_photons", "emission_offset_ns"):
        np.testing.assert_array_equal(getattr(displaced, name), getattr(draws, name),
                                      err_msg=name)
    assert displaced.capacity == draws.capacity
    assert info["attempts"] >= 1

    # Capacity check: force an impossible capacity and require the redraw to fail.
    monkeypatch.setattr(lb, "count_capacity", lambda maximum: 10 ** 9)
    with pytest.raises(RuntimeError):
        ed.sample_physical_vertex(draws, geometry, cfg, seed=20261017,
                                  run_id="test", event_id=3, max_tries=4)
    # Fiducial check: a huge displacement can never stay inside the cylinder.
    with pytest.raises(RuntimeError):
        ed.sample_physical_vertex(draws, geometry, cfg, seed=20261017,
                                  run_id="test", event_id=3, delta_m=100.0,
                                  max_tries=4)


def test_median_time_transform_semantics():
    rng = np.random.default_rng(0)
    features = np.zeros((3, 6, 9))
    hit = rng.random((3, 6)) > 0.5
    features[..., ed.HIT_INDEX] = hit.astype(float)
    features[..., ed.LOG_CHARGE_INDEX] = rng.normal(size=(3, 6)) * hit
    features[..., ed.TIME_INDEX] = np.where(hit, rng.normal(size=(3, 6)) * 2.0, 0.0)
    features[2] = 0.0                     # an event with no hits
    hit = features[..., ed.HIT_INDEX] > 0
    transformed = ed.median_time_features(features)
    # Unhit tokens stay zero; no-hit event unchanged and finite.
    np.testing.assert_array_equal(transformed[..., ed.TIME_INDEX][~hit.astype(bool)],
                                  features[..., ed.TIME_INDEX][~hit.astype(bool)])
    np.testing.assert_array_equal(transformed[2], features[2])
    assert np.isfinite(transformed).all()
    # Hit-token time median is zero per event.
    for event in range(3):
        selected = transformed[event, :, ed.TIME_INDEX][hit[event]]
        if selected.size:
            assert abs(float(np.median(selected))) <= 1e-12
    # A constant shift of the hit times leaves the output unchanged.
    shifted = features.copy()
    shifted[..., ed.TIME_INDEX] = np.where(hit, features[..., ed.TIME_INDEX] + 3.5, 0.0)
    np.testing.assert_allclose(
        ed.median_time_features(shifted)[..., ed.TIME_INDEX],
        transformed[..., ed.TIME_INDEX], atol=1e-12, rtol=0)


def test_median_time_wrapper_trains_for_two_steps(tmp_path):
    pytest.importorskip("torch")
    import torch
    from jepa_noise import lucid_bombs as lb
    from jepa_noise import point_encoder as pe
    from jepa_noise.lucid_views import ViewMasker

    lb, geometry, cfg, state, draws = _fixture()
    try:
        lb.pmt_time_rule()
    except RuntimeError:
        pytest.skip("PMT NHits time rule is not merged")
    readout = lb.make_readout(geometry, cfg)
    hits = [readout(state, draws) for _ in range(2)]
    store = lb.sparse_from_readouts(np.arange(2, dtype=np.int64), hits)
    lb.write_sparse_hits(tmp_path / "hits.npz", store)
    scaling = pe.FeatureScaling.from_geometry(geometry)
    original = ed.install_median_time_wrapper(pe)
    try:
        batch = pe.tokenise(tmp_path, geometry, split="within", seed=0,
                            scaling=scaling,
                            masker=ViewMasker(geometry, "within", n_min=0))
        features = batch.features.numpy()
        hit = features[..., ed.HIT_INDEX] > 0
        for event in range(features.shape[0]):
            selected = features[event, :, ed.TIME_INDEX][hit[event]]
            if selected.size:
                assert abs(float(np.median(selected))) <= 1e-12
        assert pe.tokenise is not original
        # Two gradient steps of the real encoder on the wrapped batches.
        torch.manual_seed(0)
        model = pe.LucidPointJepa(pe.EncoderConfig(), geometry)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        probe_tokens = model.encode(batch.features)
        target = torch.zeros(probe_tokens.shape[0], probe_tokens.shape[2])
        for _ in range(2):
            optimizer.zero_grad()
            tokens = model.encode(batch.features)
            loss = torch.mean((tokens.mean(dim=1) - target) ** 2)
            loss.backward()
            optimizer.step()
            assert torch.isfinite(loss)
    finally:
        ed.remove_median_time_wrapper(pe, original)
    assert pe.tokenise is original
    # Restoration: the same store now tokenises without the transform.
    restored = pe.tokenise(tmp_path, geometry, split="within", seed=0,
                           scaling=scaling,
                           masker=ViewMasker(geometry, "within", n_min=0))
    restored_features = restored.features.numpy()
    hit = restored_features[..., ed.HIT_INDEX] > 0
    medians = [abs(float(np.median(restored_features[e, :, ed.TIME_INDEX][hit[e]])))
               for e in range(restored_features.shape[0]) if hit[e].any()]
    assert max(medians) > 1e-6
