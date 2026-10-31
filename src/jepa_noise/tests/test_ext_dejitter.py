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


# ---------------------------------------------------------------------------
# MAE de-jitter extension (runbook 2026-10-31, §3.4)
# ---------------------------------------------------------------------------
def _mae_batch(geometry, cfg):
    from jepa_noise import lucid_bombs as lb
    from jepa_noise.lucid_views import ViewMasker
    geometry = geometry
    cfg = cfg
    run_id = "test/mae/0000"
    state = lb.state_for_run("F1", 0.05, run_id, geometry, seed=17)
    draws = lb.draw_event(geometry, cfg, run_id, 0, seed=17, domain="photon")
    readout = lb.make_readout(geometry, cfg)(state, draws)
    return readout


def test_mae_defaults_and_new_options_are_backward_compatible(tmp_path):
    pytest.importorskip("torch")
    from jepa_noise import lucid_run as lr
    import scripts.jepa_ext_dejitter as drv
    assert drv.EXT == Path("/ceph/dwong/jepa/returns/ext_dejitter_20261030")
    assert drv.EXTENSION == "dejitter_2026-10-30"
    assert drv.RESULTS == drv.REPO / "results/jepa/20261030_dejitter"
    args = drv.parser().parse_args(
        ["train", "--variant", "within_trigger", "--seed", "9000",
         "--out", "/ceph/dwong/jepa/returns/ext_dejitter_20261030/train"])
    assert args.arm == "sigreg_jepa"
    assert args.ext is None and args.extension is None
    sens = drv.parser().parse_args(
        ["sens-encode", "--label", "x", "--spec-label", "fix9000"])
    assert sens.transform == "none" and sens.spec_label == "fix9000"
    assert drv.TRANSFORMS == {
        "across_median": "median_time", "within_median": "median_time",
        "across_trigger": "none", "within_trigger": "none"}
    config = lr.load_run_config(str(drv.CONFIG))
    cells = drv.cells_for(config)
    assert cells["across_trigger"].cell_id == cells["across"].cell_id
    assert cells["across_trigger"].split == cells["across"].split
    # The run-directory guard refuses an existing run dir and accepts a new one.
    (tmp_path / "runs" / "rid").mkdir(parents=True)
    with pytest.raises(SystemExit):
        drv.refuse_existing_run_dir(tmp_path, "rid")
    drv.refuse_existing_run_dir(tmp_path, "new_rid")   # no raise


def test_mae_loss_is_shift_invariant_under_the_wrapper(tmp_path):
    pytest.importorskip("torch")
    import torch
    from jepa_noise import lucid_arms as la
    from jepa_noise import lucid_bombs as lb
    from jepa_noise import point_encoder as pe
    from jepa_noise.lucid_views import ViewMasker
    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(str(CONFIG))
    readout = _mae_batch(geometry, cfg)
    store = lb.sparse_from_readouts(np.array([0]), [readout])
    shifted = dataclasses.replace(
        store, time_ns=(np.asarray(store.time_ns, np.float32) + 37.0).astype(np.float32))
    path_a = lb.write_sparse_hits(tmp_path / "hits_a.npz", store)
    path_b = lb.write_sparse_hits(tmp_path / "hits_b.npz", shifted)
    masker = ViewMasker(geometry, "within", n_min=0)
    arms_cfg = la.load_config(None)
    arm = la.build_arm("mae_mse", geometry, config=arms_cfg, cell=None, seed=0)
    original = pe.tokenise
    patched = ed.install_median_time_wrapper(pe)
    try:
        batch_a = pe.tokenise(path_a, geometry, split="within", seed=0,
                              occupancy="natural", masker=masker)
        batch_b = pe.tokenise(path_b, geometry, split="within", seed=0,
                              occupancy="natural", masker=masker)
        out_a = arm.forward(batch_a)
        out_b = arm.forward(batch_b)
    finally:
        ed.remove_median_time_wrapper(pe, patched)
    assert pe.tokenise is original
    loss_a, loss_b = float(out_a["loss"]), float(out_b["loss"])
    assert out_a["metrics"]["hidden_hits"] > 0
    assert abs(loss_b - loss_a) <= 1e-5 * abs(loss_a)
    # Negative control: without the wrapper the constant shift changes the loss.
    batch_na = pe.tokenise(path_a, geometry, split="within", seed=0,
                           occupancy="natural", masker=masker)
    batch_nb = pe.tokenise(path_b, geometry, split="within", seed=0,
                           occupancy="natural", masker=masker)
    raw_a = float(arm.forward(batch_na)["loss"])
    raw_b = float(arm.forward(batch_nb)["loss"])
    assert abs(raw_b - raw_a) > 1e-3 * abs(raw_a)
    # Target identity: zero the head, then metrics['time'] is the mean squared
    # median-subtracted hidden hit-token time.
    with torch.no_grad():
        arm.head.weight.zero_()
        if arm.head.bias is not None:
            arm.head.bias.zero_()
    metrics = arm.forward(batch_a)["metrics"]
    features = batch_a.features.cpu().numpy()
    hidden = batch_a.hidden.cpu().numpy()
    hit = features[..., ed.HIT_INDEX] > 0
    values = features[..., ed.TIME_INDEX][hidden & hit]
    assert values.size > 0
    np.testing.assert_allclose(metrics["time"], float(np.mean(values ** 2)),
                               atol=1e-6, rtol=0)


def test_mae_two_step_cpu_training_with_wrapper_then_restore(tmp_path):
    pytest.importorskip("torch")
    import torch
    from jepa_noise import lucid_arms as la
    from jepa_noise import lucid_bombs as lb
    from jepa_noise import point_encoder as pe
    from jepa_noise.lucid_views import ViewMasker
    geometry = lb.from_synthetic()
    cfg = lb.load_generator_config(str(CONFIG))
    readout = _mae_batch(geometry, cfg)
    store = lb.sparse_from_readouts(np.array([0]), [readout])
    path = lb.write_sparse_hits(tmp_path / "hits.npz", store)
    arms_cfg = la.load_config(None)
    arm = la.build_arm("mae_mse", geometry, config=arms_cfg, cell=None, seed=0)
    optimizer = torch.optim.AdamW(arm.parameters(), lr=1e-3)
    original = pe.tokenise
    patched = ed.install_median_time_wrapper(pe)
    try:
        batch = pe.tokenise(path, geometry, split="within", seed=0,
                            occupancy="natural", masker=ViewMasker(geometry, "within", n_min=0))
        losses = []
        for _ in range(2):
            out = arm.forward(batch)
            optimizer.zero_grad()
            out["loss"].backward()
            optimizer.step()
            losses.append(float(out["loss"]))
        assert all(np.isfinite(value) for value in losses)
    finally:
        ed.remove_median_time_wrapper(pe, patched)
    assert pe.tokenise is original


def test_mode_train_refuses_an_existing_run_directory(tmp_path):
    pytest.importorskip("torch")
    import scripts.jepa_ext_dejitter as drv
    from jepa_noise import lucid_run as lr
    if not Path(drv.GEOMETRY).exists():
        pytest.skip("WCTE geometry is only available on the hosts")
    config = lr.load_run_config(str(drv.CONFIG))
    cell = drv.cell_of(config, "across_trigger")
    run_id = lr.run_id_for(cell.cell_id, "mae_mse", 9000, None)
    (tmp_path / "runs" / run_id).mkdir(parents=True)
    args = type("Args", (), {
        "config": drv.CONFIG, "generator_config": drv.GEN_CONFIG,
        "arms_config": None, "geometry": drv.GEOMETRY,
        "data_root": drv.DATA_ROOT, "device": "cpu",
        "variant": "across_trigger", "arm": "mae_mse", "seed": 9000,
        "transform": "none", "steps": 20, "probe_max_events": 200,
        "out": str(tmp_path), "ext": None, "extension": None})()
    with pytest.raises(SystemExit):
        drv.mode_train(args)
