"""Stage 4 regressions for effective-batch SIGReg, time scaling and EMA."""
from dataclasses import replace
import json
from unittest.mock import patch

import numpy as np
import pytest
import torch

from jepa_noise import ext_r2 as r2
from jepa_noise import lucid_arms as la
from jepa_noise import lucid_bombs as lb
from jepa_noise import lucid_run as lr
from jepa_noise import point_encoder as pe
from jepa_noise import r2_store as rs
from jepa_noise.lucid_views import ViewMasker, mask_from_key, token_hilbert_order
from jepa_noise.tests.r2_fixtures import small_cylinder


def _synthetic(split="within", time_scale=100.0, n_events=32):
    geometry = small_cylinder()
    rng = np.random.default_rng(87)
    reads = []
    for _ in range(n_events):
        count = int(rng.integers(36, 84))
        pmt = np.sort(rng.choice(geometry.n_pmt, count, replace=False))
        reads.append(lb.HitReadout(pmt, rng.uniform(0.3, 2.0, count),
                                   rng.normal(45.0, 20.0, count)))
    store = lb.sparse_from_readouts(np.arange(1000, 1000 + n_events), reads)
    scaling = rs.scaling_for(geometry, time_scale)
    masker = ViewMasker(geometry, split, seed=99, n_min=0)
    return geometry, scaling, store, rs.tokenise_sparse(store, geometry, scaling=scaling,
                                                        masker=masker)


def _arm(name, geometry, scaling):
    settings = la.ArmSettings.from_config()
    settings = replace(settings, sigreg=replace(settings.sigreg, num_slices=16))
    encoder = pe.EncoderConfig(d_model=16, d_pred=16, n_blocks=1, n_pred_blocks=1,
                               window=24, shift=12)
    return la.build_arm(name, geometry, settings=settings, seed=9000,
                        encoder_config=encoder, scaling=scaling)


def test_r2_shuffle_preserves_alignment_and_mixes_every_four_events():
    geometry, scaling, store, _ = _synthetic()
    runs = [f"fixture/{r}" for r in range(8)]
    manifests = [json.dumps({"runs": [{"event_id_start": 1000 + 4*i, "n_events": 4,
                                      "setting": 0.05}]}) for i in range(8)]
    domains = np.where(store.event_ids % 2 == 0, "photon", "muon")
    cohort = rs.R2Cohort("dev_train", runs, np.arange(0, 33, 4), store.event_ids, store.offsets,
                         store.pmt_id, store.charge_pe, store.time_ns,
                         {"domain": domains}, [""] * 8, manifests)
    old = rs.R2TrainingLoader(cohort, geometry, seed=9000, scaling=scaling, n_min=0)
    new = rs.R2TrainingLoader(cohort, geometry, seed=9000, scaling=scaling, n_min=0,
                              shuffle_events_within_batch=True)
    for step in (0, 30, 215, 9999):
        before, _ = old.batch_at(step)
        batch, _ = new.batch_at(step)
        replay, _ = new.batch_at(step)
        np.testing.assert_array_equal(batch.event_ids, replay.event_ids)
        assert set(batch.run_ids) == set(runs)
        assert all(len(set(batch.run_ids[i:i + 4])) > 1 for i in range(0, 32, 4))
        np.testing.assert_array_equal(batch.domains, domains[batch.event_ids - 1000])
        restore = np.argsort(batch.event_ids)
        torch.testing.assert_close(batch.features[restore], before.features, rtol=0, atol=0)
        torch.testing.assert_close(batch.target[restore], before.target, rtol=0, atol=0)


def test_spatial_patch_mask_is_keyed_and_rectangular():
    geometry, _, _, batch = _synthetic("spatial_patch")
    order = token_hilbert_order(geometry)
    patch_size = batch.keys[0]["patch_size"]
    for i, key in enumerate(batch.keys):
        replay = mask_from_key(geometry, key, batch.hit_flags[i])
        np.testing.assert_array_equal(replay.target, batch.target_index[i])
        complete = order[:(len(order) // patch_size) * patch_size].reshape(-1, patch_size)
        choices = np.isin(complete, replay.target)
        assert np.all(choices.all(axis=1) | ~choices.any(axis=1))
    assert not torch.equal(batch.target[0], batch.target[1])


class _Loader:
    def __init__(self, batch):
        self.batch = batch

    def batch_at(self, step):
        return self.batch, None


@pytest.mark.parametrize("name", ["sigreg_jepa", "jepa_repaired"])
def test_full_batch_sigreg_loss_and_gradients_match_micro_batches(tmp_path, name):
    geometry, scaling, _, batch = _synthetic()
    full, micro = [_arm(name, geometry, scaling) for _ in range(2)]
    records = []
    for model, accum in ((full, 1), (micro, 8)):
        directory = tmp_path / str(accum)
        identity = {"run_id": "probe", "arm": name, "seed": 9000,
                    "steps_scheduled": 1, "gradient_clip": None}
        records.append(lr.train_loop(
            arm=model, loader=_Loader(batch), steps=1, identity=identity, config_sha="test",
            optimizers=[torch.optim.SGD(model.parameters(), lr=1e-3)],
            checkpoint_path=directory / "checkpoint.pt", accum_steps=accum,
            snapshot_steps=(0, 1), full_batch_sigreg=True,
        ))
    for key in ("sigreg", "sigreg_context", "sigreg_target", "loss"):
        assert records[0]["metrics"][key] == pytest.approx(records[1]["metrics"][key], abs=1e-5)
    for p, q in zip(full.parameters(), micro.parameters()):
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=0)
    assert full.sigreg.sliced.global_step.item() == micro.sigreg.sliced.global_step.item()
    assert (tmp_path / "8/snapshot_step_00000.pt").is_file()
    snap = torch.load(tmp_path / "8/snapshot_step_00001.pt", weights_only=False)
    assert snap["step"] == 1 and snap["metrics"]["sigreg"] == records[1]["metrics"]["sigreg"]


def test_ema_teacher_has_no_gradient_and_latent_loss_excludes_unhit():
    geometry, scaling, _, batch = _synthetic(n_events=4)
    arm = _arm("jepa_repaired", geometry, scaling)
    before = arm.sigreg_objective(batch)
    saved = arm.sigreg_inputs(batch)
    inputs, prediction, labels, hidden, context, full = saved
    hit = batch.features[..., pe.HIT_INDEX].gather(1, hidden) > 0.5
    assert (~hit).any()
    with patch.object(arm, "sigreg_inputs", return_value=(
            inputs, prediction, labels + (~hit)[..., None] * 100, hidden, context, full)):
        after = arm.sigreg_objective(batch)
    assert before["metrics"]["prediction"] == after["metrics"]["prediction"]
    before["loss"].backward()
    assert all(p.grad is None for p in arm.target_encoder.parameters())
    target = next(arm.target_encoder.parameters())
    online = next(arm.jepa.encoder.parameters())
    old = target.clone()
    with torch.no_grad():
        online.add_(0.2)
    assert arm.update_target(0) == pytest.approx(0.992)
    torch.testing.assert_close(target, 0.992 * old + 0.008 * online)
    assert arm.settings.ema_tau(10000) == pytest.approx(0.9995)


@pytest.mark.parametrize("detector,time_scale", [("sk", 450.0), ("juno", 1000/3), ("hk", 2300/3)])
def test_balanced_mae_time_loss_and_window(detector, time_scale):
    geometry, scaling, _, batch = _synthetic("spatial_patch", time_scale=time_scale, n_events=4)
    geometry = replace(geometry, detector=detector)
    model = r2.build_r2_arm("mae_repaired", geometry, 9000, scaling=scaling)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        metrics = model(batch)["metrics"]
    assert model.time_loss_weight == pytest.approx((time_scale / 50) ** 2)
    assert metrics["time_fraction"] >= 0.15
    assert metrics["time"] == pytest.approx(metrics["time_raw"] * metrics["time_loss_weight"])
