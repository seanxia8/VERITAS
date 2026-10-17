"""WP4c tests: views/masks (§3.5.4) and the point encoder (§3.5.5), CPU only.

Fixtures are the schema-identical synthetic geometry (labelled NOT WCTE) and tiny
synthetic runs.  The amplitude number here is a **fixture** number at random
init; the registered R² >= 0.9 criterion is measured after training.
"""
import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_noise import lucid_bombs as lb
from jepa_noise import lucid_views as lv
from jepa_noise import point_encoder as pe

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "point_encoder_v0.json"
SMALL = lb.from_synthetic({"module_ids": tuple(range(4)),
                           "surface_counts": {"top": 16, "barrel": 44, "bottom": 16}})
FULL = lb.from_synthetic()


def _run(tmp_path, geometry, *, events=6, hits=50, seed=0, prefix=0):
    """Write a tiny synthetic run: `hits` distinct PMTs per event, positive charges."""
    rng = np.random.default_rng(seed)
    reads = [lb.HitReadout(
        np.sort(rng.choice(geometry.n_pmt, hits, replace=False)).astype(np.int32),
        (rng.exponential(3.0, hits) + 0.3).astype(np.float32),
        rng.normal(50.0, 5.0, hits).astype(np.float32)) for _ in range(events)]
    directory = tmp_path / f"run_{seed}_{prefix}"
    directory.mkdir(parents=True, exist_ok=True)
    ids = prefix + np.arange(events)
    lb.write_sparse_hits(directory / "hits.npz", lb.sparse_from_readouts(ids, reads))
    return directory


def _batch(tmp_path, geometry=SMALL, *, events=6, hits=50, seed=0, prefix=0,
           split="across", occupancy="natural"):
    directory = _run(tmp_path, geometry, events=events, hits=hits, seed=seed, prefix=prefix)
    return pe.tokenise(directory, geometry, split=split, seed=seed, occupancy=occupancy)


def _hit(geometry, store, index):
    hit = np.zeros(geometry.n_pmt, dtype=bool)
    span = slice(store.offsets[index], store.offsets[index + 1])
    hit[store.pmt_id[span]] = True
    return hit


@pytest.fixture(scope="module")
def small_model():
    torch.manual_seed(1)
    return pe.LucidPointJepa(pe.EncoderConfig(), SMALL)


# ------------------------------------------------------------------ §3.5.4 views
def test_hilbert_curve_is_a_continuous_bijection():
    bits = 3
    grid = np.stack(np.meshgrid(*[np.arange(1 << bits)] * 3, indexing="ij"), -1).reshape(-1, 3)
    distance = lv.hilbert_index_3d(grid, bits)
    assert np.array_equal(np.sort(distance), np.arange(grid.shape[0]))
    steps = np.abs(np.diff(grid[np.argsort(distance)], axis=0)).sum(axis=1)
    assert steps.max() == 1 and steps.min() == 1
    with pytest.raises(ValueError):
        lv.hilbert_index_3d(np.array([[1 << bits, 0, 0]]), bits)
    modules, tokens = lv.module_hilbert_order(FULL), lv.token_hilbert_order(FULL)
    assert np.array_equal(np.sort(modules), np.arange(FULL.n_modules))
    assert np.array_equal(np.sort(tokens), np.arange(FULL.n_pmt))
    assert np.array_equal(lv.module_hilbert_order(FULL), modules)  # cached, stable


def test_across_target_is_a_contiguous_hilbert_half():
    order = lv.module_hilbert_order(FULL)
    hit = np.zeros(FULL.n_pmt, dtype=bool)
    hit[np.arange(0, FULL.n_pmt, 3)] = True
    for event_id in (0, 5, 11):
        mask = lv.across_mask(FULL, event_id, hit, seed=4)
        modules = np.unique(FULL.module_index[mask.target])
        expected = order[(mask.offset + np.arange(FULL.n_modules // 2)) % FULL.n_modules]
        assert np.array_equal(np.sort(modules), np.sort(expected))
        assert mask.target.size == (FULL.n_modules // 2) * FULL.pmts_per_module
        assert mask.context.size + mask.target.size == FULL.n_pmt
        assert not np.intersect1d(mask.context, mask.target).size
        assert mask.key["geometry_sha256"] == FULL.digest


def test_across_nmin_redraws_and_exclusion_counters(tmp_path):
    store = lb.read_sparse_hits(_run(tmp_path, SMALL, events=3, hits=3, seed=1) / "hits.npz")
    masker = lv.ViewMasker(SMALL, "across", seed=0, cell="across/s")
    for index, event_id in enumerate(store.event_ids):
        mask = masker.mask(int(event_id), _hit(SMALL, store, index))
        assert not mask.accepted and mask.attempts == lb.MAX_REDRAWS
    report = masker.report()
    assert report["n_events"] == 3 and report["n_excluded"] == 3
    assert report["across_redraws"] == 3 * lb.MAX_REDRAWS
    assert report["across_redraw_rate"] == float(lb.MAX_REDRAWS)
    assert report["across_exclusion_rate"] == 1.0

    store = lb.read_sparse_hits(_run(tmp_path, SMALL, events=4, hits=50, seed=2) / "hits.npz")
    masker = lv.ViewMasker(SMALL, "across", seed=7, cell="across/mixed")
    masks = [masker.mask(int(event_id), _hit(SMALL, store, index))
             for index, event_id in enumerate(store.event_ids)]
    assert all(mask.accepted and mask.attempts == 0 for mask in masks)
    report = masker.report()
    assert 0.0 < report["context_hit_fraction"] < 1.0
    assert 0.0 < report["target_hit_fraction"] < 1.0


def test_within_partition_is_fixed_and_nine_ten_per_module():
    hit = np.ones(FULL.n_pmt, dtype=bool)
    first = lv.within_mask(FULL, 0, hit)
    second = lv.within_mask(FULL, 99, hit, n_min=20)
    assert np.array_equal(first.context, second.context)
    assert np.array_equal(first.target, second.target)
    assert np.all(FULL.pmt_id_in_mpmt[first.target] % 2 == 1)   # 1-based: odd = target
    assert np.all(FULL.pmt_id_in_mpmt[first.context] % 2 == 0)
    assert np.bincount(FULL.module_index[first.target]).min() == 10
    assert np.bincount(FULL.module_index[first.context]).min() == 9
    assert first.target.size == 10 * FULL.n_modules
    few = np.zeros(FULL.n_pmt, dtype=bool)
    few[:3] = True
    masker = lv.ViewMasker(FULL, "within", cell="within/s")
    assert not masker.mask(0, few).accepted
    report = masker.report()
    assert report["n_excluded"] == 1 and report["n_accepted"] == 0
    assert report["within_exclusion_rate"] == 1.0


def test_masks_are_deterministic_and_keys_recompute():
    hit = np.zeros(FULL.n_pmt, dtype=bool)
    hit[np.arange(7, FULL.n_pmt, 11)] = True
    for split, build, kwargs in (("across", lv.across_mask, {"seed": 5, "replicate": 2}),
                                 ("within", lv.within_mask, {})):
        first = build(FULL, 42, hit, **kwargs)
        again = build(FULL, 42, hit, **kwargs)
        assert np.array_equal(first.context, again.context)
        assert np.array_equal(first.target, again.target)
        rebuilt = lv.mask_from_key(FULL, first.key, hit)
        assert np.array_equal(rebuilt.target, first.target)
        assert np.array_equal(rebuilt.context, first.context)
        assert first.key["geometry_sha256"] == FULL.digest and first.key["event_id"] == 42
        if split == "across":
            other = build(FULL, 42, hit, seed=6, replicate=2)
            assert not np.array_equal(other.target, first.target)
            assert first.key["offset"] == first.offset


def test_occupancy_ablation_balances_context_only(small_model):
    hit = np.zeros(SMALL.n_pmt, dtype=bool)
    hit[np.arange(0, SMALL.n_pmt, 4)] = True
    natural = lv.ViewMasker(SMALL, "within", occupancy="natural").mask(0, hit)
    balanced = lv.ViewMasker(SMALL, "within", occupancy="balanced", seed=3).mask(0, hit)
    extra = balanced.extra_hidden
    assert extra.size == natural.context.size - 2 * natural.context_hits
    assert np.all(np.isin(extra, natural.context)) and not np.intersect1d(extra, natural.target).size
    assert not hit[extra].any()                       # only unhit context PMTs are dropped
    assert np.array_equal(balanced.target, natural.target)
    assert balanced.key["occupancy"] == "balanced"

    features = torch.zeros(2, SMALL.n_pmt, pe.N_FEATURES)
    hidden = torch.zeros(2, SMALL.n_pmt, dtype=torch.bool)
    hidden[:, balanced.hidden] = True
    features.requires_grad_(True)
    out = small_model.encode(features, hidden_mask=hidden)
    grads = torch.autograd.grad((out * 1.7).sum(), features)[0]
    assert float(grads[hidden].abs().max()) == 0.0


def test_hit_fractions_are_reported_per_view():
    hit = np.zeros(FULL.n_pmt, dtype=bool)
    hit[np.arange(1, FULL.n_pmt, 5)] = True
    mask = lv.within_mask(FULL, 0, hit)
    fractions = mask.fractions(hit)
    assert 0.0 <= fractions["context"] <= 1.0 and 0.0 <= fractions["target"] <= 1.0
    masker = lv.ViewMasker(FULL, "within", cell="within/s0.05")
    masker.mask(0, hit)
    report = masker.report()
    assert np.isclose(report["context_hit_fraction"], fractions["context"])
    assert np.isclose(report["target_hit_fraction"], fractions["target"])


# --------------------------------------------------------------- §3.5.5 encoder
def test_tokenise_shape_and_raw_feature_roundtrip(tmp_path):
    directory = _run(tmp_path, FULL, events=3, hits=300, seed=4)
    batch = pe.tokenise(directory, FULL, split="within", seed=0)
    store = lb.read_sparse_hits(directory / "hits.npz")
    assert tuple(batch.features.shape) == (3, FULL.n_pmt, pe.N_FEATURES)
    assert batch.features.dtype == torch.float32
    geometry_block = pe.geometry_features(FULL, batch.scaling)
    for row, _ in enumerate(batch.event_ids):
        span = slice(store.offsets[row], store.offsets[row + 1])
        pmt = store.pmt_id[span]
        charge = store.charge_pe[span].astype(np.float64)
        time_ns = store.time_ns[span].astype(np.float64)
        token = batch.features[row].numpy()
        assert np.allclose(token[pmt, pe.LOG_CHARGE_INDEX], np.log(charge), atol=1e-5)
        assert np.allclose(token[pmt, pe.TIME_INDEX], time_ns / 100.0, atol=1e-5)
        assert np.allclose(token[pmt, pe.HIT_INDEX], 1.0)
        unhit = np.setdiff1d(np.arange(FULL.n_pmt), pmt)
        assert np.all(token[unhit][:, pe.LOG_CHARGE_INDEX:] == 0.0)
        assert np.allclose(token[:, :6], geometry_block, atol=1e-6)
        assert np.array_equal(np.flatnonzero(np.isin(np.arange(FULL.n_pmt), pmt)),
                              np.flatnonzero(batch.hit_flags[row]))
    assert int(batch.target.sum()) + batch.context_index.numel() == 3 * FULL.n_pmt
    assert int((batch.target & batch.hidden).sum()) == int(batch.target.sum())
    assert len(batch.keys) == 3 and batch.report["n_accepted"] == 3
    assert np.array_equal(batch.log_charge, batch.features[..., pe.LOG_CHARGE_INDEX].numpy())


def test_leakage_zero_gradient_and_exact_invariance(tmp_path):
    for split, prefix in (("within", 0), ("across", 50)):
        batch = _batch(tmp_path, FULL, events=2, hits=200, split=split, seed=2, prefix=prefix)
        torch.manual_seed(0)
        model = pe.LucidPointJepa(pe.EncoderConfig(), FULL)
        features = batch.features.clone().requires_grad_(True)
        out = model.encode(features, hidden_mask=batch.hidden)
        projection = torch.randn(out.shape, generator=torch.Generator().manual_seed(3))
        grads = torch.autograd.grad((out * projection).sum(), features)[0]
        hidden = batch.hidden
        assert float(grads[hidden].abs().max()) == 0.0    # every feature, not only charge
        for column in (pe.LOG_CHARGE_INDEX, pe.TIME_INDEX, pe.HIT_INDEX):
            assert float(grads[..., column][hidden].abs().max()) == 0.0
        assert float(grads[~hidden].abs().max()) > 0.0
        changed = batch.features.clone()
        changed[..., pe.LOG_CHARGE_INDEX][hidden] = 9.0
        changed[..., pe.TIME_INDEX][hidden] = -2.0
        changed[..., pe.HIT_INDEX][hidden] = 0.0
        with torch.no_grad():
            before = model.encode(batch.features, hidden_mask=hidden)
            after = model.encode(changed, hidden_mask=hidden)
        assert torch.equal(before, after)


def test_amplitude_fixture_probe(tmp_path):
    batch = pe.tokenise(_run(tmp_path, FULL, events=8, hits=300, seed=9), FULL,
                        split="within", seed=0)
    torch.manual_seed(2)
    model = pe.LucidPointJepa(pe.EncoderConfig(), FULL)
    result = pe.amplitude_probe(model, batch, train_fraction=0.5)
    assert result["status"] == "fixture_not_the_registered_criterion"
    assert result["n_features"] == pe.EncoderConfig().d_model and result["n_train"] > 0
    assert result["r2"] > 0.95, result          # fixture at random init, not the criterion
    print(f"\n[fixture] amplitude probe R2 = {result['r2']:.4f} "
          f"(n_train={result['n_train']}); the registered criterion is post-training")


def test_ten_step_determinism(tmp_path):
    batch = _batch(tmp_path, SMALL, events=4, hits=45, split="within")

    def train():
        torch.manual_seed(5)
        model = pe.LucidPointJepa(pe.EncoderConfig(), SMALL)
        optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
        losses = []
        for _ in range(10):
            optimiser.zero_grad(set_to_none=True)
            loss = pe.smoke_loss(model, batch)
            loss.backward()
            optimiser.step()
            losses.append(float(loss))
        return losses

    started = time.time()
    first, second = train(), train()
    assert first == second
    assert all(np.isfinite(first))
    assert time.time() - started < 60.0


def test_no_final_norm_is_applied(small_model, tmp_path):
    batch = _batch(tmp_path, SMALL, events=2, hits=60, split="within")
    calls = []

    def hook(module, inputs, output):
        calls.append(type(module).__name__)

    handles = [module.register_forward_hook(hook) for module in small_model.encoder.modules()
               if isinstance(module, (torch.nn.LayerNorm, torch.nn.RMSNorm))]
    try:
        out = small_model.encode(batch.features, hidden_mask=batch.hidden)
    finally:
        for handle in handles:
            handle.remove()
    assert small_model.encoder.final_norm is None
    assert len(calls) == 2 * pe.EncoderConfig().n_blocks  # two pre-norms per block, no final norm
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert not torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


def test_parameter_count_and_smoke_batch_two(tmp_path):
    batch = pe.tokenise(_run(tmp_path, FULL, events=2, hits=300, seed=1), FULL,
                        split="across", seed=0)
    torch.manual_seed(0)
    model = pe.LucidPointJepa(pe.EncoderConfig(), FULL)
    n_parameters = model.n_parameters()
    assert 900_000 < n_parameters < 950_000
    loss = pe.smoke_loss(model, batch)
    assert torch.isfinite(loss)
    loss.backward()
    norms = [float(p.grad.norm()) for p in model.parameters() if p.grad is not None]
    assert norms and all(np.isfinite(norms)) and max(norms) > 0.0
    print(f"\n[record] trainable parameters = {n_parameters}; batch-2 loss = {float(loss):.4f}")


def test_config_is_frozen_and_optimiser_is_pending():
    config = json.loads(CONFIG_PATH.read_text())
    defaults = pe.EncoderConfig()
    architecture = config["architecture"]
    assert architecture["d_model"] == defaults.d_model == 128
    assert architecture["n_blocks"] == defaults.n_blocks == 4
    assert architecture["n_heads"] == defaults.n_heads == 4
    assert architecture["window"] == defaults.window == 64
    assert architecture["window_shift"] == defaults.shift == 32
    assert architecture["d_ff"] == defaults.d_ff_resolved
    assert architecture["final_norm"] is None
    assert architecture["parameter_count"] == 911296
    predictor = config["predictor"]
    assert predictor["d_pred"] == defaults.d_pred == 64
    assert predictor["n_blocks"] == defaults.n_pred_blocks == 2
    assert config["training"]["optimizer"] == "pending_review"
    assert all(value == "pending_review" for value in
               config["training"]["optimizer_detail"].values())
    assert config["features"]["order"] == list(pe.FEATURES)
    assert config["views"]["n_min"] == lb.N_MIN
    assert config["views"]["max_redraws"] == lb.MAX_REDRAWS
    assert config["views"]["occupancy_ablation"]["default"] == "natural"
    batch_32 = next(m for m in config["memory"]["measurements"] if m["batch"] == 32)
    assert batch_32["peak_process_rss_mib"] < 0.8 * 12 * 1024     # §6.2: 9.6 GB on a Titan
