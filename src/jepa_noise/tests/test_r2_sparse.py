# SPDX-License-Identifier: MIT
"""r2-sparse T1 focused tests (runbook RUNBOOK_2026-11-24_R2_SPARSE_WCTE.md §T1).

Blocking leakage tests: (a) S1/S1n/S1p predictions for one query do not change when other queries
are added or removed; (b) S2 predictor inputs are unchanged when the positions of hidden hits are
permuted; (c) the count head output does not depend on the number of queries; (d) S1n negatives
all lie inside masked patches.
"""
from __future__ import annotations

import ast
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_noise import data_roots as dr
from jepa_noise import ext_r2 as r2
from jepa_noise import lucid_bombs as lb
from jepa_noise import r2_store as rs
from jepa_noise import sparse_arms as sa
from jepa_noise.lucid_views import ViewMasker
from jepa_noise.point_encoder import HIT_INDEX, LOG_CHARGE_INDEX, TIME_INDEX

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def wcte():
    geometry = dr.geometry("wcte", dr.LUCID_V1)
    scaling = rs.scaling_for(geometry, 100.0)
    return geometry, scaling


def _store(geometry, n_events=3, seed=0, start=1_000_000):
    rng = np.random.default_rng(seed)
    offsets, pmt, q, t = [0], [], [], []
    for _ in range(n_events):
        hit = np.flatnonzero(rng.random(geometry.n_pmt) < 0.5)
        pmt.append(hit.astype(np.int32))
        q.append(np.exp(rng.normal(1.0, 1.0, hit.size)).astype(np.float32))
        tt = rng.uniform(-110.0, 50.0, hit.size).astype(np.float32)
        tt[: min(5, hit.size)] = 7.0  # ties in time
        t.append(tt)
        offsets.append(offsets[-1] + hit.size)
    return lb.SparseHitStore(np.arange(start, start + n_events, dtype=np.int64), np.asarray(offsets, np.int64),
                             np.concatenate(pmt), np.concatenate(q), np.concatenate(t))


def _batch(geometry, scaling, n_events=3, seed=0):
    store = _store(geometry, n_events, seed)
    masker = ViewMasker(geometry, "spatial_patch", seed=lb.DEFAULT_SEED)
    return rs.tokenise_sparse(store, geometry, scaling=scaling, masker=masker)


def _cfg(geometry):
    return r2.encoder_config_for("mae_repaired", geometry)


# ----------------------------------------------------------------------------- data kinds (T1.1)
def test_data_kind_registry():
    assert dr.LUCID_V1_W2 in dr.DATA_KINDS
    assert dr.pack("wcte", dr.LUCID_V1_W2).name == "wcte_lucid_v1_w2.h5"
    assert dr.sealed_pack("wcte", dr.LUCID_V1_W2).name == "wcte_lucid_v1_w2_sealed.h5"
    assert dr.wcte_train_shards(dr.LUCID_V1_W2).parts[-3:] == ("lucid_v1_w2", "wcte", "wcte_train")
    with pytest.raises(ValueError):
        dr.pack("sk", dr.LUCID_V1_W2)


def test_pack_kind_mismatch_raises(tmp_path):
    h5py = pytest.importorskip("h5py")
    for kind, attr in dr.PACK_KIND.items():
        path = tmp_path / f"{kind}.h5"
        with h5py.File(path, "w") as f:
            f.attrs["dataset_kind"] = attr
        assert dr.check_pack_kind(path, kind) == attr
        other = dr.LUCID_V1_W2 if kind == dr.LUCID_V1 else dr.LUCID_V1
        with pytest.raises(ValueError):
            dr.check_pack_kind(path, other)
    for kind, attr in dr.PACK_KIND.items():
        dr.check_manifest_kind({"dataset_kind": attr}, kind)
        other = dr.LUCID_V1_W2 if kind == dr.LUCID_V1 else dr.LUCID_V1
        with pytest.raises(ValueError):
            dr.check_manifest_kind({"dataset_kind": attr}, other)


@pytest.mark.skipif(not dr.wcte_train_shards(dr.LUCID_V1).is_dir(), reason="needs the lucid_v1 WCTE data")
def test_real_packs_refuse_the_other_kind():
    from jepa_noise.ext_r2_repair_train import WcteTrainingShards
    WcteTrainingShards(dr.wcte_train_shards(dr.LUCID_V1), data_kind=dr.LUCID_V1)
    with pytest.raises(ValueError):
        WcteTrainingShards(dr.wcte_train_shards(dr.LUCID_V1), data_kind=dr.LUCID_V1_W2)
    with pytest.raises(ValueError):
        dr.check_pack_kind(dr.pack("wcte", dr.LUCID_V1), dr.LUCID_V1_W2)
    if dr.pack("wcte", dr.LUCID_V1_W2).is_file():
        with pytest.raises(ValueError):
            dr.check_pack_kind(dr.pack("wcte", dr.LUCID_V1_W2), dr.LUCID_V1)
        with pytest.raises(ValueError):
            WcteTrainingShards(dr.wcte_train_shards(dr.LUCID_V1_W2), data_kind=dr.LUCID_V1)


# ----------------------------------------------------------------------------- embedding, placeholders, loss
def test_fourier_code(wcte):
    geometry, scaling = wcte
    code, meta = sa.fourier_code(geometry, scaling)
    assert code.shape == (geometry.n_pmt, 128)
    assert np.all(code[:, 126:] == 0)
    assert 0.65 < meta["rms_per_dim"] < 0.75
    for axis in "xyz":
        a = meta["axes"][axis]
        assert a["wavelength_max_scaled"] == 4.0 and 0 < a["wavelength_min_scaled"] < 1.0
    assert meta["median_nn_spacing_m"] > 0


def test_sentinel_placeholders(wcte):
    geometry, scaling = wcte
    b = _batch(geometry, scaling, 2)
    x = sa.sentinel_features(b.features, (50.0 + 50.0) / 100.0)
    hit = b.features[..., HIT_INDEX] > 0.5
    assert torch.all(x[..., TIME_INDEX][~hit] == 1.0)
    assert torch.all(x[..., LOG_CHARGE_INDEX][~hit] == 0.0)
    assert torch.allclose(x[..., LOG_CHARGE_INDEX][hit], torch.log1p(torch.exp(b.features[..., LOG_CHARGE_INDEX][hit])))
    assert torch.all(x[..., LOG_CHARGE_INDEX][hit] > 0)
    assert torch.equal(x[..., TIME_INDEX][hit], b.features[..., TIME_INDEX][hit])


def test_fixed_loss_terms():
    torch.manual_seed(0)
    out = torch.randn(2, 10, 3)
    ln_q, tau = torch.randn(2, 10), torch.randn(2, 10) * 0.01
    hit = torch.zeros(2, 10, dtype=torch.bool)
    hit[:, :3] = True
    valid = torch.ones_like(hit)
    terms = sa.fixed_loss(out, ln_q, tau, hit, valid, time_scale_ns=100.0)
    w = min(max(14 / 6, 0.05), 20.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(out[..., 2], hit.float(), reduction="none")
    weights = torch.where(hit, torch.tensor(w), torch.tensor(1.0))
    assert torch.allclose(terms["hit"], (bce * weights).sum() / weights.sum())
    assert torch.allclose(terms["charge"], ((out[..., 0] - ln_q)[hit] ** 2).mean())
    # 2 x Huber equals r^2 for |r| <= 0.2 (10 ns) -> the legacy weight (T / 50 ns)^2 on small residuals
    small = out.clone()
    small[..., 1] = tau + 0.04  # r = 0.04 * 100 / 50 = 0.08
    t2 = sa.fixed_loss(small, ln_q, tau, hit, valid, time_scale_ns=100.0)
    assert torch.allclose(t2["time"], torch.tensor(0.08 ** 2), atol=1e-6)
    big = out.clone()
    big[..., 1] = tau + 1.0  # r = 2 -> 2 * 0.2 * (2 - 0.1)
    t3 = sa.fixed_loss(big, ln_q, tau, hit, valid, time_scale_ns=100.0)
    assert torch.allclose(t3["time"], torch.tensor(2 * 0.2 * (2.0 - 0.1)), atol=1e-5)
    # positives-only (S1p): no hit term
    t4 = sa.fixed_loss(out, ln_q, tau, hit, valid, time_scale_ns=100.0, hit_term=False)
    assert "hit" not in t4 and torch.allclose(t4["loss"], t4["charge"] + t4["time"])
    # weight clip
    hit2 = torch.ones_like(hit)  # no negatives -> w clipped at 0.05
    t5 = sa.fixed_loss(out, ln_q, tau, hit2, valid, time_scale_ns=100.0)
    assert float(t5["pos_weight"]) == pytest.approx(0.05)


# ----------------------------------------------------------------------------- tokeniser
def test_sparse_tokeniser_orders(wcte):
    geometry, scaling = wcte
    b = _batch(geometry, scaling, 2)
    arm = sa.SparseMaeArm(geometry, encoder_config=_cfg(geometry), scaling=scaling, seed=1)
    hit = b.features[..., HIT_INDEX] > 0.5
    idx, valid = sa.gather_tokens(hit, arm.hilbert_rank[None, :])
    for row in range(2):
        sel = idx[row, valid[row]]
        assert torch.equal(torch.sort(sel).values, torch.nonzero(hit[row])[:, 0])
        assert torch.all(torch.diff(arm.hilbert_rank[sel]) > 0)
    tau = b.features[..., TIME_INDEX]
    idx, valid = sa.gather_tokens(hit, tau)
    for row in range(2):
        sel = idx[row, valid[row]]
        t = tau[row, sel]
        assert torch.all(torch.diff(t) >= 0)
        ties = torch.diff(t) == 0
        assert torch.all(torch.diff(sel)[ties] > 0)  # ties by PMT index


# ----------------------------------------------------------------------------- leakage (blocking)
def test_leakage_a_query_isolation_s1(wcte):
    geometry, scaling = wcte
    b = _batch(geometry, scaling, 2)
    arm = sa.SparseMaeArm(geometry, encoder_config=_cfg(geometry), scaling=scaling, seed=2).eval()
    x, hidden = b.features, b.hidden
    hit = x[..., HIT_INDEX] > 0.5
    with torch.no_grad():
        z, _, valid = arm.encode_hits(x, hit & ~hidden, "hilbert")
        q_all, _ = sa.gather_tokens(hidden, arm.hilbert_rank[None, :])
        full = arm.predict_queries(z, valid, q_all)
        sub = q_all[:, ::3]
        part = arm.predict_queries(z, valid, sub)
        torch.testing.assert_close(part, full[:, ::3], rtol=1e-5, atol=1e-6)
        one = arm.predict_queries(z, valid, q_all[:, 5:6])
        torch.testing.assert_close(one, full[:, 5:6], rtol=1e-5, atol=1e-6)
        perm = torch.randperm(q_all.size(1))
        torch.testing.assert_close(arm.predict_queries(z, valid, q_all[:, perm]), full[:, perm], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("queries", ["neg", "pos"])
def test_leakage_a_query_isolation_s1n_s1p(wcte, queries):
    geometry, scaling = wcte
    b = _batch(geometry, scaling, 2)
    arm = sa.SparseMaeArm(geometry, encoder_config=_cfg(geometry), scaling=scaling, seed=3,
                          queries=queries, neg_ratio=0.5).eval()
    x, hidden = b.features, b.hidden
    hit = x[..., HIT_INDEX] > 0.5
    with torch.no_grad():
        z, _, valid = arm.encode_hits(x, hit & ~hidden, "hilbert")
        qs = arm.query_set(hidden, hit)
        q_idx, q_valid = sa.gather_tokens(qs, arm.hilbert_rank[None, :])
        full = arm.predict_queries(z, valid, q_idx)
        extra = torch.cat([q_idx, q_idx[:, :7].flip(1)], dim=1)
        more = arm.predict_queries(z, valid, extra)
        torch.testing.assert_close(more[:, :q_idx.size(1)], full, rtol=1e-5, atol=1e-6)


def test_leakage_a_query_isolation_c3(wcte):
    geometry, scaling = wcte
    b = _batch(geometry, scaling, 2)
    arm = r2.build_r2_arm("mae_repaired", geometry, 4, scaling=scaling,
                          sparse_flags={"loss_mode": "fixed", "placeholder": "sentinel", "embed": "fourier",
                                        "decoder": "isolated", "count_head": True, "sentinel_time": 1.0}).eval()
    with torch.no_grad():
        feats = arm._encoder_features(b.features)
        ctx = arm.jepa.encode(feats, hidden_mask=b.hidden)
        hidden_index = arm._hidden_index(b)
        full = arm.jepa.predict(ctx, b.context_index, hidden_index)
        part = arm.jepa.predict(ctx, b.context_index, hidden_index[:, 10:20])
        torch.testing.assert_close(part, full[:, 10:20], rtol=1e-5, atol=1e-6)


def test_leakage_b_s2_hidden_positions(wcte):
    geometry, scaling = wcte
    b = _batch(geometry, scaling, 2, seed=5)
    arm = sa.SparseTimeArm(geometry, encoder_config=_cfg(geometry), scaling=scaling, seed=6).eval()
    with torch.no_grad():
        p = arm.predictor_inputs(b)
        z = arm.encoder(p["tokens"], p["v_valid"])
        out = arm.predictor(z, p["v_valid"], p["queries"])
    # permute the positions (PMT ids) of the hidden hits: their charge/time move to other hidden-hit PMTs
    feats = b.features.clone()
    for row in range(feats.size(0)):
        pm = p["q_pmt"][row, p["q_valid"][row]]
        perm = pm[torch.randperm(pm.numel())]
        feats[row, perm, 6:9] = b.features[row, pm, 6:9]
    b2 = replace(b, features=feats)
    with torch.no_grad():
        p2 = arm.predictor_inputs(b2)
        z2 = arm.encoder(p2["tokens"], p2["v_valid"])
        out2 = arm.predictor(z2, p2["v_valid"], p2["queries"])
    for key in ("tokens", "queries", "v_valid", "q_valid", "q_block", "q_tau", "v_idx"):
        assert torch.equal(p[key], p2[key]), key
    torch.testing.assert_close(out, out2)


def test_leakage_c_count_head_independent_of_queries(wcte):
    geometry, scaling = wcte
    b = _batch(geometry, scaling, 2)
    rates = []
    for queries in ("all", "neg", "pos"):
        arm = sa.SparseMaeArm(geometry, encoder_config=_cfg(geometry), scaling=scaling, seed=7,
                              queries=queries, neg_ratio=1.0).eval()
        x, hidden = b.features, b.hidden
        hit = x[..., HIT_INDEX] > 0.5
        with torch.no_grad():
            z, _, valid = arm.encode_hits(x, hit & ~hidden, "hilbert")
            rates.append(arm.count_rate(z, valid))
    torch.testing.assert_close(rates[0], rates[1])
    torch.testing.assert_close(rates[0], rates[2])


def test_leakage_d_s1n_negatives_inside_masked_patches(wcte):
    geometry, scaling = wcte
    b = _batch(geometry, scaling, 3)
    for r in (0.25, 0.5, 1.0, 2.0):
        arm = sa.SparseMaeArm(geometry, encoder_config=_cfg(geometry), scaling=scaling, seed=8,
                              queries="neg", neg_ratio=r)
        hit = b.features[..., HIT_INDEX] > 0.5
        qs = arm.query_set(b.hidden, hit)
        assert torch.all(~qs | b.hidden)                       # every query inside the masked patches
        assert torch.all(qs[b.hidden & hit])                    # every masked hit queried
        for row in range(qs.size(0)):
            n_h = int((b.hidden[row] & hit[row]).sum())
            n_u = int((b.hidden[row] & ~hit[row]).sum())
            assert int((qs[row] & ~hit[row]).sum()) == min(n_u, int(np.ceil(r * n_h)))


# ----------------------------------------------------------------------------- arms run, readout adapter
@pytest.mark.parametrize("spec", [
    ("mae_repaired", {"loss_mode": "fixed", "placeholder": "sentinel", "sentinel_time": 1.0}),
    ("mae_repaired", {"loss_mode": "fixed", "placeholder": "sentinel", "embed": "fourier", "decoder": "isolated",
                      "count_head": True, "sentinel_time": 1.0}),
    ("sparse_mae", {"queries": "all"}), ("sparse_mae", {"queries": "neg", "neg_ratio": 0.5}),
    ("sparse_mae", {"queries": "pos"}), ("sparse_time", {"mask_seed": lb.DEFAULT_SEED})])
def test_arms_forward_backward(wcte, spec):
    geometry, scaling = wcte
    arm_name, flags = spec
    b = _batch(geometry, scaling, 2)
    arm = r2.build_r2_arm(arm_name, geometry, 9, scaling=scaling, sparse_flags=flags)
    out = arm(b, step=0)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    arm.check_parameter_coverage()
    z = arm.encode_tokens(b.features)
    assert z.shape == (2, geometry.n_pmt, 128) and torch.isfinite(z).all()
    if arm_name.startswith("sparse"):
        hit = b.features[..., HIT_INDEX] > 0.5
        assert torch.all(z[~hit] == 0)


def test_defaults_unchanged(wcte):
    geometry, scaling = wcte
    a = r2.build_r2_arm("mae_repaired", geometry, 9000, scaling=scaling)
    b = r2.build_r2_arm("mae_repaired", geometry, 9000, scaling=scaling, sparse_flags=None)
    assert a.state_dict().keys() == b.state_dict().keys()
    for k in a.state_dict():
        assert torch.equal(a.state_dict()[k], b.state_dict()[k])
    assert a.count_head is None and a.loss_mode == "legacy" and a.placeholder == "legacy"


def test_adapter_equals_dense_on_synthetic(wcte):
    from jepa_noise.r2_probe_repair import moment_features
    from jepa_noise.sparse_readout import adapter_moment_features
    geometry, scaling = wcte
    b = _batch(geometry, scaling, 3)
    arm = r2.build_r2_arm("mae_repaired", geometry, 11, scaling=scaling).eval()
    with torch.no_grad():
        z = arm.encode_tokens(b.features).numpy()
    raw = b.features.numpy()
    hit = torch.as_tensor(raw[..., 8] > 0.5)
    idx, valid = sa.gather_tokens(hit, torch.arange(geometry.n_pmt)[None, :].float())
    zh = np.take_along_axis(z, idx.numpy()[..., None], axis=1)
    lat_a, full_a = adapter_moment_features(zh, idx.numpy(), valid.numpy(), raw)
    lat_d, full_d = moment_features(z, raw)
    np.testing.assert_allclose(lat_a, lat_d, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(full_a, full_d, rtol=1e-6, atol=1e-7)


@pytest.mark.skipif(not dr.pack("wcte", dr.LUCID_V1).is_file(), reason="needs the lucid_v1 WCTE pack")
def test_golden_adapter_lucid_v1_200_events():
    """T1.6 golden test: on lucid_v1, a dense model's spatial-moment features through the adapter on
    its hit tokens equal the dense path for 200 events (latent and latent + 15 raw summaries)."""
    from jepa_noise.r2_probe_repair import moment_features
    from jepa_noise.sparse_readout import adapter_moment_features
    geometry = dr.geometry("wcte", dr.LUCID_V1)
    scaling = rs.scaling_for(geometry, 100.0)
    cohort = rs.open_cohort(dr.pack("wcte", dr.LUCID_V1), "dev_probe_test", truth_fields=())
    arm = r2.build_r2_arm("mae_repaired", geometry, 9000, scaling=scaling).eval()
    masker = ViewMasker(geometry, "within", seed=0, n_min=0)
    done = 0
    for i in range(cohort.n_runs):
        store = cohort.run_store(i)
        batch = rs.tokenise_sparse(store, geometry, scaling=scaling, masker=masker)
        with torch.no_grad():
            z = arm.encode_tokens(batch.features).numpy()
        raw = batch.features.numpy()
        hit = torch.as_tensor(raw[..., 8] > 0.5)
        idx, valid = sa.gather_tokens(hit, torch.arange(geometry.n_pmt)[None, :].float())
        zh = np.take_along_axis(z, idx.numpy()[..., None], axis=1)
        lat_a, full_a = adapter_moment_features(zh, idx.numpy(), valid.numpy(), raw)
        lat_d, full_d = moment_features(z, raw)
        np.testing.assert_allclose(lat_a, lat_d, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(full_a, full_d, rtol=1e-6, atol=1e-7)
        done += len(batch.event_ids)
        if done >= 200:
            break
    assert done >= 200


# ----------------------------------------------------------------------------- W2 data path
def test_w2_merge_rules():
    from jepa_noise import lucid_w2 as w2
    dig = {"sensor": np.array([3, 5]), "pe": np.array([2.0, 1.0]), "t": np.array([100.0, 50.0]),
           "physics": np.array([True, False])}
    added = {"sensor": np.array([3, 5, 7, 3]), "pe": np.array([1.0, 1.0, 1.0, 1.0]),
             "t": np.array([-50.0, 300.0, 10.0, -150.0])}
    out, c = w2.merge_digits(dig, added, 200.0)
    # sensor 3: added at -50 is 150 ns before the cached 100 -> merged (time -50, charge 3) ; then the
    # added -150 is 100 ns before the merged digit (-50) -> merged again (time -150, charge 4)
    k = np.flatnonzero(out["sensor"] == 3)
    assert k.size == 1 and out["t"][k[0]] == -150.0 and out["pe"][k[0]] == 4.0 and out["physics"][k[0]]
    # sensor 5: added at 300 is 250 ns after the cached 50 -> separate digit
    assert np.sum(out["sensor"] == 5) == 2
    assert c["n_merge_before"] == 2 and c["n_ext_new"] == 2
    assert out["dark_ext"][k[0]]


def test_w2_window_and_proxy():
    from jepa_noise import lucid_w2 as w2
    rule = w2.window_rule({"n": 6, "trigger_window_ns": 20.0, "prompt_window_ns": [-20.0, 60.0]}, 110, 50)
    assert rule["prompt_window_ns"] == [-110.0, 50.0]
    # an isolated early physics digit (dark-opened) is skipped by the first-light proxy
    assert w2.first_light_proxy([-90.0, 0.0, 1.0, 2.0]) == 0.0
    assert w2.first_light_proxy([0.0, 5.0]) == 0.0


def test_w2_path_imports_no_toy_physics():
    """G6 (static) for the W2 re-pack path: allowed lucid_bombs names only, no r2_generator."""
    from jepa_noise import lucid_mc_generator as mg
    for rel in ("src/jepa_noise/lucid_w2.py", "scripts/sparse_w2_repack.py", "scripts/sparse_w2_pack.py"):
        tree = ast.parse((REPO / rel).read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                assert not any("r2_generator" in n or "lucid_bombs_physics" in n for n in names), rel
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "lb":
                assert node.attr in mg.ALLOWED_LB_NAMES, f"{rel}: lb.{node.attr}"
                assert node.attr not in mg.FORBIDDEN_LB_NAMES
