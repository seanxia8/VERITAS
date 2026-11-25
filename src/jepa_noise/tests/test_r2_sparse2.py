# SPDX-License-Identifier: MIT
"""r2-sparse2 T1 focused tests (runbook RUNBOOK_2026-11-25_R2_SPARSE2.md §2.2, §T1).

Blocking leakage tests:
(a) S1L: the queried PMT set is a function of the mask only;
(b) S1L: a query's input is its PMT's scaled xyz only (never charge, time or hit flag);
(c) S2L, S2Lt: predictor inputs are unchanged when the positions (and charges) of hidden hits change;
(d) S2L, S2Lt: the predictor output is unchanged when the hidden hits' PMT indices are permuted among
    themselves, holding their times (full attention among queries exposes only their times);
(e) all new cells: the encoder never sees a masked hit (exactly zero gradient of the encoder output,
    and of the loss through every non-target input column, with respect to masked hits' features).

Synthetic tokens on the frozen WCTE geometry (no generator, no physics); times are tie-free so the
time order does not depend on PMT indices.
"""
from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch

from jepa_noise import data_roots as dr
from jepa_noise import ext_r2 as r2
from jepa_noise import ext_r2_repair_train as tr
from jepa_noise import lucid_bombs as lb
from jepa_noise import r2_store as rs
from jepa_noise import sparse_arms as sa
from jepa_noise.lucid_views import ViewMasker
from jepa_noise.point_encoder import HIT_INDEX, LOG_CHARGE_INDEX, TIME_INDEX, geometry_features

SEED = 9000


@pytest.fixture(scope="module")
def wcte():
    geometry = dr.geometry("wcte", dr.LUCID_V1_W2)
    scaling = rs.scaling_for(geometry, 100.0)
    return geometry, scaling


def _store(geometry, n_events=3, seed=0, start=1_000_000, spread_ns=40.0):
    rng = np.random.default_rng(seed)
    offsets, pmt, q, t = [0], [], [], []
    for _ in range(n_events):
        hit = np.flatnonzero(rng.random(geometry.n_pmt) < 0.5)
        pmt.append(hit.astype(np.int32))
        q.append(np.exp(rng.normal(1.0, 1.0, hit.size)).astype(np.float32))
        tt = -10.0 + spread_ns * rng.permutation(hit.size) / hit.size + rng.uniform(0, 1e-3, hit.size)
        t.append(tt.astype(np.float32))   # tie-free
        offsets.append(offsets[-1] + hit.size)
    return lb.SparseHitStore(np.arange(start, start + n_events, dtype=np.int64), np.asarray(offsets, np.int64),
                             np.concatenate(pmt), np.concatenate(q), np.concatenate(t))


def _batch(geometry, scaling, n_events=3, seed=0, **kw):
    masker = ViewMasker(geometry, "spatial_patch", seed=lb.DEFAULT_SEED)
    return rs.tokenise_sparse(_store(geometry, n_events, seed, **kw), geometry, scaling=scaling, masker=masker)


def _cfg(geometry):
    return r2.encoder_config_for("mae_repaired", geometry)


CELLS = {"S1L": ("sparse_mae", {"queries": "all", "embed": "linear", "decoder": "legacy"}),
         "S2L": ("sparse_time", {"mask_seed": 20261017, "embed": "linear", "decoder": "full"}),
         "S2Lt": ("sparse_time", {"mask_seed": 20261017, "embed": "linear", "decoder": "full",
                                  "blocks": "time", "block_width_ns": 5.0})}


def _arm(wcte, cell, seed=SEED):
    geometry, scaling = wcte
    arm, flags = CELLS[cell]
    return r2.build_r2_arm(arm, geometry, seed, scaling=scaling, encoder_config=_cfg(geometry),
                           sparse_flags=dict(flags)).eval()


def _c2(wcte, seed=SEED):
    geometry, scaling = wcte
    flags = {"loss_mode": "fixed", "placeholder": "sentinel", "embed": "linear", "decoder": "legacy",
             "count_head": False, "sentinel_time": (50.0 + 50.0) / 100.0}
    return r2.build_r2_arm("mae_repaired", geometry, seed, scaling=scaling, encoder_config=_cfg(geometry),
                           sparse_flags=flags).eval()


def _with(batch, features):
    from dataclasses import replace
    return replace(batch, features=features)


# ----------------------------------------------------------------------------- T1.1 / T1.2 modules
def test_linear_cells_start_from_c2_init(wcte):
    c2 = _c2(wcte).state_dict()
    s1l = _arm(wcte, "S1L").state_dict()
    s2l = _arm(wcte, "S2L").state_dict()
    for key, value in s1l.items():
        if key.startswith("input_proj."):
            ref = c2["jepa.encoder." + key]
        elif key.startswith("encoder.blocks."):
            ref = c2["jepa." + key]
        elif key.startswith("predictor."):
            ref = c2["jepa." + key]
        elif key.startswith("head."):
            ref = c2[key]
        else:
            raise AssertionError(f"unexpected S1L parameter {key}")
        assert torch.equal(value, ref), key
    for key in ("input_proj.weight", "input_proj.bias"):
        assert torch.equal(s2l[key], c2["jepa.encoder." + key])
    assert all(torch.equal(s2l[k], c2["jepa." + k]) for k in s2l if k.startswith("encoder.blocks."))
    assert not any(k.startswith(("content.", "query.", "count_head.")) for k in s1l)


def test_linear_tokens_equal_c2_input_projection(wcte):
    geometry, scaling = wcte
    b = _batch(geometry, scaling)
    arm, c2 = _arm(wcte, "S1L"), _c2(wcte)
    x = b.features
    hit = x[..., HIT_INDEX] > 0.5
    idx, valid = sa.gather_tokens(hit, arm.hilbert_rank[None, :])
    tok = arm.token_inputs(x, idx)
    ref = c2.jepa.encoder.input_proj(c2._encoder_features(x))
    ref = ref.gather(1, idx[..., None].expand(-1, -1, ref.size(-1)))
    assert torch.allclose(tok[valid], ref[valid], atol=1e-6)


def test_legacy_predict_equals_c2_predictor_on_full_context(wcte):
    arm = _arm(wcte, "S1L")
    torch.manual_seed(0)
    ctx = torch.randn(2, 50, arm.d_model)
    q = torch.randn(2, 30, 3)
    ref = arm.predictor(ctx, q)
    out = sa.legacy_predict(arm.predictor, ctx, torch.ones(2, 50, dtype=torch.bool), q)
    assert torch.allclose(out, ref, atol=1e-5)
    # padding keys are ignored: appending padded context rows does not change the output
    pad = torch.cat([ctx, torch.randn(2, 7, arm.d_model)], 1)
    valid = torch.cat([torch.ones(2, 50, dtype=torch.bool), torch.zeros(2, 7, dtype=torch.bool)], 1)
    assert torch.allclose(sa.legacy_predict(arm.predictor, pad, valid, q), ref, atol=1e-5)


def test_legacy_decoder_refusals(wcte):
    geometry, scaling = wcte
    for flags in ({"queries": "neg", "neg_ratio": 0.5, "embed": "linear", "decoder": "legacy"},
                  {"queries": "pos", "embed": "linear", "decoder": "legacy"},
                  {"queries": "all", "embed": "fourier", "decoder": "legacy"},
                  {"queries": "all", "embed": "linear", "decoder": "full"},
                  {"queries": "all", "blocks": "time", "block_width_ns": 5.0}):
        with pytest.raises(ValueError):
            sa.build_sparse_arm("sparse_mae", geometry, SEED, encoder_config=_cfg(geometry), scaling=scaling, **flags)
    for flags in ({"decoder": "legacy"}, {"blocks": "time"}, {"block_width_ns": 5.0},
                  {"blocks": "count32", "block_width_ns": 5.0}):
        with pytest.raises(ValueError):
            sa.build_sparse_arm("sparse_time", geometry, SEED, encoder_config=_cfg(geometry), scaling=scaling, **flags)


# ----------------------------------------------------------------------------- leakage (a), (b)
def test_leakage_a_s1l_query_set_is_a_function_of_the_mask(wcte):
    geometry, scaling = wcte
    arm = _arm(wcte, "S1L")
    b1 = _batch(geometry, scaling, seed=0)
    hidden = b1.hidden
    hit1 = b1.features[..., HIT_INDEX] > 0.5
    hit2 = torch.rand(hit1.shape, generator=torch.Generator().manual_seed(5)) < 0.3   # different hits
    q1 = sa.gather_tokens(arm.query_set(hidden, hit1), arm.hilbert_rank[None, :])
    q2 = sa.gather_tokens(arm.query_set(hidden, hit2), arm.hilbert_rank[None, :])
    assert torch.equal(q1[0], q2[0]) and torch.equal(q1[1], q2[1])
    assert torch.equal(arm.query_set(hidden, hit1), hidden)


def test_leakage_b_s1l_query_input_is_scaled_xyz_only(wcte):
    geometry, scaling = wcte
    arm = _arm(wcte, "S1L")
    b = _batch(geometry, scaling)
    q_idx, q_valid = sa.gather_tokens(b.hidden, arm.hilbert_rank[None, :])
    xyz = torch.as_tensor(geometry_features(geometry, scaling)[:, :3])
    assert torch.equal(arm.query_inputs(q_idx), xyz[q_idx])
    assert arm.predictor.query_proj.in_features == 3
    # predictions do not move when the charge, time and hit flag of every queried PMT change
    x = b.features.clone()
    x2 = x.clone()
    h = b.hidden
    x2[..., LOG_CHARGE_INDEX] = torch.where(h, x[..., LOG_CHARGE_INDEX] + 3.0, x[..., LOG_CHARGE_INDEX])
    x2[..., TIME_INDEX] = torch.where(h, x[..., TIME_INDEX] * 0.3 + 0.1, x[..., TIME_INDEX])
    x2[..., HIT_INDEX] = torch.where(h, 1.0 - x[..., HIT_INDEX], x[..., HIT_INDEX])
    with torch.no_grad():
        p1 = arm(b)["prediction"]
        p2 = arm(_with(b, x2))["prediction"]
    assert torch.equal(p1, p2)


# ----------------------------------------------------------------------------- leakage (c), (d)
def _move_hidden_hits(arm, b, *, to_unhit: bool, seed=1):
    """New features where every hidden hit moves to another PMT (keeping its time): a permutation
    among the hidden hits' PMTs (``to_unhit=False``) or onto unhit PMTs with new charges."""
    lay = arm.layout(b)
    x = b.features.clone()
    gen = torch.Generator().manual_seed(seed)
    x2 = x.clone()
    for row in range(x.size(0)):
        src = lay["q_pmt"][row][lay["q_valid"][row]]
        if to_unhit:
            unhit = torch.nonzero(~lay["hit"][row], as_tuple=False)[:, 0]
            dst = unhit[torch.randperm(unhit.numel(), generator=gen)[:src.numel()]]
            assert dst.numel() == src.numel()
            x2[row, src, 6:] = 0.0
            x2[row, src, LOG_CHARGE_INDEX] = x[row, src, LOG_CHARGE_INDEX] * 0  # unhit placeholder
            x2[row, dst, TIME_INDEX] = x[row, src, TIME_INDEX]
            x2[row, dst, LOG_CHARGE_INDEX] = torch.randn(src.numel(), generator=gen)
            x2[row, dst, HIT_INDEX] = 1.0
        else:
            dst = src[torch.randperm(src.numel(), generator=gen)]
            x2[row, dst, TIME_INDEX] = x[row, src, TIME_INDEX]          # times held, PMTs permuted
            x2[row, dst, LOG_CHARGE_INDEX] = x[row, src, LOG_CHARGE_INDEX]
    return x2


@pytest.mark.parametrize("cell", ["S2L", "S2Lt"])
def test_leakage_c_s2_predictor_inputs_ignore_hidden_positions(wcte, cell):
    geometry, scaling = wcte
    arm = _arm(wcte, cell)
    b = _batch(geometry, scaling)
    p1 = arm.predictor_inputs(b)
    assert int(p1["q_valid"].sum()) > 0
    for to_unhit in (False, True):
        p2 = arm.predictor_inputs(_with(b, _move_hidden_hits(arm, b, to_unhit=to_unhit)))
        for key in ("tokens", "queries", "v_idx", "v_valid", "q_valid", "q_tau", "q_block"):
            assert torch.equal(p1[key], p2[key]), (cell, to_unhit, key)
    # queries are built from tau, the block / bin index and the mask embedding only
    assert arm.time_mlp[0].in_features == 1


@pytest.mark.parametrize("cell", ["S2L", "S2Lt"])
def test_leakage_d_s2_full_attention_output_ignores_hidden_pmt_indices(wcte, cell):
    geometry, scaling = wcte
    arm = _arm(wcte, cell)
    assert arm.predictor.attention == "full"
    b = _batch(geometry, scaling)
    with torch.no_grad():
        e1 = arm.run_predictor(arm.predictor_inputs(b))
        e2 = arm.run_predictor(arm.predictor_inputs(_with(b, _move_hidden_hits(arm, b, to_unhit=False))))
    assert torch.equal(e1, e2)


def test_full_attention_lets_queries_see_each_other(wcte):
    """Power check of (d): with full attention a query's output depends on the other queries'
    times (it does not with the isolated decoder of S2)."""
    geometry, scaling = wcte
    b = _batch(geometry, scaling)
    for cell, expect_change in (("S2L", True),):
        arm = _arm(wcte, cell)
        p = arm.predictor_inputs(b)
        with torch.no_grad():
            e1 = arm.run_predictor(p)
            q2 = p["queries"].clone()
            q2[0, 1:] = q2[0, 1:] + 0.5 * p["q_valid"][0, 1:, None]
            e2 = arm.run_predictor(dict(p, queries=q2))
        assert (not torch.allclose(e1[0, 0], e2[0, 0])) == expect_change
    iso = r2.build_r2_arm("sparse_time", geometry, SEED, scaling=scaling, encoder_config=_cfg(geometry),
                          sparse_flags={"mask_seed": 20261017}).eval()
    p = iso.predictor_inputs(b)
    with torch.no_grad():
        e1 = iso.run_predictor(p)
        q2 = p["queries"].clone()
        q2[0, 1:] = q2[0, 1:] + 0.5 * p["q_valid"][0, 1:, None]
        e2 = iso.run_predictor(dict(p, queries=q2))
    assert torch.allclose(e1[0, 0], e2[0, 0], atol=1e-6)


# ----------------------------------------------------------------------------- leakage (e)
def _masked_hits(arm, b):
    if isinstance(arm, sa.SparseMaeArm):
        return b.hidden & (b.features[..., HIT_INDEX] > 0.5)
    lay = arm.layout(b)
    return sa._mark(lay["hit"].shape, lay["q_pmt"], lay["q_valid"])


@pytest.mark.parametrize("cell", ["S1L", "S2L", "S2Lt"])
def test_leakage_e_encoder_never_sees_masked_hits(wcte, cell):
    geometry, scaling = wcte
    arm = _arm(wcte, cell).train()
    b = _batch(geometry, scaling)
    masked = _masked_hits(arm, b)
    assert int(masked.sum()) > 0
    # encoder output: exactly zero gradient w.r.t. every feature of every masked hit
    x = b.features.clone().requires_grad_(True)
    if isinstance(arm, sa.SparseMaeArm):
        hit = x[..., HIT_INDEX] > 0.5
        z, _, valid = arm.encode_hits(x, hit & ~b.hidden, "hilbert")
    else:
        p = arm.predictor_inputs(_with(b, x))
        z = arm.encoder(p["tokens"], p["v_valid"])
    (z ** 2).sum().backward()
    assert x.grad is not None and float(x.grad[masked].abs().max()) == 0.0
    # loss: zero gradient through every column that is not a target (S1L: position, direction, hit
    # flag; S2: position, direction, charge, hit flag; tau is a given query input of S2 by design)
    x = b.features.clone().requires_grad_(True)
    arm(_with(b, x))["loss"].backward()
    cols = [0, 1, 2, 3, 4, 5, HIT_INDEX] + ([] if cell == "S1L" else [LOG_CHARGE_INDEX])
    assert float(x.grad[masked][:, cols].abs().max()) == 0.0


# ----------------------------------------------------------------------------- S2Lt bins, f_own
def test_s2lt_time_bins(wcte):
    geometry, scaling = wcte
    arm = _arm(wcte, "S2Lt")
    assert arm.n_blocks == 33 and arm.block_embed.num_embeddings == 33
    tau = torch.tensor([-110.0, -105.01, -105.0, 0.0, 49.99, 50.0]) / 100.0
    assert arm.time_bin(tau).tolist() == [0, 0, 1, 22, 31, 32]
    b = _batch(geometry, scaling, n_events=4)
    lay = arm.layout(b)
    x = b.features
    for row in range(x.size(0)):
        hit = lay["hit"][row]
        bins = arm.time_bin(x[row, :, TIME_INDEX])
        nonempty = torch.unique(bins[hit])
        hidden_bins = torch.unique(lay["q_block"][row][lay["q_valid"][row]])
        assert hidden_bins.numel() == (nonempty.numel() + 1) // 2
        # every hit of a hidden bin is hidden; no visible token lies in a hidden bin
        in_hidden = torch.isin(bins, hidden_bins) & hit
        assert int(in_hidden.sum()) == int(lay["q_valid"][row].sum())
        vis = lay["v_idx"][row][lay["v_valid"][row]]
        assert not bool(torch.isin(bins[vis], hidden_bins).any())
    # keyed draw: same event, same hidden bins
    assert torch.equal(arm.layout(b)["q_block"], lay["q_block"])


def test_s2lt_loss_is_per_bin_mean(wcte):
    geometry, scaling = wcte
    arm = _arm(wcte, "S2Lt")
    b = _batch(geometry, scaling, n_events=2)
    with torch.no_grad():
        out = arm(b)
        p = arm.predictor_inputs(b)
        mass = arm.assignment_mass(arm.run_predictor(p), p)
    vals = []
    for row in range(mass.size(0)):
        v = p["q_valid"][row]
        blk, pmt, m = p["q_block"][row][v], p["q_pmt"][row][v], mass[row][v]
        for k in blk.unique():
            sel = blk == k
            lam = m[sel].sum(0)
            vals.append(float((-torch.log(lam[pmt[sel]] + sa.S2_EPS)).mean()))
    assert abs(float(out["loss"]) - float(np.mean(vals))) < 1e-5


def test_block_mass_stats(wcte):
    geometry, scaling = wcte
    b = _batch(geometry, scaling, n_events=2)
    for cell in ("S2L", "S2Lt"):
        st = _arm(wcte, cell).block_mass_stats(b)
        assert st["queries"] > 0 and st["own"] > 0
        assert st["own"] + st["other"] <= st["queries"] * (1 + 1e-5)
        assert 0 < st["chance_block_sum"] / st["n_blocks"] < 1
    s2 = r2.build_r2_arm("sparse_time", geometry, SEED, scaling=scaling, encoder_config=_cfg(geometry),
                         sparse_flags={"mask_seed": 20261017}).eval()
    st = s2.block_mass_stats(b)
    lay = s2.layout(b)
    expect = []
    for row in range(2):
        nb = len(torch.unique(lay["q_block"][row][lay["q_valid"][row]]))
        expect += [1.0 / nb] * nb
    assert abs(st["chance_block_sum"] / st["n_blocks"] - np.mean(expect)) < 1e-6


# ----------------------------------------------------------------------------- forward, CLI, defaults
@pytest.mark.parametrize("cell", list(CELLS))
def test_new_cells_forward_backward(wcte, cell):
    geometry, scaling = wcte
    arm = _arm(wcte, cell).train()
    out = arm(_batch(geometry, scaling))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    arm.check_parameter_coverage()
    grads = [p.grad for p in arm.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    if cell == "S1L":
        assert "count" not in out["metrics"] and arm.count_head is None


def _args(arm, **kw):
    base = dict(arm=arm, data_kind=dr.LUCID_V1_W2, loss=None, placeholder=None, embed=None, decoder=None,
                count_head=None, queries=None, neg_ratio=None, blocks=None, block_width_ns=None,
                mask=None, patch_size=None, time_loss_scale_ns=None, charge_loss_weight=None,
                time_reference=None)
    base.update(kw)
    return Namespace(**base)


def test_cli_suffixes_and_flags(wcte):
    _, scaling = wcte
    s1l = _args("sparse_mae", embed="linear", decoder="legacy")
    s2l = _args("sparse_time", embed="linear", decoder="full")
    s2lt = _args("sparse_time", embed="linear", decoder="full", blocks="time", block_width_ns=5.0)
    assert tr.run_suffix(s1l) == "__dw2__elin__dleg__qall"
    assert tr.run_suffix(s2l) == "__dw2__elin__dfull"
    assert tr.run_suffix(s2lt) == "__dw2__elin__dfull__bt5"
    assert tr.sparse_build_flags(s1l, scaling) == {"queries": "all", "embed": "linear", "decoder": "legacy"}
    assert tr.sparse_build_flags(s2lt, scaling) == {"mask_seed": None, "embed": "linear", "decoder": "full",
                                                    "blocks": "time", "block_width_ns": 5.0}
    # the unchanged cells: same suffix, same flags dict (so the same recipe hash)
    assert tr.run_suffix(_args("sparse_mae")) == "__dw2__qall"
    assert tr.run_suffix(_args("sparse_time")) == "__dw2"
    assert tr.sparse_build_flags(_args("sparse_mae"), scaling) == {"queries": "all"}
    assert tr.sparse_build_flags(_args("sparse_mae", queries="neg", neg_ratio=1.0), scaling) == {
        "queries": "neg", "neg_ratio": 1.0}
    assert tr.sparse_build_flags(_args("sparse_time"), scaling) == {"mask_seed": None}
    c2 = _args("mae_repaired", loss="fixed", placeholder="sentinel")
    assert tr.run_suffix(c2) == "__dw2__lfixed__psent"
    for bad in (_args("sparse_mae", decoder="full"), _args("sparse_time", decoder="legacy"),
                _args("sparse_mae", blocks="time", block_width_ns=5.0), _args("sparse_time", blocks="time"),
                _args("sparse_time", block_width_ns=5.0), _args("sparse_time", count_head=True),
                _args("mae_repaired", decoder="full", loss="fixed")):
        with pytest.raises(ValueError):
            tr.run_suffix(bad)
            tr.sparse_build_flags(bad, scaling)
            r2.build_r2_arm(bad.arm, wcte[0], SEED, scaling=scaling, encoder_config=_cfg(wcte[0]),
                            sparse_flags=tr.sparse_build_flags(bad, scaling))


def test_default_sparse_arms_unchanged(wcte):
    """S1 / S2 built with the explicit default r2-sparse2 flags equal the unchanged construction."""
    geometry, scaling = wcte
    for arm, flags, extra in (("sparse_mae", {"queries": "all"}, {"embed": "fourier", "decoder": "isolated"}),
                              ("sparse_time", {"mask_seed": 1}, {"embed": "fourier", "decoder": "isolated",
                                                                 "blocks": "count32"})):
        a = sa.build_sparse_arm(arm, geometry, SEED, encoder_config=_cfg(geometry), scaling=scaling, **flags)
        b = sa.build_sparse_arm(arm, geometry, SEED, encoder_config=_cfg(geometry), scaling=scaling,
                                **flags, **extra)
        sa_, sb = a.state_dict(), b.state_dict()
        assert sa_.keys() == sb.keys() and all(torch.equal(sa_[k], sb[k]) for k in sa_)
        assert any(k.startswith("content.") for k in sa_)
