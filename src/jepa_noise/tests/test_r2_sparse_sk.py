# SPDX-License-Identifier: MIT
"""r2-sparse-sk T1 focused tests (runbook RUNBOOK_2026-11-26_R2_SPARSE_SK.md §2.3, §2.4, §T1).

Blocking leakage tests (§2.4):
(a) S1L, S1A: the queried PMT set depends on the mask only;
(b) S1A: an event's absence tokens are unchanged when hits in masked patches are added, removed or
    moved (and so is the encoder output);
(c) S2L: predictor inputs are unchanged when hidden hits' positions are permuted; full attention among
    queries exposes only hit times;
(d) all new cells (SK-C2, SK-S1L, SK-S1A, SK-S2L, W-S1A): the gradient with respect to masked hits'
    input features is exactly zero (encoder output: every feature; loss: every non-target column);
(e) SK-C2's recipe equals SK-D0's except --loss and --placeholder (CLI-level here; the full
    field-by-field diff against the reused run is ``scripts/sparse_sk_recipe_check.sh``).

Synthetic tokens on the frozen lucid_v1 SK and lucid_v1_w2 WCTE geometries (no generator, no
physics); times are tie-free so the time order does not depend on PMT indices.
"""
from __future__ import annotations

from argparse import Namespace
from dataclasses import replace

import numpy as np
import pytest
import torch

from jepa_noise import data_roots as dr
from jepa_noise import ext_r2 as r2
from jepa_noise import ext_r2_repair_train as tr
from jepa_noise import lucid_bombs as lb
from jepa_noise import r2_store as rs
from jepa_noise import sparse_arms as sa
from jepa_noise.lucid_views import ViewMasker, default_patch_size, token_hilbert_order
from jepa_noise.point_encoder import HIT_INDEX, LOG_CHARGE_INDEX, TIME_INDEX

SEED = 9000


@pytest.fixture(scope="module")
def sk():
    geometry, scaling = tr.geometry_and_scaling("sk", dr.LUCID_V1)   # the production path
    assert scaling.time_scale_ns == 450.0
    return geometry, scaling


@pytest.fixture(scope="module")
def wcte():
    geometry = dr.geometry("wcte", dr.LUCID_V1_W2)
    return geometry, rs.scaling_for(geometry, 100.0)


DET = {"sk": "sk", "wcte": "wcte"}


def _store(geometry, n_events=2, seed=0, occupancy=0.14, start=2_000_000, spread_ns=200.0):
    rng = np.random.default_rng(seed)
    offsets, pmt, q, t = [0], [], [], []
    for _ in range(n_events):
        hit = np.flatnonzero(rng.random(geometry.n_pmt) < occupancy)
        pmt.append(hit.astype(np.int32))
        q.append(np.exp(rng.normal(1.0, 1.0, hit.size)).astype(np.float32))
        tt = -10.0 + spread_ns * rng.permutation(hit.size) / hit.size + rng.uniform(0, 1e-3, hit.size)
        t.append(tt.astype(np.float32))   # tie-free
        offsets.append(offsets[-1] + hit.size)
    return lb.SparseHitStore(np.arange(start, start + n_events, dtype=np.int64), np.asarray(offsets, np.int64),
                             np.concatenate(pmt), np.concatenate(q), np.concatenate(t))


def _batch(det, n_events=2, seed=0, split="spatial_patch", **kw):
    geometry, scaling = det
    masker = ViewMasker(geometry, split, seed=lb.DEFAULT_SEED)
    occ = kw.pop("occupancy", 0.14 if geometry.n_pmt > 5000 else 0.5)
    return rs.tokenise_sparse(_store(geometry, n_events, seed, occupancy=occ, **kw), geometry,
                              scaling=scaling, masker=masker)


def _cfg(geometry):
    return r2.encoder_config_for("mae_repaired", geometry)


def _flags(cell, geometry):
    if cell == "S1L":
        return {"queries": "all", "embed": "linear", "decoder": "legacy"}
    if cell == "S1A":
        return {"queries": "all", "embed": "linear", "decoder": "legacy", "absence_tokens": True,
                "absence_patch_size": default_patch_size(geometry.n_pmt)}
    if cell == "S2L":
        return {"mask_seed": lb.DEFAULT_SEED, "embed": "linear", "decoder": "full"}
    raise KeyError(cell)


def _arm(det, cell, seed=SEED):
    geometry, scaling = det
    name = "sparse_time" if cell == "S2L" else "sparse_mae"
    return r2.build_r2_arm(name, geometry, seed, scaling=scaling, encoder_config=_cfg(geometry),
                           sparse_flags=_flags(cell, geometry)).eval()


def _c2(det, seed=SEED):
    geometry, scaling = det
    win = dr.window_ns(geometry.detector, dr.LUCID_V1 if geometry.detector != "wcte" else dr.LUCID_V1_W2)
    flags = {"loss_mode": "fixed", "placeholder": "sentinel", "embed": "linear", "decoder": "legacy",
             "count_head": False, "sentinel_time": (win[1] + 50.0) / float(scaling.time_scale_ns)}
    return r2.build_r2_arm("mae_repaired", geometry, seed, scaling=scaling, encoder_config=_cfg(geometry),
                           sparse_flags=flags).eval()


def _with(batch, features):
    return replace(batch, features=features)


# ----------------------------------------------------------------------------- windows, encoder
def test_detector_windows_and_sentinel(sk):
    assert dr.window_ns("sk", dr.LUCID_V1) == (-20.0, 270.0)
    assert dr.window_ns("hk", dr.LUCID_V1) == (-20.0, 460.0)
    assert dr.window_ns("juno", dr.LUCID_V1) == (-20.0, 200.0)
    assert dr.window_ns("wcte", dr.LUCID_V1_W2) == dr.WINDOW_NS[dr.LUCID_V1_W2]
    assert dr.window_ns("wcte", dr.LUCID_V1) == dr.WINDOW_NS[dr.LUCID_V1]
    _, scaling = sk
    args = _args("mae_repaired", detector="sk", data_kind=dr.LUCID_V1, loss="fixed", placeholder="sentinel")
    flags = tr.sparse_build_flags(args, scaling)
    assert abs(flags["sentinel_time"] - 320.0 / 450.0) < 1e-12          # runbook §2.1: 0.711
    w = _args("mae_repaired", detector="wcte", data_kind=dr.LUCID_V1_W2, loss="fixed", placeholder="sentinel")
    assert tr.sparse_build_flags(w, rs.scaling_for(dr.geometry("wcte", dr.LUCID_V1_W2), 100.0))[
        "sentinel_time"] == 1.0                                         # WCTE C2 unchanged


def test_sparse_encoder_uses_detector_window(sk, wcte):
    assert _cfg(sk[0]).window == 384 and _cfg(sk[0]).shift == 192
    from jepa_noise.point_encoder import EncoderConfig
    assert _cfg(wcte[0]) == EncoderConfig()                             # WCTE sparse recipes unchanged
    assert _arm(sk, "S1L").encoder.config.window == 384


# ----------------------------------------------------------------------------- S1A modules
@pytest.mark.parametrize("name", ["sk", "wcte"])
def test_s1a_shares_s1l_init_and_adds_absence_modules(sk, wcte, name):
    det = sk if name == "sk" else wcte
    s1l, s1a = _arm(det, "S1L").state_dict(), _arm(det, "S1A").state_dict()
    extra = set(s1a) - set(s1l)
    assert extra == {"absence_proj.weight", "absence_proj.bias", "absence_type"}
    assert all(torch.equal(s1l[k], s1a[k]) for k in s1l)
    arm = _arm(det, "S1A")
    assert arm.absence_proj.in_features == 6 and arm.absence_proj.out_features == 128
    groups = arm.parameter_groups()
    assert all(p is not arm.absence_type and p is not arm.absence_proj.weight for p in groups["muon"])


def test_absence_patch_layout(sk):
    geometry, _ = sk
    arm = _arm(sk, "S1A")
    size = default_patch_size(geometry.n_pmt)
    assert size == 173 and arm.n_absence == geometry.n_pmt // size + 1   # 64 patches + tail of 24
    order = token_hilbert_order(geometry)
    po = arm.absence_patch_of.numpy()
    for k in (0, 17, 63):
        assert set(po[order[k * size:(k + 1) * size]]) == {k}
    assert set(po[order[64 * size:]]) == {64}
    # key: just before the patch's first PMT in the Hilbert order
    assert float(arm.absence_key[5]) == 5 * size - 0.5


def test_absence_features_values(sk):
    geometry, _ = sk
    arm = _arm(sk, "S1A")
    b = _batch(sk, n_events=1)
    x, hidden = b.features, b.hidden
    feat, present = arm.absence_features(x, hidden)
    po = arm.absence_patch_of.numpy()
    hid = hidden[0].numpy()
    hit = (x[0, :, HIT_INDEX] > 0.5).numpy()
    q = np.exp(x[0, :, LOG_CHARGE_INDEX].numpy())
    for p in range(arm.n_absence):
        members = po == p
        vis = members & ~hid
        assert bool(present[0, p]) == bool(vis.any())
        if p < 64:   # patch mask: a full patch is either visible or masked
            assert vis.sum() in (0, members.sum())
        if not vis.any():
            continue
        h = vis & hit
        expect = np.concatenate([x[0, vis, 0:3].numpy().mean(0),
                                 [h.sum() / vis.sum(), np.log1p(h.sum()), np.log1p(q[h].sum())]])
        np.testing.assert_allclose(feat[0, p].numpy(), expect, rtol=1e-5, atol=1e-5)
    n_full_visible = sum(1 for p in range(64) if bool(present[0, p]))
    assert n_full_visible == 32   # SK default mask: 32 of 64 patches hidden


def test_absence_token_placement_in_hilbert_order(sk):
    arm = _arm(sk, "S1A")
    b = _batch(sk, n_events=1)
    x, hidden = b.features, b.hidden
    hit = x[..., HIT_INDEX] > 0.5
    feat, present = arm.absence_features(x, hidden)
    sel = torch.cat([hit & ~hidden, present], 1)
    key = torch.cat([arm.hilbert_rank.float(), arm.absence_key])[None]
    idx, valid = sa.gather_tokens(sel, key)
    seq = idx[0][valid[0]].tolist()
    n = arm.n_pmt
    po = arm.absence_patch_of
    # patch ids along the sequence never decrease, and each absence token opens its patch's run
    pids = [j - n if j >= n else int(po[j]) for j in seq]
    assert pids == sorted(pids)
    for pos, j in enumerate(seq):
        if j >= n:
            assert pos == 0 or pids[pos - 1] < pids[pos]
    assert sum(1 for j in seq if j >= n) == int(present.sum())


def test_readout_excludes_absence_tokens(sk):
    arm = _arm(sk, "S1A")
    b = _batch(sk, n_events=1)
    with torch.no_grad():
        z = arm.encode_tokens(b.features)
    hit = b.features[..., HIT_INDEX] > 0.5
    assert z.shape == (1, arm.n_pmt, 128) and torch.all(z[~hit] == 0)
    assert bool((z[hit].abs().sum(-1) > 0).all())


# ----------------------------------------------------------------------------- leakage (a)
@pytest.mark.parametrize("cell", ["S1L", "S1A"])
@pytest.mark.parametrize("name", ["sk", "wcte"])
def test_leakage_a_query_set_is_a_function_of_the_mask(sk, wcte, cell, name):
    det = sk if name == "sk" else wcte
    arm = _arm(det, cell)
    b = _batch(det, seed=0)
    hidden = b.hidden
    hit1 = b.features[..., HIT_INDEX] > 0.5
    hit2 = torch.rand(hit1.shape, generator=torch.Generator().manual_seed(5)) < 0.3
    q1 = sa.gather_tokens(arm.query_set(hidden, hit1), arm.hilbert_rank[None, :])
    q2 = sa.gather_tokens(arm.query_set(hidden, hit2), arm.hilbert_rank[None, :])
    assert torch.equal(q1[0], q2[0]) and torch.equal(q1[1], q2[1])
    assert torch.equal(arm.query_set(hidden, hit1), hidden)


# ----------------------------------------------------------------------------- leakage (b)
def _perturb_masked(b, seed=3):
    """Variants of the features in which hits inside masked PMTs are added, removed and moved."""
    x = b.features
    hidden = b.hidden
    hit = x[..., HIT_INDEX] > 0.5
    gen = torch.Generator().manual_seed(seed)
    out = {}
    add = x.clone()      # every masked unhit PMT becomes a hit with a random charge and time
    m = hidden & ~hit
    add[..., HIT_INDEX] = torch.where(m, torch.ones_like(add[..., HIT_INDEX]), add[..., HIT_INDEX])
    add[..., LOG_CHARGE_INDEX] = torch.where(m, torch.randn(m.shape, generator=gen), add[..., LOG_CHARGE_INDEX])
    add[..., TIME_INDEX] = torch.where(m, torch.rand(m.shape, generator=gen), add[..., TIME_INDEX])
    out["add"] = add
    rem = x.clone()      # every masked hit removed
    m = hidden & hit
    rem[..., HIT_INDEX] = torch.where(m, torch.zeros_like(rem[..., HIT_INDEX]), rem[..., HIT_INDEX])
    rem[..., LOG_CHARGE_INDEX] = torch.where(m, torch.zeros_like(rem[..., LOG_CHARGE_INDEX]), rem[..., LOG_CHARGE_INDEX])
    out["remove"] = rem
    mov = x.clone()      # masked hits permuted among the masked PMTs (features 6..8 moved)
    for row in range(x.size(0)):
        hp = torch.nonzero(hidden[row], as_tuple=False)[:, 0]
        perm = hp[torch.randperm(hp.numel(), generator=gen)]
        mov[row, perm, 6:] = x[row, hp, 6:]
    out["move"] = mov
    return out


@pytest.mark.parametrize("name", ["sk", "wcte"])
def test_leakage_b_absence_tokens_ignore_masked_hits(sk, wcte, name):
    det = sk if name == "sk" else wcte
    arm = _arm(det, "S1A")
    b = _batch(det, seed=1)
    x, hidden = b.features, b.hidden
    hit = x[..., HIT_INDEX] > 0.5
    f1, p1 = arm.absence_features(x, hidden)
    with torch.no_grad():
        z1, i1, v1 = arm.encode_hits(x, hit & ~hidden, "hilbert", hidden=hidden)
    for kind, x2 in _perturb_masked(b).items():
        assert not torch.equal(x2, x), kind
        f2, p2 = arm.absence_features(x2, hidden)
        assert torch.equal(p1, p2), kind
        assert torch.equal(f1[p1], f2[p2]), kind
        hit2 = x2[..., HIT_INDEX] > 0.5
        with torch.no_grad():
            z2, i2, v2 = arm.encode_hits(x2, hit2 & ~hidden, "hilbert", hidden=hidden)
        assert torch.equal(i1, i2) and torch.equal(v1, v2) and torch.equal(z1, z2), kind
    # power check: changing a visible hit does change a visible patch's absence token
    x3 = x.clone()
    vis_hit = torch.nonzero((hit & ~hidden)[0], as_tuple=False)[0, 0]
    x3[0, vis_hit, LOG_CHARGE_INDEX] += 2.0
    f3, _ = arm.absence_features(x3, hidden)
    assert not torch.equal(f1, f3)


def test_leakage_b_holds_under_random_mask(sk):
    """Stage H generalisation: with a random mask a patch's token uses its visible PMTs only."""
    arm = _arm(sk, "S1A")
    b = _batch(sk, seed=2, split="random")
    x, hidden = b.features, b.hidden
    f1, p1 = arm.absence_features(x, hidden)
    assert bool(p1.all())
    for kind, x2 in _perturb_masked(b).items():
        f2, p2 = arm.absence_features(x2, hidden)
        assert torch.equal(p1, p2) and torch.equal(f1, f2), kind


# ----------------------------------------------------------------------------- leakage (c)
def _move_hidden_hits(arm, b, *, to_unhit: bool, seed=1):
    lay = arm.layout(b)
    x = b.features.clone()
    gen = torch.Generator().manual_seed(seed)
    x2 = x.clone()
    for row in range(x.size(0)):
        src = lay["q_pmt"][row][lay["q_valid"][row]]
        if to_unhit:
            unhit = torch.nonzero(~lay["hit"][row], as_tuple=False)[:, 0]
            dst = unhit[torch.randperm(unhit.numel(), generator=gen)[:src.numel()]]
            x2[row, src, 6:] = 0.0
            x2[row, dst, TIME_INDEX] = x[row, src, TIME_INDEX]
            x2[row, dst, LOG_CHARGE_INDEX] = torch.randn(src.numel(), generator=gen)
            x2[row, dst, HIT_INDEX] = 1.0
        else:
            dst = src[torch.randperm(src.numel(), generator=gen)]
            x2[row, dst, TIME_INDEX] = x[row, src, TIME_INDEX]
            x2[row, dst, LOG_CHARGE_INDEX] = x[row, src, LOG_CHARGE_INDEX]
    return x2


def test_leakage_c_s2l_predictor_inputs_ignore_hidden_positions(sk):
    arm = _arm(sk, "S2L")
    b = _batch(sk, seed=4)
    p1 = arm.predictor_inputs(b)
    assert int(p1["q_valid"].sum()) > 0
    for to_unhit in (False, True):
        p2 = arm.predictor_inputs(_with(b, _move_hidden_hits(arm, b, to_unhit=to_unhit)))
        for key in ("tokens", "queries", "v_idx", "v_valid", "q_valid", "q_tau", "q_block"):
            assert torch.equal(p1[key], p2[key]), (to_unhit, key)
    assert arm.time_mlp[0].in_features == 1
    assert arm.predictor.attention == "full"
    with torch.no_grad():
        e1 = arm.run_predictor(p1)
        e2 = arm.run_predictor(arm.predictor_inputs(_with(b, _move_hidden_hits(arm, b, to_unhit=False))))
    assert torch.equal(e1, e2)


# ----------------------------------------------------------------------------- leakage (d)
CELLS_D = [("sk", "C2"), ("sk", "S1L"), ("sk", "S1A"), ("sk", "S2L"), ("wcte", "S1A")]


@pytest.mark.parametrize("name,cell", CELLS_D)
def test_leakage_d_zero_gradient_wrt_masked_hits(sk, wcte, name, cell):
    det = sk if name == "sk" else wcte
    b = _batch(det, n_events=1, seed=5)
    hit = b.features[..., HIT_INDEX] > 0.5
    if cell == "C2":
        arm = _c2(det).train()
        masked = b.hidden & hit
        x = b.features.clone().requires_grad_(True)
        z = arm.encode_tokens(x, hidden_mask=b.hidden)
        (z ** 2).sum().backward()
        assert int(masked.sum()) > 0 and float(x.grad[masked].abs().max()) == 0.0
        # loss: zero through every non-target column (targets: ln q, tau, hit flag of hidden PMTs)
        x = b.features.clone().requires_grad_(True)
        arm(_with(b, x), step=0)["loss"].backward()
        assert float(x.grad[masked][:, [0, 1, 2, 3, 4, 5]].abs().max()) == 0.0
        return
    arm = _arm(det, cell).train()
    if cell == "S2L":
        lay = arm.layout(b)
        masked = sa._mark(lay["hit"].shape, lay["q_pmt"], lay["q_valid"])
    else:
        masked = b.hidden & hit
    assert int(masked.sum()) > 0
    x = b.features.clone().requires_grad_(True)
    if cell == "S2L":
        p = arm.predictor_inputs(_with(b, x))
        z = arm.encoder(p["tokens"], p["v_valid"])
    elif cell == "S1A":
        z, _, _ = arm.encode_hits(x, (x[..., HIT_INDEX] > 0.5) & ~b.hidden, "hilbert", hidden=b.hidden)
    else:
        z, _, _ = arm.encode_hits(x, (x[..., HIT_INDEX] > 0.5) & ~b.hidden, "hilbert")
    (z ** 2).sum().backward()
    assert x.grad is not None and float(x.grad[masked].abs().max()) == 0.0
    # S1A: also every masked PMT (hit or not) has exactly zero gradient (absence features)
    if cell == "S1A":
        assert float(x.grad[b.hidden].abs().max()) == 0.0
    x = b.features.clone().requires_grad_(True)
    arm(_with(b, x))["loss"].backward()
    cols = [0, 1, 2, 3, 4, 5, HIT_INDEX] + ([LOG_CHARGE_INDEX] if cell == "S2L" else [])
    assert float(x.grad[masked][:, cols].abs().max()) == 0.0


# ----------------------------------------------------------------------------- forward, CLI
@pytest.mark.parametrize("cell", ["S1A", "S2L"])
def test_sk_cells_forward_backward(sk, cell):
    arm = _arm(sk, cell).train()
    out = arm(_batch(sk, n_events=1, seed=6))
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    arm.check_parameter_coverage()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in arm.parameters() if p.requires_grad)
    if cell == "S1A":
        assert out["metrics"]["n_absence_tokens"] == 33   # 32 visible patches + the tail


def _args(arm, **kw):
    base = dict(arm=arm, detector="sk", data_kind=dr.LUCID_V1, loss=None, placeholder=None, embed=None,
                decoder=None, count_head=None, queries=None, neg_ratio=None, blocks=None, block_width_ns=None,
                mask=None, patch_size=None, time_loss_scale_ns=None, charge_loss_weight=None,
                time_reference=None, absence_tokens=None)
    base.update(kw)
    return Namespace(**base)


def test_cli_suffixes_flags_and_refusals(sk):
    _, scaling = sk
    s1l = _args("sparse_mae", embed="linear", decoder="legacy")
    s1a = _args("sparse_mae", embed="linear", decoder="legacy", absence_tokens=True)
    s2l = _args("sparse_time", embed="linear", decoder="full")
    c2 = _args("mae_repaired", loss="fixed", placeholder="sentinel")
    assert tr.run_suffix(s1l) == "__elin__dleg__qall"
    assert tr.run_suffix(s1a) == "__elin__dleg__abs__qall"
    assert tr.run_suffix(s2l) == "__elin__dfull"
    assert tr.run_suffix(c2) == "__lfixed__psent"
    assert tr.sparse_build_flags(s1a, scaling) == {"queries": "all", "embed": "linear", "decoder": "legacy",
                                                   "absence_tokens": True, "absence_patch_size": 173}
    assert tr.sparse_build_flags(s1l, scaling) == {"queries": "all", "embed": "linear", "decoder": "legacy"}
    w = _args("sparse_mae", detector="wcte", data_kind=dr.LUCID_V1_W2, embed="linear", decoder="legacy",
              absence_tokens=True)
    assert tr.run_suffix(w) == "__dw2__elin__dleg__abs__qall"
    assert tr.sparse_build_flags(w, scaling)["absence_patch_size"] == 32
    # Stage H: sparse_mae takes the mask and patch size; absence patches follow the mask's patches
    h = _args("sparse_mae", embed="linear", decoder="legacy", absence_tokens=True, patch_size=43)
    assert tr.run_suffix(h) == "__ps43__elin__dleg__abs__qall"
    assert tr.sparse_build_flags(h, scaling)["absence_patch_size"] == 43
    hr = _args("sparse_mae", embed="linear", decoder="legacy", absence_tokens=True, mask="random")
    assert tr.run_suffix(hr) == "__mrandom__elin__dleg__abs__qall"
    assert tr.sparse_build_flags(hr, scaling)["absence_patch_size"] == 173
    for bad in (_args("sparse_time", embed="linear", decoder="full", absence_tokens=True),
                _args("mae_repaired", loss="fixed", placeholder="sentinel", absence_tokens=True),
                _args("sparse_mae", absence_tokens=True),                        # fourier / isolated
                _args("sparse_mae", embed="linear", decoder="legacy", absence_tokens=True, queries="neg",
                      neg_ratio=1.0),
                _args("sparse_time", mask="random"), _args("sparse_time", patch_size=43)):
        with pytest.raises(ValueError):
            tr.run_suffix(bad)
            tr.sparse_build_flags(bad, scaling)
    for flags in ({"queries": "all", "absence_tokens": True},
                  {"queries": "all", "embed": "linear", "decoder": "legacy", "absence_patch_size": 173}):
        with pytest.raises(ValueError):
            sa.build_sparse_arm("sparse_mae", sk[0], SEED, encoder_config=_cfg(sk[0]), scaling=scaling, **flags)
    with pytest.raises(ValueError):
        sa.build_sparse_arm("sparse_time", sk[0], SEED, encoder_config=_cfg(sk[0]), scaling=scaling,
                            absence_tokens=True)


def test_leakage_e_c2_cli_differs_from_d0_by_loss_and_placeholder_only(sk):
    """CLI level of (e): the SK-C2 and SK-D0 argument sets differ only in --loss / --placeholder, and
    the only resulting recipe change is the r2_sparse_flags block (plus the run id)."""
    _, scaling = sk
    d0 = _args("mae_repaired")
    c2 = _args("mae_repaired", loss="fixed", placeholder="sentinel")
    diff = {k for k in vars(d0) if vars(d0)[k] != vars(c2)[k]}
    assert diff == {"loss", "placeholder"}
    assert tr.sparse_build_flags(d0, scaling) is None
    assert set(tr.sparse_build_flags(c2, scaling)) == {"loss_mode", "placeholder", "embed", "decoder",
                                                       "count_head", "sentinel_time"}


def test_step_efficiency_median():
    eff = tr.StepEfficiency("cpu", first=3, last=6)
    import time as _t
    for k in range(1, 9):
        eff(k)
        _t.sleep(0.001)
    assert eff.result["n_intervals"] == 3 and eff.result["median_step_seconds"] > 0


# ----------------------------------------------------------------------------- readout adapter (golden)
@pytest.mark.skipif(not dr.pack("sk", dr.LUCID_V1).is_file(), reason="needs the lucid_v1 SK pack")
def test_golden_adapter_sk_50_events():
    """§T1.2 golden test: on SK lucid_v1, a dense model's spatial-moment features through the adapter
    on its hit tokens equal the dense path for 50 events; and the S1A readout (hit tokens scattered)
    equals the adapter on its gathered hit tokens."""
    from jepa_noise.r2_probe_repair import moment_features
    from jepa_noise.sparse_readout import adapter_moment_features
    geometry = dr.geometry("sk", dr.LUCID_V1)
    scaling = rs.scaling_for(geometry, 450.0)
    cohort = rs.open_cohort(dr.pack("sk", dr.LUCID_V1), "dev_probe_test", truth_fields=())
    masker = ViewMasker(geometry, "within", seed=0, n_min=0)
    store = cohort.run_store(0)
    ids = store.event_ids[:50]
    batch = rs.tokenise_sparse(store, geometry, event_ids=ids, scaling=scaling, masker=masker)
    assert len(batch.event_ids) == 50
    raw = batch.features.numpy()
    hit = torch.as_tensor(raw[..., 8] > 0.5)
    idx, valid = sa.gather_tokens(hit, torch.arange(geometry.n_pmt)[None, :].float())
    dense = r2.build_r2_arm("mae_repaired", geometry, 9000, scaling=scaling,
                            encoder_config=_cfg(geometry)).eval()
    s1a = _arm((geometry, scaling), "S1A")
    for arm in (dense, s1a):
        z = []
        with torch.no_grad():
            for k in range(0, 50, 10):
                z.append(arm.encode_tokens(batch.features[k:k + 10]).numpy())
        z = np.concatenate(z)
        zh = np.take_along_axis(z, idx.numpy()[..., None], axis=1)
        lat_a, full_a = adapter_moment_features(zh, idx.numpy(), valid.numpy(), raw)
        lat_d, full_d = moment_features(z, raw)
        np.testing.assert_allclose(lat_a, lat_d, rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(full_a, full_d, rtol=1e-6, atol=1e-7)
