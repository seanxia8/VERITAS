"""r2-tier1 T1 focused tests (runbook 2026-11-22 §T1.1-T1.7)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_noise import ext_r2 as r2
from jepa_noise import ext_r2_repair_train as rt
from jepa_noise import lucid_views as lv
from jepa_noise.tests.r2_fixtures import small_cylinder
from jepa_noise.tests.test_r2_repair import _synthetic

REPO = Path(__file__).resolve().parents[3]
GOLDEN = Path(__file__).with_name("tier1_mask_golden.json")
LUCID_RUNS = Path("/ceph/dwong/jepa/returns/r2_lucid_2026-11-21/train/runs")


# ------------------------------------------------------------------ T1.1 random split
def test_random_split_registered():
    assert "random" in lv.SPLITS
    assert lv.SPLITS[:3] == ("across", "within", "spatial_patch")


def test_random_mask_deterministic_per_event_and_exact_half():
    g = small_cylinder()
    hit = np.zeros(g.n_pmt, bool)
    a = lv.random_mask(g, 1234, hit, seed=7, n_min=0)
    b = lv.random_mask(g, 1234, hit, seed=7, n_min=0)
    np.testing.assert_array_equal(a.target, b.target)
    assert a.target.size == g.n_pmt // 2
    assert np.intersect1d(a.target, a.context).size == 0
    assert a.target.size + a.context.size == g.n_pmt
    rng = lv.keyed_rng(7, lv.NS_VIEWS, "random", 0, 1234)
    np.testing.assert_array_equal(a.target, np.sort(rng.permutation(g.n_pmt)[:g.n_pmt // 2]))


def test_random_mask_differs_across_events_and_round_trips():
    g = small_cylinder()
    hit = np.zeros(g.n_pmt, bool)
    hit[::3] = True
    masker = lv.ViewMasker(g, "random", seed=11, n_min=0)
    targets = [tuple(masker.mask(e, hit).target) for e in range(50)]
    assert len(set(targets)) == 50
    m = masker.mask(42, hit)
    again = lv.mask_from_key(g, m.key, hit)
    np.testing.assert_array_equal(again.target, m.target)
    assert m.accepted and m.target_hits == int(hit[m.target].sum())
    strict = lv.random_mask(g, 42, hit, seed=11, n_min=10_000)
    assert not strict.accepted


def test_random_split_tokenises_rectangular_batches():
    _, _, _, batch = _synthetic("random", n_events=6)
    assert batch.hidden.shape[0] == 6
    assert int(batch.hidden.sum(1).unique().numel()) == 1


def test_within_and_spatial_patch_unchanged_against_base_golden():
    from jepa_noise import data_roots as dr
    golden = json.loads(GOLDEN.read_text())
    for det in ("wcte", "sk"):
        try:
            g = dr.geometry(det, "lucid_v1")
        except Exception as exc:  # pragma: no cover - host without the LUCiD config
            pytest.skip(f"lucid_v1 geometry unavailable: {exc}")
        assert g.digest == golden[f"geometry_sha256_{det}"]
        for split in ("within", "spatial_patch"):
            m = lv.ViewMasker(g, split, seed=20261017, n_min=0)
            h = hashlib.sha256()
            for e in golden["event_ids"]:
                vm = m.mask(e, np.zeros(g.n_pmt, bool))
                h.update(np.asarray(vm.target, np.int64).tobytes())
                h.update(np.asarray(vm.context, np.int64).tobytes())
            assert h.hexdigest() == golden["masks"][f"{det}/{split}"], (det, split)


# ------------------------------------------------------------------ T1.2-T1.4 flags, names
def _ns(**kw):
    base = dict(detector="sk", arm="mae_repaired", seed=9000, time_loss_scale_ns=None, mask=None,
                charge_loss_weight=None, time_reference=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_run_suffix_order_and_defaults():
    assert rt.run_suffix(_ns()) == ""
    assert rt.run_suffix(_ns(mask="spatial_patch", time_loss_scale_ns=50.0, charge_loss_weight=1.0)) == ""
    assert rt.run_suffix(_ns(mask="random", time_loss_scale_ns=450.0, charge_loss_weight=100.0)) == \
        "__mrandom__ts450__cw100"
    assert rt.run_suffix(_ns(mask="within", time_loss_scale_ns=1.5)) == "__mwithin__ts1p5"
    assert rt.run_suffix(_ns(charge_loss_weight=100.0)) == "__cw100"


@pytest.mark.parametrize("kw", [dict(mask="random"), dict(charge_loss_weight=100.0),
                                dict(time_loss_scale_ns=5.0)])
def test_variant_flags_refused_outside_mae_repaired(kw):
    with pytest.raises(ValueError):
        rt.run_suffix(_ns(arm="jepa_repaired", **kw))


def test_refuses_r2_lucid_returns(tmp_path):
    with pytest.raises(PermissionError):
        rt.refuse_lucid_returns(rt.RETURNS / rt.LUCID_EXTENSION / "train")
    assert rt.refuse_lucid_returns(tmp_path) == tmp_path.resolve()


# ------------------------------------------------------------------ T1.3 charge weight
def test_charge_loss_weight_multiplies_only_the_charge_term():
    geometry, scaling, _, batch = _synthetic("spatial_patch", time_scale=450.0, n_events=4)
    from dataclasses import replace
    geometry = replace(geometry, detector="sk")
    base = r2.build_r2_arm("mae_repaired", geometry, 9000, scaling=scaling)
    heavy = r2.build_r2_arm("mae_repaired", geometry, 9000, scaling=scaling, charge_loss_weight=100.0)
    heavy.load_state_dict(base.state_dict())
    with torch.no_grad():
        m0, m1 = base(batch)["metrics"], heavy(batch)["metrics"]
    assert "charge_loss_weight" not in m0 and "charge_weighted" not in m0
    assert m1["charge"] == pytest.approx(m0["charge"])
    assert m1["charge_weighted"] == pytest.approx(100.0 * m1["charge"], rel=1e-6)
    assert m1["loss"] == pytest.approx(100.0 * m1["charge"] + m1["hit"] + m1["time"], rel=1e-6)
    assert m0["loss"] == pytest.approx(m0["charge"] + m0["hit"] + m0["time"], rel=1e-6)
    with pytest.raises(ValueError):
        r2.build_r2_arm("jepa_repaired", geometry, 9000, scaling=scaling, charge_loss_weight=2.0)


# ------------------------------------------------------------------ T1.6 default-recipe hash, P3
def _prepare(detector, **kw):
    args = argparse.Namespace(detector=detector, arm="mae_repaired", seed=9000, data_kind="lucid_v1",
                              time_loss_scale_ns=None, mask=None, charge_loss_weight=None, data=None,
                              time_reference=None,
                              micro_batch=8, window=None, smoke=None, device="cpu",
                              memory_fraction=None)
    for k, v in kw.items():
        setattr(args, k, v)
    return rt.prepare(args)


@pytest.mark.parametrize("detector", ["sk", "wcte"])
def test_default_recipe_hash_matches_existing_run(detector):
    record = LUCID_RUNS / f"{detector}__mae_repaired__s9000" / "run_record.json"
    if not record.is_file():
        pytest.skip("r2-lucid returns unavailable on this host")
    p = _prepare(detector)
    assert p["config_sha"] == json.loads(record.read_text())["config_sha256"]


@pytest.mark.parametrize("kw,fields", [
    (dict(mask="random"), {"view_mode", "mask"}),
    (dict(mask="within"), {"view_mode", "mask"}),
    (dict(time_loss_scale_ns=450.0), {"time_loss_scale_ns"}),
    (dict(charge_loss_weight=100.0), {"charge_loss_weight"}),
])
def test_one_factor_recipes(kw, fields):
    if not (LUCID_RUNS / "wcte__mae_repaired__s9000" / "run_record.json").is_file():
        pytest.skip("LUCiD data unavailable on this host")
    base = _prepare("wcte")["recipe"]
    var = _prepare("wcte", **kw)["recipe"]
    changed = {k for k in set(base) | set(var) if base.get(k) != var.get(k)}
    assert changed == fields | {"identity"}
    assert {k for k in base["identity"] if base["identity"][k] != var["identity"][k]} == {"run_id"}
    assert var["encoder"] == base["encoder"] and var["shuffle"] is True


# ------------------------------------------------------------------ T1.5 builder (D1, D7)
def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_final(root, rid, det, arm, seed, spatial):
    v = {k: {"mean_m": spatial, "mean_over_r_fid": spatial} for k in (
        "ridge_spatial_moment_latent", "mlp_spatial_moment", "ridge_all_mean", "ridge_hit_mean",
        "ridge_weighted_means", "ridge_raw_spatial_moment")}
    tok = {"token_time_r2": 0.5, "token_log_charge_r2": 0.5}
    d = root / rid
    d.mkdir(parents=True)
    (d / "final.json").write_text(json.dumps({
        "run_id": rid, "detector": det, "arm": arm, "seed": seed, "standard": None,
        "repaired": {"vertex": v, "token": {"post_ln_unmasked_r2": tok, "masked_target_pred_r2": tok}}}))


def test_builder_keys_by_run_id_and_separates_sweeps(tmp_path):
    tvl = _load_script("lucid_toy_vs_lucid")
    _fake_final(tmp_path, "wcte__mae_repaired__s9000", "wcte", "mae_repaired", 9000, 0.0521)
    _fake_final(tmp_path, "wcte__mae_repaired__s9000__ts5", "wcte", "mae_repaired", 9000, 0.0400)
    _fake_final(tmp_path, "wcte__mae_repaired__s9000__mrandom__ts450", "wcte", "mae_repaired", 9000, 0.09)
    main = tvl.lucid_rows(tmp_path)
    assert main[("wcte", "mae_repaired", 9000)]["repaired"]["vertex"][
        "ridge_spatial_moment_latent"]["mean_m"] == 0.0521
    sweep = tvl.lucid_sweep_rows(tmp_path)
    assert set(sweep) == {"wcte__mae_repaired__s9000__ts5", "wcte__mae_repaired__s9000__mrandom__ts450"}


def test_builder_has_no_unqualified_tof_ceiling():
    text = (REPO / "scripts" / "lucid_toy_vs_lucid.py").read_text()
    assert "Approximate ToF-only ceiling" not in text
    assert "not a bound" in text


# ------------------------------------------------------------------ T1.7 evaluation entry point
def test_tier1_eval_sealed_access_requires_spec_and_purpose_and_logs(tmp_path, monkeypatch):
    ev = _load_script("tier1_eval")
    log = tmp_path / "log.md"
    with pytest.raises(ValueError):
        ev.log_sealed_access("x@10000", "", log)
    monkeypatch.setattr(ev, "READOUT_SPEC", tmp_path / "missing.md")
    with pytest.raises(PermissionError):
        ev.log_sealed_access("x@10000", "test", log)
    spec = tmp_path / "spec.md"
    spec.write_text("frozen")
    monkeypatch.setattr(ev, "READOUT_SPEC", spec)
    ev.log_sealed_access("sk__mae_repaired__s9000@10000", "unit test", log)
    ev.log_sealed_access("sk__mae_repaired__s9001@10000", "unit test 2", log)
    lines = log.read_text().splitlines()
    assert lines[0].startswith("# Sealed access log")
    assert sum("unit test" in line for line in lines) == 2


def test_tier1_eval_refuses_lucid_output_root():
    ev = _load_script("tier1_eval")
    with pytest.raises(SystemExit):
        ev.main(["--legacy-init", "--detector", "wcte", "--seed", "9000",
                 "--out-root", str(ev.LUCID_RETURNS / "x")])


def test_tier1_eval_helpers():
    ev = _load_script("tier1_eval")
    p = np.r_[np.linspace(0, 0.5, 100), np.linspace(0.5, 1.0, 100)]
    is_mu = np.r_[np.zeros(100, bool), np.ones(100, bool)]
    mu_mis, e_mis = ev.misid(p, is_mu)
    assert mu_mis <= 0.02 and e_mis <= 0.02
    a = ev.boot_indices(50, "sealed_test/photon")
    b = ev.boot_indices(50, "sealed_test/photon")
    np.testing.assert_array_equal(a, b)
    assert a.shape == (ev.N_BOOT, 50)
    s = ev.energy_summary(np.r_[-0.1, 0.0, 0.1])
    assert s["rms"] == pytest.approx(np.sqrt(0.02 / 3))


def test_attentive_probe_learns_a_simple_target():
    ev = _load_script("tier1_eval")
    rng = np.random.default_rng(0)
    n, d = 300, 16

    def make(n):
        z = rng.normal(size=(n, ev.ATTENTIVE_TOKENS, d)).astype(np.float16)
        r = rng.normal(size=(n, ev.ATTENTIVE_TOKENS, 3)).astype(np.float32)
        cnt = rng.integers(5, 40, n)
        y = np.stack([r[i, :cnt[i]].mean(0) for i in range(n)])
        return {"z": z, "r": r, "n": cnt, "y": y}

    train, val = make(n), make(100)
    predict, info = ev.fit_attentive(train, val, 0, torch.device("cpu"), epochs=30, patience=10)
    err = np.linalg.norm(predict(val) - val["y"], axis=1).mean()
    base = np.linalg.norm(val["y"] - train["y"].mean(0), axis=1).mean()
    assert err < base


# ------------------------------------------------------------------ T5.1 time reference
def test_median_time_reference_centres_hit_times_only():
    from jepa_noise import point_encoder as pe
    from jepa_noise import r2_store as rs
    _, _, _, batch = _synthetic("spatial_patch", n_events=5)
    x = batch.features
    assert rs.apply_time_reference(x, "trigger") is x
    y = rs.apply_time_reference(x, "median")
    hit = x[..., pe.HIT_INDEX] > 0.5
    for row in range(x.shape[0]):
        assert float(y[row, hit[row], pe.TIME_INDEX].median()) == pytest.approx(0.0, abs=1e-6)
        torch.testing.assert_close(y[row, ~hit[row]], x[row, ~hit[row]])
        diff = (x[row, hit[row], pe.TIME_INDEX] - y[row, hit[row], pe.TIME_INDEX]).unique()
        assert float(diff.max() - diff.min()) < 1e-5
    other = [i for i in range(pe.N_FEATURES) if i != pe.TIME_INDEX]
    torch.testing.assert_close(y[..., other], x[..., other])
    with pytest.raises(ValueError):
        rs.apply_time_reference(x, "first")


def test_time_reference_suffix_and_recipe():
    assert rt.run_suffix(_ns(time_reference="median")) == "__trmedian"
    assert rt.run_suffix(_ns(time_reference="trigger")) == ""
    with pytest.raises(ValueError):
        rt.run_suffix(_ns(arm="jepa_repaired", time_reference="median"))
    if not (LUCID_RUNS / "wcte__mae_repaired__s9000" / "run_record.json").is_file():
        pytest.skip("LUCiD data unavailable on this host")
    base = _prepare("wcte", time_reference=None)
    var = _prepare("wcte", time_reference="median")
    changed = {k for k in set(base["recipe"]) | set(var["recipe"])
               if base["recipe"].get(k) != var["recipe"].get(k)}
    assert changed == {"time_reference", "identity"}
    assert var["loader"].time_reference == "median" and base["loader"].time_reference == "trigger"
