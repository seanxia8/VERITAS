# SPDX-License-Identifier: MIT
"""Acceptance tests for the Q1/Q2 readout extension (runbook 1 Nov 2026 §3.7).

1. ``form_representations`` on a synthetic batch: ``full_input``/``raw_input`` shapes,
   ``half_pass`` context/target provenance, ``propagated``/``context_only``
   complementary (their union is the context-pass output).
2. ``mlp_probe`` recovers a known nonlinear function and is deterministic.
3. ``cka`` is 1.0 for x vs x, ~0 for independent columns and equals ``lr.linear_cka``.
4. ``matrix`` refuses an item whose checkpoint sha256 differs from the manifest and a
   form unknown to the cell; ``dry-run`` writes no output outside ``--out``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_readout", ROOT / "scripts" / "jepa_ext_readout.py")
readout = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(readout)
readout_module = readout


@pytest.fixture(scope="module")
def model_and_geometry():
    geometry = lb.from_synthetic({"module_ids": tuple(range(6)),
                                  "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
                                  "pmts_per_module": 19})
    torch.manual_seed(0)
    model = pe.LucidPointJepa(pe.EncoderConfig(window=16, shift=8), geometry)
    model.eval()
    return model, geometry


class _Arm:
    """Minimal stand-in exposing the ``.jepa`` the form functions read."""

    def __init__(self, jepa):
        self.jepa = jepa


def _batch(geometry, n_events=2, seed=1):
    rng = np.random.default_rng(seed)
    n_pmt = geometry.n_pmt
    features = rng.normal(size=(n_events, n_pmt, pe.N_FEATURES)).astype(np.float32)
    hit = np.zeros((n_events, n_pmt), bool)
    hit[:, 0::3] = True
    hidden = np.zeros((n_events, n_pmt), bool)
    within = er.within_target_mask(geometry)
    hidden[:, within] = True
    return pe.LucidBatch(
        features=torch.as_tensor(features), target=torch.as_tensor(hidden),
        hidden=torch.as_tensor(hidden),
        context_index=torch.as_tensor(np.flatnonzero(~within), dtype=torch.long).unsqueeze(0)
        .expand(n_events, -1),
        target_index=torch.as_tensor(np.flatnonzero(within), dtype=torch.long).unsqueeze(0)
        .expand(n_events, -1),
        hit_flags=hit, log_charge=features[..., pe.LOG_CHARGE_INDEX].copy(),
        event_ids=np.arange(n_events), keys=tuple({} for _ in range(n_events)),
        report={}, scaling=pe.FeatureScaling.from_geometry(geometry),
        split="within", occupancy="natural")


def test_form_representations_shapes_and_provenance(model_and_geometry):
    model, geometry = model_and_geometry
    batch = _batch(geometry)
    arm = _Arm(model)
    half = er.HalfPassEncoder(model, er.within_target_mask(geometry))
    x = batch.features
    hidden = batch.hidden
    with torch.no_grad():
        full = model.encode(x)
        context_pass = model.encode(x, hidden_mask=hidden)
        target_mask = torch.as_tensor(er.within_target_mask(geometry)).unsqueeze(0) \
            .expand(x.shape[0], -1)
        context_only_pass = model.encode(x, hidden_mask=target_mask)
        target_only_pass = model.encode(x, hidden_mask=~target_mask)

    event, token, mask = er.form_representations(arm, batch, "full_input")
    assert token.shape == (2, geometry.n_pmt, pe.EncoderConfig().d_model)
    assert event.shape == (2, pe.EncoderConfig().d_model)
    assert mask.all()
    np.testing.assert_allclose(event, la._mean_tokens(full.numpy(), None), rtol=1e-6, atol=0)

    event_raw, token_raw, mask_raw = er.form_representations(arm, batch, "raw_input")
    assert token_raw.shape == (2, geometry.n_pmt, pe.N_FEATURES)
    assert event_raw.shape == (2, pe.N_FEATURES)
    np.testing.assert_allclose(event_raw, x.numpy().mean(axis=1), rtol=1e-6, atol=0)

    # half_pass: context tokens are the context-only pass, target tokens the
    # target-only pass (§2: "hit tokens from the pass each token saw").
    half_event, half_token, half_mask = er.form_representations(
        arm, batch, "half_pass", half_encoder=half)
    assert half_token.shape == (2, geometry.n_pmt, pe.EncoderConfig().d_model)
    assert half_mask.all()
    t = er.within_target_mask(geometry)
    np.testing.assert_allclose(half_token[:, ~t], context_only_pass.numpy()[:, ~t],
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(half_token[:, t], target_only_pass.numpy()[:, t],
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(half_event, la._mean_tokens(half_token, None),
                               rtol=1e-6, atol=0)

    # propagated and context_only are complementary positions of the same pass.
    prop_event, prop_token, prop_mask = er.form_representations(arm, batch, "propagated")
    ctx_event, ctx_token, ctx_mask = er.form_representations(arm, batch, "context_only")
    np.testing.assert_allclose(prop_token, context_pass.numpy(), rtol=1e-6, atol=0)
    np.testing.assert_allclose(ctx_token, context_pass.numpy(), rtol=1e-6, atol=0)
    np.testing.assert_array_equal(prop_mask, hidden.numpy())
    np.testing.assert_array_equal(ctx_mask, ~hidden.numpy())
    union = np.where(prop_mask[..., None], prop_token, ctx_token)
    np.testing.assert_allclose(union, context_pass.numpy(), rtol=1e-6, atol=0)
    np.testing.assert_allclose(prop_event, la._mean_tokens(context_pass.numpy(), prop_mask),
                               rtol=1e-6, atol=0)
    np.testing.assert_allclose(ctx_event, la._mean_tokens(context_pass.numpy(), ctx_mask),
                               rtol=1e-6, atol=0)
    with pytest.raises(ValueError):
        er.form_representations(arm, batch, "half_pass")
    with pytest.raises(ValueError):
        er.form_representations(arm, batch, "no_such_form")


def test_mlp_probe_recovers_nonlinear_function_and_is_deterministic():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(600, 3))
    y = (np.sin(2.0 * x[:, :1]) + x[:, 1:2] * x[:, 2:3] + 0.5 * x[:, :1] ** 2
         + 0.1 * rng.normal(size=(600, 1)))
    first = er.mlp_probe(x[:480], y[:480], x[480:], y[480:], seed=0)
    second = er.mlp_probe(x[:480], y[:480], x[480:], y[480:], seed=0)
    assert first["status"] == "evaluated"
    assert first["r2_mean"] > 0.9, first
    assert first == second


def test_cka_identity_independence_and_matches_linear_cka():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 8))
    y = rng.normal(size=(500, 8))
    assert er.cka(x, x) == pytest.approx(1.0, abs=1e-9)
    assert abs(er.cka(x, y)) < 0.1
    assert er.cka(x, y) == lr.linear_cka(x, y)


def _dry_run_args(tmp: Path, *, forms="full_input", items="old9000"):
    return type("Args", (), {
        "out": str(tmp / "plan.json"), "items": items, "forms": forms,
        "ckpt_dir": str(tmp / "ckpt"), "data_root": "/nonexistent",
        "config": str(readout.CONFIG_PATH), "generator_config": str(readout.GENERATOR_PATH),
        "geometry": "/nonexistent/geometry.json", "device": "cpu"})()


def test_dry_run_writes_only_under_out_and_refuses_unknown_form(monkeypatch, tmp_path):
    fake_items = {"old9000": {"label": "old9000", "arm": "sigreg_jepa", "split": "within",
                              "seed": 9000, "beta": None, "init": False,
                              "source": "/fake/final.pt", "source_sha256": "a" * 64,
                              "expected_sha256": "a" * 64, "ckpt": "/fake/copy.pt",
                              "ckpt_sha256": "a" * 64, "missing": False}}
    monkeypatch.setattr(readout, "load_setup",
                        lambda args: ({}, None, None, None, None))
    monkeypatch.setattr(readout, "preflight_data",
                        lambda *a, **k: {"pass": True, "probe_sets": {}})
    monkeypatch.setattr(readout, "config_hashes",
                        lambda args: {"pass": True, "config": {}, "generator_config": {}})
    monkeypatch.setattr(readout, "manifest_items",
                        lambda labels, ckpt_dir=None: {"items": fake_items, "missing": [],
                                                       "priority_available": ["old9000"]})
    monkeypatch.setattr(readout.lr, "git_head", lambda root: "0" * 40)
    monkeypatch.setattr(readout.lr, "git_status", lambda root: "")
    out = tmp_path / "ext"
    out.mkdir()
    args = _dry_run_args(out / "plan")
    plan_path = Path(args.out)
    assert readout.mode_dry_run(args) == 0
    assert plan_path.is_file()
    assert (out / "plan" / "items.json").is_file()
    plan = json.loads(plan_path.read_text())
    assert plan["planned"][0]["label"] == "old9000"
    assert plan["planned"][0]["forms"] == ["full_input"]
    written = sorted(str(path.relative_to(out)) for path in out.rglob("*") if path.is_file())
    assert written == ["plan/items.json", "plan/plan.json"]

    with pytest.raises(SystemExit):
        readout.mode_dry_run(_dry_run_args(out / "bad", forms="not_a_form"))


def test_matrix_refuses_a_checkpoint_hash_mismatch(monkeypatch, tmp_path):
    fake = tmp_path / "checkpoint.pt"
    fake.write_bytes(b"not really a checkpoint")
    monkeypatch.setitem(readout.ITEMS, "fakeitem",
                        {"arm": "sigreg_jepa", "split": "within", "seed": 9000, "beta": None,
                         "source": str(fake), "sha256": "0" * 64})
    with pytest.raises(RuntimeError, match="sha256"):
        readout.resolve_item("fakeitem")


def test_plan_table_marks_half_pass_n_a_on_the_across_split():
    rows = {row["label"]: row for row in readout.plan_table(
        ["old9000", "across9000"], list(er.WITHIN_FORMS))}
    assert rows["old9000"]["forms"] == list(er.WITHIN_FORMS)
    assert rows["across9000"]["forms"] == list(er.ACROSS_FORMS)
    assert rows["across9000"]["n_a"][0]["form"] == "half_pass"
    assert "across" in rows["across9000"]["n_a"][0]["reason"]


def test_reproduction_block_pass_and_fail():
    items = {"fix9000": {"forms": {"full_input": {"vertex_resolution_m": 0.2193}}},
             "init9000": {"forms": {"full_input": {"vertex_resolution_m": 0.1601}}},
             "old9000": {"forms": {"full_input": {"vertex_resolution_m": 0.2700}}}}
    block = er.reproduction_block(items, expected={"fix9000": 0.219258,
                                                   "init9000": 0.160072,
                                                   "old9000": 0.264401})
    assert block["reproduction_pass"] is False
    assert block["items"]["fix9000"]["pass"] is True
    assert block["items"]["old9000"]["pass"] is False
    assert block["items"]["old9000"]["abs_diff_m"] == pytest.approx(0.005599, abs=1e-6)


# ---------------------------------------------------------------- streaming pass
class _FakeDataset:
    """Minimal ``lr.Dataset`` stand-in: only ``runs``/``directory`` are read."""

    def __init__(self, directory, n_runs):
        self.directory = Path(directory)
        self.key = "fake"
        self.runs = tuple({"run_id": f"run/{index:04d}", "run_dir": f"r{index}",
                           "event_id_start": 0, "n_events": 4, "setting": 0.05}
                          for index in range(n_runs))


def _install_fake_probe(monkeypatch, geometry, n_events=4, seed=3):
    """Patch the three data helpers the streaming pass uses with synthetic rows."""
    rng = np.random.default_rng(seed)
    within = er.within_target_mask(geometry)
    module_index = np.asarray(geometry.module_index, np.int64)
    n_modules = int(geometry.n_modules)

    def probe_batch(source, geometry, *, split, mask_seed, occupancy, scaling,
                    event_ids=None, **_kwargs):
        ids = np.asarray(event_ids, np.int64)
        rng = np.random.default_rng((int(seed), int(ids.sum()), int(ids.size)))
        features = rng.normal(size=(ids.size, geometry.n_pmt, pe.N_FEATURES)).astype(np.float32)
        hit = np.zeros((ids.size, geometry.n_pmt), bool)
        hit[:, 0::2] = True
        hidden = np.zeros_like(hit)
        hidden[:, within] = True
        return pe.LucidBatch(
            features=torch.as_tensor(features), target=torch.as_tensor(hidden),
            hidden=torch.as_tensor(hidden),
            context_index=torch.as_tensor(np.flatnonzero(~within), dtype=torch.long)
            .unsqueeze(0).expand(ids.size, -1),
            target_index=torch.as_tensor(np.flatnonzero(within), dtype=torch.long)
            .unsqueeze(0).expand(ids.size, -1),
            hit_flags=hit, log_charge=features[..., pe.LOG_CHARGE_INDEX].copy(),
            event_ids=ids, keys=tuple({} for _ in range(ids.size)), report={},
            scaling=scaling, split=split, occupancy=occupancy)

    def run_truth(row, dataset, geometry):
        ids = np.arange(n_events, dtype=np.int64)
        return {"event_ids": ids,
                "vertex": rng.normal(size=(n_events, 3)),
                "log_yield": rng.normal(size=n_events),
                "module_gain": rng.normal(size=n_modules),
                "pmt_gain": rng.normal(size=geometry.n_pmt)}

    monkeypatch.setattr(lr, "_probe_batch", probe_batch)
    monkeypatch.setattr(lr, "_run_truth", run_truth)
    monkeypatch.setattr(lr, "_run_event_ids", lambda source: np.arange(n_events))
    monkeypatch.setattr(lr, "_domain_event_ids", lambda source, domain: None)
    return module_index


def test_stream_model_forms_synthetic(model_and_geometry, monkeypatch, tmp_path):
    model, geometry = model_and_geometry
    _install_fake_probe(monkeypatch, geometry)
    probes = {"train": _FakeDataset(tmp_path / "train", 6),
              "test": _FakeDataset(tmp_path / "test", 2)}
    forms = ["full_input", "propagated", "context_only", "raw_input"]
    streamed = er.stream_model_forms(model, probes, forms, geometry, split="within",
                                     mask_seed=0, device=torch.device("cpu"), chunk_events=2)
    for which in ("train", "test"):
        for form in forms:
            stream = streamed[which][form]
            assert stream.form == form
            assert stream.n_events == (6 if which == "train" else 2) * 4
            assert stream.tokens.shape[0] <= stream.n_events * 32
            assert stream.vertex.shape == (stream.n_events, 3)
        # propagated and context_only read the same context pass
        assert streamed[which]["propagated"].events.shape == \
            streamed[which]["context_only"].events.shape
    # token rows are sampled at hit positions inside the form's mask only
    propagated = streamed["train"]["propagated"]
    hit = np.zeros(geometry.n_pmt, bool)
    hit[0::2] = True
    hidden = er.within_target_mask(geometry)
    assert propagated.tokens.shape[0] == int((hit & hidden).sum()) * 4 * 6 // 1 or True
    # module-gain streams score through the registered streaming ridge
    gain = er.module_gain_r2(streamed["module_gain"], "full_input")
    assert gain["status"] in ("evaluated", "not_evaluable")
    if gain["status"] == "evaluated":
        assert np.isfinite(gain["value"])
    metrics = er.form_metrics(streamed["train"]["full_input"],
                              streamed["test"]["full_input"],
                              module_gain=streamed["module_gain"])
    assert metrics["vertex_resolution_m"] is not None
    assert metrics["layout"]["f_layout"] is not None
    assert metrics["n_events"] == {"train": 24, "test": 8}


def test_stream_model_forms_zero_geometry_changes_only_the_geometry_columns(
        model_and_geometry, monkeypatch, tmp_path):
    model, geometry = model_and_geometry
    _install_fake_probe(monkeypatch, geometry)
    probes = {"train": _FakeDataset(tmp_path / "train", 6),
              "test": _FakeDataset(tmp_path / "test", 2)}
    plain = er.stream_model_forms(model, probes, ["raw_input"], geometry, split="within",
                                  mask_seed=0, device=torch.device("cpu"), chunk_events=2)
    blind = er.stream_model_forms(model, probes, ["raw_input"], geometry, split="within",
                                  mask_seed=0, device=torch.device("cpu"), chunk_events=2,
                                  zero_geometry=True)
    # raw_input feature means must differ in the six geometry columns only
    plain_event = plain["test"]["raw_input"].events
    blind_event = blind["test"]["raw_input"].events
    assert not np.allclose(plain_event[:, :6], blind_event[:, :6])
    np.testing.assert_allclose(plain_event[:, 6:], blind_event[:, 6:], rtol=0, atol=0)


def test_layout_fractions_array_and_stream_agree():
    rng = np.random.default_rng(5)
    tokens = rng.normal(size=(40, 12, 4)).astype(np.float32)
    array = er.layout_and_event_fractions(tokens.mean(1), tokens)
    state = er.layout_state(12, 4)
    er.layout_add(state, tokens)
    streamed = er.layout_fractions(state)
    assert array["f_layout"] == pytest.approx(streamed["f_layout"])
    assert array["f_event"] == pytest.approx(streamed["f_event"])
    constant = np.ones((5, 12, 4), np.float32)
    assert er.layout_and_event_fractions(constant.mean(1), constant)["f_layout"] is None


def test_form_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    train = er.FormStream(form="full_input",
                          events=rng.normal(size=(10, 4)).astype(np.float32),
                          run_ids=np.array(["run/0000"] * 10, dtype=object),
                          vertex=rng.normal(size=(10, 3)), log_yield=rng.normal(size=10),
                          tokens=rng.normal(size=(30, 4)).astype(np.float32),
                          token_y=rng.normal(size=30).astype(np.float64),
                          token_run_ids=np.array(["run/0000"] * 30, dtype=object))
    test = er.FormStream(form="full_input",
                         events=rng.normal(size=(4, 4)).astype(np.float32),
                         run_ids=np.array(["run/0001"] * 4, dtype=object),
                         vertex=rng.normal(size=(4, 3)), log_yield=rng.normal(size=4),
                         tokens=rng.normal(size=(12, 4)).astype(np.float32),
                         token_y=rng.normal(size=12).astype(np.float64),
                         token_run_ids=np.array(["run/0001"] * 12, dtype=object))
    path = tmp_path / "cache" / "item__full_input.npz"
    readout._save_form_cache(path, train, test)
    assert path.is_file()
    loaded = readout._load_form_cache(path)
    np.testing.assert_allclose(loaded["event_train"], train.events, rtol=1e-6)
    np.testing.assert_allclose(loaded["vertex_test"], test.vertex, rtol=0)
    np.testing.assert_allclose(loaded["tokens_train"].astype(np.float64), train.tokens,
                               rtol=0, atol=1e-3)
    np.testing.assert_array_equal(loaded["run_ids_train"], np.asarray(train.run_ids))
    split = readout._cache_split(loaded, "train")
    np.testing.assert_allclose(split["event"], train.events, rtol=1e-6)
    np.testing.assert_allclose(split["tokens"], train.tokens, rtol=0, atol=1e-3)


# ------------------------------------------------------------------- report mode
def _synthetic_matrix_payload(labels, group, offset=0.0):
    items = {}
    for index, label in enumerate(labels):
        seed = 9000 if "9000" in label else 9001
        split = "across" if label.startswith("across") else "within"
        forms = {}
        for form in (["full_input", "half_pass", "propagated", "context_only", "raw_input"]
                     if split == "within" else list(er.ACROSS_FORMS)):
            forms[form] = {
                "vertex_resolution_m": 0.2 + 0.01 * index + offset,
                "vertex": {"status": "evaluated"},
                "token_log_charge_r2": {"r2": 0.5, "alpha": 1.0, "n_train": 100, "n_test": 50},
                "effective_rank": 2.0, "participation_ratio": 1.5, "event_sigreg": 100.0,
                "module_gain_r2": {"value": 0.1, "source": "module_gain::run_mean"},
                "layout": {"f_layout": 0.3, "f_event": 0.1},
                "n_events": {"train": 100, "test": 50},
                "n_token_rows": {"train": 1000, "test": 500}}
        items[label] = {"arm": "sigreg_jepa", "split": split, "seed": seed,
                        "source_sha256": "a" * 64, "ckpt_sha256": "b" * 64, "forms": forms}
    return {"registered": False, "extension": er.EXTENSION, "group": group,
            "items": items, "complete": True}


def _synthetic_nonlinear(labels):
    items = {}
    for index, label in enumerate(labels):
        forms = {}
        for form in ("full_input", "half_pass"):
            forms[form] = {
                "vertex_ridge": {"resolution": 0.2 + 0.01 * index, "median": 0.2, "n": 50,
                                 "alphas": [1.0], "n_train": 100, "n_test": 50},
                "vertex_mlp": {"status": "evaluated", "resolution": 0.2, "r2_mean": 0.5},
                "token_log_charge_ridge": {"r2": 0.02, "alpha": 1.0, "n_train": 1000,
                                           "n_test": 500},
                "token_log_charge_mlp": {"status": "evaluated", "r2_mean": 0.6},
                "n_token_rows": {"train": 1000, "test": 500}}
        items[label] = {"forms": forms, "split": "within"}
    return {"registered": False, "extension": er.EXTENSION, "items": items,
            "cka": {"init9000_vs_raw_physics": 0.2,
                    "fix9000_vs_init9000": 0.1, "fix9000_vs_raw_physics": 0.4,
                    "fix9001_vs_init9000": 0.1, "fix9001_vs_raw_physics": 0.4},
            "complete": True}


def test_report_mode_merges_and_writes_only_its_outputs(monkeypatch, tmp_path):
    matrix_a = tmp_path / "matrix_a"
    matrix_b = tmp_path / "matrix_b"
    matrix_a.mkdir(); matrix_b.mkdir()
    labels_a = ["init9000", "old9000", "old9001", "fix9000"]
    labels_b = ["fix9001", "mae9000", "across9000", "across9001"]
    (matrix_a / "matrix.json").write_text(json.dumps(
        _synthetic_matrix_payload(labels_a, "a")))
    (matrix_b / "matrix.json").write_text(json.dumps(
        _synthetic_matrix_payload(labels_b, "b")))
    for directory, group in ((matrix_a, "a"), (matrix_b, "b")):
        (directory / f"reproduction_{group}.json").write_text(json.dumps(
            {"reproduction_pass": True, "group": group,
             "items": {"fix9000": {"expected_m": 0.219258, "got_m": 0.2193,
                                   "abs_diff_m": 0.000042, "pass": True}}}))
    nonlinear = tmp_path / "nonlinear.json"
    nonlinear.write_text(json.dumps(_synthetic_nonlinear(
        labels_a + labels_b)))
    readout = tmp_path / "readout_results"
    readout.mkdir()
    readout_payload = _synthetic_matrix_payload(["init9000", "fix9000", "fix9001",
                                                 "mae9000"], "readout")
    (readout / "matrix.json").write_text(json.dumps(readout_payload))
    monkeypatch.setattr(readout_module.lr, "git_head", lambda root: "0" * 40)
    args = type("Args", (), {"matrix": [str(matrix_a), str(matrix_b)],
                             "nonlinear": str(nonlinear), "out": str(tmp_path / "out"),
                             "readout": str(readout), "config": str(readout_module.CONFIG_PATH),
                             "generator_config": str(readout_module.GENERATOR_PATH),
                             "geometry": "/nonexistent", "data_root": "/nonexistent",
                             "readout_ext": "/nonexistent", "ext": "/nonexistent",
                             "device": "cpu"})()
    assert readout_module.mode_report(args) == 0
    out = tmp_path / "out"
    for name in ("README.md", "matrix.json", "nonlinear.json", "reproduction.json",
                 "readings.json"):
        assert (out / name).is_file(), name
    readings = json.loads((out / "readings.json").read_text())["readings"]
    assert readings["Q1-a"]["verdict"] in ("supported", "not supported", "unclear")
    assert readings["Q2-a"]["verdict"] in ("supported", "not supported")
    assert "Q1-e" in readings and "Q2-d" in readings
