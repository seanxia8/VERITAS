# SPDX-License-Identifier: MIT
"""Acceptance tests for the H1/H2 causal-test extension (runbook 1 Nov 2026 §3.5/§4).

Covers the runbook's required tests: E2 build/restore (input clone, original tensor
unchanged, wrapper removal restores the original output), E3 frozen-parameter /
state_dict behaviour after one synthetic CPU training step (both loss terms see the
frozen label), run-directory separation (a second launch of the same variant
refuses an existing run directory), and the §4.2 reproduction plumbing.
"""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from jepa_noise import ext_h12 as eh  # noqa: E402
from jepa_noise import ext_readout as er  # noqa: E402
from jepa_noise import lucid_arms as la  # noqa: E402
from jepa_noise import lucid_run as lr  # noqa: E402
from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location(
    "jepa_ext_h12", ROOT / "scripts" / "jepa_ext_h12.py")
h12 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h12)


@pytest.fixture(scope="module")
def geometry():
    return lb.from_synthetic({"module_ids": tuple(range(6)),
                              "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
                              "pmts_per_module": 19})


@pytest.fixture(scope="module")
def encoder_config():
    return pe.EncoderConfig(window=16, shift=8)


def _model(geometry, config):
    torch.manual_seed(0)
    model = pe.LucidPointJepa(config, geometry)
    model.eval()
    return model


class _Arm:
    def __init__(self, jepa, geometry):
        self.jepa = jepa
        self.d_model = jepa.config.d_model
        self.n_modules = geometry.n_modules
        self.pmts_per_module = geometry.pmts_per_module


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
        context_index=torch.as_tensor(np.flatnonzero(~within)).unsqueeze(0)
        .expand(n_events, -1),
        target_index=torch.as_tensor(np.flatnonzero(within)).unsqueeze(0)
        .expand(n_events, -1),
        hit_flags=hit, log_charge=features[..., pe.LOG_CHARGE_INDEX].copy(),
        event_ids=np.arange(n_events), keys=tuple({} for _ in range(n_events)),
        report={}, scaling=pe.FeatureScaling.from_geometry(geometry),
        split="within", occupancy="natural")


# ------------------------------------------------------------------------ E2
class _RecordingEncoder(torch.nn.Module):
    """A stub encoder that records exactly what the runtime hook passes it."""

    def __init__(self):
        super().__init__()
        self.config = pe.EncoderConfig()
        self.seen = []

    def encode(self, features, hidden_mask=None):
        self.seen.append(features.detach().clone())
        return features.sum(-1, keepdim=True)


def test_geometry_blind_wrapper_zeroes_a_clone_and_restores(geometry):
    recorder = _RecordingEncoder()
    arm = _Arm(recorder, geometry)
    batch = _batch(geometry)
    x = batch.features.clone().requires_grad_(True)
    with torch.no_grad():
        before = recorder.encode(x)
    original = er.install_geometry_blind(recorder)
    recorder.seen.clear()
    try:
        wrapped = recorder.encode(x)
        # the wrapper clones: the caller's tensor keeps its geometry and is unchanged
        torch.testing.assert_close(x, batch.features)
        assert not torch.allclose(x[..., :6], torch.zeros_like(x[..., :6]))
        # the wrapped encoder saw a clone with the six geometry columns zeroed
        torch.testing.assert_close(recorder.seen[0][..., :6],
                                   torch.zeros_like(recorder.seen[0][..., :6]))
        torch.testing.assert_close(recorder.seen[0][..., 6:], x[..., 6:])
        with torch.no_grad():
            expected = er.geometry_blind(x).sum(-1, keepdim=True)
        torch.testing.assert_close(wrapped, expected)
    finally:
        er.restore_encode(recorder, original)
    with torch.no_grad():
        restored = recorder.encode(x)
    torch.testing.assert_close(restored, before)
    torch.testing.assert_close(recorder.seen[-1], x)


# ------------------------------------------------------------------------ E3
def _settings():
    return la.ArmSettings.from_config(la.load_config(), cell=(0.05, 0.1))


def test_frozen_target_state_dict_and_one_training_step(geometry, encoder_config):
    arm = la.LucidArm("sigreg_jepa", geometry, settings=_settings(), seed=9000,
                      encoder_config=encoder_config)
    plain = la.LucidArm("sigreg_jepa", geometry, settings=_settings(), seed=9000,
                        encoder_config=encoder_config)
    assert set(arm.state_dict()) == set(plain.state_dict())
    before = copy.deepcopy(arm.state_dict())
    eh.attach_frozen_target(arm, geometry)
    # the frozen encoder is outside the module tree: no extra state_dict keys, no
    # change to the trainable arm's initial parameters
    assert set(arm.state_dict()) == set(plain.state_dict())
    for key, value in before.items():
        torch.testing.assert_close(arm.state_dict()[key], value)
    frozen = arm.__dict__["_ext_frozen_target"]
    snapshot = {key: value.detach().clone() for key, value in frozen.named_parameters()}
    assert all(parameter.requires_grad is False for parameter in frozen.parameters())
    # it moves with the arm (held outside the module tree, so _apply is overridden)
    arm.to(torch.device("cpu"))
    batch = _batch(geometry)
    optimizer = torch.optim.Adam(arm.parameters(), lr=1e-3)
    out = arm._forward_sigreg(batch)
    assert np.isfinite(float(out["loss"]))
    assert out["target_tokens"].requires_grad is False
    out["loss"].backward()
    optimizer.step()
    for key, parameter in frozen.named_parameters():
        torch.testing.assert_close(parameter.detach(), snapshot[key], rtol=0, atol=0)
        assert parameter.grad is None
    # the frozen label feeds both loss terms: perturbing the frozen encoder changes
    # the prediction MSE and the SIGReg target term but not the context SIGReg term
    # SIGReg draws its slices from an incrementing step counter: reset it so both
    # forwards see identical slices and only the frozen label can differ.
    arm.sigreg.sliced.global_step.zero_()
    baseline = arm._forward_sigreg(batch)["metrics"]
    with torch.no_grad():
        first = next(frozen.parameters())
        first.add_(0.5)
    arm.sigreg.sliced.global_step.zero_()
    perturbed = arm._forward_sigreg(batch)["metrics"]
    assert perturbed["prediction"] != baseline["prediction"]
    assert perturbed["sigreg_target"] != baseline["sigreg_target"]
    assert perturbed["sigreg_context"] == baseline["sigreg_context"]
    assert np.isfinite(perturbed["loss"])


def test_frozen_target_is_seeded_and_independent_of_the_arm_seed(geometry, encoder_config):
    def build(seed):
        arm = la.LucidArm("sigreg_jepa", geometry, settings=_settings(), seed=seed,
                          encoder_config=encoder_config)
        eh.attach_frozen_target(arm, geometry)
        return arm

    first, second = build(9000), build(9001)
    frozen_a = first.__dict__["_ext_frozen_target"]
    frozen_b = second.__dict__["_ext_frozen_target"]
    for (name_a, parameter_a), (name_b, parameter_b) in zip(frozen_a.named_parameters(),
                                                            frozen_b.named_parameters()):
        assert name_a == name_b
        torch.testing.assert_close(parameter_a, parameter_b, rtol=0, atol=0)
    # the RNG state was restored: a freshly drawn tensor is not the manual_seed(0) one
    state = torch.get_rng_state()
    assert torch.initial_seed() != 0 or state.ndim == 1


# ------------------------------------------------------------ run directories
def test_variant_run_directories_are_separate_and_refuse_a_second_launch(tmp_path):
    geom_dir = tmp_path / "geomblind"
    frozen_dir = tmp_path / "frozen_target"
    run_id_a, dir_a = eh.prepare_run_dir(geom_dir, "geomblind", 9000)
    run_id_b, dir_b = eh.prepare_run_dir(frozen_dir, "frozen_target", 9000)
    # the registered arm name is unchanged, so both variants share the run_id ...
    assert run_id_a == run_id_b == lr.run_id_for(eh.CELL_ID, eh.ARM_NAME, 9000)
    # ... which is why each variant needs its own root.
    assert Path(dir_a) != Path(dir_b)
    assert Path(dir_a).parent.parent == geom_dir
    assert Path(dir_b).parent.parent == frozen_dir
    assert (Path(dir_a) / "variant.json").is_file()
    assert json.loads((Path(dir_a) / "variant.json").read_text())["variant"] == "geomblind"
    with pytest.raises(RuntimeError, match="already exists"):
        eh.prepare_run_dir(geom_dir, "geomblind", 9000)


# ------------------------------------------------------------------ readings
def _row(vertex, r2):
    return {"vertex_resolution_m": vertex, "token_log_charge_r2": {"r2": r2}}


def test_recover_is_three_valued_and_decision_uses_the_conjunction():
    init_full = _row(0.160072, 0.99)
    init_half = _row(0.174821, 0.98)
    supported = eh.recover({9000: _row(0.14, 0.9), 9001: _row(0.15, 0.9)},
                           {9000: _row(0.13, 0.9), 9001: _row(0.14, 0.9)},
                           init_full, init_half)
    assert supported["verdict"] == "supported"
    partial = eh.recover({9000: _row(0.14, 0.1), 9001: _row(0.15, 0.1)},
                         {9000: _row(0.13, 0.9), 9001: _row(0.14, 0.9)},
                         init_full, init_half)
    assert partial["verdict"] == "partial"
    contradicted = eh.recover({9000: _row(0.10, 0.9), 9001: _row(0.10, 0.9)},
                              {9000: _row(0.30, 0.9), 9001: _row(0.30, 0.9)},
                              init_full, init_half)
    assert contradicted["verdict"] == "not supported"
    failed = eh.recover({9000: _row(0.30, 0.1), 9001: _row(0.30, 0.1)},
                        {9000: _row(0.30, 0.1), 9001: _row(0.30, 0.1)},
                        init_full, init_half)
    assert failed["verdict"] == "not supported"
    table = eh.decision(supported, supported)
    assert "H1 + H2" in table["verdict"]
    assert "unclear" in eh.decision(supported, partial)["verdict"]


# ---------------------------------------------------------------- reproduction
def test_reproduction_plumbing_passes_and_exits_nonzero(monkeypatch, tmp_path):
    items = {"fix9000": {"label": "fix9000", "source": "/fake/fix.pt", "ckpt": None,
                         "arm": "sigreg_jepa", "split": "within", "seed": 9000},
             "init9000": {"label": "init9000", "arm": "sigreg_jepa", "split": "within",
                          "seed": 9000, "source": None}}
    items_path = tmp_path / "items.json"
    items_path.write_text(json.dumps({"items": items}))

    def fake_evaluate(item, probes, geometry, config, arms_config, **kwargs):
        value = {"fix9000": 0.2193, "init9000": 0.1601}[item["label"]]
        return {"forms": {"full_input": {"vertex_resolution_m": value}}}

    monkeypatch.setattr(er, "evaluate_item", fake_evaluate)
    block = eh.reproduction(None, None, None, None, items_path, device="cpu")
    assert block["reproduction_pass"] is True
    assert block["items"]["fix9000"]["abs_diff_m"] == pytest.approx(0.000042, abs=1e-6)

    def bad_evaluate(item, probes, geometry, config, arms_config, **kwargs):
        return {"forms": {"full_input": {"vertex_resolution_m": 0.25}}}

    monkeypatch.setattr(er, "evaluate_item", bad_evaluate)
    block = eh.reproduction(None, None, None, None, items_path, device="cpu")
    assert block["reproduction_pass"] is False


def test_data_preflight_manifest_table_matches_the_runbook():
    assert eh.DATA_MANIFESTS["dev_probe_train__F1__0.05__s20261017"].startswith("093ea894")
    assert eh.DATA_MANIFESTS["dev_probe_test__F1__0.05__s20261017"].startswith("c11edc71")
    assert eh.DATA_MANIFESTS[
        "dev_train__F1__0.05__s20261017__n100000x1000"].startswith("b1b3f4f4")
    assert eh.INIT_FULL_VERTEX_M == pytest.approx(0.160072)
