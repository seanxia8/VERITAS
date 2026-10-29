# SPDX-License-Identifier: MIT
"""Tests for scripts/jepa_pregrid_checks.py (extension, registered: false)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("pregrid", ROOT / "scripts" / "jepa_pregrid_checks.py")
pg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pg)

from jepa_noise import lucid_bombs as lb  # noqa: E402
from jepa_noise import point_encoder as pe  # noqa: E402


@pytest.fixture(scope="module")
def model_and_geometry():
    geometry = lb.from_synthetic({"module_ids": tuple(range(6)),
                                  "surface_counts": {"top": 24, "barrel": 66, "bottom": 24},
                                  "pmts_per_module": 19})
    torch.manual_seed(0)
    model = pe.LucidPointJepa(pe.EncoderConfig(window=16, shift=8), geometry)
    model.eval()
    return model, geometry


def test_half_pass_matches_the_two_training_passes(model_and_geometry):
    model, geometry = model_and_geometry
    target = pg.within_target_mask(geometry)
    x = torch.randn(2, geometry.n_pmt, pe.N_FEATURES)
    half = pg.HalfPassEncoder(model, target)
    t = torch.as_tensor(target).unsqueeze(0).expand(2, -1)
    with torch.no_grad():
        out = half.encode(x)
        assert torch.allclose(out[t], model.encode(x, hidden_mask=~t)[t], atol=1e-6)
        assert torch.allclose(out[~t], model.encode(x, hidden_mask=t)[~t], atol=1e-6)
        # an explicit mask passes straight through
        assert torch.allclose(half.encode(x, hidden_mask=t), model.encode(x, hidden_mask=t))


def test_half_pass_target_tokens_ignore_context_inputs(model_and_geometry):
    model, geometry = model_and_geometry
    target = torch.as_tensor(pg.within_target_mask(geometry))
    x = torch.randn(1, geometry.n_pmt, pe.N_FEATURES, requires_grad=True)
    out = pg.HalfPassEncoder(model, target.numpy()).encode(x)
    out[0, target].sum().backward()
    assert float(x.grad[0, ~target].abs().max()) == 0.0
    assert float(x.grad[0, target].abs().max()) > 0.0


class _FakeArm:
    """Minimal stand-in exposing the attributes patch_stopgrad uses."""

    def __init__(self):
        self.w = torch.nn.Parameter(torch.tensor(2.0))
        self.coefficient_lambda = 0.02

    def _encode_views(self, batch):
        context = self.w * batch.c
        labels = self.w * batch.t
        prediction = context.mean(1, keepdim=True).expand_as(labels)
        return context, prediction, labels, None, None

    @staticmethod
    def _context_pooled(context, index):
        return context.mean(1)

    def sigreg(self, pooled, labels):
        v = (pooled ** 2).mean() * 0 + (labels ** 2).mean() * 0
        return {"value": v, "context": v, "target": v}


def test_stopgrad_cuts_label_gradient_in_prediction_loss():
    class B:  # noqa: D401 - tiny batch
        c = torch.ones(2, 3, 4); t = 3 * torch.ones(2, 3, 4); context_index = None
    arm = pg.patch_stopgrad(_FakeArm())
    out = arm._forward_sigreg(B)
    out["loss"].backward()
    # d/dw of (w*1 - stopgrad(w*3))^2 at w=2 is 2*(2-6)*1 = -8 per element (mean over elements)
    assert float(arm.w.grad) == pytest.approx(-8.0)
    assert out["metrics"]["label_stopgrad"] is True


def test_lambda_patch_changes_real_arm_loss_without_mutating_settings(model_and_geometry, monkeypatch):
    _, geometry = model_and_geometry
    arm = pg.la.build_arm("sigreg_jepa", geometry, settings=pg.la.ArmSettings.from_config(pg.la.load_config()), seed=0)
    original = arm.settings
    assert arm.coefficient_lambda == pytest.approx(0.02)
    assert pg.patch_lambda(arm, 0.2) is arm
    assert original.coefficient_lambda == pytest.approx(0.02)
    assert arm.coefficient_lambda == pytest.approx(0.2)

    context = torch.ones(1, 2, 3)
    prediction = torch.ones(1, 2, 3)
    labels = torch.zeros(1, 2, 3)
    monkeypatch.setattr(arm, "_encode_views", lambda batch: (context, prediction, labels, None, None))
    monkeypatch.setattr(arm, "_context_pooled", lambda value, index: value.mean(1))
    class ConstantSigreg(torch.nn.Module):
        def forward(self, pooled, target):
            value = torch.tensor(5.0)
            return {"value": value, "context": value, "target": value}
    arm.sigreg = ConstantSigreg()
    from types import SimpleNamespace
    result = arm._forward_sigreg(SimpleNamespace(context_index=None))
    assert result["loss"].item() == pytest.approx(2.0)  # 1 + 0.2 * 5


def test_statistics_helpers():
    rng = np.random.default_rng(0)
    g = rng.normal(size=(5000, 16))
    assert pg.epps_pulley(g) < 3.0
    assert pg.epps_pulley(np.zeros((5000, 16))) > 1000
    er = pg.effective_rank(g)
    assert 15 < er["effective_rank"] <= 16 and 14 < er["participation_ratio"] <= 16
    assert pg.effective_rank(np.c_[g[:, :1], np.zeros((5000, 15))])["effective_rank"] == pytest.approx(1, abs=1e-6)
    x = rng.normal(size=(4000, 5)); y = x @ np.arange(1, 6) + 0.1 * rng.normal(size=4000)
    assert pg.ridge_r2(x[:3000], y[:3000], x[3000:], y[3000:])["r2"] > 0.99


def test_reselect_beta_uses_the_new_plain(tmp_path):
    table = [{"beta": 0.1, "resolution": 0.232849}, {"beta": 1, "resolution": 0.235181},
             {"beta": 10, "resolution": 0.224621}, {"beta": 100, "resolution": 0.243416},
             {"beta": 1000, "resolution": 0.291769}]
    src = tmp_path / "step3.json"
    src.write_text(json.dumps({"plain_resolution": 0.2644, "selections": {
        "sigreg_jepa_anchored": {"table": table},
        "anchor_only": {"table": [{"beta": 1000, "resolution": 0.319391}]}}}))
    out = tmp_path / "re.json"
    pg.main(["reselect-beta", "--decision", str(src), "--plain-resolution", "0.219259",
             "--plain-source", "fixed_s9000", "--out", str(out)])
    sel = json.loads(out.read_text())["selections"]
    assert sel["sigreg_jepa_anchored"]["selected"] == 10.0   # 0.2434 > 1.1 * 0.2193 = 0.2412
    assert sel["anchor_only"]["status"] == "none_qualified"


def test_event_statistics_stream_without_changing_sample_order(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(pg.pe.FeatureScaling, "from_geometry", lambda geometry: None)
    monkeypatch.setattr(pg.lr, "_domain_event_ids", lambda source, domain: np.arange(13))
    calls = []

    def batch(source, geometry, **kwargs):
        ids = kwargs["event_ids"]
        calls.append(len(ids))
        return SimpleNamespace(features=torch.as_tensor(np.tile(ids[:, None, None], (1, 3, 2)),
                                                         dtype=torch.float32),
                               hit_flags=np.ones((len(ids), 3), bool),
                               log_charge=np.tile(ids[:, None], (1, 3)))

    monkeypatch.setattr(pg.lr, "_probe_batch", batch)

    class Model:
        def eval(self):
            return self

        def encode(self, features):
            return features

    ds = SimpleNamespace(directory=tmp_path, runs=[{"run_dir": "a"}, {"run_dir": "b"}])
    kwargs = dict(split="within", mask_seed=0, device=torch.device("cpu"),
                  max_events=11, probe_max_events=7, tokens_per_event=2)
    streamed = pg.event_level(Model(), ds, None, chunk_events=3, **kwargs)
    assert max(calls) <= 3
    assert streamed[0][:, 0].tolist() == list(range(7)) + list(range(4))
    whole = pg.event_level(Model(), ds, None, chunk_events=20, **kwargs)
    for left, right in zip(streamed, whole):
        np.testing.assert_array_equal(left, right)


def test_strong_drift_generation_is_always_unregistered(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(pg.lr, "ensure_dataset", lambda **kw: calls.append(kw) or object())
    config = {"data": {"dataset_seed": 4, "cohorts": {
        "dev_train": {}, "dev_probe_train": {}, "dev_probe_test": {}}}}
    for setting in (0.05, pg.STRONG_DRIFT_S):
        calls.clear()
        pg.datasets_for(SimpleCell(setting), config, None, None, tmp_path)
        assert calls[0]["registered"] is False
        assert [c["registered"] for c in calls[1:]] == [setting != pg.STRONG_DRIFT_S] * 2


class SimpleCell:
    train_cohort = "dev_train"

    def __init__(self, setting):
        self.setting = setting


# --------------------------------------------------------------------------- 28 Oct: anc100 beta
def _ckpt(tmp_path, arm, beta, name="final.pt"):
    path = tmp_path / name
    torch.save({"identity": {"arm": arm, "beta": beta}, "model": {"w": torch.zeros(1)}}, path)
    return path


class _Recorder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.ones(1))


def _patch_build(monkeypatch):
    calls = []

    def fake_build_arm(arm, geometry, arms_config, cell, seed, beta=None, dark_bank=None,
                       encoder_config=None):
        calls.append({"arm": arm, "beta": beta, "dark_bank": dark_bank})
        return _Recorder()

    monkeypatch.setattr(pg.lr, "build_arm", fake_build_arm)
    monkeypatch.setattr(pg.lr, "ensure_dark_bank", lambda **kw: Path("/nonexistent"))
    monkeypatch.setattr(pg.la, "dark_bank_from_store", lambda path, geometry: "DARK")
    return calls


_CFG = {"data": {"dataset_seed": 1, "dark_noise": {"records": 2}}}


def test_anchored_checkpoint_is_rebuilt_with_its_own_beta(tmp_path, monkeypatch):
    """Regression for the 26 Oct Part C queue: anc100 failed with 'requires beta'."""
    calls = _patch_build(monkeypatch)
    path = _ckpt(tmp_path, "sigreg_jepa_anchored", 100.0)
    arm, beta = pg.build_loaded("sigreg_jepa_anchored", None, None, {}, _CFG, {}, tmp_path,
                                str(path), 0)
    assert beta == 100.0 and calls[-1]["beta"] == 100.0 and calls[-1]["dark_bank"] == "DARK"
    assert float(arm.w) == 0.0  # the checkpoint weights were loaded


def test_plain_arm_still_gets_no_beta(tmp_path, monkeypatch):
    calls = _patch_build(monkeypatch)
    path = _ckpt(tmp_path, "sigreg_jepa", None)
    _, beta = pg.build_loaded("sigreg_jepa", None, None, {}, _CFG, {}, tmp_path, str(path), 0)
    assert beta is None and calls[-1]["beta"] is None and calls[-1]["dark_bank"] is None
    _, beta = pg.build_loaded("sigreg_jepa", None, None, {}, _CFG, {}, tmp_path, "init", 9000)
    assert beta is None and calls[-1]["beta"] is None


def test_beta_arm_refuses_missing_beta_mismatched_arm_and_init(tmp_path, monkeypatch):
    _patch_build(monkeypatch)
    with pytest.raises(ValueError, match="no beta"):
        pg.build_loaded("sigreg_jepa_anchored", None, None, {}, _CFG, {}, tmp_path,
                        str(_ckpt(tmp_path, "sigreg_jepa_anchored", None, "a.pt")), 0)
    with pytest.raises(ValueError, match="does not match"):
        pg.build_loaded("sigreg_jepa_anchored", None, None, {}, _CFG, {}, tmp_path,
                        str(_ckpt(tmp_path, "sigreg_jepa", None, "b.pt")), 0)
    with pytest.raises(ValueError, match="random init"):
        pg.build_loaded("sigreg_jepa_anchored", None, None, {}, _CFG, {}, tmp_path, "init", 0)


def test_report_is_written_after_each_item(tmp_path):
    report = {"registered": False, "items": {"a": {}}}
    pg._write_report(tmp_path, report, complete=False)
    assert json.loads((tmp_path / "inspect.json").read_text())["complete"] is False
    pg._write_report(tmp_path, report, complete=True)
    assert json.loads((tmp_path / "inspect.json").read_text())["complete"] is True
    assert not (tmp_path / "inspect.json.tmp").exists()
