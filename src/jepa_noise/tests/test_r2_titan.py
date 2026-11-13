"""Titan addendum: separate ledger line, JUNO only, 11 GiB micro-batch rule, profile guard."""
import pytest

from jepa_noise import ext_r2 as x


def test_default_profile_unchanged(tmp_path, monkeypatch):
    monkeypatch.delenv("R2_GPU_PROFILE", raising=False)
    assert x.gpu_profile_name() == "l40s" and x.cap_h() == 72.0
    assert x.ledger_path(tmp_path).name == "RESOURCE_LEDGER_R2.json"
    assert x.choose_micro_batch([{"micro_batch": 8, "peak_memory_gib": 19.9}]) == 8
    with pytest.raises(RuntimeError):
        x.choose_micro_batch([{"micro_batch": 8, "peak_memory_gib": 20.1}])


def test_titan_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("R2_GPU_PROFILE", "titan")
    assert x.cap_h() == 160.0
    assert x.ledger_path(tmp_path).name == "RESOURCE_LEDGER_R2_TITAN.json"
    led = x.ledger_load(tmp_path)
    assert led["line"] == "R2_titan_deepthought"
    # only JUNO is owed on the Titan line
    reserved = x.reserved_gpu_h(tmp_path, led)
    expected = 6 * x.DESIGN_MAIN_RUN_H["juno"] + x.DESIGN_RF_RUN_H["juno"] + x.eval_total_h(tmp_path, "juno")
    assert abs(reserved - expected) < 1e-9
    # eval reservation from the Titan WCTE measurement, scaled by tokens, x 12 models
    x.ledger_add(tmp_path, {"kind": "eval", "detector": "wcte", "arm": "init", "seed": 9000,
                            "gpu_h": 0.1, "wall_s": 360.0})
    assert abs(x.eval_total_h(tmp_path, "juno") - 12 * 360.0 * 10000 / 1995 / 3600) < 1e-9
    assert x.ledger_load(tmp_path)["entries"][0]["gpu"].startswith("NVIDIA GeForce GTX TITAN")
    assert not (tmp_path / "ledger" / "RESOURCE_LEDGER_R2.json").exists()
    assert x.choose_micro_batch([{"micro_batch": 4, "peak_memory_gib": 10.9},
                                 {"micro_batch": 8, "peak_memory_gib": 11.1}]) == 4
    with pytest.raises(x.StopError) as err:
        x.gpu_precheck(tmp_path, "juno", "mae_mse", 9000, 161.0)
    assert err.value.stop == "T-S1"


def test_profile_guard(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv("R2_GPU_PROFILE", "titan")
    with pytest.raises(ValueError, match="GPU profile"):
        x.train_r2(detector="sk", arm="mae_mse", seed=9000, h5_path=tmp_path / "x.h5",
                   out_root=tmp_path, device="cpu", smoke=True, steps=x.SMOKE_STEPS)
    monkeypatch.setenv("R2_GPU_PROFILE", "l40s")
    with pytest.raises(ValueError, match="GPU profile"):
        x.train_r2(detector="juno", arm="mae_mse", seed=9000, h5_path=tmp_path / "x.h5",
                   out_root=tmp_path, device="cpu", smoke=True, steps=x.SMOKE_STEPS)
