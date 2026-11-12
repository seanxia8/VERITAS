# SPDX-License-Identifier: MIT
"""Round-2 budget logic: test 9 (seed-9002 triggers, skips, HK fallback, init seeds)."""
from __future__ import annotations

import pytest

from jepa_noise import ext_r2 as x


def _readings(a="event-invariant", b="recovered", c="yes"):
    return {"R2-a": {"class": a}, "R2-b": {"class": b}, "R2-c": {"class": c}}


SEPARATED = {"mae_mse": {9000: 0.10, 9001: 0.11}, "target_posfree": {9000: 0.13, 9001: 0.14}}


def test_no_trigger():
    assert x.seed9002_triggers(_readings(), SEPARATED) == []


@pytest.mark.parametrize("kw,arm,why", [
    ({"b": "unclear"}, "target_posfree", "R2-b unclear"),
    ({"c": "unclear"}, "mae_mse", "R2-c unclear"),
    ({"a": "intermediate"}, "sigreg_jepa", "R2-a intermediate")])
def test_each_reading_trigger(kw, arm, why):
    t = x.seed9002_triggers(_readings(**kw), SEPARATED)
    assert t == [{"arm": arm, "seed": 9002, "triggers": [why]}]


def test_overlap_trigger_and_merge():
    overlap = {"mae_mse": {9000: 0.10, 9001: 0.13}, "target_posfree": {9000: 0.12, 9001: 0.14}}
    t = x.seed9002_triggers(_readings(c="unclear"), overlap)
    assert [r["arm"] for r in t] == ["target_posfree", "mae_mse"]
    assert t[1]["triggers"] == ["R2-c unclear", "mae_mse / target_posfree vertex seed ranges overlap"]
    touching = {"mae_mse": {9000: 0.10, 9001: 0.12}, "target_posfree": {9000: 0.12, 9001: 0.14}}
    assert len(x.seed9002_triggers(_readings(), touching)) == 2


def test_skip_when_budget_too_small():
    t = x.seed9002_triggers(_readings(a="intermediate", b="unclear", c="unclear"), SEPARATED)
    cost = {"sigreg_jepa": 2.5, "target_posfree": 2.5, "mae_mse": 2.0}
    plan = x.plan_seed9002(t, remaining_h=5.0, train_h=cost, eval_h=0.2)
    # order of MAIN_ARMS: sigreg_jepa (2.7 h), target_posfree (2.7 h), mae_mse (2.2 h); a
    # skipped run does not block a cheaper later one
    assert [p["arm"] for p in plan] == ["sigreg_jepa", "target_posfree", "mae_mse"]
    assert [p["decision"] for p in plan] == ["train", "skip", "train"]
    assert "remaining cap" in plan[1]["reason"]
    plan = x.plan_seed9002(t, remaining_h=5.0, train_h={**cost, "mae_mse": 2.5}, eval_h=0.2)
    assert [p["decision"] for p in plan] == ["train", "skip", "skip"]
    assert [p["decision"] for p in x.plan_seed9002(t, 1.0, cost, 0.2)] == ["skip"] * 3


def test_hk_fallback_order():
    assert x.hk_fallback(40.0, 4.0, 5.0, 2.0)["step"] == 0
    assert x.hk_fallback(27.0, 4.0, 5.0, 2.0)["step"] == 1
    assert x.hk_fallback(15.0, 4.0, 5.0, 2.0)["step"] == 2
    s = x.hk_fallback(10.0, 4.0, 5.0, 2.0)
    assert s["step"] == 3 and s["stop"] == "S1"


def test_readings_rules():
    assert x.reading_r2a({9000: 5e-4, 9001: 1e-3})["class"] == "event-invariant"
    assert x.reading_r2a({9000: 0.02, 9001: 0.01})["class"] == "event-dependent"
    assert x.reading_r2a({9000: 0.02, 9001: 0.005})["class"] == "intermediate"
    init = {9000: {"vertex": 0.20}, 9001: {"vertex": 0.21}}
    good = {s: {"vertex": 0.1, "token_r2": 0.6, "f_event": 0.05} for s in (9000, 9001)}
    assert x.reading_r2b(good, init)["class"] == "recovered"
    bad = {s: {"vertex": 0.3, "token_r2": 0.6, "f_event": 0.05} for s in (9000, 9001)}
    assert x.reading_r2b(bad, init)["class"] == "not recovered"
    mixed = {9000: good[9000], 9001: bad[9001]}
    assert x.reading_r2b(mixed, init)["class"] == "unclear"
    yes = {s: {"mean_difference_m": -0.05, "ci95_m": [-0.06, -0.04]} for s in (9000, 9001)}
    assert x.reading_r2c(yes)["class"] == "yes"
    no = {s: {"mean_difference_m": 0.05, "ci95_m": [0.04, 0.06]} for s in (9000, 9001)}
    assert x.reading_r2c(no)["class"] == "no"
    assert x.reading_r2c({9000: yes[9000], 9001: no[9001]})["class"] == "unclear"
    d = x.reading_r2d({"mae_mse": {9000: .1, 9001: .1}, "target_posfree": {9000: .2, 9001: .2},
                       "init": {9000: .3, 9001: .3}, "sigreg_jepa": {9000: .4, 9001: .4}})
    assert d["kendall_tau"] == 1.0
    assert x.kendall_tau(list(reversed(x.WCTE_ORDER)), x.WCTE_ORDER) == -1.0
    e = x.reading_r2e({"mean_difference_m": 0.02, "ci95_m": [0.01, 0.03]},
                      {"token_r2": 0.5, "f_event": 0.1, "particle": {"mue_auc": 0.9}},
                      {"token_r2": 0.6, "f_event": 0.1, "particle": {"mue_auc": 0.8}})
    assert e["confounded_by_receptive_field"] and abs(e["token_r2_difference"] + 0.1) < 1e-12
    e2 = x.reading_r2e({"mean_difference_m": 0.01, "ci95_m": [0.005, 0.02]},
                       {"token_r2": 0.5, "f_event": 0.1}, {"token_r2": 0.5, "f_event": 0.1})
    assert not e2["confounded_by_receptive_field"]


def test_init_seeds_build_and_are_reproducible():
    torch = pytest.importorskip("torch")
    from jepa_noise.tests.r2_fixtures import small_cylinder
    g = small_cylinder()
    states = {}
    for seed in x.INIT_SEEDS:
        a = x.build_r2_arm("init", g, seed).state_dict()
        b = x.build_r2_arm("init", g, seed).state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)
        states[seed] = a
    key = next(k for k in states[9000] if states[9000][k].ndim == 2)
    assert len({float(states[s][key].sum()) for s in x.INIT_SEEDS}) == 5


def test_detector_readings_from_saved_evaluations(tmp_path):
    import json
    import numpy as np
    rng = np.random.default_rng(1)
    runs = np.repeat([f"r{i}" for i in range(20)], 10)
    ids = np.arange(200)
    base = {"init": 0.30, "sigreg_jepa": 0.40, "target_posfree": 0.20, "mae_mse": 0.10,
            "mae_mse_rf": 0.08}
    for arm, v in base.items():
        for seed in (x.INIT_SEEDS if arm == "init" else (9000, 9001)):
            if arm == "mae_mse_rf" and seed != 9000:
                continue
            d = tmp_path / "eval" / x.run_name("sk", arm, seed)
            d.mkdir(parents=True)
            err = np.abs(rng.normal(v, 0.01, 200))
            metrics = {"vertex_full_m": float(err.mean()), "token_log_charge_r2": 0.9,
                       "f_event": 5e-4 if arm == "sigreg_jepa" else 0.05,
                       "particle": {"mue_auc": 0.8}}
            (d / "metrics.json").write_text(json.dumps({"metrics": metrics}))
            np.savez(d / "per_event.npz", vertex_errors=err, run_ids=runs, event_ids=ids)
    r = x.detector_readings(tmp_path, "sk")
    assert r["R2-a"]["class"] == "event-invariant"
    assert r["R2-b"]["class"] == "recovered"
    assert r["R2-c"]["class"] == "yes"
    assert r["R2-d"]["order"] == list(x.WCTE_ORDER) and r["R2-d"]["kendall_tau"] == 1.0
    assert r["R2-e"]["vertex"]["mean_difference_m"] < 0
    assert r["floor_band"]["seeds"] == list(x.INIT_SEEDS)
    assert r["seed9002_triggers"] == []
    assert x.detector_readings(tmp_path, "juno")["status"] == "incomplete"
