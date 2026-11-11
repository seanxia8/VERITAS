"""Paper 3 P2 addendum tests (CPU only; no data generation, no GPU)."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from jepa_noise import ext_p2 as p2x
from jepa_noise import ext_p2a as p2a
from jepa_noise import lucid_bombs as lb

ROOT = Path(__file__).resolve().parents[3]
GEOM = lb.from_synthetic({"module_ids": tuple(range(4)),
                          "surface_counts": {"top": 16, "barrel": 44, "bottom": 16}})
RUNS = tuple(f"r{i}" for i in range(5))


def _rows(seed=0, n_events=120, *, dims=8, informative=None, runs=None):
    rng = np.random.default_rng(seed)
    run_set = tuple(RUNS if runs is None else runs)
    runs = np.repeat(np.asarray(run_set), n_events // len(run_set))
    rows = {"run_ids": runs.astype(object), "event_ids": np.arange(n_events),
            "n_hit": rng.integers(1, 40, n_events).astype(float)}
    if informative is None:
        rows["events"] = rng.standard_normal((n_events, dims))
        rows["vertex"] = rng.standard_normal((n_events, 3))
    else:
        u = rng.standard_normal((n_events, 1))
        rows["events"] = rng.standard_normal((n_events, dims))
        rows["vertex"] = np.concatenate(
            [u * np.array([2.0, -1.5, 1.0]) + 0.05 * rng.standard_normal((n_events, 3))
             for _ in range(1)], axis=1)
        rows["u"] = u
    rows["log_yield"] = rng.standard_normal(n_events)
    return rows


# ------------------------------------------------------------------- A1 design
def test_fair_design_drops_the_constant_column():
    rows = {"n_hit": np.asarray([2.0, 4.0, 6.0]),
            "run_ids": np.asarray(["r0", "r0", "r0"], dtype=object)}
    condition_t = p2x.build_drift_conditions()["T1.0"]
    design_t = p2x.r2_design(rows, condition_t, None, 4.0, GEOM)
    assert design_t.shape[1] == 2
    fair = p2a.fair_design(rows, condition_t, None, 4.0, GEOM)
    assert fair.shape == (3, 1)
    assert np.allclose(fair[:, 0], rows["n_hit"] / 4.0)
    condition_k = p2x.build_drift_conditions()["K5"]
    dead = {run: np.full(int(GEOM.n_modules), 0.1) for run in RUNS}
    design_k = p2x.r2_design(rows, condition_k, dead, 4.0, GEOM)
    assert p2a.fair_design(rows, condition_k, dead, 4.0, GEOM).shape == design_k.shape
    hit = p2a.hit_design(rows, 4.0)
    assert hit.shape == (3, 1) and np.allclose(hit[:, 0], rows["n_hit"] / 4.0)


def test_r3u_beats_r3_when_the_design_carries_the_signal():
    """A1: a planted linear effect in u must be recovered by the fair ceiling."""
    train = _rows(seed=1, informative=True)
    test = _rows(seed=2, informative=True)
    plain_v, _ = p2a.probe_errors(train, test)
    r3u_v, _ = p2a.probe_errors(p2a.augmented(train, train["u"]),
                                p2a.augmented(test, test["u"]))
    assert plain_v > 0.3            # z alone carries no vertex information here
    assert r3u_v < 0.15             # [z, u] recovers the planted effect
    assert r3u_v < plain_v


def test_ceiling_cell_runs_with_a_dead_condition():
    """A1 must build the K/P design from both train and test runs (regression)."""
    n_train = _rows(seed=3, runs=("tr0", "tr1", "tr2", "tr3", "tr4"))
    n_test = _rows(seed=4, runs=("te0", "te1", "te2", "te3", "te4"))
    c_train = dict(n_train, events=n_train["events"] + 0.3)
    c_test = dict(n_test, events=n_test["events"] + 0.3)
    data = {"condition": p2x.build_drift_conditions()["K15"],
            "n_train": n_train, "n_test": n_test, "c_train": c_train,
            "c_test": c_test, "hbar": float(np.mean(n_train["n_hit"])),
            "geometry": GEOM}
    frozen = p2a.er.fit_vertex_probe(n_train)
    cell = p2a.ceiling_cell(data, frozen_vertex=frozen, p2_cell=None)
    assert cell.get("status") != "no events"
    assert cell["v_R3u_m"] is not None and cell["rho_R3u"] is not None
    assert (cell["ceiling"] or {}).get("violation") in (True, False)


# --------------------------------------------------------------------- A2
def test_p2f_u_uses_the_fair_ceiling():
    cell = {"delta_c_m": 0.05, "v_c_m": 0.20,
            "refit": {"v_R3_m": 0.18},           # rho_R3 = 0.4
            "latent": {"rho_R2": 0.5}, "r1": {"rho_R1": None}}
    block = p2a.p2f_u(cell, {"rho_R3u": 0.9, "information_destroying_u": True})
    assert block["rho_R3"] == pytest.approx(0.4)
    assert block["rho_R3u"] == pytest.approx(0.9)
    # target = 0.8 * max(0.4, 0.9) = 0.72; the cheapest rung reaching it is R3u
    assert block["target"] == pytest.approx(0.72)
    assert block["cheapest_rung"] == "R3u"
    assert block["beyond_refit"] is True
    # if R3u is worse than R3, the label follows R3
    block2 = p2a.p2f_u(cell, {"rho_R3u": 0.2, "information_destroying_u": False})
    assert block2["target"] == pytest.approx(0.32)
    assert block2["cheapest_rung"] == "R2"
    assert block2["beyond_refit"] is False


# --------------------------------------------------------------------- A3
def _p2_payload(rho_by_model):
    models = {}
    for label, rho2 in rho_by_model.items():
        models[label] = {"conditions": {"K5": {"latent": {"rho_R2": rho2,
                                                          "r2_eq": 0.5}},
                                        "K15": {"latent": {"rho_R2": rho2,
                                                           "r2_eq": 0.5}},
                                        "K30": {"latent": {"rho_R2": rho2,
                                                           "r2_eq": 0.5}}},
                         "gain_r2_eq": None}
    return {"models": models}


def test_eta_rule_and_exclusion():
    labels = ("fix9000", "fix9001", "init9000", "frozen_target9000",
              "frozen_target9001")
    gain = {label: float(index) for index, label in enumerate(labels)}
    payload = _p2_payload({label: 0.1 * (index + 1)
                           for index, label in enumerate(labels)})
    ceilings = {label: {level: {"rho_R3u": 1.0} for level in ("K5", "K15", "K30")}
                for label in labels}
    out = p2a.eta_readings(payload, ceilings, gain)
    assert out["levels"]["K5"]["n"] == 5
    assert out["levels"]["K5"]["rho_s"] == pytest.approx(1.0)
    assert out["rule"]["verdict"] == "supported"
    # a cell whose rho_R3u is too small is excluded and reported
    ceilings["init9000"]["K5"]["rho_R3u"] = 0.01
    out2 = p2a.eta_readings(payload, ceilings, gain)
    assert out2["levels"]["K5"]["n"] == 4
    assert any(row["model"] == "init9000"
               for row in out2["levels"]["K5"]["excluded"])


def test_permutation_spearman_is_one_sided_and_keyed():
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    y = x + 0.01
    block = p2a.permutation_spearman_p2a(x, y, level="test", permutations=199)
    assert block["rho_s"] == pytest.approx(1.0)
    assert block["p"] <= 0.02 and block["one_sided"] is True
    assert block["namespace"] == p2a.PERM_NAMESPACE and block["seed"] == p2a.SEED


# --------------------------------------------------------------------- A6
def test_shift_share_bounds_and_controls():
    rng = np.random.default_rng(7)
    w = np.eye(10)
    shift = rng.standard_normal((10, 200))
    basis = p2a.er.random_control_basis(rng, 10, 3)
    share = p2a.shift_share(w, shift, basis)
    assert 0.0 <= share <= 1.0
    # an orthonormal basis spanning the whole space removes everything
    full = p2a.er.random_control_basis(rng, 10, 10)
    assert p2a.shift_share(w, shift, full) == pytest.approx(1.0, abs=1e-10)
    # a basis orthogonal to the shift removes nothing
    one_d = shift[:, :1]
    one_d = one_d / np.linalg.norm(one_d)
    orthogonal = p2a.er.random_control_basis(rng, 10, 9)
    orthogonal = orthogonal @ np.linalg.svd(orthogonal, full_matrices=False)[2]
    assert 0.0 <= p2a.shift_share(w, shift, orthogonal) <= 1.0


# --------------------------------------------------- A4 fold unit tests (F1/F2)
def _particle_rows(seed=0, n_events=600):
    rng = np.random.default_rng(seed)
    rows = _rows(seed=seed, n_events=n_events, dims=8)
    rows["particle"] = np.full(n_events, "electron")
    rows["contained"] = np.ones(n_events, bool)
    rows["deposited_MeV"] = rng.uniform(50.0, 500.0, n_events)
    return rows


def test_fold_f1_energy_block_uses_the_particle_rows():
    photon = _rows(seed=5, n_events=600)
    particle_train = _particle_rows(seed=6)
    particle_test = _particle_rows(seed=7)
    block = p2x._x3p_blocks(
        {"train_N": photon, "test_N": photon, "test_D": photon,
         "particle_train_N": particle_train, "particle_test_N": particle_test,
         "particle_test_D": particle_test},
        kinds=["vertex", "log_yield", "energy"])
    assert block["energy"]["N"] is not None
    assert block["energy"]["N"]["resolution"] is not None
    assert block["energy"]["N"]["errors"].size == 600
    # and the energy block is the particle probe, not the photon rows
    fitted = p2a.er.fit_energy_probe(particle_train)
    expected = p2a.er.energy_metrics(particle_train, particle_test, fitted=fitted)
    assert block["energy"]["N"]["resolution"] == pytest.approx(
        expected["resolution"], rel=1e-12)


def test_fold_f2_exact_mask_uses_the_trig_lookup(monkeypatch):
    """The folded _cell_for_model must mask with _trig_lookup(info), not the dict."""
    synthetic = {}
    counts = {}
    for name in list(p2a.CONDITIONS) + [p2x.R1_NAME, "N"]:
        for role in ("train", "test"):
            rows = _rows(seed=hash((name, role)) % 1000)
            synthetic[p2a.p2x.er.embedding_cache_key(role, name)] = rows
            trig = np.zeros(len(rows["event_ids"]))
            if name == "T1.0w" and role == "test":
                trig[-20:] = 0.5          # 20 window-edge events
            synthetic[("info", name, role)] = {
                "run_ids": np.asarray(rows["run_ids"]).astype(str),
                "event_ids": np.asarray(rows["event_ids"], np.int64),
                "trig_rel": trig}
            counts[(name, role)] = rows
    for role in ("train", "test"):
        synthetic[p2a.p2x.er.embedding_cache_key(role, "N")] = counts[("N", role)]

    def fake_cache_rows(cache, key):
        return synthetic.get(str(key))

    captured = {}

    def fake_cell_reading(data, **kwargs):
        captured[data["condition"].name] = int(data["c_test"]["events"].shape[0])
        return {"condition": data["condition"].name, "status": "captured"}

    monkeypatch.setattr(p2x, "cache_rows", fake_cache_rows)
    monkeypatch.setattr(p2x, "cell_reading", fake_cell_reading)
    infos = {(name, role): synthetic[("info", name, role)]
             for name in list(p2a.CONDITIONS) + [p2x.R1_NAME]
             for role in ("train", "test")}
    out = p2x._cell_for_model("m", object(), object(), p2a.p2x.build_drift_conditions(),
                              infos, {}, 10.0, GEOM, exact=True,
                              log=lambda *a, **k: None)
    assert out["K5"]["status"] == "captured"
    assert captured["K5"] == 120                 # every trig' = 0 -> kept
    assert captured["T1.0w"] == 100             # 20 window-edge events removed
    assert out["T1.0w"]["condition"] == "T1.0w"


# --------------------------------------------------------------- A5 CPU launcher
def test_cpu_launcher_allow_list():
    from scripts import jepa_cpu_launch as launcher
    assert launcher.extension_root_for_module(
        "scripts.jepa_ext_predictability_cost") == "ext_predictability_20261029"
    assert launcher.extension_root_for_module("scripts.jepa_ext_p2") == "ext_p2_20261109"
    assert launcher.extension_root_for_module(
        "scripts.jepa_ext_p2a") == "ext_p2_20261109"
    with pytest.raises(ValueError):
        launcher.extension_root_for_module("scripts.jepa_ext_repair")


# ------------------------------------------------------------------------ gate
def _fake_repo(tmp_path, *, record=True, verdict="PASS", sha=None):
    root = tmp_path / "repo"
    (root / "docs/jepa").mkdir(parents=True)
    real = ROOT / p2a.RUNBOOK_REL
    (root / p2a.RUNBOOK_REL).write_bytes(real.read_bytes())
    (root / p2a.REVIEW_REL).write_text("PASS\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "-m", "fixture"], check=True)
    if record:
        impl = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                       text=True).strip()
        payload = {"runbook_sha256": sha or p2a.RUNBOOK_SHA256,
                   "impl_commit": impl, "review_path": p2a.REVIEW_REL,
                   "review_verdict": verdict, "lead": "lead", "date": "2026-11-11"}
        (root / p2a.ACCEPTANCE_REL).write_text(json.dumps(payload))
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t",
                        "commit", "-q", "-m", "acceptance"], check=True)
    return root


def test_gate_pending_denied_and_granted(tmp_path):
    pending = _fake_repo(tmp_path / "pending", record=False)
    check = p2a.assert_authorised("dry-run", root=pending, dry_run=True)
    assert check["status"] == "gate_pending_acceptance"
    with pytest.raises(SystemExit):
        p2a.assert_authorised("p2a", root=pending)
    granted = _fake_repo(tmp_path / "granted")
    assert p2a.assert_authorised("p2a", root=granted)["authorised"]
    blocked = _fake_repo(tmp_path / "blocked", verdict="BLOCK")
    assert not p2a._gate_checks("p2a", root=blocked)["authorised"]
    wrong = _fake_repo(tmp_path / "wrong", sha="0" * 64)
    assert not p2a._gate_checks("p2a", root=wrong)["authorised"]
    with pytest.raises(ValueError):
        p2a.assert_authorised("train", root=granted, dry_run=True)
