"""WP5 arm-objective and probe-form tests, CPU only (§3.5.6).

Fixtures are the synthetic geometry (NOT WCTE) and tiny runs; the printed
Monte-Carlo number is a fixture number, not a registered result.
"""
import os

for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_noise import lucid_arms as la
from jepa_noise import lucid_bombs as lb
from jepa_noise import point_encoder as pe
from jepa_noise import sigreg

CONFIG = la.CONFIG_PATH
SMALL = lb.from_synthetic({"module_ids": tuple(range(4)),
                           "surface_counts": {"top": 16, "barrel": 44, "bottom": 16}})
N_PMT, N_MOD, D_PROBE = SMALL.n_pmt, SMALL.n_modules, 8
CHI2 = dict(s=0.10, s_p=0.10, sigma_spe=lb.SIGMA_SPE, v_dark=lb.V_DARK)


def _run(path, *, events=2, hits=60, seed=0):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    reads = [lb.HitReadout(np.sort(rng.choice(N_PMT, hits, replace=False)).astype(np.int32),
                           (rng.exponential(3.0, hits) + 0.3).astype(np.float32),
                           rng.normal(50.0, 5.0, hits).astype(np.float32))
             for _ in range(events)]
    lb.write_sparse_hits(path / "hits.npz",
                         lb.sparse_from_readouts(np.arange(events), reads))
    return path


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    return pe.tokenise(_run(tmp_path_factory.mktemp("run")), SMALL, split="within", seed=0)


@pytest.fixture(scope="module")
def bank(tmp_path_factory):
    directory = _run(tmp_path_factory.mktemp("dark"), events=3, hits=8, seed=11)
    return la.dark_bank_from_store(lb.read_sparse_hits(directory / "hits.npz"), SMALL)


@pytest.fixture(scope="module")
def settings():
    return la.ArmSettings.from_config(la.load_config())


def _arm(arm, settings, bank, seed=0):
    return la.build_arm(arm, SMALL, settings=settings, seed=seed,
                        beta=10.0 if arm in la.BETA_ARMS else None,
                        dark_bank=bank if arm in la.ANCHOR_ARMS else None)


def _probe_runs(n_runs=7, *, noise=0.01, constant_module_gain=False, nan_tokens=False):
    """Synthetic runs whose token/raw columns carry the §3.5.6 truth targets."""
    rng = np.random.default_rng(3)
    runs = []
    for r in range(n_runs):
        events = 3
        gain_m = np.zeros(N_MOD) if constant_module_gain else rng.normal(0.0, 0.05, N_MOD)
        gain_p, vertex = rng.normal(0.0, 0.05, N_PMT), rng.normal(0.0, 0.3, (events, 3))
        log_yield, f2, age = rng.uniform(4.0, 7.0, events), rng.normal(), rng.uniform()
        tokens = noise * rng.standard_normal((events, N_PMT, D_PROBE))
        tokens[..., 0] += gain_p
        tokens[..., 1] += gain_m[SMALL.module_index]
        tokens[..., 2:5] += vertex[:, None, :]
        tokens[..., 5] += log_yield[:, None]
        tokens[..., 6] += f2
        tokens[..., 7] += age
        raw = noise * rng.standard_normal((events, N_PMT, pe.N_FEATURES))
        raw[..., 6] += gain_m[SMALL.module_index]
        raw[..., 7] += gain_p
        hit, hidden = np.ones((events, N_PMT), bool), \
            np.broadcast_to(SMALL.pmt_id_in_mpmt % 2 == 1, (events, N_PMT))
        truth = {"vertex": vertex, "log_yield": log_yield, "module_gain": gain_m,
                 "pmt_gain": gain_p, "f2_variable": f2, "age": age}
        viewed = np.full_like(tokens, np.nan) if nan_tokens else tokens
        runs.append(la.RunProbeData(f"run{r}", np.arange(events), viewed, viewed.copy(),
                                    raw, hit, hidden, SMALL.module_index, truth))
    return runs


# ------------------------------------------------------------------- config
def test_config_freezes_the_registered_numerics():
    config = la.load_config()
    settings = la.ArmSettings.from_config(config)
    assert settings.beta_grid == (0.1, 1.0, 10.0, 100.0, 1000.0)
    assert (settings.tau_start, settings.tau_end) == (0.996, 1.0)
    assert (settings.sigreg.num_slices, settings.sigreg.n_points, settings.sigreg.t_max) \
        == (1024, 17, 3.0)
    assert settings.coefficient_lambda == settings.sigreg.coefficient_lambda == 0.02
    assert settings.sigma_spe == lb.SIGMA_SPE == 0.3
    assert 0.0 < settings.v_dark == pytest.approx(lb.V_DARK, rel=1e-12)
    generator = lb.load_generator_config(Path(CONFIG).parent / "lucid_bombs_v0.json")
    assert settings.time_sigma_ns == generator.tts_sigma_ns == 1.5
    assert settings.rescale_sd == 0.01 and settings.variance_floor_pe2 == 0.0
    assert config["mae_chi2"]["variance_floor_status"].startswith("resolved_2026-10-17")
    assert settings.v_dark > 0.0
    assert config["anchor"]["dark_overlay"]["draws"] == 2
    assert config["anchor"]["dark_overlay"]["dark_only_draws"] == 1
    assert config["training"]["optimizer"] == "pending_review"
    assert all(v == "pending_review" for v in config["training"]["optimizer_detail"].values())
    assert config["probes"]["primary_form"] == la.PRIMARY_PROBE_FORM == "run_mean"
    assert config["sigreg"]["source_pin"]["commit_sha"] == sigreg.SIGREG_SOURCE_PIN["commit_sha"]
    assert config["beta"]["arms"]["sigreg_jepa_anchored"] is None   # not selected here


# --------------------------------------------------------------------- arms
def test_every_arm_forward_backward_on_batch_two(batch, settings, bank):
    losses, metrics = {}, {}
    for arm in la.ALL_ARMS:
        model = _arm(arm, settings, bank)
        extra = {"oracle": True, "twin_batch": batch} if arm == la.ORACLE_ARM else {}
        out = model(batch, step=0, **extra)
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
        assert any(float(g.abs().sum()) > 0.0 for g in grads)
        losses[arm], metrics[arm] = float(out["loss"]), out["metrics"]
    assert losses[la.ORACLE_ARM] == losses["sigreg_jepa"]      # same seed, same objective
    assert losses["sigreg_jepa_anchored"] > losses["sigreg_jepa"]
    distinct = {a: losses[a] for a in la.ALL_ARMS if a != la.ORACLE_ARM}
    assert len({round(v, 6) for v in distinct.values()}) == len(distinct)
    assert metrics["mae_chi2"]["uses_oracle_noise_information"] is True
    assert 0.0 <= metrics["mae_chi2"]["chi2_floor_fraction"] <= 1.0
    assert metrics["mae_mse"]["hidden_hits"] > 0
    print("\n[record] arm losses: " + ", ".join(f"{a}={v:.4f}" for a, v in losses.items()))


# ---------------------------------------------------------------- mae_chi2
def test_chi2_woodbury_matches_dense_on_one_module():
    rng = np.random.default_rng(0)
    mu = np.array([0.7, 1.5, 3.0, 0.2, 8.0])
    q = np.maximum(mu + rng.normal(0.0, 1.2, mu.size), 0.0)
    dense = la.chi2_dense_module(q - mu, mu, floor=1e-6, **CHI2)
    terms = la.chi2_module_terms(torch.tensor(q[None]), torch.tensor(mu[None]),
                                 torch.zeros(5, dtype=torch.long), torch.arange(5),
                                 n_modules=2, pmts_per_module=19, floor=1e-6, **CHI2)
    assert np.isclose(float(terms[0, 0]), dense, rtol=1e-9, atol=1e-12)
    assert float(terms[0, 1]) == 0.0          # a module with no hidden PMTs is neutral
    # the generator's dark-charge variance keeps C_m positive definite at mu = 0
    assert CHI2["v_dark"] > 0.0
    assert np.isfinite(la.chi2_dense_module(np.zeros(1), np.zeros(1), floor=0.0, **CHI2))
    # a non-registered v_dark = 0 covariance is only usable with a positive floor
    with pytest.raises(ValueError):
        la.chi2_dense_module(np.zeros(1), np.zeros(1), floor=0.0,
                             **{**CHI2, "v_dark": 0.0})
    assert np.isfinite(la.chi2_dense_module(np.zeros(1), np.zeros(1), floor=1e-6,
                                            **{**CHI2, "v_dark": 0.0}))


def test_monte_carlo_covariance_within_ten_percent():
    mu = np.array([4.0, 6.0, 9.0, 12.0, 8.0])
    samples = la.simulate_module_readout(mu, n_samples=400_000, seed=7, **CHI2)
    empirical = np.cov(samples, rowvar=False)
    diagonal = mu * (1.0 + CHI2["sigma_spe"] ** 2) + CHI2["v_dark"] + CHI2["s_p"] ** 2 * mu ** 2
    covariance = np.diag(diagonal) + CHI2["s"] ** 2 * np.outer(mu, mu)
    relative = np.abs(empirical - covariance) / np.abs(covariance)
    assert relative.max() < 0.10, relative
    print("\n[record] Monte-Carlo covariance relative error per entry (%):\n"
          f"{np.round(relative * 100.0, 2)}\nmax = {relative.max() * 100.0:.2f}%")


# ------------------------------------------------------------------- anchor
def test_anchor_penalty_zero_for_identity_and_positive_otherwise(batch, settings, bank):
    identity = la.build_arm("sigreg_jepa_anchored", SMALL, settings=settings, beta=10.0,
                            seed=0, dark_bank=la.DarkBank.empty(N_PMT))
    zeros = torch.zeros(batch.features.size(0), dtype=torch.long)
    ones = torch.ones(batch.features.size(0), N_MOD)
    assert torch.equal(identity.perturb(batch.features, dark_index=zeros, rescale=ones),
                       batch.features)
    z = identity._module_tokens(identity.jepa.encode(batch.features))
    assert float(identity.anchor_penalty(z, z.clone())) == 0.0
    arm = _arm("sigreg_jepa_anchored", settings, bank)
    for draws in ({"dark_index": torch.ones_like(zeros), "rescale": ones},
                  {"dark_index": zeros, "rescale": torch.full((batch.features.size(0),
                                                               N_MOD), 1.5)}):
        moved = arm.perturb(batch.features, **draws)
        assert not torch.equal(moved, batch.features)
        penalty = arm.anchor_penalty(z, arm._module_tokens(arm.jepa.encode(moved)))
        assert float(penalty) > 0.0
    with pytest.raises(ValueError):
        arm.perturb(batch.features, rescale=torch.ones(3, N_MOD))


def test_anchor_trace_denominator_carries_gradient(settings, bank):
    arm = _arm("sigreg_jepa_anchored", settings, bank)
    generator = torch.Generator().manual_seed(5)
    z = torch.randn(2, N_MOD, arm.d_model, generator=generator, requires_grad=True)
    z_tilde = z.detach() + 0.1 * torch.randn(z.shape, generator=generator)
    penalty = arm.anchor_penalty(z, z_tilde)
    gradient = torch.autograd.grad(penalty, z)[0]
    detached = (arm.beta * (z_tilde - z).pow(2).sum(-1).mean()
                / arm._trace_cov(z).detach().clamp_min(1e-6))
    gradient_detached = torch.autograd.grad(detached, z)[0]
    assert bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0.0
    assert not torch.allclose(gradient, gradient_detached)


def test_dark_variant_differs_from_the_anchored_arm(batch, settings, bank):
    anchored = _arm("sigreg_jepa_anchored", settings, bank)
    dark = _arm(la.DARK_VARIANT, settings, bank)
    full, dark_out = anchored(batch), dark(batch)
    assert dark.dark_only and not anchored.dark_only
    assert full["metrics"]["anchor_draws"] == 2 and dark_out["metrics"]["anchor_draws"] == 1
    assert float(full["anchor"]) > 0.0 and float(dark_out["anchor"]) > 0.0
    assert float(full["anchor"]) != float(dark_out["anchor"])
    assert float(full["loss"]) != float(dark_out["loss"])


def test_oracle_arm_refuses_non_oracle_access(batch, settings, bank):
    oracle, plain = _arm(la.ORACLE_ARM, settings, bank), _arm("sigreg_jepa", settings, bank)
    assert oracle.arm == la.ORACLE_ARM and plain.arm != la.ORACLE_ARM
    with pytest.raises(PermissionError):
        oracle(batch)
    with pytest.raises(PermissionError):
        oracle(batch, oracle=True)                     # no twin batch
    with pytest.raises(PermissionError):
        plain(batch, twin_batch=batch)                 # only the oracle arm may read it
    twin = oracle(batch, oracle=True, twin_batch=batch)
    assert float(twin["loss"]) == float(plain(batch)["loss"])


# -------------------------------------------------------------------- probes
def test_probe_forms_are_finite_and_run_mean_is_primary():
    runs = _probe_runs(n_runs=20)
    train, test = runs[:15], runs[15:]
    for kind in la.PROBE_KINDS:
        for form in la.PROBE_FORMS:
            if form == "run_mean" and kind in ("vertex", "log_yield"):
                with pytest.raises(ValueError):   # a per-event truth has no run-mean target
                    la.build_probe_dataset(runs, kind, form)
                continue
            result = la.probe_r2(la.build_probe_dataset(train, kind, form),
                                 la.build_probe_dataset(test, kind, form))
            assert result["status"] == "evaluated", (kind, form, result)
            assert np.isfinite(result["r2_mean"]), (kind, form, result)
            if form != "raw_input":
                assert result["r2_mean"] > 0.5, (kind, form, result)
    run_mean = la.build_probe_dataset(runs, "module_gain", "run_mean")
    assert run_mean.n_rows == len(runs) * N_MOD     # one row per (run, module)


def test_constant_target_is_not_evaluable():
    runs = _probe_runs(n_runs=6, constant_module_gain=True)
    result = la.probe_r2(la.build_probe_dataset(runs[:5], "module_gain"),
                         la.build_probe_dataset(runs[5:], "module_gain"))
    assert result["status"] == "not_evaluable"
    assert result["reason"] == "constant_probe_train_target" and result["r2"] is None


def test_raw_input_ceiling_uses_raw_features_only():
    runs = _probe_runs(n_runs=6, nan_tokens=True)
    dataset = la.build_probe_dataset(runs[:5], "module_gain", "raw_input")
    assert dataset.x.shape[1] == pe.N_FEATURES and np.isfinite(dataset.x).all()
    pooled, _ = la._module_pool(runs[0].raw, runs[0].module_index, N_MOD)
    assert np.allclose(dataset.x[:N_MOD], pooled[0])      # module-mean raw features
    result = la.probe_r2(dataset,
                         la.build_probe_dataset(runs[5:], "module_gain", "raw_input"))
    assert result["status"] == "evaluated" and np.isfinite(result["r2_mean"])
    tokens = la.build_probe_dataset(_probe_runs(n_runs=5), "module_gain", "per_event")
    assert tokens.x.shape[1] == D_PROBE                   # the encoder form, not raw


def test_probe_data_from_the_encoder(tmp_path, settings):
    model = la.build_arm("sigreg_jepa", SMALL, settings=settings, seed=0).jepa
    runs = []
    for i in range(6):
        batch = pe.tokenise(_run(tmp_path / f"r{i}", seed=i), SMALL, split="within", seed=i)
        state = lb.state_for_run("F1", 0.05, f"run{i}", SMALL, seed=1)
        rng = np.random.default_rng(i)
        truth = la.truth_from_state(SMALL, state, vertex_m=rng.normal(0.0, 0.3, (2, 3)),
                                    yield_photons=rng.uniform(100.0, 1000.0, 2))
        runs.append(la.probe_data(model, batch, truth, SMALL, run_id=f"run{i}"))
    assert runs[0].tokens.shape == (2, N_PMT, pe.EncoderConfig().d_model)
    result = la.probe_r2(la.build_probe_dataset(runs[:5], "module_gain"),
                         la.build_probe_dataset(runs[5:], "module_gain"))
    assert result["status"] == "evaluated" and np.isfinite(result["r2_mean"])


def test_random_init_floor_hook(batch):
    floor = la.random_init_floor("sigreg_jepa", SMALL, seed=3)
    again = la.random_init_floor("sigreg_jepa", SMALL, seed=3)
    assert all(torch.equal(a, b) for a, b in zip(floor.parameters(), again.parameters()))
    assert all(p.grad is None for p in floor.parameters())
    with pytest.raises(ValueError):
        la.random_init_floor("not_an_arm", SMALL)
    state = lb.state_for_run("F1", 0.05, "run0", SMALL, seed=1)
    truth = la.truth_from_state(SMALL, state, vertex_m=np.zeros((2, 3)),
                                yield_photons=np.full(2, 500.0))
    data = la.probe_data(floor, batch, truth, SMALL, run_id="run0")
    assert np.isfinite(data.tokens).all() and data.tokens.shape[1] == N_PMT


def test_parameter_groups_and_ema_update(batch, settings, bank):
    model = _arm("ijepa_ema", settings, bank)
    coverage = model.check_parameter_coverage()
    assert coverage["n_muon"] > 0 and coverage["n_adamw"] > 0
    target = [p.detach().clone() for p in model.target_encoder.parameters()]
    online = list(model.jepa.encoder.parameters())
    assert all(torch.equal(a, b) for a, b in zip(target, online))
    with torch.no_grad():
        for parameter in online:
            parameter.add_(0.05)
    tau = model.update_target(0)
    assert tau == settings.tau_start
    expected = [tau * t + (1.0 - tau) * o.detach() for t, o in zip(target, online)]
    assert all(torch.allclose(p, e, atol=1e-6)
               for p, e in zip(model.target_encoder.parameters(), expected))
    assert model.update_target(settings.total_steps) == 1.0
    with pytest.raises(ValueError):
        _arm("sigreg_jepa", settings, bank).update_target(0)
