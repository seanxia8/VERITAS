"""WP4a Tier-1 bridge tests (runbook v4 §3.5.7; v3 WP4 stays in force).

All fixtures here are synthetic and independent of ``views.py`` so the bridge is
testable while the WP2-v4 A4 base-rho integration is pending.  The registered
bridge data law is A4 at g = 0.02, nominal SNR, channel split.

Scope notes:
- The amplitude-preservation R² >= 0.9 criterion is a *trained-encoder*
  criterion (runbook v4 §3.5.7): it is measured after the smoke and on the final
  encoders.  The fixture test below only verifies the probe and the amplitude
  content of the read-out at random init; it is not the registered criterion.
- ``ijepa_ema`` is not a bridge arm and is not implemented here (out of scope).
"""

import os

# Runbook v3 §3.1: no implicit compilation for CPU checks.  Must precede any
# reconstruction_model import (the Muon module decides at import time).
os.environ.setdefault("RECONSTRUCTION_DISABLE_TORCH_COMPILE", "1")
# Keep the CPU test process light on the shared Mac; must precede numpy/torch.
for _variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import copy
import math

import numpy as np
import pytest
import torch

from jepa_noise import train as synthetic
from jepa_noise.backbone import TokenGridEncoder
from jepa_noise.jepa import (
    ABLATION_ARM,
    ALL_ARMS,
    BRIDGE_ARMS,
    build_model,
    build_optimizers,
    check_optimizer_coverage,
    load_config,
    transformer_config,
)
from jepa_noise.masks import raw_split
from jepa_noise.sigreg import SIGREG_SOURCE_PIN, SIGReg, SIGRegConfig

# Middle of the registered beta grid.  A placeholder for smoke/unit tests only:
# the registered beta is selected on the independent Tier-1 development cohort
# (v3 §1.3); this is not a selection and is not chosen from any outcome.
BETA_SMOKE = 1.0
BATCH = 2


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def products():
    """Real registered A4 bridge products: fixed scaling, overlay bank, N_actual.

    Built once from the frozen ``views.py`` API (WP2-v4 commit 10693eb): A4 at
    g = 0.02, nominal, channel split.  The synthetic fixture is used only where
    a test explicitly needs synthetic records (the scaling-fit mechanics).
    """
    return synthetic.a4_bridge_products()


def build_arm(config, products, arm, seed=0, beta=BETA_SMOKE):
    return build_model(
        config,
        arm,
        channel_mean=products["channel_mean"],
        channel_std=products["channel_std"],
        target_noise_covariance=products["target_noise_covariance"],
        noise_bank=products["noise_bank"],
        beta=beta if arm == "sigreg_jepa_anchored" else None,
        seed=seed,
    )


def batch(config, n=BATCH, seed=0):
    return synthetic.synthetic_batch(config, batch=n, seed=seed)


# ---------------------------------------------------------------- config freeze
def test_config_pins_registered_numerics(config):
    backbone = config["backbone"]
    data = config["data"]
    assert data["split"] == "channel"
    assert data["channels"] == 8 and data["n_samples"] == 256
    assert data["patch_len"] == data["patch_stride"] == 32
    assert data["context_channels"] == [0, 1, 2, 3]
    assert data["target_channels"] == [4, 5, 6, 7]
    assert "none" in data["gap"]
    assert backbone["d_model"] == 128
    assert backbone["n_head"] == 4
    assert (backbone["n_time_layers"], backbone["n_channel_layers"]) == (1, 1)
    assert backbone["depths"] == [1, 1]
    assert backbone["readout_before_final_proj_norm"] is True
    assert config["predictor"]["d_pred"] == 64
    assert config["predictor"]["n_layers"] == 2
    sigreg = config["sigreg"]
    assert (sigreg["num_slices"], sigreg["t_max"], sigreg["n_points"]) == (1024, 3.0, 17)
    assert sigreg["coefficient_lambda"] == 0.02
    assert config["objective"]["anchor"]["beta_grid"] == [0.1, 1.0, 10.0]
    assert config["optimizer"]["adamw"]["lr"] == 0.001
    assert config["optimizer"]["muon"]["lr"] == 0.001
    assert config["training"]["batch"] == 128
    assert config["training"]["steps_primary"] == 10000
    # Data products are frozen from the registered A4 generator (not pending).
    assert data["fixed_scaling"]["status"].startswith("frozen")
    assert len(data["fixed_scaling"]["values_mean_std"]["mean"]) == 8
    assert len(data["fixed_scaling"]["values_mean_std"]["std"]) == 8
    assert data["target_noise_covariance"]["status"].startswith("frozen")
    assert data["target_noise_covariance"]["n_target"] == 1024
    assert data["generator"]["base_rho"] == 0.0
    assert data["generator"]["law"] == "v4-A4-waveform-v1"
    # SIGReg source is pinned to a commit, not to a moving branch.
    source = config["sigreg"]["source"]
    assert source["commit_sha"] == SIGREG_SOURCE_PIN["commit_sha"]
    assert source["commit_sha"] in source["raw_url"]
    assert "n_target" in config["sigreg"]["n_factor_note"]
    assert len(config["decisions"]) >= 8
    assert all(isinstance(line, str) and line for line in config["decisions"])
    assert any("not a selection" in line for line in config["decisions"])
    assert "pending" in config["pre_outcome_review"]


def test_frozen_a4_products_match_registered_generator(config, products):
    """The config's frozen scaling and N_actual hashes reproduce from views.py."""
    metadata = products["metadata"]
    data = config["data"]
    frozen_scaling = data["fixed_scaling"]
    assert frozen_scaling["status"].startswith("frozen")
    assert frozen_scaling["values_mean_std"]["mean"] == metadata["fixed_scaling"]["mean"]
    assert frozen_scaling["values_mean_std"]["std"] == metadata["fixed_scaling"]["std"]
    assert frozen_scaling["sha256"] == metadata["fixed_scaling"]["sha256"]
    assert (
        frozen_scaling["n_records"],
        frozen_scaling["role"],
        frozen_scaling["seed"],
        frozen_scaling["cell"],
    ) == (4096, "noise_cal", 0, "bridge")
    assert all(value > 0 for value in frozen_scaling["values_mean_std"]["std"])
    # The fitted products are bit-identical to the frozen config values.
    assert torch.equal(
        products["channel_mean"],
        torch.tensor(frozen_scaling["values_mean_std"]["mean"], dtype=torch.float32),
    )
    assert torch.equal(
        products["channel_std"],
        torch.tensor(frozen_scaling["values_mean_std"]["std"], dtype=torch.float32),
    )

    frozen_target = data["target_noise_covariance"]
    assert frozen_target["status"].startswith("frozen")
    assert frozen_target["sha256"] == metadata["target_noise_covariance"]["sha256"]
    assert (
        frozen_target["target_raw_index_sha256"]
        == metadata["target_noise_covariance"]["target_raw_index_sha256"]
    )
    assert (
        frozen_target["generator_manifest_sha256"]
        == metadata["generator"]["manifest_sha256"]
    )
    assert (
        frozen_target["generator_noise_covariance_sha256"]
        == metadata["generator"]["noise_covariance_sha256"]
    )

    covariance = np.asarray(products["target_noise_covariance"], dtype=np.float64)
    target_index = raw_split("channel").target
    marginal = covariance[np.ix_(target_index, target_index)]
    assert marginal.shape == (1024, 1024)
    np.linalg.cholesky(marginal)  # raises if the frozen marginal is not SPD
    model = build_arm(config, products, "mae_chi2")
    assert model.target_precision_chol.shape == (1024, 1024)


# ------------------------------------------------------------------- shapes
def test_token_grid_shape_and_a4_masks(config, products):
    model = build_arm(config, products, "sigreg_jepa")
    encoder = model.encoder
    x = batch(config)
    grid = encoder(x)
    assert grid.shape == (BATCH, 8, 8, 128)
    assert encoder.encode(x, "target")["grid"].shape == (BATCH, 8, 8, 128)
    assert encoder.encode_after_time(x).shape == (2, 8, 8, 128)  # (B, C, N, d)
    assert (encoder.n_context_tokens, encoder.n_target_tokens) == (32, 32)
    assert encoder.n_gap_tokens == 0

    registered = raw_split("channel")
    assert torch.equal(encoder.context_raw_index, torch.as_tensor(registered.context))
    assert torch.equal(encoder.target_raw_index, torch.as_tensor(registered.target))
    assert encoder.gap_raw_index.numel() == 0
    partition = torch.cat([encoder.context_raw_index, encoder.target_raw_index])
    assert torch.equal(partition.sort().values, torch.arange(8 * 256))

    context_tokens = encoder.select_tokens(grid, "context")
    prediction = model.predictor(context_tokens)
    assert prediction.shape == (BATCH, 32, 128)


# ------------------------------------------------- read-out is pre-final-norm
def test_readout_is_before_final_proj_norm(config, products):
    model = build_arm(config, products, "sigreg_jepa")
    encoder = model.encoder
    x = batch(config)
    # final_proj_norm is deliberately never called by the forward pass, so the
    # read-out must equal the output of the last spatial block, i.e. exactly the
    # tensor that would enter final_proj_norm.
    captured = {}
    def capture_spatial(module, inputs, output):
        # A forward hook must return None; a returned value would replace the
        # module output.
        captured["out"] = output.detach()

    handle = encoder.spatial_layers[-1].register_forward_hook(capture_spatial)
    with torch.no_grad():
        grid = encoder(x)
    handle.remove()

    pre_norm = captured["out"].view(grid.shape)
    assert torch.equal(grid, pre_norm)
    # It is not the normalised tensor; the normalised tensor has unit RMS.
    with torch.no_grad():
        post_norm = encoder.final_proj_norm(
            pre_norm.reshape(-1, 8, 128)
        ).view(grid.shape)
    assert not torch.allclose(grid, post_norm)
    rms = lambda t: t.pow(2).mean(dim=-1).sqrt()
    assert (rms(post_norm) - 1.0).abs().max() < 1e-3
    # A test that would fail if final_proj_norm were applied to the read-out.
    assert not torch.allclose(rms(grid), torch.ones_like(rms(grid)), atol=1e-3)
    assert (rms(grid) - 1.0).abs().max() > 1e-2


# --------------------------------------------------------- fixed input scaling
def test_fixed_scaling_is_stored_fixed_not_per_event(config):
    records = synthetic.synthetic_calibration_noise(config, n_records=32, seed=5)
    transformer = transformer_config(config)
    encoder = TokenGridEncoder.from_calibration_noise(
        transformer,
        records,
        n_channels=8,
        n_samples=256,
        context_channels=(0, 1, 2, 3),
        target_channels=(4, 5, 6, 7),
    )
    mean, std = TokenGridEncoder.fit_scaling(records)
    assert torch.allclose(encoder.channel_mean, mean)
    assert torch.allclose(encoder.channel_std, std)
    assert not encoder.channel_mean.requires_grad
    assert not encoder.channel_std.requires_grad

    x = synthetic.synthetic_batch(config, batch=3, seed=2)
    captured = {}
    def capture_patch_input(module, inputs, output):
        captured["input"] = inputs[0].detach()

    handle = encoder.patch_embedding.register_forward_hook(capture_patch_input)
    with torch.no_grad():
        encoder.encode(x, "unmasked")
    handle.remove()
    expected = (
        (x - mean.view(1, -1, 1)) / std.view(1, -1, 1)
    ).reshape(3 * 8, 256).unfold(-1, 32, 32)
    assert torch.allclose(captured["input"], expected)

    # A forward pass never refits or moves the stored scaling.
    snapshot = (encoder.channel_mean.clone(), encoder.channel_std.clone())
    with torch.no_grad():
        encoder.encode(x * 3.0 + 1.0, "unmasked")
    assert torch.equal(encoder.channel_mean, snapshot[0])
    assert torch.equal(encoder.channel_std, snapshot[1])

    with torch.no_grad():
        base = encoder.encode(x, "unmasked")["grid"]
        # A per-channel constant offset must survive: a per-event z-score would
        # remove it (mean subtraction).
        offset = torch.arange(8, dtype=torch.float32) * 0.37
        shifted = encoder.encode(x + offset.view(1, -1, 1), "unmasked")["grid"]
        # A per-event z-score would map x and 2x to the same input; fixed
        # scaling keeps the amplitude.
        scaled = encoder.encode(2.0 * x, "unmasked")["grid"]
    assert not torch.allclose(base, shifted)
    assert not torch.allclose(base, scaled)


# ---------------------------------------------------------------- leakage
def test_leakage_context_and_target_views(config, products):
    model = build_arm(config, products, "sigreg_jepa")
    encoder = model.encoder
    x = batch(config, seed=3)
    context_index = encoder.context_raw_index
    target_index = encoder.target_raw_index

    context_input = x.clone().requires_grad_(True)
    grid = encoder.encode(context_input, "context")["grid"]
    grid.pow(2).sum().backward()
    flat_grad = context_input.grad.reshape(BATCH, -1)
    assert torch.count_nonzero(flat_grad[:, target_index]) == 0
    assert flat_grad[:, context_index].abs().max() > 0

    with torch.no_grad():
        base_context = encoder.encode(x, "context")["grid"]
        perturbed = x.clone()
        perturbed.reshape(BATCH, -1)[:, target_index] += 3.7
        perturbed_context = encoder.encode(perturbed, "context")["grid"]
    assert torch.equal(base_context, perturbed_context)

    target_input = x.clone().requires_grad_(True)
    target_grid = encoder.encode(target_input, "target")["grid"]
    target_grid.pow(2).sum().backward()
    flat_grad = target_input.grad.reshape(BATCH, -1)
    assert torch.count_nonzero(flat_grad[:, context_index]) == 0
    assert flat_grad[:, target_index].abs().max() > 0

    with torch.no_grad():
        base_target = encoder.encode(x, "target")["grid"]
        perturbed = x.clone()
        perturbed.reshape(BATCH, -1)[:, context_index] += 2.5
        pert_target = encoder.encode(perturbed, "target")["grid"]
    assert torch.equal(base_target, pert_target)

    # The predictor reads context outputs only, so its prediction is also
    # independent of every raw target sample.
    predictor_input = x.clone().requires_grad_(True)
    context_tokens = encoder.select_tokens(
        encoder.encode(predictor_input, "context")["grid"], "context"
    )
    model.predictor(context_tokens).pow(2).sum().backward()
    assert torch.count_nonzero(
        predictor_input.grad.reshape(BATCH, -1)[:, target_index]
    ) == 0


def test_leakage_also_holds_for_a_nonempty_gap(config, products):
    """The registered channel split has no gap; the masking code must still mask
    gap samples.  Exercised with a synthetic time-style split fixture."""
    data = copy.deepcopy(config["data"])
    data["context_channels"] = list(range(8))
    data["target_channels"] = list(range(8))
    data["context_patches"] = [0, 1, 2]
    data["target_patches"] = [5, 6, 7]
    encoder = TokenGridEncoder(
        transformer_config(config),
        channel_mean=products["channel_mean"],
        channel_std=products["channel_std"],
        n_channels=8,
        n_samples=256,
        context_channels=data["context_channels"],
        target_channels=data["target_channels"],
        context_patches=data["context_patches"],
        target_patches=data["target_patches"],
    )
    encoder.initialize(0)
    assert encoder.n_gap_tokens > 0
    assert torch.equal(
        encoder.gap_raw_index.sort().values,
        torch.as_tensor(
            sorted(
                set(range(2048))
                - set(encoder.context_raw_index.tolist())
                - set(encoder.target_raw_index.tolist())
            )
        ),
    )
    x = batch(config, seed=4)
    gap_index = encoder.gap_raw_index
    for view, forbidden in (
        ("context", torch.cat([encoder.target_raw_index, gap_index])),
        ("target", torch.cat([encoder.context_raw_index, gap_index])),
    ):
        grad_input = x.clone().requires_grad_(True)
        encoder.encode(grad_input, view)["grid"].pow(2).sum().backward()
        assert torch.count_nonzero(grad_input.grad.reshape(BATCH, -1)[:, forbidden]) == 0
        with torch.no_grad():
            base = encoder.encode(x, view)["grid"]
            perturbed = x.clone()
            perturbed.reshape(BATCH, -1)[:, forbidden] += 1.3
            changed = encoder.encode(perturbed, view)["grid"]
        assert torch.equal(base, changed)


# ------------------------------------------------------------------- masks
def test_mask_visibility_and_learned_mask_embedding(config, products):
    model = build_arm(config, products, "sigreg_jepa")
    encoder = model.encoder
    x = batch(config, seed=6)

    def first_block_input(view):
        captured = {}
        def capture_tokens(module, inputs, output):
            captured["x"] = inputs[0].detach()

        handle = encoder.temporal_layers[0].register_forward_hook(capture_tokens)
        with torch.no_grad():
            encoder.encode(x, view)
        handle.remove()
        return captured["x"].view(BATCH, 8, 8, 128)

    context_tokens = first_block_input("context")
    masked_channels = [c for c in range(8) if c not in encoder.context_channels]
    assert masked_channels == [4, 5, 6, 7]
    for channel in masked_channels:
        for patch in range(8):
            assert torch.equal(
                context_tokens[:, channel, patch, :],
                encoder.mask_embedding[channel].expand(BATCH, 128),
            )

    target_tokens = first_block_input("target")
    for channel in encoder.context_channels:
        for patch in range(8):
            assert torch.equal(
                target_tokens[:, channel, patch, :],
                encoder.mask_embedding[channel].expand(BATCH, 128),
            )

    # Masked tokens are independent of the raw samples they replace.
    perturbed = x.clone()
    perturbed.reshape(BATCH, -1)[:, encoder.target_raw_index] += 5.0
    captured = {}

    def capture_perturbed_tokens(module, inputs, output):
        captured["x"] = inputs[0].detach()

    handle = encoder.temporal_layers[0].register_forward_hook(
        capture_perturbed_tokens
    )
    with torch.no_grad():
        encoder.encode(perturbed, "context")
    handle.remove()
    perturbed_tokens = captured["x"].view(BATCH, 8, 8, 128)
    for channel in masked_channels:
        for patch in range(8):
            assert torch.equal(
                perturbed_tokens[:, channel, patch, :],
                context_tokens[:, channel, patch, :],
            )

    # ...while a visible context token does change when a *context* sample
    # changes (the reverse perturbation would have to leave it unchanged).
    visible = x.clone()
    visible.reshape(BATCH, -1)[:, encoder.context_raw_index] += 5.0
    captured = {}
    handle = encoder.temporal_layers[0].register_forward_hook(capture_perturbed_tokens)
    with torch.no_grad():
        encoder.encode(visible, "context")
    handle.remove()
    visible_tokens = captured["x"].view(BATCH, 8, 8, 128)
    assert not torch.allclose(context_tokens[:, 0, 0, :], visible_tokens[:, 0, 0, :])


# ---------------------------------------------------------------- optimizers
@pytest.mark.parametrize("arm", ALL_ARMS)
def test_optimizer_coverage_and_finite_gradients(arm, config, products):
    model = build_arm(config, products, arm)
    optimizers = build_optimizers(model)
    coverage = check_optimizer_coverage(model, optimizers)  # raises if broken
    assert coverage["n_trainable"] == coverage["n_optimizer"]
    assert coverage["n_adamw"] + coverage["n_muon"] == coverage["n_trainable"]

    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    seen = set()
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                assert parameter.requires_grad
                assert id(parameter) not in seen
                seen.add(id(parameter))
    assert seen == trainable

    # Muon only receives 2-D block matrices; the frozen read-out heads are out.
    muon_parameters = [
        p for group in optimizers[1].param_groups for p in group["params"]
    ]
    assert muon_parameters and all(p.ndim >= 2 for p in muon_parameters)
    frozen_prefixes = (
        "encoder.final_proj_norm",
        "encoder.spatial_head",
        "encoder.energy_head",
    )
    for name, parameter in model.named_parameters():
        if name.startswith(frozen_prefixes):
            assert not parameter.requires_grad
            assert id(parameter) not in seen

    out = model(batch(config, seed=7))
    out["loss"].backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


# ------------------------------------------------------------- determinism
def test_ten_step_cpu_determinism(config, products):
    batches = synthetic.make_batches(config, batch=BATCH, count=10, seed=123)

    def run(arm):
        model = build_arm(config, products, arm, seed=11)
        optimizers = build_optimizers(model)
        losses = synthetic.train_steps(model, optimizers, batches, 10)
        return losses, model

    for arm in ("sigreg_jepa", "sigreg_jepa_anchored"):
        losses_a, model_a = run(arm)
        losses_b, model_b = run(arm)
        assert losses_a == losses_b
        for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters()):
            assert torch.equal(parameter_a, parameter_b)
        assert all(math.isfinite(loss) for loss in losses_a)


# --------------------------------------------------------------- SIGReg
def test_sigreg_gaussian_and_collapse_calibration():
    config = SIGRegConfig()  # pinned defaults: 1024 slices, t_max 3, 17 points
    assert (config.num_slices, config.t_max, config.n_points) == (1024, 3.0, 17)
    reg = SIGReg(config)
    torch.manual_seed(0)
    gaussian = torch.randn(256, 128)
    near_collapse = 0.01 * torch.randn(256, 128)
    anisotropic = torch.zeros(256, 128)
    anisotropic[:, 0] = torch.randn(256)
    scaled = 2.0 * gaussian

    gaussian_value = float(reg(gaussian, gaussian)["value"])
    collapse_value = float(reg(near_collapse, near_collapse)["value"])
    anisotropic_value = float(reg(anisotropic, anisotropic)["value"])
    scaled_value = float(reg(scaled, scaled)["value"])

    assert math.isfinite(gaussian_value) and gaussian_value < 5.0
    assert collapse_value > 20.0 * gaussian_value
    assert anisotropic_value > 5.0 * gaussian_value
    assert scaled_value > 5.0 * gaussian_value

    combined = reg(gaussian, torch.randn(4, 8, 128))
    assert torch.allclose(
        combined["value"], 0.5 * (combined["context"] + combined["target"])
    )
    # Step-seeded slice directions: a fresh module gives the same number.
    fresh = SIGReg(SIGRegConfig())
    assert float(fresh(gaussian, gaussian)["value"]) == gaussian_value


# ------------------------------------------------------- covariance / anchor
def test_positive_covariance_and_clamped_anchor_denominator(config, products):
    model = build_arm(config, products, "sigreg_jepa_anchored", beta=2.0)
    z = torch.randn(16, 128)
    trace = model._trace_cov(z)
    assert float(trace) > 0
    centered = z - z.mean(dim=0, keepdim=True)
    covariance = (
        centered.double().transpose(0, 1) @ centered.double() / (z.size(0) - 1)
    )
    eigenvalues = torch.linalg.eigvalsh(covariance)
    assert float(eigenvalues.min()) > -1e-8

    identical = torch.full((4, 8), 0.5)
    assert float(model._trace_cov(identical).abs()) < 1e-12
    penalty = model.anchor_penalty(identical, identical + 0.1)
    assert torch.isfinite(penalty)
    assert float(penalty) > 0


def test_anchor_penalty_trace_enters_the_gradient(config, products):
    """Finite-difference gradcheck would fail if the trace denominator were
    detached: the penalty value depends on the trace, so its gradient must too."""
    model = build_arm(config, products, "sigreg_jepa_anchored", beta=2.0)
    generator = torch.Generator().manual_seed(0)
    z = torch.randn(4, 3, dtype=torch.float64, generator=generator)
    z_tilde = z.detach() + 0.1 * torch.randn(
        4, 3, dtype=torch.float64, generator=generator
    )
    z.requires_grad_(True)
    z_tilde.requires_grad_(True)
    assert float(model._trace_cov(z)) > 1e-6
    assert torch.autograd.gradcheck(
        lambda a, b: model.anchor_penalty(a, b), (z, z_tilde), eps=1e-6, atol=1e-5
    )


def test_anchor_penalty_uses_unmasked_pooled_embeddings(config, products):
    model = build_arm(config, products, "sigreg_jepa_anchored", beta=2.5)
    x = batch(config, n=4, seed=8)
    overlay = products["noise_bank"][:4]
    model._sample_overlay = lambda n: overlay[:n]
    out = model(x)
    with torch.no_grad():
        z = model.encoder.encode_unmasked(x)["pooled"]
        z_tilde = model.encoder.encode_unmasked(x + overlay)["pooled"]
        expected = model.anchor_penalty(z, z_tilde)
    assert torch.allclose(out["anchor"], expected, rtol=1e-6, atol=1e-7)
    assert out["metrics"]["anchor_beta"] == 2.5


# ------------------------------------------------------------- mae_chi2
def test_mae_chi2_is_the_target_marginal_chi2(config, products):
    model = build_arm(config, products, "mae_chi2")
    x = batch(config, seed=9)
    out = model(x)
    assert out["metrics"]["uses_oracle_noise_information"] is True

    index = model.encoder.target_raw_index
    full = torch.as_tensor(
        products["target_noise_covariance"], dtype=torch.float64
    )
    marginal = full.index_select(0, index).index_select(1, index)
    residual = (out["prediction"] - out["label"]).double()
    dense = torch.linalg.solve(marginal, residual.transpose(0, 1)).transpose(0, 1)
    expected = 0.5 * (residual * dense).sum(dim=1).mean()
    assert torch.allclose(out["loss"].double(), expected, rtol=1e-4, atol=1e-5)
    # The reconstruction target is the raw target samples, channel-major.
    assert torch.equal(
        out["label"], x.flatten(1).index_select(1, model.encoder.target_raw_index)
    )


# ------------------------------------------------------- parallel ablation
def test_parallel_ablation_architecture(config, products):
    model = build_arm(config, products, ABLATION_ARM)
    encoder = model.encoder
    assert encoder.parallel is True
    assert len(encoder.temporal_layers) == len(encoder.spatial_layers) == 1
    assert len(encoder.merge_layers) == 1
    assert model.predictor is not None and model.sigreg is not None
    assert model.beta is None
    x = batch(config)
    grid = encoder(x)
    assert grid.shape == (BATCH, 8, 8, 128)
    assert encoder.encode_after_time(x).shape == (2, 8, 8, 128)
    out = model(x)
    assert torch.isfinite(out["loss"])
    optimizers = build_optimizers(model)
    muon = [p for g in optimizers[1].param_groups for p in g["params"]]
    assert any(p is encoder.merge_layers[0].weight for p in muon)


# ------------------------------------------------- amplitude preservation
def _pulse_fixture(n, seed, n_samples=256, channels=8):
    generator = torch.Generator().manual_seed(seed)
    t = torch.arange(n_samples, dtype=torch.float32)
    pulse = torch.exp(-(((t - 70.0) / 18.0) ** 2)) - 0.6 * torch.exp(
        -(((t - 70.0) / 40.0) ** 2)
    )
    pulse = pulse / pulse.abs().max()
    gains = torch.linspace(0.6, 1.4, channels)
    amplitudes = 0.5 + 1.5 * torch.rand(n, channels, generator=generator)
    noise = 0.02 * torch.randn(n, channels, n_samples, generator=generator)
    x = (
        amplitudes.unsqueeze(-1)
        * gains.view(1, channels, 1)
        * pulse.view(1, 1, -1)
        + noise
    )
    return x, amplitudes


def _ridge_r2(train_x, train_y, test_x, test_y, relative_alpha=1e-6):
    train_x = train_x.double()
    test_x = test_x.double()
    train_y = train_y.double()
    test_y = test_y.double()
    mean_x = train_x.mean(dim=0, keepdim=True)
    mean_y = train_y.mean(dim=0, keepdim=True)
    centered_x = train_x - mean_x
    centered_y = train_y - mean_y
    gram = centered_x.transpose(0, 1) @ centered_x
    alpha = relative_alpha * float(torch.diagonal(gram).mean())
    ridge = torch.linalg.solve(
        gram + alpha * torch.eye(gram.size(0), dtype=torch.float64),
        centered_x.transpose(0, 1) @ centered_y,
    )
    prediction = (test_x - mean_x) @ ridge + mean_y
    residual = ((test_y - prediction) ** 2).sum(dim=0)
    total = ((test_y - mean_y) ** 2).sum(dim=0)
    return 1.0 - residual / total


def test_amplitude_preservation_probe_fixture(config, products):
    """Probe implementation + fixture-level R².

    The registered criterion (R² >= 0.9 for sigreg_jepa and mae_chi2) is
    measured on *trained* encoders, after the smoke and on the final encoders
    (runbook v4 §3.5.7).  At random init the wrapped package zero-initialises
    residual block outputs, so the read-out is a fixed linear map of the input
    and this fixture is easy; the number below is a fixture check, not the
    scientific criterion.
    """
    model = build_arm(config, products, "sigreg_jepa")
    encoder = model.encoder
    train_x, train_y = _pulse_fixture(400, 1)
    test_x, test_y = _pulse_fixture(200, 2)

    def readout_features(grid):  # (B, n_patch, C, d) -> per-channel mean over patches
        return grid.mean(dim=1).reshape(grid.size(0), -1)

    def post_time_features(post):  # (B, C, n_patch, d)
        return post.mean(dim=2).reshape(post.size(0), -1)

    with torch.no_grad():
        train_readout = readout_features(encoder.encode(train_x, "unmasked")["grid"])
        test_readout = readout_features(encoder.encode(test_x, "unmasked")["grid"])
        train_post = post_time_features(encoder.encode(train_x, "unmasked")["post_time"])
        test_post = post_time_features(encoder.encode(test_x, "unmasked")["post_time"])

    readout_r2 = _ridge_r2(train_readout, train_y, test_readout, test_y)
    post_r2 = _ridge_r2(train_post, train_y, test_post, test_y)
    print(
        "amplitude probe fixture R2: read-out mean=%.4f "
        "(per-channel %s); post-time mean=%.4f"
        % (
            float(readout_r2.mean()),
            [round(float(value), 3) for value in readout_r2],
            float(post_r2.mean()),
        )
    )
    assert readout_r2.shape == (8,) and post_r2.shape == (8,)
    assert float(readout_r2.mean()) > 0.9
    assert float(post_r2.mean()) > 0.9


# ------------------------------------------------------------------ smoke
def test_cpu_smoke_four_arms_under_sixty_seconds(config, products):
    results = synthetic.smoke_arms(
        config, arms=ALL_ARMS, batch=BATCH, steps=1, seed=0, fixture=products
    )
    assert set(results) == set(ALL_ARMS)
    for arm in ALL_ARMS:
        assert results[arm]["seconds"] < 60.0, arm
        assert results[arm]["losses"] and all(
            math.isfinite(loss) for loss in results[arm]["losses"]
        )
        assert results[arm]["n_parameters"] > 0
    print(
        "smoke seconds: "
        + ", ".join(
            "%s=%.2f" % (arm, results[arm]["seconds"]) for arm in ALL_ARMS
        )
    )
