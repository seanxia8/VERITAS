# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Dowling Wong <wangdowling@gmail.com>
"""WP-N1: a frozen channel structure makes one implied covariance span many records.

Motivation (docs/LATENT_MONITORING_PLAN_2026-09-05.md, arms plan §7.3b): the
per-channel gains and private strengths used to be redrawn inside every
``generate()`` call, so pooling records across calls mixed several
covariances and kappa(Sigma_hat^-1 Sigma) plateaued instead of converging.
"""

from __future__ import annotations

import numpy as np
import pytest

from noise_module import MultiChannelNoiseGenerator

BASE = {"noise_type": "white", "noise_power": 1.0, "sampling_frequency": 1.0}


def _kappa(implied: np.ndarray, realized: np.ndarray) -> float:
    return float(np.linalg.cond(np.linalg.solve(implied, realized)))


@pytest.mark.parametrize("mode", ["shared_private", "lowrank"])
def test_unfrozen_structure_is_redrawn_per_call(mode: str) -> None:
    gen = MultiChannelNoiseGenerator(BASE, {"mode": mode, "n_channels": 6, "corr_strength": 0.4}, seed=1)
    _, a = gen.generate(64, return_metadata=True)
    _, b = gen.generate(64, return_metadata=True)
    assert a["channel_structure_frozen"] is False
    assert not np.array_equal(a["implied_covariance"], b["implied_covariance"])


@pytest.mark.parametrize("mode", ["shared_private", "lowrank"])
def test_frozen_structure_is_reused_and_implied_covariance_is_stable(mode: str) -> None:
    gen = MultiChannelNoiseGenerator(
        BASE, {"mode": mode, "n_channels": 6, "corr_strength": 0.4, "freeze_channel_structure": True}, seed=1
    )
    X1, a = gen.generate(64, return_metadata=True)
    X2, b = gen.generate(64, return_metadata=True)
    assert a["channel_structure_frozen"] is True
    np.testing.assert_array_equal(a["implied_covariance"], b["implied_covariance"])
    # the *noise* still differs — only the structure is pinned
    assert not np.array_equal(X1, X2)
    assert gen.channel_structure(mode, 6, a.get("n_latent", 1)) is not None


def test_pooling_frozen_records_converges_where_unfrozen_plateaus() -> None:
    """The measured defect: pooled kappa converges to 1 only when frozen."""
    C, N, reps = 8, 256, 64
    cfg = {"mode": "shared_private", "n_channels": C, "corr_strength": 0.5,
           "normalize_channel_variance": False}
    frozen = MultiChannelNoiseGenerator(BASE, {**cfg, "freeze_channel_structure": True}, seed=7)
    loose = MultiChannelNoiseGenerator(BASE, cfg, seed=7)
    kappas = {}
    for name, gen in (("frozen", frozen), ("loose", loose)):
        blocks, implied = [], None
        for _ in range(reps):
            X, meta = gen.generate(N, return_metadata=True)
            blocks.append(X)
            implied = meta["implied_covariance"]
        pooled = np.cov(np.concatenate(blocks, axis=1))
        kappas[name] = _kappa(implied, pooled)
    assert kappas["frozen"] < 1.25, kappas
    assert kappas["loose"] > kappas["frozen"], kappas


def test_set_channel_structure_pins_an_explicit_covariance() -> None:
    C = 5
    gen = MultiChannelNoiseGenerator(BASE, {"mode": "shared_private", "n_channels": C, "corr_strength": 0.3}, seed=3)
    gains = np.linspace(0.9, 1.1, C)
    private = np.full(C, 1.0)
    gen.set_channel_structure("shared_private", C, gains=gains, private_strengths=private)
    _, meta = gen.generate(128, return_metadata=True)
    np.testing.assert_array_equal(meta["gains"], gains)
    np.testing.assert_array_equal(meta["private_strengths"], private)
    assert meta["channel_structure_frozen"] is True
    # analytic covariance from the pinned structure
    power = meta["implied_covariance"][0, 0] / (0.3 * gains[0] ** 2 + 0.7 * private[0] ** 2)
    expected = power * (0.3 * np.outer(gains, gains) + 0.7 * np.diag(private**2))
    np.testing.assert_allclose(meta["implied_covariance"], expected, rtol=1e-12)


def test_set_channel_structure_validates_shapes() -> None:
    gen = MultiChannelNoiseGenerator(BASE, {"mode": "lowrank", "n_channels": 4, "n_latent": 2}, seed=0)
    with pytest.raises(ValueError, match="shape"):
        gen.set_channel_structure("lowrank", 4, 2, weights=np.ones((4, 3)),
                                  latent_strengths=np.ones((4, 2)), private_strengths=np.ones(4))
    with pytest.raises(ValueError, match="needs"):
        gen.set_channel_structure("shared_private", 4, gains=np.ones(4))
    with pytest.raises(ValueError, match="No channel structure"):
        gen.set_channel_structure("independent", 4)


def test_reset_channel_structure_redraws() -> None:
    gen = MultiChannelNoiseGenerator(
        BASE, {"mode": "shared_private", "n_channels": 4, "freeze_channel_structure": True}, seed=5
    )
    _, a = gen.generate(32, return_metadata=True)
    gen.reset_channel_structure()
    _, b = gen.generate(32, return_metadata=True)
    assert not np.array_equal(a["gains"], b["gains"])
